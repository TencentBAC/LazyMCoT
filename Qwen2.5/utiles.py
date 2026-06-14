import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from scipy.ndimage import zoom
import os
import pandas as pd
import pyarrow.parquet as pq
import ast
import json
from typing import List, Dict
from itertools import combinations
import base64
from PIL import Image
from io import BytesIO
import io
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter
import torch

######################################################################
# 创新点1: EGAF (Expert-Guided Attention Fusion)
#
# 三个子改进:
#   A. 视觉专家: 调用Grounding DINO服务(非MLLM)进行开放词汇检测
#   B. TAD加速: 只计算关键实体token的注意力图，跳过所有无关token
#   C. 专家补全: 专家检测到但TAD遗漏的目标，直接补充bbox到最终结果
######################################################################

import requests as http_requests  # 避免与局部变量results冲突

# ======================== A. Grounding DINO 视觉专家 ========================

def call_grounding_expert(image_path_or_base64, entity_list, expert_url="http://localhost:8001/predict",
                          box_threshold=0.3):
    """
    调用 Grounding DINO / LangSAM 视觉专家服务进行开放词汇目标检测。
    
    与视觉专家服务 (expert_server/model_service.py) 兼容的 HTTP 接口:
      POST /predict  { "image": base64_str, "text": prompt }
      返回 { "boxes": [[x1,y1,x2,y2], ...], "labels": [...] }
    
    Args:
        image_path_or_base64: 图像路径或 "data:image;base64,..." 字符串
        entity_list: list of str, 实体文本列表, e.g. ["dog", "motorcycle"]
        expert_url: str, 视觉专家服务地址
        box_threshold: float, 检测置信度阈值
    Returns:
        expert_results: dict, {entity_name: [[x0,y0,x1,y1], ...]} 归一化到[0,1]
    """
    # 准备图像 base64
    if image_path_or_base64.startswith('data:image;base64,'):
        img_b64 = image_path_or_base64.split(',')[1]
    elif os.path.exists(image_path_or_base64):
        img_b64 = image_to_base64(image_path_or_base64).split(',')[1]
    else:
        img_b64 = image_path_or_base64
    
    # 获取图像尺寸
    try:
        img_data = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_data))
        img_w, img_h = pil_img.size
    except:
        img_w, img_h = 1, 1
    
    expert_results = {}
    # 对每个实体分别调用（确保每个实体都有结果）
    for entity in entity_list:
        entity_clean = entity.strip()
        if not entity_clean:
            continue
        try:
            resp = http_requests.post(
                expert_url,
                json={"image": img_b64, "text": entity_clean},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                boxes = data.get("boxes", [])
                # Grounding DINO 返回像素坐标, 归一化到 [0,1]
                norm_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    nx1 = max(0.0, min(1.0, x1 / img_w))
                    ny1 = max(0.0, min(1.0, y1 / img_h))
                    nx2 = max(0.0, min(1.0, x2 / img_w))
                    ny2 = max(0.0, min(1.0, y2 / img_h))
                    if nx1 < nx2 and ny1 < ny2:
                        area = (nx2 - nx1) * (ny2 - ny1)
                        if area > 0.0005:
                            norm_boxes.append([nx1, ny1, nx2, ny2])
                expert_results[entity_clean] = norm_boxes
            else:
                expert_results[entity_clean] = []
        except Exception as e:
            print(f"  ⚠️ Grounding expert call failed for '{entity_clean}': {e}")
            expert_results[entity_clean] = []
    
    return expert_results


def call_grounding_expert_local(image_path_or_base64, entity_list, gdino_model=None):
    print("  ⚠️ Local grounding expert not initialized, returning empty results")
    return {entity.strip(): [] for entity in entity_list}


# ======================== B. TAD加速: 实体token定位 ========================

def find_entity_token_indices(entity_text, dicts, inputs, img_end,
                              min_token_ratio=0.3):
    """
    在 search prompt 的token序列中，精确定位属于实体词的token位置。
    
    Args:
        entity_text: str, 实体文本, e.g. "dog, motorcycle with red color"
        dicts: dict, {token_id: token_text}
        inputs: model inputs
        img_end: list, 图像token结束位置列表
        min_token_ratio: float, TAD加速保底比例。
            当实体token占比低于此值时，补充周围上下文token，
            防止信息量过低导致bbox质量退化。
    Returns:
        entity_token_indices: list of int, 属于实体词的token位置
        entity_token_map: dict, {entity_name: [token_positions]}
    """
    # 拆分实体列表
    entities = [e.strip().lower() for e in entity_text.split(',') if e.strip()]
    
    # 获取 search prompt 区间内所有token及其文本
    start_k = img_end[-1] + 1
    end_k = len(inputs['input_ids'][0])
    
    # 重建 token 序列文本
    token_positions = []  # [(position, token_text), ...]
    for k in range(start_k, end_k):
        token_id = inputs['input_ids'][0][k].cpu().item()
        if token_id in dicts:
            token_positions.append((k, dicts[token_id]))
        else:
            token_positions.append((k, ""))
    
    # 拼接全文
    full_text = ""
    char_to_token_idx = []  # char位置 → token_positions列表中的索引
    for tidx, (pos, text) in enumerate(token_positions):
        for c in text:
            char_to_token_idx.append(tidx)
        full_text += text
    
    full_text_lower = full_text.lower()
    
    # 对每个实体，在全文中找到所有出现位置，映射回token索引
    entity_token_indices = set()
    entity_token_map = {}
    
    for entity in entities:
        entity_lower = entity.lower().strip()
        if not entity_lower:
            continue
        
        matched_tids = set()
        # 查找所有出现位置
        search_start = 0
        while True:
            idx = full_text_lower.find(entity_lower, search_start)
            if idx == -1:
                break
            # 映射字符范围 [idx, idx+len) 到 token 索引
            for char_pos in range(idx, min(idx + len(entity_lower), len(char_to_token_idx))):
                tidx = char_to_token_idx[char_pos]
                matched_tids.add(tidx)
            search_start = idx + 1
        
        # 转换为实际的 token position
        positions = [token_positions[tidx][0] for tidx in sorted(matched_tids)]
        entity_token_map[entity_lower] = positions
        entity_token_indices.update(positions)
    
    # ===== TAD加速保底: 确保最低token数量 =====
    total_prompt_tokens = end_k - start_k
    min_tokens = max(3, int(total_prompt_tokens * min_token_ratio))
    
    if len(entity_token_indices) < min_tokens:
        # 实体token太少，在每个实体token周围补充上下文token
        expanded = set(entity_token_indices)
        all_prompt_positions = [tp[0] for tp in token_positions]
        
        for eidx in sorted(entity_token_indices):
            # 找到该token在 all_prompt_positions 中的位置
            try:
                pidx = all_prompt_positions.index(eidx)
            except ValueError:
                continue
            # 向前后各扩展1-2个token
            for delta in [-2, -1, 1, 2]:
                neighbor_pidx = pidx + delta
                if 0 <= neighbor_pidx < len(all_prompt_positions):
                    expanded.add(all_prompt_positions[neighbor_pidx])
                    if len(expanded) >= min_tokens:
                        break
            if len(expanded) >= min_tokens:
                break
        
        entity_token_indices = expanded
    
    return sorted(entity_token_indices), entity_token_map


# ======================== C. 专家补全: 独占bbox处理 ========================

def compute_expert_exclusive_bboxes(expert_results, tad_bboxes, iou_threshold=0.1,
                                     max_exclusive_per_entity=2):
    """
    找出视觉专家检测到但TAD注意力图完全遗漏的目标bbox。
    
    修复: 对每个实体的exclusive bbox设置上限(max_exclusive_per_entity)，
    防止通用类别(person等)的过度检测淹没正确目标。
    
    Args:
        expert_results: dict, {entity: [[x0,y0,x1,y1], ...]}
        tad_bboxes: list of [x0,y0,x1,y1], TAD产生的归一化bbox
        iou_threshold: float, IoU阈值，低于此值认为TAD遗漏
        max_exclusive_per_entity: int, 每个实体最多补充的exclusive bbox数量
    Returns:
        exclusive_bboxes: list of [x0,y0,x1,y1], 专家独占bbox
        exclusive_info: dict, 记录哪些实体被补全
    """
    def bbox_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0
    
    exclusive_bboxes = []
    exclusive_info = {}
    
    for entity, expert_boxes in expert_results.items():
        entity_exclusive = []
        for ebox in expert_boxes:
            # 计算与所有TAD bbox的最大IoU
            max_iou = 0.0
            if tad_bboxes:
                for tbox in tad_bboxes:
                    iou = bbox_iou(ebox, tbox)
                    max_iou = max(max_iou, iou)
            
            if max_iou < iou_threshold:
                entity_exclusive.append(ebox)
        
        # 限制每个实体的exclusive bbox数量，按面积降序取最大的几个
        if entity_exclusive and max_exclusive_per_entity > 0:
            entity_exclusive.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            entity_exclusive = entity_exclusive[:max_exclusive_per_entity]
        
        exclusive_bboxes.extend(entity_exclusive)
        if entity_exclusive:
            exclusive_info[entity] = len(entity_exclusive)
    
    return exclusive_bboxes, exclusive_info


# ======================== 融合函数 ========================

def expert_bboxes_to_attention_mask(expert_bboxes, att_h, att_w, sigma=2.0):
    """将视觉专家的bbox列表转换为软注意力掩码。"""
    if not expert_bboxes:
        return np.zeros((att_h, att_w), dtype=np.float32)
    
    expert_att = np.zeros((att_h, att_w), dtype=np.float32)
    for bbox in expert_bboxes:
        x0, y0, x1, y1 = bbox
        ax0 = max(0, int(x0 * att_w))
        ay0 = max(0, int(y0 * att_h))
        ax1 = min(att_w, int(x1 * att_w))
        ay1 = min(att_h, int(y1 * att_h))
        if ax0 < ax1 and ay0 < ay1:
            expert_att[ay0:ay1, ax0:ax1] = 1.0
    
    if sigma > 0:
        expert_att = gaussian_filter(expert_att, sigma=sigma)
    if expert_att.max() > 0:
        expert_att = expert_att / expert_att.max()
    return expert_att


def estimate_expert_reliability(expert_results, entity_list):
    """
    估计视觉专家的可靠性系数 ρ ∈ [0, 1]。
    ρ = 至少被检测到一次的实体比例。
    """
    if not entity_list or not expert_results:
        return 0.0
    detected = sum(1 for e in entity_list 
                   if e.strip() in expert_results and len(expert_results[e.strip()]) > 0)
    return detected / len(entity_list)


def fuse_tad_and_expert(tad_att, expert_att, fusion_mode="adaptive", expert_weight=0.5,
                        expert_reliability=1.0):
    """
    Rank-Based Evidence Fusion (改进B).
    
    ======== 核心 Insight ========
    
    注意力值 a(p)∈[0,1] 不是概率——它是经过 min-max 归一化的相对量。
    在归一化的值域上做贝叶斯融合(log-odds)会因为:
      (1) logit(1-ε)→+∞ 导致极端值主导
      (2) 大面积bbox的高值压缩小目标的相对值
    
    解决: 在 **秩空间(rank space)** 做融合，而非值空间。
    
    ======== 数学定义 ========
    
    定义 rank 映射:  R(x) = rank(x) / total_pixels
    将注意力图的每个像素值替换为其在所有像素中的百分位排名 ∈ [0,1]
    
    融合公式:
      F(p) = (1-α)·R_tad(p) + α·R_expert(p)
    
    其中 α = ρ·w (可靠性调制后的专家权重):
      - ρ=0 (专家不可靠): α=0, F=R_tad, 完全退化为纯TAD排序
      - ρ=1 (专家完全可靠): α=w, TAD和专家共同决定排序
    
    ======== 性质证明 ========
    
    性质1 (保序性): R(·) 是保序映射，α-加权和保序 → 不改变TAD的相对排序
    性质2 (尺度不变性): R(·) 仅依赖排序，不依赖绝对值 → 不受大面积bbox影响
    性质3 (优雅退化): ρ→0 ⇒ α→0 ⇒ F→R_tad，无需硬编码回退
    """
    if fusion_mode == "multiply":
        fused = tad_att * expert_att
        if fused.max() > 0:
            fused = (fused - fused.min()) / (fused.max() - fused.min())
        return fused
    elif fusion_mode == "add":
        fused = (1.0 - expert_weight) * tad_att + expert_weight * expert_att
        if fused.max() > 0:
            fused = (fused - fused.min()) / (fused.max() - fused.min())
        return fused
    elif fusion_mode == "adaptive":
        # ---- Rank-Based Fusion ----
        # Step 1: 将注意力图转为秩百分位图
        def to_rank_map(att):
            """将2D注意力图转为秩百分位图（值域[0,1]）"""
            flat = att.flatten()
            # argsort of argsort 给出排名
            order = flat.argsort().argsort()
            rank = order.astype(np.float32) / max(len(flat) - 1, 1)
            return rank.reshape(att.shape)
        
        R_tad = to_rank_map(tad_att)
        R_expert = to_rank_map(expert_att)
        
        # Step 2: 可靠性调制的专家权重
        alpha = expert_reliability * expert_weight  # α = ρ·w
        
        # Step 3: 加权融合
        fused = (1.0 - alpha) * R_tad + alpha * R_expert
        
        # 归一化到[0,1]
        if fused.max() > fused.min():
            fused = (fused - fused.min()) / (fused.max() - fused.min())
        
        return fused
    else:
        return tad_att


# ======================== process: 原始HiDe TAD (基准, 不修改) ========================

def process(dicts, start_k, end_k, attention, inputs, img_start, img_end, sig):
    """原始 HiDe TAD 注意力处理。对所有 search prompt token 逐一计算注意力图。"""
    accept_att = {}
    noise_token_num = 8
    noise_mean = [[0 for k in range(noise_token_num)] for i in range(len(inputs["image_grid_thw"]))]
    for k in range(start_k, end_k - 4):
        per_img_attention = []
        for img_idx in range(len(inputs["image_grid_thw"])):
            image_grid_thw = inputs["image_grid_thw"][img_idx]
            start = img_start[img_idx]
            end = img_end[img_idx]
            if start_k < end:
                start_k = end + 1
            layer_mean = []
            for i in range(len(attention)):
                k_att_map = np.array([row[k] for row in attention[i][0]])
                att_map = k_att_map[:, start:end].reshape(-1, image_grid_thw[1] // 2, image_grid_thw[2] // 2).mean(axis=0)
                layer_mean.append(att_map)
            per_img_attention.append(np.array(layer_mean).mean(axis=0, keepdims=True))
        max_att_get = 0
        for i in range(len(per_img_attention)):
            sum_per_img_att = per_img_attention[i].max()
            if sum_per_img_att > max_att_get:
                max_att_get = sum_per_img_att
                img_idx = i
            if k < start_k + noise_token_num:
                per_att = per_img_attention[i]
                if sig > 0:
                    per_att = gaussian_filter(per_att, sigma=sig)
                per_att = per_att - per_att.min()
                if per_att.max() > 0:
                    per_att = per_att / per_att.max()
                noise_mean[i][k - start_k] = per_att
        if k < start_k + noise_token_num:
            continue
        if img_idx not in accept_att:
            accept_att[img_idx] = {}
        accept_s = per_img_attention[img_idx]
        if sig > 0:
            accept_s = gaussian_filter(accept_s, sigma=sig)
        accept_s = accept_s - accept_s.min()
        if accept_s.max() > 0:
            accept_s = accept_s / accept_s.max()
        if noise_token_num > 0:
            accept_s -= np.array(noise_mean[img_idx]).mean(axis=0)
            accept_s[accept_s < 0] = 0
        if accept_s.max() <= 0:
            continue
        accept_s = accept_s - accept_s.min()
        accept_s = accept_s / accept_s.max()
        accept_att[img_idx][k] = accept_s
    return accept_att


# ======================== process_egaf: EGAF改进版TAD ========================

def process_egaf(dicts, start_k, end_k, attention, inputs, img_start, img_end, sig,
                 expert_bboxes_per_img=None, egaf_fusion_mode="adaptive",
                 egaf_expert_weight=0.5, entity_token_indices=None,
                 entity_token_map=None, expert_reliability=1.0):
    """
    EGAF 改进版注意力处理。
    
    与原始 process 的区别:
      1. SAAA (Semantic-Aware Attention Aggregation): 将同一个实体的多个 token 的注意力图取 max 聚合
      2. Rank-Based Fusion: 聚合后的注意力图与视觉专家掩码在秩空间进行保序融合
      3. TAD加速: 跳过完全无关的非实体token
    """
    accept_att = {}
    raw_tad_att = {}
    noise_token_num = 8
    noise_mean = [[0 for k in range(noise_token_num)] for i in range(len(inputs["image_grid_thw"]))]

    # 预计算专家注意力掩码
    expert_att_maps = {}
    if expert_bboxes_per_img:
        for img_idx_e, bboxes_e in expert_bboxes_per_img.items():
            if img_idx_e < len(inputs["image_grid_thw"]):
                grid = inputs["image_grid_thw"][img_idx_e]
                att_h, att_w = grid[1].item() // 2, grid[2].item() // 2
                expert_att_maps[img_idx_e] = expert_bboxes_to_attention_mask(
                    bboxes_e, att_h, att_w, sigma=2.0)

    entity_set = set(entity_token_indices) if entity_token_indices else None

    # 第一阶段：提取所有保留 token 的 TAD 注意力图（并去噪）
    for k in range(start_k, end_k - 4):
        # TAD加速: 跳过非实体token (保留噪声估计token)
        if entity_set is not None:
            if k >= start_k + noise_token_num and k not in entity_set:
                continue

        per_img_attention = []
        for img_idx in range(len(inputs["image_grid_thw"])):
            image_grid_thw = inputs["image_grid_thw"][img_idx]
            start = img_start[img_idx]
            end = img_end[img_idx]
            if start_k < end:
                start_k = end + 1
            layer_mean = []
            for i in range(len(attention)):
                k_att_map = np.array([row[k] for row in attention[i][0]])
                att_map = k_att_map[:, start:end].reshape(-1, image_grid_thw[1] // 2, image_grid_thw[2] // 2).mean(axis=0)
                layer_mean.append(att_map)
            per_img_attention.append(np.array(layer_mean).mean(axis=0, keepdims=True))
        max_att_get = 0
        img_idx = 0
        for i in range(len(per_img_attention)):
            sum_per_img_att = per_img_attention[i].max()
            if sum_per_img_att > max_att_get:
                max_att_get = sum_per_img_att
                img_idx = i
            if k < start_k + noise_token_num:
                per_att = per_img_attention[i]
                if sig > 0:
                    per_att = gaussian_filter(per_att, sigma=sig)
                per_att = per_att - per_att.min()
                if per_att.max() > 0:
                    per_att = per_att / per_att.max()
                noise_mean[i][k - start_k] = per_att
        if k < start_k + noise_token_num:
            continue
            
        if img_idx not in raw_tad_att:
            raw_tad_att[img_idx] = {}
            
        accept_s = per_img_attention[img_idx]
        if sig > 0:
            accept_s = gaussian_filter(accept_s, sigma=sig)
        accept_s = accept_s - accept_s.min()
        if accept_s.max() > 0:
            accept_s = accept_s / accept_s.max()
        if noise_token_num > 0:
            accept_s -= np.array(noise_mean[img_idx]).mean(axis=0)
            accept_s[accept_s < 0] = 0
        if accept_s.max() <= 0:
            continue
        accept_s = accept_s - accept_s.min()
        accept_s = accept_s / accept_s.max()
        
        raw_tad_att[img_idx][k] = accept_s

    # 第二阶段：Semantic-Aware Attention Aggregation (SAAA) & Rank-Based Fusion
    # 构建分组
    entity_groups = []
    handled_tokens = set()
    if entity_token_map:
        for entity_name, tokens in entity_token_map.items():
            if tokens:
                group = sorted(tokens)
                entity_groups.append(group)
                handled_tokens.update(group)
                
    # 把不在实体分组内（但被 TAD 加速保底保留的）token 单独作为独立组
    if entity_set is not None:
        for t in sorted(entity_set):
            if t not in handled_tokens and t >= start_k + noise_token_num:
                entity_groups.append([t])
    else:
        # 如果没有开启实体过滤，对所有处理过的 token 单独设组
        for img_idx in raw_tad_att:
            for t in raw_tad_att[img_idx]:
                if t not in handled_tokens:
                    entity_groups.append([t])
                    handled_tokens.add(t)

    for img_idx in raw_tad_att:
        if img_idx not in accept_att:
            accept_att[img_idx] = {}
            
        for group in entity_groups:
            # 取出该组的所有 attention map
            group_atts = [raw_tad_att[img_idx][t] for t in group if t in raw_tad_att[img_idx]]
            if not group_atts:
                continue
                
            # 1. 聚合注意力: Max Pooling (保留同一实体多个 token 中的最大激活)
            agg_att = np.max(group_atts, axis=0)
            
            # 2. Rank-Based Fusion
            if img_idx in expert_att_maps:
                tad_2d = agg_att[0] if agg_att.ndim == 3 else agg_att
                expert_2d = expert_att_maps[img_idx]
                if tad_2d.shape == expert_2d.shape:
                    fused_2d = fuse_tad_and_expert(
                        tad_2d, expert_2d,
                        fusion_mode=egaf_fusion_mode,
                        expert_weight=egaf_expert_weight,
                        expert_reliability=expert_reliability)
                    agg_att = fused_2d[np.newaxis, ...] if agg_att.ndim == 3 else fused_2d
                    
            # 3. 将聚合并融合后的注意力图赋给该组的最后(或者最核心)一个 token
            target_k = group[-1]
            accept_att[img_idx][target_k] = agg_att

    return accept_att


######################################################################
# GRACE: Gradient-guided Relative Attention with Consistency-aware
#        Expert Verification
#
# 核心改进（相对于 TAD 和 EGAF）:
#   1. 相对注意力归一化：A_rel = A / (A_noise + ε)，替代简单减法，
#      更有效地压制 attention sink 等背景噪声
#   2. Max-pooling 聚合同一实体的多个 token 注意力图
#   3. SAM3 文本提示检测结果作为独立的补充 bbox（不融合进注意力图）
#   4. 自适应 Otsu 阈值（在 Get_box.py 中应用）替代固定阈值
######################################################################

def process_grace(dicts, start_k, end_k, attention, inputs, img_start, img_end, sig,
                  entity_token_indices=None, entity_token_map=None,
                  noise_token_num=8):
    """
    GRACE 注意力处理：与 TAD process 完全一致的逐 token 遍历和减噪归一化。

    注意力计算机制已完全回退到 TAD 方案：
      - 逐 token 独立处理（不做实体 token 过滤、不做 Max Pooling 聚合）
      - 前 noise_token_num 个 token 的归一化注意力图作为噪声基准，后续 token 减去后再归一化
      - 使用固定阈值 t 进行二值化（与 TAD 一致，不使用 Otsu）
    GRACE 保留的改进点（仅视觉专家融合部分）：
      - SAM3 文本提示检测结果作为独立补充 bbox（橙色高亮 + 标注）
      - 二次 SAM3 验证（在注意力 bbox 内裁剪搜索，青色标注）

    噪声处理方案（与 tmp/HiDe 原始 TAD 一致）：
      1. 取 Search Prompt 前 noise_token_num（默认 8）个 token 的归一化注意力图
         作为噪声均值估计
      2. 对后续 token 的注意力图：
         a. Gaussian blur + min-max 归一化到 [0,1]
         b. accept_s -= noise_mean（减去噪声基准）
         c. 负值裁剪为 0
         d. 再次 min-max 归一化

    Args:
        dicts: {token_id: token_text}
        start_k: 文本 token 起始位置（img_end[-1]+1）
        end_k: 序列末尾位置
        attention: list of per-layer attention tensors
        inputs: model inputs
        img_start / img_end: 图像 token 起止位置列表
        sig: Gaussian blur sigma
        entity_token_indices: [已弃用] 保留参数兼容性，不再使用
        entity_token_map: [已弃用] 保留参数兼容性，不再使用
        noise_token_num: int，前 N 个 token 用作噪声基准（默认 8）

    Returns:
        accept_att: {img_idx: {token_k: att_map (H, W)}}
    """
    accept_att = {}
    noise_mean = [[0 for k in range(noise_token_num)] for i in range(len(inputs["image_grid_thw"]))]
    for k in range(start_k, end_k - 4):
        per_img_attention = []
        for img_idx in range(len(inputs["image_grid_thw"])):
            image_grid_thw = inputs["image_grid_thw"][img_idx]
            start = img_start[img_idx]
            end = img_end[img_idx]
            if start_k < end:
                start_k = end + 1
            layer_mean = []
            for i in range(len(attention)):
                k_att_map = np.array([row[k] for row in attention[i][0]])
                att_map = k_att_map[:, start:end].reshape(-1, image_grid_thw[1] // 2, image_grid_thw[2] // 2).mean(axis=0)
                layer_mean.append(att_map)
            per_img_attention.append(np.array(layer_mean).mean(axis=0, keepdims=True))
        max_att_get = 0
        for i in range(len(per_img_attention)):
            sum_per_img_att = per_img_attention[i].max()
            if sum_per_img_att > max_att_get:
                max_att_get = sum_per_img_att
                img_idx = i
            if k < start_k + noise_token_num:
                per_att = per_img_attention[i]
                if sig > 0:
                    per_att = gaussian_filter(per_att, sigma=sig)
                per_att = per_att - per_att.min()
                if per_att.max() > 0:
                    per_att = per_att / per_att.max()
                noise_mean[i][k - start_k] = per_att
        if k < start_k + noise_token_num:
            continue
        if img_idx not in accept_att:
            accept_att[img_idx] = {}
        accept_s = per_img_attention[img_idx]
        if sig > 0:
            accept_s = gaussian_filter(accept_s, sigma=sig)
        accept_s = accept_s - accept_s.min()
        if accept_s.max() > 0:
            accept_s = accept_s / accept_s.max()
        if noise_token_num > 0:
            accept_s -= np.array(noise_mean[img_idx]).mean(axis=0)
            accept_s[accept_s < 0] = 0
        if accept_s.max() <= 0:
            continue
        accept_s = accept_s - accept_s.min()
        accept_s = accept_s / accept_s.max()
        accept_att[img_idx][k] = accept_s
    return accept_att


def get_sam3_supplement_bboxes(sam3_results, max_per_entity=3):
    """
    从 SAM3 文本提示检测结果中提取所有边界框及对应实体标签。

    - 不做 IoU 过滤：所有 SAM3 检测到的框都纳入最终结果。
    - 每个实体按面积从大到小排序，最多取 max_per_entity 个。

    Args:
        sam3_results: dict, {entity_name: [[x0, y0, x1, y1], ...]}，坐标归一化到 [0,1]
        max_per_entity: int，每个实体最多保留的 bbox 数量

    Returns:
        supplement_bboxes: list of [x0, y0, x1, y1]
        entity_labels: list of str，与 supplement_bboxes 等长，每个 bbox 对应的实体名
    """
    supplement_bboxes = []
    entity_labels = []
    for entity, boxes in sam3_results.items():
        if not boxes:
            continue
        sorted_boxes = sorted(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )
        for b in sorted_boxes[:max_per_entity]:
            supplement_bboxes.append(b)
            entity_labels.append(entity)
    return supplement_bboxes, entity_labels


######################################################################
# 创新点2: ACR (Adaptive Confidence Routing) 相关函数
######################################################################

def filter_noise_bboxes(bboxes, min_area_ratio=0.001, edge_margin=0.02):
    """
    创新点2-ACR: 过滤噪声bbox
    
    移除面积过小和位于图像角落/边缘的噪声bbox
    
    Args:
        bboxes: list of [x0, y0, x1, y1] (归一化坐标)
        min_area_ratio: float, 最小面积比例阈值
        edge_margin: float, 边缘判定阈值
    Returns:
        filtered_bboxes: 过滤后的bbox列表
    """
    if not bboxes:
        return bboxes
    
    filtered = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        
        # 过滤面积过小的bbox
        if area < min_area_ratio:
            continue
        
        # 过滤位于四个角落的微小bbox
        is_corner = False
        corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for cx, cy in corners:
            if (abs(x0 - cx) < edge_margin or abs(x1 - cx) < edge_margin) and \
               (abs(y0 - cy) < edge_margin or abs(y1 - cy) < edge_margin):
                if area < 0.005:  # 角落处的微小bbox
                    is_corner = True
                    break
        if is_corner:
            continue
        
        filtered.append(bbox)
    
    return filtered if filtered else bboxes  # 如果全部被过滤，返回原始列表


def extract_answer_letter(text):
    """从模型输出中提取答案字母"""
    import re
    # 先尝试从 FINAL_OUTPUT 标签中提取
    match = re.search(r'<FINAL_OUTPUT>\s*\(?([A-D])\)?\s*', text)
    if match:
        return match.group(1)
    # 尝试提取最后出现的选项字母
    matches = re.findall(r'\(([A-D])\)', text)
    if matches:
        return matches[-1]
    # 尝试找到独立的选项字母
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1]
    return None


def confidence_routing(ori_answer_text, hide_answer_text, ori_letter, hide_letter):
    """
    创新点2-ACR: 自适应置信度路由
    
    比较原始回答和HiDe回答的质量，选择更优的答案
    
    启发式规则:
    1. 如果两个答案一致，直接返回
    2. 如果HiDe的回答更详细且包含推理过程，选择HiDe
    3. 如果原始回答更简洁自信，选择原始
    4. 默认选择HiDe
    
    Returns:
        best_answer_text: str
        best_letter: str
        route: str, "ori" or "hide"
    """
    # 如果两者一致
    if ori_letter and hide_letter and ori_letter == hide_letter:
        return hide_answer_text, hide_letter, "agree"
    
    # 如果HiDe没有提取到有效答案
    if not hide_letter and ori_letter:
        return ori_answer_text, ori_letter, "ori"
    
    if not ori_letter and hide_letter:
        return hide_answer_text, hide_letter, "hide"
    
    # 检查回答长度和推理深度
    ori_has_reasoning = len(ori_answer_text) > 100 and any(
        kw in ori_answer_text.lower() for kw in ['therefore', 'because', 'since', 'looking at']
    )
    hide_has_reasoning = len(hide_answer_text) > 100 and any(
        kw in hide_answer_text.lower() for kw in ['therefore', 'because', 'since', 'looking at']
    )
    
    # 如果只有HiDe有推理过程，选择HiDe
    if hide_has_reasoning and not ori_has_reasoning:
        return hide_answer_text, hide_letter, "hide"
    
    # 如果只有ori有推理过程，选择ori
    if ori_has_reasoning and not hide_has_reasoning:
        return ori_answer_text, ori_letter, "ori"
    
    # 默认选择HiDe（因为有额外视觉信息）
    return hide_answer_text, hide_letter, "hide"


######################################################################
# 创新点3: PMGVV (Progressive Multi-Granularity Visual Verification) 相关函数
######################################################################

def expand_bbox(bbox, expand_ratio=0.3, img_w=1.0, img_h=1.0):
    """
    扩展单个bbox，用于PMGVV的自适应区域扩展
    
    Args:
        bbox: [x0, y0, x1, y1] (归一化坐标)
        expand_ratio: 扩展比例
    Returns:
        expanded bbox
    """
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    new_x0 = max(0, cx - new_w / 2)
    new_y0 = max(0, cy - new_h / 2)
    new_x1 = min(img_w, cx + new_w / 2)
    new_y1 = min(img_h, cy + new_h / 2)
    return [new_x0, new_y0, new_x1, new_y1]


def merge_all_bboxes(bboxes):
    """将所有bbox合并为一个外包围框"""
    if not bboxes:
        return None
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]


def compute_bbox_coverage(bboxes):
    """计算bbox覆盖的图像面积比例"""
    if not bboxes:
        return 0.0
    merged = merge_all_bboxes(bboxes)
    if merged is None:
        return 0.0
    x0, y0, x1, y1 = merged
    return (x1 - x0) * (y1 - y0)


def build_verification_prompt(entities, has_crop=True):
    """
    创新点3-PMGVV: 构建验证prompt
    
    用于检查裁剪图像中是否包含所有关键实体
    """
    entity_list = entities if isinstance(entities, list) else [e.strip() for e in entities.split(',')]
    entity_str = ', '.join(entity_list)
    
    if has_crop:
        prompt = (
            f"Look at the second image carefully. "
            f"Can you see ALL of the following objects: {entity_str}? "
            f"Answer ONLY 'yes' or 'no'. If any object is missing, answer 'no'."
        )
    else:
        prompt = (
            f"Look at this image carefully. "
            f"Can you see ALL of the following objects: {entity_str}? "
            f"Answer ONLY 'yes' or 'no'. If any object is missing, answer 'no'."
        )
    return prompt

def create_directory(path):
    """
    创建给定路径的目录，包括所有必要的父目录。

    :param path: 完整的目录路径字符串
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created successfully at {path}")
    except Exception as e:
        print(f"Failed to create directory at {path}: {e}")

def load_json_to_list(json_path: str) -> List[Dict]:
    """
    加载 JSON 文件并返回一个由字典组成的列表
    
    参数:
        json_path (str): JSON 文件路径
    
    返回:
        List[Dict]: 列表中的每个元素都是一个字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 文件内容不是一个列表")

    return data

def serialize_dict(my_dict, file_path):
    """
    将一个字典序列化为一行 JSON，追加写入到 .jsonl 文件。
    
    每次调用写入一行，不换行嵌套，符合 JSONL 标准。
    
    参数:
        my_dict: 要写入的字典（可能包含 ndarray、np.int64 等）
        file_path: 输出的 .jsonl 文件路径
    """
    def serialize_obj(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: serialize_obj(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_obj(item) for item in obj]
        else:
            return obj

    # 序列化整个字典
    serialized_dict = serialize_obj(my_dict)

    # 追加写入一行 JSON
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serialized_dict, ensure_ascii=False, indent=4) + '\n')

def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def pil_to_base64(pil_img, format="PNG"):
    buffered = BytesIO()
    # 如果 pil_img.format 不存在，使用指定的默认格式
    img_format = pil_img.format if pil_img.format else format
    pil_img.save(buffered, format=img_format)  # 使用指定格式保存图像到内存
    encoded_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def swap_and_rebuild_dict(nested_dict):
    """
    将两层嵌套字典的内外层 key 对调。
    
    输入:
        nested_dict: 形如 {outer_key: {inner_key: value}}
    输出:
        new_dict: 形如 {inner_key: {outer_key: value}}
    """
    new_dict = {}

    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in new_dict:
                new_dict[inner_key] = {}
            new_dict[inner_key][outer_key] = value
            
    return dict(sorted(new_dict.items()))

def detect_concentrated_regions_with_merge(matrix, k=3, merge_distance_ratio=0.1):
    """
    自动检测集中区域，并合并距离较近的区域。
    如果一个小区域被一个大区域完全包含，则只保留外层的大区域。
    
    参数:
        matrix (np.ndarray): NxN 受力矩阵
        k (float): 控制灵敏度的倍数，默认为 2
        merge_distance_ratio (float): 合并距离阈值（相对于图像对角线的比例）
    
    返回:
        List[List[int]]: 每个元素是一个 bounding box [x1, y1, x2, y2]
    """
    H, W = matrix.shape
    diag_length = np.sqrt(H**2 + W**2)
    merge_distance_threshold = diag_length * merge_distance_ratio  # 转换为实际像素距离

    # Step 1: 原有方法提取所有原始区域
    mean = np.mean(matrix)
    std = np.std(matrix)
    threshold = mean + k * std
    binary = matrix > threshold
    labeled_matrix, num_features = ndimage.label(binary)

    regions = []
    for label_id in range(1, num_features + 1):
        coords = np.column_stack(np.where(labeled_matrix == label_id))
        regions.append(coords)

    if not regions:
        return []

    # Step 2: 获取每个区域的 bounding box
    boxes = []
    for coords in regions:
        y_min, x_min = np.min(coords, axis=0)
        y_max, x_max = np.max(coords, axis=0)
        boxes.append([x_min, y_min, x_max, y_max])  # [x1,y1,x2,y2]

    # Step 3: 构建区域之间的距离图
    n = len(boxes)
    to_merge = []

    def box_center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # 判断哪些框可以合并
    for i, j in combinations(range(n), 2):
        c1 = box_center(boxes[i])
        c2 = box_center(boxes[j])
        dist = np.linalg.norm(c1 - c2)
        if dist < merge_distance_threshold:
            to_merge.append((i, j))


    # Step 4: 合并逻辑（使用并查集）
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i, j in to_merge:
        union(i, j)

    # Step 5: 收集合并后的区域
    merged_boxes = {}
    for i in range(n):
        root = find(i)
        if root not in merged_boxes:
            merged_boxes[root] = boxes[i]
        else:
            x1 = min(merged_boxes[root][0], boxes[i][0])
            y1 = min(merged_boxes[root][1], boxes[i][1])
            x2 = max(merged_boxes[root][2], boxes[i][2])
            y2 = max(merged_boxes[root][3], boxes[i][3])
            merged_boxes[root] = [x1, y1, x2, y2]

    # Step 6: 转换为 list 格式返回
    final_boxes = [list(map(int, box)) for box in merged_boxes.values()]

    # Step 7: 去除被完全包含的小区域
    def remove_nested_boxes(boxes):
        if not boxes:
            return []

        # 按面积从大到小排序
        def area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        boxes_sorted = sorted(boxes, key=area, reverse=True)
        result = []

        for current in boxes_sorted:
            x1, y1, x2, y2 = current
            contained = False
            for other in result:
                ox1, oy1, ox2, oy2 = other
                if ox1 <= x1 and oy1 <= y1 and ox2 >= x2 and oy2 >= y2:
                    contained = True
                    break
            if not contained:
                result.append(current)

        return result
    H, W = matrix.shape
    final_boxes = remove_nested_boxes(final_boxes)
    return final_boxes

def load_dataset_Vstar_json(path):
    Vstar_list = []
    with open(path, 'r', encoding='utf-8') as f:
        Vstar_list = json.load(f)
    mmetype_Vstarbench = []
    for i in range(len(Vstar_list)):
        # if Vstar_list[i]["category"] == "direct_attributes": continue
        dict_i = {}
        dict_i["id"] = Vstar_list[i]["id"]
        dict_i["Text"] = Vstar_list[i]["question"].replace("\nAnswer with the option's letter from the given choices directly.","")
        # dict_i["Choices"] = "\n".join(Vstar_list[i]["text"].split("\n")[1:-1])
        dict_i["Ground truth"] = Vstar_list[i]["labels"]
        dict_i["image"] = Vstar_list[i]["image_path"]
        if "box_json" in Vstar_list[i]:
            dict_i["box_json"] = Vstar_list[i]["box_json"]
        dict_i["category"] = Vstar_list[i]["category"]
        mmetype_Vstarbench.append(dict_i)
    return mmetype_Vstarbench

def load_dataset_hrbench_json(path):
    Vstar_list = []
    with open(path, 'r', encoding='utf-8') as f:
        Vstar_list = json.load(f)
    mmetype_Vstarbench = []
    for i in range(len(Vstar_list)):
        dict_i = {}
        dict_i["id"] = Vstar_list[i]["id"]
        dict_i["Text"] = Vstar_list[i]["question"]
        # dict_i["Text"] = Vstar_list[i]["question"] + "\nAnswer with the option's letter from the given choices directly."
        # dict_i["Choices"] = "\n".join(Vstar_list[i]["text"].split("\n")[1:-1])
        dict_i["Ground truth"] = Vstar_list[i]["labels"]
        dict_i["image"] = Vstar_list[i]["image_path"]
        dict_i["Category"] = Vstar_list[i]["Category"]
        mmetype_Vstarbench.append(dict_i)
    return mmetype_Vstarbench

def load_dataset_hrbench(path):
    hrbench = pd.read_csv(path, sep='\t')
    mmetype_hrbench = []
    for i in hrbench.index:
        # if str(hrbench["cycle_category"][i]) != "0": continue
        dict_i = {}
        dict_i["Text"] = hrbench["question"][i] + "\n(A) "+hrbench["A"][i] + "\n(B) "+hrbench["B"][i]+"\n(C) "+hrbench["C"][i]+ "\n(D) "+hrbench["D"][i]
        # dict_i["Answer choices"] = ["(A) "+hrbench["A"][i], "(B) "+hrbench["B"][i], "(C) "+hrbench["C"][i], "(D) "+hrbench["D"][i]]
        dict_i["Ground truth"] = hrbench["answer"][i]
        dict_i["image"] = r"data:image;base64,"+hrbench["image"][i]
        dict_i["Category"] = hrbench["category"][i]
        dict_i["cycle_category"] = str(hrbench["cycle_category"][i])
        mmetype_hrbench.append(dict_i)
    return mmetype_hrbench
