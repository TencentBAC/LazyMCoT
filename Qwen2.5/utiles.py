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
#
######################################################################

import requests as http_requests # results


def call_grounding_expert(image_path_or_base64, entity_list, expert_url="http://localhost:8001/predict",
                          box_threshold=0.3):
    """
     Grounding DINO / LangSAM 
    
     (expert_server/model_service.py) HTTP :
      POST /predict  { "image": base64_str, "text": prompt }
       { "boxes": [[x1,y1,x2,y2], ...], "labels": [...] }
    
    Args:
        image_path_or_base64: "data:image;base64,..." 
        entity_list: list of str, , e.g. ["dog", "motorcycle"]
        expert_url: str, 
        box_threshold: float, 
    Returns:
        expert_results: dict, {entity_name: [[x0,y0,x1,y1], ...]} [0,1]
    """
    if image_path_or_base64.startswith('data:image;base64,'):
        img_b64 = image_path_or_base64.split(',')[1]
    elif os.path.exists(image_path_or_base64):
        img_b64 = image_to_base64(image_path_or_base64).split(',')[1]
    else:
        img_b64 = image_path_or_base64
    
    try:
        img_data = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_data))
        img_w, img_h = pil_img.size
    except:
        img_w, img_h = 1, 1
    
    expert_results = {}
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


def find_entity_token_indices(entity_text, dicts, inputs, img_end,
                              min_token_ratio=0.3):
    """
     search prompt tokentoken
    
    Args:
        entity_text: str, , e.g. "dog, motorcycle with red color"
        dicts: dict, {token_id: token_text}
        inputs: model inputs
        img_end: list, token
        min_token_ratio: float, TAD
            tokentoken
            bbox
    Returns:
        entity_token_indices: list of int, token
        entity_token_map: dict, {entity_name: [token_positions]}
    """
    entities = [e.strip().lower() for e in entity_text.split(',') if e.strip()]
    
    start_k = img_end[-1] + 1
    end_k = len(inputs['input_ids'][0])
    
    token_positions = []  # [(position, token_text), ...]
    for k in range(start_k, end_k):
        token_id = inputs['input_ids'][0][k].cpu().item()
        if token_id in dicts:
            token_positions.append((k, dicts[token_id]))
        else:
            token_positions.append((k, ""))
    
    full_text = ""
    char_to_token_idx = [] # char → token_positions
    for tidx, (pos, text) in enumerate(token_positions):
        for c in text:
            char_to_token_idx.append(tidx)
        full_text += text
    
    full_text_lower = full_text.lower()
    
    entity_token_indices = set()
    entity_token_map = {}
    
    for entity in entities:
        entity_lower = entity.lower().strip()
        if not entity_lower:
            continue
        
        matched_tids = set()
        search_start = 0
        while True:
            idx = full_text_lower.find(entity_lower, search_start)
            if idx == -1:
                break
            for char_pos in range(idx, min(idx + len(entity_lower), len(char_to_token_idx))):
                tidx = char_to_token_idx[char_pos]
                matched_tids.add(tidx)
            search_start = idx + 1
        
        positions = [token_positions[tidx][0] for tidx in sorted(matched_tids)]
        entity_token_map[entity_lower] = positions
        entity_token_indices.update(positions)
    
    total_prompt_tokens = end_k - start_k
    min_tokens = max(3, int(total_prompt_tokens * min_token_ratio))
    
    if len(entity_token_indices) < min_tokens:
        expanded = set(entity_token_indices)
        all_prompt_positions = [tp[0] for tp in token_positions]
        
        for eidx in sorted(entity_token_indices):
            try:
                pidx = all_prompt_positions.index(eidx)
            except ValueError:
                continue
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


def compute_expert_exclusive_bboxes(expert_results, tad_bboxes, iou_threshold=0.1,
                                     max_exclusive_per_entity=2):
    """
    TADbbox
    
    : exclusive bbox(max_exclusive_per_entity)
    (person)
    
    Args:
        expert_results: dict, {entity: [[x0,y0,x1,y1], ...]}
        tad_bboxes: list of [x0,y0,x1,y1], TADbbox
        iou_threshold: float, IoUTAD
        max_exclusive_per_entity: int, exclusive bbox
    Returns:
        exclusive_bboxes: list of [x0,y0,x1,y1], bbox
        exclusive_info: dict, 
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
            max_iou = 0.0
            if tad_bboxes:
                for tbox in tad_bboxes:
                    iou = bbox_iou(ebox, tbox)
                    max_iou = max(max_iou, iou)
            
            if max_iou < iou_threshold:
                entity_exclusive.append(ebox)
        
        if entity_exclusive and max_exclusive_per_entity > 0:
            entity_exclusive.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            entity_exclusive = entity_exclusive[:max_exclusive_per_entity]
        
        exclusive_bboxes.extend(entity_exclusive)
        if entity_exclusive:
            exclusive_info[entity] = len(entity_exclusive)
    
    return exclusive_bboxes, exclusive_info


def expert_bboxes_to_attention_mask(expert_bboxes, att_h, att_w, sigma=2.0):
    """bbox"""
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
     ρ ∈ [0, 1]
    ρ = 
    """
    if not entity_list or not expert_results:
        return 0.0
    detected = sum(1 for e in entity_list 
                   if e.strip() in expert_results and len(expert_results[e.strip()]) > 0)
    return detected / len(entity_list)


def fuse_tad_and_expert(tad_att, expert_att, fusion_mode="adaptive", expert_weight=0.5,
                        expert_reliability=1.0):
    """
    Rank-Based Evidence Fusion (B).
    
    ======== Insight ========
    
     a(p)∈[0,1] min-max 
    (log-odds):
      (1) logit(1-ε)→+∞ 
      (2) bbox
    
    : **(rank space)** 
    
    ======== ========
    
     rank : R(x) = rank(x) / total_pixels
     ∈ [0,1]
    
    :
      F(p) = (1-α)·R_tad(p) + α·R_expert(p)
    
     α = ρ·w ():
      - ρ=0 (): α=0, F=R_tad, TAD
      - ρ=1 (): α=w, TAD
    
    ======== ========
    
    1 (): R(·) α- → TAD
    2 (): R(·) → bbox
    3 (): ρ→0 ⇒ α→0 ⇒ F→R_tad
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
        def to_rank_map(att):
            """2D[0,1]"""
            flat = att.flatten()
            order = flat.argsort().argsort()
            rank = order.astype(np.float32) / max(len(flat) - 1, 1)
            return rank.reshape(att.shape)
        
        R_tad = to_rank_map(tad_att)
        R_expert = to_rank_map(expert_att)
        
        alpha = expert_reliability * expert_weight  # α = ρ·w
        
        fused = (1.0 - alpha) * R_tad + alpha * R_expert
        
        if fused.max() > fused.min():
            fused = (fused - fused.min()) / (fused.max() - fused.min())
        
        return fused
    else:
        return tad_att


def process(dicts, start_k, end_k, attention, inputs, img_start, img_end, sig):
    """ HiDe TAD search prompt token """
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


def process_egaf(dicts, start_k, end_k, attention, inputs, img_start, img_end, sig,
                 expert_bboxes_per_img=None, egaf_fusion_mode="adaptive",
                 egaf_expert_weight=0.5, entity_token_indices=None,
                 entity_token_map=None, expert_reliability=1.0):
    """
    EGAF 
    
     process :
      1. SAAA (Semantic-Aware Attention Aggregation): token max 
      2. Rank-Based Fusion: 
      3. TAD: token
    """
    accept_att = {}
    raw_tad_att = {}
    noise_token_num = 8
    noise_mean = [[0 for k in range(noise_token_num)] for i in range(len(inputs["image_grid_thw"]))]

    expert_att_maps = {}
    if expert_bboxes_per_img:
        for img_idx_e, bboxes_e in expert_bboxes_per_img.items():
            if img_idx_e < len(inputs["image_grid_thw"]):
                grid = inputs["image_grid_thw"][img_idx_e]
                att_h, att_w = grid[1].item() // 2, grid[2].item() // 2
                expert_att_maps[img_idx_e] = expert_bboxes_to_attention_mask(
                    bboxes_e, att_h, att_w, sigma=2.0)

    entity_set = set(entity_token_indices) if entity_token_indices else None

    for k in range(start_k, end_k - 4):
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

    entity_groups = []
    handled_tokens = set()
    if entity_token_map:
        for entity_name, tokens in entity_token_map.items():
            if tokens:
                group = sorted(tokens)
                entity_groups.append(group)
                handled_tokens.update(group)
                
    if entity_set is not None:
        for t in sorted(entity_set):
            if t not in handled_tokens and t >= start_k + noise_token_num:
                entity_groups.append([t])
    else:
        for img_idx in raw_tad_att:
            for t in raw_tad_att[img_idx]:
                if t not in handled_tokens:
                    entity_groups.append([t])
                    handled_tokens.add(t)

    for img_idx in raw_tad_att:
        if img_idx not in accept_att:
            accept_att[img_idx] = {}
            
        for group in entity_groups:
            group_atts = [raw_tad_att[img_idx][t] for t in group if t in raw_tad_att[img_idx]]
            if not group_atts:
                continue
                
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
                    
            target_k = group[-1]
            accept_att[img_idx][target_k] = agg_att

    return accept_att


######################################################################
# GRACE: Gradient-guided Relative Attention with Consistency-aware
#        Expert Verification
#
######################################################################

def process_grace(dicts, start_k, end_k, attention, inputs, img_start, img_end, sig,
                  entity_token_indices=None, entity_token_map=None,
                  noise_token_num=8):
    """
    GRACE TAD process token 

     TAD 
      - token token Max Pooling 
      - noise_token_num token token 
      - t TAD Otsu
    GRACE 
      - SAM3 bbox + 
      - SAM3 bbox 

     tmp/HiDe TAD 
      1. Search Prompt noise_token_num 8 token 
         
      2. token 
         a. Gaussian blur + min-max [0,1]
         b. accept_s -= noise_mean
         c. 0
         d. min-max 

    Args:
        dicts: {token_id: token_text}
        start_k: token img_end[-1]+1
        end_k: 
        attention: list of per-layer attention tensors
        inputs: model inputs
        img_start / img_end: token 
        sig: Gaussian blur sigma
        entity_token_indices: [] 
        entity_token_map: [] 
        noise_token_num: int N token 8

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
     SAM3 

    - IoU SAM3 
    - max_per_entity 

    Args:
        sam3_results: dict, {entity_name: [[x0, y0, x1, y1], ...]} [0,1]
        max_per_entity: int bbox 

    Returns:
        supplement_bboxes: list of [x0, y0, x1, y1]
        entity_labels: list of str supplement_bboxes bbox 
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
######################################################################

def filter_noise_bboxes(bboxes, min_area_ratio=0.001, edge_margin=0.02):
    """
    2-ACR: bbox
    
    /bbox
    
    Args:
        bboxes: list of [x0, y0, x1, y1] ()
        min_area_ratio: float, 
        edge_margin: float, 
    Returns:
        filtered_bboxes: bbox
    """
    if not bboxes:
        return bboxes
    
    filtered = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        
        if area < min_area_ratio:
            continue
        
        is_corner = False
        corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for cx, cy in corners:
            if (abs(x0 - cx) < edge_margin or abs(x1 - cx) < edge_margin) and \
               (abs(y0 - cy) < edge_margin or abs(y1 - cy) < edge_margin):
                if area < 0.005: # bbox
                    is_corner = True
                    break
        if is_corner:
            continue
        
        filtered.append(bbox)
    
    return filtered if filtered else bboxes # 


def extract_answer_letter(text):
    """"""
    import re
    match = re.search(r'<FINAL_OUTPUT>\s*\(?([A-D])\)?\s*', text)
    if match:
        return match.group(1)
    matches = re.findall(r'\(([A-D])\)', text)
    if matches:
        return matches[-1]
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches[-1]
    return None


def confidence_routing(ori_answer_text, hide_answer_text, ori_letter, hide_letter):
    """
    2-ACR: 
    
    HiDe
    
    :
    1. 
    2. HiDeHiDe
    3. 
    4. HiDe
    
    Returns:
        best_answer_text: str
        best_letter: str
        route: str, "ori" or "hide"
    """
    if ori_letter and hide_letter and ori_letter == hide_letter:
        return hide_answer_text, hide_letter, "agree"
    
    if not hide_letter and ori_letter:
        return ori_answer_text, ori_letter, "ori"
    
    if not ori_letter and hide_letter:
        return hide_answer_text, hide_letter, "hide"
    
    ori_has_reasoning = len(ori_answer_text) > 100 and any(
        kw in ori_answer_text.lower() for kw in ['therefore', 'because', 'since', 'looking at']
    )
    hide_has_reasoning = len(hide_answer_text) > 100 and any(
        kw in hide_answer_text.lower() for kw in ['therefore', 'because', 'since', 'looking at']
    )
    
    if hide_has_reasoning and not ori_has_reasoning:
        return hide_answer_text, hide_letter, "hide"
    
    if ori_has_reasoning and not hide_has_reasoning:
        return ori_answer_text, ori_letter, "ori"
    
    return hide_answer_text, hide_letter, "hide"


######################################################################
######################################################################

def expand_bbox(bbox, expand_ratio=0.3, img_w=1.0, img_h=1.0):
    """
    bboxPMGVV
    
    Args:
        bbox: [x0, y0, x1, y1] ()
        expand_ratio: 
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
    """bbox"""
    if not bboxes:
        return None
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]


def compute_bbox_coverage(bboxes):
    """bbox"""
    if not bboxes:
        return 0.0
    merged = merge_all_bboxes(bboxes)
    if merged is None:
        return 0.0
    x0, y0, x1, y1 = merged
    return (x1 - x0) * (y1 - y0)


def build_verification_prompt(entities, has_crop=True):
    """
    3-PMGVV: prompt
    
    
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
    

    :param path: 
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created successfully at {path}")
    except Exception as e:
        print(f"Failed to create directory at {path}: {e}")

def load_json_to_list(json_path: str) -> List[Dict]:
    """
     JSON 
    
    :
        json_path (str): JSON 
    
    :
        List[Dict]: 
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON ")

    return data

def serialize_dict(my_dict, file_path):
    """
     JSON .jsonl 
    
     JSONL 
    
    :
        my_dict: ndarraynp.int64 
        file_path: .jsonl 
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

    serialized_dict = serialize_obj(my_dict)

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serialized_dict, ensure_ascii=False, indent=4) + '\n')

def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def pil_to_base64(pil_img, format="PNG"):
    buffered = BytesIO()
    img_format = pil_img.format if pil_img.format else format
    pil_img.save(buffered, format=img_format) # 
    encoded_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def swap_and_rebuild_dict(nested_dict):
    """
     key 
    
    :
        nested_dict: {outer_key: {inner_key: value}}
    :
        new_dict: {inner_key: {outer_key: value}}
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
    
    
    :
        matrix (np.ndarray): NxN 
        k (float): 2
        merge_distance_ratio (float): 
    
    :
        List[List[int]]: bounding box [x1, y1, x2, y2]
    """
    H, W = matrix.shape
    diag_length = np.sqrt(H**2 + W**2)
    merge_distance_threshold = diag_length * merge_distance_ratio # 

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

    boxes = []
    for coords in regions:
        y_min, x_min = np.min(coords, axis=0)
        y_max, x_max = np.max(coords, axis=0)
        boxes.append([x_min, y_min, x_max, y_max])  # [x1,y1,x2,y2]

    n = len(boxes)
    to_merge = []

    def box_center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    for i, j in combinations(range(n), 2):
        c1 = box_center(boxes[i])
        c2 = box_center(boxes[j])
        dist = np.linalg.norm(c1 - c2)
        if dist < merge_distance_threshold:
            to_merge.append((i, j))


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

    final_boxes = [list(map(int, box)) for box in merged_boxes.values()]

    def remove_nested_boxes(boxes):
        if not boxes:
            return []

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
