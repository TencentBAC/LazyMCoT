import os
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
import json
import torch.multiprocessing as mp
import multiprocessing
from joblib import Parallel, delayed
import time
import random
from PIL import Image
import io
import numpy as np
import base64
import gc
import base64
import multiprocessing
from multiprocessing import Pool
from accelerate import infer_auto_device_map, dispatch_model
import shutil
from inference import cycle_epoch_infer
from utiles import *
import traceback
import subprocess

Image.MAX_IMAGE_PIXELS = 28000000000

def log_error(e):
    print(f"❌ 异常发生: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")

def get_available_gpus(max_memory_mb=1000, max_gpus=None):
    """
    获取显存占用低于 max_memory_mb 的 GPU 设备 ID 列表，并按占用从小到大排序返回

    Args:
        max_memory_mb: 最大允许显存占用（MB），低于此值才认为是"可用"
        max_gpus: 最多返回几个 GPU，None 表示返回所有符合条件的

    Returns:
        按显存占用升序排列的可用 GPU ID 列表，例如 [2, 0, 3]
    """
    try:
        # 使用 nvidia-smi 获取每张 GPU 的显存使用情况
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        # 解析显存使用量（MB）
        used_memory = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        
        # 创建 (gpu_id, memory_used) 的列表并按显存使用量升序排序
        gpu_memory_pairs = [(i, mem) for i, mem in enumerate(used_memory)]
        gpu_memory_pairs.sort(key=lambda x: x[1])  # 按显存使用量从小到大排序
        
        # 筛选低于阈值的 GPU，并保留排序顺序
        available_gpus = [gpu_id for gpu_id, mem in gpu_memory_pairs if mem < max_memory_mb]
        
        # 限制返回数量
        if max_gpus is not None:
            available_gpus = available_gpus[:max_gpus]
        
        return available_gpus

    except Exception as e:
        print(f"Error detecting GPU memory: {e}")
        return []

def main(datasetdir, savedir, max_pixels, Parallels, sig, thre, para_nums=6, batch_size=4,
         enable_saaa=False, enable_acr=False, enable_pmgvv=False,
         acr_min_area=0.001, acr_edge_margin=0.02, pmgvv_expand_ratio=0.3,
         egaf_fusion_mode="adaptive", egaf_expert_weight=0.5,
         egaf_expert_url="http://localhost:8001/predict",
         skip_ori=False,
         enable_grace=False,
         grace_sam3_url="http://localhost:8001/predict",
         grace_max_sam3_per_entity=3,
         enable_router=False,
         router_report_path=None,
         router_alpha=None,
         heatmap_save_dir=None):
    if not Parallels: para_nums = 1
    dataset = load_dataset_Vstar_json(datasetdir)
    # dataset = load_dataset_hrbench(datasetdir)
    random.shuffle(dataset)
    available_gpus = get_available_gpus(max_memory_mb=8000, max_gpus=para_nums)
    if len(available_gpus) == 0:
        print("❌ 没有找到符合条件的空闲 GPU（占用显存 < 8000MB")
        return
    print(f"✅ 找到 {len(available_gpus)} 个可用 GPU（占用显存 < 8000MB）: {available_gpus}")
    print(f"📦 每张卡 batch_size = {batch_size}")
    print(f"🔬 创新点开关: SAAA={enable_saaa}, ACR={enable_acr}, PMGVV={enable_pmgvv}, "
          f"ROUTER={enable_router} (α={router_alpha if router_alpha is not None else 'trained-default(0.0)'})")
    # 分割数据集到不同 GPU 上
    # 将 dataset 划分为 num_gpus 份，每份尽量均衡
    splits = np.array_split(dataset, len(available_gpus))
    print("文件加载完成")
    if not Parallels:
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            cycle_epoch_infer(gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre, batch_size,
                              enable_saaa=enable_saaa, enable_acr=enable_acr, enable_pmgvv=enable_pmgvv,
                              acr_min_area=acr_min_area, acr_edge_margin=acr_edge_margin,
                              pmgvv_expand_ratio=pmgvv_expand_ratio,
                              egaf_fusion_mode=egaf_fusion_mode, egaf_expert_weight=egaf_expert_weight,
                              egaf_expert_url=egaf_expert_url,
                              skip_ori=skip_ori,
                              enable_grace=enable_grace,
                              grace_sam3_url=grace_sam3_url,
                              grace_max_sam3_per_entity=grace_max_sam3_per_entity,
                              enable_router=enable_router,
                              router_report_path=router_report_path,
                              router_alpha=router_alpha,
                              heatmap_save_dir=heatmap_save_dir)
    else:
        pool = Pool(processes=len(available_gpus))
        results = []
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            res = pool.apply_async(
                cycle_epoch_infer,
                args=(gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre, batch_size,
                      enable_saaa, enable_acr, enable_pmgvv,
                      acr_min_area, acr_edge_margin, pmgvv_expand_ratio,
                      egaf_fusion_mode, egaf_expert_weight, egaf_expert_url,
                      skip_ori, enable_grace, grace_sam3_url, grace_max_sam3_per_entity,
                      enable_router, router_report_path, router_alpha,
                      heatmap_save_dir),
                error_callback=log_error
            )
            results.append(res)
        pool.close()
        # 等待并获取结果（可选：获取返回值）
        for res in tqdm(results, desc="等待所有进程完成"):
            res.wait()  # 触发 error_callback
        pool.join()

if __name__ == "__main__":
    # 👇 必须放在这里！
    mp.set_start_method('spawn', force=True)
    maxp = 16384
    #并行多开线程计算，自动寻找满足条件的GPU
    Parallels = True
    #超参数
    sigma = [3]
    threshold = [0.7]
    seed = 2077
    random.seed(seed)
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    datasetdir = f"/path/to/data/benchmark.json"
    
    # ==================== 方案开关（消融实验超参数）====================

    # ── GRACE 模式（推荐，新方案）─────────────────────────────────────
    # 相对注意力归一化 + Otsu 二值化 + SAM3 文本提示补充 bbox
    # 最终 bounding_boxes = 注意力 bbox ∪ SAM3 检测 bbox
    enable_grace = True
    grace_sam3_url = "http://localhost:8002/predict"   # model_service_v2.py (SAM3) 服务地址
    grace_max_sam3_per_entity = 10                       # 每个实体最多保留的 SAM3 bbox 数

    # ── 旧 EGAF 模式（向后兼容，enable_grace=True 时自动禁用）────────
    # EGAF: Rank-Based Fusion 将专家注意力掩码融合进 TAD 注意力图
    enable_saaa = False   # enable_grace=True 时此开关不生效
    egaf_fusion_mode = "adaptive"
    egaf_expert_weight = 0.5
    egaf_expert_url = "http://localhost:8002/predict"

    # ── ACR - 自适应置信度路由 ─────────────────────────────────────
    enable_acr = False
    acr_min_area = 0.001
    acr_edge_margin = 0.02

    # ── PMGVV - 渐进式多粒度视觉验证 ─────────────────────────────────
    enable_pmgvv = False
    pmgvv_expand_ratio = 0.3
    # ==================================================================

    # ==================== Difficulty-Aware Router v2（无硬阈值）====================
    # RouterV2 (Cost-Aware Conformal Safe-Skip Router)：
    #   - 特征：2 维 [answer_topp, logit_gap_opt_nonopt]（纯首 token 分布统计）
    #   - 决策：run_grace ⟺ s(x) ≥ s_floor
    #   - s_floor 由训练集 OOF 上 (ori_wrong \ ow_gw) 的 Q_α 分位数自动得出
    #   - 不使用任何文本关键词 / regex 派生特征，跨数据集泛化性强
    #
    # 【当前 router_v2_report.json】
    #   model_type = "gbdt"
    #   训练 α     = 0.0      （严格 must-recall 100% 召回）
    #   s_floor    = -4.607   （即 must-recall OOF 分数的最小值）
    #
    # 启用后 inference.py 会自动：
    #   (1) direct-answer 改用"直接吐字母"prompt（让首 token 就是选项字母）
    #   (2) 收集完整词表 first_logits → 计算路由器特征
    #   (3) 用 RouterV2.decide 判定本样本是否走 GRACE 流程
    # 注意：enable_router=True 时 skip_ori 会被强制改为 False（需要 direct 前向）
    enable_router = True
    # 路由器配置文件（由 tools/train_router_v2.py 生成）
    router_report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "params", "Qwen2.5", "router_report.json",
    )

    # ── Router α 超参入口（推理时动态调节 s_floor，无需重训） ──
    # α 语义：允许 α 比例的 must-recall (ow\ow_gw) 样本被漏召回，
    #        换取更大的 ori skip 空间。
    #
    # ⚠️ 默认请置为 None，表示严格沿用训练时保存在 report 中的 α=0.0
    #   （=必召回 100%，cgw_saved=0，oc\cgw skip ≈ 25%，最保守最安全）。
    #
    # 仅当你明确希望在保留的 α=0 baseline 基础上做 α 敏感性 / 消融时，
    #   才设为显式数值；RouterV2.set_alpha() 会基于训练时保存的
    #   mr_oof_scores_sorted 重新取分位数，完全等价于用新 α 重训。
    #
    # 推荐档位（同 train_router_v2.py 的 α sensitivity 表）：
    # None    → 使用训练时保存的 α（= 0.0，严格 100% must-recall，默认）
    # 0.0     → 显式指定 0.0（与 None 等价，强制按 mr_oof.min() 计算 s_floor）
    # 0.005   → 允许 ~1.4 条 must-recall 漏，cgw 救回 0，oc\cgw skip ≈ 45%
    # 0.01    → 允许 ~2.8 条 must-recall 漏，cgw 救回 3，oc\cgw skip ≈ 54%  (net≈-5 条)
    # 0.02    → 允许 ~5.5 条漏，cgw 救回 5，oc\cgw skip ≈ 56%   (net≈-11 条)
    # 0.05    → 允许 ~14 条漏，cgw 救回 7，oc\cgw skip ≈ 64%    (net≈-22 条)
    #
    # 【警告】 α > 0 时净准确率通常会下降（cgw_saved - ow_leak < 0），
    # 仅在需要大幅节省 GRACE 计算、可接受小幅精度损失时使用。
    router_alpha = None   # 保持 α=0.0（训练配置）；需要敏感性实验时改为 0.005/0.01/0.02/0.05
    # =============================================================================

    # ==================== 推理控制开关 ====================
    skip_ori = False  # 跳过原始基模推理，节省时间
    # =====================================================

    # ==================== 注意力热力图保存 ====================
    # None 表示不保存；设置为目录路径则对每个样本保存聚合注意力热力图：
    #   {sample_id}_img{img_idx}_s{sigma}_t{thresh}_agg_heatmap.png
    # 图像内容：原图 + jet 热力图叠加 + 绿色注意力bbox + 橙色虚线SAM3 bbox
    # heatmap_save_dir = None
    heatmap_save_dir = None
    # =========================================================

    # 生成带方案标识的保存路径
    innovation_tag = "_GRACE" if enable_grace else ""
    if not enable_grace and enable_saaa: innovation_tag += "_EGAF"
    if enable_acr: innovation_tag += "_ACR"
    if enable_pmgvv: innovation_tag += "_PMGVV"
    if enable_router: innovation_tag += "_ROUTER"
    if not innovation_tag: innovation_tag = "_baseline"

    savejson = f"/path/to/output/results_qwen25_grace_router_v2.json"

    main(datasetdir, savejson, maxp, Parallels, sigma, threshold, 4, batch_size=1,
         enable_saaa=enable_saaa, enable_acr=enable_acr, enable_pmgvv=enable_pmgvv,
         acr_min_area=acr_min_area, acr_edge_margin=acr_edge_margin,
         pmgvv_expand_ratio=pmgvv_expand_ratio,
         egaf_fusion_mode=egaf_fusion_mode, egaf_expert_weight=egaf_expert_weight,
         egaf_expert_url=egaf_expert_url,
         skip_ori=skip_ori,
         enable_grace=enable_grace,
         grace_sam3_url=grace_sam3_url,
         grace_max_sam3_per_entity=grace_max_sam3_per_entity,
         enable_router=enable_router,
         router_report_path=router_report_path,
         router_alpha=router_alpha,
         heatmap_save_dir=heatmap_save_dir)
