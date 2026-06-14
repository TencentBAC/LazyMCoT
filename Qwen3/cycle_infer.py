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

def get_available_gpus(max_memory_mb=8000, max_gpus=None):
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

def main(datasetdir, savedir, max_pixels, Parallels, sig, thre, para_nums=6,
         batch_size=4, enable_saaa=False, enable_acr=False, enable_pmgvv=False,
         egaf_fusion_mode="adaptive", egaf_expert_weight=0.5,
         egaf_expert_url="http://localhost:8002/predict",
         skip_ori=False,
         enable_grace=False,
         grace_sam3_url="http://localhost:8002/predict",
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
    # 分割数据集到不同 GPU 上
    # 将 dataset 划分为 num_gpus 份，每份尽量均衡
    splits = np.array_split(dataset, len(available_gpus))
    print("文件加载完成")
    if not Parallels:
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            cycle_epoch_infer(gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre,
                              batch_size=batch_size,
                              enable_saaa=enable_saaa, enable_acr=enable_acr,
                              enable_pmgvv=enable_pmgvv,
                              egaf_fusion_mode=egaf_fusion_mode,
                              egaf_expert_weight=egaf_expert_weight,
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
                args=(gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre),
                kwds=dict(
                    batch_size=batch_size,
                    enable_saaa=enable_saaa, enable_acr=enable_acr,
                    enable_pmgvv=enable_pmgvv,
                    egaf_fusion_mode=egaf_fusion_mode,
                    egaf_expert_weight=egaf_expert_weight,
                    egaf_expert_url=egaf_expert_url,
                    skip_ori=skip_ori,
                    enable_grace=enable_grace,
                    grace_sam3_url=grace_sam3_url,
                    grace_max_sam3_per_entity=grace_max_sam3_per_entity,
                    enable_router=enable_router,
                    router_report_path=router_report_path,
                    router_alpha=router_alpha,
                    heatmap_save_dir=heatmap_save_dir,
                ),
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
    threshold = [0.5]
    seed = 2077
    random.seed(seed)
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)

    # ── 数据集配置 ──
    datasetdir = f"/path/to/data/benchmark.json"
    savejson = f"/path/to/output/results_qwen3_grace_router_v2.json"

    # ── 创新点开关 ──
    enable_grace = True          # GRACE 模式（SAM3 + TAD）
    grace_sam3_url = "http://localhost:8002/predict"
    grace_max_sam3_per_entity = 10
    skip_ori = False              # 跳过直接回答（节省推理时间）
    batch_size = 1               # batch 推理大小
    heatmap_save_dir = None      # 设为路径则保存热力图，如 "/path/to/output/heatmaps"

    # ── 路由器配置 ──
    # enable_router=True  ⇒ 强制 skip_ori=False（router 需要 direct-answer logits）
    enable_router = True
    router_report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "params", "Qwen3", "router_report.json",
    )
    router_alpha = None          # 覆盖 router 训练时的 α，按新 α 重算 s_floor

    main(datasetdir, savejson, maxp, Parallels, sigma, threshold, 4,
         batch_size=batch_size,
         skip_ori=skip_ori,
         enable_grace=enable_grace,
         grace_sam3_url=grace_sam3_url,
         grace_max_sam3_per_entity=grace_max_sam3_per_entity,
         enable_router=enable_router,
         router_report_path=router_report_path,
         router_alpha=router_alpha,
         heatmap_save_dir=heatmap_save_dir)
