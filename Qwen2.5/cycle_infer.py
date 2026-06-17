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
    print(f"❌ : {e}")
    print(f"Traceback:\n{traceback.format_exc()}")

def get_available_gpus(max_memory_mb=1000, max_gpus=None):
    """
     max_memory_mb GPU ID 

    Args:
        max_memory_mb: MB""
        max_gpus: GPUNone 

    Returns:
         GPU ID [2, 0, 3]
    """
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        used_memory = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        
        gpu_memory_pairs = [(i, mem) for i, mem in enumerate(used_memory)]
        gpu_memory_pairs.sort(key=lambda x: x[1]) # 
        
        available_gpus = [gpu_id for gpu_id, mem in gpu_memory_pairs if mem < max_memory_mb]
        
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
        print("❌ GPU < 8000MB")
        return
    print(f"✅ {len(available_gpus)} GPU < 8000MB: {available_gpus}")
    print(f"📦 batch_size = {batch_size}")
    print(f"🔬 : SAAA={enable_saaa}, ACR={enable_acr}, PMGVV={enable_pmgvv}, "
          f"ROUTER={enable_router} (α={router_alpha if router_alpha is not None else 'trained-default(0.0)'})")
    splits = np.array_split(dataset, len(available_gpus))
    print("")
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
        for res in tqdm(results, desc=""):
            res.wait() # error_callback
        pool.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    maxp = 16384
    Parallels = True
    sigma = [3]
    threshold = [0.7]
    seed = 2077
    random.seed(seed)
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    datasetdir = f"/path/to/data/benchmark.json"
    

    enable_grace = True
    grace_sam3_url = "http://localhost:8002/predict" # model_service_v2.py (SAM3) 
    grace_max_sam3_per_entity = 10 # SAM3 bbox 

    enable_saaa = False # enable_grace=True 
    egaf_fusion_mode = "adaptive"
    egaf_expert_weight = 0.5
    egaf_expert_url = "http://localhost:8002/predict"

    enable_acr = False
    acr_min_area = 0.001
    acr_edge_margin = 0.02

    enable_pmgvv = False
    pmgvv_expand_ratio = 0.3
    # ==================================================================

    # RouterV2 (Cost-Aware Conformal Safe-Skip Router)：
    #
    #   model_type = "gbdt"
    #
    enable_router = True
    router_report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "params", "Qwen2.5", "router_report.json",
    )

    #
    #
    #
    #
    router_alpha = None # α=0.0 0.005/0.01/0.02/0.05
    # =============================================================================

    skip_ori = False # 
    # =====================================================

    #   {sample_id}_img{img_idx}_s{sigma}_t{thresh}_agg_heatmap.png
    # heatmap_save_dir = None
    heatmap_save_dir = None
    # =========================================================

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
