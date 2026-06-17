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

def get_available_gpus(max_memory_mb=8000, max_gpus=None):
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
        print("❌ GPU < 8000MB")
        return
    print(f"✅ {len(available_gpus)} GPU < 8000MB: {available_gpus}")
    splits = np.array_split(dataset, len(available_gpus))
    print("")
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
        for res in tqdm(results, desc=""):
            res.wait() # error_callback
        pool.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    maxp = 16384
    Parallels = True
    sigma = [3]
    threshold = [0.5]
    seed = 2077
    random.seed(seed)
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)

    datasetdir = f"/path/to/data/benchmark.json"
    savejson = f"/path/to/output/results_qwen3_grace_router_v2.json"

    enable_grace = True # GRACE SAM3 + TAD
    grace_sam3_url = "http://localhost:8002/predict"
    grace_max_sam3_per_entity = 10
    skip_ori = False # 
    batch_size = 1 # batch 
    heatmap_save_dir = None # "/path/to/output/heatmaps"

    enable_router = True
    router_report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "params", "Qwen3", "router_report.json",
    )
    router_alpha = None # router α α s_floor

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
