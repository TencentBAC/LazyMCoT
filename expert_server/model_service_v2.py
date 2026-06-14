#!/usr/bin/env python3
"""
Vision expert service using SAM3 text-prompted detection + segmentation.

SAM3 directly provides masks, boxes, and scores from a text prompt,
eliminating the need for a separate grounding model.

API is identical to model_service.py:
    POST /predict  { "image": <base64>, "text": <prompt> }
    -> { "boxes": [...], "labels": [...], "masks": [...] }
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import time
import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Any

# ── Sanitise LOG_LEVEL before importing sam3 ─────────────────────────────
# sam3's logger only accepts DEBUG/INFO/WARNING/ERROR/CRITICAL.
# uvicorn (or other tools) may set LOG_LEVEL to non-standard values like
# "TRACE", which causes an AssertionError at import time.
_SAM3_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_env_log = os.environ.get("LOG_LEVEL", "").upper()
if _env_log and _env_log not in _SAM3_VALID_LEVELS:
    os.environ["LOG_LEVEL"] = "DEBUG"   # closest safe fallback

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import gc

# ── SAM3 ─────────────────────────────────────────────────────────────────
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ═══════════════════════════════════════════════════════════════════════════
# Model paths — adjust if your checkpoints live elsewhere
# ═══════════════════════════════════════════════════════════════════════════
SAM3_CKPT_PATH = "/path/to/ckpt/sam3/sam3.pt"


# ═══════════════════════════════════════════════════════════════════════════
# Device helpers
# ═══════════════════════════════════════════════════════════════════════════
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _get_device()


# ═══════════════════════════════════════════════════════════════════════════
# SAM3 wrapper (text-prompt detection + segmentation)
# ═══════════════════════════════════════════════════════════════════════════
class SAM3Model:
    """
    Wraps SAM3's build_sam3_image_model + Sam3Processor.
    Uses set_text_prompt to directly obtain masks, boxes, and scores
    from a text prompt — no external grounding model needed.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.model = build_sam3_image_model(
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            device=device,
        )
        self.processor = Sam3Processor(self.model)
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        image_pil: Image.Image,
        text_prompt: str,
    ) -> dict:
        """
        Use SAM3's native text-prompted detection + segmentation.

        Returns:
            dict with "boxes" (list[list[float]], xyxy pixel coords),
                       "masks" (list[list[list[int]]], binary masks),
                       "labels" (list[str]),
        """
        inference_state = self.processor.set_image(image_pil)
        output = self.processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt,
        )

        masks = output.get("masks", [])
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])

        # Convert tensors to numpy/lists (.float() to handle bfloat16)
        if hasattr(masks, "cpu"):
            masks = masks.cpu().float().numpy()
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().float().numpy()
        if hasattr(scores, "cpu"):
            scores = scores.cpu().float().numpy()

        # Squeeze extra dimensions: (N, 1, H, W) -> (N, H, W)
        if isinstance(masks, np.ndarray) and masks.ndim == 4:
            masks = masks.squeeze(1)

        # Convert to serialisable lists
        if hasattr(masks, "tolist"):
            masks_list = masks.astype(int).tolist()
        elif isinstance(masks, np.ndarray):
            masks_list = masks.astype(int).tolist()
        else:
            masks_list = list(masks) if masks is not None else []

        if hasattr(boxes, "tolist"):
            boxes_list = boxes.tolist()
        elif isinstance(boxes, np.ndarray):
            boxes_list = boxes.tolist()
        else:
            boxes_list = list(boxes) if boxes is not None else []

        # Build label list — use text_prompt as label for each detection
        n_detections = len(boxes_list)
        labels_list = [text_prompt] * n_detections

        return {
            "boxes": boxes_list,
            "labels": labels_list,
            "masks": masks_list,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Request / Response schemas (identical to model_service.py)
# ═══════════════════════════════════════════════════════════════════════════
class ImageRequest(BaseModel):
    image: str  # base64-encoded RGB image
    text: str   # prompt text


class PredictionResponse(BaseModel):
    boxes: List[List[float]]
    labels: List[str]
    masks: List[List[List[int]]]


# ═══════════════════════════════════════════════════════════════════════════
# Batch processor (same queuing / batching logic as model_service.py)
# ═══════════════════════════════════════════════════════════════════════════
class BatchProcessor:
    """Accumulates requests and runs inference (one-by-one via SAM3 text prompt)."""

    def __init__(
        self,
        sam: SAM3Model,
        max_batch_size: int = 8,
        max_queue_size: int = 100,
    ) -> None:
        self.sam = sam
        self.max_batch_size = max_batch_size
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = False
        self.start_time = time.time()
        self.port: int = 0  # filled after argparse

    async def _print_queue_stats(self) -> None:
        while True:
            uptime = time.time() - self.start_time
            qsize = self.request_queue.qsize()
            app.state.logger.info(
                "Port=%s | Uptime=%.2fs | Queue=%d", self.port, uptime, qsize,
            )
            await asyncio.sleep(60)

    async def add_request(self, image: Image.Image, text: str) -> Dict[str, Any]:
        if self.request_queue.full():
            raise HTTPException(status_code=503, detail="Server queue is full")

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.request_queue.put((image, text, fut))

        if not self.processing:
            asyncio.create_task(self._process_batch())

        try:
            return await asyncio.wait_for(fut, timeout=3000.0)
        except asyncio.TimeoutError as e:
            raise HTTPException(status_code=504, detail="Processing timeout") from e

    async def _process_batch(self) -> None:
        self.processing = True
        try:
            while not self.request_queue.empty():
                batch_images: List[Image.Image] = []
                batch_texts: List[str] = []
                batch_futures: List[asyncio.Future] = []

                while (
                    len(batch_images) < self.max_batch_size
                    and not self.request_queue.empty()
                ):
                    try:
                        image, text, fut = self.request_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if not fut.cancelled():
                        batch_images.append(image)
                        batch_texts.append(text)
                        batch_futures.append(fut)

                if not batch_images:
                    continue

                try:
                    # SAM3 text prompt works per-image; iterate over the batch
                    for fut, img, text in zip(batch_futures, batch_images, batch_texts):
                        result = self.sam.predict(img, text)
                        fut.set_result(result)

                    gc.collect()
                    app.state.logger.info(
                        "Batch processed | size=%d | remaining=%d",
                        len(batch_images),
                        self.request_queue.qsize(),
                    )
                except Exception as e:
                    error_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_logs")
                    try:
                        os.makedirs(error_dir, exist_ok=True)
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                        err_path = os.path.join(error_dir, f"error_{ts}.txt")
                        with open(err_path, "w", encoding="utf-8") as f:
                            f.write(f"Error: {str(e)}\n\nTexts:\n")
                            for i, text in enumerate(batch_texts):
                                f.write(f"{i}: {text}\n")

                        for i, img in enumerate(batch_images):
                            img.save(os.path.join(error_dir, f"error_{ts}_img_{i}.jpg"))

                        app.state.logger.error("Batch failed; logs saved to %s", error_dir)
                    except Exception as log_err:
                        app.state.logger.error("Batch failed: %s (also failed to save logs: %s)", e, log_err)
                    for fut in batch_futures:
                        if not fut.done():
                            fut.set_exception(e)
        finally:
            self.processing = False


# ═══════════════════════════════════════════════════════════════════════════
# CLI & FastAPI setup
# ═══════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="SAM3 vision expert service")
parser.add_argument("--port", type=int, default=8000, help="Service port")
parser.add_argument("--max_batch_size", type=int, default=8, help="Max batch size")
parser.add_argument("--sam3_ckpt", type=str, default=SAM3_CKPT_PATH, help="SAM3 checkpoint path")
args = parser.parse_args()


# ── Enable TF32 for Ampere+ GPUs ─────────────────────────────────────────
if torch.cuda.is_available():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ── Instantiate SAM3 model ───────────────────────────────────────────────
print(f"Loading SAM3 from {args.sam3_ckpt} ...")
sam3_model = SAM3Model(
    ckpt_path=args.sam3_ckpt,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print("SAM3 loaded.")

processor = BatchProcessor(
    sam=sam3_model,
    max_batch_size=args.max_batch_size,
    max_queue_size=10000,
)
processor.port = args.port


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.logger = logging.getLogger("uvicorn")
    stats_task = asyncio.create_task(processor._print_queue_stats())
    try:
        yield
    finally:
        stats_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    try:
        img_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(img_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        result = await processor.add_request(image, request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(
        app,                       # 直接传入 app 对象，避免重新 import 导致模型加载两次
        host="0.0.0.0",
        port=args.port,
        limit_concurrency=10000,
        backlog=10000,
        log_level="debug",
    )
