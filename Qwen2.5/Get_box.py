import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from utiles import *
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
import cv2
import numpy as np
import base64
from io import BytesIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter

from skimage.filters import threshold_otsu
from PIL import ImageDraw


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

_LEGEND_PALETTE = [
    (255, 140, 0), # 
    (0, 191, 255), # 
    (255, 105, 180), # 
    (138, 43, 226), # 
    (255, 215, 0), # 
    (60, 179, 113), # 
    (220, 20, 60), # 
    (100, 149, 237), # 
    (255, 69, 0), # 
    (0, 206, 209), # 
    (186, 85, 211), # 
    (154, 205, 50), # 
]


def assign_entity_colors(entity_labels):
    """
    

    Args:
        entity_labels: list[str] bbox 
    Returns:
        label2color: dict[str, tuple(R,G,B)]
        bbox_colors: list[tuple] entity_labels bbox 
    """
    label2color = {}
    bbox_colors = []
    for lb in entity_labels:
        key = (lb or "").strip()
        if key not in label2color:
            label2color[key] = _LEGEND_PALETTE[len(label2color) % len(_LEGEND_PALETTE)]
        bbox_colors.append(label2color[key])
    return label2color, bbox_colors


def add_legend_to_lpd_image(
    lpd_image_b64,
    color_label_pairs,
    position="bottom",
    margin=6,
    swatch_size=18,
    font_size=None,
    background=(245, 245, 245),
    text_color=(20, 20, 20),
    border_color=(100, 100, 100),
):
    """
     LPD "■ label" 

    - LPD 
    - LPD 
    - color_label_pairs 

    Args:
        lpd_image_b64: "data:image;base64,..." base64 
        color_label_pairs: list[(rgb_tuple, label_str)]
        position: "bottom" | "right"
        margin: /
        swatch_size: 
        font_size: None 
        background / text_color / border_color: 

    Returns:
        new_image_b64: "data:image;base64,..." 
    """
    if not color_label_pairs:
        return lpd_image_b64

    try:
        if isinstance(lpd_image_b64, str) and lpd_image_b64.startswith("data:image"):
            _b64 = lpd_image_b64.split(",", 1)[1]
        else:
            _b64 = lpd_image_b64
        main_img = Image.open(BytesIO(base64.b64decode(_b64))).convert("RGBA")
        mw, mh = main_img.size

        if font_size is None:
            font_size = max(12, int(min(mw, mh) * 0.022))
        font = _get_pil_font(font_size)

        _dummy = Image.new("RGBA", (10, 10))
        _ddraw = ImageDraw.Draw(_dummy)

        def _text_size(text):
            try:
                tb = _ddraw.textbbox((0, 0), text, font=font)
                return tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                return font_size * len(text) // 2, font_size

        item_gap = max(10, font_size // 2)
        items = []   # [(color, label, text_w, text_h)]
        row_h = max(swatch_size, font_size) + 2
        for color, label in color_label_pairs:
            label_disp = str(label) if label is not None else ""
            tw, th = _text_size(label_disp)
            items.append((color, label_disp, tw, th))

        if position == "bottom":
            avail_w = mw - 2 * margin
            lines = []   # list[list[item]]
            cur_line = []
            cur_w = 0
            for it in items:
                item_w = swatch_size + 6 + it[2]
                need_w = item_w + (item_gap if cur_line else 0)
                if cur_line and cur_w + need_w > avail_w:
                    lines.append(cur_line)
                    cur_line = [it]
                    cur_w = item_w
                else:
                    cur_line.append(it)
                    cur_w += need_w
            if cur_line:
                lines.append(cur_line)

            legend_h = margin + row_h * len(lines) + margin
            new_w = mw
            new_h = mh + legend_h
            bg_rgba = background + (255,) if len(background) == 3 else background
            new_img = Image.new("RGBA", (new_w, new_h), color=(0, 0, 0, 0))
            legend_strip = Image.new("RGBA", (new_w, legend_h), color=bg_rgba)
            new_img.paste(legend_strip, (0, mh))
            new_img.paste(main_img, (0, 0))
            draw = ImageDraw.Draw(new_img)
            draw.line([(0, mh), (new_w, mh)], fill=border_color, width=1)

            for li, line in enumerate(lines):
                y_top = mh + margin + li * row_h
                x_cur = margin
                for color, label_disp, tw, th in line:
                    sw = swatch_size
                    sy = y_top + (row_h - sw) // 2
                    draw.rectangle(
                        [x_cur, sy, x_cur + sw, sy + sw],
                        fill=tuple(int(c) for c in color),
                        outline=border_color,
                    )
                    tx = x_cur + sw + 4
                    ty = y_top + (row_h - th) // 2
                    draw.text((tx, ty), label_disp, fill=text_color, font=font)
                    x_cur += sw + 6 + tw + item_gap

        else:
            item_ws = [swatch_size + 6 + it[2] for it in items]
            legend_w = margin + max(item_ws) + margin
            needed_h = margin + row_h * len(items) + margin
            new_w = mw + legend_w
            new_h = max(mh, needed_h)
            bg_rgba = background + (255,) if len(background) == 3 else background
            new_img = Image.new("RGBA", (new_w, new_h), color=(0, 0, 0, 0))
            legend_strip = Image.new("RGBA", (legend_w, new_h), color=bg_rgba)
            new_img.paste(legend_strip, (mw, 0))
            new_img.paste(main_img, (0, 0))
            draw = ImageDraw.Draw(new_img)
            draw.line([(mw, 0), (mw, new_h)], fill=border_color, width=1)
            for i, (color, label_disp, tw, th) in enumerate(items):
                y_top = margin + i * row_h
                x_cur = mw + margin
                sw = swatch_size
                sy = y_top + (row_h - sw) // 2
                draw.rectangle(
                    [x_cur, sy, x_cur + sw, sy + sw],
                    fill=tuple(int(c) for c in color),
                    outline=border_color,
                )
                tx = x_cur + sw + 4
                ty = y_top + (row_h - th) // 2
                draw.text((tx, ty), label_disp, fill=text_color, font=font)

        buf = BytesIO()
        new_img.save(buf, format="PNG")
        return "data:image;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[add_legend] : {e}")
        return lpd_image_b64


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def bbox_is_contained(inner_box, outer_box, tolerance=0.03):
    """
     inner_box outer_box 

    :
        inner_box / outer_box: [x0, y0, x1, y1]
        tolerance: 0.033%

    :
        bool: inner_box outer_box tolerance 
    """
    x0_i, y0_i, x1_i, y1_i = inner_box
    x0_o, y0_o, x1_o, y1_o = outer_box
    return (x0_i >= x0_o - tolerance and
            y0_i >= y0_o - tolerance and
            x1_i <= x1_o + tolerance and
            y1_i <= y1_o + tolerance)


def _get_pil_font(size: int):
    """ PIL """
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def apply_sam3_highlight_to_image(
    image_path_or_b64: str,
    sam3_bboxes_norm: list,
    entity_labels: list = None,
    border_color=(255, 165, 0),
    border_thickness: int = 3,
    text_color=(255, 255, 255),
    text_bg_color=(200, 100, 0),
):
    """
     SAM3 + 

     bbox bbox
     LPD 

    :
        (modified_image_b64, expanded_bboxes_norm)
        - modified_image_b64: base64 
        - expanded_bboxes_norm: list of [x0,y0,x1,y1] sam3_bboxes_norm 
           bbox bbox 
    """
    from PIL import ImageFont
    expanded_bboxes = [list(b) for b in sam3_bboxes_norm]
    try:
        if isinstance(image_path_or_b64, str) and image_path_or_b64.startswith("data:image"):
            b64 = image_path_or_b64.split(",")[1]
        elif isinstance(image_path_or_b64, str) and os.path.exists(image_path_or_b64):
            with open(image_path_or_b64, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        else:
            b64 = image_path_or_b64
        img_data = base64.b64decode(b64)
        result_pil = Image.open(BytesIO(img_data)).convert("RGB")
        W, H = result_pil.size

        draw = ImageDraw.Draw(result_pil)
        for box in sam3_bboxes_norm:
            x0n, y0n, x1n, y1n = box
            x0 = max(0, int(x0n * W))
            y0 = max(0, int(y0n * H))
            x1 = min(W - 1, int(x1n * W))
            y1 = min(H - 1, int(y1n * H))
            if x0 >= x1 or y0 >= y1:
                continue
            for tk in range(border_thickness):
                draw.rectangle(
                    [x0 + tk, y0 + tk, x1 - tk, y1 - tk],
                    outline=border_color,
                )

        if entity_labels:
            font_size = max(12, int(H * 0.015))
            font = _get_pil_font(font_size)
            pad = 3
            occupied_rects = []

            def _rects_overlap(r1, r2):
                """ (x0,y0,x1,y1) """
                return r1[0] < r2[2] and r1[2] > r2[0] and r1[1] < r2[3] and r1[3] > r2[1]

            def _find_non_overlapping_ty(tx, ty_candidate, tw_full, th_full, direction="up"):
                """ direction ty """
                ty = ty_candidate
                max_attempts = 10
                for _ in range(max_attempts):
                    candidate_rect = (tx, ty, tx + tw_full, ty + th_full)
                    has_overlap = any(_rects_overlap(candidate_rect, occ) for occ in occupied_rects)
                    if not has_overlap:
                        return ty
                    ty += th_full + 1
                    if ty + th_full > H:
                        break
                ty = ty_candidate
                for _ in range(max_attempts):
                    ty -= th_full + 1
                    if ty < 0:
                        break
                    candidate_rect = (tx, ty, tx + tw_full, ty + th_full)
                    has_overlap = any(_rects_overlap(candidate_rect, occ) for occ in occupied_rects)
                    if not has_overlap:
                        return ty
                return ty_candidate # 

            for idx, (box, label) in enumerate(
                zip(sam3_bboxes_norm, entity_labels)
            ):
                if not label:
                    continue
                x0n, y0n, x1n, y1n = box
                x0 = max(0, int(x0n * W))
                y0 = max(0, int(y0n * H))
                y1_px = min(H - 1, int(y1n * H))

                display_label = label[:28] + "..." if len(label) > 28 else label

                try:
                    tb = draw.textbbox((0, 0), display_label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = font_size * len(display_label) // 2, font_size

                text_block_h = th + pad * 2 # +
                tw_full = tw + pad * 2 # 

                tx = min(x0, W - tw_full)
                tx = max(0, tx)

                if y0 - text_block_h >= 0:
                    ty = y0 - text_block_h
                else:
                    ty = min(y1_px + pad, H - text_block_h)
                    ty = max(0, ty)

                ty = _find_non_overlapping_ty(tx, ty, tw_full, text_block_h)
                ty = max(0, min(ty, H - text_block_h))

                if ty < y0:
                    expanded_bboxes[idx][1] = min(
                        expanded_bboxes[idx][1],
                        max(0.0, ty / H)
                    )
                else:
                    expanded_bboxes[idx][3] = max(
                        expanded_bboxes[idx][3],
                        min(1.0, (ty + text_block_h) / H)
                    )

                tx_right = tx + tw_full
                expanded_bboxes[idx][0] = min(
                    expanded_bboxes[idx][0], max(0.0, tx / W)
                )
                expanded_bboxes[idx][2] = max(
                    expanded_bboxes[idx][2], min(1.0, tx_right / W)
                )

                draw.rectangle(
                    [tx, ty, tx + tw_full, ty + text_block_h],
                    fill=text_bg_color,
                )
                draw.text((tx + pad, ty + pad), display_label,
                          fill=text_color, font=font)

                occupied_rects.append((tx, ty, tx + tw_full, ty + text_block_h))

        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        b64_out = base64.b64encode(buf.getvalue()).decode()
        return f"data:image;base64,{b64_out}", expanded_bboxes

    except Exception as e:
        print(f"[sam3_highlight] : {e}")
        return image_path_or_b64, expanded_bboxes


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def crop_image_region_b64(image_path_or_b64: str, bbox_norm: list) -> str:
    """
     bbox_norm=[x0,y0,x1,y1] base64
     image_path_or_b64
    """
    try:
        if isinstance(image_path_or_b64, str) and image_path_or_b64.startswith("data:image"):
            b64 = image_path_or_b64.split(",")[1]
        elif isinstance(image_path_or_b64, str) and os.path.exists(image_path_or_b64):
            with open(image_path_or_b64, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        else:
            b64 = image_path_or_b64
        img_data = base64.b64decode(b64)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        W, H = img.size
        x0 = max(0, int(bbox_norm[0] * W))
        y0 = max(0, int(bbox_norm[1] * H))
        x1 = min(W, int(bbox_norm[2] * W))
        y1 = min(H, int(bbox_norm[3] * H))
        if x0 >= x1 or y0 >= y1:
            return image_path_or_b64
        cropped = img.crop((x0, y0, x1, y1))
        buf = BytesIO()
        cropped.save(buf, format="PNG")
        return "data:image;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[crop_region] : {e}")
        return image_path_or_b64


def _bbox_iou(b1, b2):
    """ bbox IoU"""
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def secondary_sam3_check_on_att_bboxes(
    img_url: list,
    bounding_boxes: dict,
    entity_list: list,
    sam3_url: str,
    max_per_entity: int = 2,
    max_att_bboxes: int = 4,
    existing_sam3_bboxes_per_img: dict = None,
    iou_dedup_threshold: float = 0.8,
    max_new_bboxes: int = 4,
    draw_on_image: bool = False,
) -> tuple:
    """
     bbox max_att_bboxes SAM3
     bbox/label LPD

    
      - draw_on_image=False
      - new_entries_per_img bbox bounding_boxes 
         LPD LPD legend 

    Returns:
        (img_url_modified, found_any, new_entries_per_img)
        - img_url_modified: listdraw_on_image=False 
        - found_any: bool
        - new_entries_per_img: dict[int, list[(box, label)]] bbox 
    """
    img_url_modified = list(img_url)
    found_any = False
    new_entries_per_img = {}

    if not entity_list or not sam3_url:
        return img_url_modified, found_any, new_entries_per_img

    for imgidx, att_bboxes in bounding_boxes.items():
        if imgidx >= len(img_url) or not att_bboxes:
            continue

        src_img = img_url[imgidx]

        existing_boxes = []
        if existing_sam3_bboxes_per_img and imgidx in existing_sam3_bboxes_per_img:
            existing_boxes = existing_sam3_bboxes_per_img[imgidx]

        sorted_att = sorted(
            att_bboxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True
        )[:max_att_bboxes]

        new_bboxes_orig = []
        new_labels_orig = []

        for att_bbox in sorted_att:
            x0_a, y0_a, x1_a, y1_a = att_bbox
            w_att = x1_a - x0_a
            h_att = y1_a - y0_a
            if w_att < 0.02 or h_att < 0.02:
                continue # bbox

            cropped_b64 = crop_image_region_b64(src_img, att_bbox)
            try:
                crop_results = call_grounding_expert(
                    cropped_b64, entity_list, expert_url=sam3_url
                )
            except Exception as e:
                print(f"[secondary_sam3] SAM3 : {e}")
                continue

            for entity, bboxes in crop_results.items():
                sorted_bboxes = sorted(
                    bboxes,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    reverse=True,
                )[:max_per_entity]
                for b in sorted_bboxes:
                    ox0 = x0_a + b[0] * w_att
                    oy0 = y0_a + b[1] * h_att
                    ox1 = x0_a + b[2] * w_att
                    oy1 = y0_a + b[3] * h_att
                    area = (ox1 - ox0) * (oy1 - oy0)
                    if area < 0.0005:
                        continue
                    new_box = [ox0, oy0, ox1, oy1]
                    is_dup = False
                    for eb in existing_boxes:
                        if _bbox_iou(new_box, eb) >= iou_dedup_threshold:
                            is_dup = True
                            break
                    if not is_dup:
                        for eb in new_bboxes_orig:
                            if _bbox_iou(new_box, eb) >= iou_dedup_threshold:
                                is_dup = True
                                break
                    if is_dup:
                        continue
                    new_bboxes_orig.append(new_box)
                    new_labels_orig.append(entity[:25])
                    found_any = True

        if new_bboxes_orig and len(new_bboxes_orig) > max_new_bboxes:
            pairs = sorted(
                zip(new_bboxes_orig, new_labels_orig),
                key=lambda bl: (bl[0][2] - bl[0][0]) * (bl[0][3] - bl[0][1]),
                reverse=True,
            )[:max_new_bboxes]
            new_bboxes_orig = [p[0] for p in pairs]
            new_labels_orig = [p[1] for p in pairs]

        if new_bboxes_orig:
            new_entries_per_img[imgidx] = list(zip(new_bboxes_orig, new_labels_orig))

        if draw_on_image and new_bboxes_orig:
            img_url_modified[imgidx], _ = apply_sam3_highlight_to_image(
                img_url_modified[imgidx],
                new_bboxes_orig,
                entity_labels=None,
                border_color=(0, 200, 255),
                text_bg_color=(0, 100, 180),
            )

    return img_url_modified, found_any, new_entries_per_img


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════

def save_attention_heatmap(
    att_map: np.ndarray,
    image_path_or_b64: str,
    save_path: str,
    alpha: float = 0.55,
    colormap: str = "jet",
    bboxes_norm=None,
    sam3_bboxes_norm=None,
    title: str = "",
):
    """
    

    :
        att_map: 2D shape (H_att, W_att) [0, 1]
        image_path_or_b64: base64 
        save_path: .png / .jpg
        alpha: 0=1= 0.55
        colormap: matplotlib colormap "jet"
        bboxes_norm: list of [x0,y0,x1,y1] bbox
        sam3_bboxes_norm: list of [x0,y0,x1,y1]SAM3 bbox
        title: 
    """
    import matplotlib
    matplotlib.use("Agg") # GUI 
    import matplotlib.patches as mpatches

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    try:
        if isinstance(image_path_or_b64, str) and image_path_or_b64.startswith("data:image"):
            b64 = image_path_or_b64.split(",")[1]
        elif isinstance(image_path_or_b64, str) and os.path.exists(image_path_or_b64):
            with open(image_path_or_b64, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        else:
            b64 = image_path_or_b64
        img_data = base64.b64decode(b64)
        orig_img = Image.open(BytesIO(img_data)).convert("RGB")
    except Exception as e:
        print(f"[heatmap] : {e}")
        return

    W, H = orig_img.size # 

    att_resized = np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(att_resized)              # (H, W, 4) float [0,1]
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil  = Image.fromarray(heatmap_rgb)

    orig_arr    = np.array(orig_img, dtype=np.float32)
    heatmap_arr = np.array(heatmap_pil, dtype=np.float32)
    blended_arr = np.clip((1 - alpha) * orig_arr + alpha * heatmap_arr, 0, 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(blended_arr)
    ax.axis("off")

    legend_patches = []
    if bboxes_norm:
        for box in bboxes_norm:
            x0n, y0n, x1n, y1n = box
            rect = mpatches.Rectangle(
                (x0n * W, y0n * H), (x1n - x0n) * W, (y1n - y0n) * H,
                linewidth=2, edgecolor="#00ff00", facecolor="none"
            )
            ax.add_patch(rect)
        legend_patches.append(mpatches.Patch(edgecolor="#00ff00", facecolor="none", label="Attention bbox"))

    if sam3_bboxes_norm:
        for box in sam3_bboxes_norm:
            x0n, y0n, x1n, y1n = box
            rect = mpatches.Rectangle(
                (x0n * W, y0n * H), (x1n - x0n) * W, (y1n - y0n) * H,
                linewidth=2, edgecolor="#ff8800", facecolor="none", linestyle="--"
            )
            ax.add_patch(rect)
        legend_patches.append(mpatches.Patch(edgecolor="#ff8800", facecolor="none",
                                              linestyle="--", label="SAM3 bbox"))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
                  framealpha=0.7, facecolor="white")
    if title:
        ax.set_title(title, fontsize=9, pad=3)

    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def build_aggregated_heatmap(accept_att: dict, img_idx: int) -> np.ndarray | None:
    """
     accept_att[img_idx] token 

    :
        accept_att: {img_idx: {token_k: att_map (H, W)}}
        img_idx: 

    :
         (H, W) [0, 1] None
    """
    if img_idx not in accept_att or not accept_att[img_idx]:
        return None

    maps = []
    for k, att in accept_att[img_idx].items():
        m = att[0] if att.ndim == 3 else att   # squeeze leading dim if present
        maps.append(m)

    if not maps:
        return None

    agg = np.mean(maps, axis=0)
    if agg.max() > agg.min():
        agg = (agg - agg.min()) / (agg.max() - agg.min())
    return agg


def get_inputs(messages,processor,model):
    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages,return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs
    )
    inputs = inputs.to(model.device)
    return text,image_inputs,video_inputs,inputs,video_kwargs

def batch_get_inputs(messages_list, processor, model):
    """
     messages batched inputs
     padding_side='left' 
    
    Args:
        messages_list: list of messages messages
        processor: AutoProcessor
        model: 
    
    Returns:
        batched_inputs: tensor
    """
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = 'left'
    
    texts = []
    all_image_inputs = []
    all_video_inputs = []
    all_video_kwargs_list = []
    
    for messages in messages_list:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        if image_inputs:
            all_image_inputs.extend(image_inputs)
        if video_inputs:
            all_video_inputs.extend(video_inputs)
        all_video_kwargs_list.append(video_kwargs)
    
    merged_video_kwargs = {}
    for vk in all_video_kwargs_list:
        for key, val in vk.items():
            if key not in merged_video_kwargs:
                merged_video_kwargs[key] = val
    
    batched_inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
        **merged_video_kwargs
    )
    batched_inputs = batched_inputs.to(model.device)
    
    processor.tokenizer.padding_side = original_padding_side
    
    return batched_inputs

def batch_messages2out(model, processor, batched_inputs, return_first_logits=False,
                         option_token_ids=None, return_full_first_logits=False):
    """
     left-padded 

    Args:
        model: 
        processor: AutoProcessor
        batched_inputs: batch_get_inputs 
        return_first_logits: bool, token 
             Difficulty-Aware Router v1 A/B/C/D 4 token
        option_token_ids: dict[str,int] {"A":32,"B":33,"C":34,"D":35}
             return_first_logits=True 
        return_full_first_logits: bool, True 
            first-token logitsnp.ndarray [vocab] RouterV2 
            option_masslogit_gap 

    Returns:
        
          return_first_logits=False, return_full_first_logits=False:
              output_texts: list[str]
          return_first_logits=True, return_full_first_logits=False:
              output_texts, option_probs_list, answer_entropies
          return_full_first_logits=True  (implies return_first_logits):
              output_texts, option_probs_list, answer_entropies,
              full_first_logits_list  (list[np.ndarray|None], each [vocab])
    """
    import numpy as np
    import torch.nn.functional as F

    batched_inputs = batched_inputs.to(model.device)
    need_scores = return_first_logits or return_full_first_logits

    with torch.no_grad():
        if need_scores:
            gen_out = model.generate(
                **batched_inputs, use_cache=True, max_new_tokens=4096, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
            generated_ids = gen_out.sequences
            scores = gen_out.scores # tuple, = new_tokens [B, vocab]
        else:
            generated_ids = model.generate(
                **batched_inputs, use_cache=True, max_new_tokens=4096, do_sample=False,
            )
            scores = None

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(batched_inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    option_probs_list = None
    answer_entropies = None
    full_first_logits_list = None
    if need_scores and scores is not None and len(scores) > 0:
        first_logits = scores[0].float().cpu().numpy()   # [B, vocab]
        B = first_logits.shape[0]

        if return_first_logits and option_token_ids is not None:
            letters = ["A", "B", "C", "D"]
            opt_idx = np.array([option_token_ids[l] for l in letters], dtype=np.int64)
            opt_log = first_logits[:, opt_idx]  # [B, 4]
            # softmax over 4 options
            opt_log = opt_log - opt_log.max(axis=1, keepdims=True)
            opt_probs = np.exp(opt_log)
            opt_probs = opt_probs / opt_probs.sum(axis=1, keepdims=True)   # [B, 4]
            entropies = -np.sum(opt_probs * np.log(opt_probs + 1e-12), axis=1)  # [B]
            option_probs_list = [opt_probs[i] for i in range(opt_probs.shape[0])]
            answer_entropies = [float(entropies[i]) for i in range(entropies.shape[0])]

        if return_full_first_logits:
            full_first_logits_list = [first_logits[i].copy() for i in range(B)]

    del batched_inputs, generated_ids
    torch.cuda.empty_cache()

    if return_full_first_logits:
        return output_texts, option_probs_list, answer_entropies, full_first_logits_list
    if return_first_logits:
        return output_texts, option_probs_list, answer_entropies
    return output_texts

def messages2out(model,processor,inputs):
    inputs = inputs.to(model.device)
    end_ques = len(inputs['input_ids'][0])

    with torch.no_grad():
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=4096, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    del inputs,generated_ids
    torch.cuda.empty_cache()
    return output_text,end_ques


def messages2out_with_logits(model, processor, inputs, option_token_ids=None):
    """
     messages2out first answer token logits / entropy
    threshold-free router

    Args:
        option_token_ids: dict, e.g. {"A": 32, "B": 33, "C": 34, "D": 35}
             4 softmax + 
    Returns:
        output_text: list[str]
        end_ques: int
        first_logits: np.ndarray [vocab_size] token logits
        option_probs: np.ndarray [4] 4 option_token_ids 
        answer_entropy: float 4 softmax option_token_ids 
    """
    import numpy as np
    import torch.nn.functional as F

    inputs = inputs.to(model.device)
    end_ques = len(inputs['input_ids'][0])
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, use_cache=True, max_new_tokens=4096, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    generated_ids = gen_out.sequences
    # scores: tuple of length = num_new_tokens, each [B, vocab]
    scores = gen_out.scores
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    if scores is not None and len(scores) > 0:
        first_logits = scores[0][0].float().cpu().numpy()   # [vocab]
    else:
        first_logits = None

    option_probs = None
    answer_entropy = None
    if option_token_ids is not None and first_logits is not None:
        letters = ["A", "B", "C", "D"]
        p_full = F.softmax(torch.from_numpy(first_logits), dim=-1).numpy()
        p_opt = np.array([p_full[option_token_ids[l]] for l in letters], dtype=np.float64)
        p_opt = p_opt / max(p_opt.sum(), 1e-12)
        option_probs = p_opt
        answer_entropy = float(-np.sum(p_opt * np.log(p_opt + 1e-12)))

    del inputs, generated_ids, scores
    torch.cuda.empty_cache()
    return output_text, end_ques, first_logits, option_probs, answer_entropy
    
def messages2att(model,processor,inputs):
    inputs = inputs.to(model.device)
    end_ques = len(inputs['input_ids'][0])
    img_start = []
    img_end = []
    idx2word_dicts = {}
    need_2_att_w = []
    for i in range(len(inputs['input_ids'][0])):
        words = processor.post_process_image_text_to_text(
        torch.tensor([inputs['input_ids'][0][i]]), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        idx2word_dicts[inputs['input_ids'][0][i].cpu().item()] = words
        if inputs['input_ids'][0][i].cpu().item() == 151652:
            img_start.append(i+1)
        if inputs['input_ids'][0][i].cpu().item() == 151653:
            img_end.append(i)
    for i in range(len(inputs['input_ids'][0])):
        if i>img_end[-1]:
            need_2_att_w.append(i)
    # print(len(need_2_att_w))
    with torch.no_grad():
        out = model(**inputs, output_attentions=True,target_indices=torch.tensor(need_2_att_w))  # logits,past_key_values,Attention
    # del inputs
    # torch.cuda.empty_cache()
    attention = []
    for i in range(len(out['attentions'])):
        if out['attentions'][i] is None:
            continue
        attention.append(out['attentions'][i])
    del inputs,out
    torch.cuda.empty_cache()
    return attention,idx2word_dicts,img_start,img_end


def place_on_center(canvas_bgra, content_bgra):
    """ content (BGRA) canvas (BGRA)"""
    canvas_h, canvas_w, _ = canvas_bgra.shape
    content_h, content_w, _ = content_bgra.shape

    if content_h > canvas_h or content_w > canvas_w:
        scale = min(canvas_h / content_h, canvas_w / content_w)
        new_h, new_w = int(content_h * scale), int(content_w * scale)
        content_bgra = cv2.resize(content_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        content_h, content_w = new_h, new_w

    paste_x = (canvas_w - content_w) // 2
    paste_y = (canvas_h - content_h) // 2
    
    alpha_mask = content_bgra[:, :, 3] / 255.0
    
    for c in range(0, 3):
        canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c] = \
            alpha_mask * content_bgra[:, :, c] + \
            (1 - alpha_mask) * canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c]
            
    canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3] = \
        np.maximum(canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3], content_bgra[:, :, 3])
        
    return canvas_bgra

def decompose_bbox_by_alpha(image_bgra, bbox, alpha_threshold=10):
    """
    BBoxAlphaBBox

    Args:
        image_bgra (np.array): 4BGRA
        bbox (list or tuple): [x0, y0, x1, y1]
        alpha_threshold (int): 
                               Alpha

    Returns:
        list: BBox [x, y, w, h] 
              BBox
    """
    x0, y0, x1, y1 = bbox
    img_h, img_w, _ = image_bgra.shape

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)

    if x0 >= x1 or y0 >= y1:
        return []

    roi = image_bgra[y0:y1, x0:x1]
    alpha_channel = roi[:, :, 3] # BGRAAlpha3

    _, mask = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sub_bboxes = []
    for contour in contours:
        sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(contour)
        
        abs_x0 = x0 + sub_x
        abs_y0 = y0 + sub_y
        abs_x1 = abs_x0 + sub_w
        abs_y1 = abs_y0 + sub_h
        
        sub_bboxes.append([abs_x0, abs_y0, abs_x1, abs_y1])
        
    return sub_bboxes

def merge_overlapping_bboxes(bboxes):
    """
    BBox

    Args:
        bboxes (list): BBox [x0, y0, x1, y1] 

    Returns:
        list: BBoxBBox
    """
    if not bboxes:
        return []

    bboxes = [list(b) for b in bboxes] # 

    while True:
        merged_one = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                box1 = bboxes[i]
                box2 = bboxes[j]

                is_overlapping = not (box1[2] < box2[0] or # box1box2
                                      box1[0] > box2[2] or # box1box2
                                      box1[3] < box2[1] or # box1box2
                                      box1[1] > box2[3]) # box1box2

                if is_overlapping:
                    new_x0 = min(box1[0], box2[0])
                    new_y0 = min(box1[1], box2[1])
                    new_x1 = max(box1[2], box2[2])
                    new_y1 = max(box1[3], box2[3])
                    
                    bboxes[i] = [new_x0, new_y0, new_x1, new_y1]
                    bboxes.pop(j)
                    
                    merged_one = True
                    break # j
                else:
                    j += 1
            
            if merged_one:
                break # iwhile True
            else:
                i += 1
        
        if not merged_one:
            break
            
    return bboxes

def compact_and_center_with_relative_pos(imgidx, img_nums, image, normalized_bboxes, n=1,
                                          bbox_colors=None, bbox_labels=None,
                                          draw_color_border=True,
                                          border_thickness=2,
                                          overlay_bboxes=None):
    """
     BBox 

    :
    - n (int): n BBox 
               BBox 
    - bbox_colors / bbox_labels: [] normalized_bboxes 
        / LPD + alpha bbox 
         overlay_bboxes 
         legend bbox_colors 
    - draw_color_border (bool): overlay_bboxes
    - border_thickness (int): 
    - overlay_bboxes (list[dict] | None): bbox :
        {"bbox_norm": [x0,y0,x1,y1], "color": (R,G,B), "label": "entity name"}
        - bbox_norm: 
        - color: RGB 
        - label: 
        overlay_bboxes LPD / 
        "" x_map / y_map 
        
         SAM3 bbox bbox 
         SAM3 bbox 

    Returns:
        (imgs_list, return_bboxes, used_colors_labels)
        - imgs_list: list[str] base64 None
        - return_bboxes: list[list[float]]bbox
        - used_colors_labels: list[(rgb_tuple, label)]
           overlay_bboxes 
    """
    if not image.startswith('data:image;base64,'):
        image64 = image_to_base64(image).split(',')[1]
    elif ',' in image:
        image64 = image.split(',')[1]
    image_data = base64.b64decode(image64)
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    img_cv_bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    img_h, img_w, _ = img_cv_bgra.shape

    if not normalized_bboxes:
        return None, [], []

    initial_pixel_bboxes = []
    for n_box in normalized_bboxes:
        nx0, ny0, nx1, ny1 = n_box
        x0, y0 = int(nx0 * img_w), int(ny0 * img_h)
        x1, y1 = int(nx1 * img_w), int(ny1 * img_h)
        initial_pixel_bboxes.append([x0, y0, x1, y1])

    overlap_map = np.zeros((img_h, img_w), dtype=np.uint16)
    for bbox in initial_pixel_bboxes:
        x0, y0, x1, y1 = bbox
        if x0 < x1 and y0 < y1:
            overlap_map[y0:y1, x0:x1] += 1

    threshold_mask = (overlap_map >= n)
    if not np.any(threshold_mask):
        return None, [], []

    contributing_bboxes = []
    for bbox in initial_pixel_bboxes:
        x0, y0, x1, y1 = bbox
        if x0 < x1 and y0 < y1 and np.any(threshold_mask[y0:y1, x0:x1]):
            contributing_bboxes.append(bbox)
    if not contributing_bboxes:
        return None, [], []

    final_merged_bboxes = merge_overlapping_bboxes(contributing_bboxes)
    bboxes = np.array(final_merged_bboxes, dtype=int)

    decomposed_bboxes = []
    for bbox in bboxes:
        sub_bboxes = decompose_bbox_by_alpha(img_cv_bgra, bbox)
        decomposed_bboxes.extend(sub_bboxes)
    if not decomposed_bboxes:
        return None, [], []
    bboxes = np.array(decomposed_bboxes, dtype=int)

    # ✅ 3. "bbox_only_region.png"
    masked_img_bgra = np.zeros_like(img_cv_bgra)
    for x0, y0, x1, y1 in bboxes:
        x0_c, y0_c = max(0, x0), max(0, y0)
        x1_c, y1_c = min(img_w, x1), min(img_h, y1)
        if x0_c < x1_c and y0_c < y1_c:
            masked_img_bgra[y0_c:y1_c, x0_c:x1_c] = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
    masked_img_rgba = cv2.cvtColor(masked_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_masked = Image.fromarray(masked_img_rgba)

    x_coords = sorted(list(set(bboxes[:, [0, 2]].flatten())))
    y_coords = sorted(list(set(bboxes[:, [1, 3]].flatten())))

    x_map, new_x = {}, 0
    for i in range(len(x_coords) - 1):
        x_map[x_coords[i]] = new_x
        start_x, end_x = x_coords[i], x_coords[i+1]
        if any(b[0] < end_x and b[2] > start_x for b in bboxes):
            new_x += (end_x - start_x)
    x_map[x_coords[-1]] = new_x
    new_total_width = new_x

    y_map, new_y = {}, 0
    for i in range(len(y_coords) - 1):
        y_map[y_coords[i]] = new_y
        start_y, end_y = y_coords[i], y_coords[i+1]
        if any(b[1] < end_y and b[3] > start_y for b in bboxes):
            new_y += (end_y - start_y)
    y_map[y_coords[-1]] = new_y
    new_total_height = new_y

    x_pix_map = np.full(img_w + 1, -1, dtype=np.int32)
    y_pix_map = np.full(img_h + 1, -1, dtype=np.int32)
    for i in range(len(x_coords) - 1):
        sx, ex = x_coords[i], x_coords[i+1]
        if any(b[0] < ex and b[2] > sx for b in bboxes):
            for xx in range(max(0, sx), min(img_w, ex) + 1):
                x_pix_map[xx] = x_map[sx] + (xx - sx)
    x_pix_map[x_coords[-1]] = x_map[x_coords[-1]] if x_coords else 0
    for i in range(len(y_coords) - 1):
        sy, ey = y_coords[i], y_coords[i+1]
        if any(b[1] < ey and b[3] > sy for b in bboxes):
            for yy in range(max(0, sy), min(img_h, ey) + 1):
                y_pix_map[yy] = y_map[sy] + (yy - sy)
    y_pix_map[y_coords[-1]] = y_map[y_coords[-1]] if y_coords else 0

    composite_image_bgra = np.zeros((new_total_height, new_total_width, 4), dtype=np.uint8)
    for (x0, y0, x1, y1) in bboxes:
        y0_c, y1_c = max(0, y0), min(img_h, y1)
        x0_c, x1_c = max(0, x0), min(img_w, x1)
        if y0_c >= y1_c or x0_c >= x1_c:
            continue
        roi = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
        paste_x, paste_y = x_map[x0], y_map[y0]
        h, w, _ = roi.shape
        composite_image_bgra[paste_y : paste_y + h, paste_x : paste_x + w] = roi

    used_colors_labels = []
    _seen_color_label = set()
    if draw_color_border and overlay_bboxes:
        _rgba = cv2.cvtColor(composite_image_bgra, cv2.COLOR_BGRA2RGBA)
        _pil_comp = Image.fromarray(_rgba)
        _draw = ImageDraw.Draw(_pil_comp)

        def _map_x(px):
            px = int(max(0, min(img_w, px)))
            if x_pix_map[px] >= 0:
                return int(x_pix_map[px])
            for d in range(1, 20):
                if px - d >= 0 and x_pix_map[px - d] >= 0:
                    return int(x_pix_map[px - d])
                if px + d <= img_w and x_pix_map[px + d] >= 0:
                    return int(x_pix_map[px + d])
            return -1

        def _map_y(py):
            py = int(max(0, min(img_h, py)))
            if y_pix_map[py] >= 0:
                return int(y_pix_map[py])
            for d in range(1, 20):
                if py - d >= 0 and y_pix_map[py - d] >= 0:
                    return int(y_pix_map[py - d])
                if py + d <= img_h and y_pix_map[py + d] >= 0:
                    return int(y_pix_map[py + d])
            return -1

        for ov in overlay_bboxes:
            try:
                bx = ov.get("bbox_norm") if isinstance(ov, dict) else ov[0]
                color = ov.get("color") if isinstance(ov, dict) else ov[1]
                label = (ov.get("label") if isinstance(ov, dict) else ov[2]) or ""
            except Exception:
                continue
            if not bx or color is None:
                continue
            px0 = int(bx[0] * img_w)
            py0 = int(bx[1] * img_h)
            px1 = int(bx[2] * img_w)
            py1 = int(bx[3] * img_h)
            cx0 = _map_x(px0); cx1 = _map_x(px1)
            cy0 = _map_y(py0); cy1 = _map_y(py1)
            if cx0 < 0 or cy0 < 0 or cx1 < 0 or cy1 < 0:
                continue
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            cx0 = max(0, min(new_total_width - 1, cx0))
            cx1 = max(0, min(new_total_width - 1, cx1))
            cy0 = max(0, min(new_total_height - 1, cy0))
            cy1 = max(0, min(new_total_height - 1, cy1))
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            try:
                region_alpha = composite_image_bgra[cy0:cy1, cx0:cx1, 3]
                if region_alpha.size == 0 or not np.any(region_alpha > 0):
                    continue
            except Exception:
                pass
            c = tuple(int(v) for v in color)
            for tk in range(border_thickness):
                _draw.rectangle(
                    [cx0 + tk, cy0 + tk, cx1 - tk, cy1 - tk],
                    outline=c,
                )
            key = (c, label)
            if key not in _seen_color_label:
                _seen_color_label.add(key)
                used_colors_labels.append((c, label))
        composite_image_bgra = cv2.cvtColor(np.array(_pil_comp), cv2.COLOR_RGBA2BGRA)

    final_canvas_bgra = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    final_img_bgra = place_on_center(final_canvas_bgra, composite_image_bgra)

    final_img_rgba = cv2.cvtColor(final_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_centered = Image.fromarray(final_img_rgba)

    composite_image_rgba = cv2.cvtColor(composite_image_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result = Image.fromarray(composite_image_rgba)

    up_sclae = 1
    new_size = (round(pil_result.width * up_sclae), round(pil_result.height * up_sclae))
    pil_result = pil_result.resize(new_size, Image.Resampling.BILINEAR)

    return_bboxes = []
    for x0, y0, x1, y1 in bboxes:
        return_bboxes.append([x0/img_w, y0/img_h, x1/img_w, y1/img_h])
    return [pil_to_base64(pil_result)], return_bboxes, used_colors_labels


def _merge_overlap_with_source(bboxes, source_indices):
    """
    merge_overlapping_bboxes bbox 

    Args:
        bboxes: list[[x0,y0,x1,y1]]
        source_indices: list[int] bboxes bbox 
    Returns:
        merged_bboxes, merged_source_sets
          - merged_bboxes: list[list[int]]
          - merged_source_sets: list[set[int]] merged_bboxes 
    """
    if not bboxes:
        return [], []
    bxs = [list(b) for b in bboxes]
    srcs = [set([si]) for si in source_indices]
    while True:
        merged_one = False
        i = 0
        while i < len(bxs):
            j = i + 1
            while j < len(bxs):
                b1, b2 = bxs[i], bxs[j]
                overlapping = not (b1[2] < b2[0] or b1[0] > b2[2] or
                                   b1[3] < b2[1] or b1[1] > b2[3])
                if overlapping:
                    bxs[i] = [
                        min(b1[0], b2[0]), min(b1[1], b2[1]),
                        max(b1[2], b2[2]), max(b1[3], b2[3]),
                    ]
                    srcs[i] = srcs[i] | srcs[j]
                    bxs.pop(j); srcs.pop(j)
                    merged_one = True
                    break
                else:
                    j += 1
            if merged_one:
                break
            else:
                i += 1
        if not merged_one:
            break
    return bxs, srcs

def find_top_n_attended_regions(norm_att, n, threshold=0.5, use_otsu=False):
    """
    n

    
    1. 
    2. 
    3. ""
    4. 
    5. nn

    :
    att_map (np.ndarray): 01
    n (int): 
    threshold (float, optional): 0.5

    :
    list: [x_min, y_min, x_max, y_max]
          
    """
    att_map = np.array(norm_att)

    if use_otsu and att_map.max() > att_map.min():
        try:
            threshold = threshold_otsu(att_map)
        except Exception:
            pass # 

    binarized_map = (att_map >= threshold)
    if not np.any(binarized_map): # 
        return [], 0
        
    labeled_map = label(binarized_map, connectivity=2)
    regions = regionprops(labeled_map)

    scored_regions = []
    for region in regions:
        mask = (labeled_map == region.label)
        score = np.sum(att_map[mask])
        scored_regions.append({
            'score': score,
            'bbox': region.bbox # bbox (y0, x0, y1, x1)
        })
        # if 0 == region.bbox[0] and 0 == region.bbox[1]: return [],0

    sorted_regions = sorted(scored_regions, key=lambda r: r['score'], reverse=True)

    # if n > len(sorted_regions):
    #     n = len(sorted_regions)
    # top_n_regions = sorted_regions[:n]

    final_boxes = []
    get_num = 0
    for region in sorted_regions:
        y0, x0, y1, x1 = region['bbox']
        box_area = (y1-y0) * (x1-x0)
        # if 0 == x0 and 0 == y0 and box_area/map_area < 0.1:
        #     continue
        get_num += 1
        final_boxes.append([x0, y0, x1, y1])
        # if 0 == x0 and 0 == y0: return [],0

    # final_boxes = []
    # for region in sorted_regions:
    #     y0, x0, y1, x1 = region['bbox']
    #     final_boxes.append([x0, y0, x1, y1])

    return final_boxes, len(final_boxes)

def from_img_and_att_get_cropbox(inputs, attention, dicts, img_url, img_start, img_end, sig, thre,
                                  enable_saaa=False, entity_text="",
                                  expert_bboxes_per_img=None,
                                  egaf_fusion_mode="adaptive", egaf_expert_weight=0.5,
                                  entity_token_indices=None,
                                  entity_token_map=None,
                                  expert_reliability=1.0,
                                  enable_grace=False,
                                  use_otsu=False,
                                  sam3_supplement_bboxes_per_img=None,
                                  sam3_entity_labels_per_img=None,
                                  sam3_draw_on_image=False,
                                  sam3_draw_labels=False,
                                  heatmap_save_dir=None,
                                  sample_id="sample"):
    """
     EGAF GRACE 

    
      SAM3 bbox 
        1. bbox LPD
        2. LPD legend
        3. SAM3 

    :
        enable_saaa: bool, EGAF Rank-Based Fusion
        entity_text: str, 
        expert_bboxes_per_img: dict, {img_idx: bboxes}EGAF 
        egaf_fusion_mode / egaf_expert_weight / expert_reliability: EGAF 
        entity_token_indices: list of int, token 
        entity_token_map: dict, token 

        enable_grace: bool, GRACE + Otsu + SAM3 
        use_otsu: bool, Otsu GRACE True
        sam3_supplement_bboxes_per_img: dict, {img_idx: [[x0,y0,x1,y1],...]}，
            SAM3 bbox
        sam3_entity_labels_per_img: dict, {img_idx: [label,...]}，
             sam3_supplement_bboxes_per_img bbox 
            None 

        heatmap_save_dir: str | None, 
            None None :
              {sample_id}_img{img_idx}_s{sigma}_t{thresh}_agg_heatmap.png ()
        sample_id: str, "sample"
    """
    tmp_att = []
    for i in range(len(attention)):
        if attention[i] is None:
            continue
        tmp_att.append(attention[i])
    attention = tmp_att
    start_k = img_end[-1]+1
    end_k = len(inputs['input_ids'][0])
    results = {}
    for s in sig:
        for t in thre:
            if enable_grace:
                accept_att = process_grace(
                    dicts, start_k, end_k, attention, inputs, img_start, img_end, s)
            elif enable_saaa:
                accept_att = process_egaf(
                    dicts, start_k, end_k, attention, inputs, img_start, img_end, s,
                    expert_bboxes_per_img=expert_bboxes_per_img,
                    egaf_fusion_mode=egaf_fusion_mode,
                    egaf_expert_weight=egaf_expert_weight,
                    entity_token_indices=entity_token_indices,
                    entity_token_map=entity_token_map,
                    expert_reliability=expert_reliability)
            else:
                accept_att = process(dicts, start_k, end_k, attention, inputs, img_start, img_end, s)

            if heatmap_save_dir is not None:
                for _img_idx in accept_att:
                    agg_map = build_aggregated_heatmap(accept_att, _img_idx)
                    if agg_map is None:
                        continue
                    _sam3_bboxes = (
                        sam3_supplement_bboxes_per_img.get(_img_idx, [])
                        if sam3_supplement_bboxes_per_img else []
                    )
                    _img_src = img_url[_img_idx] if _img_idx < len(img_url) else img_url[0]
                    _fname = f"{sample_id}_img{_img_idx}_s{s}_t{t}_agg_heatmap.png"
                    _fpath = os.path.join(heatmap_save_dir, _fname)
                    save_attention_heatmap(
                        att_map=agg_map,
                        image_path_or_b64=_img_src,
                        save_path=_fpath,
                        alpha=0.55,
                        colormap="jet",
                        bboxes_norm=None, # bbox 
                        sam3_bboxes_norm=_sam3_bboxes if _sam3_bboxes else None,
                        title=f"[{sample_id}] Aggregated Attn | s={s} t={t:.2f} | {entity_text[:60]}",
                    )

            imgs_words_att_box = {}
            for img_idx in accept_att:
                accept_word_att = accept_att[img_idx]
                words_att_box = {}
                for word in accept_word_att:
                    att_map = accept_word_att[word][0]
                    boxs, rigion_nums = find_top_n_attended_regions(att_map, 100, t, use_otsu=False)
                    total_attention = np.sum(att_map)
                    img_height, img_width = att_map.shape
                    total_area = img_width * img_height
                    save_boxs = []
                    if boxs:
                        H, W = att_map.shape
                        words_att_box[word] = []
                        for box in boxs:
                            x0,y0,x1,y1 = box
                            bbox_norm = (x0 / W, y0 / H, (x1) / W, (y1) / H)
                            x0,y0,x1,y1 = bbox_norm
                            region = att_map[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]
                            region_sum = np.sum(region)
                            words_att_box[word].append(bbox_norm)
                            save_boxs.append(box)
                imgs_words_att_box[img_idx] = words_att_box

            for img_idx in imgs_words_att_box:
                max_word_idx = 0
                for words_idx in imgs_words_att_box[img_idx]:
                    max_word_idx = max(max_word_idx,words_idx)
            img_merged_boxes = swap_and_rebuild_dict(imgs_words_att_box)

            words_lines = {}
            get_words = ""
            # print(start_k,end_k)
            for i in range(start_k,end_k):
                token_idx = inputs['input_ids'][0][i].cpu().item()
                # print(i,dicts[token_idx],end="||")
                if token_idx < 151643:
                    get_words+=dicts[token_idx]
                for word in img_merged_boxes:
                    if i == word+1:
                        words_lines[word] = get_words
                        get_words = ''
            for word in img_merged_boxes:
                if i == word:
                    words_lines[word] = get_words
                    get_words = ''
            words_lines[-1] = get_words
            get_words = ''
            crop_list = {}
            bounding_boxes = {}
            highlight_imgs = []
            hide_highlight_imgs = []
            for word in img_merged_boxes:
                if not word in crop_list:
                    crop_list[word] = {}
                for imgidx in img_merged_boxes[word]:
                    if not imgidx in bounding_boxes: bounding_boxes[imgidx] = []
                    for boxid in range(len(img_merged_boxes[word][imgidx])):
                        bounding_boxes[imgidx].append(img_merged_boxes[word][imgidx][boxid])
            original_att_bboxes = {}
            for imgidx in bounding_boxes:
                original_att_bboxes[imgidx] = merge_overlapping_bboxes(
                    [list(b) for b in bounding_boxes[imgidx]]
                )

            if enable_grace:
                hide_bounding_boxes = {
                    imgidx: [list(b) for b in boxes]
                    for imgidx, boxes in bounding_boxes.items()
                }
                for imgidx, hide_boxes in hide_bounding_boxes.items():
                    src_img_hide = img_url[imgidx] if imgidx < len(img_url) else img_url[0]
                    hide_imgs, _, _ = compact_and_center_with_relative_pos(
                        imgidx, len(img_url), src_img_hide, hide_boxes,
                        bbox_colors=None, bbox_labels=None, draw_color_border=False,
                    )
                    if hide_imgs:
                        hide_highlight_imgs.extend(hide_imgs)

            #
            #
            #
            img_url_for_crop = list(img_url) # 

            # item: {"bbox_norm": [x0,y0,x1,y1], "color": (R,G,B), "label": str}
            overlay_bboxes_per_img = {}
            extra_legend_entries = [] # legend 

            if sam3_supplement_bboxes_per_img:
                _all_sam3_labels = []
                for _labels in (sam3_entity_labels_per_img or {}).values():
                    _all_sam3_labels.extend(list(_labels) if _labels else [])
                label2color_global, _ = assign_entity_colors(_all_sam3_labels)

                for sup_imgidx, sup_bboxes in sam3_supplement_bboxes_per_img.items():
                    if not sup_bboxes:
                        continue
                    sup_labels = (
                        list(sam3_entity_labels_per_img.get(sup_imgidx, []))
                        if sam3_entity_labels_per_img else []
                    )
                    while len(sup_labels) < len(sup_bboxes):
                        sup_labels.append("")

                    att_bboxes = bounding_boxes.get(sup_imgidx, [])

                    for box_idx, orig_box in enumerate(sup_bboxes):
                        label = sup_labels[box_idx] if box_idx < len(sup_labels) else ""
                        color = label2color_global.get(
                            (label or "").strip(), _LEGEND_PALETTE[0]
                        )
                        sam3_area = (orig_box[2] - orig_box[0]) * (orig_box[3] - orig_box[1])
                        SAM3_MAX_AREA_RATIO = 0.80
                        if sam3_area > SAM3_MAX_AREA_RATIO:
                            if label:
                                extra_legend_entries.append(
                                    (color, f"{label} (skipped large)")
                                )
                            continue

                        is_inside = att_bboxes and any(
                            bbox_is_contained(orig_box, att_box) for att_box in att_bboxes
                        )
                        overlay_bboxes_per_img.setdefault(sup_imgidx, []).append({
                            "bbox_norm": list(orig_box),
                            "color": color,
                            "label": label or "",
                        })
                        if not is_inside:
                            if sup_imgidx not in bounding_boxes:
                                bounding_boxes[sup_imgidx] = []
                            bounding_boxes[sup_imgidx].append(list(orig_box))

            for imgidx in bounding_boxes:
                src_img = (img_url_for_crop[imgidx]
                           if imgidx < len(img_url_for_crop) else img_url[imgidx])
                _overlays = overlay_bboxes_per_img.get(imgidx) or None
                img, bboxs, used_colors_labels = compact_and_center_with_relative_pos(
                    imgidx, len(img_url), src_img, bounding_boxes[imgidx],
                    overlay_bboxes=_overlays,
                    draw_color_border=True,
                )
                if img:
                    bounding_boxes[imgidx] = bboxs
                    _seen = set()
                    merged_legend = []
                    for cl in list(used_colors_labels) + extra_legend_entries:
                        key = (tuple(cl[0]), cl[1])
                        if key not in _seen and cl[1]:
                            _seen.add(key)
                            merged_legend.append(cl)
                    for im in img:
                        if merged_legend:
                            im = add_legend_to_lpd_image(
                                im, merged_legend, position="bottom"
                            )
                        highlight_imgs.append(im)

            if heatmap_save_dir is not None:
                for _img_idx in accept_att:
                    agg_map = build_aggregated_heatmap(accept_att, _img_idx)
                    if agg_map is None:
                        continue
                    _att_bboxes   = original_att_bboxes.get(_img_idx, [])
                    _sam3_bboxes  = (
                        sam3_supplement_bboxes_per_img.get(_img_idx, [])
                        if sam3_supplement_bboxes_per_img else []
                    )
                    _img_src  = img_url[_img_idx] if _img_idx < len(img_url) else img_url[0]
                    _fname    = f"{sample_id}_img{_img_idx}_s{s}_t{t}_agg_heatmap.png"
                    _fpath    = os.path.join(heatmap_save_dir, _fname)
                    save_attention_heatmap(
                        att_map=agg_map,
                        image_path_or_b64=_img_src,
                        save_path=_fpath,
                        alpha=0.55,
                        colormap="jet",
                        bboxes_norm=_att_bboxes if _att_bboxes else None,
                        sam3_bboxes_norm=_sam3_bboxes if _sam3_bboxes else None,
                        title=f"[{sample_id}] Attn Heatmap | s={s} t={t:.2f} | {entity_text[:60]}",
                    )

            if not str(s) in results:results[str(s)] = {}
            results[str(s)][str(t)] = [
                img_merged_boxes,
                crop_list,
                words_lines,
                highlight_imgs,
                bounding_boxes,
                hide_highlight_imgs,
            ]
    return results
