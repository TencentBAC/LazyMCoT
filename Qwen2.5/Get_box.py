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
# 颜色调色盘 + LPD 外侧图注工具
# ═══════════════════════════════════════════════════════════════════════════

# 高区分度调色盘（避开纯红/纯绿/纯蓝以免与注意力热力图色系混淆；覆盖色相尽量分散）
_LEGEND_PALETTE = [
    (255, 140, 0),     # 橙
    (0, 191, 255),     # 天蓝青
    (255, 105, 180),   # 亮粉
    (138, 43, 226),    # 紫
    (255, 215, 0),     # 金黄
    (60, 179, 113),    # 中海绿
    (220, 20, 60),     # 深红
    (100, 149, 237),   # 矢车菊蓝
    (255, 69, 0),      # 红橙
    (0, 206, 209),     # 深青
    (186, 85, 211),    # 兰花紫
    (154, 205, 50),    # 黄绿
]


def assign_entity_colors(entity_labels):
    """
    为每个实体标签分配一个固定调色盘颜色（同名实体共用同色）。

    Args:
        entity_labels: list[str]，与 bbox 一一对应的实体名
    Returns:
        label2color: dict[str, tuple(R,G,B)]
        bbox_colors: list[tuple]，与 entity_labels 等长，每个 bbox 的颜色
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
    在 LPD 输出图像的外侧追加一条图注区域，以 "■ label" 形式列出颜色与实体名。

    - 不修改原 LPD 图像像素，而是在外侧追加一条带状区域。
    - 图注条带的底色为浅灰色，避免遮盖 LPD 主图内容。
    - 若 color_label_pairs 为空，则直接返回原图（不做任何修改）。

    Args:
        lpd_image_b64: "data:image;base64,..." 或裸 base64 字符串
        color_label_pairs: list[(rgb_tuple, label_str)]，同色同实体合并后的图注项
        position: "bottom" | "right"，图注条带位置
        margin: 条带与主图/边缘的内边距
        swatch_size: 颜色方块像素大小
        font_size: 字体大小；None 时按主图尺寸自适应
        background / text_color / border_color: 图注区域的颜色

    Returns:
        new_image_b64: "data:image;base64,..." 追加图注后的图像
    """
    if not color_label_pairs:
        return lpd_image_b64

    try:
        # ── 解码主图 ────────────────────────────────────────────────────────
        if isinstance(lpd_image_b64, str) and lpd_image_b64.startswith("data:image"):
            _b64 = lpd_image_b64.split(",", 1)[1]
        else:
            _b64 = lpd_image_b64
        main_img = Image.open(BytesIO(base64.b64decode(_b64))).convert("RGBA")
        mw, mh = main_img.size

        # ── 自适应字体大小 ──────────────────────────────────────────────────
        if font_size is None:
            font_size = max(12, int(min(mw, mh) * 0.022))
        font = _get_pil_font(font_size)

        # ── 用 dummy 画布测量文本宽高 ──────────────────────────────────────
        _dummy = Image.new("RGBA", (10, 10))
        _ddraw = ImageDraw.Draw(_dummy)

        def _text_size(text):
            try:
                tb = _ddraw.textbbox((0, 0), text, font=font)
                return tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                return font_size * len(text) // 2, font_size

        # ── 计算每项宽度 ────────────────────────────────────────────────────
        item_gap = max(10, font_size // 2)
        items = []   # [(color, label, text_w, text_h)]
        row_h = max(swatch_size, font_size) + 2
        for color, label in color_label_pairs:
            label_disp = str(label) if label is not None else ""
            tw, th = _text_size(label_disp)
            items.append((color, label_disp, tw, th))

        # ── 布局：bottom 模式按行排列，宽度=主图宽度，超出换行 ──────────
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
            # RGBA 画布：主图区保持原透明度，图注区用不透明背景色
            bg_rgba = background + (255,) if len(background) == 3 else background
            new_img = Image.new("RGBA", (new_w, new_h), color=(0, 0, 0, 0))
            # 先把图注背景条带画上（不透明）
            legend_strip = Image.new("RGBA", (new_w, legend_h), color=bg_rgba)
            new_img.paste(legend_strip, (0, mh))
            # 再把主图 alpha-composite 上去（保留透明区域）
            new_img.paste(main_img, (0, 0))
            draw = ImageDraw.Draw(new_img)
            # 分隔线
            draw.line([(0, mh), (new_w, mh)], fill=border_color, width=1)

            # 绘制每一行
            for li, line in enumerate(lines):
                y_top = mh + margin + li * row_h
                x_cur = margin
                for color, label_disp, tw, th in line:
                    # 色块
                    sw = swatch_size
                    sy = y_top + (row_h - sw) // 2
                    draw.rectangle(
                        [x_cur, sy, x_cur + sw, sy + sw],
                        fill=tuple(int(c) for c in color),
                        outline=border_color,
                    )
                    # 文本
                    tx = x_cur + sw + 4
                    ty = y_top + (row_h - th) // 2
                    draw.text((tx, ty), label_disp, fill=text_color, font=font)
                    x_cur += sw + 6 + tw + item_gap

        else:
            # "right" 模式：单列排列在右侧
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
        print(f"[add_legend] 追加图注失败: {e}")
        return lpd_image_b64


# ═══════════════════════════════════════════════════════════════════════════
# SAM3 bbox 区分高亮工具
# ═══════════════════════════════════════════════════════════════════════════

def bbox_is_contained(inner_box, outer_box, tolerance=0.03):
    """
    判断 inner_box 是否（在一定容差内）被 outer_box 包含。

    参数:
        inner_box / outer_box: [x0, y0, x1, y1]，归一化坐标
        tolerance: 容差比例（相对于整图），默认 0.03（3%）

    返回:
        bool: inner_box 的四边均在 outer_box 的 tolerance 范围内
    """
    x0_i, y0_i, x1_i, y1_i = inner_box
    x0_o, y0_o, x1_o, y1_o = outer_box
    return (x0_i >= x0_o - tolerance and
            y0_i >= y0_o - tolerance and
            x1_i <= x1_o + tolerance and
            y1_i <= y1_o + tolerance)


def _get_pil_font(size: int):
    """尝试加载系统字体，失败时返回 PIL 默认字体。"""
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
    对图像中 SAM3 检测区域施加「细边框 + 实体名文字标注」。

    文字标注放在 bbox 外侧（优先上方，不足则下方），同时返回每个 bbox
    扩展后的区域（包含文字绘制区域），供 LPD 裁剪时使用，避免文字被截断。

    返回:
        (modified_image_b64, expanded_bboxes_norm)
        - modified_image_b64: 标注后图像的 base64 字符串
        - expanded_bboxes_norm: list of [x0,y0,x1,y1]，与 sam3_bboxes_norm 等长，
          每个 bbox 扩展到包含其文字标注区域。无标签的 bbox 保持原大小。
    """
    from PIL import ImageFont
    expanded_bboxes = [list(b) for b in sam3_bboxes_norm]
    try:
        # ── 解码原图 ─────────────────────────────────────────────────────
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

        # ── 1. 橙色细边框 ────────────────────────────────────────────────
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

        # ── 2. 文字标注 + 计算扩展 bbox（带防碰撞）────────────────────────
        if entity_labels:
            font_size = max(12, int(H * 0.015))
            font = _get_pil_font(font_size)
            pad = 3
            # 已占用的文字区域列表 [(tx, ty, tx_right, ty_bottom), ...]
            occupied_rects = []

            def _rects_overlap(r1, r2):
                """判断两个矩形 (x0,y0,x1,y1) 是否重叠。"""
                return r1[0] < r2[2] and r1[2] > r2[0] and r1[1] < r2[3] and r1[3] > r2[1]

            def _find_non_overlapping_ty(tx, ty_candidate, tw_full, th_full, direction="up"):
                """在 direction 方向上寻找不与已有标签重叠的 ty 位置。"""
                ty = ty_candidate
                max_attempts = 10
                for _ in range(max_attempts):
                    candidate_rect = (tx, ty, tx + tw_full, ty + th_full)
                    has_overlap = any(_rects_overlap(candidate_rect, occ) for occ in occupied_rects)
                    if not has_overlap:
                        return ty
                    # 向下偏移一个文字块高度
                    ty += th_full + 1
                    if ty + th_full > H:
                        break
                # 如果向下找不到，尝试从候选位置向上找
                ty = ty_candidate
                for _ in range(max_attempts):
                    ty -= th_full + 1
                    if ty < 0:
                        break
                    candidate_rect = (tx, ty, tx + tw_full, ty + th_full)
                    has_overlap = any(_rects_overlap(candidate_rect, occ) for occ in occupied_rects)
                    if not has_overlap:
                        return ty
                return ty_candidate  # 实在找不到就用原始位置

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

                text_block_h = th + pad * 2   # 文字块总高度（文字+内边距）
                tw_full = tw + pad * 2         # 文字块总宽度

                tx = min(x0, W - tw_full)
                tx = max(0, tx)

                # 优先放 bbox 上方，若不足则放下方
                if y0 - text_block_h >= 0:
                    ty = y0 - text_block_h
                else:
                    ty = min(y1_px + pad, H - text_block_h)
                    ty = max(0, ty)

                # 防碰撞：检测并避免与已有标签重叠
                ty = _find_non_overlapping_ty(tx, ty, tw_full, text_block_h)
                ty = max(0, min(ty, H - text_block_h))

                # 更新扩展 bbox
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

                # 横向也可能需要扩展（文字比 bbox 宽时）
                tx_right = tx + tw_full
                expanded_bboxes[idx][0] = min(
                    expanded_bboxes[idx][0], max(0.0, tx / W)
                )
                expanded_bboxes[idx][2] = max(
                    expanded_bboxes[idx][2], min(1.0, tx_right / W)
                )

                # 绘制深色背景衬底 + 文字
                draw.rectangle(
                    [tx, ty, tx + tw_full, ty + text_block_h],
                    fill=text_bg_color,
                )
                draw.text((tx + pad, ty + pad), display_label,
                          fill=text_color, font=font)

                # 记录已占用区域
                occupied_rects.append((tx, ty, tx + tw_full, ty + text_block_h))

        # ── 转 base64 返回 ───────────────────────────────────────────────
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        b64_out = base64.b64encode(buf.getvalue()).decode()
        return f"data:image;base64,{b64_out}", expanded_bboxes

    except Exception as e:
        print(f"[sam3_highlight] 高亮处理失败: {e}")
        return image_path_or_b64, expanded_bboxes


# ═══════════════════════════════════════════════════════════════════════════
# 二次 SAM3 验证工具
# ═══════════════════════════════════════════════════════════════════════════

def crop_image_region_b64(image_path_or_b64: str, bbox_norm: list) -> str:
    """
    从图像中裁剪归一化坐标 bbox_norm=[x0,y0,x1,y1] 对应的区域，返回 base64。
    失败时返回原始 image_path_or_b64。
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
        print(f"[crop_region] 裁剪失败: {e}")
        return image_path_or_b64


def _bbox_iou(b1, b2):
    """计算两个归一化 bbox 的 IoU。"""
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
    对注意力图 bbox 区域（按面积最大的 max_att_bboxes 个）裁剪后再次调用 SAM3，
    返回新发现的 bbox/label 映射供调用方送入 LPD（默认不修改原图像素）。

    新方案说明：
      - 默认 draw_on_image=False：不在原图绘制任何标注，保持原图像素纯净
      - 返回 new_entries_per_img 供调用方把 bbox 加入 bounding_boxes 并统一
        进入 LPD，在 LPD 之后通过外侧图注 legend 显示实体标签

    Returns:
        (img_url_modified, found_any, new_entries_per_img)
        - img_url_modified: list，原图列表（draw_on_image=False 时与输入一致）
        - found_any: bool，是否发现新区域
        - new_entries_per_img: dict[int, list[(box, label)]]，新发现 bbox 及其实体名
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

        # 收集该图已有的首次 SAM3 框，用于去重
        existing_boxes = []
        if existing_sam3_bboxes_per_img and imgidx in existing_sam3_bboxes_per_img:
            existing_boxes = existing_sam3_bboxes_per_img[imgidx]

        # 按面积降序取 top-N 注意力 bbox 做裁剪检测
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
                continue   # 跳过过小的注意力 bbox

            cropped_b64 = crop_image_region_b64(src_img, att_bbox)
            try:
                crop_results = call_grounding_expert(
                    cropped_b64, entity_list, expert_url=sam3_url
                )
            except Exception as e:
                print(f"[secondary_sam3] SAM3 调用失败: {e}")
                continue

            for entity, bboxes in crop_results.items():
                sorted_bboxes = sorted(
                    bboxes,
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                    reverse=True,
                )[:max_per_entity]
                for b in sorted_bboxes:
                    # 裁剪图坐标 → 原图归一化坐标
                    ox0 = x0_a + b[0] * w_att
                    oy0 = y0_a + b[1] * h_att
                    ox1 = x0_a + b[2] * w_att
                    oy1 = y0_a + b[3] * h_att
                    area = (ox1 - ox0) * (oy1 - oy0)
                    if area < 0.0005:
                        continue
                    new_box = [ox0, oy0, ox1, oy1]
                    # IoU 去重：与首次 SAM3 框和已接受的二次框比对
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

        # 按面积降序截断至 max_new_bboxes，防止噪声累积
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

        # 兼容旧调用：仅当显式要求 draw_on_image 才在原图绘制（不推荐，新方案应使用 LPD 图注）
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
# 注意力热力图保存工具
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
    将注意力图渲染为热力图并叠加到原图上保存。

    参数:
        att_map: 2D 注意力图，shape (H_att, W_att)，值域 [0, 1]
        image_path_or_b64: 原始图像路径或 base64 字符串
        save_path: 输出文件路径（.png / .jpg）
        alpha: 热力图叠加透明度，0=全透明，1=不透明，默认 0.55
        colormap: matplotlib colormap 名称，默认 "jet"
        bboxes_norm: list of [x0,y0,x1,y1]，注意力 bbox（归一化），绿色绘制
        sam3_bboxes_norm: list of [x0,y0,x1,y1]，SAM3 补充 bbox（归一化），橙色绘制
        title: 图像标题（可选）
    """
    import matplotlib
    matplotlib.use("Agg")   # 非交互后端，避免 GUI 依赖
    import matplotlib.patches as mpatches

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # ── 读取原图 ──────────────────────────────────────────────────────────
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
        print(f"[heatmap] 读取原图失败: {e}")
        return

    W, H = orig_img.size   # 原图像素宽高

    # ── 将注意力图上采样到原图尺寸 ────────────────────────────────────────
    att_resized = np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # ── 应用 colormap，生成 RGBA 热力图 ────────────────────────────────────
    cmap = plt.get_cmap(colormap)
    heatmap_rgba = cmap(att_resized)              # (H, W, 4) float [0,1]
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil  = Image.fromarray(heatmap_rgb)

    # ── 叠加：原图 * (1-alpha) + 热力图 * alpha ─────────────────────────
    orig_arr    = np.array(orig_img, dtype=np.float32)
    heatmap_arr = np.array(heatmap_pil, dtype=np.float32)
    blended_arr = np.clip((1 - alpha) * orig_arr + alpha * heatmap_arr, 0, 255).astype(np.uint8)

    # ── 用 matplotlib 绘制 bbox ──────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(blended_arr)
    ax.axis("off")

    legend_patches = []
    # 注意力 bbox（绿色实线）
    if bboxes_norm:
        for box in bboxes_norm:
            x0n, y0n, x1n, y1n = box
            rect = mpatches.Rectangle(
                (x0n * W, y0n * H), (x1n - x0n) * W, (y1n - y0n) * H,
                linewidth=2, edgecolor="#00ff00", facecolor="none"
            )
            ax.add_patch(rect)
        legend_patches.append(mpatches.Patch(edgecolor="#00ff00", facecolor="none", label="Attention bbox"))

    # SAM3 补充 bbox（橙色虚线）
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
    将 accept_att[img_idx] 中所有 token 的注意力图取均值，得到聚合热力图。

    参数:
        accept_att: {img_idx: {token_k: att_map (H, W)}}
        img_idx: 目标图像索引

    返回:
        聚合注意力图 (H, W)，值域 [0, 1]；若无数据返回 None
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
    批量处理多条 messages，返回 batched inputs。
    使用 padding_side='left' 确保生成时对齐。
    
    Args:
        messages_list: list of messages，每个元素是一条完整的对话 messages
        processor: AutoProcessor
        model: 模型
    
    Returns:
        batched_inputs: 批量化的输入 tensor
    """
    # 设置 left padding
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
    
    # 合并 video_kwargs（如果有的话）
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
    
    # 恢复 padding_side
    processor.tokenizer.padding_side = original_padding_side
    
    return batched_inputs

def batch_messages2out(model, processor, batched_inputs, return_first_logits=False,
                         option_token_ids=None, return_full_first_logits=False):
    """
    批量推理生成，支持 left-padded 输入。

    Args:
        model: 模型
        processor: AutoProcessor
        batched_inputs: batch_get_inputs 返回的批量输入
        return_first_logits: bool, 是否同时返回每条样本的首 token 选项概率与熵
            （用于 Difficulty-Aware Router v1 —— 只看 A/B/C/D 4 个 token）
        option_token_ids: dict[str,int]，如 {"A":32,"B":33,"C":34,"D":35}；
            仅在 return_first_logits=True 时使用。
        return_full_first_logits: bool, 若 True 则同时返回每条样本的完整词表
            first-token logits（np.ndarray [vocab]），供 RouterV2 计算
            全词表熵、option_mass、logit_gap 等通用特征。

    Returns:
        根据参数组合返回不同元组：
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
    # 若需要 full logits 必然也要 scores
    need_scores = return_first_logits or return_full_first_logits

    with torch.no_grad():
        if need_scores:
            gen_out = model.generate(
                **batched_inputs, use_cache=True, max_new_tokens=4096, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
            generated_ids = gen_out.sequences
            scores = gen_out.scores   # tuple, 长度 = new_tokens，每元素 [B, vocab]
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
    与 messages2out 相同，但额外返回 first answer token 的 logits / entropy，
    用于难度感知路由（threshold-free router）。

    Args:
        option_token_ids: dict, e.g. {"A": 32, "B": 33, "C": 34, "D": 35}
            若提供，则返回 4 选项 softmax 概率 + 归一化熵
    Returns:
        output_text: list[str]
        end_ques: int
        first_logits: np.ndarray [vocab_size]  首个生成 token 的 logits
        option_probs: np.ndarray [4]            4 选项概率（如 option_token_ids 提供）
        answer_entropy: float                   4 选项 softmax 熵（如 option_token_ids 提供）
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

    # 首个生成 token 的 logits（B=1 情形）
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
    """一个辅助函数，将 content 图像(BGRA)居中放置在 canvas 画布(BGRA)上"""
    canvas_h, canvas_w, _ = canvas_bgra.shape
    content_h, content_w, _ = content_bgra.shape

    if content_h > canvas_h or content_w > canvas_w:
        scale = min(canvas_h / content_h, canvas_w / content_w)
        new_h, new_w = int(content_h * scale), int(content_w * scale)
        content_bgra = cv2.resize(content_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        content_h, content_w = new_h, new_w

    paste_x = (canvas_w - content_w) // 2
    paste_y = (canvas_h - content_h) // 2
    
    # 使用Alpha通道作为蒙版来粘贴
    alpha_mask = content_bgra[:, :, 3] / 255.0
    
    # 遍历每个颜色通道
    for c in range(0, 3):
        canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c] = \
            alpha_mask * content_bgra[:, :, c] + \
            (1 - alpha_mask) * canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, c]
            
    # 更新画布的alpha通道
    canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3] = \
        np.maximum(canvas_bgra[paste_y:paste_y+content_h, paste_x:paste_x+content_w, 3], content_bgra[:, :, 3])
        
    return canvas_bgra

def decompose_bbox_by_alpha(image_bgra, bbox, alpha_threshold=10):
    """
    将单个BBox根据其Alpha通道分解为多个不包含透明区域的子BBox。

    Args:
        image_bgra (np.array): 4通道的BGRA格式图像。
        bbox (list or tuple): 单个边界框 [x0, y0, x1, y1]。
        alpha_threshold (int): 用于判断像素是否透明的阈值。
                               高于此值的Alpha被认为是不透明的。

    Returns:
        list: 一个包含多个子BBox [x, y, w, h] 的列表。
              如果BBox内没有不透明区域，则返回空列表。
    """
    x0, y0, x1, y1 = bbox
    img_h, img_w, _ = image_bgra.shape

    # 确保BBox坐标在图像范围内
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)

    if x0 >= x1 or y0 >= y1:
        return []

    # 1. 提取BBox内的区域，并获取其Alpha通道
    roi = image_bgra[y0:y1, x0:x1]
    alpha_channel = roi[:, :, 3]  # BGRA格式的Alpha通道在索引3

    # 2. 二值化Alpha通道
    # 使用cv2.THRESH_BINARY，高于阈值的像素变为255，否则为0
    _, mask = cv2.threshold(alpha_channel, alpha_threshold, 255, cv2.THRESH_BINARY)

    # 3. 寻找轮廓
    # cv2.RETR_EXTERNAL 只检测最外层的轮廓，这正是我们需要的
    # cv2.CHAIN_APPROX_SIMPLE 压缩轮廓，节省内存
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. 将轮廓转换为BBox
    sub_bboxes = []
    for contour in contours:
        # 计算轮廓的边界框 (x, y, w, h)
        sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(contour)
        
        # 将子BBox的相对坐标转换回原图的绝对坐标
        abs_x0 = x0 + sub_x
        abs_y0 = y0 + sub_y
        abs_x1 = abs_x0 + sub_w
        abs_y1 = abs_y0 + sub_h
        
        sub_bboxes.append([abs_x0, abs_y0, abs_x1, abs_y1])
        
    return sub_bboxes

def merge_overlapping_bboxes(bboxes):
    """
    合并列表中所有重叠的BBox。

    Args:
        bboxes (list): 一个包含多个BBox [x0, y0, x1, y1] 的列表。

    Returns:
        list: 一个新的BBox列表，其中所有重叠的BBox已被合并。
    """
    if not bboxes:
        return []

    # 使用索引来操作，避免在迭代时修改列表
    bboxes = [list(b) for b in bboxes] # 确保是可修改的列表

    while True:
        merged_one = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                box1 = bboxes[i]
                box2 = bboxes[j]

                # 检查是否重叠
                # 如果一个box的右边在另一个的左边之外，或者上边在下边之外，则不重叠
                is_overlapping = not (box1[2] < box2[0] or  # box1在box2左侧
                                      box1[0] > box2[2] or  # box1在box2右侧
                                      box1[3] < box2[1] or  # box1在box2上方
                                      box1[1] > box2[3])   # box1在box2下方

                if is_overlapping:
                    # 合并两个BBox
                    new_x0 = min(box1[0], box2[0])
                    new_y0 = min(box1[1], box2[1])
                    new_x1 = max(box1[2], box2[2])
                    new_y1 = max(box1[3], box2[3])
                    
                    # 用合并后的大BBox替换第一个，并删除第二个
                    bboxes[i] = [new_x0, new_y0, new_x1, new_y1]
                    bboxes.pop(j)
                    
                    # 因为我们合并了，需要从头开始重新检查
                    merged_one = True
                    break # 跳出内层j循环
                else:
                    j += 1
            
            if merged_one:
                break # 跳出外层i循环，重新开始while True
            else:
                i += 1
        
        # 如果完整遍历一次后没有任何合并发生，则结束
        if not merged_one:
            break
            
    return bboxes

def compact_and_center_with_relative_pos(imgidx, img_nums, image, normalized_bboxes, n=1,
                                          bbox_colors=None, bbox_labels=None,
                                          draw_color_border=True,
                                          border_thickness=2,
                                          overlay_bboxes=None):
    """
    将 BBox 区域紧凑排列（保留相对位置），然后居中放置在透明背景上。

    参数:
    - n (int): 一个阈值。找出所有被至少 n 个 BBox 覆盖的区域，
              然后返回所有与这些区域有交集的原始 BBox 的并集。
    - bbox_colors / bbox_labels: [已废弃路径] 为 normalized_bboxes 的每个框
        分配颜色/标签。由于 LPD 会做重叠合并 + alpha 分解，原始 bbox 粒度
        会丢失，所以推荐改用 overlay_bboxes 参数。当前为兼容性保留：
        仅用于生成 legend 条目（不再在紧凑画布上根据 bbox_colors 绘制边框）。
    - draw_color_border (bool): 是否绘制彩色边框（作用于 overlay_bboxes）。
    - border_thickness (int): 边框像素厚度。
    - overlay_bboxes (list[dict] | None): 覆盖层 bbox 列表，每个元素:
        {"bbox_norm": [x0,y0,x1,y1], "color": (R,G,B), "label": "entity name"}
        - bbox_norm: 原图归一化坐标
        - color: RGB 颜色
        - label: 图注标签
        overlay_bboxes 不影响 LPD 的裁剪 / 紧凑布局，仅作为额外的彩色
        "提示层"叠加：通过 x_map / y_map 把原图像素位置映射到紧凑画布
        位置，然后只对该位置落在紧凑画布内的部分绘制彩色细边框。
        这样即使 SAM3 bbox 被注意力 bbox 包含（或与之合并），仍然能独立
        保留 SAM3 bbox 的视觉可见性。

    Returns:
        (imgs_list, return_bboxes, used_colors_labels)
        - imgs_list: list[str]，紧凑图的 base64 列表（单元素）或 None
        - return_bboxes: list[list[float]]，bbox（原图归一化坐标）
        - used_colors_labels: list[(rgb_tuple, label)]，实际在紧凑图中
          可见并绘制了边框的 overlay_bboxes 对应的去重图注条目
    """
    # ✅ 1. 解码并统一转换为4通道BGRA格式
    if not image.startswith('data:image;base64,'):
        image64 = image_to_base64(image).split(',')[1]
    elif ',' in image:
        image64 = image.split(',')[1]
    image_data = base64.b64decode(image64)
    pil_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    img_cv_bgra = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGRA)
    img_h, img_w, _ = img_cv_bgra.shape

    # ✅ 2. 修复空BBox情况的返回值
    if not normalized_bboxes:
        return None, [], []

    # --- 坐标转换 ---
    initial_pixel_bboxes = []
    for n_box in normalized_bboxes:
        nx0, ny0, nx1, ny1 = n_box
        x0, y0 = int(nx0 * img_w), int(ny0 * img_h)
        x1, y1 = int(nx1 * img_w), int(ny1 * img_h)
        initial_pixel_bboxes.append([x0, y0, x1, y1])

    # --- 重叠阈值筛选贡献 BBox 的并集 ---
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

    # --- 分解：作用于最终的并集区域 ---
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

    # --- 紧凑排列逻辑 ---
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

    # ── 构建原图 → 紧凑图的像素级映射数组（便于 overlay_bboxes 精准映射） ──
    # 对任意原图像素坐标 (x, y)，其紧凑图坐标 = (x_pix_map[x], y_pix_map[y])，
    # 若该像素落在"被间隙挤掉"的区域，则映射为 -1（不可见）
    x_pix_map = np.full(img_w + 1, -1, dtype=np.int32)
    y_pix_map = np.full(img_h + 1, -1, dtype=np.int32)
    # x 方向：对每段 [x_coords[i], x_coords[i+1]]，若该段在 x_map 中有递进则映射；否则 -1
    for i in range(len(x_coords) - 1):
        sx, ex = x_coords[i], x_coords[i+1]
        if any(b[0] < ex and b[2] > sx for b in bboxes):
            # 该段存在于紧凑图，线性映射
            for xx in range(max(0, sx), min(img_w, ex) + 1):
                x_pix_map[xx] = x_map[sx] + (xx - sx)
    x_pix_map[x_coords[-1]] = x_map[x_coords[-1]] if x_coords else 0
    for i in range(len(y_coords) - 1):
        sy, ey = y_coords[i], y_coords[i+1]
        if any(b[1] < ey and b[3] > sy for b in bboxes):
            for yy in range(max(0, sy), min(img_h, ey) + 1):
                y_pix_map[yy] = y_map[sy] + (yy - sy)
    y_pix_map[y_coords[-1]] = y_map[y_coords[-1]] if y_coords else 0

    # ✅ 4. 创建4通道透明画布，并从4通道源图粘贴
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

    # ── overlay_bboxes：在紧凑画布上根据像素映射绘制彩色边框 ────────────
    used_colors_labels = []
    _seen_color_label = set()
    if draw_color_border and overlay_bboxes:
        _rgba = cv2.cvtColor(composite_image_bgra, cv2.COLOR_BGRA2RGBA)
        _pil_comp = Image.fromarray(_rgba)
        _draw = ImageDraw.Draw(_pil_comp)

        def _map_x(px):
            px = int(max(0, min(img_w, px)))
            # 若该位置被挤掉，向两侧回退寻找有效映射
            if x_pix_map[px] >= 0:
                return int(x_pix_map[px])
            # 向左回退
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
            # 映射到紧凑坐标
            cx0 = _map_x(px0); cx1 = _map_x(px1)
            cy0 = _map_y(py0); cy1 = _map_y(py1)
            if cx0 < 0 or cy0 < 0 or cx1 < 0 or cy1 < 0:
                continue
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            # 裁剪到紧凑画布边界
            cx0 = max(0, min(new_total_width - 1, cx0))
            cx1 = max(0, min(new_total_width - 1, cx1))
            cy0 = max(0, min(new_total_height - 1, cy0))
            cy1 = max(0, min(new_total_height - 1, cy1))
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            # 检查该框实际覆盖区域是否有可见像素（alpha>0），避免为空白区域画框
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

    # ✅ 5. 创建最终的4通道透明画布，并居中粘贴
    final_canvas_bgra = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    final_img_bgra = place_on_center(final_canvas_bgra, composite_image_bgra)

    final_img_rgba = cv2.cvtColor(final_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_centered = Image.fromarray(final_img_rgba)

    # ✅ 6. 对紧凑图进行缩放和返回
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
    merge_overlapping_bboxes 的变体：合并时同时维护每个合并 bbox 的原索引集合。

    Args:
        bboxes: list[[x0,y0,x1,y1]]
        source_indices: list[int]，与 bboxes 等长，每个 bbox 的原始索引
    Returns:
        merged_bboxes, merged_source_sets
          - merged_bboxes: list[list[int]]
          - merged_source_sets: list[set[int]]，与 merged_bboxes 一一对应
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
    从注意力图中找到前n个最受关注的连通区域。

    这个函数通过以下步骤工作：
    1. 使用阈值对注意力图进行二值化，以识别高关注度区域。
    2. 对二值化图进行连通域分析，找到所有独立的区域。
    3. 为每个区域计算一个"关注度分数"（区域内所有注意力值的总和）。
    4. 根据分数对所有区域进行降序排序。
    5. 返回排名前n的区域的边界框。如果总区域数小于n，则返回所有区域。

    参数:
    att_map (np.ndarray): 二维的注意力图，值通常在0到1之间。
    n (int): 需要寻找的顶部区域的数量。
    threshold (float, optional): 用于二值化的阈值。默认为 0.5。

    返回:
    list: 一个包含边界框的列表。每个边界框格式为 [x_min, y_min, x_max, y_max]。
          列表按关注度分数降序排列。
    """
    att_map = np.array(norm_att)

    # GRACE: 自适应 Otsu 阈值，替代固定阈值，自动适应各注意力图分布
    if use_otsu and att_map.max() > att_map.min():
        try:
            threshold = threshold_otsu(att_map)
        except Exception:
            pass  # 异常时回退到固定阈值

    binarized_map = (att_map >= threshold)
    if not np.any(binarized_map):  # 阈值化后无任何区域
        return [], 0
        
    labeled_map = label(binarized_map, connectivity=2)
    regions = regionprops(labeled_map)

    # 2. 为每个区域计算分数并存储
    scored_regions = []
    for region in regions:
        # 创建一个与att_map同样大小的掩码，其中只有当前区域为True
        mask = (labeled_map == region.label)
        # 计算该区域内所有像素在原始att_map上的注意力值总和作为分数
        score = np.sum(att_map[mask])
        scored_regions.append({
            'score': score,
            'bbox': region.bbox  # bbox格式为 (y0, x0, y1, x1)
        })
        # if 0 == region.bbox[0] and 0 == region.bbox[1]: return [],0

    # 3. 根据分数对区域进行降序排序
    sorted_regions = sorted(scored_regions, key=lambda r: r['score'], reverse=True)

    # # 4. 选择前n个区域（如果不够n个，则全选）
    # if n > len(sorted_regions):
    #     n = len(sorted_regions)
    # top_n_regions = sorted_regions[:n]

    final_boxes = []
    # 5. 提取并格式化边界框
    get_num = 0
    for region in sorted_regions:
        y0, x0, y1, x1 = region['bbox']
        # 转换为 [x_min, y_min, x_max, y_max] 格式
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
    从注意力图中提取裁剪框，支持 EGAF 和 GRACE 两种增强模式，并可保存注意力热力图。

    新方案（重要更新）：
      SAM3 bbox 不再在原图像素上绘制，改为：
        1. bbox 进入 LPD，在紧凑画布上绘制彩色细边框（不占像素空间）
        2. LPD 输出后，在图像外侧追加图注（legend），与边框颜色一一对应
        3. 不同 SAM3 实体自动分配不同颜色

    参数:
        enable_saaa: bool, 旧 EGAF 模式开关（含 Rank-Based Fusion）
        entity_text: str, 实体文本
        expert_bboxes_per_img: dict, {img_idx: bboxes}，EGAF 专家注意力掩码
        egaf_fusion_mode / egaf_expert_weight / expert_reliability: EGAF 融合参数
        entity_token_indices: list of int, 实体 token 位置（加速用）
        entity_token_map: dict, 实体分组 token 映射

        enable_grace: bool, GRACE 模式开关（相对注意力 + Otsu + SAM3 补充）
        use_otsu: bool, 是否使用 Otsu 自适应阈值（GRACE 模式下自动为 True）
        sam3_supplement_bboxes_per_img: dict, {img_idx: [[x0,y0,x1,y1],...]}，
            SAM3 文本提示检测得到的补充 bbox（归一化坐标）
        sam3_entity_labels_per_img: dict, {img_idx: [label,...]}，
            与 sam3_supplement_bboxes_per_img 等长，每个 bbox 对应的实体名；
            用于在高亮图上绘制文字标注。None 则不绘制文字。

        heatmap_save_dir: str | None, 热力图保存目录。
            None 表示不保存；非 None 则在该目录下生成如下文件:
              {sample_id}_img{img_idx}_s{sigma}_t{thresh}_agg_heatmap.png  (聚合均值热力图)
        sample_id: str, 样本唯一标识（用于文件命名），默认 "sample"
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
                # GRACE: TAD 逐 token 遍历 + Otsu 自适应阈值（无实体过滤和聚合）
                accept_att = process_grace(
                    dicts, start_k, end_k, attention, inputs, img_start, img_end, s)
            elif enable_saaa:
                # 旧 EGAF: 实体 token 加速 + Rank-Based Fusion
                accept_att = process_egaf(
                    dicts, start_k, end_k, attention, inputs, img_start, img_end, s,
                    expert_bboxes_per_img=expert_bboxes_per_img,
                    egaf_fusion_mode=egaf_fusion_mode,
                    egaf_expert_weight=egaf_expert_weight,
                    entity_token_indices=entity_token_indices,
                    entity_token_map=entity_token_map,
                    expert_reliability=expert_reliability)
            else:
                # 原始 HiDe TAD（基准）
                accept_att = process(dicts, start_k, end_k, attention, inputs, img_start, img_end, s)

            # ── 保存聚合注意力热力图 ──────────────────────────────────────────
            if heatmap_save_dir is not None:
                for _img_idx in accept_att:
                    agg_map = build_aggregated_heatmap(accept_att, _img_idx)
                    if agg_map is None:
                        continue
                    # 收集当前 s/t 组合下该图像的最终 bbox（此时 bounding_boxes 尚未建立，
                    # 等热力图保存后再建立；此处仅传注意力 bbox，SAM3 补充框在下方单独绘制）
                    _sam3_bboxes = (
                        sam3_supplement_bboxes_per_img.get(_img_idx, [])
                        if sam3_supplement_bboxes_per_img else []
                    )
                    _img_src = img_url[_img_idx] if _img_idx < len(img_url) else img_url[0]
                    _fname = f"{sample_id}_img{_img_idx}_s{s}_t{t}_agg_heatmap.png"
                    _fpath = os.path.join(heatmap_save_dir, _fname)
                    # 先不传 attention bbox（尚未计算），SAM3 bbox 可以先绘制
                    save_attention_heatmap(
                        att_map=agg_map,
                        image_path_or_b64=_img_src,
                        save_path=_fpath,
                        alpha=0.55,
                        colormap="jet",
                        bboxes_norm=None,           # 注意力 bbox 在下方覆盖更新
                        sam3_bboxes_norm=_sam3_bboxes if _sam3_bboxes else None,
                        title=f"[{sample_id}] Aggregated Attn | s={s} t={t:.2f} | {entity_text[:60]}",
                    )

            imgs_words_att_box = {}
            for img_idx in accept_att:
                accept_word_att = accept_att[img_idx]
                words_att_box = {}
                for word in accept_word_att:
                    att_map = accept_word_att[word][0]
                    # 注意力 bbox 提取始终使用固定阈值 t（不使用 Otsu），
                    # 保证 GRACE 与 HiDe 产生一致的紧凑 bbox。
                    # Otsu 对偏态注意力分布会产生系统性低阈值，导致 bbox 过大、
                    # 空间关系丢失（尤其影响 relative_position 类问题）。
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
                            # 计算 box 面积
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
            # ── 保存原始注意力 bbox 副本（用于热力图绘制，不受后续 SAM3 合并和 LPD 变换影响）──
            # 多个 token 可能对同一区域各自产生 bbox，需要先合并重叠框再保存，
            # 避免热力图上同一高注意力区域显示多个重叠的绿框。
            original_att_bboxes = {}
            for imgidx in bounding_boxes:
                original_att_bboxes[imgidx] = merge_overlapping_bboxes(
                    [list(b) for b in bounding_boxes[imgidx]]
                )

            # ── 先保留一套不含视觉专家 bbox 的 HiDe 式 LPD 输出 ────────────────
            # 该分支严格只使用注意力 bbox 和原图，供最终推理额外追加一张
            # attention-only 的 LPD 图像，不引入任何 SAM3 标注。
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

            # ── SAM3 bbox 处理（新方案：overlay 方式叠加彩色边框 + 外侧图注）──
            #
            # 设计原则：
            #   1. 原图像素完全不被装饰
            #   2. SAM3 bbox 不与注意力 bbox 合并，作为独立 overlay 层传入 LPD
            #   3. LPD 通过原图→紧凑图的像素映射，在紧凑画布上按 SAM3 bbox 的
            #      原始大小绘制彩色边框（保留 SAM3 小目标的独立可见性）
            #   4. 注意力 bbox 本身不绘制任何边框（LPD 输出与纯 HiDe 一致）
            #   5. SAM3 无检测结果时，overlay_bboxes 为空 → LPD 不画边框、
            #      legend 为空 → 最终 highlight_imgs 完全等同于 hide_highlight_imgs
            #
            # 颜色分配：使用 assign_entity_colors 按实体名稳定分色（同名实体共用色）
            #
            img_url_for_crop = list(img_url)   # 保留变量（现在不再修改）

            # 构建每张图的 overlay_bboxes 列表（仅包含 SAM3 来源的 bbox）
            # item: {"bbox_norm": [x0,y0,x1,y1], "color": (R,G,B), "label": str}
            overlay_bboxes_per_img = {}
            extra_legend_entries = []   # 被面积守卫跳过的大框，仅入 legend 提示

            if sam3_supplement_bboxes_per_img:
                # 跨实体稳定的颜色映射（同名实体共用颜色）
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
                        # 面积守卫：仅对几乎覆盖整图（≥80%）的框跳过，
                        # 如 "scene"、"image" 这种无语义区分度的全图框。
                        # 旧阈值 30% 过于激进，会误跳 "left tower" 等占图较大
                        # 但仍有明确语义的目标实体。
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
                        # 作为 overlay（不论是否被 attention 包含，都保留 SAM3 的独立可见性）
                        overlay_bboxes_per_img.setdefault(sup_imgidx, []).append({
                            "bbox_norm": list(orig_box),
                            "color": color,
                            "label": label or "",
                        })
                        # 若 SAM3 在注意力 bbox 外部，需把它也加入 bounding_boxes，
                        # 以便 LPD 裁剪时包含该区域（否则裁不到 SAM3 目标）
                        if not is_inside:
                            if sup_imgidx not in bounding_boxes:
                                bounding_boxes[sup_imgidx] = []
                            bounding_boxes[sup_imgidx].append(list(orig_box))

            # ── LPD 处理 + LPD 后追加外侧图注 ─────────────────────────────────
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
                    # 合并 LPD 内可见的 + 跳过大框 legend，去重
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

            # ── 覆盖保存带 bbox 标注的热力图 ──────────────────────────────────
            # 使用 original_att_bboxes（LPD 处理前保存的原始注意力 bbox），
            # 确保 GRACE 和 HiDe 在注意力图相同时绘制一致的绿色 bbox。
            # bounding_boxes 此时已被 LPD 和 SAM3 合并修改，不适合用于热力图标注。
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
