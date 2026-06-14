import os
from transformers import AutoTokenizer, AutoProcessor
from modeling_qwen2_5_vl_re_infer import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np
from tqdm import tqdm
from utiles import *
import json
import sys
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
from Get_box import (messages2out, messages2out_with_logits,
                     messages2att, from_img_and_att_get_cropbox,
                     get_inputs, batch_get_inputs, batch_messages2out,
                     compact_and_center_with_relative_pos,
                     secondary_sam3_check_on_att_bboxes,
                     apply_sam3_highlight_to_image,
                     assign_entity_colors, add_legend_to_lpd_image,
                     _bbox_iou, _LEGEND_PALETTE)
import shutil
import cv2
import re
import math


# ──────────────────────────────────────────────────────────────────────────────
# Router v2 辅助：从完整词表 first_logits 抽取 v2 的 13 维特征
# ──────────────────────────────────────────────────────────────────────────────
_OPT_PAT_V2 = re.compile(r"(?:^|\n|\s)\(?([A-Z])[\)\.\:]", re.MULTILINE)


def _parse_options_v2(question: str):
    """解析题干里从 A 起连续出现的选项字母；兜底返回 ABCD"""
    if not question:
        return ["A", "B", "C", "D"]
    found = set()
    for m in _OPT_PAT_V2.finditer(question):
        found.add(m.group(1))
    letters = []
    for c in range(ord("A"), ord("Z") + 1):
        ch = chr(c)
        if ch in found:
            letters.append(ch)
        else:
            break
    return letters if len(letters) >= 2 else ["A", "B", "C", "D"]


def build_answer_instr_v2(question: str) -> str:
    """
    v2 统一的"简洁直接字母作答"指令 —— 用于 direct-answer 与 GRACE 最终推理。

    设计要点：
      1. 末尾 "Answer:" 强诱导首 token = 选项字母（与 features_train_v2.jsonl
         训练分布对齐，保证 router 特征匹配）。
      2. 不含 <FINAL_OUTPUT> tag。下游评估函数取答案字符串的最后字符，
         所以只需保证最后字符是答案字母即可（首 token == 最后字符）。
      3. 这样保证 ori 分支与 GRACE 分支用同一套 prompt，不会因为 GRACE 换
         prompt 导致"对的答案被改错"（regret）。
    """
    letters = _parse_options_v2(question)
    letters_str = (", ".join(letters[:-1]) + f", or {letters[-1]}"
                   if len(letters) > 1 else letters[0])
    return (
        f"\nAnswer the multiple-choice question with ONLY the single letter "
        f"of the correct option ({letters_str}). "
        f"Do not output any word, punctuation, tag, explanation or whitespace. "
        f"Your entire response must be exactly one character — the option letter.\n"
        f"Answer:"
    )


def compute_router_v2_features(first_logits, question, option_token_ids_all):
    """
    给定 first-token vocab logits（1D np.ndarray [V]）+ question + A-Z token map，
    计算 RouterV2 可能用到的数值特征（6 维完备集合，router 按自身需要 pick）。

    返回特征：
        answer_topp                 选项 top-1 概率
        answer_margin               top1 - top2
        answer_entropy              选项 K-way 熵 (nat)
        answer_entropy_norm         选项 K-way 熵 / log(K)    ★ 推荐单特征
        option_mass                 选项 token 总概率
        logit_gap_opt_nonopt        max 选项 logit - max 非选项 logit
        vocab_full_entropy_norm     首 token 全词表归一化熵（备用）
    """
    if first_logits is None:
        return None
    try:
        import numpy as np
        logits = np.asarray(first_logits, dtype=np.float64)
        # softmax on full vocab (稳定计算)
        l = logits - logits.max()
        p_full = np.exp(l)
        p_full = p_full / max(p_full.sum(), 1e-12)

        letters = _parse_options_v2(question)
        opt_ids = []
        for L in letters:
            if L in option_token_ids_all:
                opt_ids.append(option_token_ids_all[L])
        if len(opt_ids) < 2:
            return None
        K = len(opt_ids)

        p_opt_raw = np.array([p_full[i] for i in opt_ids], dtype=np.float64)
        option_mass = float(p_opt_raw.sum())
        p_opt = p_opt_raw / max(p_opt_raw.sum(), 1e-12)
        sorted_p = np.sort(p_opt)[::-1]
        answer_topp = float(sorted_p[0])
        answer_margin = (float(sorted_p[0] - sorted_p[1])
                         if len(sorted_p) > 1 else float(sorted_p[0]))

        # K-way 选项熵（归一化到 log(K)，跨不同选项数可比）
        answer_entropy = -float((p_opt * np.log(p_opt + 1e-12)).sum())
        answer_entropy_norm = answer_entropy / max(math.log(K), 1e-12)

        # vocab full entropy (normalized)
        vocab_H = -float((p_full * np.log(p_full + 1e-12)).sum())
        V = p_full.size
        vocab_full_entropy_norm = vocab_H / max(math.log(V), 1e-12)

        # logit gap
        max_opt_logit = float(max(logits[i] for i in opt_ids))
        mask = np.ones_like(logits, dtype=bool)
        for i in opt_ids:
            mask[i] = False
        max_nonopt_logit = float(logits[mask].max())
        logit_gap_opt_nonopt = max_opt_logit - max_nonopt_logit

        return {
            "answer_topp": answer_topp,
            "answer_margin": answer_margin,
            "answer_entropy": answer_entropy,
            "answer_entropy_norm": answer_entropy_norm,
            "option_mass": option_mass,
            "logit_gap_opt_nonopt": logit_gap_opt_nonopt,
            "vocab_full_entropy_norm": vocab_full_entropy_norm,
        }
    except Exception:
        return None


def unpack_att_result(result_item):
    if len(result_item) >= 6:
        img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs = result_item[:6]
    else:
        img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes = result_item
        hide_highlight_imgs = []
    return img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs


def append_visual_inputs(content, ori_img_url, hide_highlight_imgs=None, highlight_imgs=None):
    """
    按固定顺序往 content 追加视觉输入（供最终 VLM 推理使用）：
        1) 原图（ori_img_url）
        2) HiDe 输出图（hide_highlight_imgs）——纯注意力 LPD，无 SAM3 装饰
        3) 带视觉专家 bbox 的 HiDe 输出图（highlight_imgs）——含 SAM3 彩色边框 + 图注

    若 HiDe 输出与 GRACE 输出完全一致（SAM3 无效，overlay 未触发），
    跳过第 2 张避免重复。
    """
    # 第 1 张：原图
    for img in ori_img_url:
        content.append({"type": "image", "image": img})
    # 第 2 张：HiDe 输出（仅当与 GRACE 输出不完全相同时才传）
    if hide_highlight_imgs and hide_highlight_imgs != highlight_imgs:
        for h_img in hide_highlight_imgs:
            content.append({"type": "image", "image": h_img})
    # 第 3 张：带视觉专家 bbox 的 HiDe 输出
    if highlight_imgs:
        for h_img in highlight_imgs:
            content.append({"type": "image", "image": h_img})


def once_infer(model,qwen_processor,sample,messages,img_url,ori_img_url,ques,sig,thre):
    prompt_ques = '''
You are a highly precise language analysis engine. Your sole function is to extract entities (e.g., objects, people) from a user's question, and deconstruct them into a canonical, attribute-based format by strictly following a set of rules and a thinking process.

### Thinking Process
Before generating the final output, you must internally follow these steps in order:

1.  **Identify Core Entities**: Read the entire question and identify all key noun phrases. For example, "the green surfboard," "the purple umbrella."
2.  **Deconstruct Attributes for Each Entity Individually**: **This is the critical step. Before considering the relationship between entities, look at each entity in isolation and apply Rules 2, 3, and 4 to fully deconstruct its attributes.**
    *   For instance, first process "the green surfboard" using Rule 2 to get `surfboard with green color`.
    *   Then, process "the purple umbrella" using Rule 2 to get `umbrella with purple color`.
3.  **Handle Relationships Between Entities**: After all entities have been individually deconstructed, check for spatial or logical relationships between them (Rule 5). If a relationship exists, you will list the **already deconstructed** entities as separate items.
4.  **Assemble and Normalize**:
    *   Gather all the canonical entity strings you have transformed.
    *   Convert all text to lowercase.
    *   Join all entities into a single line, separated by a comma and a space (", ").
5.  **Final Formatting**: Enclose the resulting single-line string within the `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.

### Extraction Rules

**Rule 1: Simple Entities**
If a noun is not described by any modifiers, extract the noun itself.
*   *Example*: "the scooter" becomes `scooter`.

**Rule 2: Adjective Attribute Deconstruction**
If an entity is modified by one or more adjectives, the format must be `noun with [property] [type]`. Chain multiple properties consecutively.
*   **Format**: `noun with [property1] [type1] with [property2] [type2]`
*   **Common Type Mappings**:
    *   Colors (red, blue, black, silver) -> `color`
    *   Sizes (large, small, big) -> `size`
    *   Materials (wooden, metal, plastic) -> `material`
*   *Example*: "the large blue truck" becomes `truck with large size with blue color`.

**Rule 3: Possessive Inversion**
Convert all possessive forms (e.g., `X's Y`) uniformly into the `Y of X` format.
*   *Example*: "the woman's handbag" becomes `handbag of woman`.

**Rule 4: Attributive Prepositional Phrases**
If a prepositional phrase describes a component of an entity (e.g., "in a shirt"), preserve the structure and recursively apply the rules to the entity within the phrase.
*   *Example*: "the man in the green shirt" becomes `man in a shirt with green color`.

**Rule 5: Relational Prepositional Phrases**
If a prepositional phrase describes a relationship between two separate entities (e.g., `next to`, `on the left of`), **extract the entities as separate items, but only after each entity has been fully deconstructed according to the other rules.** Do not include the relational words in the output.
*   *Example 1 (Simple)*: "the dog on the left side of the scooter" becomes `dog, scooter`.
*   **Example 2 (With Attributes)**: **"Is the green surfboard on the left side of the purple umbrella?" becomes `surfboard with green color, umbrella with purple color`.** (Note: The surfboard and umbrella are first deconstructed individually, then listed as separate entities).

**Rule 6: Compound Nouns**
Recognized compound nouns should be treated as a single entity.
*   *Example*: "the traffic light" becomes `traffic light`.

### Output Format Rules
1.  **Sole Output**: The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.
2.  **Single-Line Format**: The output must be a single continuous line of text.
3.  **Delimiter**: Multiple entities must be separated by a comma followed by a space (", ").
4.  **Lowercase**: All output characters must be in lowercase.
5.  **Content Exclusion**: The final entity string must not include articles (a, an, the), question words (what, which, is), or purely relational words (side, next to, of).

Now, following all the rules above, extract the entities from the question below:
{input_text}
'''

    prompt_messages = [{"role": "user","content": [{"type": "text", "text": prompt_ques.format(input_text=ques)}],},]
    text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(prompt_messages,qwen_processor,model)
    prompt_output_text,_ = messages2out(model,qwen_processor,inputs)
    answer_out = prompt_output_text[0].split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
    messages[-1]["content"] = messages[-1]["content"][:-1]
    outputs = {}
    
    #如果提取出了实体词
    if answer_out: 
        messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: "+answer_out})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
        attention,idx2word_dicts,img_start,img_end = messages2att(model,qwen_processor,inputs)  # Retrieve attention from model outputs
        results = from_img_and_att_get_cropbox(inputs,attention, idx2word_dicts, img_url, img_start, img_end,sig,thre)
        for s in sig:
            for t in thre:
                img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs = unpack_att_result(results[str(s)][str(t)])
                messages = [ {"role": "user","content": [],},]
                append_visual_inputs(messages[-1]["content"], ori_img_url, hide_highlight_imgs, highlight_imgs)
                #加上问题
                messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
                text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
                output_text,_ = messages2out(model,qwen_processor,inputs)
                if not str(s) in outputs:outputs[str(s)] = {}
                outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes]
                
    #没有提取出实体词
    else:
        messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: " + ques +"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
        attention,idx2word_dicts,img_start,img_end = messages2att(model,qwen_processor,inputs)  # Retrieve attention from model outputs
        results = from_img_and_att_get_cropbox(inputs,attention, idx2word_dicts, img_url, img_start, img_end,sig,thre)
        for s in sig:
            for t in thre:
                img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs = unpack_att_result(results[str(s)][str(t)])
                messages = [ {"role": "user","content": [],},]
                append_visual_inputs(messages[-1]["content"], ori_img_url, hide_highlight_imgs, highlight_imgs)
                #加上问题
                messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
                text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
                output_text,_ = messages2out(model,qwen_processor,inputs)
                if not str(s) in outputs:outputs[str(s)] = {}
                outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes]
    return outputs

def cycle_epoch_infer(gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre, batch_size=4,
                      enable_saaa=False, enable_acr=False, enable_pmgvv=False,
                      acr_min_area=0.001, acr_edge_margin=0.02,
                      pmgvv_expand_ratio=0.3,
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
    """
    核心推理函数。

    参数说明:
        enable_saaa: 旧 EGAF 模式（注意力图与专家注意力掩码 Rank-Based Fusion）
        enable_acr:  自适应置信度路由
        enable_pmgvv: 渐进式多粒度视觉验证
        enable_grace: GRACE 模式（新方案）
            - 使用相对注意力归一化（A_rel = A / A_noise）替代简单减法，压制背景噪声
            - SAM3 文本提示检测结果作为独立补充 bbox（不融合进注意力图）
            - Otsu 自适应阈值替代固定阈值
            - 最终 bounding_boxes = 注意力 bbox ∪ SAM3 bbox
        grace_sam3_url: SAM3 服务地址（model_service_v2.py）
        grace_max_sam3_per_entity: 每个实体最多保留的 SAM3 bbox 数量
        heatmap_save_dir: str | None，注意力热力图保存目录。
            None（默认）表示不保存；非 None 时每条样本每张图都会保存一张
            叠加了注意力热力图 + bbox 标注的 PNG 文件，命名规则：
            {sample_id}_img{img_idx}_s{sigma}_t{thresh}_agg_heatmap.png
    """
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    device = f"cuda:{gpu_id}"

    print(rank,len(dataset_part),device,f"batch_size={batch_size}")
    print(f"  创新点开关: EGAF={enable_saaa}(mode={egaf_fusion_mode},w={egaf_expert_weight}), ACR={enable_acr}, PMGVV={enable_pmgvv}")

    model_path = r"/path/to/ckpt/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    )

    qwen_processor = AutoProcessor.from_pretrained(model_path,use_fast=True,min_pixels=256*28*28,max_pixels=max_pixels*28*28)
    if qwen_processor.tokenizer.pad_token is None:
        qwen_processor.tokenizer.pad_token = qwen_processor.tokenizer.eos_token

    # ══════════════════════════════════════════════════════════════════════
    # 难度感知路由器（Threshold-Free Router）加载
    # ══════════════════════════════════════════════════════════════════════
    # 启用后：
    #   在第 1 步 direct-answer 阶段，额外收集首个生成 token 的词表 softmax
    #   → RouterV2 动态解析选项字母 + 全词表熵 + option_mass + logit_gap 等特征
    #   → Cost-Aware Conformal Safe-Skip 决策
    #   （或旧版 4 类 / 2 类 ThresholdFree / Conformal Router，向后兼容）
    # 若 trigger=False：直接用 direct-answer 作为最终结果，跳过 GRACE/HiDe 管道
    router = None
    router_kind = None                          # "v2_gbdt"/"v2_lr"/"conformal"/"binary"/"4class"
    option_token_ids = None                     # 仅 v1 路由器使用（ABCD）
    option_token_ids_all = None                 # A-Z 全字母 → token_id，v2 路由器使用
    if enable_router:
        try:
            # 路由器需要 direct-answer 的 first-token logits，因此强制 skip_ori=False
            if skip_ori:
                print(f"⚠️ enable_router=True 需要 direct-answer logits；强制 skip_ori=False")
                skip_ori = False
            # 加载路由器（自动识别 4 类 / 2 类）
            _rp = router_report_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "params", "Qwen2.5", "router_report.json",
            )
            # 通过查看 model_type 字段决定加载哪种路由器
            with open(_rp, "r") as _f:
                _meta = json.load(_f)
            _mtype = _meta.get("model_type", "multinomial_logreg_4class")
            # 把 tools/ 路径加进 sys.path 以便导入新版 Router
            _tools_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tools",
            )
            if _tools_dir not in sys.path:
                sys.path.insert(0, _tools_dir)
            if _mtype in ("gbdt", "lr"):
                # 推荐：RouterV2 (Cost-Aware Conformal Safe-Skip)
                from router_v2 import RouterV2
                router = RouterV2.load(_rp)
                router_kind = f"v2_{_mtype}"
                # 若用户指定了 router_alpha，按新 α 重算 s_floor
                # （需要 router report 里存有 mr_oof_scores_sorted）
                if router_alpha is not None:
                    try:
                        _old_sf = router.s_floor
                        _new_sf = router.set_alpha(float(router_alpha))
                        print(f"[router] override α = {router_alpha}  "
                              f"s_floor: {_old_sf:.4f} → {_new_sf:.4f}")
                    except Exception as _e:
                        print(f"⚠️ 无法按 router_alpha={router_alpha} 调节 s_floor: {_e}")
                        print(f"   保持训练时的 s_floor={router.s_floor:.4f} "
                              f"(α={router.alpha})")
            elif _mtype == "conformal_safe_skip_router":
                # 推荐路由器（保证 ori_wrong 高召回 + 自动 conformal 阈值）
                from router import Router
                router = Router.load(_rp)
                router_kind = "conformal"
            elif _mtype == "binary_logreg":
                # 旧版二分类保守路由器（向后兼容）
                from router import Router   # 已合并入新版
                router = Router.load(_rp)
                router_kind = "binary"
            else:
                # 4 类无阈值路由器（旧版）
                from difficulty_analysis.threshold_free_router import ThresholdFreeRouter
                router = ThresholdFreeRouter.load(_rp)
                router_kind = "4class"

            # 取 A/B/C/D 对应的 token id（v1 路由器 / 兼容）
            option_token_ids = {}
            for letter in "ABCD":
                ids = qwen_processor.tokenizer.encode(letter, add_special_tokens=False)
                option_token_ids[letter] = ids[0]
            # A-Z 全字母 token id（v2 路由器动态选项用）
            option_token_ids_all = {}
            for c in range(ord("A"), ord("Z") + 1):
                ch = chr(c)
                ids = qwen_processor.tokenizer.encode(ch, add_special_tokens=False)
                if ids:
                    option_token_ids_all[ch] = ids[0]
            print(f"[router] kind={router_kind}, loaded from {_rp}")
            print(f"[router] option token ids (ABCD): {option_token_ids}")
            if router_kind and router_kind.startswith("v2_"):
                print(f"[router] RouterV2 features: {router.features}")
                print(f"[router] decision rule: run_grace <=> "
                      f"s(x) >= s_floor = {router.s_floor:.4f}  "
                      f"(alpha={router.alpha:.3f})")
            elif router_kind == "conformal":
                print(f"[router] decision rule: run_grace ⟺ "
                      f"s(x) ≥ s_floor = {router.s_floor:.4f}  "
                      f"(α={router.alpha:.3f})")
            elif router_kind == "binary":
                print(f"[router] decision rule (legacy binary): "
                      f"s(x) ≥ s_floor = {router.s_floor:.4f}")
            else:
                print(f"[router] decision rule: run_grace = (w_c*P(C) > w_b*P(B)) "
                      f"with w_c={router.w_c}, w_b={router.w_b}")
        except Exception as _e:
            print(f"⚠️ 路由器加载失败，退回到全走 GRACE: {_e}")
            router = None
            router_kind = None

    num_samples = len(dataset_part)
    for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"GPU{gpu_id} batches"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = dataset_part[batch_start:batch_end]
        actual_batch_size = len(batch_samples)
        
        # ==================== 第1步：批量直接回答（ori结果）====================
        # skip_ori=True 时跳过此步，节省推理时间（不影响HiDe结果）
        direct_messages_list = []
        batch_img_urls = []
        batch_ori_img_urls = []
        batch_ques = []

        # Prompt 选择说明
        # -------------------------------------------------------------
        # v2 路由器用 "Answer:" direct prompt 采集训练特征。为保证推理特征
        # 分布与训练一致（避免边界样本决策漂移），v2 路由器激活时推理也用
        # 同一 prompt。
        #
        # 注意：
        #   (1) 训练期 extract_features_v2.py **必须** 使用与推理完全相同的
        #       forward 路径（model.generate(..., output_scores=True)）才能
        #       保证特征值一致 —— 这点已在 extract_features_v2.py 中修复。
        #   (2) 若 router 未启用或是 v1 conformal/binary，则保持原 tag prompt。
        use_v2_prompt = (router is not None and router_kind is not None
                         and router_kind.startswith("v2_"))

        for sample in batch_samples:
            img_url = [sample["image"]]
            ori_img_url = list(img_url)
            ques = sample["Text"]

            if use_v2_prompt:
                # v2 统一 prompt：首 token 直接是选项字母，与训练分布对齐
                instr = build_answer_instr_v2(ques)
            else:
                instr = (
                    "\nAnswer with the option's letter from the given choices letter. "
                    "The final and only output content must be enclosed within "
                    "`<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."
                )

            messages = [{"role": "user", "content": []}]
            for img in img_url:
                messages[-1]["content"].append({"type": "image", "image": img})
            messages[-1]["content"].append({"type": "text", "text": ques + instr})

            direct_messages_list.append(messages)
            batch_img_urls.append(img_url)
            batch_ori_img_urls.append(ori_img_url)
            batch_ques.append(ques)

        if skip_ori:
            direct_output_texts = [""] * actual_batch_size
            direct_entropies = [None] * actual_batch_size
            direct_topps = [None] * actual_batch_size
            direct_margins = [None] * actual_batch_size
            direct_v2_feats = [None] * actual_batch_size
        else:
            direct_v2_feats = [None] * actual_batch_size
            direct_entropies = [None] * actual_batch_size
            direct_topps = [None] * actual_batch_size
            direct_margins = [None] * actual_batch_size
            try:
                batched_inputs = batch_get_inputs(direct_messages_list, qwen_processor, model)

                # ⚠️ 为了与 4_21 版 GRACE 流程保持**GPU 状态/数值一致**，
                # direct-answer 的 ori 生成与 router 特征采集 **必须分开做两次 forward**：
                #   (1) 先用 **不带 output_scores** 的 generate 拿 ori 答案（与 4_21 完全等价）
                #   (2) 再用单独的 1-token forward 拿 first_logits 给 router
                # 如果把它们合并成一个 output_scores=True 的 forward，HuggingFace
                # 为存 scores tuple 会改变 GPU memory 分配顺序，在 bf16 + flash-attn
                # 下会让后续 GRACE forward 的浮点数值出现微小扰动，导致边界样本
                # （如 vstar id=4, id=54 这种多个选项概率接近的题）答案翻边。

                # ── Step 1: 拿 ori 答案（与 4_21 完全一致，不带 output_scores） ──
                direct_output_texts = batch_messages2out(model, qwen_processor, batched_inputs)
                torch.cuda.empty_cache()

                # ── Step 2: 若启用 router，再单独做一次 1-token forward 采特征 ──
                if router is not None and option_token_ids is not None:
                    # 重新 build 一次 inputs（batch_messages2out 消耗了原 inputs）
                    batched_inputs_r = batch_get_inputs(direct_messages_list, qwen_processor, model)
                    with torch.no_grad():
                        gen_out_r = model.generate(
                            **batched_inputs_r, use_cache=True, max_new_tokens=1,
                            do_sample=False,
                            return_dict_in_generate=True, output_scores=True,
                        )
                    if gen_out_r.scores is not None and len(gen_out_r.scores) > 0:
                        _first_logits_batch = gen_out_r.scores[0].float().cpu().numpy()   # [B, V]
                        import numpy as _np
                        # 4 选项 softmax（v1 路由器用）
                        letters = ["A", "B", "C", "D"]
                        opt_idx = _np.array([option_token_ids[L] for L in letters], dtype=_np.int64)
                        opt_log = _first_logits_batch[:, opt_idx]
                        opt_log = opt_log - opt_log.max(axis=1, keepdims=True)
                        opt_probs = _np.exp(opt_log)
                        opt_probs = opt_probs / opt_probs.sum(axis=1, keepdims=True)
                        entropies = -_np.sum(opt_probs * _np.log(opt_probs + 1e-12), axis=1)
                        direct_entropies = [float(entropies[i]) for i in range(opt_probs.shape[0])]
                        for i in range(opt_probs.shape[0]):
                            sp = _np.sort(opt_probs[i])[::-1]
                            direct_topps[i] = float(sp[0])
                            direct_margins[i] = float(sp[0] - sp[1])
                        while len(direct_topps) < actual_batch_size:
                            direct_topps.append(None); direct_margins.append(None)
                        # v2 完整特征
                        if use_v2_prompt:
                            for _i in range(actual_batch_size):
                                if _i < _first_logits_batch.shape[0]:
                                    direct_v2_feats[_i] = compute_router_v2_features(
                                        _first_logits_batch[_i], batch_ques[_i],
                                        option_token_ids_all,
                                    )
                    del gen_out_r, batched_inputs_r
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ Batch direct answer failed (batch_size={actual_batch_size}), falling back to single: {e}")
                direct_output_texts = []
                direct_entropies = []
                direct_topps = []
                direct_margins = []
                direct_v2_feats = []
                for _i2, msgs in enumerate(direct_messages_list):
                    text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(msgs,qwen_processor,model)
                    if router is not None and option_token_ids is not None:
                        out_tup = messages2out_with_logits(
                            model, qwen_processor, inputs,
                            option_token_ids=option_token_ids,
                        )
                        output_text = out_tup[0]
                        ent = out_tup[4]
                        op_probs = out_tup[3]
                        first_logits_full = out_tup[2]   # [vocab]
                        if op_probs is not None:
                            import numpy as _np
                            _sp = _np.sort(op_probs)[::-1]
                            tp = float(_sp[0]); mg = float(_sp[0] - _sp[1])
                        else:
                            tp = mg = None
                        # v2 特征（若启用 v2 prompt）
                        if use_v2_prompt and first_logits_full is not None:
                            v2f = compute_router_v2_features(
                                first_logits_full, batch_ques[_i2],
                                option_token_ids_all,
                            )
                        else:
                            v2f = None
                    else:
                        output_text, _ = messages2out(model, qwen_processor, inputs)
                        ent = tp = mg = None
                        v2f = None
                    direct_output_texts.append(output_text[0])
                    direct_entropies.append(ent)
                    direct_topps.append(tp)
                    direct_margins.append(mg)
                    direct_v2_feats.append(v2f)
            torch.cuda.empty_cache()
        
        # ==================== 第2步：批量实体提取 ====================
        prompt_ques_template = '''
You are a highly precise language analysis engine. Your sole function is to extract entities (e.g., objects, people) from a user's question, and deconstruct them into a canonical, attribute-based format by strictly following a set of rules and a thinking process.

### Thinking Process
Before generating the final output, you must internally follow these steps in order:

1.  **Identify Core Entities**: Read the entire question and identify all key noun phrases. For example, "the green surfboard," "the purple umbrella."
2.  **Deconstruct Attributes for Each Entity Individually**: **This is the critical step. Before considering the relationship between entities, look at each entity in isolation and apply Rules 2, 3, and 4 to fully deconstruct its attributes.**
    *   For instance, first process "the green surfboard" using Rule 2 to get `surfboard with green color`.
    *   Then, process "the purple umbrella" using Rule 2 to get `umbrella with purple color`.
3.  **Handle Relationships Between Entities**: After all entities have been individually deconstructed, check for spatial or logical relationships between them (Rule 5). If a relationship exists, you will list the **already deconstructed** entities as separate items.
4.  **Assemble and Normalize**:
    *   Gather all the canonical entity strings you have transformed.
    *   Convert all text to lowercase.
    *   Join all entities into a single line, separated by a comma and a space (", ").
5.  **Final Formatting**: Enclose the resulting single-line string within the `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.

### Extraction Rules

**Rule 1: Simple Entities**
If a noun is not described by any modifiers, extract the noun itself.
*   *Example*: "the scooter" becomes `scooter`.

**Rule 2: Adjective Attribute Deconstruction**
If an entity is modified by one or more adjectives, the format must be `noun with [property] [type]`. Chain multiple properties consecutively.
*   **Format**: `noun with [property1] [type1] with [property2] [type2]`
*   **Common Type Mappings**:
    *   Colors (red, blue, black, silver) -> `color`
    *   Sizes (large, small, big) -> `size`
    *   Materials (wooden, metal, plastic) -> `material`
*   *Example*: "the large blue truck" becomes `truck with large size with blue color`.

**Rule 3: Possessive Inversion**
Convert all possessive forms (e.g., `X's Y`) uniformly into the `Y of X` format.
*   *Example*: "the woman's handbag" becomes `handbag of woman`.

**Rule 4: Attributive Prepositional Phrases**
If a prepositional phrase describes a component of an entity (e.g., "in a shirt"), preserve the structure and recursively apply the rules to the entity within the phrase.
*   *Example*: "the man in the green shirt" becomes `man in a shirt with green color`.

**Rule 5: Relational Prepositional Phrases**
If a prepositional phrase describes a relationship between two separate entities (e.g., `next to`, `on the left of`), **extract the entities as separate items, but only after each entity has been fully deconstructed according to the other rules.** Do not include the relational words in the output.
*   *Example 1 (Simple)*: "the dog on the left side of the scooter" becomes `dog, scooter`.
*   **Example 2 (With Attributes)**: **"Is the green surfboard on the left side of the purple umbrella?" becomes `surfboard with green color, umbrella with purple color`.** (Note: The surfboard and umbrella are first deconstructed individually, then listed as separate entities).

**Rule 6: Compound Nouns**
Recognized compound nouns should be treated as a single entity.
*   *Example*: "the traffic light" becomes `traffic light`.

### Output Format Rules
1.  **Sole Output**: The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags.
2.  **Single-Line Format**: The output must be a single continuous line of text.
3.  **Delimiter**: Multiple entities must be separated by a comma followed by a space (", ").
4.  **Lowercase**: All output characters must be in lowercase.
5.  **Content Exclusion**: The final entity string must not include articles (a, an, the), question words (what, which, is), or purely relational words (side, next to, of).

Now, following all the rules above, extract the entities from the question below:
{input_text}
'''
        
        entity_messages_list = []
        for ques in batch_ques:
            prompt_messages = [{"role": "user","content": [{"type": "text", "text": prompt_ques_template.format(input_text=ques)}]}]
            entity_messages_list.append(prompt_messages)
        
        try:
            entity_batched_inputs = batch_get_inputs(entity_messages_list, qwen_processor, model)
            entity_output_texts = batch_messages2out(model, qwen_processor, entity_batched_inputs)
        except Exception as e:
            print(f"⚠️ Batch entity extraction failed, falling back to single: {e}")
            entity_output_texts = []
            for msgs in entity_messages_list:
                text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(msgs,qwen_processor,model)
                output_text,_ = messages2out(model,qwen_processor,inputs)
                entity_output_texts.append(output_text[0])
        torch.cuda.empty_cache()
        
        # ==================== 第3步：逐条 attention 分析 + 最终回答 ====================
        for idx in range(actual_batch_size):
            sample = batch_samples[idx]
            results = sample
            img_url = batch_img_urls[idx]
            ori_img_url = batch_ori_img_urls[idx]
            ques = batch_ques[idx]
            
            results["answer"] = {}
            if not skip_ori:
                results["answer"]["ori"] = direct_output_texts[idx]
            results["bounding_box"] = {}
            results["prompt_text"] = {}
            results["innovation_info"] = {}  # 记录创新点相关信息

            # 样本唯一 ID，用于热力图文件命名
            _sample_id = str(sample.get("id", f"rank{rank}_b{batch_start}_i{idx}"))
            # 过滤文件名中的非法字符
            _sample_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in _sample_id)

            # ══════════════════════════════════════════════════════════════
            # 难度感知路由：决定本样本是否走 GRACE
            #   v2 路由器：基于 RouterV2 Cost-Aware Conformal Safe-Skip
            #   v1 路由器：基于 Router（旧版 5 维 LogReg Conformal）/ 4 类 Threshold-Free
            # ══════════════════════════════════════════════════════════════
            should_run_grace_pipeline = True   # 默认走 GRACE（即不启用路由器时）

            # v2 分支：优先使用 v2 特征调用 RouterV2.decide
            if (router is not None and router_kind is not None
                    and router_kind.startswith("v2_")
                    and idx < len(direct_v2_feats) and direct_v2_feats[idx] is not None):
                try:
                    v2f = direct_v2_feats[idx]
                    # 把所有计算出的特征都传给 router；RouterV2 根据自身 features
                    # 列表按需取值（多余的会被忽略）。这样 router 升级换特征时
                    # 推理端无需修改。
                    trigger, p_info = router.decide(
                        question_text=ques,
                        **v2f,
                    )
                    should_run_grace_pipeline = bool(trigger)
                    # 记录 v2 特征到 innovation_info（便于事后分析）
                    for _k, _v in v2f.items():
                        results["innovation_info"][f"router_{_k}"] = float(_v)
                    if isinstance(p_info, dict):
                        for k, v in p_info.items():
                            results["innovation_info"][f"router_{k}"] = (
                                float(v) if isinstance(v, (int, float)) else v
                            )
                    results["innovation_info"]["router_trigger_grace"] = bool(trigger)
                    results["innovation_info"]["router_kind"] = router_kind
                    results["innovation_info"]["router_features"] = list(router.features)
                except Exception as _e:
                    print(f"  [router v2] 决策失败，回退到全走 GRACE: {_e}")
                    should_run_grace_pipeline = True
            # v1 分支：旧 Router（ABCD 4 选项 entropy/topp/margin）
            elif router is not None and direct_entropies[idx] is not None:
                try:
                    # 新版 Router 需要 5 个特征：entropy/topp/margin + 2 个关键词
                    trigger, p_info = router.decide(
                        answer_entropy=direct_entropies[idx],
                        answer_topp=direct_topps[idx] if idx < len(direct_topps) else None,
                        answer_margin=direct_margins[idx] if idx < len(direct_margins) else None,
                        question_text=ques,
                    )
                    should_run_grace_pipeline = bool(trigger)
                    results["innovation_info"]["router_answer_entropy"] = float(direct_entropies[idx])
                    if direct_topps[idx] is not None:
                        results["innovation_info"]["router_answer_topp"] = float(direct_topps[idx])
                    if direct_margins[idx] is not None:
                        results["innovation_info"]["router_answer_margin"] = float(direct_margins[idx])
                    # p_info 是 dict（4 类） 或包含 score/s_floor 的 dict（新版）
                    if isinstance(p_info, dict):
                        for k, v in p_info.items():
                            results["innovation_info"][f"router_{k}"] = (
                                float(v) if isinstance(v, (int, float)) else v
                            )
                    results["innovation_info"]["router_trigger_grace"] = bool(trigger)
                except Exception as _e:
                    print(f"  [router] 决策失败，回退到全走 GRACE: {_e}")
                    should_run_grace_pipeline = True

            # 若路由器判定不需要 GRACE，直接把 direct answer 作为最终结果
            if router is not None and not should_run_grace_pipeline:
                # 用 direct-answer 填写 HiDe 标准键（与 GRACE 路径保持 schema 一致）
                for s in sig:
                    for t in thre:
                        results["answer"][f"HiDe_s{s}_t{t}"] = direct_output_texts[idx]
                        results["bounding_box"][f"HiDe_s{s}_t{t}"] = {}
                results["prompt_text"][f"HiDe"] = "[router-skipped-grace]"
                results.pop("image", None)
                serialize_dict(results, savedir)
                torch.cuda.empty_cache()
                continue   # 本样本完成，处理下一条

            entity_text = entity_output_texts[idx]
            answer_out = entity_text.split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
            
            messages = [{"role": "user", "content": []}]
            for img in img_url:
                messages[-1]["content"].append({"type": "image", "image": img})
            
            outputs = {}
            if answer_out:
                entity_list = [e.strip() for e in answer_out.split(',') if e.strip()]

                # ── 初始化公共变量 ──────────────────────────────────────────────
                expert_bboxes_per_img = None
                expert_results_raw = {}
                entity_token_indices = None
                entity_token_map = {}
                expert_reliability = 0.0
                sam3_supplement_bboxes_per_img = None   # GRACE 专用
                sam3_entity_labels_per_img = None           # GRACE 专用

                # ══════════════════════════════════════════════════════════════
                # GRACE 模式（新方案）:
                #   A. SAM3 文本提示 → 独立补充 bbox（不融合进注意力图）
                #   B. Search Prompt → TAD 逐 token 注意力图 + Otsu 二值化
                #   C. 最终 bbox = 注意力 bbox ∪ SAM3 bbox
                # ══════════════════════════════════════════════════════════════
                if enable_grace:
                    # A. SAM3 文本提示检测（与 Search Prompt 关键词相同）
                    sam3_results_raw = call_grounding_expert(
                        img_url[0], entity_list,
                        expert_url=grace_sam3_url
                    )
                    all_sam3_bboxes, all_sam3_labels = get_sam3_supplement_bboxes(
                        sam3_results_raw, max_per_entity=grace_max_sam3_per_entity
                    )
                    if all_sam3_bboxes:
                        sam3_supplement_bboxes_per_img = {0: all_sam3_bboxes}
                        sam3_entity_labels_per_img = {0: all_sam3_labels}
                    results["innovation_info"]["grace_sam3_total"] = len(all_sam3_bboxes)
                    results["innovation_info"]["grace_sam3_per_entity"] = {
                        k: len(v) for k, v in sam3_results_raw.items()
                    }

                # ══════════════════════════════════════════════════════════════
                # 旧 EGAF 模式（保持向后兼容）
                # ══════════════════════════════════════════════════════════════
                elif enable_saaa:
                    expert_results_raw = call_grounding_expert(
                        img_url[0], entity_list,
                        expert_url=egaf_expert_url
                    )
                    all_expert_bboxes = []
                    for ent, boxes in expert_results_raw.items():
                        all_expert_bboxes.extend(boxes)
                    if all_expert_bboxes:
                        expert_bboxes_per_img = {0: all_expert_bboxes}
                    expert_reliability = estimate_expert_reliability(expert_results_raw, entity_list)
                    results["innovation_info"]["egaf_expert_results"] = {
                        k: len(v) for k, v in expert_results_raw.items()
                    }
                    results["innovation_info"]["egaf_expert_total"] = len(all_expert_bboxes)
                    results["innovation_info"]["egaf_reliability"] = round(expert_reliability, 3)

                # B. 构建 Search Prompt 并获取注意力图
                # GRACE / EGAF / TAD 均使用 Search Prompt 方案提取注意力；
                # GRACE 通过 process_grace（TAD 逐 token + Otsu 自适应阈值）
                # 在后处理阶段提升信噪比，而非在 token 遍历阶段做实体过滤。
                messages[-1]["content"].append(
                    {"type": "text", "text": "Search the following entities in the images: " + answer_out}
                )
                text, image_inputs, video_inputs, inputs, video_kwargs = get_inputs(
                    messages, qwen_processor, model
                )

                attention, idx2word_dicts, img_start, img_end = messages2att(
                    model, qwen_processor, inputs
                )

                # C. 实体 token 定位（仅 EGAF/SAAA 需要，GRACE 已退回 TAD 逐 token 遍历）
                if enable_saaa:
                    entity_token_indices, entity_token_map = find_entity_token_indices(
                        answer_out, idx2word_dicts, inputs, img_end,
                        min_token_ratio=0.3
                    )
                    results["innovation_info"]["entity_tokens"] = {
                        k: len(v) for k, v in entity_token_map.items()
                    }
                    n_total = len(inputs['input_ids'][0]) - img_end[-1] - 1
                    n_entity = len(entity_token_indices)
                    results["innovation_info"]["entity_token_speedup"] = f"{n_entity}/{n_total} tokens"

                # D. 注意力图 → bbox（含 SAM3 补充框插入）
                att_results = from_img_and_att_get_cropbox(
                    inputs, attention, idx2word_dicts, img_url, img_start, img_end, sig, thre,
                    enable_saaa=enable_saaa and not enable_grace,
                    enable_grace=enable_grace,
                    use_otsu=False,
                    entity_text=answer_out,
                    expert_bboxes_per_img=expert_bboxes_per_img,
                    sam3_supplement_bboxes_per_img=sam3_supplement_bboxes_per_img,
                    sam3_entity_labels_per_img=sam3_entity_labels_per_img,
                    egaf_fusion_mode=egaf_fusion_mode,
                    egaf_expert_weight=egaf_expert_weight,
                    entity_token_indices=entity_token_indices,
                    entity_token_map=entity_token_map,
                    expert_reliability=expert_reliability,
                    heatmap_save_dir=heatmap_save_dir,
                    sample_id=_sample_id,
                )

                for s in sig:
                    for t in thre:
                        img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs = unpack_att_result(att_results[str(s)][str(t)])

                        # 旧 EGAF 专家补全（仅 EGAF 模式，ρ > 0 时执行）
                        if enable_saaa and not enable_grace and expert_results_raw and expert_reliability > 0:
                            tad_all_bboxes = []
                            for imgidx in bounding_boxes:
                                tad_all_bboxes.extend(bounding_boxes[imgidx])
                            exclusive_bboxes, exclusive_info = compute_expert_exclusive_bboxes(
                                expert_results_raw, tad_all_bboxes,
                                iou_threshold=0.1, max_exclusive_per_entity=2
                            )
                            if exclusive_bboxes:
                                if 0 not in bounding_boxes:
                                    bounding_boxes[0] = []
                                bounding_boxes[0].extend(exclusive_bboxes)
                                results["innovation_info"][f"egaf_exclusive_s{s}_t{t}"] = exclusive_info
                                results["innovation_info"][f"egaf_exclusive_count_s{s}_t{t}"] = len(exclusive_bboxes)
                                new_highlight_imgs = []
                                for imgidx in bounding_boxes:
                                    new_img, new_bboxs, _ = compact_and_center_with_relative_pos(
                                        imgidx, len(img_url), img_url[imgidx], bounding_boxes[imgidx]
                                    )
                                    if new_img:
                                        bounding_boxes[imgidx] = new_bboxs
                                        new_highlight_imgs.extend(new_img)
                                if new_highlight_imgs:
                                    highlight_imgs = new_highlight_imgs
                        
                        # ====== 创新点2: ACR - 噪声bbox过滤 ======
                        if enable_acr:
                            for imgidx in bounding_boxes:
                                original_count = len(bounding_boxes[imgidx])
                                bounding_boxes[imgidx] = filter_noise_bboxes(
                                    bounding_boxes[imgidx], 
                                    min_area_ratio=acr_min_area,
                                    edge_margin=acr_edge_margin
                                )
                                filtered_count = len(bounding_boxes[imgidx])
                                results["innovation_info"][f"acr_bbox_filter_s{s}_t{t}"] = \
                                    f"{original_count}->{filtered_count}"
                        
                        # ====== 创新点3: PMGVV - 视觉验证 ======
                        verification_passed = True
                        if enable_pmgvv and highlight_imgs:
                            verify_prompt = build_verification_prompt(answer_out, has_crop=True)
                            verify_messages = [{"role": "user", "content": []}]
                            for img in ori_img_url:
                                verify_messages[-1]["content"].append({"type": "image", "image": img})
                            for h_img in highlight_imgs:
                                verify_messages[-1]["content"].append({"type": "image", "image": h_img})
                            verify_messages[-1]["content"].append({"type": "text", "text": verify_prompt})
                            
                            v_text,_,_,v_inputs,_ = get_inputs(verify_messages, qwen_processor, model)
                            v_output,_ = messages2out(model, qwen_processor, v_inputs)
                            verify_answer = v_output[0].strip().lower()
                            
                            if 'no' in verify_answer and 'yes' not in verify_answer:
                                verification_passed = False
                                results["innovation_info"][f"pmgvv_verify_s{s}_t{t}"] = "failed_expanding"
                                # 扩展bbox区域
                                for imgidx in bounding_boxes:
                                    expanded = [expand_bbox(b, pmgvv_expand_ratio) for b in bounding_boxes[imgidx]]
                                    bounding_boxes[imgidx] = expanded
                                # 重新生成裁剪图（使用扩展后的bbox）
                                new_highlight_imgs = []
                                for imgidx in bounding_boxes:
                                    new_img, new_bboxs, _ = compact_and_center_with_relative_pos(
                                        imgidx, len(img_url), img_url[imgidx], bounding_boxes[imgidx]
                                    )
                                    if new_img:
                                        bounding_boxes[imgidx] = new_bboxs
                                        new_highlight_imgs.extend(new_img)
                                if new_highlight_imgs:
                                    highlight_imgs = new_highlight_imgs
                            else:
                                results["innovation_info"][f"pmgvv_verify_s{s}_t{t}"] = "passed"
                            torch.cuda.empty_cache()
                        
                        # ── Requirement 2: 二次 SAM3 验证（在注意力 bbox 内裁剪搜索）────
                        # 新方案（overlay 流程）：
                        #   1. 二次 SAM3 返回的新 bbox 不污染 bounding_boxes
                        #   2. 把首次 SAM3 bbox + 二次 SAM3 新 bbox 一起构造成
                        #      overlay_bboxes_per_img，传给 LPD 画彩色边框
                        #   3. 若两次 SAM3 都无结果，overlay 为空 → 与 HiDe LPD 等价
                        if enable_grace and entity_list and highlight_imgs:
                            try:
                                _sec_img_url, found_secondary, new_entries_per_img = secondary_sam3_check_on_att_bboxes(
                                    img_url, bounding_boxes, entity_list,
                                    sam3_url=grace_sam3_url,
                                    max_per_entity=2,
                                    max_att_bboxes=4,
                                    max_new_bboxes=3,
                                    existing_sam3_bboxes_per_img=sam3_supplement_bboxes_per_img,
                                    draw_on_image=False,
                                )

                                if found_secondary and new_entries_per_img:
                                    # 统一颜色映射：首次 + 二次 SAM3 所有实体共用同一个映射
                                    _first_labels = []
                                    if sam3_entity_labels_per_img:
                                        for _lbs in sam3_entity_labels_per_img.values():
                                            _first_labels.extend(list(_lbs) if _lbs else [])
                                    _sec_all_labels = [
                                        lb for entries in new_entries_per_img.values()
                                        for _, lb in entries
                                    ]
                                    label2color_full, _ = assign_entity_colors(
                                        list(_first_labels) + list(_sec_all_labels)
                                    )

                                    # 构建合并的 overlay_bboxes_per_img
                                    overlay_merged = {}
                                    # 首次 SAM3：复制进 overlay（同时也要确保对应 bounding_boxes 包含该区域；
                                    # 在 from_img_and_att_get_cropbox 中已处理过，这里不需要再追加）
                                    if sam3_supplement_bboxes_per_img:
                                        for _imgidx, _bxs in sam3_supplement_bboxes_per_img.items():
                                            _lbs = (
                                                sam3_entity_labels_per_img.get(_imgidx, [])
                                                if sam3_entity_labels_per_img else []
                                            )
                                            while len(_lbs) < len(_bxs):
                                                _lbs.append("")
                                            for _bx, _lb in zip(_bxs, _lbs):
                                                _color = label2color_full.get(
                                                    (_lb or "").strip(),
                                                    _LEGEND_PALETTE[0]
                                                )
                                                overlay_merged.setdefault(_imgidx, []).append({
                                                    "bbox_norm": list(_bx),
                                                    "color": _color,
                                                    "label": _lb or "",
                                                })

                                    # 二次 SAM3：新 bbox 需加入 bounding_boxes 保证裁剪包含该区域
                                    for _imgidx, entries in new_entries_per_img.items():
                                        if _imgidx not in bounding_boxes:
                                            bounding_boxes[_imgidx] = []
                                        for _bx, _lb in entries:
                                            bounding_boxes[_imgidx].append(list(_bx))
                                            _color = label2color_full.get(
                                                (_lb or "").strip(), _LEGEND_PALETTE[1]
                                            )
                                            overlay_merged.setdefault(_imgidx, []).append({
                                                "bbox_norm": list(_bx),
                                                "color": _color,
                                                "label": _lb or "",
                                            })

                                    # 重新跑 LPD + 追加图注
                                    new_hl = []
                                    for _imgidx in bounding_boxes:
                                        _src = img_url[_imgidx] if _imgidx < len(img_url) else img_url[0]
                                        _overlays = overlay_merged.get(_imgidx) or None
                                        _img, _bboxs, _used_cl = compact_and_center_with_relative_pos(
                                            _imgidx, len(img_url), _src, bounding_boxes[_imgidx],
                                            overlay_bboxes=_overlays,
                                            draw_color_border=True,
                                        )
                                        if _img:
                                            bounding_boxes[_imgidx] = _bboxs
                                            _seen = set()
                                            _merged = []
                                            for cl in list(_used_cl):
                                                key = (tuple(cl[0]), cl[1])
                                                if key not in _seen and cl[1]:
                                                    _seen.add(key)
                                                    _merged.append(cl)
                                            for _im in _img:
                                                if _merged:
                                                    _im = add_legend_to_lpd_image(
                                                        _im, _merged, position="bottom"
                                                    )
                                                new_hl.append(_im)
                                    if new_hl:
                                        highlight_imgs = new_hl

                                results["innovation_info"][f"secondary_sam3_s{s}_t{t}"] = found_secondary
                            except Exception as _e:
                                print(f"[secondary_sam3] 跳过: {_e}")

                        # ── Requirement 3: 保存 LPD 输出图（额外保留 HiDe-only 与最终 GRACE 两套）──
                        if heatmap_save_dir:
                            import os as _os
                            _os.makedirs(heatmap_save_dir, exist_ok=True)
                            if hide_highlight_imgs and hide_highlight_imgs != highlight_imgs:
                                for _hi, _h_img in enumerate(hide_highlight_imgs):
                                    _fname = f"{_sample_id}_s{s}_t{t}_lpd_hide_out_{_hi}.png"
                                    _fpath = _os.path.join(heatmap_save_dir, _fname)
                                    try:
                                        _b64d = _h_img.split(",")[1] if "," in _h_img else _h_img
                                        with open(_fpath, "wb") as _f:
                                            _f.write(base64.b64decode(_b64d))
                                    except Exception as _e:
                                        print(f"[save_lpd] 保存失败 {_fname}: {_e}")
                            if highlight_imgs:
                                for _hi, _h_img in enumerate(highlight_imgs):
                                    _fname = f"{_sample_id}_s{s}_t{t}_lpd_out_{_hi}.png"
                                    _fpath = _os.path.join(heatmap_save_dir, _fname)
                                    try:
                                        _b64d = _h_img.split(",")[1] if "," in _h_img else _h_img
                                        with open(_fpath, "wb") as _f:
                                            _f.write(base64.b64decode(_b64d))
                                    except Exception as _e:
                                        print(f"[save_lpd] 保存失败 {_fname}: {_e}")

                        # 最终推理
                        final_messages = [{"role": "user","content": []}]
                        append_visual_inputs(final_messages[-1]["content"], ori_img_url, hide_highlight_imgs, highlight_imgs)
                        # ⚠️ GRACE 最终推理保持原始 prompt（<FINAL_OUTPUT> 模板），
                        # 不使用 v2 "Answer:" 简洁 prompt。原因：
                        #   1) GRACE 路径带多张图（原图 + HiDe LPD 图 + 覆盖 SAM3 bbox 的图），
                        #      模型需要 tag 模板的推理空间（尤其空间关系题），简洁 prompt 会损失质量
                        #   2) v2 prompt 仅用于 direct-answer 阶段采集 router 特征，
                        #      被 skip 到 ori 的样本才用 v2 prompt 的输出；被 trigger 走 GRACE 的样本
                        #      应使用原始 pipeline 的 prompt 链路
                        final_messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
                        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(final_messages,qwen_processor,model)
                        output_text,_ = messages2out(model,qwen_processor,inputs)
                        if not str(s) in outputs:outputs[str(s)] = {}
                        outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,final_messages,words_lines,img_merged_boxes,bounding_boxes]
            else:
                # 未提取到实体词时，用原始问题作为 Search Prompt（GRACE / TAD 统一）
                messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: " + ques +"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
                text, image_inputs, video_inputs, inputs, video_kwargs = get_inputs(messages, qwen_processor, model)
                attention, idx2word_dicts, img_start, img_end = messages2att(model, qwen_processor, inputs)
                att_results = from_img_and_att_get_cropbox(
                    inputs, attention, idx2word_dicts, img_url, img_start, img_end, sig, thre,
                    enable_saaa=enable_saaa and not enable_grace,
                    enable_grace=enable_grace,
                    use_otsu=False,
                    entity_text=ques,
                    expert_bboxes_per_img=None,
                    sam3_supplement_bboxes_per_img=None,
                    egaf_fusion_mode=egaf_fusion_mode,
                    egaf_expert_weight=egaf_expert_weight,
                    heatmap_save_dir=heatmap_save_dir,
                    sample_id=_sample_id,
                )
                for s in sig:
                    for t in thre:
                        img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs = unpack_att_result(att_results[str(s)][str(t)])
                        
                        if enable_acr:
                            for imgidx in bounding_boxes:
                                bounding_boxes[imgidx] = filter_noise_bboxes(
                                    bounding_boxes[imgidx],
                                    min_area_ratio=acr_min_area,
                                    edge_margin=acr_edge_margin
                                )
                        
                        final_messages = [{"role": "user","content": []}]
                        append_visual_inputs(final_messages[-1]["content"], ori_img_url, hide_highlight_imgs, highlight_imgs)
                        # GRACE 最终推理：保持原始 <FINAL_OUTPUT> prompt（同上原因）
                        final_messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
                        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(final_messages,qwen_processor,model)
                        output_text,_ = messages2out(model,qwen_processor,inputs)
                        if not str(s) in outputs:outputs[str(s)] = {}
                        outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,final_messages,words_lines,img_merged_boxes,bounding_boxes]
            
            for s in sig:
                for t in thre:
                    if str(s) in outputs and str(t) in outputs[str(s)]:
                        prompt_output_text,output_text,crop_list,highlight_imgs,final_messages,words_lines,img_merged_boxes,bounding_boxes = outputs[str(s)][str(t)]
                        
                        hide_answer_text = output_text[0]
                        
                        # ====== 创新点2: ACR - 置信度路由 ======
                        if enable_acr:
                            ori_letter = extract_answer_letter(direct_output_texts[idx])
                            hide_letter = extract_answer_letter(hide_answer_text)
                            best_text, best_letter, route = confidence_routing(
                                direct_output_texts[idx], hide_answer_text,
                                ori_letter, hide_letter
                            )
                            results["answer"][f"HiDe_s{s}_t{t}"] = best_text
                            results["answer"][f"HiDe_s{s}_t{t}_raw"] = hide_answer_text
                            results["innovation_info"][f"acr_route_s{s}_t{t}"] = route
                            results["innovation_info"][f"acr_ori_letter"] = ori_letter
                            results["innovation_info"][f"acr_hide_letter"] = hide_letter
                        else:
                            results["answer"][f"HiDe_s{s}_t{t}"] = hide_answer_text
                        
                        results["prompt_text"][f"HiDe"] = prompt_output_text[0]
                        results["bounding_box"][f"HiDe_s{s}_t{t}"] = bounding_boxes
            results.pop("image")
            serialize_dict(results,savedir)
            torch.cuda.empty_cache()
        
        print(f"GPU{gpu_id}: batch {batch_start//batch_size+1} done, saved to {savedir}")
    del model
    torch.cuda.empty_cache()
