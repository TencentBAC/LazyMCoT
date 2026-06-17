import os
from transformers import AutoTokenizer, AutoProcessor
from modeling_qwen3_vl_re_infer import Qwen3VLForConditionalGeneration
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
                     _LEGEND_PALETTE)
import shutil
import cv2
import re
import math


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def unpack_att_result(result_item):
    if len(result_item) >= 6:
        img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs = result_item[:6]
    else:
        img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes = result_item
        hide_highlight_imgs = []
    return img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs


def append_visual_inputs(content, ori_img_url, hide_highlight_imgs=None, highlight_imgs=None):
    """
     content VLM 
        1) ori_img_url
        2) HiDe hide_highlight_imgs LPD SAM3 
        3) bbox HiDe highlight_imgs SAM3 + 

     HiDe GRACE SAM3 overlay 
     2 
    """
    for img in ori_img_url:
        content.append({"type": "image", "image": img})
    if hide_highlight_imgs and hide_highlight_imgs != highlight_imgs:
        for h_img in hide_highlight_imgs:
            content.append({"type": "image", "image": h_img})
    if highlight_imgs:
        for h_img in highlight_imgs:
            content.append({"type": "image", "image": h_img})


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
_OPT_PAT_V2 = re.compile(r"(?:^|\n|\s)\(?([A-Z])[\)\.\:]", re.MULTILINE)


def _parse_options_v2(question: str):
    """ A ABCD"""
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
    v2 "" direct-answer GRACE 

    
      1. "Answer:" token = features_train_v2.jsonl
          router 
      2. <FINAL_OUTPUT> tag
          token == 
      3. ori GRACE prompt GRACE 
         prompt ""regret
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
     first-token vocab logits1D np.ndarray [V]+ question + A-Z token map
     RouterV2 7 router pick

    
        answer_topp top-1 
        answer_margin               top1 - top2
        answer_entropy K-way (nat)
        answer_entropy_norm K-way / log(K) ★ 
        option_mass token 
        logit_gap_opt_nonopt max logit - max logit
        vocab_full_entropy_norm token 
    """
    if first_logits is None:
        return None
    try:
        import numpy as np
        logits = np.asarray(first_logits, dtype=np.float64)
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
                messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
                text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
                output_text,_ = messages2out(model,qwen_processor,inputs)
                if not str(s) in outputs:outputs[str(s)] = {}
                outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,messages,words_lines,img_merged_boxes,bounding_boxes]
                
    else:
        messages[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: " + ques +"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
        attention,idx2word_dicts,img_start,img_end = messages2att(model,qwen_processor,inputs)  # Retrieve attention from model outputs
        results = from_img_and_att_get_cropbox(inputs,qwen_processor,attention, idx2word_dicts, img_url, img_start, img_end,sig,thre)
        for s in sig:
            for t in thre:
                img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs = unpack_att_result(results[str(s)][str(t)])
                messages = [ {"role": "user","content": [],},]
                append_visual_inputs(messages[-1]["content"], ori_img_url, hide_highlight_imgs, highlight_imgs)
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
    

    :
        enable_saaa: EGAF Rank-Based Fusion
        enable_acr:  
        enable_pmgvv: 
        enable_grace: GRACE 
            - A_rel = A / A_noise
            - SAM3 bbox
            - Otsu 
            - bounding_boxes = bbox ∪ SAM3 bbox
        grace_sam3_url: SAM3 model_service_v2.py
        grace_max_sam3_per_entity: SAM3 bbox 
        enable_router: RouterV2 / Conformal / 4 
        router_report_path: router router_v2_gbdt.json
        router_alpha: router α α s_floor
        heatmap_save_dir: str | None / LPD 
            None None 
    """
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    device = f"cuda:{gpu_id}"

    print(rank,len(dataset_part),device,f"batch_size={batch_size}")
    print(f" : EGAF={enable_saaa}(mode={egaf_fusion_mode},w={egaf_expert_weight}), ACR={enable_acr}, PMGVV={enable_pmgvv}, GRACE={enable_grace}, ROUTER={enable_router}")

    model_path = r"/path/to/ckpt/Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    )

    qwen_processor = AutoProcessor.from_pretrained(model_path,use_fast=True,min_pixels=256*32*32,max_pixels=max_pixels*32*32)
    if qwen_processor.tokenizer.pad_token is None:
        qwen_processor.tokenizer.pad_token = qwen_processor.tokenizer.eos_token

    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    router = None
    router_kind = None                          # "v2_gbdt"/"v2_lr"/"conformal"/"binary"/"4class"
    option_token_ids = None # v1 ABCD
    option_token_ids_all = None # A-Z → token_idv2 
    if enable_router:
        try:
            if skip_ori:
                print(f"⚠️ enable_router=True direct-answer logits skip_ori=False")
                skip_ori = False
            _rp = router_report_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "params", "Qwen3", "router_report.json",
            )
            with open(_rp, "r") as _f:
                _meta = json.load(_f)
            _mtype = _meta.get("model_type", "multinomial_logreg_4class")
            _tools_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tools",
            )
            if _tools_dir not in sys.path:
                sys.path.insert(0, _tools_dir)
            if _mtype in ("gbdt", "lr"):
                from router_v2 import RouterV2
                router = RouterV2.load(_rp)
                router_kind = f"v2_{_mtype}"
                if router_alpha is not None:
                    try:
                        _old_sf = router.s_floor
                        _new_sf = router.set_alpha(float(router_alpha))
                        print(f"[router] override α = {router_alpha}  "
                              f"s_floor: {_old_sf:.4f} → {_new_sf:.4f}")
                    except Exception as _e:
                        print(f"⚠️ router_alpha={router_alpha} s_floor: {_e}")
                        print(f" s_floor={router.s_floor:.4f} "
                              f"(α={router.alpha})")
            elif _mtype == "conformal_safe_skip_router":
                from router import Router
                router = Router.load(_rp)
                router_kind = "conformal"
            elif _mtype == "binary_logreg":
                from router import Router # 
                router = Router.load(_rp)
                router_kind = "binary"
            else:
                from difficulty_analysis.threshold_free_router import ThresholdFreeRouter
                router = ThresholdFreeRouter.load(_rp)
                router_kind = "4class"

            option_token_ids = {}
            for letter in "ABCD":
                ids = qwen_processor.tokenizer.encode(letter, add_special_tokens=False)
                option_token_ids[letter] = ids[0]
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
            print(f"⚠️ GRACE: {_e}")
            router = None
            router_kind = None

    num_samples = len(dataset_part)
    for batch_start in tqdm(range(0, num_samples, batch_size), desc=f"GPU{gpu_id} batches"):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = dataset_part[batch_start:batch_end]
        actual_batch_size = len(batch_samples)
        
        direct_messages_list = []
        batch_img_urls = []
        batch_ori_img_urls = []
        batch_ques = []

        # -------------------------------------------------------------
        use_v2_prompt = (router is not None and router_kind is not None
                         and router_kind.startswith("v2_"))

        for sample in batch_samples:
            img_url = [sample["image"]]
            ori_img_url = list(img_url)
            ques = sample["Text"]

            if use_v2_prompt:
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


                direct_output_texts = batch_messages2out(model, qwen_processor, batched_inputs)
                torch.cuda.empty_cache()

                if router is not None and option_token_ids is not None:
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
            results["innovation_info"] = {} # 

            _sample_id = str(sample.get("id", f"rank{rank}_b{batch_start}_i{idx}"))
            _sample_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in _sample_id)

            # ══════════════════════════════════════════════════════════════
            # ══════════════════════════════════════════════════════════════
            should_run_grace_pipeline = True # GRACE

            if (router is not None and router_kind is not None
                    and router_kind.startswith("v2_")
                    and idx < len(direct_v2_feats) and direct_v2_feats[idx] is not None):
                try:
                    v2f = direct_v2_feats[idx]
                    trigger, p_info = router.decide(
                        question_text=ques,
                        **v2f,
                    )
                    should_run_grace_pipeline = bool(trigger)
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
                    print(f" [router v2] GRACE: {_e}")
                    should_run_grace_pipeline = True
            elif router is not None and direct_entropies[idx] is not None:
                try:
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
                    if isinstance(p_info, dict):
                        for k, v in p_info.items():
                            results["innovation_info"][f"router_{k}"] = (
                                float(v) if isinstance(v, (int, float)) else v
                            )
                    results["innovation_info"]["router_trigger_grace"] = bool(trigger)
                except Exception as _e:
                    print(f" [router] GRACE: {_e}")
                    should_run_grace_pipeline = True

            if router is not None and not should_run_grace_pipeline:
                for s in sig:
                    for t in thre:
                        results["answer"][f"HiDe_s{s}_t{t}"] = direct_output_texts[idx]
                        results["bounding_box"][f"HiDe_s{s}_t{t}"] = {}
                results["prompt_text"][f"HiDe"] = "[router-skipped-grace]"
                results.pop("image", None)
                serialize_dict(results, savedir)
                torch.cuda.empty_cache()
                continue # 

            entity_text = entity_output_texts[idx]
            answer_out = entity_text.split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
            
            messages = [{"role": "user", "content": []}]
            for img in img_url:
                messages[-1]["content"].append({"type": "image", "image": img})
            
            outputs = {}
            if answer_out:
                entity_list = [e.strip() for e in answer_out.split(',') if e.strip()]

                expert_bboxes_per_img = None
                expert_results_raw = {}
                entity_token_indices = None
                entity_token_map = {}
                expert_reliability = 0.0
                sam3_supplement_bboxes_per_img = None # GRACE 
                sam3_entity_labels_per_img = None # GRACE 

                # ══════════════════════════════════════════════════════════════
                # ══════════════════════════════════════════════════════════════
                if enable_grace:
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

                messages[-1]["content"].append(
                    {"type": "text", "text": "Search the following entities in the images: " + answer_out}
                )
                text, image_inputs, video_inputs, inputs, video_kwargs = get_inputs(
                    messages, qwen_processor, model
                )

                attention, idx2word_dicts, img_start, img_end = messages2att(
                    model, qwen_processor, inputs
                )

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
                                for imgidx in bounding_boxes:
                                    expanded = [expand_bbox(b, pmgvv_expand_ratio) for b in bounding_boxes[imgidx]]
                                    bounding_boxes[imgidx] = expanded
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

                                    overlay_merged = {}
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
                                print(f"[secondary_sam3] : {_e}")

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
                                        print(f"[save_lpd] {_fname}: {_e}")
                            if highlight_imgs:
                                for _hi, _h_img in enumerate(highlight_imgs):
                                    _fname = f"{_sample_id}_s{s}_t{t}_lpd_out_{_hi}.png"
                                    _fpath = _os.path.join(heatmap_save_dir, _fname)
                                    try:
                                        _b64d = _h_img.split(",")[1] if "," in _h_img else _h_img
                                        with open(_fpath, "wb") as _f:
                                            _f.write(base64.b64decode(_b64d))
                                    except Exception as _e:
                                        print(f"[save_lpd] {_fname}: {_e}")

                        final_messages = [{"role": "user","content": []}]
                        append_visual_inputs(final_messages[-1]["content"], ori_img_url, hide_highlight_imgs, highlight_imgs)
                        final_messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices letter. The final and only output content must be enclosed within `<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."})
                        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(final_messages,qwen_processor,model)
                        output_text,_ = messages2out(model,qwen_processor,inputs)
                        if not str(s) in outputs:outputs[str(s)] = {}
                        outputs[str(s)][str(t)] = [[answer_out],output_text,crop_list,highlight_imgs,final_messages,words_lines,img_merged_boxes,bounding_boxes]
            else:
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
