import os
from utiles_internvl import *
from utiles_internvl import _LEGEND_PALETTE
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import io
import base64
import gc
import sys
import multiprocessing
from multiprocessing import Pool
from accelerate import infer_auto_device_map, dispatch_model
import shutil
import json
import torch.multiprocessing as mp
from joblib import Parallel, delayed
import time
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = 28000000000
from scipy.ndimage import zoom
import subprocess


# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
def _direct_answer_two_pass(model, tokenizer, pixel_values, question,
                             generation_config, need_first_logits=False):
    """
    Returns:
        ori_text: str
        first_logits: np.ndarray [vocab] or None
    """
    try:
        response, _ = model.chat(tokenizer, pixel_values, question, generation_config,
                                  history=None, return_history=True)
        ori_text = response if isinstance(response, str) else (response[0] if response else "")
    except Exception as e:
        import traceback as _tb
        print(f"[direct_answer] model.chat failed: {e}")
        _tb.print_exc()
        ori_text = ""

    first_logits = None
    if need_first_logits:
        try:
            _pv, input_ids, attention_mask, gen_cfg = get_input(
                model, tokenizer, pixel_values, question, dict(generation_config),
                history=None, return_history=True,
            )
            with torch.no_grad():
                if _pv is not None:
                    vit_embeds = model.extract_feature(_pv)
                    input_embeds = model.language_model.get_input_embeddings()(input_ids)
                    B, N, C = input_embeds.shape
                    input_embeds = input_embeds.reshape(B * N, C)
                    flat_ids = input_ids.reshape(B * N)
                    selected = (flat_ids == model.img_context_token_id)
                    if selected.sum() != 0:
                        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
                    input_embeds = input_embeds.reshape(B, N, C)
                else:
                    input_embeds = model.language_model.get_input_embeddings()(input_ids)
                gen_out = model.language_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=gen_cfg.get("pad_token_id", tokenizer.eos_token_id),
                )
            if gen_out.scores is not None and len(gen_out.scores) > 0:
                first_logits = gen_out.scores[0][0].float().cpu().numpy()
            del gen_out, input_embeds
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback as _tb
            print(f"[direct_answer] 1-token forward failed: {e}")
            _tb.print_exc()
            first_logits = None
    return ori_text, first_logits


def cycle_epoch_infer(gpu_id, rank, dataset_part, savedir, CoT, cycle_times, sig, thre,
                      enable_grace=False,
                      grace_sam3_url="http://localhost:8002/predict",
                      grace_max_sam3_per_entity=3,
                      enable_router=False,
                      router_report_path=None,
                      router_alpha=None,
                      skip_ori=False,
                      heatmap_save_dir=None):
    """InternVL3 + HiDe TAD + Router/GRACE GPU 

    :
        enable_grace: GRACESAM3 bbox + LPD overlay + SAM3
        grace_sam3_url: SAM3 HTTP 
        grace_max_sam3_per_entity: SAM3 bbox 
        enable_router: RouterV2 (Cost-Aware Conformal Safe-Skip)
        router_report_path: router v2 JSON 
            <repo>/tools/internvl3/router_v2_report.json
        router_alpha: α s_floorNone 
        skip_ori: direct-answer router False
    """
    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    device = f"cuda:{gpu_id}"
    print(rank, len(dataset_part), device,
          f"enable_grace={enable_grace}, enable_router={enable_router}, skip_ori={skip_ori}")

    path = r"/path/to/ckpt/InternVL3-8B-Instruct"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        attn_implementation="eager",
        trust_remote_code=True,
        device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    for ly in range(model.language_model.model.config.num_hidden_layers):
        model.language_model.model.layers[ly].self_attn.forward = types.MethodType(
            layer_forward, model.language_model.model.layers[ly].self_attn)
    model.language_model.model.forward = types.MethodType(
        qwen2_forward, model.language_model.model)

    # "tensor a (28) must match b (128) at non-singleton dimension 3"。
    try:
        _lm_cfg = model.language_model.model.config
        _lm_cfg._attn_implementation = "eager"
        if hasattr(_lm_cfg, "_attn_implementation_internal"):
            _lm_cfg._attn_implementation_internal = "eager"
        for _ly in model.language_model.model.layers:
            if hasattr(_ly.self_attn, "config"):
                _ly.self_attn.config._attn_implementation = "eager"
    except Exception as _e:
        print(f"[warn] force eager attn failed: {_e}")

    # "tensor a (28) must match b (128) at non-singleton dimension 3"。
    def _compat_update_causal_mask(self, attention_mask, inputs_embeds, cache_position,
                                    past_key_values=None, output_attentions=False):
        _ccm = None
        try:
            from transformers.masking_utils import create_causal_mask as _ccm  # type: ignore
        except Exception:
            try:
                from transformers.models.qwen2.modeling_qwen2 import create_causal_mask as _ccm  # type: ignore
            except Exception:
                _ccm = None
        out = None
        if _ccm is not None:
            try:
                out = _ccm(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=None,
                )
            except Exception:
                out = None
            if out is not None and out.dim() != 4:
                out = None
        if out is None:
            bsz, q_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            kv_len = past_len + q_len
            dtype, device = inputs_embeds.dtype, inputs_embeds.device
            min_val = torch.finfo(dtype).min
            mask = torch.full((q_len, kv_len), min_val, dtype=dtype, device=device)
            i = torch.arange(q_len, device=device).unsqueeze(1)
            j = torch.arange(kv_len, device=device).unsqueeze(0)
            mask = mask.masked_fill(j <= (i + past_len), 0.0)
            out = mask[None, None, :, :].expand(bsz, 1, q_len, kv_len).contiguous()
            if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape[-1] == kv_len:
                pad = (1.0 - attention_mask.to(dtype))[:, None, None, :] * min_val
                out = out + pad
        return out
    model.language_model.model._update_causal_mask = types.MethodType(
        _compat_update_causal_mask, model.language_model.model)

    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    router = None
    router_kind = None
    option_token_ids_all = None
    if enable_router:
        try:
            if skip_ori:
                print("⚠️ enable_router=True direct-answer logits skip_ori=False")
                skip_ori = False
            _rp = router_report_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "params", "Internvl", "router_report.json",
            )
            with open(_rp, "r") as _f:
                _meta = json.load(_f)
            _mtype = _meta.get("model_type", "gbdt")
            _tools_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "tools",
            )
            if _tools_dir not in sys.path:
                sys.path.insert(0, _tools_dir)
            from router_v2 import RouterV2   # type: ignore
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

            option_token_ids_all = {}
            for c in range(ord("A"), ord("Z") + 1):
                ch = chr(c)
                ids = tokenizer.encode(ch, add_special_tokens=False)
                if ids:
                    option_token_ids_all[ch] = ids[0]
            print(f"[router] kind={router_kind}, loaded from {_rp}")
            print(f"[router] features: {router.features}")
            print(f"[router] decision rule: run_grace ⟺ "
                  f"s(x) ≥ s_floor = {router.s_floor:.4f}  (alpha={router.alpha:.3f})")
        except Exception as _e:
            print(f"⚠️ GRACE: {_e}")
            router = None
            router_kind = None

    use_v2_prompt = (router is not None and router_kind is not None
                     and router_kind.startswith("v2_"))

    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    for sample in tqdm(dataset_part):
        try:
            results = sample
            pil_img = Image.open(sample["image"]).convert('RGBA')
            img_url = [pil_img]
            pixel_values_list = []
            block_indices = []
            for img in img_url:
                pixel_value, block_index = load_image(img, max_num=128)
                pixel_values_list.append(pixel_value)
                block_indices.append(block_index)
            pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).to(device=model.device)

            ques = sample["Text"]
            if use_v2_prompt:
                direct_question_text = ques + build_answer_instr_v2(ques)
            else:
                direct_question_text = ques + (
                    "\nAnswer with the option's letter from the given choices letter. "
                    "The final and only output content must be enclosed within "
                    "`<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."
                )
            direct_question_prompted = "<image>\n" + direct_question_text

            results["answer"] = {}
            results["enetity"] = {}
            results["innovation_info"] = {}
            _sample_id = str(sample.get("id", f"rank{rank}_{time.time_ns()}"))
            _sample_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in _sample_id)

            direct_first_logits = None
            direct_v2_feat = None
            if skip_ori and router is None:
                direct_text = ""
            else:
                _need_logits = (router is not None and option_token_ids_all is not None)
                direct_text, direct_first_logits = _direct_answer_two_pass(
                    model, tokenizer, pixel_values, direct_question_prompted,
                    generation_config, need_first_logits=_need_logits,
                )
                if not skip_ori:
                    results["answer"]["ori"] = direct_text
                if use_v2_prompt and direct_first_logits is not None:
                    direct_v2_feat = compute_router_v2_features(
                        direct_first_logits, ques, option_token_ids_all,
                    )

            should_run_grace_pipeline = True
            if router is not None and use_v2_prompt and direct_v2_feat is not None:
                try:
                    trigger, p_info = router.decide(
                        question_text=ques, **direct_v2_feat,
                    )
                    should_run_grace_pipeline = bool(trigger)
                    for _k, _v in direct_v2_feat.items():
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
                    print(f" [router] GRACE: {_e}")
                    should_run_grace_pipeline = True

            if router is not None and not should_run_grace_pipeline:
                for i in range(cycle_times):
                    results["answer"][f"TAD_{i+1}"] = direct_text
                    results["enetity"][f"TAD_{i+1}"] = "[router-skipped-grace]"
                results.pop("image", None)
                serialize_dict(results, savedir)
                torch.cuda.empty_cache()
                continue

            if CoT:
                for i in range(cycle_times):
                    torch.cuda.empty_cache()
                    output_text, crop_list, highlight_imgs, hide_highlight_imgs, \
                        pixel_values, block_indices, words_lines, \
                        img_merged_boxes, bounding_boxes, prompt_ques = once_cot_infer_v2(
                            model, tokenizer, pixel_values, block_indices,
                            direct_question_prompted, # VLM prompt
                            generation_config, img_url, sig, thre,
                            ques_for_entity=ques,
                            enable_grace=enable_grace,
                            grace_sam3_url=grace_sam3_url,
                            grace_max_sam3_per_entity=grace_max_sam3_per_entity,
                            sample_id=_sample_id,
                            heatmap_save_dir=heatmap_save_dir,
                        )
                    results["answer"][f"TAD_{i+1}"] = output_text[0] if isinstance(output_text, (list, tuple)) else output_text
                    results["enetity"][f"TAD_{i+1}"] = prompt_ques
                    for h_img in highlight_imgs:
                        img_url.append(h_img)
                    if enable_grace:
                        results["innovation_info"][f"grace_sam3_total_TAD_{i+1}"] = len(
                            [b for bx in bounding_boxes.values() for b in bx]
                        )

            results.pop("image", None)
            serialize_dict(results, savedir)
            torch.cuda.empty_cache()
            print(savedir)
        except Exception as _samp_e:
            import traceback as _tb
            _img_path = sample.get("image", "<unknown>") if isinstance(sample, dict) else "<not-dict>"
            print(f"[rank={rank}] , image={_img_path}: {_samp_e}")
            _tb.print_exc()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
    del model
    torch.cuda.empty_cache()


def once_cot_infer_v2(model, tokenizer, pixel_values, block_indices, question,
                      generation_config, img_url, sig, thre,
                      ques_for_entity=None,
                      enable_grace=False,
                      grace_sam3_url="http://localhost:8002/predict",
                      grace_max_sam3_per_entity=3,
                      sample_id="sample",
                      heatmap_save_dir=None):
    """ once_cot_infer GRACE SAM3 bbox → LPD overlay + SAM3 + hide 

    :
        question: VLM prompt "<image>\n"
        ques_for_entity: None question
    Returns:
        output_text, crop_list, highlight_imgs, hide_highlight_imgs,
        pixel_values, block_indices, words_lines, img_merged_boxes,
        bounding_boxes, prompt_output_text
    """
    raw_question = ques_for_entity if ques_for_entity is not None else question
    prompt_ques = """Your task is to extract entities from a user's question. You must follow a strict set of rules to deconstruct and reformat these entities into a canonical, attribute-based format. The output should be a single line of comma-separated values.

Extraction Rules:

Deconstruct Object Descriptions: For any object described with adjectives, first state the core noun, then list its properties using a with [property] format.

Example Transformation: "the large blue truck" becomes truck with large size with blue color.
Example Transformation: "the man in the green shirt" becomes man in a shirt with green color.
Standardize Possessives: Convert possessive forms (like X's Y) into an of structure (Y of X).

Examples:

Question: Which one is closer to the camera, the black vehicle or the silver vehicle?
Answer: vehicle with black color, vehicle with silver color

Question: What is the color of the woman's handbag? Blue or white?
Answer: handbag of woman

Question: What is the man in the green shirt holding next to the wooden table?
Answer: man in a shirt with green color, table with wooden material

Question: What is the color of the guard's glove?
Answer: glove of guard

Question: Is the dog on the left or right side of the scooter?
Answer: dog, scooter

Now, extract entities from the question: """
    prompt_ques_full = prompt_ques + raw_question.split("<image>\n")[-1].split("\n")[0] + "\nAnswer: "
    prompt_output_text, _ = messages2out(
        model, tokenizer, None, prompt_ques_full, generation_config,
        history=None, return_history=True,
    )
    ent_text = prompt_output_text if isinstance(prompt_output_text, str) else (
        prompt_output_text[0] if prompt_output_text else "")
    entity_list = [e.strip() for e in ent_text.split(',') if e.strip()]

    sam3_supplement_bboxes_per_img = None
    sam3_entity_labels_per_img = None
    if enable_grace and entity_list:
        try:
            sam3_results_raw = call_grounding_expert(
                img_url[0], entity_list, expert_url=grace_sam3_url
            )
            all_sam3_bboxes, all_sam3_labels = get_sam3_supplement_bboxes(
                sam3_results_raw, max_per_entity=grace_max_sam3_per_entity
            )
            if all_sam3_bboxes:
                sam3_supplement_bboxes_per_img = {0: all_sam3_bboxes}
                sam3_entity_labels_per_img = {0: all_sam3_labels}
        except Exception as _e:
            print(f"[GRACE] SAM3 : {_e}")

    search_question = "<image>\n" + "Search the following entities in the images: " + ent_text
    attention, idx2word_dicts, img_start, img_end = messages2att(
        model, tokenizer, pixel_values, search_question, generation_config,
        history=None, return_history=True,
    )
    att_unpacked = from_img_and_att_get_cropbox(
        model, tokenizer, pixel_values, block_indices, search_question,
        generation_config, attention, idx2word_dicts, img_url,
        img_start, img_end, None, sig, thre,
        history=None, return_history=True,
        enable_grace=enable_grace,
        sam3_supplement_bboxes_per_img=sam3_supplement_bboxes_per_img,
        sam3_entity_labels_per_img=sam3_entity_labels_per_img,
        heatmap_save_dir=heatmap_save_dir,
        sample_id=sample_id,
        entity_text=ent_text,
        return_hide_copy=enable_grace,
    )
    if enable_grace:
        img_merged_boxes, crop_list, words_lines, highlight_imgs, \
            bounding_boxes, hide_highlight_imgs = att_unpacked
    else:
        img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes = att_unpacked
        hide_highlight_imgs = []

    if enable_grace and entity_list and highlight_imgs:
        try:
            found_secondary, new_entries_per_img = secondary_sam3_check_on_att_bboxes(
                img_url, bounding_boxes, entity_list,
                sam3_url=grace_sam3_url,
                max_per_entity=2, max_att_bboxes=4, max_new_bboxes=3,
                existing_sam3_bboxes_per_img=sam3_supplement_bboxes_per_img,
            )
            if found_secondary and new_entries_per_img:
                _first_labels = []
                if sam3_entity_labels_per_img:
                    for _lbs in sam3_entity_labels_per_img.values():
                        _first_labels.extend(list(_lbs) if _lbs else [])
                _sec_all_labels = [
                    lb for entries in new_entries_per_img.values() for _, lb in entries
                ]
                label2color_full, _ = assign_entity_colors(
                    list(_first_labels) + list(_sec_all_labels)
                )
                overlay_merged = {}
                if sam3_supplement_bboxes_per_img:
                    for _imgidx, _bxs in sam3_supplement_bboxes_per_img.items():
                        _lbs = list(sam3_entity_labels_per_img.get(_imgidx, [])) \
                            if sam3_entity_labels_per_img else []
                        while len(_lbs) < len(_bxs):
                            _lbs.append("")
                        for _bx, _lb in zip(_bxs, _lbs):
                            _color = label2color_full.get((_lb or "").strip(), _LEGEND_PALETTE[0])
                            overlay_merged.setdefault(_imgidx, []).append({
                                "bbox_norm": list(_bx), "color": _color, "label": _lb or "",
                            })
                for _imgidx, entries in new_entries_per_img.items():
                    if _imgidx not in bounding_boxes:
                        bounding_boxes[_imgidx] = []
                    for _bx, _lb in entries:
                        bounding_boxes[_imgidx].append(list(_bx))
                        _color = label2color_full.get((_lb or "").strip(), _LEGEND_PALETTE[1])
                        overlay_merged.setdefault(_imgidx, []).append({
                            "bbox_norm": list(_bx), "color": _color, "label": _lb or "",
                        })
                new_hl = []
                for _imgidx in bounding_boxes:
                    _src = img_url[_imgidx] if _imgidx < len(img_url) else img_url[0]
                    _overlays = overlay_merged.get(_imgidx) or None
                    _img, _bboxs, _used_cl = compact_and_center_with_relative_pos(
                        _imgidx, len(img_url), _src, bounding_boxes[_imgidx],
                        overlay_bboxes=_overlays, draw_color_border=bool(_overlays),
                    )
                    if _img is not None:
                        bounding_boxes[_imgidx] = _bboxs
                        _seen = set(); _merged = []
                        for cl in list(_used_cl):
                            key = (tuple(cl[0]), cl[1])
                            if key not in _seen and cl[1]:
                                _seen.add(key); _merged.append(cl)
                        if _merged:
                            _img = add_legend_to_lpd_image(_img, _merged, position="bottom")
                        new_hl.append(_img)
                if new_hl:
                    highlight_imgs = new_hl
        except Exception as _e:
            print(f"[secondary_sam3] : {_e}")

    if heatmap_save_dir is not None:
        try:
            import os as _os
            _os.makedirs(heatmap_save_dir, exist_ok=True)
            if enable_grace and hide_highlight_imgs and hide_highlight_imgs != highlight_imgs:
                for _hi, _h_img in enumerate(hide_highlight_imgs):
                    _fp = _os.path.join(
                        heatmap_save_dir,
                        f"{sample_id}_s{sig}_t{thre}_lpd_hide_out_{_hi}.png",
                    )
                    save_pil_lpd(_h_img, _fp)
            if highlight_imgs:
                for _hi, _h_img in enumerate(highlight_imgs):
                    _fp = _os.path.join(
                        heatmap_save_dir,
                        f"{sample_id}_s{sig}_t{thre}_lpd_out_{_hi}.png",
                    )
                    save_pil_lpd(_h_img, _fp)
        except Exception as _e:
            print(f"[save_lpd] : {_e}")

    final_images = list(img_url)
    if enable_grace and hide_highlight_imgs and hide_highlight_imgs != highlight_imgs:
        for h in hide_highlight_imgs:
            final_images.append(h)
    for h in highlight_imgs:
        final_images.append(h)

    for _new_img in final_images[len(img_url):]:
        try:
            pixel_values_tmp, _bi = load_image(_new_img, max_num=128)
            pixel_values = torch.cat(
                [pixel_values, pixel_values_tmp.to(torch.bfloat16).to(model.device)],
                dim=0,
            )
            block_indices = block_indices + [_bi]
        except Exception as _e:
            print(f"[final_merge] load_image : {_e}")

    extra_n = len(final_images) - len(img_url)
    final_question = ("<image>\n" * extra_n) + question if extra_n > 0 else question

    pixel_values, input_ids, attention_mask, generation_config = get_input(
        model, tokenizer, pixel_values, final_question, generation_config,
        history=None, return_history=True,
    )
    output_text, _ = messages2out(
        model, tokenizer, pixel_values, final_question, generation_config,
        history=None, return_history=True,
    )
    return (output_text, crop_list, highlight_imgs, hide_highlight_imgs,
            pixel_values, block_indices, words_lines, img_merged_boxes,
            bounding_boxes, ent_text)


def get_available_gpus(max_memory_mb=1000, max_gpus=None):
    """ GPU"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        used_memory = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        gpu_memory_pairs = [(i, mem) for i, mem in enumerate(used_memory)]
        gpu_memory_pairs.sort(key=lambda x: x[1])
        available_gpus = [gpu_id for gpu_id, mem in gpu_memory_pairs if mem < max_memory_mb]
        if max_gpus is not None:
            available_gpus = available_gpus[:max_gpus]
        return available_gpus
    except Exception as e:
        print(f"Error detecting GPU memory: {e}")
        return []


def main(datasetdir, savedir, CoT, cycle_times, Parallels, sig, thre,
         para_nums=6,
         enable_grace=False,
         grace_sam3_url="http://localhost:8002/predict",
         grace_max_sam3_per_entity=3,
         enable_router=False,
         router_report_path=None,
         router_alpha=None,
         skip_ori=False,
         heatmap_save_dir=None):
    dataset = load_dataset_Vstar_json(datasetdir)
    random.shuffle(dataset)
    available_gpus = get_available_gpus(max_memory_mb=96000 - 40000, max_gpus=para_nums)
    if len(available_gpus) == 0:
        print("❌ GPU > 40000MB")
        return
    print(f"✅ {len(available_gpus)} GPU: {available_gpus}")
    splits = np.array_split(dataset, len(available_gpus))
    print("")
    if not Parallels:
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            cycle_epoch_infer(
                gpu_id, rank, dataset_part, savedir, CoT, cycle_times, sig, thre,
                enable_grace=enable_grace,
                grace_sam3_url=grace_sam3_url,
                grace_max_sam3_per_entity=grace_max_sam3_per_entity,
                enable_router=enable_router,
                router_report_path=router_report_path,
                router_alpha=router_alpha,
                skip_ori=skip_ori,
                heatmap_save_dir=heatmap_save_dir,
            )
    else:
        pool = Pool(processes=len(available_gpus))
        async_results = []
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            ar = pool.apply_async(
                cycle_epoch_infer,
                args=(gpu_id, rank, dataset_part, savedir, CoT, cycle_times, sig, thre),
                kwds=dict(
                    enable_grace=enable_grace,
                    grace_sam3_url=grace_sam3_url,
                    grace_max_sam3_per_entity=grace_max_sam3_per_entity,
                    enable_router=enable_router,
                    router_report_path=router_report_path,
                    router_alpha=router_alpha,
                    skip_ori=skip_ori,
                    heatmap_save_dir=heatmap_save_dir,
                ),
            )
            async_results.append((rank, gpu_id, ar))
        pool.close()
        for rank, gpu_id, ar in async_results:
            try:
                ar.get() # + 
            except Exception as _sub_e:
                import traceback as _tb
                print(f"[pool] worker rank={rank} gpu={gpu_id} FAILED: {_sub_e}")
                _tb.print_exc()
        pool.join()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    maxp = [16384]
    CoT = [True]
    Parallels = True
    cycle_times = 1
    sigma = [1]
    threshold = [0.4]
    seed = 2077
    random.seed(seed)

    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    ENABLE_GRACE = False
    GRACE_SAM3_URL = "http://localhost:8002/predict"
    GRACE_MAX_SAM3_PER_ENTITY = 10

    ENABLE_ROUTER = False
    ROUTER_REPORT_PATH = None # None → params/Internvl/router_report.json
    ROUTER_ALPHA = None # None → α
    SKIP_ORI = False
    HEATMAP_SAVE_DIR = None # LPD/

    current_time = time.localtime()
    formatted_time = time.strftime("%Y-%m-%d", current_time)
    save_dir = f'/path/to/output/internvl/{formatted_time}'
    create_directory(save_dir)
    for maxpp in maxp:
        for coti in CoT:
            for sig in sigma:
                for thre in threshold:
                    datasetdir = f"/path/to/data/benchmark.json"
                    _tag = []
                    if ENABLE_GRACE: _tag.append("GRACE")
                    if ENABLE_ROUTER: _tag.append(f"Router-a{ROUTER_ALPHA if ROUTER_ALPHA is not None else 'def'}")
                    _tag_str = "-".join(_tag) if _tag else "HiDe"
                    savejson = f'{save_dir}/Vstar-{_tag_str}-internvl3-{cycle_times}-15layer-sig{sig}-thre{thre}-norm.json'
                    main(
                        datasetdir, savejson, coti, cycle_times, Parallels, sig, thre,
                        enable_grace=ENABLE_GRACE,
                        grace_sam3_url=GRACE_SAM3_URL,
                        grace_max_sam3_per_entity=GRACE_MAX_SAM3_PER_ENTITY,
                        enable_router=ENABLE_ROUTER,
                        router_report_path=ROUTER_REPORT_PATH,
                        router_alpha=ROUTER_ALPHA,
                        skip_ori=SKIP_ORI,
                        heatmap_save_dir=HEATMAP_SAVE_DIR,
                    )
