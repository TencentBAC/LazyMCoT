"""
================================================================================
RELABEL + EXTRACT (LITE) — 只提取 train_router_v2.py 真正使用的 CORE5 统计特征
================================================================================

基于 tools/relabel_and_extract.py 精简, 去掉耗时最大的视觉注意力 forward
以及 router 训练根本用不到的其它字段。每条样本只跑 *一次* generate (max_new_tokens=1)
即可完成 relabel 和特征提取, 相比完整版约节省 ~50% GPU 时间。

保留的 router 相关字段 (严格对齐 train_router_v2.py 的 DEFAULT_FEATS):
    metadata:
        id, image_path, label            # label = 'ori_correct' / 'ori_wrong'
        question, labels, category       # 便于后续诊断 / 划分
        ori_pred, ori_raw, ori_correct   # relabel 字段
        model_type
    CORE5 features:
        answer_topp              (选项 softmax top1)
        answer_margin            (选项 softmax top1 - top2)
        vocab_full_entropy_norm  (全词表首 token 熵 / log V)
        option_mass              (首 token 对选项字母的总概率)
        logit_gap_opt_nonopt     (max 选项 logit - max 非选项 logit)

**不再提取** (相对 relabel_and_extract.py):
    - p_options / option_probs / answer_entropy / answer_entropy_norm
    - vocab_full_entropy / vocab_top10_entropy
    - 所有 vatt_* 视觉注意力特征 (节省一次 output_attentions forward)
    - has_spatial_keyword / has_fine_detail_keyword / question_length_tokens
    - parsed_options / num_options / num_visual_tokens

输出:
  <out_dir>/
    ori_preds_rank{r}.jsonl     # 每条样本的 ori + CORE5 (per-rank shard)
    ori_preds_all.jsonl         # 合并去重后的完整 jsonl
    features_train_v2.jsonl     # 供 train_router_v2.py 直接消费
    ori_correct.json / ori_wrong.json            # 按 ori_pred==labels 划分

默认数据集 (--sources 不传时):
  - DeepScan/data/vstar_bench2/vstar_bench.json
  - DeepScan/data/TreeBench/TreeBench.json
  - DeepScan/data/HR-Bench/hr_bench_8k.json
  - DeepScan/data/HR-Bench/hr_bench_4k.json

用法:
  python relabel_and_extract_lite.py --model_type qwen3_vl \
      --model_path /ckpt/Qwen3-VL-8B-Instruct \
      --gpu_ids 0,1,2,3 --out_dir /data/relabeled_qwen3_lite
"""
from __future__ import annotations
import os
# 显存碎片缓解 (必须在 import torch 之前)
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

import sys
import json
import argparse
import re
import math
import glob

_THIS = os.path.dirname(os.path.abspath(__file__))
_HIDE_ROOT = os.path.dirname(_THIS)
_QWEN25 = os.path.join(_HIDE_ROOT, "Qwen2.5")
_QWEN3 = os.path.join(_HIDE_ROOT, "Qwen3")
_INTERNVL = os.path.join(_HIDE_ROOT, "Internvl")  # OpenGVLab 原版工具集
                                                  # (load_image / get_input / conversation)


# ==============================================================================
# Section 1. Prompt / 答案解析 / 选项解析
# ==============================================================================
_OPT_PAT = re.compile(r"(?:^|\n|\s)\(?([A-Z])[\)\.\:]", re.MULTILINE)


def parse_options(question: str):
    if not question:
        return ["A", "B", "C", "D"]
    found = set()
    for m in _OPT_PAT.finditer(question):
        found.add(m.group(1))
    letters = []
    for c in range(ord("A"), ord("Z") + 1):
        ch = chr(c)
        if ch in found:
            letters.append(ch)
        else:
            break
    return letters if len(letters) >= 2 else ["A", "B", "C", "D"]


def build_answer_instr_direct(letters):
    letters_str = (", ".join(letters[:-1]) + f", or {letters[-1]}"
                   if len(letters) > 1 else letters[0])
    return (
        f"\nAnswer the multiple-choice question with ONLY the single letter "
        f"of the correct option ({letters_str}). "
        f"Do not output any word, punctuation, tag, explanation or whitespace. "
        f"Your entire response must be exactly one character — the option letter.\n"
        f"Answer:"
    )


def build_answer_instr_tag(letters):
    return (
        "\nAnswer with the option's letter from the given choices letter. "
        "The final and only output content must be enclosed within "
        "`<FINAL_OUTPUT>` and `</FINAL_OUTPUT>` tags."
    )


def extract_letter(s: str, valid_letters=None) -> str:
    s = str(s or "").strip()

    def _find(text, restrict):
        for c in text:
            if c.isupper() and c.isalpha():
                if (restrict is None) or (c in restrict):
                    return c
        return ""

    m = re.search(r"<FINAL_OUTPUT>(.*?)</FINAL_OUTPUT>", s)
    if m:
        inner = m.group(1).strip()
        if not inner:
            return ""
        res = _find(inner, valid_letters)
        return res or _find(inner, None)
    s_clean = re.sub(r"FINAL_OUTPUT", "", s, flags=re.IGNORECASE)
    return _find(s_clean, valid_letters)


# ==============================================================================
# Section 2. Model Adapter — 只需要 generate_first_token_logits, 不需要 attention
# ==============================================================================
class BaseVLMAdapter:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None

    def option_token_id(self, letter: str):
        ids = self.tokenizer.encode(letter, add_special_tokens=False)
        return ids[0] if ids else None

    def close(self):
        import torch
        try:
            del self.model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


class Qwen25VLAdapter(BaseVLMAdapter):
    def load(self, model_path, device, max_pixels, **kw):
        import torch
        if _QWEN25 not in sys.path:
            sys.path.insert(0, _QWEN25)
        from transformers import AutoProcessor
        from modeling_qwen2_5_vl_re_infer import Qwen2_5_VLForConditionalGeneration
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True,
            min_pixels=256 * 28 * 28, max_pixels=max_pixels * 28 * 28,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_inputs(self, messages):
        from Get_box import batch_get_inputs
        return batch_get_inputs([messages], self.processor, self.model)

    def generate_first_token_logits(self, inputs):
        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs, use_cache=True, max_new_tokens=1,
                do_sample=False, return_dict_in_generate=True,
                output_scores=True,
            )
        scores0 = out.scores[0][0].float().cpu()
        try: del out
        except Exception: pass
        return scores0


class Qwen3VLAdapter(BaseVLMAdapter):
    def load(self, model_path, device, max_pixels, **kw):
        import torch
        if _QWEN3 not in sys.path:
            sys.path.insert(0, _QWEN3)
        from transformers import AutoProcessor
        try:
            from modeling_qwen3_vl_re_infer import Qwen3VLForConditionalGeneration
        except Exception:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, use_fast=True,
            min_pixels=256 * 28 * 28, max_pixels=max_pixels * 28 * 28,
        )
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_inputs(self, messages):
        try:
            from Get_box import batch_get_inputs
            return batch_get_inputs([messages], self.processor, self.model)
        except Exception:
            return _standard_build_inputs(self.processor, self.model, messages)

    def generate_first_token_logits(self, inputs):
        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs, use_cache=True, max_new_tokens=1,
                do_sample=False, return_dict_in_generate=True,
                output_scores=True,
            )
        scores0 = out.scores[0][0].float().cpu()
        try: del out
        except Exception: pass
        return scores0


class InternVL3Adapter(BaseVLMAdapter):
    """对齐 Internvl/cycle_inference_internvl.py 的 OpenGVLab 原版加载方式:
      - AutoModel + AutoTokenizer (trust_remote_code=True)
      - 不使用 AutoProcessor / apply_chat_template
      - 使用 Internvl/utiles_internvl.py 的 load_image_from_path / get_input 构造
        pixel_values 和 input_ids (动态 patch + thumbnail + IMG_CONTEXT 模板)
      - 直接调用 model.generate(pixel_values=, input_ids=, attention_mask=, ...)
        拿首 token logits

    适用 ckpt: OpenGVLab/InternVL3-8B-Instruct 等 *未带 -hf 后缀* 的官方仓库。
    """
    def load(self, model_path, device, max_pixels, **kw):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if _INTERNVL not in sys.path:
            sys.path.insert(0, _INTERNVL)
        # 必须在 import utiles_internvl 之前打补丁: utiles_internvl 顶部
        # `from transformers.models.qwen2.modeling_qwen2 import *` 在新版
        # transformers 上拿不到 Tuple / Optional / Cache 等符号会抛 NameError
        # _patch_internvl_typing()
        # 引入与 cycle_inference_internvl.py 相同的工具 (含 layer_forward / qwen2_forward,
        # 但 lite 版只用 load_image_from_path + get_input, 不需要 patch attention)
        from utiles_internvl import load_image_from_path, get_input  # noqa: F401
        self._iv_load_image = load_image_from_path
        self._iv_get_input = get_input

        self.device = device
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, use_flash_attn=True,
            trust_remote_code=True, device_map=device,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = None  # 不使用

        # max_pixels (Qwen 语义) -> max_num (InternVL 切片块数上限) 的映射:
        # Qwen 用 max_pixels * 28*28 像素上限, InternVL 用 max_num 个 448x448 块。
        # 默认 max_num=128 与原仓库 once_cot_infer 一致, 同时允许通过 max_pixels
        # 显式覆盖 (传入数字即被解释为 max_num)。
        self._iv_max_num = int(kw.get("internvl_max_num", 128))
        # 缓存 IMG_CONTEXT token id (用于 build_inputs)
        self._img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>")

    def build_inputs(self, messages):
        """messages 形如:
            [{"role":"user","content":[{"type":"image","image": <path>}, {"type":"text","text": <q>}]}]
        我们提取出 image_path 和 question, 用 OpenGVLab 原版的 load_image_from_path
        + get_input 构造 (pixel_values, input_ids, attention_mask)。
        """
        import torch
        # 解析 messages
        image_path = None
        text = ""
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    image_path = item.get("image")
                elif item.get("type") == "text":
                    text = item.get("text", "")
        assert image_path is not None, "InternVL3Adapter requires an image"
        # 必须加 <image>\n 前缀, 与 cycle_inference_internvl.py 第 70 行一致
        question = "<image>\n" + text

        pixel_values, _block_indices = self._iv_load_image(
            image_path, max_num=self._iv_max_num)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)

        gen_cfg = dict(max_new_tokens=1, do_sample=False)
        pixel_values, input_ids, attention_mask, gen_cfg = self._iv_get_input(
            self.model, self.tokenizer, pixel_values, question, gen_cfg,
            history=None, return_history=False)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "_eos_token_id": gen_cfg.get("eos_token_id", None),
            "_pad_token_id": gen_cfg.get("pad_token_id", None),
        }

    def generate_first_token_logits(self, inputs):
        import torch
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        eos_id = inputs.get("_eos_token_id", None)
        pad_id = inputs.get("_pad_token_id", None)
        # 兜底: 如果 _pad_token_id 没拿到, 直接从 self.tokenizer 取;
        # InternVL3-8B-Instruct: pad_token="<\n>" (id=151643), 不同于 eos (151645).
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        # NOTE: 不要传 use_cache —— InternVL3 wrapper (modeling_internvl_chat.py)
        # 内部 self.language_model.generate(...) 已经强制传了 use_cache=True,
        # 这里再传会触发: "got multiple values for keyword argument 'use_cache'"
        gen_kwargs = dict(
            max_new_tokens=1, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if pad_id is not None:
            # 显式指定 pad_token_id, 避免 HF 打 "Setting pad_token_id to
            # eos_token_id:151645 for open-end generation" 的 warning.
            gen_kwargs["pad_token_id"] = pad_id
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        scores0 = out.scores[0][0].float().cpu()
        try: del out
        except Exception: pass
        return scores0


class LlavaNextAdapter(BaseVLMAdapter):
    def load(self, model_path, device, max_pixels, **kw):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import LlavaNextForConditionalGeneration as _M
        except Exception:
            from transformers import AutoModelForImageTextToText as _M
        self.device = device
        try:
            self.model = _M.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2", device_map=device,
            )
        except Exception:
            self.model = _M.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=device,
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_inputs(self, messages):
        return _standard_build_inputs(self.processor, self.model, messages)

    def generate_first_token_logits(self, inputs):
        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        scores0 = out.scores[0][0].float().cpu()
        try: del out
        except Exception: pass
        return scores0


def make_adapter(model_type: str, model_path: str):
    t = model_type.lower()
    if t == "auto":
        s = (model_path or "").lower()
        if "qwen3" in s:
            t = "qwen3_vl"
        elif "qwen2" in s or "qwen-2" in s or "qwen_2" in s:
            t = "qwen2_5_vl"
        elif "internvl" in s:
            t = "internvl3"
        elif "llava" in s:
            t = "llava_next"
        else:
            t = "qwen2_5_vl"
    return {
        "qwen2_5_vl": Qwen25VLAdapter,
        "qwen3_vl":   Qwen3VLAdapter,
        "internvl3":  InternVL3Adapter,
        "llava_next": LlavaNextAdapter,
    }[t](), t


def _standard_build_inputs(processor, model, messages):
    try:
        orig_side = processor.tokenizer.padding_side
        processor.tokenizer.padding_side = "left"
    except Exception:
        orig_side = None
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    finally:
        if orig_side is not None:
            try: processor.tokenizer.padding_side = orig_side
            except Exception: pass
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    else:
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
    return inputs


# ==============================================================================
# Section 3. 数据合并 / 重新划分
# ==============================================================================
def load_and_merge(paths):
    merged = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"[warn] {path} not found, skipped")
            continue
        data = json.load(open(path))
        for r in data:
            k = (str(r["id"]), r["image_path"])
            if k not in merged:
                merged[k] = {
                    "id": str(r["id"]),
                    "image_path": r["image_path"],
                    "question": r["question"],
                    "labels": r.get("labels", ""),
                    "category": r.get("category", ""),
                }
    return list(merged.values())


def repartition(preds, out_dir):
    oc, ow = [], []
    for rec in preds:
        gt = rec.get("labels", "")
        sample = {
            "id": rec["id"],
            "question": rec["question"],
            "labels": gt,
            "image_path": rec["image_path"],
            "category": rec.get("category", ""),
        }
        ori_correct = (rec["ori_pred"] == gt)
        (oc if ori_correct else ow).append(sample)

    os.makedirs(out_dir, exist_ok=True)
    json.dump(oc, open(os.path.join(out_dir, "ori_correct.json"), "w"),
              ensure_ascii=False, indent=4)
    json.dump(ow, open(os.path.join(out_dir, "ori_wrong.json"), "w"),
              ensure_ascii=False, indent=4)
    print("\n" + "=" * 70)
    print(f"Repartition complete:")
    print(f"  ori_correct.json : {len(oc)}")
    print(f"  ori_wrong.json   : {len(ow)}")
    print(f"  → output dir: {out_dir}")
    print("=" * 70)


# ==============================================================================
# Section 4. 单样本推理 (仅一次 generate → CORE5)
# ==============================================================================
def infer_and_extract_one(sample, adapter: BaseVLMAdapter,
                          prompt_style: str):
    import numpy as np, torch
    import torch.nn.functional as F

    question = sample["question"]
    letters = parse_options(question)
    if prompt_style == "direct":
        instr = build_answer_instr_direct(letters)
    else:
        instr = build_answer_instr_tag(letters)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": sample["image_path"]},
            {"type": "text", "text": question + instr},
        ],
    }]
    inputs = adapter.build_inputs(messages)

    # 一次 generate → 首 token logits
    first_logits = adapter.generate_first_token_logits(inputs)  # torch [V]
    p_full = F.softmax(first_logits, dim=-1).cpu().numpy()
    logits_np = first_logits.cpu().numpy()

    # top-1 token (用于 ori_raw)
    top1_id = int(np.argmax(p_full))
    top1_text = adapter.tokenizer.decode([top1_id], skip_special_tokens=True).strip()

    # 选项 token ids
    opt_ids = []
    valid_letters = []
    for L in letters:
        tid = adapter.option_token_id(L)
        if tid is not None:
            opt_ids.append(tid)
            valid_letters.append(L)
    if len(opt_ids) < 2:
        raise RuntimeError(f"invalid option tokens: {letters}")

    # ── CORE5 特征 ──
    p_opt_raw = np.array([p_full[i] for i in opt_ids], dtype=np.float64)
    option_mass = float(p_opt_raw.sum())
    p_opt = p_opt_raw / max(p_opt_raw.sum(), 1e-12)
    sorted_p_opt = np.sort(p_opt)[::-1]
    answer_topp = float(sorted_p_opt[0])
    answer_margin = float(sorted_p_opt[0] - sorted_p_opt[1]) \
        if len(sorted_p_opt) > 1 else float(sorted_p_opt[0])

    vocab_full_entropy = float(-(p_full * np.log(p_full + 1e-12)).sum())
    V = p_full.size
    vocab_full_entropy_norm = vocab_full_entropy / max(math.log(V), 1e-12)

    max_opt_logit = float(max(logits_np[i] for i in opt_ids))
    mask = np.ones_like(logits_np, dtype=bool)
    for i2 in opt_ids:
        mask[i2] = False
    max_nonopt_logit = float(logits_np[mask].max())
    logit_gap_opt_nonopt = max_opt_logit - max_nonopt_logit

    # 决定 ori_letter
    valid_set = set(valid_letters)
    ori_letter = extract_letter(top1_text, valid_letters=valid_set)
    if not ori_letter or ori_letter not in valid_set:
        ori_letter = valid_letters[int(np.argmax(p_opt))]

    # 清理
    del inputs, first_logits
    try: torch.cuda.empty_cache()
    except Exception: pass

    return {
        "ori_letter": ori_letter,
        "ori_raw": top1_text,
        # CORE5
        "answer_topp": answer_topp,
        "answer_margin": answer_margin,
        "vocab_full_entropy_norm": vocab_full_entropy_norm,
        "option_mass": option_mass,
        "logit_gap_opt_nonopt": logit_gap_opt_nonopt,
    }


# ==============================================================================
# Section 5. 多卡 worker + 断点续跑
# ==============================================================================
def _load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    for line in open(path):
        try:
            r = json.loads(line)
            done.add((str(r["id"]), r["image_path"]))
        except Exception:
            continue
    return done


def run_worker(rank, world_size, gpu_id, samples, out_dir, args):
    import torch
    from tqdm import tqdm

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ori_preds_rank{rank}.jsonl")
    done = _load_done(out_path)

    print(f"[rank {rank}/{world_size}] loading {args.model_type} on cuda:{gpu_id} "
          f"({len(samples)} samples, {len(done)} resume)")

    adapter, resolved_t = make_adapter(args.model_type, args.model_path)
    adapter.load(args.model_path, device=device, max_pixels=args.max_pixels)
    print(f"[rank {rank}] adapter = {resolved_t}")

    fout = open(out_path, "a")
    ok = err = 0
    pbar = tqdm(samples, desc=f"[rank {rank}] {resolved_t}@gpu{gpu_id}",
                position=rank, leave=True, dynamic_ncols=True,
                initial=len(done), total=len(samples))
    for sample in pbar:
        k = (str(sample["id"]), sample["image_path"])
        if k in done:
            continue
        try:
            result = infer_and_extract_one(
                sample, adapter,
                prompt_style=args.prompt_style,
            )
            gt = sample.get("labels", "")
            rec = {
                # 元数据
                "id": str(sample["id"]),
                "image_path": sample["image_path"],
                "question": sample["question"],
                "labels": gt,
                "category": sample.get("category", ""),
                # relabel
                "ori_pred": result["ori_letter"],
                "ori_raw": result["ori_raw"],
                "ori_correct": bool(result["ori_letter"] == gt),
                # CORE5
                "answer_topp": result["answer_topp"],
                "answer_margin": result["answer_margin"],
                "vocab_full_entropy_norm": result["vocab_full_entropy_norm"],
                "option_mass": result["option_mass"],
                "logit_gap_opt_nonopt": result["logit_gap_opt_nonopt"],
                # 资源标记
                "model_type": resolved_t,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            ok += 1
            pbar.set_postfix(ok=ok, err=err,
                             acc=f"{ok/max(ok+err,1):.1%}")
        except Exception as e:
            err += 1
            pbar.set_postfix(ok=ok, err=err, last_err=str(e)[:30])
            import traceback
            traceback.print_exc()
            try: torch.cuda.empty_cache()
            except Exception: pass

    pbar.close()
    fout.close()
    adapter.close()
    print(f"[rank {rank}] DONE ok={ok} err={err}  → {out_path}")


# ==============================================================================
# Section 6. Main
# ==============================================================================
def build_features_train_jsonl(preds, out_path):
    """输出 train_router_v2.py 直接消费的 jsonl: label + CORE5 + id/image_path"""
    with open(out_path, "w") as fw:
        for r in preds:
            lab = "ori_correct" if r["ori_correct"] else "ori_wrong"
            rec = {
                "id": r["id"],
                "image_path": r["image_path"],
                "label": lab,
                "question": r["question"],
                # CORE5 (train_router_v2.py 的 DEFAULT_FEATS)
                "answer_topp": r.get("answer_topp"),
                "answer_margin": r.get("answer_margin"),
                "vocab_full_entropy_norm": r.get("vocab_full_entropy_norm"),
                "option_mass": r.get("option_mass"),
                "logit_gap_opt_nonopt": r.get("logit_gap_opt_nonopt"),
                "model_type": r.get("model_type"),
            }
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    DATA_DIR = "/path/to/data"

    # ── I/O ──
    p.add_argument("--sources", nargs="+", default=[
        f"{DATA_DIR}/vstar_bench2/vstar_bench.json",
        f"{DATA_DIR}/TreeBench/TreeBench.json",
        f"{DATA_DIR}/HR-Bench/hr_bench_8k.json",
        f"{DATA_DIR}/HR-Bench/hr_bench_4k.json",
    ], help="源数据集 JSON (将按 (id,image_path) 去重合并)")
    p.add_argument("--out_dir", default=f"{DATA_DIR}/relabeled_qwen3_lite",
                   help="输出目录")

    # ── 模型 ──
    p.add_argument("--model_type",
                   choices=["qwen2_5_vl", "qwen3_vl", "internvl3",
                            "llava_next", "auto"],
                   default="qwen2_5_vl")
    p.add_argument("--model_path",
                   default="/path/to/ckpt/Qwen3-VL-8B-Instruct")
    p.add_argument("--max_pixels", type=int, default=16384)
    p.add_argument("--prompt_style", choices=["direct", "tag"], default="direct")

    # ── 多卡 ──
    p.add_argument("--gpu_ids", default="0",
                   help="GPU 列表, 逗号分隔 (如 '0,1,2,3')")

    # ── 控制 ──
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--merge_only", action="store_true",
                   help="仅合并 preds_rank*.jsonl → 重新划分, 不做推理")
    p.add_argument("--skip_repartition", action="store_true")

    args = p.parse_args()

    # ── Step 1: 合并源数据集 ──
    print(f"[step 1] merging {len(args.sources)} source JSONs")
    samples = load_and_merge(args.sources)
    print(f"  unique samples: {len(samples)}")
    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"  limited to first {len(samples)} (debug)")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 2: 推理 + CORE5 特征 ──
    if not args.merge_only:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",") if g.strip()]
        world_size = len(gpu_ids)
        print(f"\n[step 2] launching lite inference on {world_size} GPU(s): {gpu_ids}")

        shards = [[] for _ in range(world_size)]
        for i, s in enumerate(samples):
            shards[i % world_size].append(s)
        for r, gid in enumerate(gpu_ids):
            print(f"  rank {r} (gpu {gid}): {len(shards[r])} samples")

        if world_size == 1:
            run_worker(0, 1, gpu_ids[0], shards[0], args.out_dir, args)
        else:
            import torch.multiprocessing as mp
            mp.set_start_method("spawn", force=True)
            procs = []
            for r, gid in enumerate(gpu_ids):
                proc = mp.Process(
                    target=run_worker,
                    args=(r, world_size, gid, shards[r], args.out_dir, args),
                )
                proc.start()
                procs.append(proc)
            for proc in procs:
                proc.join()
            print(f"\n[step 2] all workers finished")

    # ── Step 3: 合并 preds ──
    print(f"\n[step 3] merging preds from all ranks")
    pred_files = sorted(glob.glob(os.path.join(args.out_dir, "ori_preds_rank*.jsonl")))
    preds = []
    seen = set()
    for pf in pred_files:
        for line in open(pf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            k = (r["id"], r["image_path"])
            if k in seen:
                continue
            seen.add(k)
            preds.append(r)
    print(f"  total unique preds: {len(preds)}")

    all_preds_path = os.path.join(args.out_dir, "ori_preds_all.jsonl")
    with open(all_preds_path, "w") as fw:
        for r in preds:
            fw.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  merged → {all_preds_path}")

    n_oc = sum(1 for r in preds if r.get("ori_correct"))
    print(f"  ori_correct rate: {n_oc}/{len(preds)} = "
          f"{n_oc/max(len(preds),1):.2%}")

    # ── Step 3b: features_train_v2.jsonl ──
    feats_path = os.path.join(args.out_dir, "features_train_v2.jsonl")
    build_features_train_jsonl(preds, feats_path)
    print(f"  features_train_v2.jsonl → {feats_path}")

    # ── Step 4: repartition ──
    if args.skip_repartition:
        print("\n[step 4] --skip_repartition: done.")
        return

    print(f"\n[step 4] repartitioning into 2 JSONs")
    repartition(preds, args.out_dir)


# def _patch_internvl_typing():
#     """`Internvl/utiles_internvl.py` 顶部用 `from transformers.models.qwen2.modeling_qwen2 import *`
#     来获取 layer_forward / qwen2_forward 函数签名所需的 typing 与 transformers 内部符号
#     (Tuple / Optional / Cache / DynamicCache / FlashAttentionKwargs / Unpack /
#      BaseModelOutputWithPast / repeat_kv / apply_rotary_pos_emb /
#      eager_attention_forward / ALL_ATTENTION_FUNCTIONS / logger 等)。

#     新版 transformers 的 modeling_qwen2 不再把这些符号 re-export, 因此 utiles_internvl
#     在 import 时会抛 NameError。

#     本函数在不修改原文件的情况下, 把这些名字提前注入到 modeling_qwen2 模块的 globals,
#     确保后续 `from ... import *` 能拿到它们。任何注入失败的名字都被静默跳过, 因为
#     utiles_internvl 真正用到的只有签名 typing 和 layer_forward 内部那几个工具。
#     """
#     try:
#         from transformers.models.qwen2 import modeling_qwen2 as _M
#     except Exception:
#         return

#     # typing 标注 (NameError 的真正源头)
#     import typing as _typing
#     for _name in ("Tuple", "Optional", "Union", "Callable", "List"):
#         if not hasattr(_M, _name):
#             setattr(_M, _name, getattr(_typing, _name))

#     # transformers 内部符号 (best-effort, 跨版本路径不一)
#     def _try(attr, *importers):
#         if hasattr(_M, attr):
#             return
#         for imp in importers:
#             try:
#                 obj = imp()
#                 if obj is not None:
#                     setattr(_M, attr, obj)
#                     return
#             except Exception:
#                 continue

#     _try("Cache",
#          lambda: __import__("transformers.cache_utils", fromlist=["Cache"]).Cache)
#     _try("DynamicCache",
#          lambda: __import__("transformers.cache_utils",
#                             fromlist=["DynamicCache"]).DynamicCache)
#     _try("BaseModelOutputWithPast",
#          lambda: __import__("transformers.modeling_outputs",
#                             fromlist=["BaseModelOutputWithPast"]).BaseModelOutputWithPast)
#     _try("FlashAttentionKwargs",
#          lambda: __import__("transformers.modeling_flash_attention_utils",
#                             fromlist=["FlashAttentionKwargs"]).FlashAttentionKwargs)
#     _try("Unpack",
#          lambda: __import__("typing_extensions", fromlist=["Unpack"]).Unpack,
#          lambda: __import__("typing", fromlist=["Unpack"]).Unpack)
#     _try("logger",
#          lambda: __import__("transformers.utils.logging",
#                             fromlist=["get_logger"]).get_logger("modeling_qwen2"))

#     # 这几个一般 modeling_qwen2 自己就 export 了, 但兜底注入
#     _try("repeat_kv",
#          lambda: __import__("transformers.models.llama.modeling_llama",
#                             fromlist=["repeat_kv"]).repeat_kv)
#     _try("apply_rotary_pos_emb",
#          lambda: __import__("transformers.models.llama.modeling_llama",
#                             fromlist=["apply_rotary_pos_emb"]).apply_rotary_pos_emb)
#     _try("eager_attention_forward",
#          lambda: __import__("transformers.models.llama.modeling_llama",
#                             fromlist=["eager_attention_forward"]).eager_attention_forward)
#     _try("ALL_ATTENTION_FUNCTIONS",
#          lambda: __import__("transformers.modeling_utils",
#                             fromlist=["ALL_ATTENTION_FUNCTIONS"]).ALL_ATTENTION_FUNCTIONS)


if __name__ == "__main__":
    main()
