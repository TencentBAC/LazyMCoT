import os
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from conversation import get_conv_template
import types
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
import json
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from collections import defaultdict
import matplotlib.patches as patches
from typing import Optional, Tuple, Union, Callable

SELECT_LAYER = [15]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    block_indices = [] # <--- 
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        block_indices.append([i % (target_width // image_size), i // (target_width // image_size)])
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        block_indices.append([-1,-1])
    return processed_images,block_indices

def load_image_from_path(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGBA')
    transform = build_transform(input_size=input_size)
    images,block_indices = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values,block_indices

def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images,block_indices = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values,block_indices

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            verbose=False):

    if history is None and pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    generation_config['eos_token_id'] = eos_token_id
    # "Setting pad_token_id to eos_token_id:151645 for open-end generation"。
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        pad_id = eos_token_id
    generation_config['pad_token_id'] = pad_id
    return pixel_values,input_ids,attention_mask,generation_config

def get_attention(model,
            pixel_values = None,
            input_ids = None,
            attention_mask = None,
            visual_features = None,
            output_hidden_states = None,
            target_indices = None):
    with torch.no_grad():
        assert model.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = model.extract_feature(pixel_values)
            input_embeds = model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = model.language_model.get_input_embeddings()(input_ids)
        outputs = model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                target_indices=target_indices,
            )
    return outputs

from transformers.models.qwen2.modeling_qwen2 import *  # noqa: F401,F403  (kept for backward compat)
import torch.nn.functional as F

# --- transformers.cache_utils: Cache / DynamicCache ---
from transformers.cache_utils import Cache, DynamicCache  # type: ignore

# --- transformers.modeling_outputs: BaseModelOutputWithPast ---
from transformers.modeling_outputs import BaseModelOutputWithPast  # type: ignore

try:
    from typing import Unpack  # type: ignore
except ImportError:  # pragma: no cover
    from typing_extensions import Unpack  # type: ignore

try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs  # type: ignore
except Exception:  # pragma: no cover
    try:
        from transformers.utils import FlashAttentionKwargs  # type: ignore
    except Exception:
        from typing import TypedDict
        class FlashAttentionKwargs(TypedDict, total=False):  # type: ignore
            pass

from transformers.utils import logging as _hf_logging  # type: ignore
logger = _hf_logging.get_logger(__name__)

from transformers.models.qwen2.modeling_qwen2 import (  # type: ignore
    apply_rotary_pos_emb,
    repeat_kv,
)

try:
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward  # type: ignore
except Exception:  # pragma: no cover
    from transformers.models.llama.modeling_llama import eager_attention_forward  # type: ignore

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # type: ignore
except Exception:  # pragma: no cover
    try:
        from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS  # type: ignore
    except Exception:
        ALL_ATTENTION_FUNCTIONS = {}  # type: ignore

def layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    past_key_values: Optional[Cache] = None, # 4.57 
    cache_position: Optional[torch.LongTensor] = None,
    target_indices=None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
     eager attention , transformers eager_attention_forward,
     4.57 / 4.58 chunked attn_weights GRACE 
    """
    if past_key_value is None and past_key_values is not None:
        past_key_value = past_key_values

    input_shape = hidden_states.shape[:-1]               # [B, Q]
    hidden_shape = (*input_shape, -1, self.head_dim)     # [B, Q, H, D]

    _hd = self.head_dim
    if hidden_states.dim() == 2:
        # [Q, H] -> [1, Q, H]
        hidden_states_4d = hidden_states.unsqueeze(0)
    else:
        hidden_states_4d = hidden_states
    bsz, q_len = hidden_states_4d.shape[0], hidden_states_4d.shape[1]

    _q = self.q_proj(hidden_states_4d)  # [B, Q, num_heads*D]
    _k = self.k_proj(hidden_states_4d)  # [B, Q, num_kv_heads*D]
    _v = self.v_proj(hidden_states_4d)  # [B, Q, num_kv_heads*D]
    query_states = _q.view(bsz, q_len, -1, _hd).transpose(1, 2)  # [B, Hq, Q, D]
    key_states   = _k.view(bsz, q_len, -1, _hd).transpose(1, 2)  # [B, Hkv, Q, D]
    value_states = _v.view(bsz, q_len, -1, _hd).transpose(1, 2)  # [B, Hkv, Q, D]

    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        if hasattr(self, "rotary_emb"):
            try:
                cos, sin = self.rotary_emb(value_states, position_ids)
            except Exception:
                cos, sin = self.rotary_emb(value_states, seq_len=value_states.shape[-2])
        else:
            raise RuntimeError("layer_forward: missing position_embeddings and rotary_emb")

    try:
        if cos.dim() == 4:
            from transformers.models.qwen2.modeling_qwen2 import rotate_half  # type: ignore
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states   = (key_states   * cos) + (rotate_half(key_states)   * sin)
        else:
            try:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            except TypeError:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)
    except Exception as _rope_e:
        print(f"[layer_forward RoPE FAIL] q.shape={query_states.shape} k.shape={key_states.shape} "
              f"cos.shape={getattr(cos,'shape',None)} sin.shape={getattr(sin,'shape',None)} "
              f"head_dim={self.head_dim} err={_rope_e}")
        raise

    if past_key_value is not None and hasattr(past_key_value, "update"):
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    #
    # ---------------------------------------------------------------------------
    scaling = getattr(self, "scaling", None)
    if scaling is None:
        scaling = 1.0 / math.sqrt(self.head_dim)

    Q_len = query_states.shape[-2]
    K_len = key_states.shape[-2]
    if Q_len == K_len:
        _is_causal = True
        _sdpa_mask = None
    elif Q_len == 1:
        _is_causal = False
        _sdpa_mask = None
    else:
        _is_causal = True
        _sdpa_mask = None

    try:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,                          # [B, Hq, Q, D]
            key_states,                            # [B, Hkv, K, D]
            value_states,                          # [B, Hkv, K, D]
            attn_mask=_sdpa_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=_is_causal,
            scale=scaling,
            enable_gqa=True,
        )
    except TypeError:
        _k_full = repeat_kv(key_states,   self.num_key_value_groups)
        _v_full = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, _k_full, _v_full,
            attn_mask=_sdpa_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=_is_causal,
            scale=scaling,
        )
        del _k_full, _v_full
    attn_output = attn_output.transpose(1, 2).contiguous()

    _need_attn = bool(getattr(self, "_need_attn_weights", False))
    if "output_attentions" in kwargs:
        _need_attn = bool(kwargs["output_attentions"])
    if _need_attn:
        def chunked_attention(query_states, key_states, head_dim, target_indices=None, chunk_size=512, attention_mask=None):
            """
             target_indices query token
            
            :
                query_states: [B, H, Q_LEN, D]
                key_states:   [B, H, K_LEN, D]
                head_dim:     D
                target_indices: Optional[List[int] or Tensor] attention query token 
                chunk_size: target token
                attention_mask: Optional[Tensor] [B, 1, Q_LEN, K_LEN] 
                
            :
                attn_weights: [B, H, Q_LEN, K_LEN] 0
            """
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            # query_states = query_states.permute(0, 2, 1, 3).contiguous()
            # key_states = key_states.permute(0, 2, 1, 3).contiguous()
            # print("query_states shape: ", query_states.shape)
            # print("key_states shape: ", key_states.shape)
            B, H, Q_LEN, D = query_states.shape
            _, _, K_LEN, _ = key_states.shape
            device = query_states.device
            dtype = query_states.dtype

            key_states = key_states.to(dtype)

            # print(B,H,Q_LEN,K_LEN)
            # attn_weights = torch.zeros(B, 1, Q_LEN,K_LEN,device=device, dtype=dtype)

            cpu_attn_weight = [[[None for _ in range(Q_LEN)]] for _ in range(B)]

            scale = 1.0 / math.sqrt(head_dim)

            if target_indices is None:
                target_indices = torch.arange(Q_LEN, device=device)
            else:
                if not isinstance(target_indices, torch.Tensor):
                    target_indices = torch.tensor(target_indices, dtype=torch.long, device=device)
                else:
                    target_indices = target_indices.to(device).long()

            num_targets = len(target_indices)
            # print(num_targets)
            selected_query_states = query_states.index_select(dim=2, index=target_indices)  # [B, H, num_targets, D]

            for i in range(0, num_targets, chunk_size):
                end_i = min(i + chunk_size, num_targets)
                current_indices = target_indices[i:end_i] # chunk 
                q_chunk = selected_query_states[:, :, i:end_i, :]  # [B, H, chunk_q, D]
                B,H,_,_ = q_chunk.shape
                attn_chunk = torch.zeros(B, 1, end_i - i, K_LEN, device=device, dtype=dtype)
                for h in range(H):
                    q_chunk_h = q_chunk[:, h, :, :][:,None]
                    key_states_h = key_states[:, h, :, :][:,None]
                    attn_chunk_h = torch.matmul(q_chunk_h, key_states_h.transpose(2, 3)) * scale  # [B, H, chunk_q, K_LEN]
                    if attention_mask is not None:
                        causal_mask = attention_mask.index_select(dim=2, index=current_indices)  # [B, 1, chunk_q, K_LEN]
                        attn_chunk_h += causal_mask

                    # Fix precision issues
                    if dtype in (torch.float16, torch.bfloat16):
                        attn_chunk_h = torch.where(torch.isinf(attn_chunk_h), torch.zeros_like(attn_chunk_h), attn_chunk_h)

                    attn_chunk_h = F.softmax(attn_chunk_h, dim=-1, dtype=torch.float32).to(dtype)
                    attn_chunk += attn_chunk_h
                    del attn_chunk_h
                attn_chunk_cpu = (attn_chunk/H).detach().cpu().float().numpy()  # [B, 1, chunk_q, K_LEN]
                # print("attn_chunk_cpu shape: ",attn_chunk_cpu.shape)
                del attn_chunk
                
                for b in range(B):
                    # print(end_i,i)
                    for j in range(end_i - i): # chunk index
                        q_idx = current_indices[j].item() # query token index
                        cpu_attn_weight[b][0][q_idx] = attn_chunk_cpu[b, 0, j] # numpy array

                torch.cuda.empty_cache()
            # cpu_attn_weight = attn_weights.cpu().float().numpy()
            # del attn_weights
            torch.cuda.empty_cache()
            return cpu_attn_weight
        # print(query_states.shape, key_states.shape)
        #B H Q_LEN K_LEN
        attn_weights = chunked_attention(query_states, key_states, self.head_dim, target_indices, chunk_size=128)
    else:
        attn_weights = None
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    self._last_attn_weights = attn_weights
    return attn_output, attn_weights

def qwen2_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        target_indices=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = None
        if hasattr(self, "_update_causal_mask"):
            try:
                causal_mask = self._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )
            except Exception:
                causal_mask = None
        if causal_mask is None:
            try:
                from transformers.masking_utils import create_causal_mask as _create_causal_mask  # 4.46+
            except Exception:
                try:
                    from transformers.models.qwen2.modeling_qwen2 import create_causal_mask as _create_causal_mask # 
                except Exception:
                    _create_causal_mask = None
            if _create_causal_mask is not None:
                try:
                    causal_mask = _create_causal_mask(
                        config=self.config,
                        input_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                    )
                except Exception:
                    causal_mask = None
                if causal_mask is not None and causal_mask.dim() != 4:
                    causal_mask = None
            if causal_mask is None:
                bsz, q_len = inputs_embeds.shape[0], inputs_embeds.shape[1]
                past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
                kv_len = past_len + q_len
                dtype = inputs_embeds.dtype
                device = inputs_embeds.device
                min_val = torch.finfo(dtype).min
                mask = torch.full((q_len, kv_len), min_val, dtype=dtype, device=device)
                i = torch.arange(q_len, device=device).unsqueeze(1)
                j = torch.arange(kv_len, device=device).unsqueeze(0)
                mask = mask.masked_fill(j <= (i + past_len), 0.0)
                causal_mask = mask[None, None, :, :].expand(bsz, 1, q_len, kv_len).contiguous()
                if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape[-1] == kv_len:
                    pad = (1.0 - attention_mask.to(dtype))[:, None, None, :] * min_val
                    causal_mask = causal_mask + pad
        # -----------------------------------------------------

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = ()
        compute_layer_index_sum = 0
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            compute_layer_index_sum += 1
            if compute_layer_index_sum in SELECT_LAYER and output_attentions:
                output_attentions_cp = True
            else:
                output_attentions_cp = False
            try:
                decoder_layer.self_attn._need_attn_weights = output_attentions_cp
            except Exception:
                pass
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions_cp,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    target_indices,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions_cp,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    target_indices=target_indices,
                    **flash_attn_kwargs,
                )

            if isinstance(layer_outputs, torch.Tensor):
                hidden_states = layer_outputs
                if output_attentions_cp:
                    _aw = getattr(decoder_layer.self_attn, "_last_attn_weights", None)
                    if _aw is not None:
                        all_self_attns += (_aw,)
            else:
                hidden_states = layer_outputs[0]
                if output_attentions_cp:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

def messages2out(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True):
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    for i in range(len(input_ids[0])):
        if tokenizer.decode(input_ids[0][i]) == "<|im_end|>":
            end_ques = i
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,history=None, return_history=True)
    return response,end_ques

def messages2att(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True):
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    img_start = []
    img_end = []
    idx2word_dicts = {}
    need_2_att_w = []
    split_nums = len(pixel_values)
    for i in range(len(input_ids[0])):
        words = tokenizer.decode(input_ids[0][i])
        idx2word_dicts[input_ids[0][i].cpu().item()] = words
        if input_ids[0][i].cpu().item() == 151665:
            all_img_start=i+1
        if input_ids[0][i].cpu().item() == 151666:
            all_img_end=i
    for i in range(len(input_ids[0])):
        if i>all_img_end:
            need_2_att_w.append(i)
    # print(all_img_start,all_img_end)
    img_start = list(range(all_img_start,all_img_end-256+1,256))
    img_end = list(range(all_img_start+256,all_img_end+1,256))
    out = get_attention(model, pixel_values, input_ids, attention_mask, target_indices=need_2_att_w)
    # print(out['attentions'])
    return out['attentions'],idx2word_dicts,img_start,img_end

from sklearn.cluster import DBSCAN
import base64
import io
import cv2
import PIL.Image as Image
from io import BytesIO

def image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"

def process_notsave(block_indices, start_k, end_k, attention, input_ids, img_url, img_start, img_end, sig):
    accept_att = {}
    noise_token_num = 10
    noise_mean = [[0 for k in range(noise_token_num-1)] for i in range(len(block_indices))]
    for k in range(start_k,end_k):
        if input_ids[0][k].cpu().item() >= 151643:
            continue
        max_att_mean = 0
        max_att_sum = 0
        for per in range(len(block_indices)):
            if len(block_indices[per]) == 1:
                start = [img_start[max_att_sum]]
                end = [img_end[max_att_sum]]
            else:
                start = img_start[max_att_sum: max_att_sum + len(block_indices[per]) - 1]
                end = img_end[max_att_sum: max_att_sum + len(block_indices[per]) - 1]
            max_att_sum += len(block_indices[per])
            layer_sum = []
            for i in range(len(attention)):
                block_sum = []
                for block in range(len(start)):
                    k_att_map = []
                    for row in attention[i][0]:
                        k_att_map.append(row[k])
                    k_att_map = np.array(k_att_map)
                    attention_map = k_att_map[:,start[block]:end[block]].reshape(-1,16,16).mean(axis=0)
                    block_sum.append(attention_map)
                # noise_mean = noise_mean/len(start)
                if len(block_indices[per]) == 1:
                    block_loc = [0,0]
                else:
                    block_loc = block_indices[per][-2]
                attention_map = np.zeros([(block_loc[1]+1)*16,(block_loc[0]+1)*16])
                for block in range(len(start)):
                    attention_map[block_indices[per][block][1]*16: (block_indices[per][block][1]+1)*16, block_indices[per][block][0]*16: (block_indices[per][block][0]+1)*16] = block_sum[block]
                layer_sum.append(attention_map)
            mean_layer_sum = np.array(layer_sum).mean(axis=0,keepdims=True)
            sum_per_img_att = mean_layer_sum.max()
            # print(sum_per_img_att)
            if max_att_mean < sum_per_img_att:
                max_att_mean = sum_per_img_att
                img_idx = per
                accept_att_map = mean_layer_sum
            if sig>0: mean_layer_sum = gaussian_filter(mean_layer_sum, sigma=sig)
            mean_layer_sum = mean_layer_sum - mean_layer_sum.min()
            mean_layer_sum = mean_layer_sum / mean_layer_sum.max()
            # print(k,start_k,end_k)
            if k < start_k+noise_token_num:
                noise_mean[per][start_k-k] = mean_layer_sum
        if k >= start_k+noise_token_num:
            # accept_att_map = noise_mean[per]/9
            if sig>0: accept_att_map = gaussian_filter(accept_att_map, sigma=sig)
            accept_att_map = accept_att_map - accept_att_map.min()
            accept_att_map = accept_att_map / accept_att_map.max()
            # print(np.array(noise_mean[img_idx]))
            accept_att_map = accept_att_map - np.array(noise_mean[img_idx]).mean(axis=0)
            accept_att_map[accept_att_map<0] = 0
            if accept_att_map.max() == 0: continue
            accept_att_map = accept_att_map - accept_att_map.min()
            accept_att_map = accept_att_map / accept_att_map.max()
            # accept_att_map = accept_att_map + noise_mean[img_idx].mean()/9
        else:
            continue
        if not img_idx in accept_att:
            accept_att[img_idx] = {}
        accept_att[img_idx][k]=accept_att_map
    return accept_att

def extract_cluster_boxes_normalized(
    attention_map, 
    eps_normalized=0.1, 
    min_samples=2):
    """
    :
        attention_map: numpy array (H, W) [0~1]
        eps_normalized: [0, 1]
        min_samples: 
    
    :
        cluster_boxes: [(x_min, y_min, x_max, y_max), ...]
    """
    H, W = attention_map.shape

    coords = np.column_stack(np.where(attention_map > 0.5)) # 

    if len(coords) == 0:
        return []

    diag_length = np.sqrt(H**2 + W**2)
    eps_actual = diag_length * eps_normalized # eps 

    clustering = DBSCAN(eps=eps_actual, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0) # 


    cluster_boxes = []
    for label in range(n_clusters):
        idxs = coords[labels == label]
        y_min, x_min = idxs.min(axis=0)
        y_max, x_max = idxs.max(axis=0)
        cluster_boxes.append((x_min, y_min, x_max+1, y_max+1))

    return cluster_boxes

def merge_duplicate_boxes_to_dict_avg(data, iou_threshold=0.5):
    """
     box seq_id
     box 
    
    :
        data: dict {seq_id: [box1, box2, ...]}
        iou_threshold: float
    
    :
        dict {final_seq_id: [merged_box1, merged_box2, ...]}
    """

    all_boxes = []
    for seq_id, boxes in data.items():
        for box in boxes:
            all_boxes.append((seq_id, box))

    parent = list(range(len(all_boxes)))

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    n = len(all_boxes)
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(all_boxes[i][1], all_boxes[j][1])
            if iou > iou_threshold:
                union(i, j)

    clusters = defaultdict(list)
    for idx, (seq_id, box) in enumerate(all_boxes):
        root = find(idx)
        clusters[root].append((seq_id, box))

    merged_results = []
    for cluster in clusters.values():
        final_seq_id = max(seq_id for seq_id, _ in cluster)
        boxes_in_cluster = [box for _, box in cluster]
        avg_box = average_boxes(boxes_in_cluster)
        merged_results.append((final_seq_id, avg_box))

    result = defaultdict(list)
    for seq_id, box in merged_results:
        result[seq_id].append(tuple(round(c, 6) for c in box)) # 

    for seq_id in result:
        seen = set()
        unique_boxes = []
        for box in result[seq_id]:
            t = tuple(box)
            if t not in seen:
                seen.add(t)
                unique_boxes.append(box)
        result[seq_id] = unique_boxes

    return dict(sorted(result.items()))

def Add_box_border(mbbox, radius=0.05):
    x0 = 0 if mbbox[0] - radius < 0 else mbbox[0] - radius
    y0 = 0 if mbbox[1] - radius < 0 else mbbox[1] - radius
    x1 = 1 if mbbox[2] + radius > 1 else mbbox[2] + radius
    y1 = 1 if mbbox[3] + radius > 1 else mbbox[3] + radius
    return (x0, y0, x1, y1)


from collections import defaultdict

def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_1, x2_1 = sorted([x1_1, x2_1])
    y1_1, y2_1 = sorted([y1_1, y2_1])
    x1_2, x2_2 = sorted([x1_2, x2_2])
    y1_2, y2_2 = sorted([y1_2, y2_2])

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou

def average_boxes(boxes):
    """
     bounding boxes box
    """
    n = len(boxes)
    sum_x1 = sum(box[0] for box in boxes)
    sum_y1 = sum(box[1] for box in boxes)
    sum_x2 = sum(box[2] for box in boxes)
    sum_y2 = sum(box[3] for box in boxes)

    avg_x1 = sum_x1 / n
    avg_y1 = sum_y1 / n
    avg_x2 = sum_x2 / n
    avg_y2 = sum_y2 / n

    return (avg_x1, avg_y1, avg_x2, avg_y2)

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

def swap_and_rebuild_dict(nested_dict):
    """
     key 
    
    :
        nested_dict: {outer_key: {inner_key: value}}
    :
        new_dict: {inner_key: {outer_key: value}}
    """
    new_dict = {}

    for outer_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            if inner_key not in new_dict:
                new_dict[inner_key] = {}
            new_dict[inner_key][outer_key] = value
            
    return dict(sorted(new_dict.items()))

def pil_to_base64(pil_img, format="PNG"):
    buffered = BytesIO()
    img_format = pil_img.format if pil_img.format else format
    pil_img.save(buffered, format=img_format) # 
    encoded_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded_str}"


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
                                          overlay_bboxes=None,
                                          draw_color_border=True,
                                          border_thickness=2):
    """
     BBox 

    GRACE / InternVL:
      - n (int): n BBox 
                 BBox 
      - overlay_bboxes (list[dict] | None):
             bbox {"bbox_norm": [x0,y0,x1,y1], "color": (R,G,B), "label": "entity name"}
             LPD /
      - draw_color_border (bool): overlay_bboxes
      - border_thickness (int): 

    :
      - `(pil_img, return_norm_bboxes)` 
        `(pil_img, return_norm_bboxes, used_colors_labels)`。
        baseline TAD`used_colors_labels` 
        

    Returns:
        (pil_result, return_norm_bboxes, used_colors_labels)
    """
    if isinstance(image, str):
        if image.startswith('data:image;base64,'):
            image64 = image.split(',')[1]
            image_data = base64.b64decode(image64)
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
        elif os.path.exists(image):
            pil_img = Image.open(image).convert("RGBA")
        else:
            image_data = base64.b64decode(image)
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGBA")
    else:
        pil_img = image
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

    masked_img_bgra = np.zeros_like(img_cv_bgra) 
    for x0, y0, x1, y1 in bboxes:
        x0_c, y0_c = max(0, x0), max(0, y0)
        x1_c, y1_c = min(img_w, x1), min(img_h, y1)
        if x0_c < x1_c and y0_c < y1_c:
            masked_img_bgra[y0_c:y1_c, x0_c:x1_c] = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
            
    masked_img_rgba = cv2.cvtColor(masked_img_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result_masked = Image.fromarray(masked_img_rgba)
    pil_result_masked.save(f"{img_nums}_{imgidx}_result_transparent_bg.png")

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
    # ori_area = img_w*img_h
    # return_bboxes = []
    for x0, y0, x1, y1 in bboxes:
        y0_c, y1_c = max(0, y0), min(img_h, y1)
        x0_c, x1_c = max(0, x0), min(img_w, x1)
        if y0_c >= y1_c or x0_c >= x1_c: continue
        # area = (x1-x0)*(y1-y0)
        # if area/ori_area > 0.25 or area/ori_area < 0.001: continue
        # return_bboxes.append([x0, y0, x1, y1])
        roi = img_cv_bgra[y0_c:y1_c, x0_c:x1_c]
        paste_x, paste_y = x_map[x0], y_map[y0]
        h, w, _ = roi.shape
        composite_image_bgra[paste_y : paste_y + h, paste_x : paste_x + w] = roi

    used_colors_labels = []
    _seen_color_label = set()
    if draw_color_border and overlay_bboxes:
        from PIL import ImageDraw
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
            px0 = int(bx[0] * img_w); py0 = int(bx[1] * img_h)
            px1 = int(bx[2] * img_w); py1 = int(bx[3] * img_h)
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
    pil_result_centered.save(f"{img_nums}_{imgidx}_result_transparent_bg_center.png")

    composite_image_rgba = cv2.cvtColor(composite_image_bgra, cv2.COLOR_BGRA2RGBA)
    pil_result = Image.fromarray(composite_image_rgba)

    up_sclae = 1
    new_size = (round(pil_result.width * up_sclae), round(pil_result.height * up_sclae))
    pil_result = pil_result.resize(new_size, Image.Resampling.BILINEAR)
    pil_result.save(f"{img_nums}_{imgidx}_result.png")
    
    return_norm_bboxes = []
    for x0, y0, x1, y1 in bboxes:
        return_norm_bboxes.append([x0/img_w, y0/img_h, x1/img_w, y1/img_h])
    return pil_result, return_norm_bboxes, used_colors_labels

def hot_attention_map_show(image, attention_map, bounding_boxes=None, alpha=0.5, save=None):
    """
    
     colorbar 0~1 0.5

    Args:
        image (np.array): (H, W, 3) [0,1] [0,255]
        attention_map (np.array): (h, w) [0,1]
        bounding_boxes (list, optional): 
             [x0, y0, x1, y1] attention_map 
             None
        alpha (float, optional): 0.5
        save (str, optional): None
    """
    img_height, img_width, _ = image.shape
    attn_height, attn_width = attention_map.shape

    zoom_h = img_height / attn_height
    zoom_w = img_width / attn_width

    attention_resized = zoom(attention_map, (zoom_h, zoom_w))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
    ax.imshow(image)

    cmap = 'hot'

    im = ax.imshow(attention_resized, cmap=cmap, alpha=alpha)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Attention Score', fontsize=16)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
    cbar.ax.tick_params(labelsize=14)

    if bounding_boxes:
        for box in bounding_boxes:
            x0, y0, x1, y1 = box
            scaled_x0 = x0 * zoom_w
            scaled_y0 = y0 * zoom_h
            scaled_x1 = x1 * zoom_w
            scaled_y1 = y1 * zoom_h
            width = scaled_x1 - scaled_x0
            height = scaled_y1 - scaled_y0

            rect = patches.Rectangle(
                (scaled_x0, scaled_y0),
                width,
                height,
                linewidth=8,
                edgecolor='lime', # 
                facecolor='none',
                linestyle='-',
                alpha=0.9
            )
            ax.add_patch(rect)

    ax.axis('off')

    fig.tight_layout(pad=0)

    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
    else:
        plt.show()

def from_img_and_att_get_cropbox(model, tokenizer, pixel_values, block_indices, question, generation_config,attention, idx2word_dicts, img_url, img_start, img_end, end_ques,sig,thre, history=None, return_history=True,
                                  enable_grace=False,
                                  sam3_supplement_bboxes_per_img=None,
                                  sam3_entity_labels_per_img=None,
                                  heatmap_save_dir=None,
                                  sample_id="sample",
                                  entity_text="",
                                  return_hide_copy=False):
    """ GRACE
      - sam3_supplement_bboxes_per_img: dict[img_idx] -> list[[x0,y0,x1,y1]] 
      - sam3_entity_labels_per_img: dict[img_idx] -> list[str] bbox 
      - heatmap_save_dir / sample_id: None 
          {sample_id}_img{img_idx}_s{sigma}_t{thresh}_agg_heatmap.png
      - entity_text: heatmap title
      - return_hide_copy: SAM3 overlay HiDe LPD 

    :
         return_hide_copy=False:
            img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes
        :
            img_merged_boxes, crop_list, words_lines, highlight_imgs, bounding_boxes, hide_highlight_imgs
    """
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    start_k = img_end[-1]
    for i in range(len(input_ids[0])):
        if tokenizer.decode(input_ids[0][i]) == "<|im_end|>":
            end_ques = i
    end_k = end_ques
    accept_att = process_notsave(block_indices, start_k, end_k, attention, input_ids, img_url, img_start, img_end, sig)

    if heatmap_save_dir is not None:
        try:
            os.makedirs(heatmap_save_dir, exist_ok=True)
            for _img_idx in accept_att:
                _agg = build_aggregated_heatmap(accept_att, _img_idx)
                if _agg is None:
                    continue
                _sam3 = (
                    sam3_supplement_bboxes_per_img.get(_img_idx, [])
                    if sam3_supplement_bboxes_per_img else []
                )
                _src = img_url[_img_idx] if _img_idx < len(img_url) else img_url[0]
                _fp = os.path.join(
                    heatmap_save_dir,
                    f"{sample_id}_img{_img_idx}_s{sig}_t{thre}_agg_heatmap.png",
                )
                save_attention_heatmap(
                    att_map=_agg,
                    image_any=_src,
                    save_path=_fp,
                    alpha=0.55, colormap="jet",
                    bboxes_norm=None,
                    sam3_bboxes_norm=_sam3 if _sam3 else None,
                    title=f"[{sample_id}] Aggregated Attn | s={sig} t={thre:.2f} | {entity_text[:60]}",
                )
        except Exception as _e:
            print(f"[heatmap] : {_e}")
    # print(accept_att)
    imgs_words_att_box = {}
    for img_idx in accept_att:
        # image = plt.imread(img_url[img_idx])
        accept_word_att = accept_att[img_idx]
        words_att_box = {}
        pil_img = np.array(img_url[img_idx])[:,:,-1]
        for word in accept_word_att:
            att_map = accept_word_att[word][0]
            # att_map = gaussian_filter(att_map, sigma=3)
            H,W = att_map.shape
            att_map = att_map * (cv2.resize(pil_img,(W,H))>127)
            # att_map = att_map-att_map.min()
            # att_map = att_map/att_map.max()
            # hot_attention_map_show(image,att_map,save=f'{img_idx}_{word}_{dicts[inputs["input_ids"][0][word].cpu().item()].replace(r"/",r"[]")}_{np.std(att_map):.06f}_{calculate_attention_entropy(att_map):.06f}.png')
            # threshold = np.percentile(att_map, 75)
            # att_map = np.where(att_map < threshold, 0, att_map)
            boxs, rigion_nums = find_top_n_attended_regions(att_map, 100, thre)
            # att_map[att_map<0.5] = 0
            # max_val_coords = np.unravel_index(np.argmax(att_map), att_map.shape)
            # hot_attention_map_show(image,att_map,save=f'{img_idx}_{word}_{dicts[inputs["input_ids"][0][word].cpu().item()].replace(r"/",r"[]")}_threshold_{np.std(att_map):.06f}_{calculate_attention_entropy(att_map):.06f}.png')
            # binarized_map = att_map >= 0.5
            # labeled_map = label(binarized_map, connectivity=2)
            # target_label = labeled_map[max_val_coords]
            # regions = regionprops(labeled_map)
            # for region in regions:
            #     if region.label == target_label:
                    # y0,x0,y1,x1 = region.bbox
                    # boxs = [[x0,y0,x1,y1]]
            # boxs = extract_cluster_boxes_normalized(att_map)
            # print(boxs)
            # total_attention = np.sum(att_map)
            img_height, img_width = att_map.shape
            # total_area = img_width * img_height
            # min_area = total_area*0.001  # 0.1%
            # max_area = total_area*0.25    # 1%
            save_boxs = []
            if boxs:
                H, W = att_map.shape
                words_att_box[word] = []
                for box in boxs:
                    x0,y0,x1,y1 = box
                    # if x0 == 0 and y0 == 0: continue
                    box_area = (x1 - x0) * (y1 - y0)
                    # if not (min_area <= box_area):
                    #     continue
                    bbox_norm = (x0 / W, y0 / H, x1 / W, y1 / H)
                    # Ambbox = Add_box_border(bbox_norm,radius=0.05)
                    Ambbox = bbox_norm
                    x0,y0,x1,y1 = Ambbox
                    # region = att_map[int(y0*H):int(y1*H)+1, int(x0*W):int(x1*W)+1]
                    # region_sum = np.sum(region)
                    # if region_sum > total_attention / 2:
                    words_att_box[word].append(Ambbox)
                    save_boxs.append(box)
            # image = cv2.resize(np.array(img_url[img_idx]), (np.array(img_url[img_idx]).shape[1] // 4, np.array(img_url[img_idx]).shape[0] // 4))
            # image = np.array(image)
            # hot_attention_map_show(image,att_map,save=f'{img_idx}_{word}_{tokenizer.decode(input_ids[0][word]).replace(r"/",r"[]")}.png',bounding_boxes=save_boxs)
        imgs_words_att_box[img_idx] = words_att_box

    for img_idx in imgs_words_att_box:
        max_word_idx = 0
        for words_idx in imgs_words_att_box[img_idx]:
            max_word_idx = max(max_word_idx,words_idx)

    # img_merged_boxes = {}
    # for img_idx in imgs_words_att_box:
    #     merged_boxes = imgs_words_att_box[img_idx]
    #     flag = True
    #     while flag:
    #         tmp_merged = merge_duplicate_boxes_to_dict_avg(merged_boxes)
    #         if tmp_merged == merged_boxes:
    #             flag = False
    #         merged_boxes = tmp_merged
    #     img_merged_boxes[img_idx] = merged_boxes
    # img_merged_boxes = swap_and_rebuild_dict(img_merged_boxes)
    img_merged_boxes = swap_and_rebuild_dict(imgs_words_att_box)

    words_lines = {}
    get_words = ""
    # print(start_k,end_k)
    for i in range(start_k,end_k):
        token_idx = input_ids[0][i].cpu().item()
        # print(i,dicts[token_idx],end="||")
        # if token_idx < 151643:
            # get_words+=dicts[token_idx]
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
    # print(img_merged_boxes)
    # print(words_lines)
    image_list = []
    for i in range(len(img_url)):
    #     if not img_url[i].startswith('data:image;base64,'):
    #         image64 = image_to_base64(img_url[i]).split(',')[1]
    #     elif ',' in img_url[i]:
    #         image64 = img_url[i].split(',')[1]
    #     image_data = base64.b64decode(image64)
        # image = np.array(Image.open(io.BytesIO(image_data)))
        image_list.append(img_url[i])
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
    for _imgidx in bounding_boxes:
        try:
            original_att_bboxes[_imgidx] = merge_overlapping_bboxes(
                [list(b) for b in bounding_boxes[_imgidx]]
            )
        except Exception:
            original_att_bboxes[_imgidx] = [list(b) for b in bounding_boxes[_imgidx]]

    if (enable_grace or return_hide_copy):
        _hide_bounds = {k: [list(b) for b in v] for k, v in bounding_boxes.items()}
        for _imgidx, _boxes in _hide_bounds.items():
            if not _boxes:
                continue
            _hi_img, _, _ = compact_and_center_with_relative_pos(
                _imgidx, len(img_url), img_url[_imgidx], _boxes,
                overlay_bboxes=None, draw_color_border=False,
            )
            if _hi_img is not None:
                hide_highlight_imgs.append(_hi_img)

    overlay_bboxes_per_img = {}
    extra_legend_entries = []
    if enable_grace and sam3_supplement_bboxes_per_img:
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
            for box_idx, orig_box in enumerate(sup_bboxes):
                label = sup_labels[box_idx] if box_idx < len(sup_labels) else ""
                color = label2color_global.get(
                    (label or "").strip(), _LEGEND_PALETTE[0]
                )
                sam3_area = (orig_box[2] - orig_box[0]) * (orig_box[3] - orig_box[1])
                if sam3_area > 0.80:
                    if label:
                        extra_legend_entries.append((color, f"{label} (skipped large)"))
                    continue
                overlay_bboxes_per_img.setdefault(sup_imgidx, []).append({
                    "bbox_norm": list(orig_box),
                    "color": color,
                    "label": label or "",
                })
                if sup_imgidx not in bounding_boxes:
                    bounding_boxes[sup_imgidx] = []
                bounding_boxes[sup_imgidx].append(list(orig_box))

    for imgidx in bounding_boxes:
        # highlight_imgs.append(blur_non_roi_base64(img_url[imgidx],bounding_boxes[imgidx]))
        if not bounding_boxes[imgidx]: continue
        _overlays = overlay_bboxes_per_img.get(imgidx) or None
        img, bboxs, used_colors_labels = compact_and_center_with_relative_pos(
            imgidx, len(img_url), img_url[imgidx], bounding_boxes[imgidx],
            overlay_bboxes=_overlays,
            draw_color_border=bool(_overlays),
        )
        bounding_boxes[imgidx] = bboxs
        if img is None:
            continue
        if enable_grace and (used_colors_labels or extra_legend_entries):
            _seen = set()
            merged_legend = []
            for cl in list(used_colors_labels) + extra_legend_entries:
                key = (tuple(cl[0]), cl[1])
                if key not in _seen and cl[1]:
                    _seen.add(key)
                    merged_legend.append(cl)
            if merged_legend:
                img = add_legend_to_lpd_image(img, merged_legend, position="bottom")
        highlight_imgs.append(img)
        # highlight_imgs.append(compact_and_center_with_relative_pos(imgidx,len(img_url),img_url[imgidx],bounding_boxes[imgidx]))
        # highlight_imgs.append(compact_and_center_with_relative_pos_in_ori(image_list[imgidx],bounding_boxes[imgidx]))

    if heatmap_save_dir is not None:
        try:
            for _img_idx in accept_att:
                _agg = build_aggregated_heatmap(accept_att, _img_idx)
                if _agg is None:
                    continue
                _att_bbs = original_att_bboxes.get(_img_idx, [])
                _sam3 = (
                    sam3_supplement_bboxes_per_img.get(_img_idx, [])
                    if sam3_supplement_bboxes_per_img else []
                )
                _src = img_url[_img_idx] if _img_idx < len(img_url) else img_url[0]
                _fp = os.path.join(
                    heatmap_save_dir,
                    f"{sample_id}_img{_img_idx}_s{sig}_t{thre}_agg_heatmap.png",
                )
                save_attention_heatmap(
                    att_map=_agg,
                    image_any=_src,
                    save_path=_fp,
                    alpha=0.55, colormap="jet",
                    bboxes_norm=_att_bbs if _att_bbs else None,
                    sam3_bboxes_norm=_sam3 if _sam3 else None,
                    title=f"[{sample_id}] Attn Heatmap | s={sig} t={thre:.2f} | {entity_text[:60]}",
                )
        except Exception as _e:
            print(f"[heatmap] : {_e}")

    if return_hide_copy or enable_grace:
        return img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes,hide_highlight_imgs
    return img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes

def find_top_n_attended_regions(att_map, n, threshold=0.5):
    """
    n

    
    1. 
    2. 
    3. 
    4. 
    5. nn

    :
    att_map (np.ndarray): 01
    n (int): 
    threshold (float, optional): 0.5

    :
    list: [x_min, y_min, x_max, y_max]
          
    """
    map_area = att_map.shape[0] * att_map.shape[1]
    binarized_map = att_map >= threshold
    if not np.any(binarized_map): # 
        return [],0
        
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
        # if box_area/map_area < 0.001:
        #     continue
        get_num += 1
        final_boxes.append([x0, y0, x1, y1])
        # if 0 == x0 and 0 == y0: return [],0

    # final_boxes = []
    # for region in sorted_regions:
    #     y0, x0, y1, x1 = region['bbox']
    #     final_boxes.append([x0, y0, x1, y1])

    return final_boxes, len(final_boxes)

def once_cot_infer(model,tokenizer,pixel_values,block_indices,question,generation_config, img_url,sig,thre):

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

#     prompt_ques = """You are an AI assistant for advanced, structured entity extraction. Your task is to identify key entities from a text (question and options) based on a hierarchical logic, and then format them according to specific rules.

# Part 1: Core Extraction Logic
# You must first determine the type of question to decide what to extract.

# Specific Inquiry: If the question asks about a specific, named entity (e.g., "Is there a red bicycle?"), you must only extract that entity from the question. Do not extract anything from the options in this case.

# General Inquiry with Context: If the question asks about a general placeholder entity with descriptive context (like location or attributes, e.g., "What is the object on the left?"), you must extract both:

# The general entity and its context from the question.
# All specific candidate entities from the meaningful options (e.g., "A. cat, B. dog").
# Pure General Inquiry: If the question is purely general without a useful placeholder (e.g., "Which is correct?", "What do you see?"), you must only extract the specific entities from the meaningful options.

# Exclusion Rule: Always ignore generic options like "Yes", "No", "True", "False", "All of the above", or "None of the above".

# Part 2: Formatting Rules
# After extracting entities, you must reformat them as follows:

# Adjective Formatting: If an entity has a descriptive adjective (e.g., "white rabbit"), reformat it as [Noun] with [Adjective].
# Example: white rabbit -> rabbit with white.
# Location Formatting: If an entity has a locational description (e.g., "object in the upper right corner"), reformat it as [Noun] on the [Location].
# Example: object in the upper right corner -> object on the upper right.
# Combined Formatting: If an entity has both, apply both rules.
# Example: blue car on the left -> car with blue on the left.
# Part 3: Output Format
# The final output must be a single string containing all processed and formatted entities, separated by commas.

# Examples
# Example 1: Specific Inquiry (Logic #1)

# Input Text:
# "Can you see a red bicycle in the picture? A. Yes, B. No"

# Expected Output:
# bicycle with red

# Example 2: General Inquiry with Context (Logic #2) - Your New Example

# Input Text:
# "What is the object in the upper right corner? A. A cat, B. A dog"

# Expected Output:
# object on the upper right, cat, dog

# Example 3: Pure General Inquiry (Logic #3)

# Input Text:
# "Based on the picture, which option is correct? A. There is a cat. B. There is a dog. C. There is a giraffe."

# Expected Output:
# cat, dog, giraffe

# Example 4: Combined Formatting

# Input Text:
# "What do you see in the image? A. A blue car on the left, B. A large house"

# Expected Output:
# car with blue on the left, house with large

# Example 5:

# Input Text:
# "What is the number of persons in the image?\n(A) 17\n(B) 14\n(C) 24\n(D) 13\n(E) The image does not feature the related information."

# Expected Output:
# persons

# Example 6:

# Input Text:
# "How many characters are there in the picture?\n(A) 2.\n(B) 3.\n(C) 4.\n(D) 1.\n(E) The image does not feature the related information."

# Expected Output:
# characters

# Example 7:

# Input Text:
# "What color is the shed on the right window of the house with solar panels on the roof in the left area of the picture?\n(A) Red\n(B) White\n(C) Green\n(D) Blue\n(E) This image doesn't feature the color."

# Expected Output:
# shed on the right window of the house with solar panels on the roof in the left area

# Now, process the following text directly:

# Input Text: """
    # prompt_ques += '\"'+question.split("<image>\n")[-1].replace("\nAnswer with the option's letter from the given choices directly.","")+'\"'+"\nExpected Output: "
    prompt_ques += question.split("<image>\n")[-1].split("\n")[0]+"\nAnswer: "
    prompt_output_text,end_ques = messages2out(model, tokenizer, None, prompt_ques, generation_config, history=None, return_history=True)
    # print(prompt_ques)
    # print(prompt_output_text)
    attention,idx2word_dicts,img_start,img_end = messages2att(model, tokenizer, pixel_values, "Search the following entities in the images: "+prompt_output_text, generation_config, history=None, return_history=True)  # Retrieve attention from model outputs
    img_merged_boxes,crop_list,words_lines,highlight_imgs,bounding_boxes = from_img_and_att_get_cropbox(model, tokenizer, pixel_values, block_indices, "Search the following entities in the images: "+prompt_output_text, generation_config,attention, idx2word_dicts, img_url, img_start, img_end, end_ques,sig,thre, history=None, return_history=True)
    # print(highlight_imgs)
    for h_img in highlight_imgs:
        # print(h_img)
        pixel_values_tmp,block_indices_tmp = load_image(h_img, max_num=128)
        pixel_values = torch.cat([pixel_values,pixel_values_tmp.to(torch.bfloat16).to(model.device)],dim=0)
        block_indices = block_indices + [block_indices_tmp]
    pixel_values,input_ids,attention_mask,generation_config = get_input(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    output_text,end_ques = messages2out(model, tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
    return output_text,crop_list,highlight_imgs,pixel_values,block_indices,words_lines,img_merged_boxes,bounding_boxes,prompt_output_text

def create_directory(path):
    """
    

    :param path: 
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created successfully at {path}")
    except Exception as e:
        print(f"Failed to create directory at {path}: {e}")

import jsonlines
def load_dataset_Vstar_json(path):
    Vstar_list = []
    with open(path, 'r', encoding='utf-8') as f:
        Vstar_list = json.load(f)
    mmetype_Vstarbench = []
    for i in range(len(Vstar_list)):
        dict_i = {}
        dict_i["id"] = Vstar_list[i]["id"]
        dict_i["Text"] = Vstar_list[i]["question"]
        # dict_i["Choices"] = "\n".join(Vstar_list[i]["text"].split("\n")[1:-1])
        dict_i["Ground truth"] = Vstar_list[i]["labels"]
        dict_i["image"] = Vstar_list[i]["image_path"]
        if "box_json" in Vstar_list[i]:
            dict_i["box_json"] = Vstar_list[i]["box_json"]
        dict_i["category"] = Vstar_list[i]["category"]
        mmetype_Vstarbench.append(dict_i)
    return mmetype_Vstarbench

def serialize_dict(my_dict, file_path):

    """
     JSON .jsonl 
    
     JSONL 
    
    :
        my_dict: ndarraynp.int64 
        file_path: .jsonl 
    """
    def serialize_obj(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: serialize_obj(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [serialize_obj(item) for item in obj]
        else:
            return obj

    serialized_dict = serialize_obj(my_dict)

    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(serialized_dict, ensure_ascii=False, indent=4) + '\n')


# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
import re as _re_grace
import math as _math_grace
from PIL import ImageDraw, ImageFont  # type: ignore
try:
    import requests as http_requests   # type: ignore
except Exception:   # pragma: no cover
    http_requests = None


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
    """ (label2color, bbox_colors)"""
    label2color = {}
    bbox_colors = []
    for lb in entity_labels:
        key = (lb or "").strip()
        if key not in label2color:
            label2color[key] = _LEGEND_PALETTE[len(label2color) % len(_LEGEND_PALETTE)]
        bbox_colors.append(label2color[key])
    return label2color, bbox_colors


def _get_pil_font(size: int):
    """ PIL """
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


def _ensure_pil_rgba(img_or_path):
    """InternVL PIL PIL / base64 / PIL(RGBA)"""
    if isinstance(img_or_path, Image.Image):
        return img_or_path.convert("RGBA") if img_or_path.mode != "RGBA" else img_or_path
    if isinstance(img_or_path, str):
        if img_or_path.startswith("data:image"):
            b64 = img_or_path.split(",", 1)[1]
            return Image.open(BytesIO(base64.b64decode(b64))).convert("RGBA")
        if os.path.exists(img_or_path):
            return Image.open(img_or_path).convert("RGBA")
        try:
            return Image.open(BytesIO(base64.b64decode(img_or_path))).convert("RGBA")
        except Exception:
            raise ValueError("[_ensure_pil_rgba] ")
    raise TypeError(f"[_ensure_pil_rgba] unsupported type: {type(img_or_path)}")


def pil_to_base64(pil_img, format="PNG"):
    """PIL → 'data:image;base64,...' SAM3 HTTP POST """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    pil_img.save(buf, format=format)
    return "data:image;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def add_legend_to_lpd_image(
    lpd_image,
    color_label_pairs,
    position="bottom",
    margin=6,
    swatch_size=18,
    font_size=None,
    background=(245, 245, 245),
    text_color=(20, 20, 20),
    border_color=(100, 100, 100),
):
    """ LPD ■ label

    :
        lpd_image: PIL.ImageInternVL LPD base64 
        color_label_pairs: list[(rgb_tuple, label_str)]
    :
        PIL.Image RGBA
    """
    if not color_label_pairs:
        return lpd_image
    try:
        main_img = _ensure_pil_rgba(lpd_image)
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
        items = []
        row_h = max(swatch_size, font_size) + 2
        for color, label in color_label_pairs:
            label_disp = str(label) if label is not None else ""
            tw, th = _text_size(label_disp)
            items.append((color, label_disp, tw, th))

        if position == "bottom":
            avail_w = mw - 2 * margin
            lines = []
            cur_line, cur_w = [], 0
            for it in items:
                item_w = swatch_size + 6 + it[2]
                need_w = item_w + (item_gap if cur_line else 0)
                if cur_line and cur_w + need_w > avail_w:
                    lines.append(cur_line); cur_line = [it]; cur_w = item_w
                else:
                    cur_line.append(it); cur_w += need_w
            if cur_line:
                lines.append(cur_line)
            legend_h = margin + row_h * len(lines) + margin
            new_w, new_h = mw, mh + legend_h
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
                    draw.rectangle([x_cur, sy, x_cur + sw, sy + sw],
                                   fill=tuple(int(c) for c in color), outline=border_color)
                    tx = x_cur + sw + 4
                    ty = y_top + (row_h - th) // 2
                    draw.text((tx, ty), label_disp, fill=text_color, font=font)
                    x_cur += sw + 6 + tw + item_gap
        else:
            item_ws = [swatch_size + 6 + it[2] for it in items]
            legend_w = margin + max(item_ws) + margin
            needed_h = margin + row_h * len(items) + margin
            new_w, new_h = mw + legend_w, max(mh, needed_h)
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
                draw.rectangle([x_cur, sy, x_cur + sw, sy + sw],
                               fill=tuple(int(c) for c in color), outline=border_color)
                tx = x_cur + sw + 4
                ty = y_top + (row_h - th) // 2
                draw.text((tx, ty), label_disp, fill=text_color, font=font)
        return new_img
    except Exception as e:
        print(f"[add_legend] : {e}")
        return lpd_image


def call_grounding_expert(image_pil_or_b64, entity_list,
                          expert_url="http://localhost:8002/predict",
                          box_threshold=0.3):
    """ SAM3 / Grounding DINO HTTP 

    :
        image_pil_or_b64: PIL.Image / 'data:image;base64,...' / 
        entity_list: list[str] 
        expert_url: expert_server/model_service 
    :
        dict[entity] -> list[[x0,y0,x1,y1]] 
    """
    if http_requests is None:
        print("[SAM3] requests ")
        return {e: [] for e in entity_list}
    try:
        pil = _ensure_pil_rgba(image_pil_or_b64)
        if pil.mode != "RGB":
            pil_rgb = pil.convert("RGB")
        else:
            pil_rgb = pil
        buf = BytesIO()
        pil_rgb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_w, img_h = pil_rgb.size
    except Exception as e:
        print(f"[SAM3] : {e}")
        return {e2.strip(): [] for e2 in entity_list}

    expert_results = {}
    for entity in entity_list:
        entity_clean = entity.strip()
        if not entity_clean:
            continue
        try:
            resp = http_requests.post(
                expert_url,
                json={"image": img_b64, "text": entity_clean},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                boxes = data.get("boxes", [])
                norm_boxes = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    nx1 = max(0.0, min(1.0, x1 / img_w))
                    ny1 = max(0.0, min(1.0, y1 / img_h))
                    nx2 = max(0.0, min(1.0, x2 / img_w))
                    ny2 = max(0.0, min(1.0, y2 / img_h))
                    if nx1 < nx2 and ny1 < ny2:
                        area = (nx2 - nx1) * (ny2 - ny1)
                        if area > 0.0005:
                            norm_boxes.append([nx1, ny1, nx2, ny2])
                expert_results[entity_clean] = norm_boxes
            else:
                expert_results[entity_clean] = []
        except Exception as e:
            print(f"  ⚠️ SAM3 call failed for '{entity_clean}': {e}")
            expert_results[entity_clean] = []
    return expert_results


def get_sam3_supplement_bboxes(sam3_results, max_per_entity=3):
    """ SAM3 (bboxes, labels) K """
    supplement_bboxes = []
    entity_labels = []
    for entity, boxes in sam3_results.items():
        if not boxes:
            continue
        sorted_boxes = sorted(
            boxes,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            reverse=True,
        )
        for b in sorted_boxes[:max_per_entity]:
            supplement_bboxes.append(b)
            entity_labels.append(entity)
    return supplement_bboxes, entity_labels


def filter_noise_bboxes(bboxes, min_area_ratio=0.001, edge_margin=0.02):
    """ / bbox"""
    if not bboxes:
        return bboxes
    filtered = []
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        area = (x1 - x0) * (y1 - y0)
        if area < min_area_ratio:
            continue
        is_corner = False
        corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for cx, cy in corners:
            if (abs(x0 - cx) < edge_margin or abs(x1 - cx) < edge_margin) and \
               (abs(y0 - cy) < edge_margin or abs(y1 - cy) < edge_margin):
                if area < 0.005:
                    is_corner = True
                    break
        if is_corner:
            continue
        filtered.append(bbox)
    return filtered if filtered else bboxes


def extract_answer_letter(text):
    """ (A-E)"""
    if not text:
        return None
    m = _re_grace.search(r"<FINAL_OUTPUT>\s*\(?([A-E])\)?\s*", text)
    if m:
        return m.group(1)
    ms = _re_grace.findall(r"\(([A-E])\)", text)
    if ms:
        return ms[-1]
    ms = _re_grace.findall(r"\b([A-E])\b", text)
    if ms:
        return ms[-1]
    tail = text.strip()
    if tail and tail[-1] in "ABCDE":
        return tail[-1]
    return None


def _bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def crop_image_region_b64(image_pil_or_b64, bbox_norm):
    """ 'data:image;base64,...' """
    try:
        pil = _ensure_pil_rgba(image_pil_or_b64)
        pil_rgb = pil.convert("RGB")
        W, H = pil_rgb.size
        x0 = max(0, int(bbox_norm[0] * W))
        y0 = max(0, int(bbox_norm[1] * H))
        x1 = min(W, int(bbox_norm[2] * W))
        y1 = min(H, int(bbox_norm[3] * H))
        if x0 >= x1 or y0 >= y1:
            return pil_to_base64(pil_rgb)
        cropped = pil_rgb.crop((x0, y0, x1, y1))
        return pil_to_base64(cropped)
    except Exception as e:
        print(f"[crop_region] : {e}")
        try:
            return pil_to_base64(_ensure_pil_rgba(image_pil_or_b64).convert("RGB"))
        except Exception:
            return ""


def secondary_sam3_check_on_att_bboxes(
    img_url, bounding_boxes, entity_list, sam3_url,
    max_per_entity=2, max_att_bboxes=4,
    existing_sam3_bboxes_per_img=None,
    iou_dedup_threshold=0.8,
    max_new_bboxes=3,
):
    """ bbox SAM3 

    : (found_any, new_entries_per_img)
        new_entries_per_img: dict[img_idx] -> list[(box_norm, label)]
    """
    found_any = False
    new_entries_per_img = {}
    if not entity_list or not sam3_url:
        return found_any, new_entries_per_img
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
            reverse=True,
        )[:max_att_bboxes]
        new_bboxes_orig = []
        new_labels_orig = []
        for att_bbox in sorted_att:
            x0_a, y0_a, x1_a, y1_a = att_bbox
            w_att = x1_a - x0_a
            h_att = y1_a - y0_a
            if w_att < 0.02 or h_att < 0.02:
                continue
            cropped_b64 = crop_image_region_b64(src_img, att_bbox)
            if not cropped_b64:
                continue
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
                            is_dup = True; break
                    if not is_dup:
                        for eb in new_bboxes_orig:
                            if _bbox_iou(new_box, eb) >= iou_dedup_threshold:
                                is_dup = True; break
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
    return found_any, new_entries_per_img


_OPT_PAT_V2 = _re_grace.compile(r"(?:^|\n|\s)\(?([A-Z])[\)\.\:]", _re_grace.MULTILINE)


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
    """v2 direct-answer 'Answer:' token = """
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
    """ token vocab logits Router v2 7 

    :
        dict None
    """
    if first_logits is None:
        return None
    try:
        logits = np.asarray(first_logits, dtype=np.float64)
        l = logits - logits.max()
        p_full = np.exp(l); p_full = p_full / max(p_full.sum(), 1e-12)

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
        answer_entropy_norm = answer_entropy / max(_math_grace.log(K), 1e-12)
        vocab_H = -float((p_full * np.log(p_full + 1e-12)).sum())
        V = p_full.size
        vocab_full_entropy_norm = vocab_H / max(_math_grace.log(V), 1e-12)
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


# ═══════════════════════════════════════════════════════════════════════════
#   3. *_lpd_out_*.png       GRACE LPD  (SAM3 overlay + legend)
# ═══════════════════════════════════════════════════════════════════════════

def build_aggregated_heatmap(accept_att: dict, img_idx: int):
    """ accept_att[img_idx] token , [0,1]

    accept_att : {img_idx: {token_k: att_map (H, W) (1, H, W)}}
    """
    if img_idx not in accept_att or not accept_att[img_idx]:
        return None
    maps = []
    for _k, att in accept_att[img_idx].items():
        if att is None:
            continue
        arr = np.asarray(att)
        if arr.ndim == 3:
            arr = arr[0]
        maps.append(arr)
    if not maps:
        return None
    base_shape = maps[0].shape
    uni = []
    for m in maps:
        if m.shape != base_shape:
            m = cv2.resize(m.astype(np.float32), (base_shape[1], base_shape[0]))
        uni.append(m)
    agg = np.mean(uni, axis=0)
    if agg.max() > agg.min():
        agg = (agg - agg.min()) / (agg.max() - agg.min())
    return agg


def _load_pil_any(image_any):
    """ PIL / base64 / PIL.Image (RGB)"""
    from PIL import Image as _Image
    if isinstance(image_any, _Image.Image):
        return image_any.convert("RGB")
    if isinstance(image_any, str):
        if image_any.startswith("data:image"):
            _b64 = image_any.split(",", 1)[1]
            return _Image.open(io.BytesIO(base64.b64decode(_b64))).convert("RGB")
        if os.path.exists(image_any):
            return _Image.open(image_any).convert("RGB")
        try:
            return _Image.open(io.BytesIO(base64.b64decode(image_any))).convert("RGB")
        except Exception:
            pass
    # numpy array
    if isinstance(image_any, np.ndarray):
        arr = image_any
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return _Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    raise ValueError(f"[_load_pil_any] : {type(image_any)}")


def save_attention_heatmap(
    att_map,
    image_any,
    save_path,
    alpha=0.55,
    colormap="jet",
    bboxes_norm=None,
    sam3_bboxes_norm=None,
    title="",
):
    """ jet , attention bbox, SAM3 bbox

    image_any PIL / base64 / / np.ndarray
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.patches as _mpatches

    try:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    except Exception:
        pass

    try:
        orig_img = _load_pil_any(image_any)
    except Exception as _e:
        print(f"[heatmap] : {_e}")
        return
    W, H = orig_img.size

    att_resized = np.array(
        Image.fromarray((np.clip(att_map, 0, 1) * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    cmap = _plt.get_cmap(colormap)
    heatmap_rgba = cmap(att_resized)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    orig_arr = np.array(orig_img, dtype=np.float32)
    heatmap_arr = heatmap_rgb.astype(np.float32)
    blended_arr = np.clip((1 - alpha) * orig_arr + alpha * heatmap_arr, 0, 255).astype(np.uint8)

    fig, ax = _plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(blended_arr)
    ax.axis("off")

    legend_patches = []
    if bboxes_norm:
        for box in bboxes_norm:
            x0n, y0n, x1n, y1n = box
            rect = _mpatches.Rectangle(
                (x0n * W, y0n * H), (x1n - x0n) * W, (y1n - y0n) * H,
                linewidth=2, edgecolor="#00ff00", facecolor="none",
            )
            ax.add_patch(rect)
        legend_patches.append(_mpatches.Patch(edgecolor="#00ff00", facecolor="none", label="Attention bbox"))

    if sam3_bboxes_norm:
        for box in sam3_bboxes_norm:
            x0n, y0n, x1n, y1n = box
            rect = _mpatches.Rectangle(
                (x0n * W, y0n * H), (x1n - x0n) * W, (y1n - y0n) * H,
                linewidth=2, edgecolor="#ff8800", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
        legend_patches.append(_mpatches.Patch(edgecolor="#ff8800", facecolor="none", linestyle="--", label="SAM3 bbox"))

    if legend_patches:
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
                  framealpha=0.7, facecolor="white")
    if title:
        ax.set_title(title, fontsize=9, pad=3)

    _plt.tight_layout(pad=0)
    try:
        _plt.savefig(save_path, bbox_inches="tight", dpi=100)
    except Exception as _e:
        print(f"[heatmap] {save_path}: {_e}")
    _plt.close(fig)


def save_pil_lpd(pil_or_b64, save_path):
    """ GRACE / HiDe LPD (PIL/base64/np.ndarray) PNG"""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    except Exception:
        pass
    try:
        im = _load_pil_any(pil_or_b64)
        im.save(save_path)
    except Exception as _e:
        print(f"[save_lpd] {save_path}: {_e}")

