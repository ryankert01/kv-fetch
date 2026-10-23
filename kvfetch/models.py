"""Model / tokenizer loading and prompt formatting."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from kvfetch.tracing.tracer import TRACED_ATTN, register_traced_attention


@dataclass
class ModelDims:
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_position_embeddings: int


def model_dims(model) -> ModelDims:
    c = model.config
    return ModelDims(
        num_layers=c.num_hidden_layers,
        num_heads=c.num_attention_heads,
        num_kv_heads=c.num_key_value_heads,
        head_dim=getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads,
        max_position_embeddings=c.max_position_embeddings,
    )


def load_model(name: str, cfg: dict, device: str | torch.device, traced: bool = False,
               attn_implementation: str | None = None, config_overrides: dict | None = None):
    """Load (model, tokenizer). ``traced=True`` routes attention through the tracer
    (SDPA for the output, explicit fp32 softmax for the recorded probabilities).
    ``attn_implementation`` may also name an implementation registered by the caller."""
    mcfg = cfg["models"]
    if traced:
        register_traced_attention()
        attn = TRACED_ATTN
    else:
        attn = attn_implementation or mcfg["attn_implementation"]
    hf_config = AutoConfig.from_pretrained(name, local_files_only=mcfg["local_files_only"])
    for k, v in (config_overrides or {}).items():
        setattr(hf_config, k, v)
    model = AutoModelForCausalLM.from_pretrained(
        name, config=hf_config, dtype=getattr(torch, mcfg["dtype"]), attn_implementation=attn,
        local_files_only=mcfg["local_files_only"],
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(name, local_files_only=mcfg["local_files_only"])
    return model, tok


def stop_token_ids(name: str, tok, cfg: dict) -> list[int]:
    ids = set()
    try:
        g = GenerationConfig.from_pretrained(name, local_files_only=cfg["models"]["local_files_only"])
        e = g.eos_token_id
        ids.update(e if isinstance(e, list) else [e] if e is not None else [])
    except OSError:
        pass
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(ids)


def format_prompt(tok, user_text: str, assistant_prefix: str, cfg: dict) -> str:
    """Chat-template the user turn, open the assistant turn, append ``assistant_prefix``
    (RULER-style answer prefix). The template is rendered as text and tokenized with
    ``add_special_tokens=False`` so BOS appears exactly once."""
    kwargs = cfg["models"].get("chat_template_kwargs", {}) or {}
    head = tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                   add_generation_prompt=True, **kwargs)
    return head + assistant_prefix


def encode(tok, text: str, offsets: bool = False):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=offsets)
    return enc
