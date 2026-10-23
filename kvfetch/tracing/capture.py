"""Run one traced generation and package it for ``schema.save_trace``."""
from __future__ import annotations

import numpy as np
import torch

from kvfetch.models import model_dims
from kvfetch.tracing.decode import GenResult, generate
from kvfetch.tracing.tracer import AttentionTracer


def capture(model, input_ids: torch.Tensor, *, layers: list[int], cfg: dict, decode_steps: int,
            stop_ids: list[int], suppress_stop: bool, prefill_chunk_size: int | None = None,
            mass_dtype: str | None = None, followups: list[list[int]] | None = None,
            eot_ids: list[int] | None = None, turn_max_new: int | None = None) -> tuple[GenResult, AttentionTracer]:
    dims = model_dims(model)
    tcfg = cfg["trace"]
    tracer = AttentionTracer(
        layers=layers, num_heads=dims.num_heads, num_kv_heads=dims.num_kv_heads,
        page_size=cfg["definitions"]["page_size"], prompt_len=int(input_ids.numel()),
        decode_steps=decode_steps, obs_window=tcfg["prefill_obs_window"],
        flush_every=tcfg["flush_every"], device=input_ids.device,
        mass_dtype=mass_dtype or tcfg["mass_dtype"], head_dim=dims.head_dim,
        record_q=tcfg.get("record_q", False), head_reduce=tcfg.get("head_reduce", "none"),
    )
    with tracer:
        res, cache = generate(model, input_ids, max_new_tokens=decode_steps, stop_ids=stop_ids,
                              suppress_stop=suppress_stop, tracer=tracer,
                              prefill_chunk_size=prefill_chunk_size or cfg["models"]["prefill_chunk_size"],
                              followups=followups, eot_ids=eot_ids, turn_max_new=turn_max_new,
                              return_cache=True)
    if tcfg.get("key_summary", False):
        tracer.finalize_key_summary(cache)
    del cache
    if tracer.steps_recorded != len(res.tokens):
        raise RuntimeError(f"traced {tracer.steps_recorded} steps for {len(res.tokens)} tokens")
    return res, tracer


def trace_extras(tracer: AttentionTracer, res: GenResult) -> dict[str, np.ndarray]:
    """Optional tensors for ``save_trace(extra=...)``, trimmed to the recorded steps."""
    n = len(res.tokens)
    out = {"token_kind": np.array(res.kinds, dtype=np.int8)}
    if tracer.record_q:
        out["q"] = tracer.q[:n]
    if tracer.key_summary is not None:
        out["key_summary"] = tracer.key_summary
    return out


def trace_meta(*, prov: dict, tracer: AttentionTracer, res: GenResult, task: str, sample_id: str,
               **extra) -> dict:
    return {
        **prov,
        "task": task, "sample_id": sample_id,
        "layers": tracer.layers, "num_heads": tracer.num_heads, "num_kv_heads": tracer.num_kv_heads,
        "head_granularity": "query" if tracer.head_reduce == "none" else tracer.head_reduce,
        "prompt_len": tracer.prompt_len, "n_steps": len(res.tokens),
        "obs_window": tracer.obs_window,
        "natural_stop_step": res.natural_stop_step,
        "turns": res.turns,
        "prefill_s": res.prefill_s, "decode_s": res.decode_s,
        **extra,
    }
