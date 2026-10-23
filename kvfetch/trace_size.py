"""Trace size estimator (spec §7 Phase 0, first item).

Dense ("full") format stores m(p,t,l,h) for every page in the max page index space
P = ceil((ctx + T) / page_size). The top-K format stores, per (t,l,h), K page
indices + K masses + one residual (mass outside the top K).

M_w is not stored in the dense format: it is an exact window sum of m and is
derived at analysis time. In the top-K format it is not exactly derivable (pages
that leave the top K lose their mass), so the top-K rows also report the cost of
storing a separate top-K of M_w.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


@dataclass
class SizeRow:
    scenario: str
    page_size: int
    head_kind: str
    heads: int
    layers: int
    ctx: int
    decode_steps: int
    samples: int
    pages: int
    dense_bytes_per_sample: int
    obs_bytes_per_sample: int
    topk: int
    topk_bytes_per_sample: int
    topk_with_mw_bytes_per_sample: int
    max_budget_pages: int
    topk_covers_max_budget: bool

    def as_dict(self) -> dict:
        d = asdict(self)
        d["dense_total_bytes"] = (self.dense_bytes_per_sample + self.obs_bytes_per_sample) * self.samples
        d["topk_total_bytes"] = (self.topk_bytes_per_sample + self.obs_bytes_per_sample) * self.samples
        d["topk_with_mw_total_bytes"] = (self.topk_with_mw_bytes_per_sample + self.obs_bytes_per_sample) * self.samples
        return d


def index_bytes(n_pages: int) -> int:
    return 2 if n_pages <= np.iinfo(np.uint16).max + 1 else 4


def estimate(*, scenario: str, ctx: int, decode_steps: int, layers: int, heads: int, head_kind: str,
             samples: int, page_size: int, topk: int, mass_dtype: str, residual_bytes: int,
             obs_window: int, max_budget_frac: float) -> SizeRow:
    mb = _DTYPE_BYTES[mass_dtype]
    pages = math.ceil((ctx + decode_steps) / page_size)
    series = decode_steps * layers * heads
    dense = series * pages * mb
    obs = obs_window * layers * heads * pages * mb  # obs window is always stored dense
    k = min(topk, pages)
    topk_row = k * (mb + index_bytes(pages)) + residual_bytes
    max_budget_pages = math.ceil(max_budget_frac * (ctx // page_size))
    return SizeRow(
        scenario=scenario, page_size=page_size, head_kind=head_kind, heads=heads, layers=layers,
        ctx=ctx, decode_steps=decode_steps, samples=samples, pages=pages,
        dense_bytes_per_sample=dense, obs_bytes_per_sample=obs, topk=k,
        topk_bytes_per_sample=series * topk_row,
        topk_with_mw_bytes_per_sample=2 * series * topk_row,
        max_budget_pages=max_budget_pages,
        topk_covers_max_budget=k >= max_budget_pages,
    )


def resolve_model_dims(model_name: str, local_files_only: bool) -> dict:
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    return {
        "layers": c.num_hidden_layers,
        "heads": {"query": c.num_attention_heads, "kv": c.num_key_value_heads},
    }


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    raise AssertionError


def estimate_all(cfg: dict) -> list[SizeRow]:
    ts = cfg["trace_size"]
    rows = []
    for sc in ts["scenarios"]:
        layers, heads = sc.get("layers"), sc.get("heads")
        if "model" in sc:
            dims = resolve_model_dims(sc["model"], cfg["models"]["local_files_only"])
            layers = dims["layers"] if layers == "all" else layers
            heads = heads or dims["heads"]
        for ps in ts["page_sizes"]:
            for kind, h in heads.items():
                rows.append(estimate(
                    scenario=sc["name"], ctx=sc["ctx"], decode_steps=sc["decode_steps"],
                    layers=layers, heads=h, head_kind=kind, samples=sc["samples"], page_size=ps,
                    topk=ts["topk"], mass_dtype=cfg["trace"]["mass_dtype"],
                    residual_bytes=ts["residual_bytes"], obs_window=cfg["trace"]["prefill_obs_window"],
                    max_budget_frac=max(cfg["definitions"]["budget_fracs"]),
                ))
    return rows
