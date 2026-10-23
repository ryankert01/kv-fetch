"""On-disk trace format: one safetensors file per sample.

Tensors
  format == "full":  mass      [T, L, H, P]  fp16   m(p, s, l, h)
  format == "topk":  topk_idx  [T, L, H, K]  uint16/int32 page index
                     topk_val  [T, L, H, K]  fp16   mass of those pages
                     residual  [T, L, H]     fp32   mass outside the top K
  always:            obs_mass  [W, L, H, P]  fp16   last W prompt queries (end-of-prefill signal)
                     tokens    [T]           int32  generated token ids (token s+1 is produced at step s)
                     prompt_ids [prompt_len] int32
  optional (extra):  token_kind [T]           int8   0 = generated, 1 = fed (follow-up user turn)
                     q          [T, L, H, d]  fp16   post-RoPE decode query (q[0] = last prompt token)
                     key_summary [L, 3, KVH, P, d] fp16  per-page post-RoPE key mean / min / max
Metadata: one JSON string under key "kvfetch" (provenance, layers, heads, spans, ...).
M_w is not stored: it is an exact window sum of m (see kvfetch.metrics.importance).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

SCHEMA_VERSION = 1
STEP_CONVENTION = ("step s in [0,T) is the attention of the query producing generated token s+1; "
                   "s=0 is the last prompt token; keys visible at step s: [0, prompt_len+s)")


@dataclass
class Trace:
    meta: dict
    obs_mass: np.ndarray
    tokens: np.ndarray
    prompt_ids: np.ndarray
    mass: np.ndarray | None = None          # dense [T, L, H, P]
    topk_idx: np.ndarray | None = None
    topk_val: np.ndarray | None = None
    residual: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.tokens.shape[0])

    def dense(self) -> np.ndarray:
        """Dense m. For top-K traces, mass outside the top K is dropped (see ``residual``)."""
        if self.mass is not None:
            return self.mass
        T, L, H, K = self.topk_idx.shape
        out = np.zeros((T, L, H, self.meta["n_pages"]), dtype=self.topk_val.dtype)
        np.put_along_axis(out, self.topk_idx.astype(np.int64), self.topk_val, axis=-1)
        return out


def to_topk(mass: np.ndarray, k: int) -> dict[str, np.ndarray]:
    P = mass.shape[-1]
    k = min(k, P)
    idx = np.argpartition(-mass.astype(np.float32), k - 1, axis=-1)[..., :k]
    val = np.take_along_axis(mass, idx, axis=-1)
    residual = mass.astype(np.float32).sum(-1) - val.astype(np.float32).sum(-1)
    idx_dtype = np.uint16 if P <= np.iinfo(np.uint16).max + 1 else np.int32
    return {"topk_idx": idx.astype(idx_dtype), "topk_val": val,
            "residual": np.maximum(residual, 0).astype(np.float32)}


def save_trace(path: str | Path, *, meta: dict, mass: np.ndarray, obs_mass: np.ndarray,
               tokens: np.ndarray, prompt_ids: np.ndarray, fmt: str = "full", topk: int = 512,
               extra: dict[str, np.ndarray] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, schema_version=SCHEMA_VERSION, format=fmt, step_convention=STEP_CONVENTION,
                n_pages=int(mass.shape[-1]), shape_TLHP=list(mass.shape))
    tensors = {"obs_mass": np.ascontiguousarray(obs_mass), "tokens": tokens.astype(np.int32),
               "prompt_ids": prompt_ids.astype(np.int32)}
    if fmt == "full":
        tensors["mass"] = np.ascontiguousarray(mass)
    elif fmt == "topk":
        tensors.update(to_topk(mass, topk))
        meta["topk"] = int(tensors["topk_idx"].shape[-1])
    else:
        raise ValueError(fmt)
    for k, v in (extra or {}).items():
        if k in tensors:
            raise ValueError(f"extra tensor {k!r} clashes with a schema tensor")
        tensors[k] = np.ascontiguousarray(v)
    save_file(tensors, str(path), metadata={"kvfetch": json.dumps(meta)})
    return path


def load_meta(path: str | Path) -> dict:
    with safe_open(str(path), framework="numpy") as f:
        return json.loads(f.metadata()["kvfetch"])


def load_trace(path: str | Path) -> Trace:
    with safe_open(str(path), framework="numpy") as f:
        meta = json.loads(f.metadata()["kvfetch"])
        t = {k: f.get_tensor(k) for k in f.keys()}
    return Trace(meta=meta, obs_mass=t.pop("obs_mass"), tokens=t.pop("tokens"),
                 prompt_ids=t.pop("prompt_ids"), mass=t.pop("mass", None),
                 topk_idx=t.pop("topk_idx", None), topk_val=t.pop("topk_val", None),
                 residual=t.pop("residual", None), extra=t)
