"""Importance quantities (spec §3) on arrays shaped [T, S, P]:
T decode steps, S independent series (layer x head, flattened), P pages.

Index conventions
  m[t]        attention mass at step t (see tracing.schema.STEP_CONVENTION)
  M_w[t]      sum_{s=t+1}^{t+w} m[s]; defined for t in [0, T-w)
  score[t]    a predictor's score using information up to and including step t
"""
from __future__ import annotations

import math

import numpy as np


def future_mass(m: np.ndarray, w: int) -> np.ndarray:
    """M_w[t] = sum_{s=t+1}^{t+w} m[s] for t in [0, T-w). Accumulates in fp32."""
    T = m.shape[0]
    if T <= w:
        raise ValueError(f"need more than w={w} steps, got {T}")
    c = np.cumsum(m.astype(np.float32), axis=0)
    return c[w:] - c[:-w]  # row t: c[t+w] - c[t] = sum m[t+1..t+w]


def eligible_pages(prompt_len: int, n_pages: int, page_size: int, pinned: dict) -> np.ndarray:
    """Pages subject to the tiering decision: full prompt pages minus pinned ones."""
    if not pinned.get("decode_region", True):
        raise NotImplementedError("unpinned decode region (time-varying eligibility) not supported")
    n_full = prompt_len // page_size  # the partial last prompt page holds generated tokens -> pinned
    mask = np.zeros(n_pages, dtype=bool)
    lo = pinned.get("sink_pages", 0)
    hi = n_full - pinned.get("recent_prompt_pages", 0)
    mask[lo:max(lo, hi)] = True
    return mask


def budget_pages(frac: float, n_eligible: int, rounding: str = "ceil") -> int:
    x = frac * n_eligible
    b = math.ceil(x) if rounding == "ceil" else math.floor(x) if rounding == "floor" else round(x)
    return max(1, min(int(b), n_eligible))


def topb_mask(score: np.ndarray, B: int, eligible: np.ndarray) -> np.ndarray:
    """Boolean [..., P] mask of the B highest-scoring eligible pages (ties: arbitrary)."""
    s = np.where(eligible, score.astype(np.float32), -np.inf)
    idx = np.argpartition(-s, B - 1, axis=-1)[..., :B]
    out = np.zeros(score.shape, dtype=bool)
    np.put_along_axis(out, idx, True, axis=-1)
    return out


def topb_cutoff(score: np.ndarray, B: int, eligible: np.ndarray) -> np.ndarray:
    """Value of the B-th largest eligible score, shape [...]."""
    s = score[..., eligible].astype(np.float32)
    return np.partition(s, s.shape[-1] - B, axis=-1)[..., s.shape[-1] - B]


def head_view(m: np.ndarray, view: str, num_kv_heads: int) -> np.ndarray:
    """[T, L, H, P] query-head trace -> requested view. ``kv_mean`` averages each GQA
    group (query head h reads KV head h // G), which is the granularity tiers act on."""
    if view == "query":
        return m
    T, L, H, P = m.shape
    g = H // num_kv_heads
    grouped = m.reshape(T, L, num_kv_heads, g, P).astype(np.float32)
    if view == "kv_mean":
        return grouped.mean(3)
    if view == "kv_max":
        return grouped.max(3)
    raise ValueError(view)


def flatten_series(x: np.ndarray) -> np.ndarray:
    """[T, L, H, P] -> [T, L*H, P]."""
    return x.reshape(x.shape[0], -1, x.shape[-1])
