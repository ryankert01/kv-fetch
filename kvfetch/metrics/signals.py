"""Cheap signals that could predict page importance, evaluated on stored traces.

Every signal produces score[t] [H, P] from information available at step t: past attention,
the current (or an extrapolated) query, and per-page key summaries. Query heads read their
GQA group's key summary. ``bytes_per_page`` is the per-page, per-KV-head storage the signal
needs on the GPU (fp16), the cost axis of spec Phase 2.
"""
from __future__ import annotations

import numpy as np

from kvfetch.metrics import predictors
from kvfetch.metrics.importance import topb_mask_var

KEY_MEAN, KEY_MIN, KEY_MAX = 0, 1, 2


def _expand(ks: np.ndarray, num_heads: int) -> np.ndarray:
    """[KVH, P, d] -> [H, P, d] (query head h reads KV head h // G)."""
    return np.repeat(ks, num_heads // ks.shape[0], axis=0)


def extrapolate_q(q: np.ndarray, k: int, j: int) -> np.ndarray:
    """q_hat[t] = q[t] + (k / j) * (q[t] - q[t - j]): linear extrapolation k steps ahead from a
    j-step slope (q[t] itself for t < j)."""
    out = q.astype(np.float32).copy()
    out[j:] += (k / j) * (q[j:].astype(np.float32) - q[:-j].astype(np.float32))
    return out


def signal_scores(name: str, *, m: np.ndarray, q: np.ndarray, key_summary: np.ndarray, scaling: float,
                  sc: dict, k: int) -> np.ndarray:
    """score [T, H, P] for signal ``name``. m [T, H, P]; q [T, H, d]; key_summary [3, KVH, P, d]."""
    H = m.shape[1]
    if name == "attn_last":
        return predictors.last_step(m)
    if name == "attn_ema":
        return predictors.ema(m, sc["ema_alpha"])
    kmean = _expand(key_summary[KEY_MEAN].astype(np.float32), H)
    if name == "key_mean":
        return np.einsum("thd,hpd->thp", q.astype(np.float32), kmean) * scaling
    if name == "quest":
        # Quest upper bound sum_d max(q_d kmax_d, q_d kmin_d) = q+ . kmax + q- . kmin (kmax >= kmin)
        kmin = _expand(key_summary[KEY_MIN].astype(np.float32), H)
        kmax = _expand(key_summary[KEY_MAX].astype(np.float32), H)
        qf = q.astype(np.float32)
        return (np.einsum("thd,hpd->thp", np.maximum(qf, 0), kmax)
                + np.einsum("thd,hpd->thp", np.minimum(qf, 0), kmin)) * scaling
    if name == "q_trend":
        qh = extrapolate_q(q, k, sc["q_slope_steps"])
        return np.einsum("thd,hpd->thp", qh, kmean) * scaling
    raise ValueError(name)


def bytes_per_page(name: str, head_dim: int) -> int:
    """Per page, per KV head, fp16 on the GPU (attention signals: one score per query head
    in the group is reported as one value per KV head for comparability)."""
    return {"attn_last": 2, "attn_ema": 2, "key_mean": 2 * head_dim, "quest": 4 * head_dim,
            "q_trend": 2 * head_dim}[name]


def event_recall_from_mask(mask: np.ndarray, events: tuple[np.ndarray, ...], k: int) -> float:
    """Share of events (t, h, p) whose page was in ``mask`` (a signal's top-B, [T, H, P]) at
    step t - k. Events with t < k are skipped. A random predictor scores ~frac."""
    t, h, p = events
    ok = t >= k
    return float(mask[t[ok] - k, h[ok], p[ok]].mean()) if ok.any() else float("nan")


def mass_recall_from_masks(pred: np.ndarray, best: np.ndarray, m: np.ndarray, eligible: np.ndarray,
                           k: int) -> float:
    """Mean over (t, h) of m[t+k] mass on pred[t] / m[t+k] mass on its own top-B best[t+k]."""
    T = m.shape[0]
    if T <= k:
        return float("nan")
    truth = m[k:].astype(np.float32) * eligible[k:, None, :]
    num = (truth * pred[:T - k]).sum(-1)
    den = (truth * best[k:]).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.where(den > 0, num / den, np.nan)))


def event_recall(score: np.ndarray, eligible: np.ndarray, frac: float, events: tuple[np.ndarray, ...],
                 k: int, rounding: str = "ceil") -> float:
    mask, _, _ = topb_mask_var(score, frac, eligible, rounding)
    return event_recall_from_mask(mask, events, k)


def mass_recall_k(score: np.ndarray, m: np.ndarray, eligible: np.ndarray, frac: float, k: int,
                  rounding: str = "ceil") -> float:
    pred, _, _ = topb_mask_var(score, frac, eligible, rounding)
    best, _, _ = topb_mask_var(m.astype(np.float32) * eligible[:, None, :], frac, eligible, rounding)
    return mass_recall_from_masks(pred, best, m, eligible, k)
