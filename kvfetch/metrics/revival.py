"""Revival: a page is important, goes dormant, and becomes important again.

All functions take the instantaneous mass m [T, S, P] (S series = heads of one layer, or
KV groups) and a per-step eligibility mask [T, P] (see importance.eligibility_matrix).
"High" uses the trajectory hysteresis: enter when m reaches the top-B cutoff of that step,
leave when it falls below exit_ratio of the cutoff. Steps where a page is not eligible
count as not-high, so a page that ages into eligibility starts with an onset, never a
revival.
"""
from __future__ import annotations

import numpy as np

from kvfetch.metrics.importance import budget_pages, topb_mask_var
from kvfetch.metrics.trajectory import high_state

GAP_BINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 1 << 30]


def high_matrix(m: np.ndarray, eligible: np.ndarray, budget_frac: float, exit_ratio: float,
                rounding: str = "ceil") -> np.ndarray:
    """[T, S, P] bool high-state with hysteresis, relative to each step's top-B cutoff."""
    _, cut, _ = topb_mask_var(m, budget_frac, eligible, rounding)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((cut[..., None] > 0) & eligible[:, None, :], m / cut[..., None], 0.0)
    T, S, P = r.shape
    return high_state(r.reshape(T, S * P), exit_ratio).reshape(T, S, P)


def reentries(high: np.ndarray):
    """For high [T, N]: (reentry [T, N] bool, gap [T, N] int).
    reentry[t] marks an out->in transition at t for a column that was high before;
    gap[t] is the number of not-high steps since it was last high (valid where reentry)."""
    T, N = high.shape
    steps = np.arange(T)[:, None]
    last_high = np.maximum.accumulate(np.where(high, steps, -1), axis=0)
    prev_last = np.vstack([np.full((1, N), -1), last_high[:-1]])
    prev_high = np.vstack([np.zeros((1, N), bool), high[:-1]])
    reentry = high & ~prev_high & (prev_last >= 0)
    gap = steps - prev_last - 1
    return reentry, gap


def revived_mask(high: np.ndarray, reentry: np.ndarray, gap: np.ndarray, D: int) -> np.ndarray:
    """[T, N]: high steps that belong to an in-run which started as a revival (gap >= D)."""
    T, N = high.shape
    steps = np.arange(T)[:, None]
    entry = high & ~np.vstack([np.zeros((1, N), bool), high[:-1]])
    last_entry = np.maximum.accumulate(np.where(entry, steps, -1), axis=0)
    is_rev = reentry & (gap >= D)
    flag = np.take_along_axis(is_rev, np.clip(last_entry, 0, None), axis=0) & (last_entry >= 0)
    return high & flag


def revival_stats(m: np.ndarray, eligible: np.ndarray, rc: dict, rounding: str = "ceil") -> dict:
    """Per-sample revival statistics for one layer/view. ``rc`` is the ``revival`` config."""
    T, S, P = m.shape
    high = high_matrix(m, eligible, rc["budget_frac"], rc["exit_ratio"], rounding).reshape(T, S * P)
    reentry, gap = reentries(high)
    ever = high.any(0)
    gaps = gap[reentry]
    out = {"n_steps": T, "n_series": S, "n_ever_high": int(ever.sum()),
           "gap_hist": np.histogram(gaps, bins=GAP_BINS)[0].tolist(), "gaps": gaps.astype(np.int32)}
    el = np.broadcast_to(eligible[:, None, :], (T, S, P)).reshape(T, S * P)  # series-major, like m
    mf = m.reshape(T, S * P).astype(np.float32) * el
    high_mass = float((mf * high).sum())
    for D in rc["dormancy_steps"]:
        rev = reentry & (gap >= D)
        n_rev_pages = int(rev.any(0).sum())
        out[f"D{D}"] = {
            "revival_rate": n_rev_pages / max(1, int(ever.sum())),      # ever-high pages that revive
            "revivals": int(rev.sum()),
            "revivals_per_1k_steps_per_series": 1000 * float(rev.sum()) / (T * S),
            "revived_mass_share": float((mf * revived_mask(high, reentry, gap, D)).sum()) / high_mass
            if high_mass > 0 else float("nan"),
        }
    return out


def eviction_regret(m: np.ndarray, eligible: np.ndarray, init_score: np.ndarray, frac: float,
                    rounding: str = "ceil") -> np.ndarray:
    """H2O/SnapKV-style permanent eviction, per series. Scores start at ``init_score`` [S, P]
    (e.g. prefill-window mass) and accumulate each step's mass; whenever more than B_t
    eligible pages remain, the lowest-scoring ones are evicted for good. Returns [S]: share of
    eligible attention mass, over all steps, that lands on already-evicted pages.
    Approximation: scores accumulate full-attention mass (H2O would see attention renormalised
    over the pages it kept)."""
    T, S, P = m.shape
    score = init_score.astype(np.float64).copy()
    evicted = np.zeros((S, P), bool)
    lost = np.zeros(S)
    total = np.zeros(S)
    arangeP = np.arange(P)
    for t in range(T):
        el = eligible[t]
        cand = el[None, :] & ~evicted
        B = budget_pages(frac, int(el.sum()), rounding) if el.any() else 0
        excess = cand.sum(-1) - B
        if (excess > 0).any():
            order = np.argsort(np.where(cand, score, np.inf), axis=-1, kind="stable")
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.broadcast_to(arangeP, order.shape), axis=-1)
            evicted |= cand & (ranks < excess[:, None])
        mt = m[t].astype(np.float64)
        total += (mt * el).sum(-1)
        lost += (mt * (evicted & el)).sum(-1)
        score += mt
    return np.where(total > 0, lost / np.maximum(total, 1e-30), np.nan)


def q_stats(q: np.ndarray, lags: list[int], var_frac: float) -> dict:
    """Query trajectory smoothness for q [T, H, d]: mean cosine similarity at each lag, and the
    per-head effective dimension (PCA components reaching ``var_frac`` explained variance)."""
    qf = q.astype(np.float32)
    qn = qf / np.maximum(np.linalg.norm(qf, axis=-1, keepdims=True), 1e-12)
    T = q.shape[0]
    auto = {int(k): float((qn[:-k] * qn[k:]).sum(-1).mean()) if T > k else float("nan") for k in lags}
    dims = []
    for h in range(q.shape[1]):
        x = qf[:, h] - qf[:, h].mean(0)
        sv = np.linalg.svd(x, compute_uv=False) ** 2
        if sv.sum() <= 0:
            dims.append(0)
            continue
        dims.append(int(np.searchsorted(np.cumsum(sv) / sv.sum(), var_frac) + 1))
    return {"autocorr": auto, "eff_dim_median": float(np.median(dims)), "eff_dim": dims}
