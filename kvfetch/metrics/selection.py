"""recall@B,k, lead time and churn (spec §3). Arrays are [T, S, P] unless noted."""
from __future__ import annotations

import numpy as np

from kvfetch.metrics.importance import topb_mask

# lead-time status codes
OK, NEVER_TRUE, CENSORED, MISSED = 0, 1, 2, 3
STATUS_NAMES = {OK: "ok", NEVER_TRUE: "never_true", CENSORED: "censored", MISSED: "missed"}


def mass_recall(pred: np.ndarray, truth: np.ndarray, B: int, eligible: np.ndarray) -> np.ndarray:
    """(truth mass covered by pred) / (truth mass of the true top-B). Mass-weighted.
    pred: bool [..., P]; truth: [..., P]. NaN where the true top-B has zero mass."""
    t = np.where(eligible, truth.astype(np.float32), 0.0)
    covered = (t * pred).sum(-1)
    best = (t * topb_mask(t, B, eligible)).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(best > 0, covered / best, np.nan)


def recall_at_k(score: np.ndarray, target: np.ndarray, k: int, B: int, eligible: np.ndarray) -> np.ndarray:
    """Predict at t (``score[t]``, info <= t) the top-B of ``target[t+k]``.
    ``target`` is indexed by step (use M_w for the primary target). Returns [n, S]
    for t in [0, n), n = min(len(score), len(target) - k)."""
    n = min(score.shape[0], target.shape[0] - k)
    if n <= 0:
        raise ValueError(f"k={k} leaves no aligned steps")
    pred = topb_mask(score[:n], B, eligible)
    return mass_recall(pred, target[k:k + n], B, eligible)


def sets_for_next_step(score: np.ndarray, B: int, eligible: np.ndarray) -> np.ndarray:
    """Resident set intended FOR step t: top-B of score[t-1]. Row 0 is empty (no
    information before step 0). Returns bool [T, S, P]."""
    out = np.zeros(score.shape, dtype=bool)
    out[1:] = topb_mask(score[:-1], B, eligible)
    return out


def churn(sets: np.ndarray, B: int) -> np.ndarray:
    """|S(t) xor S(t-1)| / B for t >= 1. Range [0, 2]. Returns [T-1, S]."""
    return (sets[1:] ^ sets[:-1]).sum(-1) / B


def _first_index(mask: np.ndarray) -> np.ndarray:
    """First t with mask[t] along axis 0, or -1."""
    any_ = mask.any(0)
    return np.where(any_, mask.argmax(0), -1)


def lead_times(pred_for: np.ndarray, true_sets: np.ndarray, eligible: np.ndarray, *,
               mode: str = "first", first_eval_step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Lead time per (series, page): t_true - t_pred, where t_true is the first step the
    page is in ``true_sets`` and t_pred the step whose resident set (``pred_for[t]`` =
    set intended FOR step t) first contains it. Negative = late (stall).

    mode="first":      t_pred = first step ever predicted (spec definition).
    mode="contiguous": if predicted at t_true, t_pred = start of the unbroken predicted
                       run ending at t_true; else t_pred = next predicted step after t_true.
    Pages whose first true step is < ``first_eval_step`` are CENSORED (already needed
    before any prediction exists). Returns (lead [S, P] float, status [S, P] int).
    """
    T = true_sets.shape[0]
    t_true = _first_index(true_sets)  # [S, P]
    status = np.full(t_true.shape, OK, dtype=np.int8)
    status[t_true < 0] = NEVER_TRUE
    status[(t_true >= 0) & (t_true < first_eval_step)] = CENSORED

    if mode == "first":
        t_pred = _first_index(pred_for)
    elif mode == "contiguous":
        tt = np.clip(t_true, 0, T - 1)
        at_true = np.take_along_axis(pred_for, tt[None], 0)[0]
        steps = np.arange(T).reshape(T, *([1] * (pred_for.ndim - 1)))
        # last step <= t that was NOT predicted
        last_gap = np.maximum.accumulate(np.where(~pred_for, steps, -1), axis=0)
        run_start = np.take_along_axis(last_gap, tt[None], 0)[0] + 1
        # first predicted step >= t (T if none)
        nxt = np.flip(np.minimum.accumulate(np.flip(np.where(pred_for, steps, T), 0), axis=0), 0)
        next_pred = np.take_along_axis(nxt, tt[None], 0)[0]
        t_pred = np.where(at_true, run_start, np.where(next_pred < T, next_pred, -1))
    else:
        raise ValueError(mode)

    status[(status == OK) & (t_pred < 0)] = MISSED
    lead = np.where(status == OK, (t_true - t_pred).astype(np.float32), np.nan)
    status[:, ~eligible] = NEVER_TRUE  # ineligible pages are not part of the decision
    lead[:, ~eligible] = np.nan
    return lead, status


def lead_summary(lead: np.ndarray, status: np.ndarray, required_lead: int = 0) -> dict:
    events = (status == OK) | (status == MISSED)
    n = int(events.sum())
    ok = lead[status == OK]
    return {
        "events": n,
        "censored": int((status == CENSORED).sum()),
        "missed_rate": float((status == MISSED).sum() / n) if n else float("nan"),
        "median": float(np.median(ok)) if ok.size else float("nan"),
        "p10": float(np.percentile(ok, 10)) if ok.size else float("nan"),
        "p90": float(np.percentile(ok, 90)) if ok.size else float("nan"),
        "on_time_rate": float((ok >= required_lead).sum() / n) if n else float("nan"),
    }
