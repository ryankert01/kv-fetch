"""Trajectory classifier (spec §3). Thresholds live in config ``trajectory``.

A page's level at step t is r(t) = m(t) / tau(t), where tau(t) is the top-B cutoff of
m at step t for the same series (B = trajectory.budget_frac of eligible pages). So
r >= 1 means "inside top-B". Classification uses m, not M_w: a forward window sum
turns every step change into a w-step ramp and would make sudden jumps look gradual.

High state uses hysteresis: enter at r >= 1, leave at r < exit_ratio.
Classes, first match wins:
  stable-low    never high
  stable-high   high for >= stable_high_min_frac of steps
  oscillating   >= oscillating_min_transitions in<->out transitions
  sudden-jump   first onset after >= min_initial_low_steps steps, rising from deep-low
  gradual-rise    (r < floor_ratio) to high within <= / > sudden_max_rise_steps steps
  other         e.g. high from the start then falls; onset never preceded by deep-low
Independently of the class, every onset that follows a deep-low period gets an
``onset_kind`` (sudden / gradual) and ``rise_steps``, so oscillating pages still
contribute rise statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kvfetch.metrics.importance import budget_pages, topb_cutoff

CLASSES = ["stable-low", "stable-high", "oscillating", "sudden-jump", "gradual-rise", "other"]
STABLE_LOW, STABLE_HIGH, OSCILLATING, SUDDEN, GRADUAL, OTHER = range(len(CLASSES))
ONSET_NONE, ONSET_SUDDEN, ONSET_GRADUAL = 0, 1, 2


@dataclass
class TrajectoryResult:
    cls: np.ndarray          # [N] int, index into CLASSES
    onset_kind: np.ndarray   # [N] int, ONSET_*
    onset_step: np.ndarray   # [N] int, first high step or -1
    rise_steps: np.ndarray   # [N] float, onset - last deep-low step (nan if no rise)
    frac_high: np.ndarray    # [N]
    transitions: np.ndarray  # [N]


def level_ratio(m: np.ndarray, eligible: np.ndarray, budget_frac: float, rounding: str = "ceil",
                ema_alpha: float | None = None) -> np.ndarray:
    """[T, S, P] mass -> [T, S, P] ratio to the per-step top-B cutoff (eligible pages)."""
    x = m.astype(np.float32)
    if ema_alpha is not None:
        for t in range(1, x.shape[0]):
            x[t] = ema_alpha * x[t] + (1 - ema_alpha) * x[t - 1]
    B = budget_pages(budget_frac, int(eligible.sum()), rounding)
    tau = topb_cutoff(x, B, eligible)[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tau > 0, x / tau, 0.0)


def high_state(r: np.ndarray, exit_ratio: float) -> np.ndarray:
    """Hysteresis over axis 0 of r [T, N]: enter at r >= 1, leave at r < exit_ratio."""
    h = np.zeros(r.shape, dtype=bool)
    h[0] = r[0] >= 1
    for t in range(1, r.shape[0]):
        h[t] = np.where(h[t - 1], r[t] >= exit_ratio, r[t] >= 1)
    return h


def classify_ratio(r: np.ndarray, tc: dict) -> TrajectoryResult:
    """r: [T, N] level ratio series. ``tc`` is the ``trajectory`` config block."""
    T, N = r.shape
    h = high_state(r, tc["exit_ratio"])
    frac_high = h.mean(0)
    transitions = (h[1:] != h[:-1]).sum(0)
    ever = h.any(0)
    onset = np.where(ever, h.argmax(0), -1)

    # last deep-low step strictly before the onset
    deep = r < tc["floor_ratio"]
    steps = np.arange(T)[:, None]
    before = steps < np.where(onset >= 0, onset, 0)[None]
    last_deep = np.where((deep & before).any(0), np.where(deep & before, steps, -1).max(0), -1)
    is_rise = (onset >= tc["min_initial_low_steps"]) & (last_deep >= 0)
    rise_steps = np.where(is_rise, (onset - last_deep).astype(np.float32), np.nan)
    sudden = is_rise & (rise_steps <= tc["sudden_max_rise_steps"])
    onset_kind = np.where(sudden, ONSET_SUDDEN, np.where(is_rise, ONSET_GRADUAL, ONSET_NONE))

    cls = np.full(N, OTHER, dtype=np.int8)
    rest = np.ones(N, dtype=bool)
    for code, cond in (
        (STABLE_LOW, ~ever),
        (STABLE_HIGH, frac_high >= tc["stable_high_min_frac"]),
        (OSCILLATING, transitions >= tc["oscillating_min_transitions"]),
        (SUDDEN, onset_kind == ONSET_SUDDEN),
        (GRADUAL, onset_kind == ONSET_GRADUAL),
    ):
        sel = rest & cond
        cls[sel] = code
        rest &= ~sel
    return TrajectoryResult(cls=cls, onset_kind=onset_kind.astype(np.int8), onset_step=onset,
                            rise_steps=rise_steps, frac_high=frac_high, transitions=transitions)


def classify(m: np.ndarray, eligible: np.ndarray, tc: dict, rounding: str = "ceil") -> TrajectoryResult:
    """[T, S, P] mass -> per-(series, eligible page) classes, flattened to N = S * n_eligible."""
    r = level_ratio(m, eligible, tc["budget_frac"], rounding, tc.get("smooth_ema_alpha"))
    r = r[..., eligible].reshape(r.shape[0], -1)
    return classify_ratio(r, tc)


def class_shares(res: TrajectoryResult) -> dict:
    n = res.cls.size
    ever_high = res.cls != STABLE_LOW
    out = {"n_pages": n, "all": {}, "ever_high": {}}
    for i, name in enumerate(CLASSES):
        c = int((res.cls == i).sum())
        out["all"][name] = c / n if n else float("nan")
        if i != STABLE_LOW:
            out["ever_high"][name] = c / ever_high.sum() if ever_high.any() else float("nan")
    rises = res.onset_kind != ONSET_NONE
    out["onsets_after_low"] = int(rises.sum())
    out["gradual_share_of_onsets"] = float((res.onset_kind == ONSET_GRADUAL).sum() / rises.sum()) if rises.any() else float("nan")
    rs = res.rise_steps[rises]
    out["rise_steps"] = {"median": float(np.median(rs)) if rs.size else float("nan"),
                         "p25": float(np.percentile(rs, 25)) if rs.size else float("nan"),
                         "p75": float(np.percentile(rs, 75)) if rs.size else float("nan"),
                         "p90": float(np.percentile(rs, 90)) if rs.size else float("nan")}
    return out
