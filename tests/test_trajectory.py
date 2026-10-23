from pathlib import Path

import numpy as np

from kvfetch.config import load_config
from kvfetch.metrics.trajectory import (CLASSES, GRADUAL, ONSET_GRADUAL, ONSET_SUDDEN, OSCILLATING, OTHER,
                                        STABLE_HIGH, STABLE_LOW, SUDDEN, class_shares, classify,
                                        classify_ratio)

TC = {"budget_frac": 0.1, "exit_ratio": 0.5, "floor_ratio": 0.1, "smooth_ema_alpha": None,
      "stable_high_min_frac": 0.9, "min_initial_low_steps": 4, "sudden_max_rise_steps": 2,
      "oscillating_min_transitions": 4}
T = 100
t = np.arange(T, dtype=np.float32)


def _series() -> dict[str, np.ndarray]:
    return {
        "high": np.full(T, 2.0),
        "low": np.full(T, 0.05),
        "near_but_below": np.full(T, 0.9),
        "ramp": np.clip((t - 40) / 20, 0, 2),               # deep-low until 41, r >= 1 at 60
        "jump": np.where(t >= 60, 2.0, 0.0),
        "square": np.where((t // 10) % 2 == 1, 2.0, 0.0),
        "fall": np.where(t < 50, 2.0, 0.0),
        "hysteresis": np.where(t < 30, 0.0, np.where(t % 2 == 0, 1.1, 0.6)),
        "mid_start": np.where(t < 50, 0.5, 1.2),            # never deep-low before onset
    }


def test_canonical_trajectories():
    s = _series()
    names = list(s)
    res = classify_ratio(np.stack([s[n] for n in names], 1).astype(np.float32), TC)
    got = {n: CLASSES[c] for n, c in zip(names, res.cls)}
    assert got == {
        "high": "stable-high", "low": "stable-low", "near_but_below": "stable-low",
        "ramp": "gradual-rise", "jump": "sudden-jump", "square": "oscillating", "fall": "other",
        "hysteresis": "sudden-jump", "mid_start": "other",
    }
    i = {n: k for k, n in enumerate(names)}
    assert res.onset_step[i["ramp"]] == 60 and res.rise_steps[i["ramp"]] == 60 - 41
    assert res.rise_steps[i["jump"]] == 1 and res.onset_kind[i["jump"]] == ONSET_SUDDEN
    assert res.transitions[i["hysteresis"]] == 1  # flicker above exit_ratio is not a transition
    # oscillating pages still report the kind of their first onset
    assert res.onset_kind[i["square"]] == ONSET_SUDDEN
    assert np.isnan(res.rise_steps[i["fall"]])


def test_classify_end_to_end_with_rank_cutoff():
    # P = 21 pages: page 0 is a heavy sink (ineligible); 20 eligible -> B = 2 at 10%
    P = 21
    m = np.full((T, 1, P), 0.01, np.float32)
    m[:, 0, 0] = 5.0                                    # sink: must not set the cutoff
    m[:, 0, 1] = m[:, 0, 2] = 1.0                       # two incumbents
    m[:, 0, 3] = np.where(t >= 60, 2.0, 0.0)            # jumper
    m[:, 0, 4] = np.clip((t - 20) / 20, 0, 1.5)         # riser: deep-low until 21, reaches cutoff at 40
    elig = np.ones(P, bool)
    elig[0] = False
    res = classify(m, elig, TC)
    cls = res.cls.reshape(1, P - 1)[0]                  # eligible pages, in order
    assert cls[0] == STABLE_HIGH and cls[1] == STABLE_HIGH
    assert cls[2] == SUDDEN
    assert cls[3] == GRADUAL and res.rise_steps.reshape(1, P - 1)[0, 3] == 40 - 21
    assert (cls[4:] == STABLE_LOW).all()
    shares = class_shares(res)
    assert shares["all"]["stable-low"] == (P - 5) / (P - 1)
    assert shares["gradual_share_of_onsets"] == 0.5


def test_config_has_every_threshold():
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "base.yaml")
    assert set(TC) <= set(cfg["trajectory"])
    classify_ratio(np.stack(list(_series().values()), 1).astype(np.float32), cfg["trajectory"])


def test_class_codes_cover_all_names():
    assert sorted([STABLE_LOW, STABLE_HIGH, OSCILLATING, SUDDEN, GRADUAL, OTHER]) == list(range(len(CLASSES)))
    assert ONSET_GRADUAL != ONSET_SUDDEN
