"""Reference predictors, only to exercise the metric pipeline end to end and to give the
small eval a naive baseline. The Phase 2 predictor comparison is NOT implemented here.
Every predictor returns score[t] using information up to and including step t."""
from __future__ import annotations

import numpy as np


def last_step(m: np.ndarray) -> np.ndarray:
    """score[t] = m[t]: InfiniGen-style locality assumption."""
    return m.astype(np.float32)


def ema(m: np.ndarray, alpha: float) -> np.ndarray:
    """score[t] = alpha m[t] + (1 - alpha) score[t-1]: H2O-style accumulated importance."""
    out = m.astype(np.float32).copy()
    for t in range(1, out.shape[0]):
        out[t] = alpha * out[t] + (1 - alpha) * out[t - 1]
    return out


def oracle(target: np.ndarray, k: int, n: int) -> np.ndarray:
    """score[t] = target[t+k]: upper bound for recall@B,k (sanity check: recall == 1)."""
    return target[k:k + n].astype(np.float32)
