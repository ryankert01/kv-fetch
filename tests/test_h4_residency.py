"""scripts/h4_residency.simulate (vectorised LRU page cache) against a plain per-policy Python LRU."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import h4_residency as R


def _reference(need, pre, aged, cap):
    T, P = need.shape
    last = {}                                             # page -> use order (higher = more recent)
    sync, traffic = [], []
    for t in range(T):
        for p in range(P):
            if aged[t, p]:
                last[p] = 3 * t
        traffic.append(sum(1 for p in range(P) if pre[t, p] and p not in last))
        for p in range(P):
            if pre[t, p]:
                last[p] = 3 * t + 1
        sync.append(sum(1 for p in range(P) if need[t, p] and p not in last))
        for p in range(P):
            if need[t, p]:
                last[p] = 3 * t + 2
        excess = len(last) - int(cap[t])
        if excess > 0:
            for p in sorted(last, key=lambda p: (last[p], p))[:excess]:
                del last[p]
    return sync, traffic


def test_simulate_matches_reference():
    g = torch.Generator().manual_seed(0)
    T, K, P = 60, 3, 24
    need = torch.rand((T, K, P), generator=g) < 0.2
    pre = torch.rand((T, K, P), generator=g) < 0.15
    pre[:, 0] = False                                     # a policy without prefetch
    aged = torch.rand((T, P), generator=g) < 0.05
    cap = torch.randint(3, 12, (T, K), generator=g)
    sync, traffic = R.simulate(need, pre, aged, cap)
    for k in range(K):
        s, tr = _reference(need[:, k], pre[:, k], aged, cap[:, k])
        assert sync[:, k].tolist() == s
        assert traffic[:, k].tolist() == tr
