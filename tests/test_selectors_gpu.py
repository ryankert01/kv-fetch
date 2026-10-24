import numpy as np
import pytest
import torch

from kvfetch import selectors_gpu as G
from kvfetch.selectors import revived_weight, run_selector, static_select
from kvfetch.sim import rank_desc


def _inputs(T=40, S=2, P=12, seed=0):
    """Random continuous scores (no ties), a growing eligible window, pinned sink and recent pages."""
    rng = np.random.default_rng(seed)
    m = rng.dirichlet(np.ones(P), size=(T, S)).astype(np.float32)
    qb = rng.normal(size=(T, S, P)).astype(np.float32)
    qm = rng.normal(size=(T, S, P)).astype(np.float32)
    exist = np.zeros((T, P), bool)
    elig = np.zeros((T, P), bool)
    for t in range(T):
        n = min(P, 6 + t // 6)
        exist[t, :n] = True
        elig[t, 1:max(1, n - 2)] = True
    init = rng.dirichlet(np.ones(P), size=S).astype(np.float32)
    b = (2 + np.arange(T) % 3).astype(np.int64)
    return m, qb, qm, init, elig, exist, b


def test_rank_and_static_select_match_numpy():
    m, qb, _, _, elig, _, b = _inputs()
    t = lambda x: torch.from_numpy(x)
    assert np.array_equal(G.rank_desc(t(qb), t(elig)).numpy(), rank_desc(qb, elig))
    for kind, score in [("logit", qb), ("mass", m)]:
        ranks = G.static_ranks(t(score), t(elig), kind)
        for mode in ["per_head", "vote", "sum"]:
            got = G.select_from_ranks(ranks, t(elig), t(b), mode).numpy()
            assert np.array_equal(got, static_select(score, elig, b, mode, kind)), (kind, mode)


SPECS = [
    {"quest_share": 1.0},
    {"quest_share": 0.0, "history": "last"},
    {"quest_share": 0.5, "history": "peak", "recent_share": 0.5},
    {"fuse": "sum", "history_weight": 0.25, "history": "last", "mean_mix": 0.5},
    {"fuse": "max", "history": "peak"},
]


@pytest.mark.parametrize("shared", [False, "vote", "sum"])
def test_replay_matches_run_selector(shared):
    m, qb, qm, init, elig, exist, b = _inputs()
    t = lambda x: torch.from_numpy(x)
    sources = torch.stack([t(qb), t(qm)])
    specs = SPECS + [{"fuse": "sum", "history_weight": 0.5}]
    src = [0] * len(SPECS) + [1]                                       # the last one scores with qm
    bk = torch.stack([t(b)] * len(specs))
    got = np.zeros((len(specs),) + m.shape, bool)
    for step, chosen in G.replay(specs, sources, src, bk, m=t(m), init=t(init), eligible=t(elig), exist=t(exist),
                                 shared=shared, qm=t(qm)):
        got[:, step] = chosen.numpy()
    for i, spec in enumerate(specs):
        want = run_selector({**spec, "shared": shared}, m=m, qb=qb if src[i] == 0 else qm, init=init, eligible=elig,
                            exist=exist, b=b, qm=qm)
        assert np.array_equal(got[i], want), (i, spec)


def test_revived_weight_matches_numpy():
    T = 60
    m = np.full((T, 1, 4), 0.05, np.float32)
    m[:5, 0, 0] = m[25:, 0, 0] = 0.9                     # page 0 busy, dormant for 20 steps, busy again
    m[5:25, 0, 1] = 0.9
    m += np.random.default_rng(0).uniform(0, 1e-3, m.shape).astype(np.float32)
    elig = np.ones((T, 4), bool)
    rc = {"budget_frac": 0.25, "exit_ratio": 0.5, "dormancy_main": 16}
    got = G.revived_weight(torch.from_numpy(m), torch.from_numpy(elig), rc).numpy()
    want = revived_weight(m, elig, rc)
    assert want.sum() > 0 and np.allclose(got, want)
