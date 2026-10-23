import numpy as np
import pytest

from kvfetch.sim import (coverage, evict_context, existing_pages, prefetch_sets, quest_bound, simulate_evict,
                         simulate_offload, topk_abs)

CFG = {"snapkv_window": 4, "important_mult": 1.0, "rkv_lambda": 0.5, "q_slope_steps": 2}


def test_topk_abs_exact_counts_and_allowed():
    score = np.array([[[3, 2, 1, 0]], [[3, 2, 1, 0]]], np.float32)
    allowed = np.array([[False, True, True, True], [True, True, True, True]])
    mask = topk_abs(score, np.array([2, 5]), allowed)
    assert mask[0, 0].tolist() == [False, True, True, False]
    assert mask[1, 0].sum() == 4                       # budget above n_allowed -> all allowed pages


def _two_page_trace():
    """T=12, S=1, P=4: page 0 sink; pages 1,2,3 eligible from t=0. Page 1 is busy early and
    then idle; page 2 is idle early and busy late; page 3 is background. Budget keeps 1 of 3."""
    T, P = 12, 4
    m = np.full((T, 1, P), 0.01, np.float32)
    m[:, 0, 0] = 0.5
    m[:6, 0, 1] = 0.4
    m[6:, 0, 2] = 0.5                                  # future mass 3.06 > page 1's 2.46
    elig = np.zeros((T, P), bool)
    elig[:, 1:] = True
    exist = np.ones((T, P), bool)
    init = np.array([[0.0, 0.5, 0.0, 0.1]], np.float32)  # prefill: page 1 looked useful, page 2 not
    return m, elig, exist, init


def test_oracle_eviction_beats_h2o_on_a_late_page():
    m, elig, exist, init = _two_page_trace()
    ctx = evict_context(m, init, np.random.default_rng(0).normal(size=(1, 4, 8)).astype(np.float32), exist, CFG)
    B = 2                                              # sink (pinned) + 1 selectable page
    cov_h2o = simulate_evict("h2o", m=m, eligible=elig, exist=exist, B=B, ctx=ctx)
    cov_orc = simulate_evict("oracle", m=m, eligible=elig, exist=exist, B=B, ctx=ctx)
    # H2O keeps page 1 (best prefill score); the oracle keeps page 2 (larger remaining future mass)
    tot = m.sum(-1)[:, 0]
    want_h2o = np.mean((m[:, 0, 0] + m[:, 0, 1]) / tot)
    want_orc = np.mean((m[:, 0, 0] + m[:, 0, 2]) / tot)
    assert cov_h2o == pytest.approx(want_h2o, rel=1e-5)
    assert cov_orc == pytest.approx(want_orc, rel=1e-5) and cov_orc > cov_h2o
    assert simulate_evict("h2o", m=m, eligible=elig, exist=exist, B=B, ctx=ctx) == pytest.approx(cov_h2o)  # no shared state


def test_streaming_keeps_newest_pages():
    m, elig, exist, init = _two_page_trace()
    ctx = evict_context(m, init, np.zeros((1, 4, 8), np.float32) + 1, exist, CFG)
    cov = simulate_evict("streaming", m=m, eligible=elig, exist=exist, B=2, ctx=ctx)
    tot = m.sum(-1)[:, 0]
    assert cov == pytest.approx(np.mean((m[:, 0, 0] + m[:, 0, 3]) / tot), rel=1e-5)   # keeps page 3


def test_offload_stalls_on_demand_vs_perfect_prefetch():
    T, S, P = 6, 1, 4
    m = np.ones((T, S, P), np.float32)
    elig = np.ones((T, P), bool)
    sel = np.zeros((T, S, P), bool)
    sel[:3, 0, 1] = True
    sel[3:, 0, 2] = True                               # selection switches page at t=3
    on_demand = simulate_offload(m=m, sel=sel, prefetch=None, eligible=elig, capacity=1)
    assert on_demand["stall_pages_per_step"] == pytest.approx(2 / T)     # t=0 and t=3
    perfect = simulate_offload(m=m, sel=sel, prefetch=sel, eligible=elig, capacity=2)
    assert perfect["stall_pages_per_step"] == 0 and perfect["nostall_coverage_of_selected"] == 1.0
    assert perfect["prefetch_pages_per_step"] == pytest.approx(2 / T)
    pinned = np.zeros((T, 1, P), bool)
    pinned[:, 0, 0] = True
    skip = simulate_offload(m=m, sel=sel, prefetch=None, eligible=elig, capacity=1, base=pinned)
    # 4 hit steps cover pinned + selected (2 of 4 pages); the 2 stall steps keep only the pinned page
    assert skip["skip_stall_coverage"] == pytest.approx((4 * 2 / 4 + 2 * 1 / 4) / T)


def test_quest_bound_and_stale_prefetch_shape():
    rng = np.random.default_rng(1)
    T, H, S, P, d = 5, 4, 2, 6, 8
    q = rng.normal(size=(T, H, d)).astype(np.float32)
    keys = rng.normal(size=(S, P, 16, d)).astype(np.float32)
    ks = np.stack([keys.mean(2), keys.min(2), keys.max(2)])
    b = quest_bound(q, ks, 1.0, chunk=2)
    assert b.shape == (T, S, P)
    true_max = np.einsum("thd,spnd->thspn", q, keys).max(-1)            # [T, H, S, P]
    for h in range(H):
        assert (b[:, h // 2] >= true_max[:, h, h // 2] - 1e-4).all()    # bound >= any head in the group
    elig = np.ones((T, P), bool)
    B = np.full(T, 2)
    sel = topk_abs(b, B, elig)
    pf = prefetch_sets("stale_q", q=q, ks=ks, scaling=1.0, sel=sel, B=B, eligible=elig, L=1, cfg=CFG)
    assert pf.shape == sel.shape and (pf[1:] == sel[:-1]).all()          # stale by exactly one step
    assert prefetch_sets("oasis_ub", q=q, ks=ks, scaling=1.0, sel=sel, B=B, eligible=elig, L=1, cfg=CFG) is sel


def test_existing_pages():
    e = existing_pages(prompt_len=20, T=3, P=3, page_size=16)
    assert e.tolist() == [[True, True, False], [True, True, False], [True, True, False]]
    assert coverage(np.ones((1, 1, 2), bool), np.ones((1, 1, 2), np.float32)) == 1.0
