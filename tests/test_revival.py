import numpy as np
import pytest

from kvfetch.metrics.importance import eligibility_matrix, topb_mask_var
from kvfetch.metrics.revival import (eviction_regret, high_matrix, q_stats, reentries, revival_stats,
                                     revived_mask)

RC = {"budget_frac": 0.5, "exit_ratio": 0.5, "dormancy_steps": [2, 4], "dormancy_main": 4}


def test_eligibility_aged_pages_join_as_they_age():
    # prompt 64 tokens, page 16, recent window 32: at step t, pages ending <= 64 + t - 32 are old
    pinned = {"sink_pages": 1, "decode_region": "aged", "recent_tokens": 32}
    e = eligibility_matrix(64, 40, 7, 16, pinned)
    assert np.flatnonzero(e[0]).tolist() == [1]          # page 1 ends at 32
    assert np.flatnonzero(e[16]).tolist() == [1, 2]      # horizon 48
    assert np.flatnonzero(e[39]).tolist() == [1, 2, 3]   # horizon 71: page 4 ends at 80 -> not yet
    assert not e[:, 0].any()                              # sink never eligible
    static = eligibility_matrix(64, 5, 7, 16, {"sink_pages": 1, "decode_region": True})
    assert np.flatnonzero(static[3]).tolist() == [1, 2, 3]


def test_topb_mask_var_budget_follows_eligible_count():
    score = np.array([[[5, 4, 3, 2]], [[5, 4, 3, 2]]], np.float32)   # [T=2, S=1, P=4]
    elig = np.array([[True, True, False, False], [True, True, True, True]])
    mask, cut, B = topb_mask_var(score, 0.5, elig)
    assert B.tolist() == [1, 2]
    assert mask[0, 0].tolist() == [True, False, False, False]
    assert mask[1, 0].tolist() == [True, True, False, False] and cut[1, 0] == 4


def test_reentry_gaps_and_dormancy_threshold():
    col = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1], bool)[:, None]   # gaps 6 then 2
    re, gap = reentries(col)
    assert np.flatnonzero(re[:, 0]).tolist() == [8, 12]
    assert gap[re].tolist() == [6, 2]
    rev = revived_mask(col, re, gap, D=4)
    assert np.flatnonzero(rev[:, 0]).tolist() == [8, 9]  # only the run after the 6-step gap


def _toy_trace():
    """T=20, S=1, P=4 (page 0 = sink, ineligible). Page 1 is a steady incumbent; page 2 is
    important for steps 0-3, dormant 4-11 (well below the cutoff, no ties), important again
    12-19; page 3 is background."""
    T, P = 20, 4
    m = np.full((T, 1, P), 0.01, np.float32)
    m[:, 0, 0] = 1.0
    m[:, 0, 1] = 0.30
    m[:4, 0, 2] = m[12:, 0, 2] = 0.40
    m[4:12, 0, 2] = 0.002              # cutoff is then page 3's 0.01; 0.2 of it < exit ratio 0.5
    elig = np.broadcast_to(np.array([False, True, True, True]), (T, P))
    return m, elig


def test_revival_stats_on_toy_trace():
    m, elig = _toy_trace()
    s = revival_stats(m, elig, RC)                       # B = ceil(0.5 * 3) = 2
    # pages 1, 2 and 3: the background page 3 enters the top-2 while page 2 sleeps (rank-based)
    assert s["n_ever_high"] == 3
    assert s["gaps"].tolist() == [8]                      # page 2: dormant steps 4..11
    assert s["D4"]["revival_rate"] == pytest.approx(1 / 3) and s["D4"]["revivals"] == 1
    assert s["D4"]["revivals"] == s["D2"]["revivals"]   # the single gap (8) exceeds both D
    # revived run = page 2 at steps 12..19: 8 * 0.40 of all high mass
    high_mass = 20 * 0.30 + 4 * 0.40 + 8 * 0.40 + 8 * 0.01  # page 3 is in top-2 while page 2 sleeps
    assert s["D4"]["revived_mass_share"] == pytest.approx(8 * 0.40 / high_mass, rel=1e-5)


def test_eviction_regret_loses_the_reviving_page():
    m, elig = _toy_trace()
    m = m.copy()
    m[:4, 0, 2] = 0.02                 # page 2 starts quiet this time, so the prefill score is low
    init = np.array([[0.0, 0.5, 0.0, 0.1]])  # prefill-window mass: page 3 looks more useful
    reg = eviction_regret(m, elig, init, frac=0.5)        # keep 2 of 3 -> page 2 evicted at t=0
    eligible_mass = m[:, 0, 1:].sum()
    assert reg[0] == pytest.approx(m[:, 0, 2].sum() / eligible_mass, rel=1e-6)
    assert eviction_regret(m, elig, init, frac=1.0)[0] == 0.0   # nothing evicted at full budget


def test_q_stats_smooth_rotation_in_a_plane():
    t = np.arange(200) * 0.01
    q = np.zeros((200, 1, 8), np.float32)
    q[:, 0, 0], q[:, 0, 1] = np.cos(t), np.sin(t)
    s = q_stats(q, [1, 64], 0.9)
    assert s["autocorr"][1] == pytest.approx(np.cos(0.01), abs=1e-4)
    assert s["autocorr"][64] == pytest.approx(np.cos(0.64), abs=1e-4)
    assert s["eff_dim"] == [2] or s["eff_dim"] == [1]     # an arc: 1-2 components reach 90 %


def test_high_matrix_ignores_ineligible_steps():
    m, elig = _toy_trace()
    elig = elig.copy()
    elig[:10, 2] = False               # page 2 not yet eligible: its early peak is not "high"
    h = high_matrix(m, elig, 0.5, 0.5)
    assert not h[:10, 0, 2].any() and h[12:, 0, 2].all()
