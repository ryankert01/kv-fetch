import numpy as np
import pytest

from kvfetch.metrics import predictors
from kvfetch.metrics.importance import (budget_pages, eligible_pages, future_mass, head_view, topb_cutoff,
                                        topb_mask)
from kvfetch.metrics.selection import (CENSORED, MISSED, NEVER_TRUE, OK, churn, lead_summary, lead_times,
                                       mass_recall, recall_at_k, sets_for_next_step)


def test_future_mass_is_forward_window_sum():
    T, w = 12, 3
    m = np.tile(np.arange(T, dtype=np.float32)[:, None, None], (1, 1, 2))  # m[t] = t
    M = future_mass(m, w)
    assert M.shape == (T - w, 1, 2)
    for t in range(T - w):
        assert M[t, 0, 0] == sum(range(t + 1, t + w + 1))


def test_eligible_pages_pins_sink_and_decode_region():
    # prompt of 100 tokens, page 16 -> full pages 0..5, page 6 partial (holds generated tokens)
    e = eligible_pages(100, 9, 16, {"sink_pages": 1, "recent_prompt_pages": 0, "decode_region": True})
    assert e.tolist() == [False, True, True, True, True, True, False, False, False]
    e = eligible_pages(100, 9, 16, {"sink_pages": 1, "recent_prompt_pages": 2, "decode_region": True})
    assert np.flatnonzero(e).tolist() == [1, 2, 3]


def test_budget_pages_rounding():
    assert budget_pages(0.05, 100) == 5
    assert budget_pages(0.01, 150) == 2  # ceil(1.5)
    assert budget_pages(0.001, 10) == 1  # at least one page


def test_topb_mask_and_cutoff_respect_eligibility():
    s = np.array([[9.0, 1.0, 5.0, 3.0, 4.0]])
    elig = np.array([False, True, True, True, True])
    assert np.flatnonzero(topb_mask(s, 2, elig)[0]).tolist() == [2, 4]
    assert topb_cutoff(s, 2, elig)[0] == 4.0


def test_recall_is_mass_weighted():
    truth = np.array([[0.0, 0.5, 0.3, 0.1, 0.1]])
    elig = np.array([False, True, True, True, True])
    best2 = 0.5 + 0.3
    pred = np.array([[False, True, False, True, False]])
    assert mass_recall(pred, truth, 2, elig)[0] == pytest.approx((0.5 + 0.1) / best2)
    pred = np.array([[False, False, True, True, False]])
    assert mass_recall(pred, truth, 2, elig)[0] == pytest.approx((0.3 + 0.1) / best2)


def test_recall_at_k_alignment_and_oracle():
    T, P, k = 20, 6, 3
    elig = np.ones(P, bool)
    target = np.full((T, 1, P), 0.01, np.float32)
    for t in range(T):
        target[t, 0, t % P] = 1.0  # the hot page at step t is t % P
    # a predictor that knows target[t+k] gets recall 1 at horizon k, ~0 at another horizon
    score = predictors.oracle(target, k, T - k)
    assert np.allclose(recall_at_k(score, target, k, 1, elig), 1.0)
    assert recall_at_k(score, target, k + 1, 1, elig).max() < 0.1


def _linear_riser_trace(T=64):
    """Page 0: sink (ineligible). Page 1: incumbent at 33/64. Page 2: rises (t-10)/32
    from t=10, first exceeds the incumbent at t=27. Page 3: background."""
    m = np.zeros((T, 1, 4), np.float32)
    m[:, 0, 0] = 2.0
    m[:, 0, 1] = 33 / 64
    m[:, 0, 2] = np.maximum(np.arange(T) - 10, 0) / 32
    m[:, 0, 3] = 1 / 16
    return m, np.array([False, True, True, True])


def _trend(m, h):
    """score[t] = m[t] + h * (m[t] - m[t-1]): linear extrapolation h steps ahead."""
    s = m.copy()
    s[1:] = m[1:] + h * (m[1:] - m[:-1])
    return s


def test_lead_time_linear_riser_known_values():
    m, elig = _linear_riser_trace()
    true_sets = topb_mask(m, 1, elig)
    assert true_sets[:, 0, 2].argmax() == 27 and not true_sets[26, 0, 2]

    # last-step predictor sees the page one step too late: lead -1 (stall)
    lead, status = lead_times(sets_for_next_step(predictors.last_step(m), 1, elig), true_sets, elig)
    assert status[0, 2] == OK and lead[0, 2] == -1

    # extrapolating h steps ahead buys exactly h-1 steps of lead on a linear riser
    for h in (2, 4, 8):
        lead, status = lead_times(sets_for_next_step(_trend(m, h), 1, elig), true_sets, elig)
        assert status[0, 2] == OK and lead[0, 2] == h - 1
    # the incumbent is top-1 from step 0: censored, not an event
    assert status[0, 1] == CENSORED
    assert status[0, 0] == NEVER_TRUE and status[0, 3] == NEVER_TRUE  # sink is ineligible


def test_lead_time_modes_late_and_missed():
    T = 30
    elig = np.ones(4, bool)
    true_sets = np.zeros((T, 1, 4), bool)
    pred_for = np.zeros((T, 1, 4), bool)
    true_sets[15:, 0, 0] = True   # page 0 needed from 15
    pred_for[5:8, 0, 0] = True    # predicted early, dropped,
    pred_for[12:, 0, 0] = True    # re-predicted at 12
    true_sets[15:, 0, 1] = True   # page 1: predicted only after the need -> late by 3
    pred_for[18:, 0, 1] = True
    true_sets[15:, 0, 2] = True   # page 2: never predicted -> missed
    lead, status = lead_times(pred_for, true_sets, elig, mode="first")
    assert lead[0, 0] == 10 and lead[0, 1] == -3 and status[0, 2] == MISSED and status[0, 3] == NEVER_TRUE
    lead, status = lead_times(pred_for, true_sets, elig, mode="contiguous")
    assert lead[0, 0] == 3 and lead[0, 1] == -3 and status[0, 2] == MISSED
    s = lead_summary(lead, status)
    assert s["events"] == 3 and s["missed_rate"] == pytest.approx(1 / 3)
    assert s["on_time_rate"] == pytest.approx(1 / 3)


def test_churn():
    sets = np.zeros((4, 1, 4), bool)
    sets[0, 0, [0, 1]] = True
    sets[1, 0, [0, 1]] = True    # no change
    sets[2, 0, [2, 3]] = True    # full swap
    sets[3, 0, [2, 0]] = True    # one swap
    assert churn(sets, 2)[:, 0].tolist() == [0.0, 2.0, 1.0]


def test_head_view_kv_groups_are_contiguous():
    T, L, H, kv, P = 2, 1, 8, 2, 3
    m = np.zeros((T, L, H, P), np.float32)
    m[:, :, :4, 0] = 1.0  # query heads 0..3 read KV head 0
    m[:, :, 4:, 1] = 1.0  # query heads 4..7 read KV head 1
    v = head_view(m, "kv_mean", kv)
    assert v.shape == (T, L, kv, P)
    assert v[0, 0, 0].tolist() == [1, 0, 0] and v[0, 0, 1].tolist() == [0, 1, 0]
