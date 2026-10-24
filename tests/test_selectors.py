import numpy as np
import pytest

from kvfetch.selectors import _shared_topk, revived_weight, run_selector, scoring_bytes, upper_bound


def _late_page_trace():
    """T=8, S=1, P=4: page 0 pinned sink; pages 1-3 eligible. Page 1 is busy early, page 2 late.
    The prefill window saw only page 1, so history has never observed page 2."""
    T, P = 8, 4
    m = np.full((T, 1, P), 0.01, np.float32)
    m[:, 0, 0] = 0.5
    m[:4, 0, 1] = 0.4
    m[4:, 0, 2] = 0.5
    qb = np.zeros((T, 1, P), np.float32)
    qb[:4, 0, 1] = 1.0
    qb[4:, 0, 2] = 1.0                                   # the current query points at the busy page
    elig = np.zeros((T, P), bool)
    elig[:, 1:] = True
    exist = np.ones((T, P), bool)
    init = np.array([[0.5, 0.4, 0.0, 0.0]], np.float32)
    return dict(m=m, qb=qb, init=init, eligible=elig, exist=exist)


def test_history_cannot_see_unattended_pages_but_quest_can():
    tr = _late_page_trace()
    b = np.ones(8, int)
    hist = run_selector({"quest_share": 0.0, "history": "last"}, b=b, **tr)
    assert hist[:, 0, 1].all()                           # page 2's rise was never observed
    quest = run_selector({"quest_share": 1.0}, b=b, **tr)
    assert quest[:4, 0, 1].all() and quest[4:, 0, 2].all()


def test_hybrid_splits_budget_without_duplicates():
    tr = _late_page_trace()
    sel = run_selector({"quest_share": 0.5, "history": "last"}, b=np.full(8, 2), **tr)
    assert (sel.sum(-1) == 2).all()
    assert sel[:4, 0, 1].all() and sel[4:, 0, 2].all()   # Quest's half follows the busy page
    assert sel[:4, 0, 3].all() and sel[4, 0, 1]          # history: newest unseen page, then page 1
    assert sel[5, 0, 3]                                  # ... whose observed mass fell below page 3's
    assert not sel[:, 0, 0].any()                        # pinned pages are never "selected"


def test_recent_share_keeps_newest_eligible_page():
    tr = _late_page_trace()
    sel = run_selector({"quest_share": 1.0, "recent_share": 0.5}, b=np.full(8, 2), **tr)
    assert sel[:, 0, 3].all() and (sel.sum(-1) == 2).all()


def test_peak_history_remembers_and_last_history_renormalises():
    tr = _late_page_trace()
    tr["init"] = np.array([[0.5, 0.2, 0.0, 0.3]], np.float32)   # prefill liked page 3 most
    tr["m"][:, 0, 3] = 0.0                               # ... but decoding never attends to it
    b = np.ones(8, int)
    last = run_selector({"quest_share": 0.0, "history": "last"}, b=b, **tr)
    assert last[0, 0, 3] and last[1:, 0, 1].all()        # sees 0 on page 3, moves to page 1 (0.2)
    peak = run_selector({"quest_share": 0.0, "history": "peak"}, b=b, **tr)
    assert peak[:, 0, 3].all()                           # peak keeps the 0.3 it once saw


def test_shared_topk_counts_nominations_then_best_score():
    ninf = -np.inf
    score = np.array([[ninf, 5, 1, 4], [ninf, 1, 3, 4]], np.float32)
    assert _shared_topk(score, 1)[0].tolist() == [False, True, False, False]   # tie 1-1, page 1 scores 5
    two = _shared_topk(score, 2)
    assert two[0].tolist() == [False, True, False, True] and (two[0] == two[1]).all()


def test_upper_bounds():
    m = np.array([[[0.0, 0.6, 0.3, 0.1], [0.0, 0.1, 0.3, 0.6]]], np.float32)   # T=1, S=2
    elig = np.array([[False, True, True, True]])
    b = np.array([1])
    assert upper_bound("true_topB", m=m, eligible=elig, b=b)[0].tolist() == [[False, True, False, False],
                                                                              [False, False, False, True]]
    shared = upper_bound("shared_topB", m=m, eligible=elig, b=b)
    assert shared[0, 0].tolist() == [False, False, True, False] or shared[0, 0].tolist() == [False, True, False, False]
    assert (shared[0, 0] == shared[0, 1]).all()


def test_revived_weight_marks_reentry_after_dormancy():
    T = 40
    m = np.full((T, 1, 2), 0.05, np.float32)
    m[:5, 0, 0] = m[25:, 0, 0] = 0.9                     # page 0 busy, dormant for 20 steps, busy again
    m[5:25, 0, 1] = 0.9
    elig = np.ones((T, 2), bool)
    rc = {"budget_frac": 0.5, "exit_ratio": 0.5, "dormancy_main": 16}
    w = revived_weight(m, elig, rc)
    assert w[25:, 0, 0].sum() == pytest.approx(0.9 * 15) and w[:25].sum() == 0 and w[:, 0, 1].sum() == 0


def test_scoring_bytes():
    assert scoring_bytes({"quest_share": 1.0}, 128, 2, 4) == 512
    assert scoring_bytes({"quest_share": 0.0}, 128, 2, 4) == 4
    assert scoring_bytes({"quest_share": 0.5}, 128, 2, 4) == 516


def test_fused_selector_adds_unobserved_quest_page_and_keeps_observed_one():
    tr = _late_page_trace()
    sel = run_selector({"fuse": "max", "history": "last"}, b=np.full(8, 2), **tr)
    assert (sel.sum(-1) == 2).all() and sel[4:, 0, 2].all()


def test_mean_mix_uses_mean_logits():
    tr = _late_page_trace()
    qm = np.zeros_like(tr["qb"])
    qm[:, 0, 3] = 10.0                                   # the mean key points at page 3
    sel = run_selector({"quest_share": 1.0, "mean_mix": 0.5}, b=np.ones(8, int), qm=qm, **tr)
    assert sel[:, 0, 3].all()
    assert scoring_bytes({"quest_share": 1.0, "mean_mix": 0.5}, 128, 2, 4) == 768
    assert scoring_bytes({"fuse": "sum"}, 128, 2, 4) == 516


def test_static_select_modes():
    from kvfetch.selectors import static_select
    score = np.array([[[0.0, 5.0, 1.0, 4.0], [0.0, 1.0, 3.0, 4.0]]], np.float32)   # T=1, S=2
    elig = np.array([[False, True, True, True]])
    b = np.array([1])
    assert static_select(score, elig, b, "per_head")[0].tolist() == [[False, True, False, False],
                                                                     [False, False, False, True]]
    vote = static_select(score, elig, b, "vote")
    assert vote[0, 0].tolist() == [False, True, False, False] and (vote[0, 0] == vote[0, 1]).all()
    mass = np.array([[[0.0, 0.5, 0.1, 0.4], [0.0, 0.1, 0.4, 0.5]]], np.float32)
    assert static_select(mass, elig, b, "sum", "mass")[0, 0].tolist() == [False, False, False, True]


def test_shared_sum_in_run_selector_picks_one_set():
    tr = _late_page_trace()
    tr = {k: (np.concatenate([v, v], 1) if k in ("m", "qb") else v) for k, v in tr.items()}
    tr["init"] = np.concatenate([tr["init"], tr["init"]], 0)
    sel = run_selector({"fuse": "sum", "history_weight": 0.25, "shared": "sum"}, b=np.ones(8, int), **tr)
    assert (sel[:, 0] == sel[:, 1]).all() and (sel.sum(-1) == 1).all() and sel[4:, 0, 2].all()
