import numpy as np
import pytest

from kvfetch.metrics.signals import event_recall, extrapolate_q, mass_recall_k, signal_scores

SC = {"ema_alpha": 0.5, "q_slope_steps": 2}


def _toy(seed=0, T=6, H=4, KVH=2, P=5, d=8):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(T, H, d)).astype(np.float32)
    keys = rng.normal(size=(KVH, P, 16, d)).astype(np.float32)        # 16 tokens per page
    ks = np.stack([keys.mean(2), keys.min(2), keys.max(2)])            # [3, KVH, P, d]
    m = rng.random((T, H, P)).astype(np.float32)
    return q, keys, ks, m


def test_quest_is_an_upper_bound_and_matches_brute_force():
    q, keys, ks, m = _toy()
    s = signal_scores("quest", m=m, q=q, key_summary=ks, scaling=1.0, sc=SC, k=1)
    G = q.shape[1] // keys.shape[0]
    brute = np.zeros_like(s)
    for h in range(q.shape[1]):
        kmin, kmax = ks[1, h // G], ks[2, h // G]
        brute[:, h] = np.maximum(q[:, h, None, :] * kmin[None], q[:, h, None, :] * kmax[None]).sum(-1)
        exact_max = np.einsum("td,pnd->tpn", q[:, h], keys[h // G]).max(-1)   # best real logit per page
        assert (brute[:, h] >= exact_max - 1e-4).all()
    assert np.allclose(s, brute, atol=1e-4)


def test_key_mean_uses_the_gqa_group():
    q, keys, ks, m = _toy()
    s = signal_scores("key_mean", m=m, q=q, key_summary=ks, scaling=0.5, sc=SC, k=1)
    assert np.allclose(s[:, 3], 0.5 * q[:, 3] @ ks[0, 1].T, atol=1e-5)   # head 3 reads KV head 1


def test_extrapolate_q_is_linear():
    t = np.arange(10, dtype=np.float32)
    q = np.stack([t, 2 * t], -1)[:, None, :]                           # [T, 1, 2], slope (1, 2)
    qh = extrapolate_q(q, k=4, j=2)
    assert np.allclose(qh[5, 0], q[5, 0] + 4 * np.array([1, 2]))
    assert np.allclose(qh[0], q[0])                                    # no history yet


def test_event_recall_and_mass_recall():
    T, H, P = 5, 1, 4
    elig = np.ones((T, P), bool)
    score = np.zeros((T, H, P), np.float32)
    score[:, 0, 2] = 1.0                                               # always ranks page 2 first
    events = (np.array([3, 4]), np.array([0, 0]), np.array([2, 1]))    # page 2 at t=3, page 1 at t=4
    assert event_recall(score, elig, 0.25, events, k=1) == 0.5         # B = 1: only page 2 is predicted
    m = np.zeros((T, H, P), np.float32)
    m[:, 0, 2] = 1.0
    assert mass_recall_k(score, m, elig, 0.25, k=2) == pytest.approx(1.0)
