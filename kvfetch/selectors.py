"""Training-free page selectors for tiered KV (EXPERIMENTS.md, E1), replayed on attention traces.

A selector picks b_t eligible pages per KV head for step t. It may use the current query, through
the Quest bound on page key summaries, and attention it has observed: the prefill observation
window, then at every step the renormalised mass on the pages it attended (its selection plus the
pinned sink/recent pages). It never reads attention on pages it did not attend; only the upper
bounds read full attention.

A candidate spec (config ``selectors.candidates``) splits the budget in order:
  recent_share  of b_t: the newest eligible pages (one set for all KV heads),
  quest_share   of the rest: best Quest bound,
  the remainder: best observed history, ``history: last`` (mass when last attended) or ``peak``.
``shared: true`` (or ``vote``) picks one set for all KV heads of the layer by per-head nomination
counts (LessIsMore-style), ties broken by the best per-head score; ``shared: sum`` takes the pages
with the largest score summed over heads (for attention-share-like scores, i.e. ``fuse``).
One shared set is what a vLLM paged kernel can read unchanged: a block holds all KV heads.
``fuse: sum|max`` replaces the split for the non-recent budget with one score: the softmax of the
Quest bounds over allowed pages (a pseudo attention share) combined with the observed history,
``(1 - history_weight) * quest + history_weight * history`` or their maximum.
``mean_mix`` blends the query's logit on the page's mean key into the Quest bound (caller passes
``qm``): ``(1 - mean_mix) * bound + mean_mix * mean_logit``.
"""
from __future__ import annotations

import numpy as np

from kvfetch.metrics.revival import high_matrix, reentries, revived_mask
from kvfetch.metrics.signals import KEY_MEAN
from kvfetch.sim import group_max, mask_from_ranks, rank_desc

# Unobserved pages tie at zero history; prefer newer pages among them (order only, not a weight).
_RECENCY_TIEBREAK = 1e-9


def _topk(score: np.ndarray, k: int) -> np.ndarray:
    """[S, P] score (-inf = not allowed) -> bool mask of the k best allowed pages per row."""
    out = np.zeros(score.shape, bool)
    k = min(k, score.shape[-1])
    if k <= 0:
        return out
    idx = np.argpartition(-score, k - 1, axis=-1)[:, :k]
    np.put_along_axis(out, idx, True, axis=-1)
    return out & np.isfinite(score)


def _shared_topk(score: np.ndarray, k: int) -> np.ndarray:
    """One set for all rows: rank pages by how many rows nominate them in their own top-k, then by
    the best score any row gives them."""
    S, P = score.shape
    nom = _topk(score, k).sum(0)
    best = np.where(np.isfinite(score), score, -np.inf).max(0)
    order = np.lexsort((-best, -nom))                     # primary: nominations, then best score
    out = np.zeros(P, bool)
    out[order[:min(k, int(np.isfinite(best).sum()))]] = True
    return np.broadcast_to(out, (S, P)).copy()


def _sum_topk(score: np.ndarray, k: int) -> np.ndarray:
    """One set for all rows: the k allowed pages with the largest score summed over rows."""
    ok = np.isfinite(score)
    tot = np.where(ok.any(0), np.where(ok, score, 0.0).sum(0), -np.inf)
    return np.broadcast_to(_topk(tot[None], k)[0], score.shape).copy()


_PICK = {False: _topk, True: _shared_topk, "vote": _shared_topk, "sum": _sum_topk}


def page_shares(score: np.ndarray, eligible: np.ndarray, kind: str) -> np.ndarray:
    """[T, S, P] -> per-head attention-share-like scores over eligible pages, zero elsewhere: the
    softmax of logit-like scores (``kind='logit'``, e.g. Quest bounds) or the normalised mass
    (``'mass'``, e.g. speculated attention)."""
    allowed = eligible[:, None, :]
    if kind == "logit":
        z = np.where(allowed, score, -np.inf)
        z = np.exp(z - np.where(allowed.any(-1, keepdims=True), z.max(-1, keepdims=True), 0.0))
    elif kind == "mass":
        z = np.where(allowed, score, 0.0)
    else:
        raise ValueError(kind)
    tot = z.sum(-1, keepdims=True)
    return np.divide(z, tot, out=np.zeros_like(z), where=tot > 0)


def static_select(score: np.ndarray, eligible: np.ndarray, b: np.ndarray, mode: str, kind: str = "logit") -> np.ndarray:
    """Stateless top-b_t selection of [T, S, P] scores (higher is better) per KV head
    (``per_head``) or as one set for all KV heads: ``vote`` (nomination counts, then best score)
    or ``sum`` (summed ``page_shares``). Returns bool [T, S, P]."""
    T, S, P = score.shape
    if mode == "per_head":
        return mask_from_ranks(rank_desc(score, eligible), b, eligible)
    if mode == "sum":
        key = page_shares(score, eligible, kind).sum(1, keepdims=True)
    elif mode == "vote":
        nom = mask_from_ranks(rank_desc(score, eligible), b, eligible).sum(1)                     # [T, P]
        best_rank = rank_desc(np.where(eligible[:, None, :], score, -np.inf).max(1, keepdims=True), eligible)[:, 0]
        key = (nom.astype(np.float32) * (P + 1) - best_rank)[:, None, :]
    else:
        raise ValueError(mode)
    return np.broadcast_to(mask_from_ranks(rank_desc(key, eligible), b, eligible), score.shape)


def mean_logit(q: np.ndarray, ks: np.ndarray, scaling: float, chunk: int = 256) -> np.ndarray:
    """Query logit on each page's mean key, max over the GQA group: [T, H, d] x [3, S, P, d] -> [T, S, P]."""
    T, H, d = q.shape
    S, P = ks.shape[1], ks.shape[2]
    kmean_t = np.repeat(ks[KEY_MEAN].astype(np.float32), H // S, 0).transpose(0, 2, 1)   # [H, d, P]
    out = np.empty((T, S, P), np.float32)
    for a in range(0, T, chunk):
        qf = q[a:a + chunk].astype(np.float32).transpose(1, 0, 2)                        # [H, t, d]
        out[a:a + chunk] = group_max((np.matmul(qf, kmean_t) * scaling).transpose(1, 0, 2), S)
    return out


def _quest_share_scores(qb_t: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Softmax of the Quest bounds over allowed pages per KV head: [S, P], zero where not allowed."""
    z = np.where(allowed, qb_t, -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def run_selector(spec: dict, *, m: np.ndarray, qb: np.ndarray, init: np.ndarray, eligible: np.ndarray,
                 exist: np.ndarray, b: np.ndarray, qm: np.ndarray | None = None) -> np.ndarray:
    """Replay one selector. m, qb [T, S, P]; init [S, P] mean prefill-window mass; eligible, exist
    [T, P]; b [T] selectable pages per step; qm [T, S, P] mean-key logits (for ``mean_mix``).
    Returns the selection [T, S, P] (pinned pages not included)."""
    T, S, P = m.shape
    q_share, r_share = spec.get("quest_share", 1.0), spec.get("recent_share", 0.0)
    hist_kind, shared = spec.get("history", "last"), spec.get("shared", False)
    fuse, w_hist = spec.get("fuse"), spec.get("history_weight", 0.5)
    if spec.get("mean_mix"):
        qb = (1.0 - spec["mean_mix"]) * qb + spec["mean_mix"] * qm
    pick = _PICK[shared]
    hist = init.astype(np.float32).copy()
    tiebreak = _RECENCY_TIEBREAK * np.arange(P, dtype=np.float32) / P
    sel = np.zeros((T, S, P), bool)
    for t in range(T):
        allowed = eligible[t]
        bt = int(min(b[t], allowed.sum()))
        chosen = np.zeros((S, P), bool)
        n_rec = int(round(r_share * bt))
        if n_rec:
            chosen[:, np.flatnonzero(allowed)[-n_rec:]] = True
        n_q = 0 if fuse else int(round(q_share * (bt - n_rec)))
        if fuse and bt - n_rec > 0:
            qs = _quest_share_scores(qb[t], allowed)
            f = (1.0 - w_hist) * qs + w_hist * hist if fuse == "sum" else np.maximum(qs, hist)
            chosen |= pick(np.where(allowed & ~chosen, f + tiebreak, -np.inf), bt - n_rec)
        if n_q:
            chosen |= pick(np.where(allowed & ~chosen, qb[t], -np.inf), n_q)
        n_h = bt - int(chosen.sum(-1).max())
        if n_h > 0:
            chosen |= pick(np.where(allowed & ~chosen, hist + tiebreak, -np.inf), n_h)
        sel[t] = chosen
        att = chosen | (exist[t] & ~allowed)[None, :]
        mt = np.where(att, m[t], 0.0)
        tot = mt.sum(-1, keepdims=True)
        obs = np.divide(mt, tot, out=np.zeros_like(mt), where=tot > 0)
        hist = np.where(att, obs, hist) if hist_kind == "last" else np.maximum(hist, obs)
    return sel


def upper_bound(name: str, *, m: np.ndarray, eligible: np.ndarray, b: np.ndarray) -> np.ndarray:
    """true_topB: best b_t pages per KV head by true attention; shared_topB: best one set for all
    KV heads (by summed attention)."""
    if name == "true_topB":
        return mask_from_ranks(rank_desc(m, eligible), b, eligible)
    if name == "shared_topB":
        tot = m.sum(1, keepdims=True)
        return np.broadcast_to(mask_from_ranks(rank_desc(tot, eligible), b, eligible), m.shape)
    raise ValueError(name)


def revived_weight(m: np.ndarray, eligible: np.ndarray, rc: dict) -> np.ndarray:
    """[T, S, P] attention mass on revived page-steps (re-entered the high state after at least
    ``rc['dormancy_main']`` dormant steps), zero elsewhere. ``rc`` is the ``revival`` config."""
    T, S, P = m.shape
    high = high_matrix(m, eligible, rc["budget_frac"], rc["exit_ratio"]).reshape(T, S * P)
    reentry, gap = reentries(high)
    rev = revived_mask(high, reentry, gap, rc["dormancy_main"]).reshape(T, S, P)
    return np.where(rev, m, 0.0)


def scoring_bytes(spec: dict, head_dim: int, summary_bytes: int, history_bytes: int) -> int:
    """GPU bytes a selector reads per page per KV head per step: Quest reads key min and max,
    ``mean_mix`` also the mean key, history one score."""
    fused = bool(spec.get("fuse"))
    uses_quest = fused or spec.get("quest_share", 1.0) > 0
    uses_hist = fused or spec.get("quest_share", 1.0) < 1
    n_vec = (2 if uses_quest else 0) + (1 if spec.get("mean_mix") else 0)
    return n_vec * head_dim * summary_bytes + (history_bytes if uses_hist else 0)
