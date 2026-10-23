"""Offline comparison of KV-cache methods at equal GPU budget, replayed on attention traces.

Inputs per (sample, layer): m [T, S, P] KV-group attention mass, q [T, H, d] per query head,
key summaries [3, S, P, d], eligibility [T, P] (definitions.pinned). Pages are 16 tokens.
Pinned pages (attention sink + recent window, i.e. existing but not yet eligible) are always
resident and count toward the budget B (pages), as cache budgets do in the eviction papers.

Two families:
* Eviction (permanent drop): a kept set that only shrinks except for pages ageing in. Scores use
  information up to t-1 (the oracle variant reads the future: the ForesightKV upper bound).
  Metric: coverage = share of step-t attention on resident pages (the rest is lost for good).
* Offload: every step attends to a selected set S_t (Quest-style bound with the current query);
  pages live in host memory unless cached on GPU (capacity = Tier 1 + Tier 2). A prefetcher
  decides L steps early which pages to bring in; selected pages not on GPU are fetched on demand
  (a stall). Metrics: stalled pages/step, prefetched pages/step, no-stall coverage.
Method implementations are page-level approximations of the published token-level policies;
see each score function.
"""
from __future__ import annotations

import numpy as np

from kvfetch.metrics.signals import KEY_MAX, KEY_MEAN, KEY_MIN, extrapolate_q


# ----------------------------------------------------------------- helpers
def existing_pages(prompt_len: int, T: int, P: int, page_size: int) -> np.ndarray:
    """[T, P]: page holds at least one token visible at step t (keys [0, prompt_len + t))."""
    return (np.arange(P)[None, :] * page_size) < (prompt_len + np.arange(T)[:, None])


def rank_desc(score: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """[T, S, P] -> int32 rank of each page by descending score among allowed pages (ties by
    index). Rank once, then threshold for any number of budgets with ``mask_from_ranks``."""
    s = np.where(allowed[:, None, :], score, -np.inf)
    order = np.argsort(-s, axis=-1, kind="stable")
    ranks = np.empty(order.shape, np.int32)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(score.shape[-1], dtype=np.int32), order.shape), axis=-1)
    return ranks


def mask_from_ranks(ranks: np.ndarray, B: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Exactly min(B_t, n_allowed_t) allowed pages per (t, s)."""
    return (ranks < np.minimum(B, allowed.sum(-1))[:, None, None]) & allowed[:, None, :]


def topk_abs(score: np.ndarray, B: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """[T, S, P] score, per-step absolute budget B [T], allowed [T, P] -> bool mask."""
    return mask_from_ranks(rank_desc(score, allowed), B, allowed)


def coverage(mask: np.ndarray, m: np.ndarray) -> float:
    """Mean over (t, s) of attention mass on mask / total attention mass."""
    tot = m.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmean(np.where(tot > 0, (m * mask).sum(-1) / tot, np.nan)))


def group_max(x: np.ndarray, S: int) -> np.ndarray:
    """[T, H, P] -> [T, S, P]: max over each GQA group (a page is needed if any head needs it)."""
    T, H, P = x.shape
    return x.reshape(T, S, H // S, P).max(2)


def quest_bound(q: np.ndarray, ks: np.ndarray, scaling: float, chunk: int = 256) -> np.ndarray:
    """Quest upper bound per KV group: [T, H, d] x [3, S, P, d] -> [T, S, P] (fp32), chunked."""
    T, H, d = q.shape
    S, P = ks.shape[1], ks.shape[2]
    g = H // S
    kmin = np.repeat(ks[KEY_MIN].astype(np.float32), g, 0)
    kmax = np.repeat(ks[KEY_MAX].astype(np.float32), g, 0)
    out = np.empty((T, S, P), np.float32)
    kmax_t, kmin_t = kmax.transpose(0, 2, 1), kmin.transpose(0, 2, 1)        # [H, d, P]
    for a in range(0, T, chunk):
        qf = q[a:a + chunk].astype(np.float32).transpose(1, 0, 2)              # [H, t, d]
        b = (np.matmul(np.maximum(qf, 0), kmax_t) + np.matmul(np.minimum(qf, 0), kmin_t)) * scaling
        out[a:a + chunk] = group_max(b.transpose(1, 0, 2), S)                  # batched BLAS, [t, S, P]
    return out


# ----------------------------------------------------------------- eviction family
def evict_scores(name: str, *, m: np.ndarray, t: int, ctx: dict) -> np.ndarray:
    """Keep-priority [S, P] at step t (higher = keep) from information up to t-1 (except oracle)."""
    S, P = m.shape[1], m.shape[2]
    if name == "streaming":          # StreamingLLM: keep the most recent pages
        return np.broadcast_to(np.arange(P, dtype=np.float32), (S, P))
    if name == "h2o":                # H2O: accumulated attention
        return ctx["cum"][t - 1] if t > 0 else ctx["init"]
    if name == "tova":               # TOVA: last step's attention
        return m[t - 1] if t > 0 else ctx["init"]
    if name == "snapkv":             # SnapKV applied at decode time: attention over a recent window
        w = ctx["cfg"]["snapkv_window"]
        if t == 0:
            return ctx["init"]
        lo = max(0, t - w)
        return ctx["cum"][t - 1] - (ctx["cum"][lo - 1] if lo > 0 else 0)
    if name == "raas":               # RaaS: timestamp of the last step the page was important
        return ctx["last_imp"][t - 1].astype(np.float32) if t > 0 else ctx["init"]
    if name == "lazy":               # LazyEviction: keep pages still within their max recurrence interval
        if t == 0:
            return ctx["init"]
        since = (t - 1) - ctx["last_imp"][t - 1]
        mri = np.maximum(ctx["mri"][t - 1], 1)
        return -(since / mri).astype(np.float32)          # most overdue (relative to own MRI) evicted first
    if name == "rkv":                # R-KV: importance minus redundancy (key-mean cosine to other kept pages)
        lam = ctx["cfg"]["rkv_lambda"]
        imp = evict_scores("snapkv", m=m, t=t, ctx=ctx)
        imp = imp / np.maximum(imp.max(-1, keepdims=True), 1e-12)
        red = np.full((S, P), -1.0, np.float32)
        for s_ in range(S):          # only kept-vs-kept similarities matter (<= budget pages)
            idx = np.flatnonzero(ctx["kept"][s_])
            if idx.size > 1:
                red[s_, idx] = ctx["key_sim"][s_][np.ix_(idx, idx)].max(-1)
        return lam * imp - (1 - lam) * red
    if name == "oracle":             # ForesightKV upper bound: remaining future attention
        return ctx["future"][t]
    raise ValueError(name)


def simulate_evict(name: str, *, m: np.ndarray, eligible: np.ndarray, exist: np.ndarray, B: int,
                   ctx: dict) -> float:
    """Permanent eviction at budget B pages (incl. pinned). Evictions happen when pages age into
    eligibility; between such events the kept set is fixed, so coverage is computed in blocks."""
    T, S, P = m.shape
    ctx = dict(ctx, evicted=np.zeros((S, P), bool), kept=None)   # per-run state
    pinned = exist & ~eligible
    kept = np.zeros((S, P), bool)
    resident = np.zeros((T, S, P), bool)
    events = [0] + [t for t in range(1, T) if (eligible[t] & ~eligible[t - 1]).any()] + [T]
    for a, b in zip(events[:-1], events[1:]):
        kept |= eligible[a][None, :] & ~ctx["evicted"]
        sel_budget = max(0, B - int(pinned[a].sum()))
        excess = kept.sum(-1) - sel_budget
        if (excess > 0).any():
            ctx["kept"] = kept
            pr = np.where(kept, evict_scores(name, m=m, t=a, ctx=ctx), np.inf)
            order = np.argsort(pr, axis=-1, kind="stable")
            ranks = np.empty_like(order)
            np.put_along_axis(ranks, order, np.broadcast_to(np.arange(P), order.shape), axis=-1)
            drop = kept & (ranks < excess[:, None])
            kept &= ~drop
            ctx["evicted"] |= drop
        resident[a:b] = kept[None] | pinned[a:b, None, :]
    return coverage(resident, m)


def evict_context(m: np.ndarray, init: np.ndarray, key_mean: np.ndarray, exist: np.ndarray, cfg: dict) -> dict:
    """Precomputed per-trace state for the eviction scores."""
    T, S, P = m.shape
    cum = np.cumsum(m, axis=0)
    n_exist = exist.sum(-1)[:, None, None]
    imp = m >= cfg["important_mult"] / np.maximum(n_exist, 1)        # above-uniform attention
    steps = np.arange(T, dtype=np.int32)[:, None, None]
    first = np.argmax(exist, axis=0).astype(np.int32)                 # creation step per page
    last_imp_raw = np.maximum.accumulate(np.where(imp, steps, np.int32(-1)), axis=0)
    last_imp = np.where(last_imp_raw < 0, first[None, None, :], last_imp_raw)
    # max recurrence interval so far: largest gap between consecutive important steps
    prev = np.vstack([np.full((1, S, P), -1, np.int32), last_imp_raw[:-1]])
    gap = np.where(imp & (prev >= 0), steps - prev, np.int32(0))
    mri = np.maximum.accumulate(gap, axis=0)
    kn = key_mean / np.maximum(np.linalg.norm(key_mean, axis=-1, keepdims=True), 1e-6)
    key_sim = np.einsum("spd,sqd->spq", kn, kn)
    key_sim[:, np.arange(P), np.arange(P)] = -1.0                    # ignore self
    future = cum[-1][None] - np.vstack([np.zeros((1, S, P)), cum[:-1]])  # sum_{s >= t} m
    return {"cum": cum, "init": init, "last_imp": last_imp, "mri": mri, "key_sim": key_sim,
            "future": future, "cfg": cfg}


# ----------------------------------------------------------------- offload family
def simulate_offload(*, m: np.ndarray, sel: np.ndarray, prefetch: np.ndarray | None, eligible: np.ndarray,
                     capacity: int) -> dict:
    """GPU cache of ``capacity`` selectable pages (LRU), attended set sel [T, S, P], prefetch set
    decided early for each step (or None = on demand only). Returns per-KV-head step means."""
    T, S, P = m.shape
    gpu = np.zeros((S, P), bool)
    last_used = np.full((S, P), -1, np.int64)
    stalls = pref = 0.0
    covered = np.zeros((T, S, P), bool)
    for t in range(T):
        el = eligible[t][None, :]
        if prefetch is not None:
            new = prefetch[t] & ~gpu & el
            pref += new.sum()
            gpu |= new
        hit = sel[t] & gpu
        miss = sel[t] & ~gpu
        stalls += miss.sum()
        covered[t] = hit
        gpu |= miss
        last_used[sel[t]] = t
        gpu &= el                                     # pages that left eligibility are pinned, not cached
        over = gpu.sum(-1) - capacity
        for s_ in np.flatnonzero(over > 0):          # LRU: drop the least recently attended, unselected pages
            pr = np.where(gpu[s_] & ~sel[t, s_], last_used[s_], np.iinfo(np.int64).max)
            gpu[s_, np.argpartition(pr, over[s_] - 1)[:over[s_]]] = False
    n = T * S
    return {"stall_pages_per_step": stalls / n, "prefetch_pages_per_step": pref / n,
            "nostall_coverage_of_selected": coverage(covered, m * sel)}


def prefetch_query(name: str, q: np.ndarray, L: int, cfg: dict) -> np.ndarray | None:
    """Row t = the query a prefetcher uses to choose pages FOR step t, decided at step t - L.
    None for on-demand (no prefetch); the true q for the OasisKV upper bound (perfect draft)."""
    if name == "on_demand":
        return None
    if name == "oasis_ub":           # perfect draft: the true query of step t is known L steps early
        return q
    if name == "stale_q":            # InfiniGen-like: rank with the latest known query (q_{t-L})
        src = q
    elif name == "q_trend":          # ours: extrapolate the latest known query L steps ahead
        src = extrapolate_q(q, L, cfg["q_slope_steps"])
    else:
        raise ValueError(name)
    return np.concatenate([np.repeat(src[:1], L, 0), src[:-L]]) if L > 0 else src


def prefetch_sets(name: str, *, q: np.ndarray, ks: np.ndarray, scaling: float, sel: np.ndarray,
                  B: np.ndarray, eligible: np.ndarray, L: int, cfg: dict) -> np.ndarray | None:
    """Pages to prefetch FOR step t, decided at step t - L (information up to t - L)."""
    if name == "oasis_ub":
        return sel
    qp = prefetch_query(name, q, L, cfg)
    return None if qp is None else topk_abs(quest_bound(qp, ks, scaling), B, eligible)
