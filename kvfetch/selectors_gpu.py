"""Torch versions of the page-selection replay (kvfetch/selectors.py, kvfetch/sim.py), for E2b.

Same rules, run on the GPU: ranking [T, S, P] scores on the CPU dominated E2b's time, and the
stateful selectors replay many variants (specs x budgets) in one step loop instead of one Python
loop per variant. tests/test_selectors_gpu.py checks each function against its numpy original.
"""
from __future__ import annotations

import torch

from kvfetch.metrics.importance import budget_pages
from kvfetch.selectors import _RECENCY_TIEBREAK

_NEG = float("-inf")


def _ranks(score: torch.Tensor) -> torch.Tensor:
    """Rank of each entry along the last axis by descending score, ties by index (int64)."""
    order = torch.argsort(-score, dim=-1, stable=True)
    pos = torch.arange(score.shape[-1], device=score.device).expand_as(order)
    return torch.empty_like(order).scatter_(-1, order, pos)


def rank_desc(score: torch.Tensor, allowed: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """sim.rank_desc: [T, S, P] -> int16 rank among allowed pages ([T, P]), chunked over T."""
    out = torch.empty(score.shape, dtype=torch.int16, device=score.device)
    for a in range(0, score.shape[0], chunk):
        s = score[a:a + chunk].masked_fill(~allowed[a:a + chunk, None, :], _NEG)
        out[a:a + chunk] = _ranks(s).to(torch.int16)
    return out


def mask_from_ranks(ranks: torch.Tensor, B: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """sim.mask_from_ranks: exactly min(B_t, n_allowed_t) allowed pages per (t, s)."""
    return (ranks < torch.minimum(B, allowed.sum(-1))[:, None, None]) & allowed[:, None, :]


def page_shares(score: torch.Tensor, eligible: torch.Tensor, kind: str) -> torch.Tensor:
    """selectors.page_shares."""
    allowed = eligible[:, None, :]
    if kind == "logit":
        z = score.masked_fill(~allowed, _NEG)
        mx = torch.where(allowed.any(-1, keepdim=True), z.amax(-1, keepdim=True), torch.zeros((), device=z.device))
        z = torch.exp(z - mx)
    elif kind == "mass":
        z = torch.where(allowed, score, torch.zeros((), device=score.device))
    else:
        raise ValueError(kind)
    tot = z.sum(-1, keepdim=True)
    return torch.where(tot > 0, z / tot.clamp(min=torch.finfo(z.dtype).tiny), torch.zeros((), device=z.device))


def static_ranks(score: torch.Tensor, eligible: torch.Tensor, kind: str = "logit") -> dict[str, torch.Tensor]:
    """The budget-independent ranks behind selectors.static_select: per KV head, of the summed page
    shares, and of the best score any head gives a page (the vote tie-break)."""
    best = score.masked_fill(~eligible[:, None, :], _NEG).amax(1, keepdim=True)
    return {"per_head": rank_desc(score, eligible),
            "sum": rank_desc(page_shares(score, eligible, kind).sum(1, keepdim=True), eligible),
            "best": rank_desc(best, eligible)[:, 0]}


def select_from_ranks(ranks: dict, eligible: torch.Tensor, b: torch.Tensor, mode: str) -> torch.Tensor:
    """selectors.static_select from precomputed ``static_ranks``: bool [T, S, P]."""
    shape = ranks["per_head"].shape
    if mode == "per_head":
        return mask_from_ranks(ranks["per_head"], b, eligible)
    if mode == "sum":
        return mask_from_ranks(ranks["sum"], b, eligible).expand(shape)
    if mode == "vote":
        nom = mask_from_ranks(ranks["per_head"], b, eligible).sum(1)                         # [T, P]
        key = (nom * (shape[-1] + 1) - ranks["best"].long()).float()[:, None, :]
        return mask_from_ranks(rank_desc(key, eligible), b, eligible).expand(shape)
    raise ValueError(mode)


def _pick(score: torch.Tensor, k: torch.Tensor, shared) -> torch.Tensor:
    """selectors._topk / _shared_topk / _sum_topk for K variants at once: score [K, S, P]
    (-inf = not allowed), k [K] -> bool [K, S, P]."""
    ok = torch.isfinite(score)
    if not shared:
        return (_ranks(score) < k[:, None, None]) & ok
    if shared == "sum":
        tot = torch.where(ok.any(1), torch.where(ok, score, 0.0).sum(1), _NEG)              # [K, P]
        sel = (_ranks(tot) < k[:, None]) & torch.isfinite(tot)
    elif shared in (True, "vote"):
        nom = ((_ranks(score) < k[:, None, None]) & ok).sum(1)                              # [K, P]
        best = torch.where(ok, score, _NEG).amax(1)
        o1 = torch.argsort(-best, dim=-1, stable=True)                                      # then by best score
        order = o1.gather(-1, torch.argsort(-nom.gather(-1, o1), dim=-1, stable=True))      # nominations first
        pos = torch.arange(score.shape[-1], device=score.device).expand_as(order)
        sel = (torch.empty_like(order).scatter_(-1, order, pos) < k[:, None]) & torch.isfinite(best)
    else:
        raise ValueError(shared)
    return sel[:, None, :].expand_as(score)


def replay(specs: list[dict], sources: torch.Tensor, src: list[int], b: torch.Tensor, *, m: torch.Tensor,
           init: torch.Tensor, eligible: torch.Tensor, exist: torch.Tensor, shared=False,
           qm: torch.Tensor | None = None):
    """selectors.run_selector for K variants in one step loop. Variant k follows ``specs[k]``
    (its ``shared`` key is ignored: all variants use ``shared``), scores pages with
    ``sources[src[k]]`` ([U, T, S, P]) as its Quest bounds and has budget b[k] ([K, T]).
    Yields (t, chosen [K, S, P] bool), the selection at step t (pinned pages not included)."""
    T, S, P = m.shape
    dev = m.device
    K = len(specs)
    par = lambda key, default: torch.tensor([float(default if s.get(key) is None else s[key]) for s in specs], device=dev)
    r_share, q_share, w = par("recent_share", 0.0), par("quest_share", 1.0), par("history_weight", 0.5)
    mm = par("mean_mix", 0.0)[:, None, None]
    is_fuse = torch.tensor([bool(s.get("fuse")) for s in specs], device=dev)
    is_max = torch.tensor([s.get("fuse") == "max" for s in specs], device=dev)[:, None, None]
    peak = torch.tensor([s.get("history", "last") == "peak" for s in specs], device=dev)[:, None, None]
    if qm is None and bool((mm > 0).any()):
        raise ValueError("mean_mix needs qm")
    src_t = torch.tensor(src, device=dev)
    hist = init.float()[None].repeat(K, 1, 1)
    tiebreak = _RECENCY_TIEBREAK * torch.arange(P, device=dev, dtype=torch.float32) / P
    zero = torch.zeros((), device=dev)
    n_allowed = eligible.sum(-1)
    from_end = eligible.flip(1).long().cumsum(1).flip(1)                                    # 1 = newest allowed page
    for t in range(T):
        allowed = eligible[t]
        al = allowed[None, None, :]
        bt = torch.minimum(b[:, t], n_allowed[t])
        n_rec = torch.round(r_share * bt).long()
        chosen = (allowed & (from_end[t] <= n_rec[:, None]))[:, None, :].expand(K, S, P).clone()
        rem = bt - n_rec
        qbt = sources[src_t, t]
        if qm is not None:
            qbt = torch.where(mm > 0, (1.0 - mm) * qbt + mm * qm[t], qbt)
        z = qbt.masked_fill(~al, _NEG)
        z = torch.exp(z - z.amax(-1, keepdim=True))
        qs = z / z.sum(-1, keepdim=True)
        f = torch.where(is_max, torch.maximum(qs, hist), (1.0 - w)[:, None, None] * qs + w[:, None, None] * hist)
        chosen |= _pick(torch.where(al & ~chosen, f + tiebreak, _NEG), torch.where(is_fuse, rem, 0), shared)
        n_q = torch.where(is_fuse, 0, torch.round(q_share * rem).long())
        chosen |= _pick(torch.where(al & ~chosen, qbt, _NEG), n_q, shared)
        n_h = (bt - chosen.sum(-1).amax(-1)).clamp(min=0)
        chosen |= _pick(torch.where(al & ~chosen, hist + tiebreak, _NEG), n_h, shared)
        yield t, chosen
        att = chosen | (exist[t] & ~allowed)[None, None, :]
        mt = torch.where(att, m[t][None], zero)
        tot = mt.sum(-1, keepdim=True)
        obs = torch.where(tot > 0, mt / tot.clamp(min=torch.finfo(mt.dtype).tiny), zero)
        hist = torch.where(peak, torch.maximum(hist, obs), torch.where(att, obs, hist))


def revived_weight(m: torch.Tensor, eligible: torch.Tensor, rc: dict) -> torch.Tensor:
    """selectors.revived_weight: attention mass on revived page-steps [T, S, P], zero elsewhere."""
    T, S, P = m.shape
    n = eligible.sum(-1).tolist()
    B = torch.tensor([budget_pages(rc["budget_frac"], k) if k > 0 else 0 for k in n], device=m.device)
    s = m.float().masked_fill(~eligible[:, None, :], _NEG)
    srt = torch.sort(s, dim=-1, descending=True).values
    cut = srt.gather(-1, (B - 1).clamp(min=0)[:, None, None].expand(T, S, 1))[..., 0]
    r = torch.where((cut[..., None] > 0) & eligible[:, None, :], m / cut[..., None].clamp(min=torch.finfo(m.dtype).tiny), 0.0)
    r = r.reshape(T, S * P)
    high = torch.empty(r.shape, dtype=torch.bool, device=m.device)
    high[0] = r[0] >= 1
    for t in range(1, T):                                                   # hysteresis (trajectory.high_state)
        high[t] = torch.where(high[t - 1], r[t] >= rc["exit_ratio"], r[t] >= 1)
    steps = torch.arange(T, device=m.device)[:, None]
    neg1 = torch.full((1, S * P), -1, device=m.device, dtype=torch.long)
    last_high = torch.cummax(torch.where(high, steps, -1), 0).values                       # revival.reentries
    prev_last = torch.cat([neg1, last_high[:-1]])
    prev_high = torch.cat([torch.zeros_like(high[:1]), high[:-1]])
    reentry = high & ~prev_high & (prev_last >= 0)
    gap = steps - prev_last - 1
    entry = high & ~prev_high                                                              # revival.revived_mask
    last_entry = torch.cummax(torch.where(entry, steps, -1), 0).values
    is_rev = reentry & (gap >= rc["dormancy_main"])
    flag = is_rev.gather(0, last_entry.clamp(min=0)) & (last_entry >= 0)
    return torch.where((high & flag).reshape(T, S, P), m, 0.0)
