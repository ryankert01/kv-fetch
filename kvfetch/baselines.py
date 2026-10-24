"""Published KV selectors emulated offline with full keys (EXPERIMENTS.md, E2).

Ground truth is the exact token attention recomputed from the traced queries and the replayed
keys (scripts/dump_keys.py): a [c, S, N] = mean over each GQA group of softmax(q k^T / sqrt(d)).
Every selector picks, for each step and KV head, a set of tokens on top of the pinned tokens
(sink + recent window, as in E1) within the same token budget; coverage is the attention mass on
selected + pinned tokens. Granularity differs by method: pages of 16 tokens (Quest, ArkVale, ours),
chunks of 8 (ShadowKV), single tokens (InfiniGen, LessIsMore, exact top-k).

Emulations are optimistic where the original approximates further (noted per function), so a
selector that beats them here beats the published method's best case on this metric.
"""
from __future__ import annotations

import torch


def logits_by_group(q: torch.Tensor, k: torch.Tensor, scaling: float) -> torch.Tensor:
    """q [c, H, d], k [S, M, d] -> per-head logits [c, S, g, M] (query head h reads KV head h // g)."""
    c, H, d = q.shape
    S = k.shape[0]
    return torch.einsum("csgd,smd->csgm", q.reshape(c, S, H // S, d), k) * scaling


def attention_by_group(q: torch.Tensor, k: torch.Tensor, n_keys: torch.Tensor, scaling: float) -> torch.Tensor:
    """Exact attention per query head: q [c, H, d], k [S, N, d], n_keys [c] -> [c, S, g, N]
    (zero beyond each step's n_keys)."""
    logits = logits_by_group(q, k, scaling)
    valid = torch.arange(k.shape[1], device=q.device)[None, :] < n_keys[:, None]
    return torch.softmax(logits.masked_fill(~valid[:, None, None, :], float("-inf")), -1)


def topk_mask(score: torch.Tensor, k: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """score [c, S, M] (higher is better), k [c] per-step count, allowed [c, M] -> bool [c, S, M]."""
    s = score.masked_fill(~allowed[:, None, :], float("-inf"))
    kmax = int(k.max().item()) if k.numel() else 0
    out = torch.zeros_like(s, dtype=torch.bool)
    if kmax <= 0:
        return out
    idx = torch.topk(s, min(kmax, s.shape[-1]), dim=-1).indices                     # [c, S, kmax]
    keep = torch.arange(idx.shape[-1], device=s.device)[None, None, :] < k[:, None, None]
    out.scatter_(-1, idx, keep.expand_as(idx))
    return out & allowed[:, None, :]


def expand(mask_units: torch.Tensor, unit: int, n_tokens: int) -> torch.Tensor:
    """[c, S, U] unit mask -> [c, S, n_tokens] token mask (unit = tokens per unit)."""
    return mask_units.repeat_interleave(unit, -1)[..., :n_tokens]


def unit_summaries(k: torch.Tensor, unit: int) -> dict[str, torch.Tensor]:
    """Per-unit key summaries over complete and partial units: mean, min, max, ball radius, and
    the lowest cosine similarity of a key to its unit mean (ShadowKV's outlier test)."""
    S, N, d = k.shape
    U = -(-N // unit)
    pad = U * unit - N
    kp = torch.nn.functional.pad(k, (0, 0, 0, pad))
    valid = torch.nn.functional.pad(torch.ones(N, device=k.device, dtype=torch.bool), (0, pad)).reshape(U, unit)
    ku = kp.reshape(S, U, unit, d)
    cnt = valid.sum(-1).clamp(min=1).to(k.dtype)[None, :, None]
    mean = (ku * valid[None, :, :, None]).sum(2) / cnt
    big = torch.finfo(k.dtype).max
    kmin = ku.masked_fill(~valid[None, :, :, None], big).amin(2)
    kmax = ku.masked_fill(~valid[None, :, :, None], -big).amax(2)
    dist = (ku - mean[:, :, None, :]).norm(dim=-1).masked_fill(~valid[None], 0.0)
    cos = torch.nn.functional.cosine_similarity(ku, mean[:, :, None, :].expand_as(ku), dim=-1).masked_fill(~valid[None], 1.0)
    return {"mean": mean, "min": kmin, "max": kmax, "radius": dist.amax(-1), "min_cos": cos.amin(-1)}


def quest_scores(q: torch.Tensor, summ: dict, scaling: float) -> torch.Tensor:
    """Quest: max over the group of sum_d max(q_d kmin_d, q_d kmax_d). [c, S, U]."""
    c, H, d = q.shape
    S = summ["max"].shape[0]
    qg = q.reshape(c, S, H // S, d)
    b = (torch.einsum("csgd,sud->csgu", qg.clamp(min=0), summ["max"])
         + torch.einsum("csgd,sud->csgu", qg.clamp(max=0), summ["min"])) * scaling
    return b.amax(2)


def arkvale_scores(q: torch.Tensor, summ: dict, scaling: float) -> torch.Tensor:
    """ArkVale ball digest: q . center + |q| * radius, max over the group. [c, S, U]."""
    c, H, d = q.shape
    S = summ["mean"].shape[0]
    qg = q.reshape(c, S, H // S, d)
    b = torch.einsum("csgd,sud->csgu", qg, summ["mean"]) + qg.norm(dim=-1)[..., None] * summ["radius"][None, :, None, :]
    return (b * scaling).amax(2)


def landmark_scores(q: torch.Tensor, summ: dict, scaling: float) -> torch.Tensor:
    """ShadowKV landmarks: q . chunk-mean key, max over the group. [c, S, U]. (ShadowKV also
    approximates the keys it then attends with a low-rank projection; here they are exact.)"""
    return logits_by_group(q, summ["mean"], scaling).amax(2)


def partial_channel_scores(q: torch.Tensor, k: torch.Tensor, channels: torch.Tensor, scaling: float) -> torch.Tensor:
    """InfiniGen-style speculation with r channels per query head: estimated attention from
    partial logits, averaged over the group. channels [H, r]. [c, S, N]. (InfiniGen speculates
    from the previous layer's input; the current query used here is the optimistic case.)"""
    c, H, d = q.shape
    S, N, _ = k.shape
    g = H // S
    qs = torch.gather(q, 2, channels[None].expand(c, -1, -1))                       # [c, H, r]
    kh = k.repeat_interleave(g, 0)                                                   # [H, N, d]
    ks = torch.gather(kh, 2, channels[:, None, :].expand(-1, N, -1))                 # [H, N, r]
    return torch.einsum("chr,hnr->chn", qs, ks).reshape(c, S, g, N) * scaling


def pick_channels(q0: torch.Tensor, k_prompt: torch.Tensor, r: int) -> torch.Tensor:
    """Channels with the largest |q_d| * mean |k_d| over the prompt, per query head. [H, r]."""
    H, d = q0.shape
    S = k_prompt.shape[0]
    imp = q0.abs() * k_prompt.abs().mean(1).repeat_interleave(H // S, 0)
    return torch.topk(imp, r, dim=-1).indices


def shared_token_set(att_heads: torch.Tensor, k: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """LessIsMore unified selection: every query head nominates its top-k tokens; tokens are
    ranked by nomination count, then by the highest attention any head gives them; the top k
    form one set for all heads. att_heads [c, Hq, N]; returns bool [c, N]."""
    c, Hq, N = att_heads.shape
    a = att_heads.masked_fill(~allowed[:, None, :], -1.0)
    kmax = int(k.max().item()) if k.numel() else 0
    out = torch.zeros((c, N), dtype=torch.bool, device=a.device)
    if kmax <= 0:
        return out
    kk = min(kmax, N)
    idx = torch.topk(a, kk, dim=-1).indices
    keep = torch.arange(kk, device=a.device)[None, None, :] < k[:, None, None]
    nom = torch.zeros_like(a).scatter_(-1, idx, keep.expand_as(idx).to(a.dtype)).sum(1)   # [c, N]
    best = a.amax(1).clamp(min=0)                                                   # in [0, 1]
    rank = (nom + best).masked_fill(~allowed, -1.0)                                 # count first, then best
    sel = torch.topk(rank, kk, dim=-1).indices
    keep2 = torch.arange(kk, device=a.device)[None, :] < k[:, None]
    out.scatter_(-1, sel, keep2)
    return out & allowed
