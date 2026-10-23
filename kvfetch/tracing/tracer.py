"""Attention tracing via the HF AttentionInterface (no custom kernels).

The traced attention function delegates the actual attention output to HF's SDPA
implementation, so model outputs are bit-identical to an untraced SDPA run. For
traced layers it additionally recomputes softmax(q K^T * scale) in fp32 for the
queries of interest and reduces it to per-page mass.

Step convention (also recorded in every trace file):
  step s in [0, T) is the attention of the query that PRODUCES generated token s+1.
  s = 0 is the last prompt token (computed during prefill); s >= 1 is the query of
  generated token s. Keys visible at step s: positions [0, prompt_len + s).
Head convention: query heads, in HF order; query head h reads KV head h // G.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

TRACED_ATTN = "kvtrace_sdpa"
_ACTIVE: "AttentionTracer | None" = None


def active_tracer() -> "AttentionTracer | None":
    return _ACTIVE


def make_traced(inner):
    """Wrap an HF attention function so the active tracer observes (query, key) first."""
    def traced(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        tracer = _ACTIVE
        if tracer is not None:
            if kwargs.get("sliding_window") is not None:
                raise NotImplementedError("tracer assumes full attention (sliding_window set)")
            tracer.observe(module.layer_idx, query, key, scaling)
        return inner(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
    return traced


def register_traced_attention(name: str = TRACED_ATTN, inner=sdpa_attention_forward, mask: str = "sdpa") -> None:
    AttentionInterface.register(name, make_traced(inner))
    # Without a mask function HF skips mask creation for unknown implementations, and
    # SDPA's is_causal=True is top-left aligned: wrong for chunked prefill.
    AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[mask])


def page_reduce(probs: torch.Tensor, page_size: int, n_pages: int) -> torch.Tensor:
    """[..., N] token probabilities -> [..., n_pages] page mass (zero-padded)."""
    n = probs.shape[-1]
    pad = n_pages * page_size - n
    if pad < 0:
        raise ValueError(f"{n} keys exceed {n_pages} pages of {page_size}")
    return F.pad(probs, (0, pad)).view(*probs.shape[:-1], n_pages, page_size).sum(-1)


def attention_probs(q: torch.Tensor, k: torch.Tensor, scaling: float,
                    q_pos: torch.Tensor | None = None) -> torch.Tensor:
    """q: [Hq, nq, d], k: [Hkv, N, d] -> fp32 probs [Hq, nq, N].
    ``q_pos`` (absolute positions of the queries) applies the causal mask."""
    hq, nq, d = q.shape
    hkv, n, _ = k.shape
    g = hq // hkv
    s = torch.bmm(q.float().reshape(hkv, g * nq, d), k.float().transpose(1, 2)).mul_(scaling)
    s = s.view(hq, nq, n)
    if q_pos is not None:
        future = torch.arange(n, device=k.device)[None, :] > q_pos[:, None]  # [nq, N]
        s.masked_fill_(future[None], float("-inf"))
    return torch.softmax(s, dim=-1)


class AttentionTracer:
    def __init__(self, *, layers: list[int], num_heads: int, num_kv_heads: int, page_size: int,
                 prompt_len: int, decode_steps: int, obs_window: int, flush_every: int,
                 device: torch.device | str, mass_dtype: str = "float16", head_dim: int | None = None,
                 record_q: bool = False, head_reduce: str = "none"):
        self.layers = list(layers)
        self.slot = {l: i for i, l in enumerate(self.layers)}
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.page_size, self.prompt_len, self.T = page_size, prompt_len, decode_steps
        # keys visible at the last step: prompt_len + T - 1
        self.n_pages = math.ceil((prompt_len + decode_steps - 1) / page_size)
        self.obs_window = min(obs_window, prompt_len)
        self.flush_every = flush_every
        self.device = torch.device(device)
        self.tdtype = getattr(torch, mass_dtype)
        # "kv_mean": store mass averaged over each GQA group (the unit tiers act on), 1/G the size.
        # Queries (record_q) always stay per query head.
        if head_reduce not in ("none", "kv_mean"):
            raise ValueError(head_reduce)
        self.head_reduce = head_reduce
        L, P = len(self.layers), self.n_pages
        H = num_kv_heads if head_reduce == "kv_mean" else num_heads
        self.mass = np.zeros((decode_steps, L, H, P), dtype=np.dtype(mass_dtype))
        self.obs = torch.zeros((self.obs_window, L, H, P), dtype=self.tdtype, device=self.device)
        self._buf = torch.zeros((flush_every, L, H, P), dtype=self.tdtype, device=self.device)
        self._buf_start = 1  # decode step held in _buf[0]
        # optional: post-RoPE decode queries, same step convention as mass (q[0] = last prompt token)
        self.record_q = record_q
        if record_q:
            if head_dim is None:
                raise ValueError("record_q needs head_dim")
            self.q = np.zeros((decode_steps, L, num_heads, head_dim), dtype=np.dtype(mass_dtype))
            self._qbuf = torch.zeros((flush_every, L, num_heads, head_dim), dtype=self.tdtype, device=self.device)
        self.key_summary: np.ndarray | None = None
        self._mode: str | None = None
        self._q_start = 0
        self._step = 0
        self.steps_recorded = 0

    # -- context management -------------------------------------------------
    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None:
            raise RuntimeError("another tracer is active")
        _ACTIVE = self
        return self

    def __exit__(self, *exc):
        global _ACTIVE
        _ACTIVE = None
        return False

    def begin_prefill_chunk(self, q_start: int) -> None:
        self._mode, self._q_start = "prefill", q_start

    def finish_prefill(self) -> None:
        self._mode = None
        if self.obs_window > 0:
            self.mass[0] = self.obs[-1].cpu().numpy()
            self.steps_recorded = 1

    def begin_decode_step(self, step: int) -> None:
        if not 1 <= step < self.T:
            raise IndexError(step)
        self._mode, self._step = "decode", step

    def end_decode_step(self) -> None:
        self._mode = None
        self.steps_recorded = self._step + 1
        if self._step - self._buf_start + 1 == self.flush_every:
            self.flush()

    def flush(self) -> None:
        n = self.steps_recorded - self._buf_start
        if n > 0:
            self.mass[self._buf_start:self._buf_start + n] = self._buf[:n].cpu().numpy()
            if self.record_q:
                self.q[self._buf_start:self._buf_start + n] = self._qbuf[:n].cpu().numpy()
        self._buf_start = self.steps_recorded

    @torch.no_grad()
    def finalize_key_summary(self, cache) -> None:
        """Per-page post-RoPE key mean / min / max for the traced layers, from the final cache:
        [L, 3, KVH, P, d]. Keys never change once written, so the summary of a page is valid at
        every step after the page is complete. The partial last page is padded by replication."""
        P, ps = self.n_pages, self.page_size
        out = None
        for slot, layer in enumerate(self.layers):
            k = cache.layers[layer].keys[0].float()                     # [KVH, N, d]
            kvh, n, d = k.shape
            x = F.pad(k.transpose(1, 2), (0, P * ps - n), mode="replicate").view(kvh, d, P, ps)
            stats = torch.stack([x.mean(-1), x.amin(-1), x.amax(-1)]).transpose(-1, -2)  # [3, KVH, P, d]
            if out is None:
                out = np.zeros((len(self.layers), 3, kvh, P, d), dtype=self.mass.dtype)
            out[slot] = stats.to(self.tdtype).cpu().numpy()
        self.key_summary = out

    def _reduce(self, mass: torch.Tensor) -> torch.Tensor:
        """[Hq, ..., P] -> [KVH, ..., P] mean over each GQA group when head_reduce == kv_mean."""
        if self.head_reduce == "none":
            return mass
        g = self.num_heads // self.num_kv_heads
        return mass.reshape(self.num_kv_heads, g, *mass.shape[1:]).mean(1)

    def obs_numpy(self) -> np.ndarray:
        return self.obs.cpu().numpy()

    # -- recording ----------------------------------------------------------
    def observe(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, scaling: float | None) -> None:
        slot = self.slot.get(layer_idx)
        if slot is None or self._mode is None:
            return
        if query.shape[0] != 1:
            raise NotImplementedError("tracer supports batch size 1")
        scaling = scaling if scaling is not None else query.shape[-1] ** -0.5
        q, k = query[0], key[0]  # [Hq, nq, d], [Hkv, N, d]
        if self._mode == "decode":
            if q.shape[1] != 1:
                raise RuntimeError("decode step with more than one query")
            probs = attention_probs(q, k, scaling)[:, 0]  # [Hq, N]
            self._buf[self._step - self._buf_start, slot] = self._reduce(
                page_reduce(probs, self.page_size, self.n_pages)).to(self.tdtype)
            if self.record_q:
                self._qbuf[self._step - self._buf_start, slot] = q[:, 0].to(self.tdtype)
        else:
            lo = self.prompt_len - self.obs_window
            pos = torch.arange(self._q_start, self._q_start + q.shape[1], device=q.device)
            rows = (pos >= lo) & (pos < self.prompt_len)
            if not bool(rows.any()):
                return
            probs = attention_probs(q[:, rows], k, scaling, q_pos=pos[rows])  # [Hq, nr, N]
            mass = self._reduce(page_reduce(probs, self.page_size, self.n_pages)).to(self.tdtype)  # [H, nr, P]
            self.obs[(pos[rows] - lo), slot] = mass.transpose(0, 1)
            last = pos == self.prompt_len - 1
            if self.record_q and bool(last.any()):  # step 0 = the last prompt token's query
                self.q[0, slot] = q[:, last][:, 0].to(self.tdtype).cpu().numpy()
