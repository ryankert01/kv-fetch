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


def traced_sdpa_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    tracer = _ACTIVE
    if tracer is not None:
        if kwargs.get("sliding_window") is not None:
            raise NotImplementedError("tracer assumes full attention (sliding_window set)")
        tracer.observe(module.layer_idx, query, key, scaling)
    return sdpa_attention_forward(module, query, key, value, attention_mask,
                                  dropout=dropout, scaling=scaling, **kwargs)


def register_traced_attention() -> None:
    AttentionInterface.register(TRACED_ATTN, traced_sdpa_attention)
    # Without a mask function HF skips mask creation for unknown implementations, and
    # SDPA's is_causal=True is top-left aligned: wrong for chunked prefill.
    AttentionMaskInterface.register(TRACED_ATTN, ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["sdpa"])


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
                 device: torch.device | str, mass_dtype: str = "float16"):
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
        L, H, P = len(self.layers), num_heads, self.n_pages
        self.mass = np.zeros((decode_steps, L, H, P), dtype=np.dtype(mass_dtype))
        self.obs = torch.zeros((self.obs_window, L, H, P), dtype=self.tdtype, device=self.device)
        self._buf = torch.zeros((flush_every, L, H, P), dtype=self.tdtype, device=self.device)
        self._buf_start = 1  # decode step held in _buf[0]
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
        self._buf_start = self.steps_recorded

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
            self._buf[self._step - self._buf_start, slot] = page_reduce(probs, self.page_size, self.n_pages).to(self.tdtype)
        else:
            lo = self.prompt_len - self.obs_window
            pos = torch.arange(self._q_start, self._q_start + q.shape[1], device=q.device)
            rows = (pos >= lo) & (pos < self.prompt_len)
            if not bool(rows.any()):
                return
            probs = attention_probs(q[:, rows], k, scaling, q_pos=pos[rows])  # [Hq, nr, N]
            mass = page_reduce(probs, self.page_size, self.n_pages).to(self.tdtype)  # [Hq, nr, P]
            self.obs[(pos[rows] - lo), slot] = mass.transpose(0, 1)
