"""Batched sampling with page-sparse attention (EXPERIMENTS.md, E3: online accuracy).

Each slot of the batch holds one sequence with its full KV cache on the GPU (the host tier is
simulated: E3 measures accuracy, not speed). At every decode step and layer a method picks one set
of pages (``page_size`` tokens = one vLLM block) for all KV heads of the layer, within the token
budget left after the pinned pages (sink page + recent window, as in E1/E2b); attention then reads
only those pages and the pinned ones. Methods (``Method.name``):
  full    dense attention (HF SDPA), the reference,
  quest   Quest bounds per KV head, one set by nomination counts (E2b's best Quest mode, ``vote``),
  ours    a ``selectors.candidates`` spec with one set by summed scores (E2b's ``sum``),
  snapkv  SnapKV-style permanent eviction: when pages age into eligibility, keep the pages with the
          most attention over the last ``window`` steps (summed over KV heads), drop the rest,
  oracle  the pages with the most true attention, summed over KV heads (the page-set ceiling).
History (ours) and SnapKV's window read only the attention the method itself computed, renormalised
over the pages it attended (the observability rule of E1/E2b).

Slots run at their own lengths (continuous batching): positions, cache writes and masks are per
slot, so a finished sequence is replaced by a new prompt without padding. Prefill is dense.

Hybrid models (``config.layer_types``, e.g. Qwen3.5: Gated DeltaNet + full attention): only the
full-attention layers hold KV and are page-selected; the linear-attention layers' conv and recurrent
states live in per-slot buffers that this engine also serves (the HF linear-attention cache protocol).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface

from kvfetch.selectors import _RECENCY_TIEBREAK

SPARSE_ATTN = "kvfetch_sparse"
METHODS = ("full", "quest", "ours", "snapkv", "oracle")
_NEG = float("-inf")
_UNIT_TIEBREAK = 1e-4      # int4p2: the two pages of a unit share its bound; prefer the later one
_ENGINE: "SparseEngine | None" = None


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    if kwargs.get("sliding_window") is not None:
        raise NotImplementedError("sparse decoding assumes full attention layers")
    return _ENGINE.attend(module.layer_idx, query, key, value, scaling)


class _LinearState:
    """One linear-attention layer's buffers, shaped as HF's ``LinearAttentionLayer`` exposes them."""
    record_past = False

    def __init__(self, conv: torch.Tensor, recurrent: torch.Tensor):
        self.conv_states, self.recurrent_states = {0: conv}, {0: recurrent}

    def reset_slot(self, b: int) -> None:
        self.conv_states[0][b].zero_()
        self.recurrent_states[0][b].zero_()


def register_sparse_attention() -> None:
    """Make ``attn_implementation=SPARSE_ATTN`` route every attention layer to the active engine.
    Masks are never built: callers pass ``attention_mask={"full_attention": None}``."""
    AttentionInterface.register(SPARSE_ATTN, _attention)


@dataclass
class Method:
    name: str
    budget_tokens: int | None = None             # total attended tokens incl. pinned (None: full)
    spec: dict = field(default_factory=dict)     # ours: a selectors.candidates entry
    window: int = 32                             # snapkv observation window (steps)
    protect_prompt: bool = False                 # snapkv: evict prompt pages only when the prompt alone overflows

    @property
    def label(self) -> str:
        return self.name if self.budget_tokens is None else f"{self.name}@{self.budget_tokens}"


def _qk(qg: torch.Tensor, K: torch.Tensor, scaling: float) -> torch.Tensor:
    """qg [B, S, g, d], K [B, S, N, d] (bf16) -> logits [B, S, g, N], fp32-accumulated and fp32 out."""
    B, S, g, d = qg.shape
    N = K.shape[2]
    return torch.bmm(qg.reshape(B * S, g, d), K.reshape(B * S, N, d).transpose(1, 2),
                     out_dtype=torch.float32).view(B, S, g, N) * scaling


def _pv(p: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """p [B, S, g, N] fp32, V [B, S, N, d] -> [B, S, g, d] fp32 (probabilities rounded to V's dtype,
    as in flash attention)."""
    B, S, g, N = p.shape
    return torch.bmm(p.to(V.dtype).reshape(B * S, g, N), V.reshape(B * S, N, -1), out_dtype=torch.float32).view(B, S, g, -1)


def page_summaries(k: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """k [..., ps, S, d] keys of some pages, valid [..., ps] -> (min, max, mean) [..., S, d] in fp32
    over the valid tokens (selectors.py / baselines.unit_summaries)."""
    x = k.float()
    v = valid[..., None, None]
    cnt = valid.sum(-1).clamp(min=1)[..., None, None].float()
    return (x.masked_fill(~v, float("inf")).amin(-3), x.masked_fill(~v, _NEG).amax(-3),
            (x * v).sum(-3) / cnt)


class SparseEngine:
    """KV cache + per-step page selection for a batch of independent slots. Passed to the HF model
    as ``past_key_values``: ``update`` writes each slot's new key at its own length."""

    def __init__(self, model, method: Method, *, batch: int, max_len: int, page_size: int, sink_pages: int,
                 recent_tokens: int, obs_window: int, device, shadow: bool = False):
        if method.name not in METHODS:
            raise ValueError(method.name)
        c = model.config
        types = getattr(c, "layer_types", None) or ["full_attention"] * c.num_hidden_layers
        self.attn_layers = [i for i, t in enumerate(types) if t == "full_attention"]
        self.lin_layers = [i for i, t in enumerate(types) if t == "linear_attention"]
        if len(self.attn_layers) + len(self.lin_layers) != c.num_hidden_layers:
            raise NotImplementedError(f"layer types {sorted(set(types))}")
        self.aslot = {l: i for i, l in enumerate(self.attn_layers)}     # model layer -> KV slot
        self.L, self.S, self.H = len(self.attn_layers), c.num_key_value_heads, c.num_attention_heads
        self.d = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        self.g = self.H // self.S
        self.B, self.ps, self.sink, self.R, self.W = batch, page_size, sink_pages, recent_tokens, obs_window
        self.m = method
        self.P = math.ceil(max_len / page_size)          # page columns; column P is a dummy target
        self.Lmax = self.P * page_size
        self.dev = torch.device(device)
        dt = next(model.parameters()).dtype
        L, B, S, P, d = self.L, self.B, self.S, self.P, self.d
        self.k = torch.zeros((L, B, S, self.Lmax, d), dtype=dt, device=self.dev)
        self.v = torch.zeros_like(self.k)
        self.layers = [None] * c.num_hidden_layers        # linear-attention layers: per-slot state buffers
        if self.lin_layers:
            conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim + c.linear_num_value_heads * c.linear_value_head_dim
            sdt = getattr(torch, getattr(c, "mamba_ssm_dtype", None) or "float32")
            for l in self.lin_layers:
                self.layers[l] = _LinearState(
                    torch.zeros((B, conv_dim, c.linear_conv_kernel_dim), dtype=dt, device=self.dev),
                    torch.zeros((B, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim),
                                dtype=sdt, device=self.dev))
        self.lens = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.ar = torch.arange(B, device=self.dev)
        self.tok_ar = torch.arange(page_size, device=self.dev)
        self.page_ar = torch.arange(P + 1, device=self.dev)
        self.kmax_pages = 0 if method.budget_tokens is None else min(method.budget_tokens // page_size, self.P + 1)
        self.np_pinned = sink_pages + math.ceil(recent_tokens / page_size) + 1   # widest pinned set
        self.summ = method.name in ("quest", "ours")
        if self.summ:
            shape = (L, B, S, P + 1, d)
            self.kmin = torch.zeros(shape, dtype=dt, device=self.dev)       # min / max of bf16 keys: exact
            self.kmaxs = torch.zeros(shape, dtype=dt, device=self.dev)
            sp = method.spec
            self.bound_fmt, self.mean_fmt = sp.get("bound_fmt", "bf16"), sp.get("mean_fmt", "fp32")
            need_mean = bool(sp.get("mean_mix")) or self.bound_fmt == "none"
            self.kmean = torch.zeros(shape, device=self.dev) if need_mean else None
            if self.mean_fmt == "chan":           # fp16 mean on the d / 4 channels with the largest |q| |k| over the prompt
                self.chan = torch.zeros((L, B, S, d // 4), dtype=torch.long, device=self.dev)
        if method.name == "ours":
            sp = method.spec
            if not sp.get("fuse") or sp.get("recent_share") or sp.get("history", "last") != "last":
                raise NotImplementedError("ours: fused specs with last-observed history only")
            self.hist = torch.zeros((L, B, S, P + 1), device=self.dev)
            self.tiebreak = _RECENCY_TIEBREAK * torch.arange(P + 1, device=self.dev, dtype=torch.float32) / P
        if method.name == "snapkv":
            self.ring = torch.zeros((L, method.window, B, S, P + 1), device=self.dev)
            self.wsum = torch.zeros((L, B, S, P + 1), device=self.dev)
            self.kept = torch.zeros((L, B, P + 1), dtype=torch.bool, device=self.dev)
            self.evicted = torch.zeros_like(self.kept)
            self.kept_order = torch.arange(P + 1, device=self.dev, dtype=torch.float32) * 1e-7   # kept pages first
            self.prompt_pages = torch.zeros(B, dtype=torch.long, device=self.dev)    # protect_prompt
        self.ptr = torch.zeros((), dtype=torch.long, device=self.dev)    # decode steps run (snapkv window slot)
        self.mode: str | None = None
        self.slot = 0
        self.record: dict | None = None          # {layer: list of [B, P] bool} when set (checks)
        # shadow (checks): select and observe as the method would, but return dense attention, so the
        # hidden states (and queries) follow a dense trace and selections can be replayed offline
        self.shadow = shadow

    # ------------------------------------------------------------------ cache protocol
    def update(self, k_new, v_new, layer_idx, cache_kwargs=None):
        li = self.aslot[layer_idx]
        if self.mode == "prefill":
            b, n = self.slot, k_new.shape[2]
            self.k[li, b, :, :n] = k_new[0]
            self.v[li, b, :, :n] = v_new[0]
            if self.summ:
                np_ = math.ceil(n / self.ps)
                x = F.pad(k_new[0], (0, 0, 0, np_ * self.ps - n)).view(self.S, np_, self.ps, self.d).permute(1, 2, 0, 3)
                valid = (torch.arange(np_ * self.ps, device=self.dev) < n).view(np_, self.ps)
                self._write_summ(li, b, slice(0, np_), *page_summaries(x, valid))
            return k_new, v_new
        pos = self.lens
        self.k[li][self.ar, :, pos] = k_new[:, :, 0]
        self.v[li][self.ar, :, pos] = v_new[:, :, 0]
        if self.summ:                                   # the page holding the new key (final once full)
            cp = pos // self.ps
            tok = cp[:, None] * self.ps + self.tok_ar
            x = self.k[li][self.ar[:, None], :, tok]    # [B, ps, S, d]
            mn, mx, me = page_summaries(x, tok <= pos[:, None])
            self.kmin[li][self.ar, :, cp] = mn.to(self.kmin.dtype)
            self.kmaxs[li][self.ar, :, cp] = mx.to(self.kmaxs.dtype)
            if self.kmean is not None:
                self.kmean[li][self.ar, :, cp] = me
        return self.k[li], self.v[li]

    def _write_summ(self, li, b, pages, mn, mx, me):
        self.kmin[li, b, :, pages] = mn.transpose(0, 1)
        self.kmaxs[li, b, :, pages] = mx.transpose(0, 1)
        if self.kmean is not None:
            self.kmean[li, b, :, pages] = me.transpose(0, 1)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    # ------------------------------------------------------------------ linear-attention cache protocol
    def has_previous_state(self, layer_idx: int | None = None, state_idx: int | None = None) -> bool:
        return self.mode == "decode"

    def update_conv_state(self, conv_states, layer_idx, state_idx: int = 0, conv_kernel_size: int | None = None, **kw):
        """Prefill only (decode updates the buffer in place): keep the last kernel-size inputs of slot
        ``self.slot`` and return the (left zero-padded) inputs the causal conv needs."""
        if self.mode != "prefill":
            raise RuntimeError("conv state update outside prefill")
        buf = self.layers[layer_idx].conv_states[0]
        ks = buf.shape[-1]
        full = conv_states if conv_states.shape[-1] >= ks else F.pad(conv_states, (ks - conv_states.shape[-1], 0))
        buf[self.slot].copy_(full[0, :, -ks:])
        return full

    def update_recurrent_state(self, recurrent_states, layer_idx, state_idx: int = 0, **kw):
        buf = self.layers[layer_idx].recurrent_states[0]
        if self.mode == "prefill":
            buf[self.slot].copy_(recurrent_states[0])
        else:
            buf.copy_(recurrent_states)
        return buf

    # ------------------------------------------------------------------ slots and steps
    def reset_slot(self, b: int) -> None:
        self.lens[b] = 0
        for st in self.layers:
            if st is not None:
                st.reset_slot(b)
        if self.m.name == "ours":
            self.hist[:, b] = 0
        if self.m.name == "snapkv":
            self.ring[:, :, b] = 0
            self.wsum[:, b] = 0
            self.kept[:, b] = False
            self.evicted[:, b] = False
            self.prompt_pages[b] = 0

    def begin_prefill(self, b: int) -> None:
        self.reset_slot(b)
        self.mode, self.slot = "prefill", b

    def end_prefill(self, n: int) -> None:
        self.lens[self.slot] = n
        if self.m.name == "snapkv":
            self.prompt_pages[self.slot] = -(-n // self.ps)
        self.mode = None

    def begin_step(self, lc: int) -> None:
        """Per-step page plan shared by all layers, from GPU state only (CUDA-graph safe). ``lc`` (a
        multiple of the page size) bounds the keys dense attention reads: at least the longest slot
        after this step's key is written."""
        n = self.lens + 1                                       # keys visible incl. the new one
        ps, sink = self.ps, self.sink
        hi = ((n - self.R) // ps).clamp(min=sink)               # eligible pages: [sink, hi)
        npages = (n + ps - 1) // ps
        e = hi - sink
        n_pinned = sink + npages - hi
        rec = hi[:, None] + torch.arange(self.np_pinned - sink, device=self.dev)
        pinned = torch.cat([torch.arange(sink, device=self.dev).expand(self.B, sink),
                            torch.where(rec < npages[:, None], rec, self.P)], 1)
        b = (self.kmax_pages - n_pinned).clamp(min=0)
        self.plan = {"n": n, "hi": hi, "pinned": pinned, "k_sel": torch.minimum(b, e), "b": b,
                     "elig": (self.page_ar >= sink) & (self.page_ar < hi[:, None]),
                     "lc": lc}
        self.mode = "decode"

    def end_step(self, active: torch.Tensor) -> None:
        self.lens += active.long()
        self.ptr += 1
        self.mode = None

    # ------------------------------------------------------------------ attention
    def attend(self, layer_idx, q, K, V, scaling):
        li = self.aslot[layer_idx]
        if self.mode == "prefill":
            return self._prefill(li, q, K, V, scaling)
        if self.m.name == "full" or self.shadow:
            dense = self._dense(q, K, V, scaling)
            if self.m.name == "full":
                return dense, None
        B, S, g, d, ps = self.B, self.S, self.g, self.d, self.ps
        qg = q[:, :, 0].view(B, S, g, d)
        sel = self._select(li, qg, K, scaling)                                     # [B, Kmax], P = none
        pages = torch.cat([sel, self.plan["pinned"]], 1)                           # [B, Kp]
        tok = pages[:, :, None] * ps + self.tok_ar
        valid = (pages[:, :, None] < self.P) & (tok < self.plan["n"][:, None, None])
        tok = tok.clamp(max=self.Lmax - 1).view(B, -1)
        kk = K[self.ar[:, None], :, tok].transpose(1, 2)                           # [B, S, N, d]
        vv = V[self.ar[:, None], :, tok].transpose(1, 2)
        p = torch.softmax(_qk(qg, kk, scaling).masked_fill(~valid.view(B, 1, 1, -1), _NEG), -1)
        out = _pv(p, vv)
        if self.m.name in ("ours", "snapkv"):
            self._observe(li, pages, p.mean(2).view(B, S, -1, ps).sum(-1))
        if self.record is not None and layer_idx in self.record:
            m = torch.zeros((B, self.P + 1), dtype=torch.bool, device=self.dev)
            m[self.ar[:, None], sel] = True
            self.record[layer_idx].append(m[:, :self.P].cpu())
        if self.shadow:
            return dense, None
        return out.reshape(B, 1, self.H, d).to(q.dtype), None

    def _dense(self, q, K, V, scaling):
        lc, B = self.plan["lc"], self.B
        s = _qk(q[:, :, 0].view(B, self.S, self.g, self.d), K[:, :, :lc], scaling)
        s.masked_fill_(torch.arange(lc, device=self.dev) >= self.plan["n"][:, None, None, None], _NEG)
        return _pv(torch.softmax(s, -1), V[:, :, :lc]).reshape(B, 1, self.H, self.d).to(q.dtype)

    def _prefill(self, li, q, K, V, scaling):
        out = F.scaled_dot_product_attention(q, K, V, is_causal=True, scale=scaling, enable_gqa=True)
        if self.summ and self.mean_fmt == "chan":
            n = K.shape[2]
            imp = q[0, :, n - 1].float().abs().view(self.S, self.g, self.d).mean(1) * K[0, :, :n].float().abs().mean(1)
            self.chan[li, self.slot] = torch.topk(imp, self.d // 4, dim=-1).indices
        if self.m.name in ("ours", "snapkv"):
            # attention of the last obs_window prompt queries, as the tracer's prefill window
            n, b = K.shape[2], self.slot
            w = min(self.W, n)
            qw = q[0, :, n - w:].float().view(self.S, self.g, w, self.d)
            s = torch.einsum("sgwd,snd->sgwn", qw, K[0].float()) * scaling
            future = torch.arange(n, device=self.dev)[None, :] > torch.arange(n - w, n, device=self.dev)[:, None]
            pm = torch.softmax(s.masked_fill(future, _NEG), -1).mean(1)                # [S, w, n]
            np_ = math.ceil(n / self.ps)
            pm = F.pad(pm, (0, np_ * self.ps - n)).view(self.S, w, np_, self.ps).sum(-1)
            if self.m.name == "ours":
                self.hist[li, b, :, :np_] = pm.mean(1)
            else:                                    # step 0 (last prompt query) opens the window
                row = torch.zeros((self.S, self.P + 1), device=self.dev)
                row[:, :np_] = pm[:, -1]
                self.ring[li, :, b].index_copy_(0, ((self.ptr - 1) % self.m.window).view(1), row[None])
                self.wsum[li, b] = row
        return out.transpose(1, 2), None

    def _select(self, li, qg, K, scaling) -> torch.Tensor:
        """[B, Kmax] page ids for this layer (unused entries = P)."""
        pl, P = self.plan, self.P
        elig, k_sel, kmax = pl["elig"], pl["k_sel"], self.kmax_pages
        name = self.m.name
        if name == "snapkv":
            return self._snapkv(li, elig, pl["b"])
        if name == "oracle":
            lc = pl["lc"]
            s = _qk(qg, K[:, :, :lc], scaling)                                                   # [B, S, g, lc]
            s.masked_fill_(torch.arange(lc, device=self.dev) >= pl["n"][:, None, None, None], _NEG)
            m = torch.softmax(s, -1).mean(2).view(self.B, self.S, lc // self.ps, self.ps).sum(-1)
            key = F.pad(m.sum(1), (0, P + 1 - m.shape[-1])).masked_fill(~elig, _NEG)
            return self._take(key.topk(kmax, -1).indices, k_sel)
        bound = self._bound(li, qg, scaling)                                              # [B, S, P+1]
        if name == "quest":
            sc = bound.masked_fill(~elig[:, None, :], _NEG)
            top = sc.topk(kmax, -1).indices                                                 # per-head nominations
            nomv = (torch.arange(kmax, device=self.dev) < k_sel[:, None, None]).float()
            nom = torch.zeros((self.B, P + 1), device=self.dev).scatter_add_(1, top.reshape(self.B, -1),
                                                                           nomv.expand_as(top).reshape(self.B, -1))
            best = sc.amax(1)
            order = torch.argsort(-best, dim=-1, stable=True)
            rb = torch.empty_like(order).scatter_(-1, order, self.page_ar.expand_as(order))
            key = (nom * (P + 2) - rb.float()).masked_fill(~elig, _NEG)
            return self._take(key.topk(kmax, -1).indices, k_sel)
        sp = self.m.spec                                                                    # ours
        qb = bound
        mm = sp.get("mean_mix") or 0.0
        if self.bound_fmt == "none":
            qb = self._mean_logit(li, qg, scaling)
        elif mm:
            qb = (1.0 - mm) * qb + mm * self._mean_logit(li, qg, scaling)
        qs = torch.softmax(qb.masked_fill(~elig[:, None, :], _NEG), -1).nan_to_num(0.0)
        hist, w = self.hist[li], sp.get("history_weight", 0.5)
        f = torch.maximum(qs, hist) if sp["fuse"] == "max" else (1.0 - w) * qs + w * hist
        key = (f + self.tiebreak).sum(1).masked_fill(~elig, _NEG)
        return self._take(key.topk(kmax, -1).indices, k_sel)

    def _bound(self, li, qg, scaling):
        """Quest bound per KV head [B, S, P+1] from min / max summaries stored as bf16 (exact for bf16
        keys), int4 with one fp16 scale per (head, page), or int4 over 2-page units (H2 formats;
        rounded outward so they stay bounds)."""
        if self.bound_fmt == "none":
            return None
        kmin, kmax = self.kmin[li], self.kmaxs[li]
        if self.bound_fmt == "int4p2":
            P1 = kmin.shape[2]
            pad = (0, 0, 0, P1 % 2)
            kmin = F.pad(kmin.float(), pad, value=float("inf")).view(self.B, self.S, -1, 2, self.d).amin(3)
            kmax = F.pad(kmax.float(), pad, value=float("-inf")).view(self.B, self.S, -1, 2, self.d).amax(3)
        if self.bound_fmt in ("int4", "int4p2"):
            sc = torch.maximum(kmin.float().abs(), kmax.float().abs()).amax(-1, keepdim=True).clamp(min=1e-8) / 7
            sc = sc.half().float()
            kmin = (torch.floor(kmin.float() / sc).clamp(-8, 7) * sc).to(qg.dtype)
            kmax = (torch.ceil(kmax.float() / sc).clamp(-8, 7) * sc).to(qg.dtype)
        elif self.bound_fmt != "bf16":
            raise ValueError(self.bound_fmt)
        b = (_qk(qg.clamp(min=0), kmax, scaling) + _qk(qg.clamp(max=0), kmin, scaling)).amax(2)
        if self.bound_fmt == "int4p2":
            b = b.repeat_interleave(2, -1)[..., :self.P + 1] + _UNIT_TIEBREAK * (self.page_ar % 2)
        return b

    def _mean_logit(self, li, qg, scaling):
        """Mean-key logit per KV head [B, S, P+1], max over the group: fp32, fp16 or fp8 (e4m3) means,
        or fp16 means on the slot's d / 4 prompt-chosen channels."""
        km = self.kmean[li]
        if self.mean_fmt == "fp16":
            km = km.half().float()
        elif self.mean_fmt == "fp8":
            km = km.to(torch.float8_e4m3fn).float()
        elif self.mean_fmt == "chan":
            ch = self.chan[li]                                                       # [B, S, c]
            qs = torch.gather(qg.float(), 3, ch[:, :, None, :].expand(-1, -1, self.g, -1))
            ms = torch.gather(km.half().float(), 3, ch[:, :, None, :].expand(-1, -1, km.shape[2], -1))
            return torch.einsum("bsgc,bspc->bsgp", qs, ms).amax(2) * scaling
        elif self.mean_fmt != "fp32":
            raise ValueError(self.mean_fmt)
        return torch.einsum("bsgd,bspd->bsgp", qg.float(), km).amax(2) * scaling

    def _take(self, idx: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.arange(idx.shape[1], device=self.dev) < k[:, None], idx, self.P)

    def _snapkv(self, li, elig, b) -> torch.Tensor:
        """Permanent eviction (sim.simulate_evict, one set per layer): pages join when they age into
        eligibility; whenever the kept pages exceed this step's budget the lowest-scoring ones are
        dropped. (Until 2026-10-26, H0 audit: trimmed only when a page joined, so on the steps where a
        new recent page opened the set ran one page over budget: ~3% more tokens at 512 in E3.)"""
        kept, evicted = self.kept[li], self.evicted[li]
        kept |= elig & ~evicted
        excess = (kept.sum(1) - b).clamp(min=0)
        prio = self.wsum[li].sum(1)
        if self.m.protect_prompt:                       # generated pages go first: window sums <= window * S
            prio = prio + (self.m.window * self.S + 1.0) * (self.page_ar < self.prompt_pages[:, None])
        prio = prio.masked_fill(~kept, float("inf"))
        order = torch.argsort(prio, dim=-1, stable=True)
        rank = torch.empty_like(order).scatter_(-1, order, self.page_ar.expand_as(order))
        drop = kept & (rank < excess[:, None])
        kept &= ~drop
        evicted |= drop
        idx = (kept.float() + self.kept_order).topk(self.kmax_pages, -1).indices
        return self._take(idx, kept.sum(1))

    def _observe(self, li, pages, obs):
        """obs [B, S, Kp]: renormalised attention on the attended pages (unused entries = P)."""
        idx = pages[:, None, :].expand(-1, self.S, -1)
        if self.m.name == "ours":
            self.hist[li].scatter_(2, idx, obs)
            self.hist[li, :, :, self.P] = 0
        else:
            dense = torch.zeros((self.B, self.S, self.P + 1), device=self.dev).scatter_(2, idx, obs)
            dense[:, :, self.P] = 0
            slot = (self.ptr % self.m.window).view(1)
            self.wsum[li] += dense - self.ring[li].index_select(0, slot)[0]
            self.ring[li].index_copy_(0, slot, dense[None])


# ---------------------------------------------------------------------- sampling and the batch loop
def sample_token(logits: torch.Tensor, u: torch.Tensor, temperature: float, top_k: int, top_p: float,
                 presence_penalty: float = 0.0, seen: torch.Tensor | None = None) -> torch.Tensor:
    """HF's temperature -> top-k -> top-p sampling, driven by one uniform per row (inverse CDF), so
    every method sees the same random numbers for a given (problem, sample). u [B] in [0, 1).
    ``presence_penalty`` (vLLM's semantics) is subtracted from the logits of tokens already generated
    (``seen`` [B, V] bool), before temperature."""
    if presence_penalty:
        logits = logits.float() - presence_penalty * seen[:, :logits.shape[-1]].float()
    if temperature <= 0:
        return logits.argmax(-1)
    vals, idx = (logits.float() / temperature).topk(top_k, -1)                  # sorted descending
    p = torch.softmax(vals, -1)
    keep = (p.cumsum(-1) - p) < top_p                                           # the first is always kept
    p = p * keep
    c = (p / p.sum(-1, keepdim=True)).cumsum(-1)
    j = (c <= u[:, None]).sum(-1).clamp(max=top_k - 1)
    return idx.gather(-1, j[:, None])[:, 0]


@dataclass
class Job:
    key: tuple
    prompt_ids: list[int]
    seed: list[int]                               # np.random.default_rng seed (same for every method)
    forced: list[int] | None = None               # teacher forcing (checks): generated tokens to feed


@dataclass
class _Slot:
    job: Job
    rng: np.random.Generator
    tokens: list[int]
    n_prompt: int


@torch.inference_mode()
def run_batch(model, engine: SparseEngine, jobs: list[Job], *, stop_ids: list[int], max_new_tokens: int,
              sampling: dict, on_done, log_every: int = 0, logits_out: dict | None = None,
              graphs: bool = True, lc_bucket: int = 1024, lc_unit: int | None = None) -> None:
    """Continuous batching over ``jobs``; calls ``on_done(job, tokens, stopped)`` per finished job
    (tokens = generated ids, the stop token excluded). ``logits_out`` (checks) collects each job's
    next-token logits per step, keyed by job key. With ``graphs`` each decode step (plan, forward,
    sampling) replays a CUDA graph, one per ``lc_bucket`` of dense length for methods that read all
    keys: Python launch overhead otherwise dominates (thousands of small kernels per step). ``lc_unit``
    (checks) overrides the dense-length rounding without graphs (default: one page), e.g. to the graph
    bucket so that graphs on and off run the same kernels."""
    global _ENGINE
    _ENGINE = engine
    try:
        _run(model, engine, jobs, stop_ids=stop_ids, max_new_tokens=max_new_tokens, sampling=sampling, on_done=on_done,
             log_every=log_every, logits_out=logits_out, lc_bucket=lc_bucket, lc_unit=lc_unit,
             graphs=graphs and logits_out is None and engine.record is None)    # checks read tensors per step
    finally:
        _ENGINE = None                            # the engine holds the KV cache: let the caller free it


def _run(model, engine, jobs, *, stop_ids, max_new_tokens, sampling, on_done, log_every, logits_out, graphs,
         lc_bucket, lc_unit=None) -> None:
    dev, B = engine.dev, engine.B
    queue = list(jobs)
    slots: list[_Slot | None] = [None] * B
    lens = np.zeros(B, dtype=np.int64)
    stops = set(stop_ids)
    mask_dict = {"full_attention": None, "linear_attention": None}
    step = 0
    ids_t = torch.zeros((B, 1), dtype=torch.long, device=dev)
    u_t = torch.zeros(B, device=dev)
    act_t = torch.zeros(B, dtype=torch.bool, device=dev)
    pp = sampling.get("presence_penalty", 0.0)
    seen_t = torch.zeros((B, model.config.vocab_size), dtype=torch.bool, device=dev) if pp else None
    dense = engine.m.name in ("full", "oracle") or engine.shadow

    def step_fn(lc: int):
        if seen_t is not None:                  # the token fed this step was generated: penalise it from now on
            seen_t.scatter_(1, ids_t, True)
        pos = engine.lens.clone()
        engine.begin_step(lc)
        out = model(input_ids=ids_t, position_ids=pos[:, None], cache_position=pos, past_key_values=engine,
                    attention_mask=mask_dict, use_cache=True)
        engine.end_step(act_t)
        logits = out.logits[:, -1]
        return logits, sample_token(logits, u_t, **sampling, seen=seen_t)

    def bucket(n: int) -> int:
        if not dense:
            return engine.Lmax
        unit = lc_bucket if graphs else (lc_unit or engine.ps)
        return min(engine.Lmax, -(-n // unit) * unit)

    cuda_graphs = {}
    if graphs:          # capture before any slot is filled: warm-up steps write only to empty slots
        pool = torch.cuda.graph_pool_handle()
        for lc in sorted({bucket(n) for n in range(1, engine.Lmax + 1, lc_bucket if dense else engine.Lmax)} | {engine.Lmax}):
            side = torch.cuda.Stream(device=dev)
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                for _ in range(2):
                    step_fn(lc)
            torch.cuda.current_stream(dev).wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                outs = step_fn(lc)
            cuda_graphs[lc] = (g, outs)
        for b in range(B):
            engine.reset_slot(b)

    def first_token(b: int, job: Job) -> None:
        ids = torch.tensor([job.prompt_ids], device=dev)
        n = ids.shape[1]
        engine.begin_prefill(b)
        if seen_t is not None:
            seen_t[b] = False
        out = model(input_ids=ids, position_ids=torch.arange(n, device=dev)[None], cache_position=torch.arange(n, device=dev),
                    past_key_values=engine, attention_mask=mask_dict, use_cache=True, logits_to_keep=1)
        engine.end_prefill(n)
        lens[b] = n
        sl = _Slot(job, np.random.default_rng(job.seed), [], n)
        slots[b] = sl
        if logits_out is not None:
            logits_out.setdefault(job.key, []).append(out.logits[0, -1].float().cpu())
        if job.forced is not None:
            sl.tokens.append(job.forced[0])
        else:
            u = torch.tensor([sl.rng.random()], device=dev)
            sl.tokens.append(int(sample_token(out.logits[:, -1], u, **sampling, seen=None if seen_t is None else seen_t[b:b + 1])[0]))

    def finished(sl: _Slot) -> bool:
        t = sl.tokens
        cap = max_new_tokens if sl.job.forced is None else min(max_new_tokens, len(sl.job.forced))
        return (t and t[-1] in stops) or len(t) >= cap

    def retire(b: int) -> None:
        sl = slots[b]
        stopped = bool(sl.tokens) and sl.tokens[-1] in stops
        on_done(sl.job, sl.tokens[:-1] if stopped else sl.tokens, stopped)
        slots[b] = None
        engine.reset_slot(b)
        lens[b] = 0

    while True:
        for b in range(B):
            while slots[b] is not None and finished(slots[b]):
                retire(b)
            while slots[b] is None and queue:
                first_token(b, queue.pop(0))
                if finished(slots[b]):
                    retire(b)
        active_np = np.array([s is not None for s in slots])
        if not active_np.any():
            break
        ids_t.copy_(torch.tensor([[s.tokens[-1] if s is not None else 0] for s in slots]))
        act_t.copy_(torch.from_numpy(active_np))
        u_t.copy_(torch.tensor([s.rng.random() if s is not None and s.job.forced is None else 0.0 for s in slots]))
        lc = bucket(int(lens.max()) + 1)
        if graphs:
            g, (logits, toks_t) = cuda_graphs[lc]
            g.replay()
        else:
            logits, toks_t = step_fn(lc)
        lens += active_np
        if logits_out is not None:
            for b, sl in enumerate(slots):
                if sl is not None:
                    logits_out.setdefault(sl.job.key, []).append(logits[b].float().cpu())
        toks = toks_t.tolist()
        for b, sl in enumerate(slots):
            if sl is not None:
                sl.tokens.append(sl.job.forced[len(sl.tokens)] if sl.job.forced is not None else toks[b])
        step += 1
        if log_every and step % log_every == 0:
            n_act = int(active_np.sum())
            print(f"[e3] step {step} active {n_act} queue {len(queue)} max_len {int(lens.max())}", flush=True)
