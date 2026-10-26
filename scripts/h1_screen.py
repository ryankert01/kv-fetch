"""H1 (EXPERIMENTS.md, "Hybrid models"): offline screen of page selection on a hybrid model and on a
transformer, same problems, same pipeline. Run with .venv-hybrid.

  CUDA_VISIBLE_DEVICES=3 .venv-hybrid/bin/python scripts/h1_screen.py configs/hybrid.yaml screen --model qwen35_4b [--shard 0/2]
  CUDA_VISIBLE_DEVICES=3 .venv-hybrid/bin/python scripts/h1_screen.py configs/hybrid.yaml fidelity --model qwen3_4b
  .venv-hybrid/bin/python scripts/h1_screen.py configs/hybrid.yaml merge

Sequences: sample 0 of each explore problem from a full-attention sampling run (hybrid.h1.sequences),
first h1.max_gen_tokens generated tokens. One forward over prompt + generated tokens gives every
decode query and key of the traced attention layers (decode query s, at position prompt_len - 1 + s,
sees keys [0, prompt_len + s), as the tracer's steps); exact attention per page is recomputed from them
(scripts/compare_pages.py, E2b). Every method picks one page set per layer for all KV heads (oracle
also per KV head) on top of the pinned pages (sink + recent window), within the attended budget.

screen, per layer and budget, missing attention of:
  oracle|sum, oracle|per_head    true top pages (the ceilings),
  quest|vote, quest|sum, mean|sum  Quest bounds and mean-key logits (current query),
  ours|sum                       Quest bound + h1.ours.mean_mix * mean-key logit, summed shares (E3's gain),
  quest@la{k}|vote, ours@la{k}|sum  the same with the query predicted from the hidden state entering
                                 layer L - k (H-c: the attention layer's input norm, q projection, q norm
                                 and RoPE applied to it),
  snapkv|evict                   prompt-protected SnapKV-style eviction, window sums of the attention it
                                 observed (renormalised over attended pages), as kvfetch/sparse_decode.py,
  ours+{prior}{w}|sum            (hybrid only, H-d) ours mixed with a GDN page prior from the three GDN
                                 layers before the attention layer: "beta" = mean write strength of the
                                 page's tokens, "forgot" = sum of beta * (1 - retention since written),
plus page churn (pages new to the set per step) and each prior's Spearman correlation with the true
page mass. fidelity (H-b offline): next-token KL(dense || sparse) of teacher-forced forwards where
every attention layer attends only its selected + pinned pages, vs the missing attention it had.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

from kvfetch import selectors_gpu as G
from kvfetch.baselines import attention_by_group, landmark_scores, quest_scores, unit_summaries
from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.models import encode, format_prompt, load_model
from kvfetch.provenance import provenance, write_json
from kvfetch.sim import existing_pages
from compare_pages import Scorer, fit_pages
from run_e3 import problems, thinking

CAPTURE = "h1_capture"
SPARSE = "h1_sparse"
_STATE: dict = {}


# ---------------------------------------------------------------------------- sequences and capture
def sequences(cfg: dict, mkey: str, tok) -> list[dict]:
    h1 = cfg["hybrid"]["h1"]
    root = Path(h1["sequences"][mkey])
    out = []
    for s in problems({**cfg, "e3": cfg["hybrid"]["h3"]}, tok):
        f = root / s.task / f"{s.index:04d}_s{h1['sample']}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        prompt = encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"]
        if len(prompt) != r["prompt_tokens"]:
            raise RuntimeError(f"{f}: prompt has {len(prompt)} tokens, the run had {r['prompt_tokens']}")
        gen = r["token_ids"] if "token_ids" in r else encode(tok, r["text"])["input_ids"]   # E3 stored text only
        out.append({"task": s.task, "index": s.index, "prompt": prompt, "gen": gen[:h1["max_gen_tokens"]],
                    "retokenized": "token_ids" not in r, "score": r["score"], "stopped": r["stopped"]})
    return out


def _capture_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    st = _STATE
    if module.layer_idx in st.get("layers", ()):
        st["q"][module.layer_idx] = query[0, :, st["q0"]:].clone()          # [H, T, d] (bf16)
        st["k"][module.layer_idx] = key[0].clone()                          # [S, N, d]
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kw)


def register() -> None:
    for name, fn in ((CAPTURE, _capture_attention), (SPARSE, _sparse_attention)):
        AttentionInterface.register(name, fn)
        AttentionMaskInterface.register(name, ALL_MASK_ATTENTION_FUNCTIONS._global_mapping["sdpa"])


def attn_qk(attn, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """The attention layer's own q / k path (projection, per-head norm, RoPE) applied to x [1, N, D]
    (already through the layer's input norm). Gated attention (Qwen3.5) splits off the gate."""
    shape, hd = x.shape[:-1], attn.head_dim
    qp = attn.q_proj(x)
    if qp.shape[-1] == 2 * attn.config.num_attention_heads * hd:
        q, _ = torch.chunk(qp.view(*shape, -1, hd * 2), 2, dim=-1)
    else:
        q = qp.view(*shape, -1, hd)
    q = attn.q_norm(q.reshape(*shape, -1, hd)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(x).view(*shape, -1, hd)).transpose(1, 2)
    return sys.modules[type(attn).__module__].apply_rotary_pos_emb(q, k, cos, sin)


@torch.no_grad()
def capture(model, ids: list[int], pl: int, layers: list[int], lookahead: list[int], gdn: bool) -> dict:
    """One forward over ids; returns per traced layer: q [T, H, d] (decode queries), k [S, N, d], the
    lookahead queries {k: [T, H, d]} and (gdn) the write strength / log-decay of the GDN layers
    before it: beta, g [3, N, Hv]."""
    dec = model.model.layers
    need = sorted({l - k for l in layers for k in [0] + lookahead if l - k >= 0})
    h_in, pos = {}, {}
    hooks = [dec[l].register_forward_pre_hook(lambda m, a, kw, l=l: h_in.__setitem__(l, kw["hidden_states"] if "hidden_states" in kw else a[0]), with_kwargs=True)
             for l in need]
    def keep_pos(m, a, kw):                                       # a pre-hook must return None (no change)
        pos.setdefault("cos_sin", kw["position_embeddings"])
    hooks.append(dec[layers[0]].self_attn.register_forward_pre_hook(keep_pos, with_kwargs=True))
    _STATE.update(layers=set(layers), q={}, k={}, q0=pl - 1)
    x = torch.tensor([ids], device=next(model.parameters()).device)
    try:
        model.model(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    cos, sin = pos["cos_sin"]
    out = {}
    for l in layers:
        attn = dec[l].self_attn
        o = {"q": _STATE["q"][l].transpose(0, 1).contiguous(), "k": _STATE["k"][l].float(), "la": {}}
        q_self, _ = attn_qk(attn, dec[l].input_layernorm(h_in[l]), cos, sin)
        o["q_self_err"] = float((q_self[0, :, pl - 1:].transpose(0, 1).float() - o["q"].float()).abs().max())
        for k in lookahead:
            if l - k >= 0:
                qk, _ = attn_qk(attn, dec[l].input_layernorm(h_in[l - k]), cos, sin)
                o["la"][k] = qk[0, :, pl - 1:].transpose(0, 1).contiguous()
        if gdn:
            bs, gs = [], []
            for j in (l - 3, l - 2, l - 1):
                la = getattr(dec[j], "linear_attn", None)
                if la is None:
                    continue
                xj = dec[j].input_layernorm(h_in[j]) if j in h_in else None
                if xj is None:
                    continue
                bs.append(la.in_proj_b(xj)[0].float().sigmoid())
                gs.append(-la.A_log.float().exp() * F.softplus(la.in_proj_a(xj)[0].float() + la.dt_bias))
            if bs:
                o["beta"], o["g"] = torch.stack(bs), torch.stack(gs)
        out[l] = o
    _STATE.clear()
    del h_in
    return out


# ---------------------------------------------------------------------------- selection and scoring
def budgets_for(btoks, ps, exist_np, elig_np, device):
    return {str(b): torch.from_numpy(np.maximum(0, b // ps - (exist_np & ~elig_np).sum(-1))).to(device) for b in btoks}


def page_scores(q: torch.Tensor, k: torch.Tensor, pl: int, P: int, ps: int, chunk: int, summ: dict) -> dict:
    """Exact page mass m, Quest bounds and mean-key logits for queries q [T, H, d]: each [T, S, P]."""
    T, H, d = q.shape
    scaling = d ** -0.5
    n_keys = torch.arange(T, device=q.device) + pl
    out = {n: torch.empty((T, k.shape[0], P), device=q.device) for n in ("m", "quest", "mean")}
    pad = P * ps - k.shape[1]
    for a in range(0, T, chunk):
        sl = slice(a, min(T, a + chunk))
        qc = q[sl].float()
        att = attention_by_group(qc, k, n_keys[sl], scaling).mean(2)
        out["m"][sl] = F.pad(att, (0, pad)).reshape(*att.shape[:-1], P, ps).sum(-1)
        out["quest"][sl] = fit_pages(quest_scores(qc, summ, scaling), P)
        out["mean"][sl] = fit_pages(landmark_scores(qc, summ, scaling), P)
    return out


def bound_scores(q: torch.Tensor, P: int, chunk: int, summ: dict) -> tuple[torch.Tensor, torch.Tensor]:
    T, H, d = q.shape
    scaling = d ** -0.5
    qu, me = torch.empty((T, summ["max"].shape[0], P), device=q.device), torch.empty((T, summ["max"].shape[0], P), device=q.device)
    for a in range(0, T, chunk):
        sl = slice(a, min(T, a + chunk))
        qc = q[sl].float()
        qu[sl] = fit_pages(quest_scores(qc, summ, scaling), P)
        me[sl] = fit_pages(landmark_scores(qc, summ, scaling), P)
    return qu, me


def q4_bounds(summ: dict) -> dict:
    """Quest bounds stored as int4 with one fp16 scale per (KV head, unit); min rounded down and max
    up, so they stay bounds."""
    s = torch.maximum(summ["min"].abs(), summ["max"].abs()).amax(-1, keepdim=True).clamp(min=1e-8) / 7
    s = s.half().float()
    return {"min": torch.floor(summ["min"] / s).clamp(-8, 7) * s, "max": torch.ceil(summ["max"] / s).clamp(-8, 7) * s}


def partial_mean_scores(q: torch.Tensor, mean: torch.Tensor, ch: torch.Tensor, scaling: float) -> torch.Tensor:
    """Mean-key logits from c channels per KV head (ch [S, c]; fp16 means), max over the group. [T, S, U]."""
    T, H, d = q.shape
    S = mean.shape[0]
    qg = q.reshape(T, S, H // S, d)
    qs = torch.gather(qg, 3, ch[None, :, None, :].expand(T, S, H // S, -1))
    ms = torch.gather(mean.half().float(), 2, ch[:, None, :].expand(S, mean.shape[1], -1))
    return (torch.einsum("tsgc,suc->tsgu", qs, ms) * scaling).amax(2)


def h2_scores(q: torch.Tensor, k: torch.Tensor, pl: int, P: int, ps: int, chunk: int, summ: dict) -> dict:
    """H2 summary formats (GPU metadata per token per KV head, d = head dim): mean16 (d * 2 / ps),
    mean8 (fp8 e4m3 mean, d / ps), quest4 (int4 bounds, d / ps + scales), meanc (fp16 mean on the d / 4
    channels with the largest |q| * |k| over the prompt, d / 2 / ps), quest4p2 (int4 bounds over
    2-page units, scored per unit), lite = quest4p2 + meanc mixed like ours, ours48 = quest4 + mean8."""
    T, H, d = q.shape
    S = k.shape[0]
    scaling = d ** -0.5
    q4 = q4_bounds(summ)
    m8 = {"mean": summ["mean"].to(torch.float8_e4m3fn).float()}
    m16 = {"mean": summ["mean"].half().float()}
    s2 = q4_bounds(unit_summaries(k, 2 * ps))
    imp = q[0].float().abs().reshape(S, H // S, d).mean(1) * k[:, :pl].abs().mean(1)          # [S, d]
    ch = torch.topk(imp, d // 4, dim=-1).indices
    out = {n: torch.empty((T, S, P), device=q.device) for n in ("mean16", "mean8", "quest4", "meanc", "quest4p2")}
    for a in range(0, T, chunk):
        sl = slice(a, min(T, a + chunk))
        qc = q[sl].float()
        out["mean16"][sl] = fit_pages(landmark_scores(qc, m16, scaling), P)
        out["mean8"][sl] = fit_pages(landmark_scores(qc, m8, scaling), P)
        out["quest4"][sl] = fit_pages(quest_scores(qc, q4, scaling), P)
        out["meanc"][sl] = fit_pages(partial_mean_scores(qc, summ["mean"], ch, scaling), P)
        out["quest4p2"][sl] = fit_pages(quest_scores(qc, s2, scaling).repeat_interleave(2, -1), P)
    return out


H2_BYTES = {   # GPU metadata bytes per token per KV head, as a function of head dim d and page size ps
    "quest": lambda d, ps: 2 * d * 2 / ps, "ours": lambda d, ps: 3 * d * 2 / ps, "mean": lambda d, ps: d * 2 / ps,
    "mean16": lambda d, ps: d * 2 / ps, "mean8": lambda d, ps: d / ps, "quest4": lambda d, ps: (d + 2) / ps,
    "meanc": lambda d, ps: d / 2 / ps, "quest4p2": lambda d, ps: (d + 2) / (2 * ps),
    "lite": lambda d, ps: d / 2 / ps + (d + 2) / (2 * ps), "ours48": lambda d, ps: 2 * d / ps + 2 / ps,
}


def churn(sel: torch.Tensor) -> float:
    """Mean pages per step that were not in the previous step's set (shared sets: head 0)."""
    s = sel[:, 0]
    return float((s[1:] & ~s[:-1]).sum(-1).float().mean()) if s.shape[0] > 1 else 0.0


def snapkv_evict(m: torch.Tensor, elig: torch.Tensor, exist: torch.Tensor, b: torch.Tensor, window: int,
                 prompt_pages: int, protect: bool) -> torch.Tensor:
    """sparse_decode.SparseEngine._snapkv offline, one set per layer: pages join when they age into
    eligibility; whenever the kept pages exceed the step's budget the lowest window-sum ones are
    dropped for good.
    The window holds the attention the method observed (renormalised over attended pages); step 0
    (the last prompt query, dense in prefill) opens it. Returns the kept sets [T, P]."""
    T, S, P = m.shape
    dev = m.device
    ring = torch.zeros((window, S, P), device=dev)
    ring[0] = m[0]
    wsum = m[0].clone()
    kept = torch.zeros(P, dtype=torch.bool, device=dev)
    evicted = torch.zeros_like(kept)
    out = torch.zeros((T, P), dtype=torch.bool, device=dev)
    is_prompt = (torch.arange(P, device=dev) < prompt_pages).float() * ((window * S + 1.0) if protect else 0.0)
    page_ar = torch.arange(P, device=dev)
    tiny = torch.finfo(torch.float32).tiny
    for t in range(T):                                   # no host syncs: the loop stays on the GPU queue
        kept |= elig[t] & ~evicted
        excess = (kept.sum() - b[t]).clamp(min=0)
        prio = (wsum.sum(0) + is_prompt).masked_fill(~kept, float("inf"))
        order = torch.argsort(prio, stable=True)
        rank = torch.empty_like(order).scatter_(0, order, page_ar)
        drop = kept & (rank < excess)
        kept &= ~drop
        evicted |= drop
        out[t] = kept
        if t > 0:
            att = kept | (exist[t] & ~elig[t])
            mt = torch.where(att, m[t], 0.0)
            obs = mt / mt.sum(-1, keepdim=True).clamp(min=tiny)
            slot = t % window
            wsum += obs - ring[slot]
            ring[slot] = obs
    return out


def gdn_priors(beta: torch.Tensor, g: torch.Tensor, pl: int, T: int, P: int, ps: int, chunk: int) -> dict:
    """beta, g [J, N, Hv] (J GDN layers) -> page priors [T, P]: "beta" (mean write strength of the
    page's tokens; static) and "forgot" (sum over the page's tokens j of beta_j * (1 - exp(G_t - G_j)),
    G the inclusive cumulative log-decay, i.e. how much of what was written has decayed by step t)."""
    J, N, Hv = beta.shape
    pad = P * ps - N
    bp = F.pad(beta.mean((0, 2)), (0, pad)).reshape(P, ps)
    cnt = F.pad(torch.ones(N, device=beta.device), (0, pad)).reshape(P, ps).sum(-1).clamp(min=1)
    out = {"beta": (bp.sum(-1) / cnt)[None].expand(T, P)}
    G = g.cumsum(1)                                                            # [J, N, Hv]
    forgot = torch.empty((T, P), device=beta.device)
    pos = torch.arange(N, device=beta.device)
    for a in range(0, T, chunk):
        ts = torch.arange(a, min(T, a + chunk), device=beta.device)
        qpos = pl - 1 + ts                                                     # query position of step t
        ret = torch.exp((G[:, qpos][:, :, None, :] - G[:, None, :, :]).clamp(max=0))   # [J, c, N, Hv]
        f = (beta[:, None] * (1 - ret)).mean((0, 3))                           # [c, N]
        f = f.masked_fill(pos[None, :] > qpos[:, None], 0.0)
        forgot[a:a + len(ts)] = F.pad(f, (0, pad)).reshape(len(ts), P, ps).sum(-1)
    out["forgot"] = forgot
    return out


def spearman(a: torch.Tensor, b: torch.Tensor, allowed: torch.Tensor) -> float:
    """Mean over rows of the Spearman correlation of a and b [R, P] over the allowed pages."""
    rs = []
    for i in range(a.shape[0]):
        al = allowed[i]
        if int(al.sum()) < 3:
            continue
        x, y = a[i][al], b[i][al]
        rx, ry = G._ranks(-x).float(), G._ranks(-y).float()
        rx, ry = rx - rx.mean(), ry - ry.mean()
        den = float((rx.norm() * ry.norm()))
        if den > 0:
            rs.append(float((rx * ry).sum()) / den)
    return float(np.mean(rs)) if rs else float("nan")


def analyze_layer(c: dict, pl: int, cfg: dict, device: str, prompt_pages: int) -> dict:
    h1, defs, ps = cfg["hybrid"]["h1"], cfg["definitions"], cfg["definitions"]["page_size"]
    q, k = c["q"], c["k"]
    T, N = q.shape[0], k.shape[1]
    P = math.ceil((pl + T - 1) / ps)
    elig_np = eligibility_matrix(pl, T, P, ps, defs["pinned"])
    exist_np = existing_pages(pl, T, P, ps)
    elig, exist = torch.from_numpy(elig_np).to(device), torch.from_numpy(exist_np).to(device)
    pinned = (exist & ~elig)[:, None, :]
    budgets = budgets_for(h1["budgets_tokens"], ps, exist_np, elig_np, device)
    summ = unit_summaries(k, ps)
    sc = page_scores(q, k, pl, P, ps, h1["step_chunk"], summ)
    m = sc["m"]
    S = m.shape[1]
    score = Scorer(m, torch.zeros_like(m), pinned)
    mm = h1["ours"]["mean_mix"]
    res = {b: {} for b in budgets}
    churns = {}
    rank_sets = {"quest": (sc["quest"], "vote"), "mean": (sc["mean"], "sum"),
                 "ours": ((1 - mm) * sc["quest"] + mm * sc["mean"], "sum")}
    h2 = h2_scores(q, k, pl, P, ps, h1["step_chunk"], summ)
    for n, main in (("mean16", "sum"), ("mean8", "sum"), ("quest4", "vote"), ("meanc", "sum"), ("quest4p2", "vote")):
        rank_sets[n] = (h2[n], main)
    rank_sets["lite"] = ((1 - mm) * h2["quest4p2"] + mm * h2["meanc"], "sum")
    rank_sets["ours48"] = ((1 - mm) * h2["quest4"] + mm * h2["mean8"], "sum")
    del h2
    for la, ql in c["la"].items():
        qu, me = bound_scores(ql, P, h1["step_chunk"], summ)
        rank_sets[f"quest@la{la}"] = (qu, "vote")
        rank_sets[f"ours@la{la}"] = ((1 - mm) * qu + mm * me, "sum")
        del qu, me
    for btok, b in budgets.items():
        sel = G.mask_from_ranks(G.rank_desc(m.sum(1, keepdim=True), elig), b, elig).expand(T, S, P)
        res[btok]["oracle|sum"] = score.whole(sel, b, False)
        if btok == "512":
            churns["oracle|sum"] = churn(sel)
        res[btok]["oracle|per_head"] = score.whole(G.mask_from_ranks(G.rank_desc(m, elig), b, elig), b, True)
    for name, (s_, main_mode) in rank_sets.items():
        ranks = G.static_ranks(s_, elig, "logit")
        modes = ["vote", "sum"] if name == "quest" else [main_mode]
        for btok, b in budgets.items():
            for mode in modes:
                sel = G.select_from_ranks(ranks, elig, b, mode)
                res[btok][f"{name}|{mode}"] = score.whole(sel, b, False)
                if btok == "512" and mode == main_mode:
                    churns[f"{name}|{mode}"] = churn(sel)
        del ranks
    h3 = cfg["hybrid"]["h3"]
    for btok, b in budgets.items():
        kept = snapkv_evict(m, elig, exist, b, h3["snapkv_window"], prompt_pages, h3["snapkv_protect_prompt"])
        res[btok]["snapkv|evict"] = score.whole(kept[:, None, :].expand(T, S, P) & elig[:, None, :], b, False)
    out = {"missing": res, "churn_512": churns, "q_self_err": c["q_self_err"], "n_steps": T, "n_pages": P,
           "head_dim": int(q.shape[-1])}
    if "beta" in c:
        pri = gdn_priors(c["beta"], c["g"], pl, T, P, ps, h1["step_chunk"])
        rows = torch.arange(0, T, 16, device=device)
        out["prior_spearman"] = {n: spearman(p[rows], m.sum(1)[rows], elig[rows]) for n, p in pri.items()}
        ours_sh = G.page_shares(rank_sets["ours"][0], elig, "logit").sum(1)                 # [T, P]
        for n, p in pri.items():
            pn = G.page_shares(p[:, None, :], elig, "mass")[:, 0]                            # [T, P]
            for w in h1["gdn_prior_weights"]:
                key = ((1 - w) * ours_sh + w * S * pn)[:, None, :]
                ranks = G.rank_desc(key, elig)
                for btok, b in budgets.items():
                    res[btok][f"ours+{n}{w:g}|sum"] = score.whole(G.mask_from_ranks(ranks, b, elig).expand(T, S, P), b, False)
                del ranks
    del sc, m, score, rank_sets
    torch.cuda.empty_cache()
    return out


def screen(cfg: dict, args) -> None:
    h1, mc = cfg["hybrid"]["h1"], cfg["hybrid"]["models"][args.model]
    model, tok = load_model(mc["name"], cfg, args.device, attn_implementation=CAPTURE)
    seqs = sequences(cfg, args.model, tok)
    i, n = map(int, args.shard.split("/"))
    prov = provenance(cfg, model=mc["name"], context_length=None, h1=h1)
    hybrid = bool(getattr(model.config, "layer_types", None))
    ps = cfg["definitions"]["page_size"]
    for s in seqs[i::n]:
        out = Path(h1["shard_dir"]) / "screen" / args.model / s["task"] / f"{s['index']:04d}.json"
        if out.exists():
            continue
        pl = len(s["prompt"])
        ids = s["prompt"] + s["gen"][:-1]                     # T = len(gen) decode queries
        cap = capture(model, ids, pl, mc["layers"], h1["lookahead_layers"], gdn=hybrid)
        layers = {}
        for l in mc["layers"]:
            layers[str(l)] = analyze_layer(cap[l], pl, cfg, args.device, prompt_pages=-(-pl // ps))
            print(f"[h1] {args.model} {s['task']}/{s['index']} layer {l} done (q self err {layers[str(l)]['q_self_err']:.2e})", flush=True)
        del cap
        torch.cuda.empty_cache()
        write_json(out, {"provenance": prov, "task": s["task"], "index": s["index"], "prompt_len": pl,
                         "n_gen": len(s["gen"]), "retokenized": s["retokenized"], "score": s["score"],
                         "stopped": s["stopped"], "layers": layers})


# ---------------------------------------------------------------------------- fidelity (H-b offline)
def _sparse_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    """Teacher-forced sparse attention for every decode query (rows >= q0): one page set per layer from
    the live queries and keys, by _STATE['method'] at _STATE['budget']; prompt rows stay dense."""
    st = _STATE
    if st.get("method") is None:                                       # the dense reference
        return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kw)
    q, k, v = query[0], key[0], value[0]                               # [H, N, d], [S, N, d]
    H, N, d = q.shape
    S = k.shape[0]
    scaling = scaling if scaling is not None else d ** -0.5
    ps, q0 = st["ps"], st["q0"]
    P = -(-N // ps)
    T = N - q0
    elig, exist, b = st["elig"][:T, :P], st["exist"][:T, :P], st["b"][:T]
    summ = unit_summaries(k.float(), ps) if st["method"] in ("quest", "ours") else None
    out = torch.empty_like(q)
    dense = sdpa_attention_forward(module, query[:, :, :q0], key, value, attention_mask[:, :, :q0] if attention_mask is not None else None,
                                   scaling=scaling, **kw)[0]                                # [1, q0, H, d]
    out[:, :q0] = dense[0].transpose(0, 1)
    miss = []
    for a in range(0, T, st["chunk"]):
        rows = torch.arange(a, min(T, a + st["chunk"]), device=q.device)
        qc = q[:, q0 + rows].transpose(0, 1).float()                                         # [c, H, d]
        n_keys = q0 + 1 + rows
        att = attention_by_group(qc, k.float(), n_keys, scaling)                            # [c, S, g, N]
        mpage = F.pad(att.mean(2), (0, P * ps - N)).reshape(len(rows), S, P, ps).sum(-1)     # [c, S, P]
        el = elig[rows]
        if st["method"] == "oracle":
            key_s = mpage.sum(1)
        elif st["method"] == "recent":
            key_s = torch.zeros_like(mpage[:, 0])
        else:
            qu = fit_pages(quest_scores(qc, summ, scaling), P)
            if st["method"] == "quest":
                key_s = None
                sel = G.select_from_ranks(G.static_ranks(qu, el, "logit"), el, b[rows], "vote")[:, 0]
            else:
                me = fit_pages(landmark_scores(qc, summ, scaling), P)
                key_s = G.page_shares((1 - st["mean_mix"]) * qu + st["mean_mix"] * me, el, "logit").sum(1)
        if st["method"] == "recent":
            sel = torch.zeros_like(el)
        elif key_s is not None:
            sel = G.mask_from_ranks(G.rank_desc(key_s[:, None, :], el), b[rows], el)[:, 0]
        allowed_p = sel | (exist[rows] & ~el)                                               # + pinned
        tm = allowed_p.repeat_interleave(ps, -1)[:, :N] & (torch.arange(N, device=q.device)[None] < n_keys[:, None])
        miss.append(1 - (mpage * allowed_p[:, None, :]).sum(-1).mean(1))                    # per row
        lg = torch.einsum("csgd,snd->csgn", qc.reshape(len(rows), S, H // S, d), k.float()) * scaling
        p = torch.softmax(lg.masked_fill(~tm[:, None, None, :], float("-inf")), -1)
        o = torch.einsum("csgn,snd->csgd", p.to(v.dtype).float(), v.float()).reshape(len(rows), H, d)
        out[:, q0 + rows] = o.transpose(0, 1).to(q.dtype)
    st["miss"].append(torch.cat(miss))
    return out.transpose(0, 1)[None].contiguous(), None


@torch.no_grad()
def fidelity(cfg: dict, args) -> None:
    h1, mc = cfg["hybrid"]["h1"], cfg["hybrid"]["models"][args.model]
    fc, defs, ps = h1["fidelity"], cfg["definitions"], cfg["definitions"]["page_size"]
    model, tok = load_model(mc["name"], cfg, args.device, attn_implementation=SPARSE)
    seqs = sequences(cfg, args.model, tok)[:fc["n_sequences"]]
    prov = provenance(cfg, model=mc["name"], context_length=None, fidelity=fc)
    configs = [("oracle", 256), ("oracle", 512), ("quest", 512), ("ours", 512), ("quest", 1024), ("ours", 1024), ("recent", 0)]
    for s in seqs:
        out = Path(h1["shard_dir"]) / "fidelity" / args.model / s["task"] / f"{s['index']:04d}.json"
        if out.exists():
            continue
        pl = len(s["prompt"])
        gen = s["gen"][:fc["max_gen_tokens"]]
        ids = s["prompt"] + gen[:-1]
        T = len(gen)
        P = math.ceil((pl + T - 1) / ps)
        elig_np, exist_np = eligibility_matrix(pl, T, P, ps, defs["pinned"]), existing_pages(pl, T, P, ps)
        x = torch.tensor([ids], device=args.device)
        _STATE.clear()
        h_ref = model.model(input_ids=x, use_cache=False).last_hidden_state[0, pl - 1:]          # [T, D]
        res = {}
        for method, btok in configs:
            b = torch.from_numpy(np.maximum(0, btok // ps - (exist_np & ~elig_np).sum(-1))).to(args.device)
            _STATE.clear()
            _STATE.update(method=method, ps=ps, q0=pl - 1, chunk=fc["query_chunk"], mean_mix=h1["ours"]["mean_mix"],
                          elig=torch.from_numpy(elig_np).to(args.device), exist=torch.from_numpy(exist_np).to(args.device),
                          b=b, miss=[])
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0, pl - 1:]
            kl, agree = torch.empty(T, device=args.device), torch.empty(T, device=args.device)
            for a in range(0, T, 512):                                                         # [T, V] logits do not fit
                ref = model.lm_head(h_ref[a:a + 512]).float().log_softmax(-1)
                lp = model.lm_head(h[a:a + 512]).float().log_softmax(-1)
                kl[a:a + 512] = (ref.exp() * (ref - lp)).sum(-1)
                agree[a:a + 512] = (ref.argmax(-1) == lp.argmax(-1)).float()
            miss = torch.stack(_STATE["miss"]).mean(0)                                         # per row, mean over layers
            res[f"{method}@{btok}"] = {"kl": float(kl.mean()), "kl_p90": float(kl.quantile(0.9)), "top1_agree": float(agree.mean()),
                                        "missing": float(miss.mean()), "n_layers": len(_STATE["miss"])}
            del h, kl
            print(f"[h1-fid] {args.model} {s['task']}/{s['index']} {method}@{btok}: KL {res[f'{method}@{btok}']['kl']:.4f} "
                  f"agree {res[f'{method}@{btok}']['top1_agree']:.3f} missing {res[f'{method}@{btok}']['missing']:.4f}", flush=True)
        _STATE.clear()
        write_json(out, {"provenance": prov, "task": s["task"], "index": s["index"], "prompt_len": pl, "n_gen": T,
                         "results": res})
        del h_ref
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------- merge
def merge(cfg: dict) -> None:
    h1 = cfg["hybrid"]["h1"]
    root = Path(h1["shard_dir"])
    out = {"screen": {}, "fidelity": {}}
    for mkey in cfg["hybrid"]["models"]:
        rows = [json.loads(p.read_text()) for p in sorted((root / "screen" / mkey).glob("*/*.json"))]
        if rows:
            agg: dict = {}
            per_layer: dict = {}
            churn_: dict = {}
            spear: dict = {}
            for r in rows:
                for l, v in r["layers"].items():
                    for btok, d in v["missing"].items():
                        for name, x in d.items():
                            agg.setdefault(btok, {}).setdefault(name, []).append(x["missing"])
                            per_layer.setdefault(btok, {}).setdefault(name, {}).setdefault(l, []).append(x["missing"])
                    for name, x in v["churn_512"].items():
                        churn_.setdefault(name, []).append(x)
                    for name, x in (v.get("prior_spearman") or {}).items():
                        spear.setdefault(name, []).append(x)
            by_task = {}
            for task in sorted({r["task"] for r in rows}):
                tr = [r for r in rows if r["task"] == task]
                by_task[task] = {btok: {name: float(np.mean([v["missing"][btok][name]["missing"] for r in tr for v in r["layers"].values()
                                                             if name in v["missing"][btok]]))
                                        for name in agg[btok]} for btok in agg}
            out["screen"][mkey] = {
                "n_sequences": len(rows), "retokenized": sum(r["retokenized"] for r in rows),
                "mean_gen_tokens": float(np.mean([r["n_gen"] for r in rows])),
                "missing": {b: {n: float(np.mean(v)) for n, v in d.items()} for b, d in agg.items()},
                "missing_by_task": by_task,
                "missing_by_layer": {b: {n: {l: float(np.mean(x)) for l, x in d.items()} for n, d in dd.items()} for b, dd in per_layer.items()},
                "churn_512": {n: float(np.mean(v)) for n, v in churn_.items()},
                "prior_spearman": {n: float(np.nanmean(v)) for n, v in spear.items()},
                "max_q_self_err": max(v["q_self_err"] for r in rows for v in r["layers"].values()),
                "head_dim": rows[0]["layers"][next(iter(rows[0]["layers"]))]["head_dim"],
            }
        fr = [json.loads(p.read_text()) for p in sorted((root / "fidelity" / mkey).glob("*/*.json"))]
        if fr:
            keys = fr[0]["results"].keys()
            out["fidelity"][mkey] = {"n_sequences": len(fr), **{k: {m: float(np.mean([r["results"][k][m] for r in fr]))
                                                                    for m in fr[0]["results"][k]} for k in keys}}
    sc = out["screen"]
    g = h1["gate"]
    bt = str(g["budget_tokens"])
    if "qwen35_4b" in sc and "qwen3_4b" in sc:
        a, b = sc["qwen35_4b"]["missing"][bt], sc["qwen3_4b"]["missing"][bt]
        share = lambda d: (d["quest|vote"] - d["oracle|sum"]) / d["quest|vote"]
        out["gate_h1"] = {"budget_tokens": g["budget_tokens"],
                          "oracle": {"qwen35_4b": a["oracle|sum"], "qwen3_4b": b["oracle|sum"]},
                          "quest_gap_share": {"qwen35_4b": share(a), "qwen3_4b": share(b)},
                          "i_passed": a["oracle|sum"] <= b["oracle|sum"], "ii_passed": share(a) >= share(b)}
        la = max(h1["lookahead_layers"])
        cost = {n: 100 * (a[f"{n}@la{la}|{m}"] - a[f"{n}|{m}"]) for n, m in (("quest", "vote"), ("ours", "sum"))}
        out["gate_hc"] = {"lookahead": la, "cost_points": cost, "max_points": g["lookahead_max_points"],
                          "passed": all(v <= g["lookahead_max_points"] for v in cost.values())}
    if "qwen35_4b" in sc:
        t = sc["qwen35_4b"]["missing_by_task"]
        if "aime24" in t and "aime25" in t:
            cands = [n for n in t["aime24"][bt] if n.startswith("ours+")]
            if cands:
                best = min(cands, key=lambda n: t["aime24"][bt][n])
                red = 1 - t["aime25"][bt][best] / t["aime25"][bt]["ours|sum"]
                out["gate_hd"] = {"chosen_on_aime24": best, "aime25_ours": t["aime25"][bt]["ours|sum"],
                                  "aime25_with_prior": t["aime25"][bt][best], "reduction": red,
                                  "min_reduction": g["gdn_prior_min_reduction"], "passed": red >= g["gdn_prior_min_reduction"]}
    ps = cfg["definitions"]["page_size"]
    for mkey, s in sc.items():
        d = s["head_dim"]
        s["metadata_share"] = {n: f(d, ps) / (2 * d * 2) for n, f in H2_BYTES.items()}
    h2c = cfg["hybrid"]["h2"]
    if "qwen35_4b" in sc and "aime24" in sc["qwen35_4b"]["missing_by_task"]:
        t, share = sc["qwen35_4b"]["missing_by_task"], sc["qwen35_4b"]["metadata_share"]
        ok = [n for n in share if share[n] <= h2c["metadata_max_share"] and any(k.startswith(n + "|") for k in t["aime24"][bt])]
        ref = t["aime24"][bt]["ours|sum"]
        main = lambda n: next(k for k in t["aime24"][bt] if k.startswith(n + "|") and "@" not in k)
        fine = [n for n in ok if 100 * (t["aime24"][bt][main(n)] - ref) <= h2c["within_points"]]
        pick = min(fine, key=lambda n: share[n]) if fine else None
        out["h2_choice"] = {"candidates": {n: {"share": share[n], "aime24": t["aime24"][bt][main(n)],
                                               "aime25": t["aime25"][bt][main(n)] if "aime25" in t else None} for n in ok},
                            "ours_aime24": ref, "ours_aime25": t.get("aime25", {}).get(bt, {}).get("ours|sum"),
                            "chosen": pick, "rule": h2c}
    for mkey, s in sc.items():
        print(f"\n== {mkey}: {s['n_sequences']} sequences, mean {s['mean_gen_tokens']:.0f} generated tokens, max q self err {s['max_q_self_err']:.1e}")
        names = sorted(s["missing"][bt], key=lambda n: s["missing"][bt][n])
        print(f"{'method':24s} " + " ".join(f"{b:>7s}" for b in s["missing"]))
        for n in names:
            print(f"{n:24s} " + " ".join(f"{100 * s['missing'][b].get(n, float('nan')):7.2f}" for b in s["missing"]))
        print("churn@512 (new pages/step):", {n: round(v, 2) for n, v in s["churn_512"].items()})
        if s["prior_spearman"]:
            print("GDN prior Spearman vs page mass:", {n: round(v, 3) for n, v in s["prior_spearman"].items()})
    for mkey, f in out["fidelity"].items():
        print(f"\n== fidelity {mkey} ({f['n_sequences']} sequences)")
        for k, v in f.items():
            if isinstance(v, dict):
                print(f"{k:14s} KL {v['kl']:.4f}  p90 {v['kl_p90']:.4f}  top1 {v['top1_agree']:.3f}  missing {100 * v['missing']:.2f}%")
    for k in ("gate_h1", "gate_hc", "gate_hd", "h2_choice"):
        if k in out:
            print(k, json.dumps(out[k]))
    write_json(Path(cfg["paths"]["results_dir"]) / "h1_explore.json",
               {"provenance": provenance(cfg, model=[m["name"] for m in cfg["hybrid"]["models"].values()], context_length=None, h1=h1), **out})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("mode", choices=["screen", "fidelity", "merge"])
    p.add_argument("--model", default="qwen35_4b")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = thinking(config_from_args(args))
    register()
    {"screen": screen, "fidelity": fidelity, "merge": lambda c, a: merge(c)}[args.mode](cfg, args)


if __name__ == "__main__":
    main()
