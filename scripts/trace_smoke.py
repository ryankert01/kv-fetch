"""2K-context validation of the tracing hook and trace schema (spec §10 step 2).

For each model in ``smoke.models``:
  1a. fp32 isolation check: the tracer wraps HF's own eager_attention_forward, so the
      tracer's recompute and HF's eager attention weights see the SAME q/k in the SAME
      forward. Must agree to fp32 noise. Chunked prefill with the boundary inside the
      prefill observation window. Catches head-mapping, step off-by-one, causal-mask and
      page-reduction bugs.
  1b. fp32 cross-path check: traced SDPA decode vs. an independent teacher-forced eager
      ``output_attentions`` forward. Gross-error detector (e.g. broken chunked-prefill
      mask); a small residual is genuine SDPA-vs-eager numerics.
  2.  bf16 (production precision): tracing must not change generated tokens; rows sum
      to 1; top-B agreement with bf16 eager; schema round-trip (full and top-K).
"""
import argparse
import copy
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, load_model, model_dims, stop_token_ids
from kvfetch.provenance import provenance, write_json
from kvfetch.tasks.ruler import RulerNiah
from kvfetch.tracing.capture import capture, trace_meta
from kvfetch.tracing.decode import generate
from kvfetch.tracing.schema import load_trace, save_trace
from kvfetch.tracing.tracer import active_tracer, page_reduce, register_traced_attention

EAGER_REF = "kvtrace_eager_ref"
_RECORDER: "EagerRecorder | None" = None


class EagerRecorder:
    """Stores HF eager attention weights in the tracer's (step, layer) layout."""

    def __init__(self, *, T: int, L: int, H: int, P: int, W: int):
        self.mass = np.zeros((T, L, H, P), np.float32)
        self.obs = np.zeros((W, L, H, P), np.float32)
        # max |logit| per row: fp32 rounding of a logit of size x is ~eps*x, so the
        # admissible probability error scales with it (Qwen2.5 has |logit| ~ 1e4 heads)
        self.mass_logit = np.zeros((T, L, H), np.float32)
        self.obs_logit = np.zeros((W, L, H), np.float32)


def _row_logit_max(module, query, key, scaling) -> torch.Tensor:
    g = module.num_key_value_groups
    k = key[0].repeat_interleave(g, dim=0)  # [Hq, N, d]
    return (torch.matmul(query[0], k.transpose(1, 2)) * scaling).abs().amax(-1)  # [Hq, nq]


def recording_eager(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    eager_fn = sys.modules[type(module).__module__].eager_attention_forward
    out, w = eager_fn(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
    tr, rec = active_tracer(), _RECORDER
    if tr is not None and rec is not None and module.layer_idx in tr.slot and tr._mode is not None:
        slot = tr.slot[module.layer_idx]
        if tr._mode == "decode":
            rec.mass[tr._step, slot] = page_reduce(w[0, :, 0].float(), tr.page_size, tr.n_pages).cpu().numpy()
            rec.mass_logit[tr._step, slot] = _row_logit_max(module, query, key, scaling)[:, 0].cpu().numpy()
        else:
            lo = tr.prompt_len - tr.obs_window
            pos = torch.arange(tr._q_start, tr._q_start + query.shape[2], device=query.device)
            rows = (pos >= lo) & (pos < tr.prompt_len)
            if bool(rows.any()):
                idx = (pos[rows] - lo).cpu().numpy()
                m = page_reduce(w[0][:, rows].float(), tr.page_size, tr.n_pages).transpose(0, 1)
                rec.obs[idx, slot] = m.cpu().numpy()
                # causal mask is irrelevant for a magnitude bound; include all keys
                rec.obs_logit[idx, slot] = _row_logit_max(module, query[:, :, rows], key, scaling).T.cpu().numpy()
    return out, w


def bounded_diff(ours: np.ndarray, ref: np.ndarray, logit_max: np.ndarray, atol: float, rtol: float) -> dict:
    diff = np.abs(ours - ref)
    allowed = atol + rtol * logit_max[..., None]
    return {"max_abs_diff": float(diff.max()), "max_diff_over_bound": float((diff / allowed).max())}


@torch.inference_mode()
def eager_page_mass(model, prompt_ids: torch.Tensor, tokens: list[int], *, page_size: int, n_pages: int,
                    obs_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference m from eager output_attentions. Returns (mass [T,L,H,P], obs [W,L,H,P]) in fp32."""
    n = prompt_ids.numel()
    full = torch.cat([prompt_ids, torch.tensor(tokens[:-1], device=prompt_ids.device)])[None]
    out = model(input_ids=full, output_attentions=True, use_cache=False)
    T = len(tokens)
    steps = torch.arange(n - 1, n - 1 + T, device=full.device)
    obs = torch.arange(n - obs_window, n, device=full.device)
    m = torch.stack([page_reduce(a[0][:, steps].float(), page_size, n_pages) for a in out.attentions], 0)
    o = torch.stack([page_reduce(a[0][:, obs].float(), page_size, n_pages) for a in out.attentions], 0)
    # [L, H, T, P] -> [T, L, H, P]
    return m.permute(2, 0, 1, 3).cpu().numpy(), o.permute(2, 0, 1, 3).cpu().numpy()


def topb_overlap(a: np.ndarray, b: np.ndarray, frac: float) -> float:
    P = a.shape[-1]
    B = max(1, int(np.ceil(frac * P)))
    ia = np.argpartition(-a, B - 1, axis=-1)[..., :B]
    ib = np.argpartition(-b, B - 1, axis=-1)[..., :B]
    ma = np.zeros(a.shape, bool)
    mb = np.zeros(b.shape, bool)
    np.put_along_axis(ma, ia, True, -1)
    np.put_along_axis(mb, ib, True, -1)
    return float((ma & mb).sum(-1).mean() / B)


def check_model(name: str, cfg: dict, device: str) -> dict:
    sm = cfg["smoke"]
    ps, W = cfg["definitions"]["page_size"], cfg["trace"]["prefill_obs_window"]
    report: dict = {"model": name}

    global _RECORDER
    cfg32 = copy.deepcopy(cfg)
    cfg32["models"]["dtype"] = "float32"

    # ---- 1a. fp32 isolation: tracer recompute vs HF eager weights, same forward
    register_traced_attention(EAGER_REF, inner=recording_eager, mask="eager")
    model, tok = load_model(name, cfg32, device, attn_implementation=EAGER_REF)
    dims = model_dims(model)
    stops = stop_token_ids(name, tok, cfg)
    sample = RulerNiah(sm["task"], cfg).samples(tok, max_seq_length=sm["context_length"], num_samples=1, seed=0)[0]
    ids = torch.tensor(encode(tok, sample.prompt)["input_ids"], device=device)
    n = ids.numel()
    straddle_chunk = n - W // 2  # chunk boundary inside the obs window
    layers = list(range(dims.num_layers))
    T = sm["decode_steps"]
    _RECORDER = EagerRecorder(T=T, L=len(layers), H=dims.num_heads, P=math.ceil((n + T - 1) / ps), W=min(W, n))
    res_i, tr_i = capture(model, ids, layers=layers, cfg=cfg, decode_steps=sm["decode_steps"], stop_ids=stops,
                          suppress_stop=True, prefill_chunk_size=straddle_chunk, mass_dtype="float32")
    rec = _RECORDER
    rec.mass[0], rec.mass_logit[0] = rec.obs[-1], rec.obs_logit[-1]  # step 0 = last prompt query
    iso_m = bounded_diff(tr_i.mass, rec.mass, rec.mass_logit, sm["fp32_atol"], sm["fp32_logit_rtol"])
    iso_o = bounded_diff(tr_i.obs_numpy(), rec.obs, rec.obs_logit, sm["fp32_atol"], sm["fp32_logit_rtol"])
    _RECORDER = None
    del model
    torch.cuda.empty_cache()

    # ---- 1b. fp32 cross-path: traced SDPA decode vs teacher-forced eager
    model, _ = load_model(name, cfg32, device, traced=True)
    res, tr = capture(model, ids, layers=layers, cfg=cfg, decode_steps=sm["decode_steps"], stop_ids=stops,
                      suppress_stop=True, prefill_chunk_size=straddle_chunk, mass_dtype="float32")
    del model
    torch.cuda.empty_cache()
    eager, _ = load_model(name, cfg32, device, attn_implementation="eager")
    ref_m, ref_o = eager_page_mass(eager, ids, res.tokens, page_size=ps, n_pages=tr.n_pages, obs_window=tr.obs_window)
    del eager
    torch.cuda.empty_cache()
    d_m = float(np.abs(tr.mass - ref_m).max())
    d_o = float(np.abs(tr.obs_numpy() - ref_o).max())
    report["fp32"] = {"prompt_len": n, "prefill_chunk": straddle_chunk,
                      "isolation_decode": iso_m, "isolation_obs": iso_o,
                      "max_logit": float(max(rec.mass_logit.max(), rec.obs_logit.max())),
                      "crosspath_max_abs_diff_decode": d_m, "crosspath_max_abs_diff_obs": d_o,
                      "pass": (max(iso_m["max_diff_over_bound"], iso_o["max_diff_over_bound"]) <= 1.0
                               and max(d_m, d_o) <= sm["fp32_crosspath_atol"])}

    # ---- 2. bf16 checks -----------------------------------------------------
    model, _ = load_model(name, cfg, device, traced=True)
    res16, tr16 = capture(model, ids, layers=layers, cfg=cfg, decode_steps=sm["decode_steps"], stop_ids=stops,
                          suppress_stop=True, prefill_chunk_size=straddle_chunk)
    plain = generate(model, ids, max_new_tokens=sm["decode_steps"], stop_ids=stops, suppress_stop=True,
                     prefill_chunk_size=straddle_chunk)  # same model object, tracer inactive
    del model
    torch.cuda.empty_cache()
    sums = tr16.mass.astype(np.float32).sum(-1)
    eager16, _ = load_model(name, cfg, device, attn_implementation="eager")
    ref16, _ = eager_page_mass(eager16, ids, res16.tokens, page_size=ps, n_pages=tr16.n_pages, obs_window=tr16.obs_window)
    del eager16
    torch.cuda.empty_cache()
    overlap = topb_overlap(tr16.mass.astype(np.float32), ref16, sm["bf16_topb_frac"])

    prov = provenance(cfg, model=name, context_length=n)
    meta = trace_meta(prov=prov, tracer=tr16, res=res16, task=sm["task"], sample_id="smoke-0")
    with tempfile.TemporaryDirectory() as d:
        pf = save_trace(Path(d) / "full.safetensors", meta=meta, mass=tr16.mass, obs_mass=tr16.obs_numpy(),
                        tokens=np.array(res16.tokens), prompt_ids=ids.cpu().numpy(), fmt="full")
        pk = save_trace(Path(d) / "topk.safetensors", meta=meta, mass=tr16.mass, obs_mass=tr16.obs_numpy(),
                        tokens=np.array(res16.tokens), prompt_ids=ids.cpu().numpy(), fmt="topk",
                        topk=cfg["trace"]["topk"])
        tf, tk = load_trace(pf), load_trace(pk)
        roundtrip_full = bool(np.array_equal(tf.mass, tr16.mass) and np.array_equal(tf.obs_mass, tr16.obs_numpy())
                              and tf.tokens.tolist() == res16.tokens and tf.prompt_ids.tolist() == ids.tolist()
                              and tf.meta["layers"] == layers)
        dense_k = tk.dense()
        kept = dense_k > 0
        resid_err = float(np.abs(tk.residual + dense_k.astype(np.float32).sum(-1)
                                 - tr16.mass.astype(np.float32).sum(-1)).max())
        roundtrip_topk = bool(np.array_equal(dense_k[kept], tr16.mass[kept]) and resid_err < 1e-3)
        sizes = {"full_bytes": pf.stat().st_size, "topk_bytes": pk.stat().st_size}

    report["bf16"] = {
        "tokens_identical_traced_vs_untraced": res16.tokens == plain.tokens,
        "max_row_sum_err": float(np.abs(sums - 1).max()),
        "topb_overlap_vs_eager": overlap,
        "roundtrip_full": roundtrip_full, "roundtrip_topk": roundtrip_topk, "topk_residual_err": resid_err,
        **sizes,
        "decode_s_per_step_traced_all_layers": res16.decode_s / max(1, len(res16.tokens) - 1),
        "decode_s_per_step_untraced": plain.decode_s / max(1, len(plain.tokens) - 1),
    }
    b = report["bf16"]
    b["pass"] = bool(b["tokens_identical_traced_vs_untraced"] and b["max_row_sum_err"] <= sm["row_sum_atol"]
                     and overlap >= sm["bf16_min_topb_overlap"] and roundtrip_full and roundtrip_topk)
    report["pass"] = report["fp32"]["pass"] and b["pass"]
    report["provenance"] = prov
    report["answer"] = sample.answers
    report["generated"] = tok.decode(res16.natural_tokens())
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = config_from_args(args)
    reports = []
    for name in cfg["smoke"]["models"]:
        r = check_model(name, cfg, args.device)
        reports.append(r)
        f, b = r["fp32"], r["bf16"]
        print(f"[smoke] {name}: pass={r['pass']} | fp32 isolation max|dm| decode={f['isolation_decode']['max_abs_diff']:.1e} "
              f"obs={f['isolation_obs']['max_abs_diff']:.1e} (worst/bound "
              f"{max(f['isolation_decode']['max_diff_over_bound'], f['isolation_obs']['max_diff_over_bound']):.2f}, "
              f"max|logit|={f['max_logit']:.0f}); cross-path decode={f['crosspath_max_abs_diff_decode']:.1e} "
              f"obs={f['crosspath_max_abs_diff_obs']:.1e} | bf16 tokens_same={b['tokens_identical_traced_vs_untraced']} "
              f"row_sum_err={b['max_row_sum_err']:.1e} topB_overlap={b['topb_overlap_vs_eager']:.3f} "
              f"roundtrip={b['roundtrip_full']}/{b['roundtrip_topk']} | step "
              f"{1e3 * b['decode_s_per_step_untraced']:.1f}ms -> {1e3 * b['decode_s_per_step_traced_all_layers']:.1f}ms (all layers)",
              flush=True)
    write_json(cfg["smoke"]["out"], {"all_pass": all(r["pass"] for r in reports), "reports": reports})


if __name__ == "__main__":
    main()
