"""Checks for E3's sparse decoding (kvfetch/sparse_decode.py), run before any accuracy job.

  CUDA_VISIBLE_DEVICES=1 python scripts/check_e3.py configs/reasoning.yaml logits [--steps 300]
  CUDA_VISIBLE_DEVICES=1 python scripts/check_e3.py configs/reasoning.yaml replay [--trace math500/0012] [--steps 1500]

logits: teacher-force HF's greedy continuation of a few AIME prompts; every method with a budget
  larger than the context must give HF's next-token logits up to kernel noise (as the dense path
  does), at batch 1 and in a mixed-length batch.
replay: teacher-force an explore trace in shadow mode (dense attention, so queries follow the
  trace) and compare each layer's online page sets with E2b's offline replay of the same method
  (kvfetch/selectors_gpu.py) fed the queries and keys the engine computed. Sets must be identical
  up to float ties.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import DynamicCache

import torch.nn.functional as F

from kvfetch import selectors_gpu as G
from kvfetch.baselines import attention_by_group, landmark_scores, quest_scores, unit_summaries
from kvfetch import sparse_decode as SD
from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.models import encode, format_prompt, load_model, stop_token_ids
from kvfetch.sim import existing_pages
from kvfetch.tasks.sweep import LOADERS
from kvfetch.tracing.schema import load_trace
from compare_pages import fit_pages


def thinking(cfg: dict) -> dict:
    kw = dict(cfg["models"].get("chat_template_kwargs", {}) or {}, enable_thinking=True)
    return {**cfg, "models": {**cfg["models"], "chat_template_kwargs": kw}}


def make_engine(model, cfg, method, batch, max_len, device, shadow=False):
    defs = cfg["definitions"]
    return SD.SparseEngine(model, method, batch=batch, max_len=max_len, page_size=defs["page_size"],
                           sink_pages=defs["pinned"]["sink_pages"], recent_tokens=defs["pinned"]["recent_tokens"],
                           obs_window=cfg["trace"]["prefill_obs_window"], device=device, shadow=shadow)


def methods(cfg, budget):
    e3 = cfg["e3"]
    return [SD.Method("quest", budget), SD.Method("ours", budget, spec=cfg["selectors"]["candidates"][e3["ours"]]),
            SD.Method("snapkv", budget, window=e3["snapkv_window"]), SD.Method("oracle", budget)]


def check_logits(cfg, args, device):
    name = cfg["sweep"]["model"]
    model, tok = load_model(name, cfg, device, attn_implementation=SD.SPARSE_ATTN)
    ref, _ = load_model(name, cfg, device)
    stops = stop_token_ids(name, tok, cfg)
    samples = LOADERS["aime"](tok, cfg["sweep"]["tasks"]["aime24"], 0, 6)
    prompts = [encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"] for s in samples[::2][:3]]
    refs, ref_logits = [], []
    with torch.inference_mode():
        for p in prompts:                   # HF: SDPA + DynamicCache, greedy, one token per forward
            cache, toks, lg = DynamicCache(), [], []
            out = ref(input_ids=torch.tensor([p], device=device), past_key_values=cache, use_cache=True)
            for _ in range(args.steps):
                lg.append(out.logits[0, -1].float().cpu())
                toks.append(int(lg[-1].argmax()))
                out = ref(input_ids=torch.tensor([[toks[-1]]], device=device), past_key_values=cache, use_cache=True)
            refs.append(toks)
            ref_logits.append(torch.stack(lg))
    del ref
    torch.cuda.empty_cache()
    max_len = max(map(len, prompts)) + args.steps + 1
    greedy = {"temperature": 0, "top_k": 1, "top_p": 1.0}
    big = cfg["definitions"]["page_size"] * (max_len // cfg["definitions"]["page_size"] + 1)
    for method, batch in [(SD.Method("full"), 1), (SD.Method("full"), 4)] + [(m, 4) for m in methods(cfg, big)]:
        eng = make_engine(model, cfg, method, batch, max_len, device)
        lo = {}
        jobs = [SD.Job(key=(i,), prompt_ids=p, seed=[i], forced=refs[i]) for i, p in enumerate(prompts)]
        SD.run_batch(model, eng, jobs, stop_ids=[], max_new_tokens=args.steps, sampling=greedy,
                     on_done=lambda *a: None, logits_out=lo)
        cols = []
        for i in range(len(prompts)):
            a, b = torch.stack(lo[(i,)]), ref_logits[i]
            cols.append(f"{float((a - b).abs().mean()):.4f}/{float((a - b).abs().max()):.2f}/"
                        f"{float((a.argmax(-1) == b.argmax(-1)).float().mean()):.3f}")
        print(f"{method.label:14s} batch {batch}  mean|dlogit|/max/argmax agree: " + "  ".join(cols), flush=True)
        del eng
        torch.cuda.empty_cache()


def offline_scores(q: torch.Tensor, k: torch.Tensor, n0: int, P: int, ps: int, chunk: int = 128) -> dict:
    """E2b's inputs (scripts/compare_pages.page_scores) from given queries [T, H, d] and keys [S, N, d]
    (query j sees keys [0, n0 + j)): exact page mass m, Quest bounds and mean-key logits, each [T, S, P]."""
    T, d = q.shape[0], q.shape[-1]
    scaling = d ** -0.5
    summ = unit_summaries(k, ps)
    n_keys = torch.arange(T, device=q.device) + n0
    out = {n: torch.empty((T, k.shape[0], P), device=q.device) for n in ("m", "quest", "mean")}
    pad = P * ps - k.shape[1]
    for a in range(0, T, chunk):
        sl = slice(a, min(T, a + chunk))
        att = attention_by_group(q[sl], k, n_keys[sl], scaling).mean(2)
        out["m"][sl] = F.pad(att, (0, pad)).reshape(*att.shape[:-1], P, ps).sum(-1)
        out["quest"][sl] = fit_pages(quest_scores(q[sl], summ, scaling), P)
        out["mean"][sl] = fit_pages(landmark_scores(q[sl], summ, scaling), P)
    return out


def check_replay(cfg, args, device):
    """Online selections vs the offline replay (selectors_gpu, as in E2b) fed the same queries and
    keys the engine computed. Also reports how far those inputs are from the trace's (replayed on
    another kernel path: bf16 differences of a few percent, which flip near-tied pages)."""
    defs, ps = cfg["definitions"], cfg["definitions"]["page_size"]
    task, idx = args.trace.split("/")
    tr = load_trace(Path(cfg["sweep"]["trace_dir"]) / task / f"{idx}.safetensors")
    meta, pl = tr.meta, tr.meta["prompt_len"]
    if meta.get("split") != "explore":
        raise SystemExit("checks use explore traces only")
    T = min(args.steps, tr.mass.shape[0])
    keys = load_file(str(Path(cfg["e2"]["keys_dir"]) / task / f"{idx}.safetensors"))["keys"]
    model, _ = load_model(cfg["sweep"]["model"], cfg, device, attn_implementation=SD.SPARSE_ATTN)
    btok = cfg["e3"]["check_budget_tokens"]
    spec = cfg["selectors"]["candidates"][cfg["e3"]["ours"]]
    layers = meta["layers"]
    if getattr(model.config, "layer_types", None):    # engine state below is indexed by model layer
        raise SystemExit("replay check: Qwen3-style models only (every layer full attention)")
    for method in methods(cfg, btok)[:2]:
        eng = make_engine(model, cfg, method, 1, pl + T + 1, device, shadow=True)
        eng.record = {layer: [] for layer in layers}
        qs, init = {layer: [] for layer in layers}, {}
        attend, end_prefill = eng.attend, eng.end_prefill

        def hook(li, q, K, V, scaling):
            if li in qs and eng.mode == "decode":
                qs[li].append(q[0, :, 0].float().clone())
            return attend(li, q, K, V, scaling)

        def snap(n):
            if method.name == "ours":
                init.update({layer: eng.hist[layer, 0].clone() for layer in layers})
            end_prefill(n)
        eng.attend, eng.end_prefill = hook, snap
        SD.run_batch(model, eng, [SD.Job(key=(0,), prompt_ids=tr.prompt_ids.tolist(), seed=[0], forced=tr.tokens[:T].tolist())],
                     stop_ids=[], max_new_tokens=T, sampling={"temperature": 0, "top_k": 1, "top_p": 1.0}, on_done=lambda *a: None,
                     graphs=False)                                     # the hooks read tensors per step
        for li, layer in enumerate(layers):
            q = torch.stack(qs[layer])                                    # decode step k = trace step k + 1
            Tn = q.shape[0]
            N = pl + Tn
            k = eng.k[layer, 0, :, :N].float()
            P = eng.P
            q_tr = torch.from_numpy(tr.extra["q"][1:Tn + 1, li].astype(np.float32)).to(device)
            k_tr = keys[li, :, :N].to(device).float()
            rel = lambda a, b: float((a - b).norm() / b.norm())
            # online decode step j is trace step j + 1 (step 0 is the dense prefill): replay steps 1..Tn
            sc = offline_scores(q, k, pl + 1, P, ps)
            elig = torch.from_numpy(eligibility_matrix(pl, Tn + 1, P, ps, defs["pinned"])[1:]).to(device)
            exist = torch.from_numpy(existing_pages(pl, Tn + 1, P, ps)[1:]).to(device)
            b = (btok // ps - (exist & ~elig).sum(-1)).clamp(min=0)
            if method.name == "ours":
                gen = G.replay([spec], sc["quest"][None], [0], b[None], m=sc["m"], init=init[layer][:, :P], eligible=elig,
                               exist=exist, shared="sum", qm=sc["mean"])
                off = torch.stack([c[0, 0] for _, c in gen])
            else:
                off = G.select_from_ranks(G.static_ranks(sc["quest"], elig), elig, b, "vote")[:, 0]
            a, o = torch.stack([r[0] for r in eng.record[layer]]).to(device), off
            nontrivial = elig.sum(-1) > b                              # more eligible pages than budget
            same = (a == o).all(-1)
            inter, union = (a & o).sum(-1).float(), (a | o).sum(-1).float()
            jac = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))[nontrivial]
            m = sc["m"].mean(1)                                                # attention share per page
            gap = ((m * (a & ~o)).sum(-1) - (m * (o & ~a)).sum(-1))[nontrivial]
            print(f"{method.label:12s} layer {layer:2d}: {int(nontrivial.sum())}/{Tn} steps select, identical sets "
                  f"{float(same[nontrivial].float().mean()):.4f}, mean Jaccard {float(jac.mean()):.5f}, coverage gap "
                  f"mean {float(gap.mean()):+.1e} max |{float(gap.abs().max()):.1e}| | inputs vs trace: "
                  f"q {rel(q, q_tr):.3f}, k {rel(k, k_tr):.3f} (relative norm)", flush=True)
        del eng
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("check", choices=["logits", "replay"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--trace", default="math500/0012")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = thinking(config_from_args(args))
    SD.register_sparse_attention()
    if args.check == "logits":
        args.steps = args.steps or 300
        check_logits(cfg, args, args.device)
    else:
        args.steps = args.steps or 1500
        check_replay(cfg, args, args.device)


if __name__ == "__main__":
    main()
