"""H4 (EXPERIMENTS.md, "Hybrid models"): the pages a host-resident KV must move per decode step, on
Qwen3.5-4B's own sequences (sample h1.sample of H3's full run, every h4.residency.every-th explore
problem, up to h4.residency.max_gen_tokens), for H3's selector ("ours" = hybrid.h2.chosen) at
h4.residency.budget_tokens. Run with .venv-hybrid.

  CUDA_VISIBLE_DEVICES=4 .venv-hybrid/bin/python scripts/h4_residency.py configs/hybrid.yaml screen
  .venv-hybrid/bin/python scripts/h4_residency.py configs/hybrid.yaml merge

Policies, per attention layer and step (the sink + recent pages stay on the GPU; a page that ages out
of the recent window is resident, it was just written; the GPU page cache holds c x budget pages, LRU):
  exact c      attend the pages selected with the step's own query (as H3): the selected pages not
               resident are fetched before the layer can attend (c = 1 is H1's churn),
  la3 c+k      exact, plus a prefetch of the top k x budget pages ranked with the query predicted from
               the hidden state three layers earlier (H-c), issued during the three GDN layers before
               the attention layer: prefetch traffic and the residual synchronous misses; selection
               (accuracy) is unchanged,
  lag1 c       attend the pages selected at the previous step: their fetch overlaps a whole step; the
               cost is the missing attention of the stale set vs the exact one.
merge combines the per-step page counts with results/h4_costs_<model>.json (layer times under CUDA
graphs, host-to-GPU copy bandwidth) into a step-time model per batch and a capacity estimate at 32K.
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

from kvfetch import selectors_gpu as G
from kvfetch import sparse_decode as SD
from kvfetch.baselines import attention_by_group, landmark_scores, unit_summaries
from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.models import encode, format_prompt, load_model
from kvfetch.provenance import provenance, write_json
from kvfetch.sim import existing_pages
from compare_pages import Scorer, fit_pages
from run_e3 import problems, thinking
import h1_screen as H1


def policies(rc: dict) -> list[dict]:
    la = rc["prefetch"]["lookahead"]
    out = [{"name": f"exact c{c:g}", "need": "exact", "cache": c, "pre": 0} for c in rc["cache_multiples"]]
    out += [{"name": f"la{la} c{c:g}+k{k:g}", "need": "exact", "cache": c, "pre": k} for c, k in rc["prefetch"]["cache_plus"]]
    out += [{"name": f"lag1 c{c:g}", "need": "lag1", "cache": c, "pre": 0} for c in rc["lag1_cache_multiples"]]
    return out


def simulate(need: torch.Tensor, pre: torch.Tensor, aged: torch.Tensor, cap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """LRU page cache per policy. need, pre [T, K, P] (pages attended / prefetched at step t), aged
    [T, P] (pages leaving the pinned recent window at t: resident), cap [T, K] (cache pages). Returns
    the synchronous misses and the prefetch traffic per step, each [T, K]. Use order within a step:
    aged pages (written earlier), then prefetched, then attended; ties evict the lower page index."""
    T, K, P = need.shape
    dev = need.device
    last = torch.full((K, P), -1, dtype=torch.long, device=dev)          # 3 t + {0, 1, 2}; -1: not resident
    sync = torch.zeros((T, K), dtype=torch.long, device=dev)
    traffic = torch.zeros_like(sync)
    ar = torch.arange(P, device=dev).expand(K, P)
    big = torch.iinfo(torch.long).max
    for t in range(T):                                    # no host syncs: the loop stays on the GPU queue
        last = torch.where(aged[t][None], 3 * t, last)
        traffic[t] = (pre[t] & (last < 0)).sum(-1)
        last = torch.where(pre[t], 3 * t + 1, last)
        sync[t] = (need[t] & (last < 0)).sum(-1)
        last = torch.where(need[t], 3 * t + 2, last)
        res = last >= 0
        excess = (res.sum(-1) - cap[t]).clamp(min=0)
        order = torch.argsort(torch.where(res, last, big), dim=-1, stable=True)
        rank = torch.empty_like(order).scatter_(-1, order, ar)
        last = last.masked_fill(res & (rank < excess[:, None]), -1)
    return sync, traffic


def summary_scores(q: torch.Tensor, summ: dict, P: int, chunk: int, fmt: str) -> torch.Tensor:
    """Mean-key logits [T, S, P] in the H2 format of hybrid.h2.chosen (mean16 / mean8)."""
    mean = {"mean16": summ["mean"].half().float(), "mean8": summ["mean"].to(torch.float8_e4m3fn).float()}[fmt]
    T, S = q.shape[0], mean.shape[0]
    out = torch.empty((T, S, P), device=q.device)
    for a in range(0, T, chunk):
        out[a:a + chunk] = fit_pages(landmark_scores(q[a:a + chunk].float(), {"mean": mean}, q.shape[-1] ** -0.5), P)
    return out


def page_mass(q: torch.Tensor, k: torch.Tensor, pl: int, P: int, ps: int, chunk: int) -> torch.Tensor:
    T = q.shape[0]
    n_keys = torch.arange(T, device=q.device) + pl
    m = torch.empty((T, k.shape[0], P), device=q.device)
    pad = P * ps - k.shape[1]
    for a in range(0, T, chunk):
        sl = slice(a, min(T, a + chunk))
        att = torch.nn.functional.pad(attention_by_group(q[sl].float(), k, n_keys[sl], q.shape[-1] ** -0.5), (0, pad))
        m[sl] = att.reshape(*att.shape[:-1], P, ps).sum(-1).mean(2)
    return m


def analyze_layer(c: dict, pl: int, cfg: dict, device: str) -> dict:
    rc, defs, ps = cfg["hybrid"]["h4"]["residency"], cfg["definitions"], cfg["definitions"]["page_size"]
    fmt = cfg["hybrid"]["h2"]["chosen"]
    q, k = c["q"], c["k"]
    T = q.shape[0]
    P = math.ceil((pl + T - 1) / ps)
    elig_np = eligibility_matrix(pl, T, P, ps, defs["pinned"])
    exist_np = existing_pages(pl, T, P, ps)
    elig, exist = torch.from_numpy(elig_np).to(device), torch.from_numpy(exist_np).to(device)
    b = H1.budgets_for([rc["budget_tokens"]], ps, exist_np, elig_np, device)[str(rc["budget_tokens"])]
    summ = unit_summaries(k, ps)
    m = page_mass(q, k, pl, P, ps, rc["step_chunk"])
    S = m.shape[1]
    score = Scorer(m, torch.zeros_like(m), (exist & ~elig)[:, None, :])
    sel = G.select_from_ranks(G.static_ranks(summary_scores(q, summ, P, rc["step_chunk"], fmt), elig, "logit"), elig, b, "sum")[:, 0]
    la = rc["prefetch"]["lookahead"]
    la_ranks = G.static_ranks(summary_scores(c["la"][la], summ, P, rc["step_chunk"], fmt), elig, "logit")
    lag = torch.zeros_like(sel)
    lag[1:] = sel[:-1]
    lag[0] = sel[0]                                       # step 0: the prompt's last query (dense prefill)
    aged = elig.clone()
    aged[1:] &= ~elig[:-1]
    pols = policies(rc)
    need = torch.stack([sel if p["need"] == "exact" else lag for p in pols], 1)
    pre = torch.stack([G.select_from_ranks(la_ranks, elig, (p["pre"] * b).long(), "sum")[:, 0] if p["pre"] else torch.zeros_like(sel)
                       for p in pols], 1)
    cap = torch.stack([(p["cache"] * b).long() for p in pols], 1)
    sync, traffic = simulate(need, pre, aged, cap)
    out = {"n_steps": T, "n_pages": P, "budget_pages_mean": float(b.float().mean()),
           "missing": {"exact": score.whole(sel[:, None].expand(T, S, P), b, False)["missing"],
                       "lag1": score.whole(lag[:, None].expand(T, S, P) & elig[:, None], b, False)["missing"]},
           "policies": {}}
    for i, p in enumerate(pols):
        s, tr = sync[:, i].float(), traffic[:, i].float()
        out["policies"][p["name"]] = {"sync_mean": float(s.mean()), "sync_p99": float(s.quantile(0.99)),
                                      "prefetch_mean": float(tr.mean()), **p}
    del m, score, la_ranks, need, pre
    torch.cuda.empty_cache()
    return out


def screen(cfg: dict, args) -> None:
    h1, rc = cfg["hybrid"]["h1"], cfg["hybrid"]["h4"]["residency"]
    mc = cfg["hybrid"]["models"][rc["model"]]
    model, tok = load_model(mc["name"], cfg, args.device, attn_implementation=H1.CAPTURE)
    prov = provenance(cfg, model=mc["name"], context_length=None, residency=rc, h2_chosen=cfg["hybrid"]["h2"]["chosen"])
    defs = cfg["definitions"]
    eng = SD.SparseEngine(model, SD.Method("full"), batch=1, max_len=defs["page_size"], page_size=defs["page_size"],
                          sink_pages=defs["pinned"]["sink_pages"], recent_tokens=defs["pinned"]["recent_tokens"],
                          obs_window=cfg["trace"]["prefill_obs_window"], device=args.device)
    sizes = {"weights": sum(t.numel() * t.element_size() for t in model.parameters()),   # the engine's own buffers
             "state_per_seq": sum(t.numel() * t.element_size() for l in eng.lin_layers
                                  for t in (eng.layers[l].conv_states[0], eng.layers[l].recurrent_states[0])),
             "kv_per_token": 2 * eng.k[:, 0, :, 0].numel() * eng.k.element_size(),
             "n_layers": model.config.num_hidden_layers, "n_attn": eng.L, "head_dim": eng.d}
    del eng
    if sizes["kv_per_token"] != cfg["hybrid"]["h4"]["kv_bytes_per_token"]:
        raise RuntimeError(f"h4.kv_bytes_per_token {cfg['hybrid']['h4']['kv_bytes_per_token']} != the engine's {sizes['kv_per_token']}")
    for s in problems({**cfg, "e3": cfg["hybrid"]["h3"]}, tok)[::rc["every"]][:args.limit]:
        out = Path(rc["shard_dir"]) / s.task / f"{s.index:04d}.json"
        f = Path(rc["sequences"]) / s.task / f"{s.index:04d}_s{h1['sample']}.json"
        if out.exists() or not f.exists():
            continue
        r = json.loads(f.read_text())
        prompt = encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"]
        if len(prompt) != r["prompt_tokens"]:
            raise RuntimeError(f"{f}: prompt has {len(prompt)} tokens, the run had {r['prompt_tokens']}")
        gen = r["token_ids"][:args.max_gen or rc["max_gen_tokens"]]
        pl = len(prompt)
        cap = H1.capture(model, prompt + gen[:-1], pl, mc["layers"], [rc["prefetch"]["lookahead"]], gdn=False)
        layers = {}
        for l in mc["layers"]:
            if cap[l]["q_self_err"] > 1e-2:
                raise RuntimeError(f"recomputed queries differ from the model's at layer {l}: {cap[l]['q_self_err']}")
            layers[str(l)] = analyze_layer(cap[l], pl, cfg, args.device)
            print(f"[h4] {s.task}/{s.index} layer {l}: " + ", ".join(f"{n} {v['sync_mean']:.2f}+{v['prefetch_mean']:.2f}"
                                                                    for n, v in layers[str(l)]["policies"].items()), flush=True)
        del cap
        torch.cuda.empty_cache()
        if args.max_gen:
            print("dry run (--max-gen): not written")
            continue
        write_json(out, {"provenance": prov, "sizes": sizes, "task": s.task, "index": s.index, "prompt_len": pl, "n_gen": len(gen),
                         "stopped": r["stopped"], "layers": layers})


# ---------------------------------------------------------------------------- merge: cost model
def copy_model(copy: dict) -> tuple[float, float]:
    """Least-squares fit t = a + bytes / bw to the measured staged gathers (1-16 MiB; the 64 MiB point
    is slower and only reached by batches far above the measured ones). Returns (a in s, bw in B/s)."""
    xs = [int(s) * 2 ** 20 for s in copy["gather_gbps"] if int(s) <= 16]
    ts = [x / (copy["gather_gbps"][str(x // 2 ** 20)] * 1e9) for x in xs]
    slope, a = np.polyfit(xs, ts, 1)
    return float(max(a, 0.0)), float(1 / slope)


def merge(cfg: dict) -> None:
    h4, rc = cfg["hybrid"]["h4"], cfg["hybrid"]["h4"]["residency"]
    mkey = rc["model"]
    rows = [json.loads(p.read_text()) for p in sorted(Path(rc["shard_dir"]).glob("*/*.json"))]
    if not rows:
        raise SystemExit("no residency shards")
    names = list(rows[0]["layers"][next(iter(rows[0]["layers"]))]["policies"])
    pol = {n: {k: float(np.mean([v["policies"][n][k] for r in rows for v in r["layers"].values()]))
               for k in ("sync_mean", "sync_p99", "prefetch_mean")} for n in names}
    for n in names:
        pol[n].update({k: rows[0]["layers"][next(iter(rows[0]["layers"]))]["policies"][n][k] for k in ("need", "cache", "pre")})
    miss = {k: float(np.mean([v["missing"][k] for r in rows for v in r["layers"].values()])) for k in ("exact", "lag1")}
    costs = json.loads((Path(cfg["paths"]["results_dir"]) / f"h4_costs_{mkey}.json").read_text())
    a, bw = copy_model(costs["copy"])
    bw_pcie = max(costs["copy"]["contiguous_gbps"].values()) * 1e9
    sz = rows[0]["sizes"]
    n_attn, n_gdn = sz["n_attn"], sz["n_layers"] - sz["n_attn"]
    window_layers = n_gdn // n_attn
    page = h4["page_bytes"]
    lay = costs["layers"]
    t_copy = lambda nbytes, bw_: (a if nbytes > 0 else 0.0) + nbytes / bw_
    model = []
    for row in lay:
        if row["method"] != f"quest@{rc['budget_tokens']}":
            continue
        B, n = row["batch"], row["context"]
        gdn = next(r["gdn_layer_ms"] for r in lay if r["batch"] == B and r["context"] == n and r["method"] == "full") / 1e3
        full = next(r["attention_layer_ms"] for r in lay if r["batch"] == B and r["context"] == n and r["method"] == "full") / 1e3
        attn = row["attention_layer_ms"] / 1e3
        step_gpu = n_gdn * gdn + n_attn * attn
        out = {"batch": B, "context": n, "step_gpu_sparse_ms": 1e3 * step_gpu,
               "step_gpu_full_ms": 1e3 * (n_gdn * gdn + n_attn * full), "gdn_window_ms": 1e3 * window_layers * gdn, "policies": {}}
        for name, p in pol.items():
            for link, bw_ in (("staged", bw), ("pcie", bw_pcie)):
                sync = t_copy(p["sync_mean"] * B * page, bw_)
                if p["need"] == "lag1":                   # all of a step's fetches overlap the next step
                    over = max(0.0, n_attn * sync - step_gpu)
                else:
                    pre = t_copy(p["prefetch_mean"] * B * page, bw_)
                    over = n_attn * (max(0.0, pre - window_layers * gdn) + sync)
                out["policies"].setdefault(name, {})[link] = {"overhead_ms": 1e3 * over, "throughput_ratio": step_gpu / (step_gpu + over)}
        model.append(out)
    # capacity at the longest context: GPU bytes per sequence, all-on-GPU vs tiered
    hc = cfg["hybrid"]["models"][mkey]
    n_ctx = max(h4["contexts"])
    kv = n_ctx * h4["kv_bytes_per_token"]
    share = H1.H2_BYTES[cfg["hybrid"]["h2"]["chosen"]](sz["head_dim"], cfg["definitions"]["page_size"]) / (2 * sz["head_dim"] * 2)
    pinned_pages = cfg["definitions"]["pinned"]["sink_pages"] + cfg["definitions"]["pinned"]["recent_tokens"] // cfg["definitions"]["page_size"]
    budget_pages = rc["budget_tokens"] // cfg["definitions"]["page_size"] - pinned_pages
    free = (rc["gpu_mem_gib"] - rc["reserve_gib"]) * 2 ** 30 - sz["weights"]
    st = sz["state_per_seq"]
    cap = {"context": n_ctx, "kv_bytes": kv, "state_bytes": st, "weights_bytes": sz["weights"], "free_gpu_bytes": free,
           "metadata_share": share, "all_on_gpu": {"per_seq_bytes": kv + st, "max_batch": int(free // (kv + st))}}
    for c in sorted({p["cache"] for p in pol.values()}):
        per = st + share * kv + n_attn * page * (pinned_pages + c * budget_pages)
        cap[f"tiered c{c:g}"] = {"per_seq_bytes": per, "max_batch_gpu": int(free // per),
                                 "max_batch_host": int(rc["host_kv_gib"] * 2 ** 30 // kv)}
    res = {"provenance": provenance(cfg, model=hc["name"], context_length=None, residency=rc, h4=h4),
           "n_sequences": len(rows), "mean_gen_tokens": float(np.mean([r["n_gen"] for r in rows])),
           "missing_512": miss, "policies": pol, "copy_fit": {"latency_us": 1e6 * a, "staged_gbps": bw / 1e9, "pcie_gbps": bw_pcie / 1e9},
           "step_model": model, "capacity": cap}
    print(f"{len(rows)} sequences, mean {res['mean_gen_tokens']:.0f} generated tokens; missing at {rc['budget_tokens']}: "
          f"exact {100 * miss['exact']:.2f}%, lag1 {100 * miss['lag1']:.2f}%")
    print(f"copy fit: {1e6 * a:.1f} us + bytes / {bw / 1e9:.2f} GB/s (staged); PCIe {bw_pcie / 1e9:.1f} GB/s")
    for n, p in pol.items():
        print(f"{n:16s} sync {p['sync_mean']:.2f} (p99 {p['sync_p99']:.0f})  prefetch {p['prefetch_mean']:.2f} pages/step/layer")
    for m in model:
        print(f"B {m['batch']:3d} n {m['context']:6d}: step {m['step_gpu_sparse_ms']:.2f} ms, window {m['gdn_window_ms']:.2f} ms | " +
              "  ".join(f"{n} {v['staged']['throughput_ratio']:.2f}/{v['pcie']['throughput_ratio']:.2f}" for n, v in m["policies"].items()))
    print(json.dumps(cap, indent=1))
    write_json(Path(cfg["paths"]["results_dir"]) / f"h4_residency_{mkey}.json", res)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("mode", choices=["screen", "merge"])
    p.add_argument("--limit", type=int, default=None, help="first n sequences (dry runs)")
    p.add_argument("--max-gen", type=int, default=None, help="dry run: cut generated tokens, write nothing")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = thinking(config_from_args(args))
    H1.register()
    {"screen": screen, "merge": lambda c, a: merge(c)}[args.mode](cfg, args)


if __name__ == "__main__":
    main()
