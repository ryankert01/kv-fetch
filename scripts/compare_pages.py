"""E2b (EXPERIMENTS.md): every selector at vLLM block granularity (16-token pages), equal page budgets.

  CUDA_VISIBLE_DEVICES=1 python scripts/compare_pages.py configs/reasoning.yaml [--split explore] [--shard 0/1]

Like E2 (scripts/compare_baselines.py), ground truth is the exact attention recomputed from the
traced queries and the replayed keys. Token-level methods are adapted to pages: InfiniGen's
speculated token attention (partial key channels, current query: its best case) is summed per page.
Each method runs in three placement modes:
  per_head  one page set per KV head (needs a head-aware kernel; a block holds all heads, so
            whole-block transfers fetch the union over heads: reported as blocks per budget page),
  vote/sum  one page set per layer for all KV heads (a sub-block-table: vLLM's paged kernels and
            block transfers unchanged), by nomination counts or by summed attention shares.
Per method: missing attention, revived attention captured, and GPU metadata bytes per token per KV
head. One JSON per trace under e2b.shard_dir; --merge writes results/pages_<split>.json, prints the
table and evaluates the E2b gate.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from safetensors.torch import load_file

from kvfetch.baselines import (arkvale_scores, attention_by_group, landmark_scores, partial_channel_scores,
                               pick_channels, quest_scores, unit_summaries)
from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.provenance import provenance, write_json
from kvfetch.selectors import revived_weight, run_selector, scoring_bytes, static_select, upper_bound
from kvfetch.sim import coverage, existing_pages, mask_from_ranks, rank_desc
from kvfetch.tracing.schema import load_meta, load_trace

MODES = ["per_head", "vote", "sum"]


def fit_pages(score: torch.Tensor, P: int, fill: float = float("-inf")) -> torch.Tensor:
    """Scores over pages with at least one replayed key; pad to the trace's P."""
    return torch.nn.functional.pad(score, (0, max(0, P - score.shape[-1])), value=fill)[..., :P]


def page_scores(tr, li: int, keys: torch.Tensor, cfg: dict, device: str) -> dict[str, np.ndarray]:
    """Exact page mass and every method's page scores for one traced layer, each [T, S, P] (fp32)."""
    e2b, ps = cfg["e2b"], cfg["definitions"]["page_size"]
    T, _, S, P = tr.mass.shape
    pl = tr.meta["prompt_len"]
    k = keys.to(device).float()
    q = torch.from_numpy(tr.extra["q"][:, li].astype(np.float32)).to(device)
    N, (H, d) = k.shape[1], q.shape[1:]
    scaling = d ** -0.5
    summ = unit_summaries(k, ps)
    n_keys = torch.arange(T, device=device) + pl
    chans = {r: pick_channels(q[0], k[:, :pl], max(1, int(round(r * d)))) for r in e2b["infinigen_ratios"]}
    names = ["m", "quest", "arkvale", "mean"] + [f"infinigen_r{r:g}" for r in e2b["infinigen_ratios"]]
    out = {n: np.zeros((T, S, P), np.float32) for n in names}
    pad = P * ps - N
    to_pages = lambda x: torch.nn.functional.pad(x, (0, pad)).reshape(*x.shape[:-1], P, ps).sum(-1)
    for a0 in range(0, T, e2b["step_chunk"]):
        sl = slice(a0, min(T, a0 + e2b["step_chunk"]))
        out["m"][sl] = to_pages(attention_by_group(q[sl], k, n_keys[sl], scaling).mean(2)).cpu().numpy()
        out["quest"][sl] = fit_pages(quest_scores(q[sl], summ, scaling), P).cpu().numpy()
        out["arkvale"][sl] = fit_pages(arkvale_scores(q[sl], summ, scaling), P).cpu().numpy()
        out["mean"][sl] = fit_pages(landmark_scores(q[sl], summ, scaling), P).cpu().numpy()
        seen = torch.arange(N, device=device)[None, :] < n_keys[sl, None]          # sink + recent + eligible
        for r, ch in chans.items():
            spec = partial_channel_scores(q[sl], k, ch, scaling).masked_fill(~seen[:, None, None, :], float("-inf"))
            out[f"infinigen_r{r:g}"][sl] = to_pages(torch.softmax(spec, -1).mean(2)).cpu().numpy()
    out["min_cos"] = np.broadcast_to(fit_pages(summ["min_cos"], P, 1.0).cpu().numpy()[None], (T, S, P))
    return out


def shadowkv_select(sc: dict[str, np.ndarray], elig: np.ndarray, b: np.ndarray, mode: str, share: float) -> np.ndarray:
    """ShadowKV at page granularity: a share of the budget for outlier pages (lowest key cosine to
    the page mean, shared: lowest over heads), the rest by the page-mean landmark."""
    n_out = np.round(share * b).astype(int)
    if mode == "per_head":
        outl = mask_from_ranks(rank_desc(-sc["min_cos"], elig), n_out, elig)
        return mask_from_ranks(rank_desc(np.where(outl, np.inf, sc["mean"]), elig), b, elig)
    outl = mask_from_ranks(rank_desc(-sc["min_cos"].min(1, keepdims=True), elig), n_out, elig)[:, 0]   # [T, P]
    return outl[:, None, :] | static_select(sc["mean"], elig & ~outl, b - n_out, mode)


def metadata_bytes(cfg: dict, d: int) -> dict[str, float]:
    """GPU bytes per token per KV head each method keeps to score pages (fp16 summaries)."""
    sb, hb, ps = cfg["selectors"]["summary_bytes"], cfg["selectors"]["history_bytes"], cfg["definitions"]["page_size"]
    out = {"quest": 2 * d * sb / ps, "arkvale": (d + 1) * sb / ps, "shadowkv_page": d * sb / ps}
    for r in cfg["e2b"]["infinigen_ratios"]:
        out[f"infinigen_r{r:g}"] = round(r * d) * sb
        out[f"ours:infinigen_r{r:g}+hist"] = round(r * d) * sb + hb / ps
    for name in cfg["e2b"]["ours"]:
        out[f"ours:{name}"] = scoring_bytes(cfg["selectors"]["candidates"][name], d, sb, hb) / ps
    return out


def analyze(path: Path, cfg: dict, device: str) -> dict:
    e2b, defs, cands = cfg["e2b"], cfg["definitions"], cfg["selectors"]["candidates"]
    ps = defs["page_size"]
    tr = load_trace(path)
    meta = tr.meta
    keys_all = load_file(str(Path(cfg["e2"]["keys_dir"]) / path.parent.name / path.name))["keys"]
    T, _, S, P = tr.mass.shape
    pl = meta["prompt_len"]
    elig = eligibility_matrix(pl, T, P, ps, defs["pinned"])
    exist = existing_pages(pl, T, P, ps)
    pinned = (exist & ~elig)[:, None, :]
    out = {"task": meta["task"], "sample_id": meta["sample_id"], "n_steps": T, "score": meta.get("score"),
           "head_dim": int(tr.extra["q"].shape[-1]), "layers": {}}
    for li, layer in enumerate(meta["layers"]):
        with torch.no_grad():
            sc = page_scores(tr, li, keys_all[li], cfg, device)
        torch.cuda.empty_cache()
        m = sc["m"]
        init = tr.obs_mass[:, li].astype(np.float32).mean(0)
        rev = revived_weight(m, elig, cfg["revival"])
        rev_tot = float(rev.sum())
        res = {}
        for btok in e2b["budgets_tokens"]:
            b = np.maximum(0, btok // ps - pinned[:, 0].sum(-1))
            r = {}

            def record(name: str, sel: np.ndarray, per_head: bool) -> None:
                att = sel | pinned
                v = {"missing": 1.0 - coverage(att, m),
                     "revived_captured": float((rev * att).sum() / rev_tot) if rev_tot > 0 else float("nan")}
                if per_head:
                    v["blocks_per_budget_page"] = float((sel.any(1).sum(-1) / np.maximum(b, 1)).mean())
                r[name] = v

            record("true_page|per_head", upper_bound("true_topB", m=m, eligible=elig, b=b), True)
            record("true_page|sum", upper_bound("shared_topB", m=m, eligible=elig, b=b), False)
            for mode in MODES:
                for name, kind in [("quest", "logit"), ("arkvale", "logit")] + \
                                  [(f"infinigen_r{x:g}", "mass") for x in e2b["infinigen_ratios"]]:
                    record(f"{name}|{mode}", static_select(sc[name], elig, b, mode, kind), mode == "per_head")
                record(f"shadowkv_page|{mode}", shadowkv_select(sc, elig, b, mode, e2b["shadowkv_outlier_share"]),
                       mode == "per_head")
                shared = {"per_head": False}.get(mode, mode)
                for name in e2b["ours"]:
                    record(f"ours:{name}|{mode}", run_selector({**cands[name], "shared": shared}, m=m, qb=sc["quest"],
                                                               init=init, eligible=elig, exist=exist, b=b, qm=sc["mean"]),
                           mode == "per_head")
                for x in e2b["infinigen_ratios"]:
                    spec = {"fuse": "sum", "history_weight": e2b["plugin_history_weight"], "history": "last", "shared": shared}
                    log_spec = np.log(np.maximum(sc[f"infinigen_r{x:g}"], np.finfo(np.float32).tiny))
                    record(f"ours:infinigen_r{x:g}+hist|{mode}",
                           run_selector(spec, m=m, qb=log_spec, init=init, eligible=elig, exist=exist, b=b),
                           mode == "per_head")
            res[str(btok)] = r
        out["layers"][str(layer)] = res
        print(f"[e2b] {path.parent.name}/{path.name} layer {layer} done", flush=True)
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    groups = {t: [r for r in rows if r["task"] == t] for t in sorted({r["task"] for r in rows})}
    groups["ALL"] = rows
    for task, rs in groups.items():
        s = {"n_samples": len(rs)}
        for btok in rs[0]["layers"][next(iter(rs[0]["layers"]))]:
            vals: dict[str, list] = {}
            for r in rs:
                for l in r["layers"].values():
                    for n, v in l[btok].items():
                        vals.setdefault(n, []).append(v)
            s[btok] = {n: {k: float(np.nanmean([x[k] for x in v])) for k in v[0]} for n, v in vals.items()}
        out[task] = s
    return out


def best_mode(table: dict, method: str, modes: list[str]) -> tuple[str, float]:
    """The placement mode (among ``modes``) in which a method misses least: every method, ours and
    the baselines alike, is credited with its better shared rule."""
    got = [(m, table[f"{method}|{m}"]["missing"]) for m in modes if f"{method}|{m}" in table]
    return min(got, key=lambda x: x[1])


def gate(summary: dict, cfg: dict, d: int) -> dict:
    g, mb = cfg["e2b"]["gate"], metadata_bytes(cfg, d)
    ours = [n for n in mb if n.startswith("ours:") and mb[n] <= g["ours_max_bytes"]]
    out = {"spec": g, "budgets": {}}
    for btok in map(str, g["budgets_tokens"]):
        t = summary["ALL"][btok]
        best_ours = min(((n, *best_mode(t, n, g["modes"])) for n in ours), key=lambda x: x[2])
        base = min(((n, *best_mode(t, n, g["modes"])) for n in g["page_baselines"]), key=lambda x: x[2])
        rival = (g["equal_bytes_rival"], *best_mode(t, g["equal_bytes_rival"], g["modes"]))
        red = 1.0 - best_ours[2] / base[2]
        out["budgets"][btok] = {"ours": best_ours, "best_page_baseline": base, "equal_bytes_rival": rival,
                                "reduction_vs_page_baseline": red,
                                "passed": bool(red >= g["reduction_min"] and best_ours[2] <= rival[2])}
    out["passed"] = all(v["passed"] for v in out["budgets"].values())
    return out


def _run_shard(args_tuple) -> None:
    cfg, split, shard, device = args_tuple
    torch.set_num_threads(cfg["selectors"]["blas_threads"])
    i, n = map(int, shard.split("/"))
    paths = [pth for pth in sorted(Path(cfg["sweep"]["trace_dir"]).rglob("*.safetensors"))
             if load_meta(pth).get("split") == split][i::n]
    prov = provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"], split=split)
    for pth in paths:
        out = Path(cfg["e2b"]["shard_dir"]) / split / pth.parent.name / f"{pth.stem}.json"
        if out.exists():
            continue
        if not (Path(cfg["e2"]["keys_dir"]) / pth.parent.name / pth.name).exists():
            print(f"[e2b] {pth.parent.name}/{pth.name}: keys not dumped yet, skipped", flush=True)
            continue
        write_json(out, {"provenance": prov, "result": analyze(pth, cfg, device)})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="explore", choices=["explore", "confirm"])
    p.add_argument("--shard", default="0/1", help="i/n: this run takes every n-th trace from i")
    p.add_argument("--workers", type=int, default=1, help="processes on this GPU, each a sub-shard")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge", action="store_true", help="combine shard outputs, print the table and the gate")
    args = p.parse_args()
    cfg = config_from_args(args)
    e2b = cfg["e2b"]
    if args.merge:
        rows = [json.load(open(f))["result"] for f in sorted((Path(e2b["shard_dir"]) / args.split).rglob("*.json"))]
        summary = summarize(rows)
        d = rows[0]["head_dim"]
        mb = metadata_bytes(cfg, d)
        a = summary["ALL"]
        print(f"split={args.split} traces={len(rows)}   missing attention %, (xN = whole blocks fetched per budget page)")
        bud = list(map(str, e2b["budgets_tokens"]))
        methods = sorted({n.split("|")[0] for n in a[bud[0]]}, key=lambda n: min(v["missing"] for k, v in a[bud[1]].items() if k.split("|")[0] == n))
        print(f"{'method':>30s} {'B/tok':>6s} " + " ".join(f"{b + '|' + mo:>17s}" for b in bud for mo in MODES))
        for n in methods:
            cells = []
            for b in bud:
                for mo in MODES:
                    v = a[b].get(f"{n}|{mo}")
                    cells.append("-" if v is None else f"{100 * v['missing']:6.2f}" + (f" x{v['blocks_per_budget_page']:4.1f}" if "blocks_per_budget_page" in v else ""))
            print(f"{n:>30s} {mb.get(n, float('nan')):6.2f} " + " ".join(f"{c:>17s}" for c in cells))
        gt = gate(summary, cfg, d)
        for b, v in gt["budgets"].items():
            print(f"gate @{b}: ours {v['ours']}, best page baseline {v['best_page_baseline']}, "
                  f"rival {v['equal_bytes_rival']}, reduction {100 * v['reduction_vs_page_baseline']:.1f}% -> {'pass' if v['passed'] else 'FAIL'}")
        write_json(f"results/pages_{args.split}.json",
                   {"provenance": provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"],
                                             split=args.split, n_traces=len(rows)),
                    "metadata_bytes_per_token": mb, "summary": summary, "gate": gt, "per_sample": rows})
        print(f"\nwrote results/pages_{args.split}.json")
        return
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # read by spawned workers' BLAS
        os.environ[var] = str(cfg["selectors"]["blas_threads"])
    i, n = map(int, args.shard.split("/"))
    subs = [(cfg, args.split, f"{i + j * n}/{n * args.workers}", args.device) for j in range(args.workers)]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_run_shard, subs))


if __name__ == "__main__":
    main()
