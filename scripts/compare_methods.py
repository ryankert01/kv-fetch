"""Compare KV-cache methods at equal GPU budget on long reasoning traces (offline replay).

  python scripts/compare_methods.py configs/reasoning.yaml [--split explore] [--workers 8]

Eviction methods -> attention coverage (the rest is lost for good). Keep-all (Quest, true top-B)
-> coverage without memory saving. Offload methods attend to the Quest selection and differ in
how many selected pages must be fetched on demand (stalls) given an L-step prefetch lookahead.
"""
import argparse
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.metrics.signals import KEY_MEAN
from kvfetch.provenance import provenance, write_json
from kvfetch.sim import (coverage, evict_context, existing_pages, mask_from_ranks, prefetch_query, quest_bound,
                         rank_desc, simulate_evict, simulate_offload)
from kvfetch.tracing.schema import load_meta, load_trace


def analyze(path: Path, cfg: dict) -> dict:
    cc, defs = cfg["compare"], cfg["definitions"]
    ps = defs["page_size"]
    tr = load_trace(path)
    meta = tr.meta
    if meta.get("head_granularity") != "kv_mean":
        raise ValueError("compare_methods expects KV-group traces (trace.head_reduce: kv_mean)")
    T, L, S, P = tr.mass.shape
    elig = eligibility_matrix(meta["prompt_len"], T, P, ps, defs["pinned"])
    exist = existing_pages(meta["prompt_len"], T, P, ps)
    pinned = exist & ~elig
    n_pinned = pinned.sum(-1)
    out = {"task": meta["task"], "n_steps": T, "score": meta.get("score"), "layers": {}}
    for li, layer in enumerate(meta["layers"]):
        m = tr.mass[:, li].astype(np.float32)
        q = tr.extra["q"][:, li]
        ks = tr.extra["key_summary"][li]
        scaling = q.shape[-1] ** -0.5
        init = tr.obs_mass[:, li].astype(np.float32).sum(0)
        ctx = evict_context(m, init, ks[KEY_MEAN].astype(np.float32), exist, cc)
        quest_r = rank_desc(quest_bound(q, ks, scaling), elig)
        true_r = rank_desc(m, elig)
        pf_r = {}                                          # (prefetcher, L) -> ranks, computed once
        for Lk in cc["lookahead_steps"]:
            for pf_name in cc["prefetch_methods"]:
                qp = None if pf_name == "oasis_ub" else prefetch_query(pf_name, q, Lk, cc)
                pf_r[(pf_name, Lk)] = None if qp is None else rank_desc(quest_bound(qp, ks, scaling), elig)
        res: dict = {}
        for btok in cc["budgets_tokens"]:
            B = btok // ps
            bsel = np.maximum(0, B - n_pinned)
            r = {"evict": {}, "keep_all": {}, "offload": {}}
            for name in cc["evict_methods"]:
                r["evict"][name] = simulate_evict(name, m=m, eligible=elig, exist=exist, B=B, ctx=ctx)
            sel = mask_from_ranks(quest_r, bsel, elig)
            r["keep_all"]["quest"] = coverage(sel | pinned[:, None, :], m)
            r["keep_all"]["true_topB"] = coverage(mask_from_ranks(true_r, bsel, elig) | pinned[:, None, :], m)
            cap = int(cc["gpu_capacity_mult"] * max(1, int(bsel.max())))
            for Lk in cc["lookahead_steps"]:
                for pf_name in cc["prefetch_methods"]:
                    rk = pf_r[(pf_name, Lk)]
                    pf = sel if pf_name == "oasis_ub" else (None if rk is None else mask_from_ranks(rk, bsel, elig))
                    r["offload"][f"{pf_name}@L{Lk}"] = simulate_offload(m=m, sel=sel, prefetch=pf, eligible=elig,
                                                                        capacity=cap, base=pinned[:, None, :])
            res[str(btok)] = r
        out["layers"][str(layer)] = res
    return out


def _one(job):
    path, cfg = job
    return analyze(path, cfg)


def summarize(rows: list[dict], cfg: dict) -> dict:
    cc = cfg["compare"]
    out = {}
    for task in sorted({r["task"] for r in rows}) + ["ALL"]:
        rs = [r for r in rows if task == "ALL" or r["task"] == task]
        cells = [(l, r) for r in rs for l in r["layers"].values()]
        s = {"n_samples": len(rs), "mean_steps": float(np.mean([r["n_steps"] for r in rs]))}
        for btok in map(str, cc["budgets_tokens"]):
            b = {"evict": {}, "keep_all": {}, "offload": {}}
            for fam in ("evict", "keep_all"):
                for name in cells[0][0][btok][fam]:
                    b[fam][name] = float(np.nanmean([l[btok][fam][name] for l, _ in cells]))
            for name in cells[0][0][btok]["offload"]:
                b["offload"][name] = {k: float(np.nanmean([l[btok]["offload"][name][k] for l, _ in cells]))
                                      for k in cells[0][0][btok]["offload"][name]}
            s[btok] = b
        out[task] = s
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="explore", choices=["explore", "confirm"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    cc = cfg["compare"]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(cc["blas_threads"])        # read by BLAS as each spawned worker imports numpy
    paths = [pth for pth in sorted(Path(cfg["sweep"]["trace_dir"]).rglob("*.safetensors"))
             if load_meta(pth).get("split") == args.split
             and (args.tasks is None or load_meta(pth)["task"] in args.tasks)]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_one, [(pth, cfg) for pth in paths]))
    summary = summarize(rows, cfg)
    a = summary["ALL"]
    print(f"split={args.split} traces={len(rows)} mean steps={a['mean_steps']:.0f}")
    print("\nATTENTION COVERAGE (eviction = rest lost; keep-all = no memory saving)")
    names = cc["evict_methods"] + ["quest", "true_topB"]
    print(f"{'budget':>7s} " + " ".join(f"{n:>9s}" for n in names))
    for btok in map(str, cc["budgets_tokens"]):
        vals = [a[btok]["evict"][n] for n in cc["evict_methods"]] + [a[btok]["keep_all"][n] for n in ("quest", "true_topB")]
        print(f"{btok:>7s} " + " ".join(f"{v:9.3f}" for v in vals))
    print("\nOFFLOAD: stalled pages per step per KV head (lower is better) | no-stall coverage of the selection")
    keys = list(a[str(cc["budgets_tokens"][0])]["offload"])
    print(f"{'budget':>7s} " + " ".join(f"{k:>18s}" for k in keys))
    for btok in map(str, cc["budgets_tokens"]):
        o = a[btok]["offload"]
        print(f"{btok:>7s} " + " ".join(f"{o[k]['stall_pages_per_step']:8.3f}|{o[k]['nostall_coverage_of_selected']:.3f}" .rjust(18) for k in keys))
    out = args.out or f"results/compare_{args.split}.json"
    write_json(out, {"provenance": provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"],
                                              split=args.split, n_traces=len(rows)), "summary": summary, "per_sample": rows})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
