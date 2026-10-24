"""E1 (EXPERIMENTS.md): replay training-free page selectors on long reasoning traces.

  python scripts/compare_selectors.py configs/reasoning.yaml [--split explore] [--workers 8]

Per (trace, layer, budget): missing attention (1 - coverage of selection + pinned pages) and the
share of revived attention the selection captures. Candidates see only the current query (Quest
bound) and attention they observed; the upper bounds read true attention.
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
from kvfetch.provenance import provenance, write_json
from kvfetch.selectors import mean_logit, revived_weight, run_selector, scoring_bytes, upper_bound
from kvfetch.sim import coverage, existing_pages, quest_bound
from kvfetch.tracing.schema import load_meta, load_trace


def analyze(path: Path, cfg: dict) -> dict:
    sc, defs = cfg["selectors"], cfg["definitions"]
    ps = defs["page_size"]
    tr = load_trace(path)
    meta = tr.meta
    if meta.get("head_granularity") != "kv_mean":
        raise ValueError("compare_selectors expects KV-group traces (trace.head_reduce: kv_mean)")
    T, L, S, P = tr.mass.shape
    elig = eligibility_matrix(meta["prompt_len"], T, P, ps, defs["pinned"])
    exist = existing_pages(meta["prompt_len"], T, P, ps)
    pinned = (exist & ~elig)[:, None, :]
    n_pinned = pinned[:, 0].sum(-1)
    out = {"task": meta["task"], "sample_id": meta["sample_id"], "n_steps": T, "score": meta.get("score"),
           "layers": {}}
    for li, layer in enumerate(meta["layers"]):
        m = tr.mass[:, li].astype(np.float32)
        q = tr.extra["q"][:, li]
        ks, scaling = tr.extra["key_summary"][li], q.shape[-1] ** -0.5
        qb = quest_bound(q, ks, scaling)
        qm = mean_logit(q, ks, scaling) if any(c.get("mean_mix") for c in sc["candidates"].values()) else None
        init = tr.obs_mass[:, li].astype(np.float32).mean(0)
        rev = revived_weight(m, elig, cfg["revival"])
        rev_tot = float(rev.sum())
        res: dict = {}
        for btok in sc["budgets_tokens"]:
            b = np.maximum(0, btok // ps - n_pinned)
            r: dict = {}
            names = list(sc["candidates"]) + list(sc["upper_bounds"])
            for name in names:
                if name in sc["candidates"]:
                    sel = run_selector(sc["candidates"][name], m=m, qb=qb, init=init, eligible=elig, exist=exist, b=b,
                                       qm=qm)
                else:
                    sel = upper_bound(name, m=m, eligible=elig, b=b)
                att = sel | pinned
                r[name] = {"missing": 1.0 - coverage(att, m),
                           "revived_captured": float((rev * att).sum()) / rev_tot if rev_tot > 0 else float("nan")}
            res[str(btok)] = r
        out["layers"][str(layer)] = res
    return out


def _one(job):
    path, cfg = job
    return analyze(path, cfg)


def summarize(rows: list[dict], cfg: dict) -> dict:
    sc = cfg["selectors"]
    names = list(sc["candidates"]) + list(sc["upper_bounds"])
    out = {}
    groups = {t: [r for r in rows if r["task"] == t] for t in sorted({r["task"] for r in rows})}
    groups["ALL"] = rows
    for task, rs in groups.items():
        cells = [l for r in rs for l in r["layers"].values()]
        s = {"n_samples": len(rs), "mean_steps": float(np.mean([r["n_steps"] for r in rs]))}
        for btok in map(str, sc["budgets_tokens"]):
            s[btok] = {n: {k: float(np.nanmean([c[btok][n][k] for c in cells])) for k in ("missing", "revived_captured")}
                       for n in names}
        out[task] = s
    per_layer = {}
    for layer in rows[0]["layers"]:
        per_layer[layer] = {btok: {n: float(np.mean([r["layers"][layer][btok][n]["missing"] for r in rows])) for n in names}
                            for btok in map(str, sc["budgets_tokens"])}
    out["ALL"]["per_layer_missing"] = per_layer
    return out


def gate(summary: dict, cfg: dict, head_dim: int) -> dict:
    sc, g = cfg["selectors"], cfg["selectors"]["gate"]
    a = summary["ALL"][str(g["budget_tokens"])]
    quest, top = a["quest"]["missing"], a["true_topB"]["missing"]
    target = quest - g["gap_closed_min"] * (quest - top)
    cost = {n: scoring_bytes(spec, head_dim, sc["summary_bytes"], sc["history_bytes"]) for n, spec in sc["candidates"].items()}
    ok = [n for n in sc["candidates"] if a[n]["missing"] <= target and cost[n] <= g["cost_ratio_max"] * cost["quest"]]
    best = min(sc["candidates"], key=lambda n: a[n]["missing"])
    return {"budget_tokens": g["budget_tokens"], "quest_missing": quest, "true_topB_missing": top,
            "target_missing": target, "scoring_bytes": cost, "passing": ok, "best_candidate": best,
            "best_gap_closed": (quest - a[best]["missing"]) / (quest - top) if quest > top else float("nan"),
            "passed": bool(ok)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="explore", choices=["explore", "confirm"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    sc = cfg["selectors"]
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(sc["blas_threads"])        # read by BLAS as each spawned worker imports numpy
    paths = [pth for pth in sorted(Path(cfg["sweep"]["trace_dir"]).rglob("*.safetensors"))
             if load_meta(pth).get("split") == args.split
             and (args.tasks is None or load_meta(pth)["task"] in args.tasks)]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(_one, [(pth, cfg) for pth in paths]))
    summary = summarize(rows, cfg)
    head_dim = load_trace(paths[0]).extra["q"].shape[-1]
    g = gate(summary, cfg, head_dim)
    a = summary["ALL"]
    names = list(sc["candidates"]) + list(sc["upper_bounds"])
    print(f"split={args.split} traces={len(rows)} mean steps={a['mean_steps']:.0f}")
    print("\nMISSING ATTENTION % (lower is better) | revived attention captured % at each budget")
    print(f"{'selector':>16s} " + " ".join(f"{b:>14s}" for b in map(str, sc["budgets_tokens"])))
    for n in names:
        print(f"{n:>16s} " + " ".join(f"{100 * a[b][n]['missing']:6.2f} | {100 * a[b][n]['revived_captured']:5.1f}"
                                      for b in map(str, sc["budgets_tokens"])))
    print(f"\nGATE @ {g['budget_tokens']} tokens: Quest {100 * g['quest_missing']:.2f}%, top-B {100 * g['true_topB_missing']:.2f}%, "
          f"target <= {100 * g['target_missing']:.2f}%; best {g['best_candidate']} closes {100 * g['best_gap_closed']:.0f}% "
          f"of the gap; passing: {g['passing'] or 'none'}")
    out = args.out or f"results/selectors_{args.split}.json"
    write_json(out, {"provenance": provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"],
                                              split=args.split, n_traces=len(rows)),
                     "summary": summary, "gate": g, "per_sample": rows})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
