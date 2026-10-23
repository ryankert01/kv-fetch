"""Revival analysis of the sweep traces, per dataset. Reads only one split (default: the
exploration half); the confirm half stays untouched for the decision point.

  python scripts/analyze_revival.py configs/sweep.yaml [--split explore] [--workers 8]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix
from kvfetch.metrics.revival import GAP_BINS, eviction_regret, q_stats, revival_stats
from kvfetch.provenance import provenance, write_json
from kvfetch.tracing.schema import load_meta, load_trace


def kv_mean(x: np.ndarray, num_kv_heads: int) -> np.ndarray:
    """[..., H, P] -> [..., KVH, P]: mean over each GQA group (query head h reads KV head h // G)."""
    *lead, H, P = x.shape
    return x.reshape(*lead, num_kv_heads, H // num_kv_heads, P).mean(-2)


def analyze(path: Path, cfg: dict) -> dict:
    tr = load_trace(path)
    meta = tr.meta
    rc, defs = cfg["revival"], cfg["definitions"]
    T, L, H, P = tr.mass.shape
    elig = eligibility_matrix(meta["prompt_len"], T, P, defs["page_size"], defs["pinned"])
    out = {"task": meta["task"], "index": int(meta["sample_id"].rsplit("-", 1)[1]), "n_steps": T,
           "prompt_len": meta["prompt_len"], "fed_share": float(tr.extra["token_kind"].mean()),
           "eligible_pages_end": int(elig[-1].sum()), "layers": {}}
    for li, layer in enumerate(meta["layers"]):
        mq = tr.mass[:, li].astype(np.float32)                       # [T, H, P]
        mk = kv_mean(mq, meta["num_kv_heads"])                         # [T, KVH, P]
        rs = {"query": revival_stats(mq, elig, rc, defs["budget_rounding"]),
              "kv_mean": revival_stats(mk, elig, rc, defs["budget_rounding"])}
        init = kv_mean(tr.obs_mass[:, li].astype(np.float32).sum(0), meta["num_kv_heads"])  # prefill-window score
        regret = {str(f): float(np.nanmean(eviction_regret(mk, elig, init, f, defs["budget_rounding"])))
                  for f in rc["eviction_budget_fracs"]}
        qs = q_stats(tr.extra["q"][:, li], rc["q_lags"], rc["q_pca_var"])
        out["layers"][str(layer)] = {"revival": rs, "eviction_regret": regret, "q": qs}
    return out


def _one(job):
    return analyze(*job)


def summarize(rows: list[dict], cfg: dict) -> dict:
    rc = cfg["revival"]
    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    out = {}
    for task, rs in sorted(by_task.items()):
        layers = list(rs[0]["layers"])
        s: dict = {"n_samples": len(rs), "mean_steps": float(np.mean([r["n_steps"] for r in rs])),
                   "fed_share": float(np.mean([r["fed_share"] for r in rs]))}
        for view in ("query", "kv_mean"):
            v = {}
            for D in rc["dormancy_steps"]:
                key = f"D{D}"
                vals = [r["layers"][l]["revival"][view][key] for r in rs for l in layers]
                v[key] = {m: float(np.nanmean([x[m] for x in vals]))
                          for m in ("revival_rate", "revived_mass_share", "revivals_per_1k_steps_per_series")}
                v[key]["by_layer"] = {l: float(np.nanmean([r["layers"][l]["revival"][view][key]["revival_rate"]
                                                           for r in rs])) for l in layers}
            gaps = np.concatenate([r["layers"][l]["revival"][view]["gaps"] for r in rs for l in layers])
            v["gap_hist"] = {"bins": GAP_BINS[:-1], "counts": np.histogram(gaps, bins=GAP_BINS)[0].tolist()}
            v["gap_median"] = float(np.median(gaps)) if gaps.size else float("nan")
            s[view] = v
        s["eviction_regret"] = {f: float(np.mean([r["layers"][l]["eviction_regret"][f] for r in rs for l in layers]))
                                for f in rs[0]["layers"][layers[0]]["eviction_regret"]}
        s["q_autocorr"] = {k: float(np.mean([r["layers"][l]["q"]["autocorr"][k] for r in rs for l in layers]))
                           for k in rs[0]["layers"][layers[0]]["q"]["autocorr"]}
        s["q_eff_dim_median"] = float(np.median([r["layers"][l]["q"]["eff_dim_median"] for r in rs for l in layers]))
        out[task] = s
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="explore", choices=["explore", "confirm"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    paths = [pth for pth in sorted(Path(cfg["sweep"]["trace_dir"]).rglob("*.safetensors"))
             if load_meta(pth).get("split") == args.split]
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_one, [(pth, cfg) for pth in paths]))
    summary = summarize(rows, cfg)
    D = cfg["revival"]["dormancy_main"]
    print(f"split={args.split}  traces={len(rows)}  main dormancy D={D} steps")
    print(f"{'task':16s} {'n':>3s} {'steps':>6s} {'revive%':>8s} {'revived-attn%':>14s} {'gap med':>8s} "
          f"{'H2O regret@5%':>14s} {'@10%':>6s} {'q cos@1':>8s} {'q cos@16':>9s} {'q dim':>6s}")
    for task, s in summary.items():
        k = s["kv_mean"][f"D{D}"]
        print(f"{task:16s} {s['n_samples']:3d} {s['mean_steps']:6.0f} {100 * k['revival_rate']:7.1f}% "
              f"{100 * k['revived_mass_share']:13.1f}% {s['kv_mean']['gap_median']:8.0f} "
              f"{100 * s['eviction_regret']['0.05']:13.1f}% {100 * s['eviction_regret']['0.1']:5.1f}% "
              f"{s['q_autocorr'][1]:8.3f} {s['q_autocorr'][16]:9.3f} {s['q_eff_dim_median']:6.0f}")
    for r in rows:  # drop raw gap arrays before writing JSON
        for l in r["layers"].values():
            for v in l["revival"].values():
                v.pop("gaps", None)
    out = args.out or f"results/revival_{args.split}.json"
    write_json(out, {"provenance": provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"],
                                              split=args.split, n_traces=len(rows)),
                     "summary": summary, "per_sample": rows})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
