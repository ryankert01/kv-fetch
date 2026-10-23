"""Which cheap signal predicts revivals? For every revival event (a page re-entering the top-B
after >= dormancy_main dormant steps), check whether each signal had it in its own top-B k steps
earlier (event recall); also report the signal's mass recall of the attention k steps ahead.

  python scripts/analyze_signals.py configs/sweep.yaml [--split explore] [--workers 8]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligibility_matrix, topb_mask_var
from kvfetch.metrics.revival import high_matrix, reentries
from kvfetch.metrics.signals import bytes_per_page, event_recall_from_mask, mass_recall_from_masks, signal_scores
from kvfetch.provenance import provenance, write_json
from kvfetch.tracing.schema import load_meta, load_trace


def analyze(path: Path, cfg: dict) -> dict:
    tr = load_trace(path)
    meta, sg, rc, defs = tr.meta, cfg["signals"], cfg["revival"], cfg["definitions"]
    T, L, H, P = tr.mass.shape
    rnd = defs["budget_rounding"]
    elig = eligibility_matrix(meta["prompt_len"], T, P, defs["page_size"], defs["pinned"])
    q_all, ks_all = tr.extra["q"], tr.extra["key_summary"]
    scaling = q_all.shape[-1] ** -0.5
    out = {"task": meta["task"], "layers": {}}
    for li, layer in enumerate(meta["layers"]):
        m = tr.mass[:, li].astype(np.float32)
        high = high_matrix(m, elig, rc["budget_frac"], rc["exit_ratio"], rnd)
        re, gap = reentries(high.reshape(T, H * P))
        ev_t, ev_n = np.nonzero(re & (gap >= rc["dormancy_main"]))
        events = (ev_t, ev_n // P, ev_n % P)
        best, _, _ = topb_mask_var(m * elig[:, None, :], sg["budget_frac"], elig, rnd)
        res = {"n_events": int(ev_t.size)}
        for name in sg["names"]:
            res[name] = {}
            ks = sg["ks"] if name == "q_trend" else [None]
            masks = {}
            for kk in ks:  # q_trend extrapolates k steps; the others do not depend on k
                score = signal_scores(name, m=m, q=q_all[:, li], key_summary=ks_all[li], scaling=scaling, sc=sg,
                                      k=kk or 1)
                masks[kk] = topb_mask_var(score, sg["budget_frac"], elig, rnd)[0]
                del score
            for k in sg["ks"]:
                mask = masks[k if name == "q_trend" else None]
                res[name][str(k)] = {"event_recall": event_recall_from_mask(mask, events, k),
                                     "mass_recall": mass_recall_from_masks(mask, best, m, elig, k)}
        out["layers"][str(layer)] = res
    return out


def _one(job):
    return analyze(*job)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--split", default="explore", choices=["explore", "confirm"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    sg = cfg["signals"]
    paths = [pth for pth in sorted(Path(cfg["sweep"]["trace_dir"]).rglob("*.safetensors"))
             if load_meta(pth).get("split") == args.split]
    head_dim = None
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_one, [(pth, cfg) for pth in paths]))
    if paths:
        head_dim = load_trace(paths[0]).extra["q"].shape[-1]
    summary: dict = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task]
        cells = [l for r in rs for l in r["layers"].values()]
        n_ev = sum(c["n_events"] for c in cells)
        s = {"n_samples": len(rs), "n_revival_events": n_ev, "signals": {}}
        for name in sg["names"]:
            s["signals"][name] = {"bytes_per_page": bytes_per_page(name, head_dim), "k": {}}
            for k in map(str, sg["ks"]):
                w = np.array([c["n_events"] for c in cells], float)
                er = np.array([c[name][k]["event_recall"] for c in cells], float)
                ok = ~np.isnan(er) & (w > 0)
                s["signals"][name]["k"][k] = {
                    "event_recall": float((er[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else float("nan"),
                    "mass_recall": float(np.nanmean([c[name][k]["mass_recall"] for c in cells]))}
        summary[task] = s
    print(f"split={args.split} traces={len(rows)}  event recall = revived page already in the signal's top "
          f"{100 * sg['budget_frac']:.0f}% k steps before it wakes (random ~ {100 * sg['budget_frac']:.0f}%)")
    hdr = "".join(f"{n:>12s}" for n in sg["names"])
    for k in map(str, sg["ks"]):
        print(f"\nk = {k}{'':10s}{hdr}   (event recall | mass recall)")
        for task, s in summary.items():
            cells = "".join(f"  {s['signals'][n]['k'][k]['event_recall']:4.2f}|{s['signals'][n]['k'][k]['mass_recall']:4.2f}"
                            for n in sg["names"])
            print(f"{task:16s}{cells}   events={s['n_revival_events']}")
    out = args.out or f"results/signals_{args.split}.json"
    write_json(out, {"provenance": provenance(cfg, model=cfg["sweep"]["model"], context_length=cfg["sweep"]["max_context"],
                                              split=args.split, n_traces=len(rows)),
                     "summary": summary, "per_sample": rows})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
