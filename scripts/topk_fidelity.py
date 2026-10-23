"""Open item §9.3: would top-K + residual traces change our metrics?

Reconstructs each dense small-eval trace from its top-K pages (residual dropped), reruns
the small-eval analysis, and reports only the DIFFERENCE vs. the dense trace, so the
metric levels themselves are not exposed before the Phase 1 decision values are fixed.
K is reported both absolutely and as a fraction of pages: at 32K / page 16 (2064 pages),
K=128 has the same page fraction as K=512 at 128K (8320 pages).

  python scripts/topk_fidelity.py configs/small_eval.yaml --trace-dir traces/small_eval/<tag> --K 128 256 512
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from kvfetch.analysis import analyze_trace
from kvfetch.config import add_config_args, config_from_args
from kvfetch.metrics.importance import eligible_pages
from kvfetch.provenance import provenance, write_json
from kvfetch.tracing.schema import Trace, load_meta, load_trace, to_topk


def _recon(tr: Trace, k: int) -> Trace:
    t = to_topk(tr.mass, k)
    sparse = Trace(meta=tr.meta, obs_mass=tr.obs_mass, tokens=tr.tokens, prompt_ids=tr.prompt_ids,
                   topk_idx=t["topk_idx"], topk_val=t["topk_val"], residual=t["residual"])
    return Trace(meta=tr.meta, obs_mass=tr.obs_mass, tokens=tr.tokens, prompt_ids=tr.prompt_ids, mass=sparse.dense())


def _one(job) -> dict:
    path, cfg, ks = job
    acfg = cfg["small_eval"]["analysis"]
    tr = load_trace(path)
    elig = eligible_pages(tr.meta["prompt_len"], tr.meta["n_pages"], cfg["definitions"]["page_size"],
                          cfg["definitions"]["pinned"])
    base = analyze_trace(tr, cfg, acfg)
    out = {"task": tr.meta["task"], "n_pages": tr.meta["n_pages"], "per_k": {}}
    m = tr.mass.astype(np.float32)
    for k in ks:
        rt = _recon(tr, k)
        a = analyze_trace(rt, cfg, acfg)
        kept_elig = (rt.mass.astype(np.float32)[..., elig].sum(-1) / np.maximum(m[..., elig].sum(-1), 1e-12)).mean()
        d = {"eligible_mass_kept": float(kept_elig)}
        for view in cfg["definitions"]["head_views"]:
            b, s = base["summary"][view], a["summary"][view]
            dv = {key: s[key] - b[key] for key in b if key.startswith("recall_")}
            for name in acfg["predictors"]:
                dv[f"lead_median_{name}"] = s[f"lead_{name}"]["median"] - b[f"lead_{name}"]["median"]
            tb, ts = base["pool"][view]["traj"], a["pool"][view]["traj"]
            dv["class_agreement"] = float((tb.cls == ts.cls).mean())
            dv["max_abs_class_share_delta"] = float(max(abs(s["trajectory"]["all"][c] - b["trajectory"]["all"][c])
                                                        for c in b["trajectory"]["all"]))
            d[view] = dv
        out["per_k"][k] = d
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--K", type=int, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="results/topk_fidelity.json")
    args = p.parse_args()
    cfg = config_from_args(args)
    paths = sorted(Path(args.trace_dir).rglob("*.safetensors"))
    with ProcessPoolExecutor(args.workers) as ex:
        per = list(ex.map(_one, [(pth, cfg, args.K) for pth in paths]))

    summary = {}
    for k in args.K:
        rows = [r["per_k"][k] for r in per]
        s = {"page_fraction": float(np.mean([k / r["n_pages"] for r in per])),
             "eligible_mass_kept_mean": float(np.mean([r["eligible_mass_kept"] for r in rows])),
             "eligible_mass_kept_min": float(np.min([r["eligible_mass_kept"] for r in rows]))}
        for view in cfg["definitions"]["head_views"]:
            keys = rows[0][view].keys()
            s[view] = {key: {"mean": float(np.nanmean([r[view][key] for r in rows])),
                             "worst": float(np.nanmax(np.abs([r[view][key] for r in rows]))) if "agreement" not in key
                             else float(np.nanmin([r[view][key] for r in rows]))} for key in keys}
        summary[k] = s
        q = s["query"]
        print(f"K={k} ({100 * s['page_fraction']:.1f}% of pages): eligible mass kept {s['eligible_mass_kept_mean']:.4f} "
              f"(min {s['eligible_mass_kept_min']:.4f}) | worst |d recall| "
              f"{max(v['worst'] for kk, v in q.items() if kk.startswith('recall_')):.4f} | class agreement "
              f"mean {q['class_agreement']['mean']:.4f} min {q['class_agreement']['worst']:.4f} | worst |d lead median| "
              f"{max(v['worst'] for kk, v in q.items() if kk.startswith('lead_')):.1f}")
    meta = load_meta(paths[0])
    write_json(args.out, {"provenance": provenance(cfg, model=meta["model"], context_length=meta["context_length"],
                                                   trace_dir=args.trace_dir, n_traces=len(paths)),
                          "summary": summary})


if __name__ == "__main__":
    main()
