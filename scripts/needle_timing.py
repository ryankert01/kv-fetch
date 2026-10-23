"""When does the RULER needle become important? (task-design check, not a trajectory statistic)

For each small-eval run, averages the attention mass on the queried needle's pages over the
strongest retrieval heads in traced layers and over samples, at every position from the
last prompt queries (prefill observation window) through the decode steps.
x = 0 is decode step 0 (the last prompt token, which produces the first answer token).

  python scripts/needle_timing.py configs/small_eval.yaml --runs <tag> [<tag> ...]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from kvfetch.config import add_config_args, config_from_args
from kvfetch.provenance import provenance, write_json
from kvfetch.tracing.schema import load_trace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--runs", nargs="+", required=True, help="small-eval result tags (<model>__<ctx>)")
    p.add_argument("--heads", type=int, default=3, help="strongest probe heads (in traced layers) to average")
    p.add_argument("--out", default="results/needle_timing.json")
    args = p.parse_args()
    cfg = config_from_args(args)
    se = cfg["small_eval"]
    task = next(t["name"] for t in se["tasks"] if t["kind"] == "ruler")
    out = {}
    for tag in args.runs:
        res = json.load(open(Path(se["out_dir"]) / f"{tag}.json"))
        heads = [(h["layer"], h["head"]) for h in res["probe"]["top_heads"] if h["layer"] in res["traced_layers"]][:args.heads]
        curves, stops = [], []
        for path in sorted(Path(res["trace_dir"], task).glob("*.safetensors")):
            tr = load_trace(path)
            pages = tr.meta["query_needle_pages"]
            stops.append(tr.meta["natural_stop_step"])
            for layer, head in heads:
                li = tr.meta["layers"].index(layer)
                obs = tr.obs_mass[:, li, head][:, pages].astype(np.float32).sum(-1)   # x = -(W-1) .. 0
                dec = tr.mass[1:, li, head][:, pages].astype(np.float32).sum(-1)      # x = 1 .. T-1
                curves.append(np.concatenate([obs, dec]))
        c = np.stack(curves)
        W = tr.obs_mass.shape[0]
        out[tag] = {"model": res["provenance"]["model"], "heads": heads, "n_samples": len(stops),
                    "x": list(range(-(W - 1), c.shape[1] - (W - 1))),
                    "mean_needle_mass": c.mean(0).round(4).tolist(),
                    "natural_stop_steps": stops}
        print(f"{tag}: heads {heads}, mass at x=-31 {c[:, 0].mean():.3f}, x=0 {c[:, W - 1].mean():.3f}, "
              f"x=40 {c[:, W - 1 + 40].mean():.3f}; stops {sorted(set(s for s in stops if s is not None))}")
    write_json(args.out, {"provenance": provenance(cfg, model=None, context_length=se["context_length"],
                                                   task=task, runs=args.runs), "runs": out})


if __name__ == "__main__":
    main()
