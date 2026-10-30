"""Compare the vime rollout steps at watermark 0 and 0.05 (scripts/rl/vime_verify.py), per model.

Each pair's runs sample independently, so they do slightly different work: the report gives the step time, the
output tokens and the output rate, and the changes in each from watermark 0 to 0.05. From each run's /metrics series
it also gives the engine's mean running requests and KV-cache usage while requests wait (the KV-bound phase): with
spec decoding, vLLM holds extra state blocks per hybrid request, which shows as fewer running at the same usage.

Usage: vime_verify_report.py CONFIG [--set k=v]   writes config vime.report.out
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402

KEYS = ("rollout_time_s", "output_tokens", "output_tok_s", "preemptions", "response_len_mean", "truncated_ratio",
        "num_gpus", "spec_acceptance")      # the last two only in runs from after vime.num_gpus / speculative_config


def saturated(path):
    """Mean running requests and KV usage over the polls with requests waiting."""
    rows = [r for r in map(json.loads, open(path)) if r.get("vllm:num_requests_waiting", 0) > 0]
    return {"polls": len(rows),
            "mean_running": sum(r["vllm:num_requests_running"] for r in rows) / len(rows),
            "mean_kv_usage": sum(r["vllm:kv_cache_usage_perc"] for r in rows) / len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    rc = cfg["vime"]["report"]
    runs = Path(rc["runs_dir"])
    pairs = {}
    for model, ids in rc["pairs"].items():
        res = {k: json.load(open(runs / job / "vime_verify.json")) for k, job in ids.items()}
        a, b = res["wm0"]["summary"], res["wm05"]["summary"]
        pairs[model] = {
            "wm0": {k: a.get(k) for k in KEYS}, "wm05": {k: b.get(k) for k in KEYS},
            "engine_watermark": {k: r["evidence"]["server_info_scheduler_watermark"] for k, r in res.items()},
            "saturated": {k: saturated(runs / job / "metrics.jsonl") for k, job in ids.items()},
            "step_time_change": b["rollout_time_s"] / a["rollout_time_s"] - 1,
            "output_tokens_change": b["output_tokens"] / a["output_tokens"] - 1,
            "output_rate_change": b["output_tok_s"] / a["output_tok_s"] - 1,
        }
        print(f"{model}: step {a['rollout_time_s']:.0f} s -> {b['rollout_time_s']:.0f} s "
              f"({pairs[model]['step_time_change']:+.1%}), output tokens {pairs[model]['output_tokens_change']:+.1%}, "
              f"rate {pairs[model]['output_rate_change']:+.1%}, preemptions {a['preemptions']:.0f} -> {b['preemptions']:.0f}, "
              f"running while waiting {pairs[model]['saturated']['wm0']['mean_running']:.1f} at KV "
              f"{pairs[model]['saturated']['wm0']['mean_kv_usage']:.1%} (wm0)")
    prov = provenance(cfg, model=list(rc["pairs"]), context_length=None, script=str(Path(__file__).resolve()),
                      runs={m: ids for m, ids in rc["pairs"].items()})
    json.dump({"provenance": prov, "pairs": pairs}, open(rc["out"], "w"), indent=1)


if __name__ == "__main__":
    main()
