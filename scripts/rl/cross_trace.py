"""A like-for-like trace for kill test R1: one model's prompts with another model's response lengths.

The two models' own traces (scripts/rl/rollout_trace.py) differ in response lengths, so comparing their replay
headroom mixes the memory layout with the workload. This trace keeps the prompt token ids of --prompts (the
model to be replayed, its own tokenizer) and takes each sample's length from --lengths, matched by problem and
sample index. Both traces must come from the same rollout settings, so problem i is the same problem in both.

Usage: cross_trace.py CONFIG OUT_DIR --prompts TRACE.json --lengths TRACE.json --name NAME
Writes OUT_DIR/trace_<NAME>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--lengths", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    tp, tl = json.load(open(args.prompts)), json.load(open(args.lengths))
    keys = ("prompts", "num_prompts", "n", "max_tokens", "temperature", "top_p", "seed")
    sp, sl = tp["provenance"]["settings"], tl["provenance"]["settings"]
    if any(sp[k] != sl[k] for k in keys) or len(tp["trace"]) != len(tl["trace"]):
        sys.exit("the two traces come from different rollout settings")
    trace = [{"prompt_token_ids": p["prompt_token_ids"], "lengths": q["lengths"], "finish": q["finish"]}
             for p, q in zip(tp["trace"], tl["trace"])]
    lens = [n for r in trace for n in r["lengths"]]
    summary = {"requests": len(lens), "output_tokens": sum(lens), "output_mean": sum(lens) / len(lens),
               "output_max": max(lens), "output_median": sorted(lens)[len(lens) // 2]}
    print(json.dumps(summary, indent=1))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prov = provenance(cfg, model=tp["provenance"]["model"], context_length=None, script=str(Path(__file__).resolve()),
                      settings=sp, prompts_from=str(Path(args.prompts).resolve()), lengths_from=str(Path(args.lengths).resolve()))
    json.dump({"provenance": prov, "engine": tp["engine"], "summary": summary, "trace": trace},
              open(out / f"trace_{args.name}.json", "w"))


if __name__ == "__main__":
    main()
