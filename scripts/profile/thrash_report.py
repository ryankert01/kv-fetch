"""Summary of thrash_trace.py runs: where preemptions come from and what the recomputation costs.

Per run: throughput, steps; preemptions by trigger (a running request in decode, at a block boundary or
not, or in a prefill / recompute chunk, whose allocate_slots failed); tokens scheduled as decode, new
prefill and recompute; the prefix-cache hit on readmission as a share of what the request had computed;
how soon a readmitted request is preempted again; step time in steps with and without recompute work.
A decode step at computed position c needs a new block (attention, and in align mode one state block per
Mamba group) when c is a multiple of the block size.

Usage: thrash_report.py RUN.json [RUN.json ...]   (thrash_trace.py outputs; writes thrash_report.json next to the first)
"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path


def report(path: Path) -> dict:
    d = json.load(open(path))
    s, bs = d["summary"], d["engine"]["block_size"]
    steps = d["steps"]
    dur = [b[0] - a[0] for a, b in zip(steps, steps[1:])]
    with_rc = [t for t, x in zip(dur, steps) if x[5] > 0]
    without = [t for t, x in zip(dur, steps) if x[5] == 0 and x[4] == 0]
    trig = Counter()
    for p in d["preemptions"]:
        t = p["trigger"]
        if t is None:
            trig["none"] += 1
        elif t["phase"] == "decode":
            trig["decode, block boundary" if t["computed"] % bs == 0 else "decode, inside a block"] += 1
        else:
            trig[t["phase"]] += 1
    # steps from each readmission to the same request's next preemption
    pre_by_req: dict[str, list[int]] = {}
    for p in d["preemptions"]:
        pre_by_req.setdefault(p["req"], []).append(p["step"])
    again = []
    for r in d["readmissions"]:
        nxt = [x for x in pre_by_req.get(r["req"], []) if x > r["step"]]
        if nxt:
            again.append(nxt[0] - r["step"])
    victims = d["preemptions"]
    n = max(1, len(victims))
    res = {
        "run": path.stem, "engine": d["engine"],
        "output_tok_s": s["output_tok_s"], "wall_s": s["wall_s"], "steps": s["steps"], "ms_per_step": s["ms_per_step"],
        "mean_running": s["mean_running"], "mean_waiting": s["mean_waiting"],
        "preemptions": s["preemptions"], "triggers": dict(trig.most_common()),
        "requests_preempted": s["requests_preempted"], "preemptions_per_preempted_request": s["preemptions_per_preempted_request"],
        "victim_computed_mean": s["victim_computed_mean"],
        "victim_blocks_per_group_mean": [sum(v["blocks"][g] for v in victims) / n for g in range(len(victims[0]["blocks"]))] if victims else [],
        "decode_tokens": s["decode_tokens"], "new_prefill_tokens": s["new_prefill_tokens"], "recompute_tokens": s["recompute_tokens"],
        "recompute_per_new_prefill": s["recompute_tokens"] / max(1, s["new_prefill_tokens"]),
        "readmissions": s["readmissions"], "readmit_hit_share": s["readmit_hit_tokens"] / max(1, s["readmit_before_tokens"]),
        "readmitted_preempted_again": len(again), "steps_to_next_preemption_median": st.median(again) if again else None,
        "steps_with_recompute": len(with_rc), "ms_step_with_recompute": st.mean(with_rc) * 1e3 if with_rc else None,
        "ms_step_decode_only": st.mean(without) * 1e3 if without else None,
        "time_share_steps_with_recompute": sum(with_rc) / max(1e-9, sum(dur)),
        "free_blocks_mean": st.mean(x[7] for x in steps),
    }
    return res


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    rows = [report(p) for p in paths]
    for r in rows:
        print(f"== {r['run']}: {r['output_tok_s']:.0f} tok/s, {r['steps']} steps x {r['ms_per_step']:.1f} ms, running {r['mean_running']:.0f}, "
              f"waiting {r['mean_waiting']:.0f}, free blocks {r['free_blocks_mean']:.1f} of {r['engine']['num_gpu_blocks']}")
        print(f"   preemptions {r['preemptions']} ({r['requests_preempted']} requests; per request {r['preemptions_per_preempted_request']}); "
              f"triggers {r['triggers']}")
        print(f"   victim: {r['victim_computed_mean']:.0f} computed tokens, blocks per group {[round(x, 1) for x in r['victim_blocks_per_group_mean']]}")
        print(f"   tokens: decode {r['decode_tokens']/1e6:.2f}M, new prefill {r['new_prefill_tokens']/1e6:.2f}M, recompute {r['recompute_tokens']/1e6:.2f}M "
              f"({r['recompute_per_new_prefill']:.1f}x new prefill); readmissions {r['readmissions']}, prefix hit {r['readmit_hit_share']:.0%} of computed")
        print(f"   readmitted then preempted again: {r['readmitted_preempted_again']} (median {r['steps_to_next_preemption_median']} steps later)")
        print(f"   steps with recompute: {r['steps_with_recompute']} at {r['ms_step_with_recompute'] or 0:.1f} ms vs decode-only {r['ms_step_decode_only'] or 0:.1f} ms; "
              f"{r['time_share_steps_with_recompute']:.0%} of the time")
    json.dump({"sources": [str(p) for p in paths], "runs": rows}, open(paths[0].parent / "thrash_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
