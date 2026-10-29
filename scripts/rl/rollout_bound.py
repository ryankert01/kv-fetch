"""How far each replayed rollout step is from a no-preemption bound with the same memory (kill test R1).

Step time model, fitted per run by least squares over its own steps (scripts/rl/rollout_replay.py):
    step_s = a + b * decode_tokens + c * context_tokens + d * (new_prefill + recompute tokens)
The fit is on sums over windows of `fit_window` consecutive steps (config `rollout.bound`): with async
scheduling a step's recorded interval (schedule to next schedule) carries the previous step's GPU work, so
per-step rows are off by one and fit badly (R2 0.05 on qwen3_4b m70_wm0); the model is additive, so window
sums keep it exact while the one-step lag washes out (R2 0.95 at 50 steps).
a is the per-step fixed cost (weights, launches), b the per-request decode cost (for a hybrid this includes
the recurrent state round trip), c the attention read per context token, d the prefill cost per token.

Under this model a rollout step's time is a * steps + (work terms). The work terms are fixed by the trace
except for recompute, so a memory policy can only (1) cut the number of steps by running more requests at
once and (2) avoid recompute. The bound takes the fewest steps any schedule could use and no recompute:
    steps_min = max(longest response, sum of per-request block-steps / pool blocks, total decode / max_num_seqs)
where a request holding context x uses ceil(x / block_size) attention blocks plus its state blocks (3 per
request for Qwen3.5, none for a dense model), i.e. perfect packing with known lengths. It is optimistic
(no fragmentation, no align-mode transient blocks, prefill overlapped), which is what a kill test needs:
headroom = 1 - bound / measured makespan is an upper limit on what any admission / preemption / swap
policy can win.

Usage: rollout_bound.py REPLAY.json [REPLAY.json ...] --config CONFIG --out OUT.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402


def fit(steps: list[list], makespan: float, window: int) -> tuple[np.ndarray, float]:
    t = [s[0] for s in steps] + [makespan]
    dur = np.diff(np.array(t))
    X = np.array([[1.0, s[3], s[8], s[4] + s[5]] for s in steps])
    n = len(steps) // window * window
    X, dur = X[:n].reshape(-1, window, 4).sum(1), dur[:n].reshape(-1, window).sum(1)
    coef, *_ = np.linalg.lstsq(X, dur, rcond=None)
    pred = X @ coef
    r2 = 1 - ((dur - pred) ** 2).sum() / ((dur - dur.mean()) ** 2).sum()
    return coef, float(r2)


def ceil_sum(n: int, bs: int) -> int:
    """sum of ceil(x / bs) for x = 1 .. n"""
    q, r = divmod(n, bs)
    return bs * q * (q + 1) // 2 + r * (q + 1)


def bound_terms(trace: list[dict], bs: int, state_blocks: int, blocks: int, max_seqs: int) -> dict:
    area = dec = ctx = pre = 0
    longest = 0
    for r in trace:
        P = len(r["prompt_token_ids"])
        for L in r["lengths"]:
            pre += P
            dec += L - 1                                  # the first token comes out of the prefill step
            ctx += (L - 1) * P + (L - 1) * L // 2         # context read by each decode step
            # block-steps: attention blocks at context P + k for k = 0 .. L, plus the state blocks
            area += ceil_sum(P + L, bs) - ceil_sum(P - 1, bs) + state_blocks * (L + 1)
            longest = max(longest, L)
    steps = max(longest, math.ceil(area / blocks), math.ceil(dec / max_seqs))
    return {"steps_min": steps, "steps_longest": longest, "steps_memory": math.ceil(area / blocks),
            "steps_seqs": math.ceil(dec / max_seqs), "decode": dec, "context": ctx, "prefill": pre}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    bc = cfg["rollout"]["bound"]
    rows = []
    for path in map(Path, args.runs):
        d = json.load(open(path))
        e, s, st = d["engine"], d["summary"], d["steps"]
        if len(st) < bc["fit_min_steps"]:
            print(f"{path.name}: {len(st)} steps, too few to fit", file=sys.stderr)
            continue
        trace = json.load(open(d["provenance"]["trace"]))["trace"]
        coef, r2 = fit(st, s["makespan_s"], bc["fit_window"])
        a, b, c, dd = coef
        bt = bound_terms(trace, e["block_size"], e["state_blocks_per_request"], e["num_gpu_blocks"], e["max_num_seqs"])
        t_bound = a * bt["steps_min"] + b * bt["decode"] + c * bt["context"] + dd * bt["prefill"]
        ctx_actual = sum(x[8] for x in st)
        t_model = a * len(st) + b * s["decode_tokens"] + c * ctx_actual + dd * (s["new_prefill_tokens"] + s["recompute_tokens"])
        rows.append({
            "run": path.stem, "block_size": e["block_size"], "blocks": e["num_gpu_blocks"], "watermark": e["watermark"],
            "makespan_s": s["makespan_s"], "steps": len(st), "preemptions": s["preemptions"], "recompute_tokens": s["recompute_tokens"],
            "mean_running": s["mean_running"], "fit": {"a_ms": a * 1e3, "b_us_per_decode": b * 1e6, "c_ns_per_ctx": c * 1e9,
                                                       "d_us_per_prefill": dd * 1e6, "r2": r2, "window": bc["fit_window"]},
            "model_s": t_model, "bound": bt, "bound_s": t_bound, "headroom": 1 - t_bound / s["makespan_s"],
            "waste_s": {"extra_steps": a * (len(st) - bt["steps_min"]), "recompute": dd * s["recompute_tokens"],
                        "extra_context": c * (ctx_actual - bt["context"])}})
    for r in rows:
        f, w = r["fit"], r["waste_s"]
        print(f"{r['run']:28s} makespan {r['makespan_s']:7.1f}s  model {r['model_s']:7.1f}s (R2 {f['r2']:.2f})  bound {r['bound_s']:7.1f}s  "
              f"headroom {r['headroom']:5.1%}  steps {r['steps']} vs min {r['bound']['steps_min']} "
              f"(longest {r['bound']['steps_longest']}, memory {r['bound']['steps_memory']})  preempt {r['preemptions']}  "
              f"waste: steps {w['extra_steps']:.0f}s, recompute {w['recompute']:.0f}s, context {w['extra_context']:.0f}s  "
              f"fit a={f['a_ms']:.1f}ms b={f['b_us_per_decode']:.0f}us c={f['c_ns_per_ctx']:.2f}ns d={f['d_us_per_prefill']:.0f}us")
    json.dump({"sources": args.runs, "runs": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
