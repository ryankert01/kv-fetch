"""Per-decode-step host work from an nsys capture (engine_profile.py under `nsys profile`, profiler=cuda).

From the exported sqlite: kernels inside vs outside CUDA graphs (count and GPU time per step), CUDA
runtime calls per step (launches, graph launches, copies, event syncs), GPU idle time between kernels,
and the kernels launched outside graphs by name. Steps = engine iterations, counted by vLLM's
once-per-step sampler bookkeeping kernel (`_post_update_kernel`).

Usage: nsys_steps.py RUN_DIR [RUN_DIR ...]   (each with nsys.sqlite; writes RUN_DIR/nsys_steps.json)
"""
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyse(run: Path) -> dict:
    db = sqlite3.connect(run / "nsys.sqlite")
    K = db.execute("select start, end, graphNodeId, (select value from StringIds where id = shortName) "
                   "from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    R = db.execute("select (select value from StringIds where id = nameId), start, end from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    steps = max(sum(1 for k in K if k[3] == "_post_update_kernel"), 1)
    idle, end = 0, K[0][1]
    for s, e, _, _ in K[1:]:
        if s > end:
            idle += s - end
        end = max(end, e)
    span = K[-1][1] - K[0][0]
    calls = Counter(n.split("_v")[0] for n, _, _ in R)
    call_t = defaultdict(int)
    for n, s, e in R:
        call_t[n.split("_v")[0]] += e - s
    eager = [k for k in K if not k[2]]
    graph_t = sum(k[1] - k[0] for k in K if k[2])
    eager_t = sum(k[1] - k[0] for k in eager)
    by_name = Counter(k[3] for k in eager)
    name_t = defaultdict(int)
    for k in eager:
        name_t[k[3]] += k[1] - k[0]
    per = lambda x: x / steps
    return {"steps": steps, "span_ms_per_step": per(span) / 1e6, "gpu_idle_ms_per_step": per(idle) / 1e6,
            "graph_kernel_ms_per_step": per(graph_t) / 1e6, "eager_kernel_ms_per_step": per(eager_t) / 1e6,
            "eager_kernels_per_step": per(len(eager)), "graph_kernels_per_step": per(len(K) - len(eager)),
            "runtime_calls_per_step": {n: per(c) for n, c in calls.most_common(15)},
            "runtime_ms_per_step": {n: per(call_t[n]) / 1e6 for n, _ in calls.most_common(15)},
            "eager_kernels": {n: {"per_step": per(c), "us_per_step": per(name_t[n]) / 1e3} for n, c in by_name.most_common(40)}}


def main():
    for d in sys.argv[1:]:
        run = Path(d)
        r = analyse(run)
        json.dump(r, open(run / "nsys_steps.json", "w"), indent=1)
        print(f"{run.name}: steps {r['steps']} span {r['span_ms_per_step']:.2f} ms idle {r['gpu_idle_ms_per_step']:.2f} ms | "
              f"graph kernels {r['graph_kernels_per_step']:.0f} ({r['graph_kernel_ms_per_step']:.2f} ms) eager kernels "
              f"{r['eager_kernels_per_step']:.1f} ({r['eager_kernel_ms_per_step']:.2f} ms) | calls/step "
              + ", ".join(f"{n} {c:.1f}" for n, c in list(r["runtime_calls_per_step"].items())[:6]))


if __name__ == "__main__":
    main()
