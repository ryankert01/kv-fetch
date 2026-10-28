"""P2: per-kernel-class GPU time from vLLM torch-profiler traces (engine_profile.py output).

Each engine step carries a GPU annotation `execute_context_<n>(<tokens>)_generation_<m>(<tokens>)`
around the model forward; logits (lm_head) and sampling run after it, outside the annotation. A step's
window therefore runs from the end of the previous annotation to the end of its own, and every kernel
starting in it belongs to the step. Steps are grouped into decode-only (no context request) and
prefill-only (no generation request); mixed steps are kept apart. Per group: step count, mean window
per step, kernel time per class, GPU busy share of the window (idle = the rest), and the time between
annotations (`outside_ms_per_step`: logits, sampling and any idle time).

Usage: parse_trace.py RESULTS_DIR [MODEL ...]   (reads RESULTS_DIR/<model>.json, writes RESULTS_DIR/<model>_kernels.json)
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = [   # first match wins
    ("gdn_recurrent", r"fused_recurrent_gated_delta_rule|fused_sigmoid_gating_delta_rule"),
    ("gdn_chunk", r"chunk_|recompute_w_u|merge_16x16|solve_tril|l2norm"),
    ("conv1d", r"causal_conv1d"),
    ("gdn_aux", r"_fused_post_conv|fused_gdn_gating|gated_rms|rmsnorm_gated|olmo_hybrid_gdn"),
    ("mamba_state", r"mamba|batch_memcpy"),
    ("attention", r"flash_fwd|flash::|fmha|flashinfer|paged_attention|unified_attention|_fwd_kernel_stage"),
    ("attn_aux", r"qk_rmsnorm_rope|reshape_and_cache|rotary|rope"),
    ("moe", r"moe|marlin"),
    ("gemm", r"gemm|cutlass|cublas|xmma|nvjet|splitKreduce|gemv"),
    ("norm_act", r"rms_norm|silu|gelu|triton_poi|triton_red|triton_per|sigmoid"),
    ("sampling", r"sample|gumbel|topk|top_k|argmax|softmax|logsumexp|penalt"),
    ("copy_misc", r"Memcpy|Memset|copy|gather|index|Fill|Cat|elementwise|scatter|zero"),
]
STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def kclass(name: str) -> str:
    for c, pat in CLASSES:
        if re.search(pat, name):
            return c
    return "other"


def parse(path: Path) -> dict:
    with gzip.open(path) if path.suffix == ".gz" else open(path) as f:
        ev = json.load(f)["traceEvents"]
    steps = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation" and STEP.match(e.get("name", ""))),
                   key=lambda e: e["ts"])
    kernels = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")), key=lambda e: e["ts"])
    groups = defaultdict(lambda: {"steps": 0, "span_us": 0.0, "busy_us": 0.0, "gap_us": 0.0, "class_us": Counter(),
                                  "kernel_us": Counter(), "ctx_tokens": 0, "gen_tokens": 0})
    ki, prev_end = 0, None
    for s in steps:
        nctx, tctx, ngen, tgen = map(int, STEP.match(s["name"]).groups())
        g = groups["decode" if nctx == 0 else "prefill" if ngen == 0 else "mixed"]
        a, b = (s["ts"] if prev_end is None else prev_end), s["ts"] + s["dur"]
        while ki < len(kernels) and kernels[ki]["ts"] < a:
            ki += 1
        busy_end = a
        j = ki
        while j < len(kernels) and kernels[j]["ts"] < b:
            k = kernels[j]
            g["class_us"][kclass(k["name"])] += k["dur"]
            g["kernel_us"][k["name"][:120]] += k["dur"]
            st, en = max(k["ts"], busy_end), k["ts"] + k["dur"]
            if en > st:
                g["busy_us"] += en - st
                busy_end = en
            j += 1
        ki = j
        g["steps"] += 1
        g["span_us"] += b - a
        g["ctx_tokens"] += tctx
        g["gen_tokens"] += tgen
        g["gap_us"] += s["ts"] - a
        prev_end = b
    out = {}
    for name, g in groups.items():
        n = g["steps"]
        out[name] = {"steps": n, "span_ms_per_step": g["span_us"] / n / 1e3,
                     "outside_ms_per_step": g["gap_us"] / n / 1e3, "busy_share": g["busy_us"] / g["span_us"],
                     "ctx_tokens_per_step": g["ctx_tokens"] / n, "gen_tokens_per_step": g["gen_tokens"] / n,
                     "class_ms_per_step": {c: v / n / 1e3 for c, v in g["class_us"].most_common()},
                     "top_kernels_ms_per_step": {k: v / n / 1e3 for k, v in g["kernel_us"].most_common(25)}}
    return out


def main():
    root = Path(sys.argv[1])
    models = sys.argv[2:] or [p.stem for p in root.glob("*.json") if not p.stem.endswith("_kernels")]
    for m in models:
        meta = json.load(open(root / f"{m}.json"))
        res = {"source": f"{m}.json", "classes": [c for c, _ in CLASSES] + ["other"], "profiles": {}}
        for p in meta["profiles"]:
            trace = next(root / f for f in p["files"] if f.endswith((".json.gz", ".json")))
            res["profiles"][p["name"]] = parse(trace)
            for grp, r in res["profiles"][p["name"]].items():
                top = ", ".join(f"{c} {v:.2f}" for c, v in list(r["class_ms_per_step"].items())[:6])
                print(f"{m} {p['name']:18s} {grp:8s} steps {r['steps']:3d} window {r['span_ms_per_step']:7.2f} ms "
                      f"outside {r['outside_ms_per_step']:5.2f} busy {r['busy_share']:.2f} | {top}")
        json.dump(res, open(root / f"{m}_kernels.json", "w"), indent=1)


if __name__ == "__main__":
    main()
