"""Automatic analysis of every harness run: candidate bottlenecks under the selection rule, ablation deltas,
hybrid vs dense, serving metrics. Writes results/profile/findings.json and findings.md.

Decode roofline per profiled point (pure-decode steps, window from the previous step's forward end to this
one's, so logits and sampling count): each kernel class's time per step against the
time its bytes need at the measured read bandwidth -- weights for GEMMs (MoE: experts the batch touches),
KV for attention, recurrent state read + write for GDN, conv state for conv1d; GEMMs also against the
bf16 FLOP peak. Idle time (gaps between steps, bubbles inside them) is pure overhead. A candidate
qualifies when share >= qualify_share and efficiency <= qualify_efficiency, or overhead >= qualify_overhead_share.

Usage: findings.py CONFIG
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "profile"))
from kvfetch.config import load_config  # noqa: E402
from microbench import weight_bytes  # noqa: E402

_consts = {}


def model_consts(name: str) -> dict:
    if name in _consts:
        return _consts[name]
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(name, local_files_only=True))
    c = json.load(open(d / "config.json"))
    t = c.get("text_config", c)
    types = t.get("layer_types")
    n_attn = types.count("full_attention") if types else t["num_hidden_layers"]
    n_lin = types.count("linear_attention") if types else 0
    hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    out = {"n_attn": n_attn, "n_lin": n_lin, "kv_per_token": n_attn * t["num_key_value_heads"] * hd * 2 * 2,
           "ssm_bytes": 0, "conv_bytes": 0, "experts": t.get("num_experts"), "topk": t.get("num_experts_per_tok")}
    if n_lin and "linear_num_value_heads" in t:
        kd, vd = t["linear_key_head_dim"], t["linear_value_head_dim"]
        sb = 4 if t.get("mamba_ssm_dtype", "float32") == "float32" else 2
        out["ssm_bytes"] = n_lin * t["linear_num_value_heads"] * kd * vd * sb
        out["conv_bytes"] = n_lin * (t["linear_conv_kernel_dim"] - 1) * (2 * t["linear_num_key_heads"] * kd + t["linear_num_value_heads"] * vd) * 2
    w = weight_bytes(d)
    tied = c.get("tie_word_embeddings", t.get("tie_word_embeddings", False))
    lm = w.get("lm_head", w.get("embed", 0) if tied else 0)
    out["dense_weights"] = sum(v for k, v in w.items() if k not in ("vision", "mtp", "embed", "lm_head", "moe_experts")) + lm
    out["expert_weights"] = w.get("moe_experts", 0)
    _consts[name] = out
    return out


def decode_rows(prof: dict, C: int, mc: dict, bw: float, peak: float) -> dict:
    g = prof["decode"]
    b = round(g["gen_tokens_per_step"])
    step = g["span_ms_per_step"]                    # window = previous annotation end -> this one's end
    cls = g["class_ms_per_step"]
    wbytes = mc["dense_weights"]
    if mc["experts"]:
        wbytes += mc["expert_weights"] * (1 - (1 - mc["topk"] / mc["experts"]) ** b)
    ideal = {"gemm": max(wbytes / bw, 2 * (wbytes / 2) * b / peak),
             "attention": b * C * mc["kv_per_token"] / bw,
             "gdn_recurrent": b * 2 * mc["ssm_bytes"] / bw if mc["ssm_bytes"] else None,
             "conv1d": b * 2 * mc["conv_bytes"] / bw if mc["conv_bytes"] else None}
    rows = {}
    for k, v in cls.items():
        key = "gemm" if k in ("gemm", "moe") else k
        r = rows.setdefault(key, {"ms": 0.0})
        r["ms"] += v
    for k, r in rows.items():
        r["share"] = r["ms"] / step
        if ideal.get(k) is not None:
            r["ideal_ms"] = ideal[k] * 1e3
            r["efficiency"] = r["ideal_ms"] / r["ms"] if r["ms"] else None
    idle = (1 - g["busy_share"]) * step
    rows["gpu_idle"] = {"ms": idle, "share": idle / step, "overhead": True}
    total = wbytes + b * C * mc["kv_per_token"] + b * 2 * (mc["ssm_bytes"] + mc["conv_bytes"])   # every byte a step must move
    return {"batch": b, "context": C, "step_ms": step, "rows": rows, "weight_bytes": wbytes, "step_bytes": total,
            "step_ideal_ms": total / bw * 1e3, "step_efficiency": total / bw * 1e3 / step}


def main():
    cfg = load_config(sys.argv[1])
    hc, fc = cfg["harness"], cfg["findings"]
    mb = json.load(open(REPO / fc["microbench"]))
    bw = max(r["GBps"] for r in mb["read"]) * 1e9
    peak = max(r["TFLOPs"] for r in mb["gemm"]) * 1e12
    models = cfg["profile"]["models"]
    qroot, runs_root = REPO / hc["queue_dir"], REPO / hc["runs_dir"]
    jobs = {p.stem: json.loads(p.read_text()) for p in sorted((qroot / "done").glob("*.json"))}
    out = {"ceilings": {"read_GBps": bw / 1e9, "bf16_TFLOPs": peak / 1e12}, "decode_points": [], "prefill_points": [],
           "candidates": [], "sweeps": {}, "ablations": [], "serve": [], "pairs": []}
    engine_runs = defaultdict(list)
    for jid, job in jobs.items():
        rd = runs_root / jid
        if job["kind"] == "engine":
            m = job["model"]
            f, kf = rd / f"{m}.json", rd / f"{m}_kernels.json"
            if not f.exists():
                continue
            res = json.load(open(f))
            engine_runs[m].append((jid, job, res))
            out["sweeps"][jid] = {"model": m, "sets": job.get("sets", []), "engine": res["engine"], "gpu": res["provenance"].get("gpu"),
                                  "decode": [{k: d[k] for k in ("context", "batch", "step_ms", "tokens_per_s")} for d in res["decode"]],
                                  "prefill": [{k: d[k] for k in ("prompt_tokens", "median_s", "tokens_per_s")} for d in res["prefill"]]}
            if not kf.exists():
                continue
            mc = model_consts(models[m]["name"])
            for pname, prof in json.load(open(kf))["profiles"].items():
                if pname.startswith("decode_") and "decode" in prof:
                    C = int(re.match(r"decode_(\d+)_b\d+", pname).group(1))
                    row = decode_rows(prof, C, mc, bw, peak)
                    row.update(run=jid, model=m, kind=models[m]["kind"], sets=job.get("sets", []), point=pname)
                    out["decode_points"].append(row)
                elif pname.startswith("prefill_") and "prefill" in prof:
                    g = prof["prefill"]
                    T = g["ctx_tokens_per_step"]
                    step = g["span_ms_per_step"]
                    gemm = g["class_ms_per_step"].get("gemm", 0) + g["class_ms_per_step"].get("moe", 0)
                    flops = 2 * (mc["dense_weights"] + mc["expert_weights"] * (mc["topk"] or 0) / (mc["experts"] or 1)) / 2 * T
                    rows = {k: {"ms": v, "share": v / step} for k, v in g["class_ms_per_step"].items()}
                    rows["gemm_all"] = {"ms": gemm, "share": gemm / step, "efficiency": flops / peak * 1e3 / gemm if gemm else None}
                    idle = (1 - g["busy_share"]) * step
                    rows["gpu_idle"] = {"ms": idle, "share": idle / step, "overhead": True}
                    out["prefill_points"].append({"run": jid, "model": m, "kind": models[m]["kind"], "sets": job.get("sets", []),
                                                  "point": pname, "tokens_per_step": T, "step_ms": step, "rows": rows})
        elif job["kind"] == "serve":
            for f in sorted(rd.glob("*.json")):
                if f.name == "serve_index.json":
                    continue
                r = json.load(open(f))
                mkey, setting, wl, conc = re.match(r"(.+)_(default|prefix_off|[a-z0-9]+)_(reasoning|chat)_c(\d+)\.json", f.name).groups()
                out["serve"].append({"run": jid, "model": mkey, "setting": setting, "workload": wl, "concurrency": int(conc),
                                     **{k: r.get(k) for k in ("completed", "duration", "output_throughput", "total_token_throughput",
                                                              "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms")}})
    # candidates
    for pts, phase in ((out["decode_points"], "decode"), (out["prefill_points"], "prefill")):
        for p in pts:
            for k, r in p["rows"].items():
                eff = r.get("efficiency")
                over = r.get("overhead", False)
                qual = (over and r["share"] >= fc["qualify_overhead_share"]) or \
                       (r["share"] >= fc["qualify_share"] and eff is not None and eff <= fc["qualify_efficiency"])
                watch = r["share"] >= fc["watch_share"] and (over or eff is None or eff <= fc["qualify_efficiency"])
                if qual or watch:
                    gain = r["share"] * (1 - min(eff, 1)) if eff is not None else r["share"]
                    out["candidates"].append({"phase": phase, "class": k, "model": p["model"], "kind": p["kind"],
                                              "point": p["point"], "run": p["run"], "sets": p["sets"], "share": r["share"],
                                              "efficiency": eff, "gain_bound": gain, "status": "qualifies" if qual else "watch",
                                              "bound_known": eff is not None or over})
    out["candidates"].sort(key=lambda c: -c["gain_bound"])
    # ablations: same model, runs with different --set lists, decode points matched on (context, batch)
    for m, rs in engine_runs.items():
        base = next(((j, r) for j, job, r in rs if not job.get("sets")), None)
        if base is None:
            continue
        gpu = base[1]["provenance"].get("gpu")
        bd = {(d["context"], d["batch"]): d["step_ms"] for d in base[1]["decode"]}
        bp = {d["prompt_tokens"]: d["median_s"] for d in base[1]["prefill"]}
        for j, job, r in rs:
            if not job.get("sets") or r["provenance"].get("gpu") != gpu:   # compare on the same GPU type only
                continue
            dd = [{"context": d["context"], "batch": d["batch"], "step_ms": d["step_ms"],
                   "delta": d["step_ms"] / bd[(d["context"], d["batch"])] - 1} for d in r["decode"] if (d["context"], d["batch"]) in bd]
            pp = [{"prompt_tokens": d["prompt_tokens"], "delta": d["median_s"] / bp[d["prompt_tokens"]] - 1}
                  for d in r["prefill"] if d["prompt_tokens"] in bp]
            out["ablations"].append({"model": m, "run": j, "base": base[0], "sets": job["sets"], "why": job.get("why"),
                                     "decode": dd, "prefill": pp})
    # hybrid vs dense pairs from the base sweeps
    for h, d in cfg["profile"].get("pairs", []):
        hb = next((r for j, job, r in engine_runs.get(h, []) if not job.get("sets")), None)
        db = next((r for j, job, r in engine_runs.get(d, []) if not job.get("sets")), None)
        if hb and db and hb["provenance"].get("gpu") == db["provenance"].get("gpu"):
            hd_ = {(x["context"], x["batch"]): x["tokens_per_s"] for x in hb["decode"]}
            dd_ = {(x["context"], x["batch"]): x["tokens_per_s"] for x in db["decode"]}
            maxb = lambda r, C: max([x["batch"] for x in r["decode"] if x["context"] == C] or [0])
            out["pairs"].append({"hybrid": h, "dense": d,
                                 "decode_ratio": {f"{C}x{b}": hd_[(C, b)] / dd_[(C, b)] for (C, b) in hd_ if (C, b) in dd_},
                                 "max_batch": {C: [maxb(hb, C), maxb(db, C)] for C in sorted({x['context'] for x in hb['decode']} | {x['context'] for x in db['decode']})},
                                 "capacity_tokens": [hb["engine"]["capacity_tokens"], db["engine"]["capacity_tokens"]],
                                 "prefill_ratio": {x["prompt_tokens"]: x["tokens_per_s"] / y["tokens_per_s"]
                                                   for x in hb["prefill"] for y in db["prefill"] if x["prompt_tokens"] == y["prompt_tokens"]}})
    json.dump(out, open(REPO / "results/profile/findings.json", "w"), indent=1)
    write_md(out, REPO / "results/profile/findings.md")


def write_md(o: dict, path: Path):
    L = [f"# Findings (auto-generated by scripts/profile/findings.py)\n",
         f"Ceilings: read {o['ceilings']['read_GBps']:.0f} GB/s, bf16 {o['ceilings']['bf16_TFLOPs']:.0f} TFLOP/s.\n",
         "## Candidates (gain bound = share x (1 - efficiency); no bound -> share)\n",
         "| status | phase | class | model | point | sets | share | eff | gain bound |", "|---|---|---|---|---|---|---|---|---|"]
    for c in o["candidates"][:40]:
        eff = f"{c['efficiency']:.2f}" if c["efficiency"] is not None else "-"
        L.append(f"| {c['status']} | {c['phase']} | {c['class']} | {c['model']} | {c['point']} | {' '.join(c['sets']) or 'base'} | "
                 f"{c['share']:.2f} | {eff} | {c['gain_bound']:.2f} |")
    L += ["\n## Decode roofline (per profiled point; step eff = all bytes the step must move / read BW / step time)\n",
          "| model | sets | point | step ms | step eff | class: ms (share, eff) |", "|---|---|---|---|---|---|"]
    for p in o["decode_points"]:
        parts = []
        for k, r in sorted(p["rows"].items(), key=lambda kv: -kv[1]["ms"])[:7]:
            e = f", {r['efficiency']:.2f}" if r.get("efficiency") is not None else ""
            parts.append(f"{k} {r['ms']:.2f} ({r['share']:.0%}{e})")
        L.append(f"| {p['model']} | {' '.join(p['sets']) or 'base'} | {p['point']} | {p['step_ms']:.2f} | {p['step_efficiency']:.2f} | {'; '.join(parts)} |")
    L += ["\n## Prefill (per profiled point)\n", "| model | sets | point | step ms | class: ms (share) |", "|---|---|---|---|---|"]
    for p in o["prefill_points"]:
        parts = [f"{k} {r['ms']:.1f} ({r['share']:.0%}" + (f", eff {r['efficiency']:.2f}" if r.get("efficiency") else "") + ")"
                 for k, r in sorted(p["rows"].items(), key=lambda kv: -kv[1]["ms"])[:7]]
        L.append(f"| {p['model']} | {' '.join(p['sets']) or 'base'} | {p['point']} | {p['step_ms']:.1f} | {'; '.join(parts)} |")
    if o["ablations"]:
        L += ["\n## Ablations (step time vs the model's base run; negative = faster)\n"]
        for a in o["ablations"]:
            dec = ", ".join(f"{d['context']}x{d['batch']} {d['delta']:+.1%}" for d in a["decode"])
            pre = ", ".join(f"{d['prompt_tokens']} {d['delta']:+.1%}" for d in a["prefill"])
            L.append(f"- {a['model']} `{' '.join(a['sets'])}` ({a['why'] or ''}): decode {dec or '-'}; prefill {pre or '-'}")
    if o["pairs"]:
        L += ["\n## Hybrid vs dense (base runs)\n"]
        for p in o["pairs"]:
            L.append(f"- {p['hybrid']} / {p['dense']}: capacity {p['capacity_tokens'][0]:,} vs {p['capacity_tokens'][1]:,} tokens; "
                     f"max batch by context {p['max_batch']}; decode tok/s ratio "
                     + ", ".join(f"{k} {v:.2f}" for k, v in p["decode_ratio"].items())
                     + "; prefill ratio " + ", ".join(f"{k} {v:.2f}" for k, v in p["prefill_ratio"].items()))
    if o["serve"]:
        L += ["\n## Serving\n", "| model | setting | workload | conc | out tok/s | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms |", "|---|---|---|---|---|---|---|---|"]
        for s in sorted(o["serve"], key=lambda s: (s["workload"], s["concurrency"], s["model"], s["setting"])):
            f = lambda v, d=0: f"{v:.{d}f}" if isinstance(v, (int, float)) else "-"
            L.append(f"| {s['model']} | {s['setting']} | {s['workload']} | {s['concurrency']} | {f(s['output_throughput'])} | "
                     f"{f(s['median_ttft_ms'])} | {f(s['p99_ttft_ms'])} | {f(s['median_tpot_ms'], 1)} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
