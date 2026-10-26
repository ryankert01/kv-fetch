"""H3 (EXPERIMENTS.md, "Hybrid models"): AIME accuracy with page-sparse decoding on hybrid models
(kvfetch/sparse_decode.py), and Qwen3-4B's prompt-protected eviction for H-b. Run with .venv-hybrid.

  CUDA_VISIBLE_DEVICES=1 .venv-hybrid/bin/python scripts/run_h3.py configs/hybrid.yaml --model qwen35_4b   # worker
  .venv-hybrid/bin/python scripts/run_h3.py configs/hybrid.yaml --merge                                      # tables + gates

As scripts/run_e3.py: a unit is (model, run, group of h3.sample_group samples of every explore
problem), claimed in priority order (hybrid.h3.runs tiers). One JSON per (model, run, problem, sample)
under h3.out_dir/<model>/, with the generated token ids (H1 replays them). --merge writes
results/h3_<split>.json: per model avg@k, truncation, paired differences, the H3 gate and the H-b gate.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from kvfetch import sparse_decode as SD
from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, format_prompt, load_model, stop_token_ids
from kvfetch.provenance import provenance, write_json
from kvfetch.tasks.sweep import score
from run_e3 import bootstrap_ci, claim, problems, thinking


def h3_view(cfg: dict) -> dict:
    """run_e3.problems reads e3.tasks / e3.split."""
    return {**cfg, "e3": cfg["hybrid"]["h3"]}


def ours_spec(cfg: dict) -> dict:
    return {"fuse": "sum", "history_weight": 0.0, "history": "last", **cfg["hybrid"]["h1"]["ours"]}


def parse_run(label: str, cfg: dict) -> SD.Method:
    """method@budget; "ours" is H2's chosen summary format (hybrid.h2.chosen -> engine_specs)."""
    name, _, budget = label.partition("@")
    h2, h3 = cfg["hybrid"]["h2"], cfg["hybrid"]["h3"]
    spec = {}
    if name == "ours":
        if not h2.get("chosen"):
            raise SystemExit("ours runs need hybrid.h2.chosen (H2's pick)")
        es = dict(h2["engine_specs"][h2["chosen"]])
        name = es.pop("method")
        spec = {**ours_spec(cfg), **es} if name == "ours" else es
    return SD.Method(name, int(budget) if budget else None, spec=spec,
                     window=h3["snapkv_window"], protect_prompt=h3["snapkv_protect_prompt"])


def units(cfg: dict, mkey: str) -> list[tuple[str, int]]:
    h3 = cfg["hybrid"]["h3"]
    n_groups = -(-h3["samples"] // h3["sample_group"])
    return [(label, g) for tier in h3["runs"][mkey] for g in range(n_groups) for label in tier]


def out_path(cfg: dict, mkey: str, label: str, task: str, index: int, sample: int) -> Path:
    return Path(cfg["hybrid"]["h3"]["out_dir"]) / mkey / label.replace("@", "_") / task / f"{index:04d}_s{sample}.json"


def stops_for(name: str, tok, model, cfg: dict) -> list[int]:
    """Generation config and tokenizer EOS, plus the model config's (Qwen3.5-4B ships no generation config)."""
    e = model.config.eos_token_id
    return sorted(set(stop_token_ids(name, tok, cfg)) | set(e if isinstance(e, list) else [e] if e is not None else []))


def work(cfg: dict, args) -> None:
    h3, defs = cfg["hybrid"]["h3"], cfg["definitions"]
    mc = cfg["hybrid"]["models"][args.model]
    SD.register_sparse_attention()
    model, tok = load_model(mc["name"], cfg, args.device, attn_implementation=SD.SPARSE_ATTN)
    stops = stops_for(mc["name"], tok, model, cfg)
    probs = problems(h3_view(cfg), tok)[:args.limit]
    prompts = {(s.task, s.index): encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"] for s in probs}
    max_new = args.max_new or h3["max_new_tokens"][args.model]
    max_len = max(map(len, prompts.values())) + max_new + 1
    task_id = {t: i for i, t in enumerate(h3["tasks"])}
    prov = provenance(cfg, model=mc["name"], context_length=max_len, max_new_tokens=max_new, sampling=mc["sampling"],
                      stop_ids=stops)
    for label, g in units(cfg, args.model):
        if args.runs and label not in args.runs:
            continue
        if not claim(Path(h3["claims_dir"]) / args.model / f"{label}__g{g}.claim"):
            continue
        method = parse_run(label, cfg)
        samples = range(g * h3["sample_group"], min(h3["samples"], (g + 1) * h3["sample_group"]))
        jobs = [SD.Job(key=(s.task, s.index, k), prompt_ids=prompts[(s.task, s.index)],
                       seed=[h3["seed"], task_id[s.task], s.index, k])
                for k in samples for s in probs if not out_path(cfg, args.model, label, s.task, s.index, k).exists()]
        if not jobs:
            continue
        refs = {(s.task, s.index): s.answers for s in probs}
        t0, n_tok = time.perf_counter(), [0]
        print(f"[h3] {args.model} unit {label} samples {list(samples)}: {len(jobs)} jobs", flush=True)

        def done(job, toks, stopped):
            task, index, k = job.key
            text = tok.decode(toks)
            sc = score(task, [text], refs[(task, index)])
            n_tok[0] += len(toks)
            write_json(out_path(cfg, args.model, label, task, index, k), {
                "model_key": args.model, "task": task, "index": index, "sample": k, "run": label,
                "method": method.name, "spec": method.spec, "budget_tokens": method.budget_tokens, "prompt_tokens": len(job.prompt_ids),
                "n_gen": len(toks), "stopped": stopped, "score": sc, "answer": refs[(task, index)][0][0],
                "text": text, "token_ids": [int(t) for t in toks], "provenance": prov})
            dt = time.perf_counter() - t0
            print(f"[h3] {args.model} {label} {task}/{index} s{k}: score {sc} n_gen {len(toks)} stopped {stopped} | "
                  f"unit {n_tok[0] / dt:.0f} tok/s", flush=True)

        eng = SD.SparseEngine(model, method, batch=args.batch or h3["batch_size"][args.model], max_len=max_len,
                              page_size=defs["page_size"], sink_pages=defs["pinned"]["sink_pages"],
                              recent_tokens=defs["pinned"]["recent_tokens"],
                              obs_window=cfg["trace"]["prefill_obs_window"], device=args.device)
        SD.run_batch(model, eng, jobs, stop_ids=stops, max_new_tokens=max_new, sampling=mc["sampling"], on_done=done,
                     log_every=args.log_every)
        del eng
        torch.cuda.empty_cache()
        print(f"[h3] {args.model} unit {label} g{g} finished in {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------- merge
def model_summary(rows: list[dict], labels: list[str], tasks: list[str]) -> tuple[dict, dict, dict]:
    by = {}
    for r in rows:
        by.setdefault(r["run"], {}).setdefault((r["task"], r["index"]), {})[r["sample"]] = r
    summary, per_problem = {}, {}
    for label in labels:
        d = by.get(label, {})
        if not d:
            continue
        acc = {p: np.mean([r["score"] for r in d[p].values()]) for p in d}
        flat = [r for p in d for r in d[p].values()]
        per_problem[label] = {f"{t}/{i}": {"acc": float(acc[(t, i)]), "n": len(d[(t, i)])} for t, i in d}
        summary[label] = {
            "n_problems": len(d), "n_samples": len(flat),
            "samples_per_problem": [int(min(len(v) for v in d.values())), int(max(len(v) for v in d.values()))],
            "acc": float(np.mean(list(acc.values()))),
            "acc_by_task": {t: float(np.mean([a for (tt, _), a in acc.items() if tt == t])) for t in tasks
                            if any(tt == t for tt, _ in acc)},
            "mean_gen_tokens": float(np.mean([r["n_gen"] for r in flat])),
            "truncated": float(np.mean([not r["stopped"] for r in flat])),
            "acc_if_stopped": float(np.mean([r["score"] for r in flat if r["stopped"]])) if any(r["stopped"] for r in flat) else None,
        }
    return by, summary, per_problem


def paired(by: dict, a: str, b: str) -> dict | None:
    """Mean over problems of acc_a - acc_b on the samples both runs have (paired by sample index)."""
    if a not in by or b not in by:
        return None
    diffs = []
    for p in sorted(set(by[a]) & set(by[b])):
        ks = sorted(set(by[a][p]) & set(by[b][p]))
        if ks:
            diffs.append(np.mean([by[a][p][k]["score"] - by[b][p][k]["score"] for k in ks]))
    if not diffs:
        return None
    x = np.array(diffs)
    return {"diff_points": 100 * float(x.mean()), "ci95_points": [100 * v for v in bootstrap_ci(x, 10000, 0)],
            "n_problems": len(x), "wins": int((x > 0).sum()), "losses": int((x < 0).sum())}


def merge(cfg: dict) -> None:
    h3 = cfg["hybrid"]["h3"]
    out, bys = {"models": {}}, {}
    prov = None
    for mkey in h3["runs"]:
        rows = [json.loads(p.read_text()) for p in sorted((Path(h3["out_dir"]) / mkey).glob("*/*/*.json"))]
        if not rows:
            continue
        prov = prov or rows[0]["provenance"]
        labels = [l for tier in h3["runs"][mkey] for l in tier]
        by, summary, per_problem = model_summary(rows, labels, h3["tasks"])
        bys[mkey] = by
        comparisons = {}
        for a in summary:
            if a != "full":
                c = paired(by, a, "full")
                if c:
                    comparisons[f"{a} - full"] = c
        budgets = sorted({int(l.split("@")[1]) for l in labels if "@" in l})
        for btok in budgets:
            for x, y in [("ours", "quest"), ("ours", "snapkv"), ("ours", "oracle"), ("oracle", "quest"), ("quest", "snapkv")]:
                c = paired(by, f"{x}@{btok}", f"{y}@{btok}")
                if c:
                    comparisons[f"{x}@{btok} - {y}@{btok}"] = c
        out["models"][mkey] = {"summary": summary, "comparisons": comparisons, "per_problem": per_problem,
                               "runs_git": sorted({json.dumps(r["provenance"]["git"]) for r in rows})}
        print(f"\n== {mkey}")
        print(f"{'run':14s} {'acc':>6s} {'aime24':>7s} {'aime25':>7s} {'gen tok':>8s} {'trunc':>6s} {'acc|stop':>8s}  samples/problem")
        for label, s in summary.items():
            bt = s["acc_by_task"]
            st = s["acc_if_stopped"]
            print(f"{label:14s} {100 * s['acc']:6.1f} {100 * bt.get('aime24', float('nan')):7.1f} "
                  f"{100 * bt.get('aime25', float('nan')):7.1f} {s['mean_gen_tokens']:8.0f} {100 * s['truncated']:5.1f}% "
                  f"{100 * st if st is not None else float('nan'):8.1f}  {s['samples_per_problem']}")
        for k, c in comparisons.items():
            print(f"{k:28s} {c['diff_points']:+6.1f} pts  CI95 [{c['ci95_points'][0]:+.1f}, {c['ci95_points'][1]:+.1f}]  "
                  f"wins {c['wins']} losses {c['losses']} of {c['n_problems']}")
    def complete(s: dict, labels: list[str]) -> bool:
        """Every run has every problem (as many as any run of the model) with all its samples."""
        n = max(v["n_problems"] for v in s.values())
        return all(l in s and s[l]["n_problems"] == n and s[l]["samples_per_problem"][0] >= h3["samples"] for l in labels)

    m35 = out["models"].get("qwen35_4b")
    gc = h3["gate"]
    if m35:
        s, btok = m35["summary"], gc["budget_tokens"]
        base = [f"{m}@{btok}" for m in gc["baselines"] if f"{m}@{btok}" in s]
        if f"ours@{btok}" in s and base:
            diffs = {b: paired(bys["qwen35_4b"], f"ours@{btok}", b) for b in base}
            best = min(base, key=lambda b: diffs[b]["diff_points"])                    # the strongest, paired
            c = diffs[best]
            from h1_screen import H2_BYTES
            h2 = cfg["hybrid"]["h2"]
            share = H2_BYTES[h2["chosen"]](256, cfg["definitions"]["page_size"]) / (2 * 256 * 2)
            done = complete(s, base + [f"ours@{btok}"])
            margin_ok, meta_ok = c["diff_points"] >= gc["margin_points"], share <= h2["metadata_max_share"]
            out["gate"] = {"best_baseline": best, "margin_points": c["diff_points"], "ci95_points": c["ci95_points"],
                           "required_points": gc["margin_points"], "ours_format": h2["chosen"], "metadata_share": share,
                           "max_metadata_share": h2["metadata_max_share"], "complete": done,
                           "passed": bool(done and margin_ok and meta_ok)}
            print(f"\nH3 GATE @{btok}: ours ({h2['chosen']}, metadata {100 * share:.2f}% of KV) - {best} = "
                  f"{c['diff_points']:+.1f} pts (need >= {gc['margin_points']}) -> "
                  f"{'PASS' if out['gate']['passed'] else 'FAIL'}{'' if done else ' (incomplete: not decided)'}")
    hb = h3["hb_gate"]
    lab = f"snapkv@{hb['budget_tokens']}"
    losses = {m: v["comparisons"].get(f"{lab} - full") for m, v in out["models"].items()}
    if all(losses.get(m) for m in ("qwen35_4b", "qwen3_4b")):
        l35, l3 = -losses["qwen35_4b"]["diff_points"], -losses["qwen3_4b"]["diff_points"]
        ratio = l35 / l3 if l3 > 0 else float("inf")
        done = all(complete(out["models"][m]["summary"], ["full", lab]) for m in ("qwen35_4b", "qwen3_4b"))
        out["hb_gate"] = {"loss_points": {"qwen35_4b": l35, "qwen3_4b": l3}, "ratio": ratio, "complete": done,
                          "max_ratio": hb["max_loss_ratio"], "passed": bool(done and ratio <= hb["max_loss_ratio"])}
        print(f"H-b GATE: {lab} loss vs full: qwen35 {l35:.1f} pts, qwen3 {l3:.1f} pts, ratio {ratio:.2f} "
              f"(need <= {hb['max_loss_ratio']}) -> {'PASS' if out['hb_gate']['passed'] else 'FAIL'}"
              f"{'' if done else ' (incomplete: not decided)'}")
    write_json(Path(cfg["paths"]["results_dir"]) / f"h3_{h3['split']}.json",
               {"provenance": provenance(cfg, model=[m["name"] for m in cfg["hybrid"]["models"].values()],
                                         context_length=None, h3=h3), **out})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--model", default="qwen35_4b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--runs", nargs="*", default=None, help="only these run labels")
    p.add_argument("--limit", type=int, default=None, help="first N problems (dry runs)")
    p.add_argument("--max-new", type=int, default=None, help="override h3.max_new_tokens (dry runs)")
    p.add_argument("--batch", type=int, default=None, help="override h3.batch_size")
    p.add_argument("--log-every", type=int, default=0)
    args = p.parse_args()
    cfg = thinking(config_from_args(args))
    if args.limit or args.max_new:            # dry runs never mix with real outputs
        h3 = cfg["hybrid"]["h3"]
        h3["out_dir"], h3["claims_dir"] = h3["out_dir"] + "_dry", h3["claims_dir"] + "_dry"
    if args.merge:
        merge(cfg)
    else:
        work(cfg, args)


if __name__ == "__main__":
    main()
