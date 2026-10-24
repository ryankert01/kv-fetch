"""E3 pilot (EXPERIMENTS.md): AIME accuracy with page-sparse decoding (kvfetch/sparse_decode.py).

  CUDA_VISIBLE_DEVICES=1 python scripts/run_e3.py configs/reasoning.yaml            # worker: claims units
  python scripts/run_e3.py configs/reasoning.yaml --merge                           # table + gate

A unit is (run, group of e3.sample_group samples of every explore problem); workers claim units in
priority order (e3.runs tiers; within a tier, the first sample group of every run before the second),
so avg@4 of the main comparison lands before avg@8. One JSON per (run, problem, sample) under
e3.out_dir; a worker skips outputs that exist and takes over claims of dead workers on this host.
--merge writes results/e3_<split>.json: avg@k per run and task, generated length, truncation rate,
paired differences (bootstrap over problems) and the E3 pilot gate.
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from kvfetch import sparse_decode as SD
from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, format_prompt, load_model, stop_token_ids
from kvfetch.provenance import provenance, write_json
from kvfetch.tasks.sweep import LOADERS, score

SPLIT_PARITY = {"explore": 0, "confirm": 1}


def thinking(cfg: dict) -> dict:
    kw = dict(cfg["models"].get("chat_template_kwargs", {}) or {}, enable_thinking=True)
    return {**cfg, "models": {**cfg["models"], "chat_template_kwargs": kw}}


def parse_run(label: str, cfg: dict) -> SD.Method:
    name, _, budget = label.partition("@")
    e3 = cfg["e3"]
    variants = e3.get("variants") or {}
    if name in variants:                     # an ablation of ours: the ours engine with another spec
        name, spec = "ours", variants[name]
    else:
        spec = cfg["selectors"]["candidates"][e3["ours"]] if name == "ours" else {}
    return SD.Method(name, int(budget) if budget else None, spec=spec, window=e3["snapkv_window"])


def units(cfg: dict) -> list[tuple[str, int]]:
    e3 = cfg["e3"]
    n_groups = -(-e3["samples"] // e3["sample_group"])
    return [(label, g) for tier in e3["runs"] for g in range(n_groups) for label in tier]


def problems(cfg: dict, tok) -> list:
    """Explore (or confirm) problems of e3.tasks, as the reasoning traces' loaders build them."""
    sw, e3 = cfg["sweep"], cfg["e3"]
    parity = SPLIT_PARITY[e3["split"]]
    out = []
    for task in e3["tasks"]:
        tc = sw["tasks"][task]
        for s in LOADERS[tc.get("loader", task)](tok, tc, 0, tc.get("n", sw["samples_per_task"])):
            if s.index % 2 == parity:
                out.append(s)
    return out


def out_path(cfg: dict, label: str, task: str, index: int, sample: int) -> Path:
    return Path(cfg["e3"]["out_dir"]) / label.replace("@", "_") / task / f"{index:04d}_s{sample}.json"


def claim(path: Path) -> bool:
    """Atomic claim; a claim left by a dead process on this host is taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        info = dict(kv.split("=", 1) for kv in path.read_text().split())
        if info.get("host") != socket.gethostname() or Path(f"/proc/{info.get('pid')}").exists():
            return False
        path.unlink(missing_ok=True)
        return claim(path)
    os.write(fd, f"pid={os.getpid()} host={socket.gethostname()} device={os.environ.get('CUDA_VISIBLE_DEVICES')}\n".encode())
    os.close(fd)
    return True


def work(cfg: dict, args) -> None:
    e3, sw, defs = cfg["e3"], cfg["sweep"], cfg["definitions"]
    SD.register_sparse_attention()
    model, tok = load_model(sw["model"], cfg, args.device, attn_implementation=SD.SPARSE_ATTN)
    stops = stop_token_ids(sw["model"], tok, cfg)
    probs = problems(cfg, tok)[:args.limit]
    prompts = {(s.task, s.index): encode(tok, format_prompt(tok, s.turns[0], "", cfg))["input_ids"] for s in probs}
    max_new = args.max_new or sw["max_new_tokens"]
    max_len = max(map(len, prompts.values())) + max_new + 1
    task_id = {t: i for i, t in enumerate(e3["tasks"])}
    prov = provenance(cfg, model=sw["model"], context_length=max_len, max_new_tokens=max_new, sampling=e3["sampling"])
    for label, g in units(cfg):
        if args.runs and label not in args.runs:
            continue
        if not claim(Path(e3["claims_dir"]) / f"{label}__g{g}.claim"):
            continue
        method = parse_run(label, cfg)
        samples = range(g * e3["sample_group"], min(e3["samples"], (g + 1) * e3["sample_group"]))
        jobs = [SD.Job(key=(s.task, s.index, k), prompt_ids=prompts[(s.task, s.index)],
                       seed=[e3["seed"], task_id[s.task], s.index, k])
                for k in samples for s in probs if not out_path(cfg, label, s.task, s.index, k).exists()]
        if not jobs:
            continue
        refs = {(s.task, s.index): s.answers for s in probs}
        t0, n_tok = time.perf_counter(), [0]
        print(f"[e3] unit {label} samples {list(samples)}: {len(jobs)} jobs", flush=True)

        def done(job, toks, stopped):
            task, index, k = job.key
            text = tok.decode(toks)
            sc = score(task, [text], refs[(task, index)])
            n_tok[0] += len(toks)
            write_json(out_path(cfg, label, task, index, k), {
                "task": task, "index": index, "sample": k, "run": label, "method": method.name,
                "budget_tokens": method.budget_tokens, "prompt_tokens": len(job.prompt_ids), "n_gen": len(toks),
                "stopped": stopped, "score": sc, "answer": refs[(task, index)][0][0], "text": text,
                "provenance": prov})
            dt = time.perf_counter() - t0
            print(f"[e3] {label} {task}/{index} s{k}: score {sc} n_gen {len(toks)} stopped {stopped} | "
                  f"unit {n_tok[0] / dt:.0f} tok/s", flush=True)

        eng = SD.SparseEngine(model, method, batch=args.batch or e3["batch_size"], max_len=max_len,
                              page_size=defs["page_size"], sink_pages=defs["pinned"]["sink_pages"],
                              recent_tokens=defs["pinned"]["recent_tokens"],
                              obs_window=cfg["trace"]["prefill_obs_window"], device=args.device)
        SD.run_batch(model, eng, jobs, stop_ids=stops, max_new_tokens=max_new, sampling=e3["sampling"], on_done=done,
                     log_every=args.log_every)
        del eng
        torch.cuda.empty_cache()
        print(f"[e3] unit {label} g{g} finished in {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------- merge
def bootstrap_ci(x: np.ndarray, n: int, seed: int, level: float = 0.95) -> list[float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [float(np.quantile(means, (1 - level) / 2)), float(np.quantile(means, (1 + level) / 2))]


def merge(cfg: dict) -> None:
    e3 = cfg["e3"]
    rows = [json.loads(p.read_text()) for p in sorted(Path(e3["out_dir"]).glob("*/*/*.json"))]
    if not rows:
        raise SystemExit("no outputs")
    labels = [l for tier in e3["runs"] for l in tier]
    by = {}
    for r in rows:
        by.setdefault(r["run"], {}).setdefault((r["task"], r["index"]), {})[r["sample"]] = r
    probs_all = sorted({k for d in by.values() for k in d})
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
            "acc_by_task": {t: float(np.mean([a for (tt, _), a in acc.items() if tt == t])) for t in e3["tasks"]
                            if any(tt == t for tt, _ in acc)},
            "mean_gen_tokens": float(np.mean([r["n_gen"] for r in flat])),
            "truncated": float(np.mean([not r["stopped"] for r in flat])),
            "acc_if_stopped": float(np.mean([r["score"] for r in flat if r["stopped"]])) if any(r["stopped"] for r in flat) else None,
        }

    def paired(a: str, b: str, n_min: int = 1) -> dict | None:
        """Mean over problems of acc_a - acc_b, on problems where both have >= n_min samples, using
        the samples both have (paired by sample index too)."""
        if a not in by or b not in by:
            return None
        diffs = []
        for p in probs_all:
            ks = sorted(set(by[a].get(p, {})) & set(by[b].get(p, {})))
            if len(ks) >= n_min:
                diffs.append(np.mean([by[a][p][k]["score"] - by[b][p][k]["score"] for k in ks]))
        if not diffs:
            return None
        x = np.array(diffs)
        return {"diff_points": 100 * float(x.mean()), "ci95_points": [100 * v for v in bootstrap_ci(x, 10000, 0)],
                "n_problems": len(x), "wins": int((x > 0).sum()), "losses": int((x < 0).sum())}

    comparisons = {}
    budgets = sorted({int(l.split("@")[1]) for l in labels if "@" in l})
    for btok in budgets:
        for other in ["full"] + [f"{m}@{btok}" for m in ("quest", "snapkv", "oracle")]:
            c = paired(f"ours@{btok}", other)
            if c:
                comparisons[f"ours@{btok} - {other}"] = c
        for v in e3.get("variants") or {}:
            c = paired(f"ours@{btok}", f"{v}@{btok}")
            if c:
                comparisons[f"ours@{btok} - {v}@{btok}"] = c
        for m in ("quest", "snapkv", "oracle", *(e3.get("variants") or {})):
            c = paired(f"{m}@{btok}", "full")
            if c:
                comparisons[f"{m}@{btok} - full"] = c
    gc = e3["gate"]
    btok = gc["budget_tokens"]
    base = [f"{m}@{btok}" for m in gc["baselines"] if f"{m}@{btok}" in summary]
    gate = None
    if f"ours@{btok}" in summary and base:
        best = max(base, key=lambda l: summary[l]["acc"])
        c = paired(f"ours@{btok}", best)
        complete = all(summary[l]["samples_per_problem"][0] >= e3["samples"] for l in base + [f"ours@{btok}"])
        gate = {"best_baseline": best, "margin_points": c["diff_points"], "ci95_points": c["ci95_points"],
                "required_points": gc["margin_points"], "passed": c["diff_points"] >= gc["margin_points"],
                "complete_avg_at": e3["samples"] if complete else None}
    print(f"{'run':14s} {'acc':>6s} {'aime24':>7s} {'aime25':>7s} {'gen tok':>8s} {'trunc':>6s}  samples/problem")
    for label, s in summary.items():
        bt = s["acc_by_task"]
        print(f"{label:14s} {100 * s['acc']:6.1f} {100 * bt.get('aime24', float('nan')):7.1f} "
              f"{100 * bt.get('aime25', float('nan')):7.1f} {s['mean_gen_tokens']:8.0f} {100 * s['truncated']:5.1f}%  "
              f"{s['samples_per_problem']}")
    for k, c in comparisons.items():
        print(f"{k:28s} {c['diff_points']:+6.1f} pts  CI95 [{c['ci95_points'][0]:+.1f}, {c['ci95_points'][1]:+.1f}]  "
              f"wins {c['wins']} losses {c['losses']} of {c['n_problems']}")
    if gate:
        print(f"GATE @{btok}: ours - {gate['best_baseline']} = {gate['margin_points']:+.1f} pts (need >= "
              f"{gate['required_points']}) -> {'PASS' if gate['passed'] else 'FAIL'}"
              f"{'' if gate['complete_avg_at'] else ' (incomplete: not yet avg@' + str(e3['samples']) + ')'}")
    prov = rows[0]["provenance"]
    write_json(Path(cfg["paths"]["results_dir"]) / f"e3_{e3['split']}.json", {
        "provenance": provenance(cfg, model=prov["model"], context_length=prov["context_length"],
                                 runs_git=sorted({json.dumps(r["provenance"]["git"]) for r in rows}),
                                 sampling=e3["sampling"]),
        "summary": summary, "comparisons": comparisons, "gate": gate, "per_problem": per_problem})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--runs", nargs="*", default=None, help="only these run labels")
    p.add_argument("--limit", type=int, default=None, help="first N problems (dry runs)")
    p.add_argument("--max-new", type=int, default=None, help="override sweep.max_new_tokens (dry runs)")
    p.add_argument("--batch", type=int, default=None, help="override e3.batch_size")
    p.add_argument("--log-every", type=int, default=0)
    args = p.parse_args()
    cfg = thinking(config_from_args(args))
    if args.merge:
        merge(cfg)
    else:
        work(cfg, args)


if __name__ == "__main__":
    main()
