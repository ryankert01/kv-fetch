"""RULER needle + multi-key gate for one (model, length) job, or a shard of all jobs.

  python scripts/run_model_gate.py configs/model_gate.yaml --model Qwen/Qwen3-4B --length 32768 --device cuda:0
  python scripts/run_model_gate.py configs/model_gate.yaml --worker 0/6 --device cuda:0   # shard of all jobs
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, load_model, stop_token_ids
from kvfetch.provenance import provenance, write_json
from kvfetch.tasks.ruler import RulerNiah, string_match_all
from kvfetch.tracing.decode import generate


def job_path(cfg: dict, model: str, length: int) -> Path:
    return Path(cfg["gate"]["out_dir"]) / f"{model.replace('/', '__')}__{length}.json"


def all_jobs(cfg: dict) -> list[tuple[str, int]]:
    g = cfg["gate"]
    jobs = [(m, L) for m in g["candidates"] + g["reference_candidates"] for L in g["lengths"]]
    return sorted(jobs, key=lambda j: -j[1])  # longest first for better packing


def run_job(cfg: dict, model_name: str, length: int, device: str) -> dict:
    g = cfg["gate"]
    model, tok = load_model(model_name, cfg, device)
    max_pos = model.config.max_position_embeddings
    if length > max_pos:
        print(f"[gate] {model_name}: length {length} > max_position_embeddings {max_pos}; running anyway")
    stops = stop_token_ids(model_name, tok, cfg)
    per_task, records = {}, []
    t_start = time.perf_counter()
    for task in g["tasks"]:
        gen = RulerNiah(task, cfg)
        samples = gen.samples(tok, max_seq_length=length, num_samples=g["num_samples"], seed=g["seed"])
        scores = []
        for s in samples:
            ids = torch.tensor(encode(tok, s.prompt)["input_ids"], device=device)
            r = generate(model, ids, max_new_tokens=cfg["ruler"]["tokens_to_generate"], stop_ids=stops,
                         suppress_stop=False, prefill_chunk_size=cfg["models"]["prefill_chunk_size"])
            pred = tok.decode(r.natural_tokens(), skip_special_tokens=True)
            score = string_match_all(pred, s.answers)
            scores.append(score)
            records.append({"task": task, "index": s.index, "prompt_tokens": int(ids.numel()),
                            "answers": s.answers, "pred": pred, "score": score,
                            "prefill_s": r.prefill_s, "decode_s": r.decode_s})
        per_task[task] = 100.0 * sum(scores) / len(scores)
        print(f"[gate] {model_name} @ {length}: {task} = {per_task[task]:.1f}", flush=True)
    groups = {name: sum(per_task[t] for t in ts) / len(ts) for name, ts in g["groups"].items()}
    passed = all(v >= g["pass_threshold"] for v in groups.values())
    out = {
        "provenance": provenance(cfg, model=model_name, context_length=length,
                                 num_samples=g["num_samples"], seed=g["seed"],
                                 max_position_embeddings=max_pos),
        "per_task": per_task, "groups": groups, "pass_threshold": g["pass_threshold"], "passed": passed,
        "wall_s": time.perf_counter() - t_start, "records": records,
    }
    write_json(job_path(cfg, model_name, length), out)
    del model
    torch.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--model")
    p.add_argument("--length", type=int)
    p.add_argument("--worker", help="i/n: run every n-th job starting at i")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-done", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)

    if args.worker:
        i, n = map(int, args.worker.split("/"))
        jobs = all_jobs(cfg)[i::n]
    else:
        jobs = [(args.model, args.length)]
    for m, L in jobs:
        if args.skip_done and job_path(cfg, m, L).exists():
            continue
        run_job(cfg, m, L, args.device)


if __name__ == "__main__":
    main()
