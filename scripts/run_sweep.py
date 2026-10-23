"""Revival sweep: trace natural (multi-turn where the task has turns) decoding on several
datasets. Work-queue workers, one per GPU, claim (task, shard) jobs.

  python scripts/run_sweep.py configs/sweep.yaml --claim --device cuda:0 [--tasks math500 qmsum]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from kvfetch.config import add_config_args, config_from_args
from kvfetch.models import encode, followup_turns, format_prompt, load_model, stop_token_ids
from kvfetch.provenance import provenance, write_json
from kvfetch.tasks.sweep import LOADERS, score
from kvfetch.tracing.capture import capture, trace_extras, trace_meta
from kvfetch.tracing.schema import save_trace


def with_thinking(cfg: dict, think: bool) -> dict:
    kw = dict(cfg["models"].get("chat_template_kwargs", {}) or {}, enable_thinking=think)
    return {**cfg, "models": {**cfg["models"], "chat_template_kwargs": kw}}


def claim(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"pid={os.getpid()} device={os.environ.get('CUDA_VISIBLE_DEVICES')}\n".encode())
    os.close(fd)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--claim", action="store_true", help="work-queue mode across workers")
    p.add_argument("--limit", type=int, default=None, help="samples per task (dry runs)")
    args = p.parse_args()
    cfg = config_from_args(args)
    sw = cfg["sweep"]
    tasks = args.tasks or list(sw["tasks"])
    n_per = args.limit or sw["samples_per_task"]
    budget = sw["max_context"] - sw["max_new_tokens"] - sw["context_margin"]

    model, tok = load_model(sw["model"], cfg, args.device, traced=True)
    stops = stop_token_ids(sw["model"], tok, cfg)
    prov = provenance(cfg, model=sw["model"], context_length=sw["max_context"], traced_layers=sw["layers"],
                      max_new_tokens=sw["max_new_tokens"])
    trace_dir, rec_dir = Path(sw["trace_dir"]), Path(sw["records_dir"])
    samples_cache: dict = {}
    jobs = [(t, k) for t in tasks for k in range(math.ceil(n_per / sw["shard_size"]))]

    while True:
        job = next((j for j in jobs if (not args.claim)
                    or claim(Path(sw["claims_dir"]) / f"{j[0]}__{j[1]}.claim")), None)
        if job is None:
            break
        jobs.remove(job)
        task, shard = job
        tc = sw["tasks"][task]
        if task not in samples_cache:
            samples_cache[task] = LOADERS[task](tok, tc, budget, n_per)
        for s in samples_cache[task][shard * sw["shard_size"]:(shard + 1) * sw["shard_size"]]:
            rec_path = rec_dir / task / f"{s.index:04d}.json"
            if rec_path.exists():
                continue
            cfg_s = with_thinking(cfg, s.think)
            ids_list = encode(tok, format_prompt(tok, s.turns[0], "", cfg_s))["input_ids"]
            if len(ids_list) > sw["max_context"] - sw["max_new_tokens"]:
                print(f"[sweep] skip {task}/{s.index}: prompt {len(ids_list)} tokens", flush=True)
                continue
            ids = torch.tensor(ids_list, device=args.device)
            fu, eot = followup_turns(tok, s.turns[0], s.turns[1:], cfg_s) if len(s.turns) > 1 else (None, None)
            t0 = time.perf_counter()
            res, tr = capture(model, ids, layers=sw["layers"], cfg=cfg, decode_steps=sw["max_new_tokens"],
                              stop_ids=stops, suppress_stop=False, followups=fu, eot_ids=eot,
                              turn_max_new=tc.get("turn_max_new"))
            texts = [tok.decode(res.turn_tokens(i)) for i in range(len(res.turns))]
            sc = score(task, texts, s.answers)
            split = "explore" if s.index % 2 == 0 else "confirm"
            meta = trace_meta(prov=prov, tracer=tr, res=res, task=task, sample_id=f"{task}-{s.index}", split=split,
                              think=s.think, score=sc, n_user_turns=len(s.turns), **s.extra)
            n = len(res.tokens)
            save_trace(trace_dir / task / f"{s.index:04d}.safetensors", meta=meta, mass=tr.mass[:n],
                       obs_mass=tr.obs_numpy(), tokens=np.array(res.tokens), prompt_ids=ids.cpu().numpy(),
                       fmt=cfg["trace"]["format"], extra=trace_extras(tr, res))
            write_json(rec_path, {"task": task, "index": s.index, "split": split, "think": s.think,
                                  "prompt_tokens": len(ids_list), "n_steps": n, "n_fed": int(sum(res.kinds)),
                                  "turns": res.turns, "score": sc, "answers": s.answers,
                                  "turn_texts": [t[:3000] for t in texts], "extra": s.extra,
                                  "prefill_s": res.prefill_s, "decode_s": res.decode_s,
                                  "wall_s": time.perf_counter() - t0, "git": prov["git"]})
            print(f"[sweep] {task}/{s.index} ({split}) prompt={len(ids_list)} steps={n} fed={sum(res.kinds)} "
                  f"turns={len(res.turns)} score={sc} {time.perf_counter() - t0:.0f}s", flush=True)
            del tr
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
