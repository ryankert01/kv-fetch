"""Replay a GRPO rollout trace under a memory budget: one synchronous rollout step, scheduler instrumented.

All P x n requests of the trace (scripts/rl/rollout_trace.py) are added at once, as an RL framework hands a
rollout batch to the engine, each forced to its recorded length (ignore_eos, max_tokens = length), sampled at
the RL temperature. The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0) with the setting's
memory share and admission knobs (config `rollout.replay.settings`). The night-1 recorder
(scripts/profile/thrash_trace.py) logs every step, preemption and readmission; this script adds, per step,
the total context of the running requests (for the step-time fit in rollout_bound.py) and each request's
finish time. The makespan is the rollout step's time.

Usage (venv-vllm32, one GPU): rollout_replay.py CONFIG OUT_DIR --trace TRACE.json --model KEY --setting NAME [--set k=v]
Writes OUT_DIR/<model>_<setting>.json.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # the scheduler lives in this process
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profile"))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402
from thrash_trace import Recorder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, rc = cfg["profile"], cfg["rollout"]
    mc = pc["models"][args.model]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    trace = json.load(open(args.trace))

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    llm = LLM(model=mc["name"], **{**pc["engine"], **mc.get("engine", {}), **rc["replay"]["engine"],
                                    **rc["replay"]["settings"][args.setting]}, disable_log_stats=False)
    eng = llm.llm_engine
    sched = eng.engine_core.engine_core.scheduler
    vc = eng.vllm_config
    cache = vc.cache_config
    groups = [type(g.kv_cache_spec).__name__ for g in sched.kv_cache_config.kv_cache_groups]
    info = {"block_size": cache.block_size, "num_gpu_blocks": cache.num_gpu_blocks,
            "enable_prefix_caching": cache.enable_prefix_caching, "mamba_cache_mode": getattr(cache, "mamba_cache_mode", None),
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens, "max_num_seqs": vc.scheduler_config.max_num_seqs,
            "watermark": getattr(vc.scheduler_config, "watermark", None), "async_scheduling": vc.scheduler_config.async_scheduling,
            "kv_groups": groups, "state_blocks_per_request": sum(g == "MambaSpec" for g in groups)}
    print("engine:", info, flush=True)
    llm.generate([{"prompt_token_ids": trace["trace"][0]["prompt_token_ids"]}], SamplingParams(max_tokens=8, ignore_eos=True),
                 use_tqdm=False)

    rec = Recorder(sched)
    ctx: list[int] = []                                 # per step: total computed tokens of the running requests
    inner = sched.schedule

    def schedule(*a, **kw):
        o = inner(*a, **kw)
        ctx.append(sum(r.num_computed_tokens for r in sched.running))
        return o
    sched.schedule = schedule

    reqs = [(f"p{i}s{j}", r["prompt_token_ids"], max(1, L)) for i, r in enumerate(trace["trace"]) for j, L in enumerate(r["lengths"])]
    t0 = time.perf_counter()
    for rid, p, L in reqs:
        eng.add_request(rid, {"prompt_token_ids": p},
                        SamplingParams(max_tokens=L, ignore_eos=True, temperature=rc["temperature"], top_p=rc["top_p"],
                                       detokenize=False, output_kind=RequestOutputKind.FINAL_ONLY))
    finished, gen = {}, 0
    while len(finished) < len(reqs):
        for o in eng.step():
            if o.finished:
                finished[o.request_id] = time.perf_counter() - t0
                gen += len(o.outputs[0].token_ids)
    wall = time.perf_counter() - t0

    st = rec.steps
    for row, c in zip(st, ctx):
        row.append(c)
    tot = lambda k: sum(s[k] for s in st)
    per_req = Counter(p["req"] for p in rec.preemptions)
    summary = {"requests": len(reqs), "makespan_s": wall, "steps": len(st), "ms_per_step": wall / len(st) * 1e3,
               "output_tokens": gen, "output_tok_s": gen / wall, "decode_tokens": tot(3), "new_prefill_tokens": tot(4),
               "recompute_tokens": tot(5), "preemptions": len(rec.preemptions), "requests_preempted": len(per_req),
               "readmissions": len(rec.readmissions),
               "preempt_triggers": dict(Counter((p["trigger"] or {}).get("phase", "none") for p in rec.preemptions)),
               "mean_running": sum(s[1] for s in st) / len(st), "max_running": max(s[1] for s in st),
               "mean_waiting": sum(s[2] for s in st) / len(st), "steps_with_recompute": sum(1 for s in st if s[5] > 0),
               "last_finish_s": max(finished.values()), "median_finish_s": sorted(finished.values())[len(finished) // 2]}
    print(json.dumps(summary, indent=1), flush=True)
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()), settings=rc,
                      vllm=vllm.__version__, setting=args.setting, trace=str(Path(args.trace).resolve()),
                      load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "engine": info, "summary": summary, "steps_columns":
               ["t", "running", "waiting", "decode", "new_prefill", "recompute", "preempted", "free_blocks", "context"],
               "steps": st, "preemptions": rec.preemptions, "readmissions": rec.readmissions, "finished_s": finished},
              open(out / f"{args.model}_{args.setting}.json", "w"))


if __name__ == "__main__":
    main()
