"""The serving saturation run, offline with the scheduler instrumented: where preemptions come from and what they cost.

Drives an in-process LLMEngine (VLLM_ENABLE_V1_MULTIPROCESSING=0) at a fixed concurrency like `vllm bench serve
--max-concurrency`: `concurrency` requests in flight, a new one added when one finishes, `requests` in all.
Prompts are random token ids, input/output lengths uniform in [L(1 - range_ratio), L(1 + range_ratio)] like the
bench's random dataset, output forced (ignore_eos). Wrapped scheduler calls record, per step:
  tokens scheduled, split into decode, new prefill (positions never computed before) and recompute (positions
  the request had computed before it was preempted), running / waiting counts, free blocks, step wall time;
  every preemption (the victim's computed tokens and blocks held, and what triggered it: the running request
  whose allocate_slots failed, in decode or in a prefill chunk); every readmission of a preempted request
  (computed tokens before preemption, prefix-cache hit on readmission).

Usage (venv-vllm32, one GPU): thrash_trace.py CONFIG OUT_DIR --model KEY --setting NAME [--set k=v]
Writes OUT_DIR/<model>_<setting>.json.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # the scheduler lives in this process
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def lengths(rng: random.Random, mean: int, ratio: float) -> int:
    return rng.randint(max(1, int(mean * (1 - ratio))), int(mean * (1 + ratio)))


class Recorder:
    """Wraps Scheduler.schedule, Scheduler._preempt_request and KVCacheManager.allocate_slots."""

    def __init__(self, sched, num_prompt: dict[str, int]):
        self.s, self.num_prompt = sched, num_prompt
        self.max_seen: dict[str, int] = {}          # highest position computed so far, over all lives
        self.steps: list[list] = []                 # [t, running, waiting, decode, new_prefill, recompute, preempted, free_blocks]
        self.preemptions: list[dict] = []
        self.readmissions: list[dict] = []
        self.trigger: dict | None = None
        self.n_preempted_step = 0
        self.t0 = time.perf_counter()
        orig_schedule, orig_preempt = sched.schedule, sched._preempt_request
        kvm = sched.kv_cache_manager
        orig_alloc = kvm.allocate_slots

        def allocate_slots(request, num_new_tokens, *a, **kw):
            blocks = orig_alloc(request, num_new_tokens, *a, **kw)
            if blocks is None and request.status.name == "RUNNING":
                phase = "decode" if request.num_computed_tokens >= request.num_tokens - 1 else (
                    "recompute" if request.num_preemptions > 0 else "prefill")
                self.trigger = {"req": request.request_id, "phase": phase, "num_new_tokens": num_new_tokens,
                                "computed": request.num_computed_tokens, "free_blocks": kvm.block_pool.get_num_free_blocks()}
            return blocks

        def preempt(request, *a, **kw):
            groups = kvm.coordinator.get_blocks(request.request_id)
            rec = {"step": len(self.steps), "req": request.request_id, "computed": request.num_computed_tokens,
                   "tokens": request.num_tokens, "blocks": [sum(1 for b in g if not b.is_null) for g in groups],
                   "n": request.num_preemptions + 1, "trigger": dict(self.trigger) if self.trigger else None}
            self.preemptions.append(rec)
            self.n_preempted_step += 1
            return orig_preempt(request, *a, **kw)

        def schedule(*a, **kw):
            self.trigger, self.n_preempted_step = None, 0
            was_preempted = {r.request_id for r in sched.waiting if r.status.name == "PREEMPTED"}
            out = orig_schedule(*a, **kw)
            dec = new = rec = 0
            for rid, n in out.num_scheduled_tokens.items():
                r = sched.requests[rid]
                if rid in was_preempted:            # readmitted: computed tokens now = prefix-cache hit + this chunk
                    self.readmissions.append({"step": len(self.steps), "req": rid, "before": self.max_seen.get(rid, 0),
                                              "hit": r.num_computed_tokens - n, "chunk": n, "n": r.num_preemptions})
            for rid, n in out.num_scheduled_tokens.items():
                r = sched.requests[rid]
                end = r.num_computed_tokens
                start = end - n
                mx = self.max_seen.get(rid, 0)
                P = self.num_prompt[rid]
                rc = max(0, min(end, mx) - start)
                nw = max(0, min(end, P) - max(start, mx))
                rec += rc
                new += nw
                dec += n - rc - nw
                self.max_seen[rid] = max(mx, end)
            self.steps.append([round(time.perf_counter() - self.t0, 4), len(sched.running), len(sched.waiting), dec, new, rec,
                               self.n_preempted_step, kvm.block_pool.get_num_free_blocks()])
            return out

        sched.schedule, sched._preempt_request, kvm.allocate_slots = schedule, preempt, allocate_slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, tc = cfg["profile"], cfg["profile"]["thrash"]
    mc = pc["models"][args.model]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    llm = LLM(model=mc["name"], **{**pc["engine"], **mc.get("engine", {}), **tc["engine"], **tc["settings"][args.setting]},
              disable_log_stats=False)
    eng = llm.llm_engine
    sched = eng.engine_core.engine_core.scheduler
    vc = eng.vllm_config
    cache = vc.cache_config
    info = {"block_size": cache.block_size, "num_gpu_blocks": cache.num_gpu_blocks,
            "enable_prefix_caching": cache.enable_prefix_caching, "mamba_cache_mode": getattr(cache, "mamba_cache_mode", None),
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens, "max_num_seqs": vc.scheduler_config.max_num_seqs,
            "watermark": getattr(vc.scheduler_config, "watermark", None), "async_scheduling": vc.scheduler_config.async_scheduling,
            "kv_groups": [type(g.kv_cache_spec).__name__ for g in sched.kv_cache_config.kv_cache_groups]}
    print("engine:", info, flush=True)

    rng = random.Random(tc["seed"])
    lo, hi = pc["prompt_token_range"]
    reqs = []
    for i in range(tc["requests"]):
        il, ol = lengths(rng, tc["input"], tc["range_ratio"]), lengths(rng, tc["output"], tc["range_ratio"])
        reqs.append((f"r{i}", [rng.randrange(lo, hi) for _ in range(il)], ol))
    llm.generate([{"prompt_token_ids": reqs[0][1][:256]}], SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)

    num_prompt = {rid: len(p) for rid, p, _ in reqs}
    recd = Recorder(sched, num_prompt)
    pending = list(reqs)
    in_flight, finished, gen_tokens = 0, {}, 0
    t0 = time.perf_counter()

    def add():
        nonlocal in_flight
        rid, p, ol = pending.pop(0)
        eng.add_request(rid, {"prompt_token_ids": p}, SamplingParams(max_tokens=ol, ignore_eos=True, temperature=0, detokenize=False,
                                                                     output_kind=RequestOutputKind.FINAL_ONLY))
        in_flight += 1

    while pending and in_flight < tc["concurrency"]:
        add()
    while in_flight:
        for o in eng.step():
            if o.finished:
                finished[o.request_id] = time.perf_counter() - t0
                gen_tokens += len(o.outputs[0].token_ids)
                in_flight -= 1
                if pending:
                    add()
    wall = time.perf_counter() - t0

    st = recd.steps
    tot = lambda k: sum(s[k] for s in st)
    trig = Counter((p["trigger"] or {}).get("phase", "none") for p in recd.preemptions)
    per_req = Counter(p["req"] for p in recd.preemptions)
    summary = {"wall_s": wall, "steps": len(st), "ms_per_step": wall / len(st) * 1e3, "output_tokens": gen_tokens,
               "output_tok_s": gen_tokens / wall, "decode_tokens": tot(3), "new_prefill_tokens": tot(4), "recompute_tokens": tot(5),
               "preemptions": len(recd.preemptions), "readmissions": len(recd.readmissions),
               "readmit_before_tokens": sum(r["before"] for r in recd.readmissions),
               "readmit_hit_tokens": sum(r["hit"] for r in recd.readmissions),
               "preempt_triggers": dict(trig), "requests_preempted": len(per_req),
               "preemptions_per_preempted_request": Counter(per_req.values()),
               "victim_computed_mean": sum(p["computed"] for p in recd.preemptions) / max(1, len(recd.preemptions)),
               "victim_blocks_mean": sum(sum(p["blocks"]) for p in recd.preemptions) / max(1, len(recd.preemptions)),
               "mean_running": sum(s[1] for s in st) / len(st), "mean_waiting": sum(s[2] for s in st) / len(st),
               "steps_with_recompute": sum(1 for s in st if s[5] > 0)}
    summary["preemptions_per_preempted_request"] = {str(k): v for k, v in sorted(summary["preemptions_per_preempted_request"].items())}
    print(json.dumps(summary, indent=1), flush=True)
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()), settings=tc,
                      vllm=vllm.__version__, setting=args.setting, load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "engine": info, "summary": summary, "steps_columns":
               ["t", "running", "waiting", "decode", "new_prefill", "recompute", "preempted", "free_blocks"],
               "steps": st, "preemptions": recd.preemptions, "readmissions": recd.readmissions,
               "finished_s": finished}, open(out / f"{args.model}_{args.setting}.json", "w"))


if __name__ == "__main__":
    main()
