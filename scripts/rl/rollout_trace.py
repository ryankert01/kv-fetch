"""Record a GRPO rollout trace: real sampled response lengths for P prompts x n samples.

The prompts are MATH problems (config `rollout.prompts`, filtered by level, a seeded sample of `num_prompts`),
chat template applied, sampled n times each at the RL temperature up to `max_tokens`. The trace keeps each
prompt's token ids and each sample's response length and finish reason; scripts/rl/rollout_replay.py replays
it with every request forced to its recorded length, so replays under different memory settings do the same
work. The recording run is itself a rollout step on a roomy engine; its wall time and preemptions are kept.

Usage (venv-vllm32, one GPU): rollout_trace.py CONFIG OUT_DIR --model KEY [--set k=v]
Writes OUT_DIR/trace_<model>.json.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


def load_prompts(rc: dict) -> list[str]:
    pc = rc["prompts"]
    rows = [json.loads(line) for line in open(pc["jsonl"])]
    rows = [r for r in rows if r.get("level") in pc["levels"]]
    random.Random(rc["seed"]).shuffle(rows)
    return [f"{r[pc['field']]}\n{pc['instruction']}" for r in rows[:rc["num_prompts"]]]


def preemptions(llm) -> float:
    return sum(m.value for m in llm.get_metrics() if m.name == "vllm:num_preemptions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc, rc = cfg["profile"], cfg["rollout"]
    mc = pc["models"][args.model]
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=mc["name"], **{**pc["engine"], **mc.get("engine", {}), **rc["trace_engine"]}, disable_log_stats=False)
    tok = llm.get_tokenizer()
    prompts = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
                          add_special_tokens=False) for p in load_prompts(rc)]
    sp = SamplingParams(n=rc["n"], temperature=rc["temperature"], top_p=rc["top_p"], max_tokens=rc["max_tokens"], seed=rc["seed"])
    p0 = preemptions(llm)
    t = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    wall = time.perf_counter() - t
    trace = [{"prompt_token_ids": p, "lengths": [len(c.token_ids) for c in o.outputs],
              "finish": [c.finish_reason for c in o.outputs]} for p, o in zip(prompts, outs)]
    lens = [n for r in trace for n in r["lengths"]]
    capped = sum(f == "length" for r in trace for f in r["finish"])
    summary = {"requests": len(lens), "prompt_tokens_mean": sum(len(p) for p in prompts) / len(prompts),
               "output_tokens": sum(lens), "output_mean": sum(lens) / len(lens), "output_max": max(lens),
               "output_median": sorted(lens)[len(lens) // 2], "capped": capped, "wall_s": wall,
               "output_tok_s": sum(lens) / wall, "preemptions": preemptions(llm) - p0}
    print(json.dumps(summary, indent=1), flush=True)
    cache = llm.llm_engine.vllm_config.cache_config
    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()), settings=rc,
                      vllm=vllm.__version__, load_avg=list(os.getloadavg()))
    json.dump({"provenance": prov, "engine": {"block_size": cache.block_size, "num_gpu_blocks": cache.num_gpu_blocks},
               "summary": summary, "trace": trace}, open(out / f"trace_{args.model}.json", "w"))


if __name__ == "__main__":
    main()
