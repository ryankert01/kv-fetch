"""P1 + P2: step-time sweeps and torch-profiler traces of one model on upstream vLLM.

P1 (unprofiled): prefill time vs prompt length at batch 1 (fresh random prompts, so no prefix-cache hit),
and decode step time vs batch x context. A decode point prefills its prompts once, re-sends them with
max_tokens 1 (a cache hit, T_hit) and with max_tokens 1 + N (T_gen); step = (T_gen - T_hit) / N.
P2 (profiled): torch-profiler traces at a few prefill and decode points; scripts/profile/parse_trace.py
turns them into per-kernel-class time.

Usage (venv-vllm32, one GPU): engine_profile.py CONFIG OUT_DIR --model KEY [--skip-sweep] [--skip-profile] [--set k=v]
Writes OUT_DIR/<model>.json and OUT_DIR/traces/<model>/<point>/.
"""
import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kvfetch.config import load_config  # noqa: E402
from kvfetch.provenance import provenance  # noqa: E402


class Prompts:
    def __init__(self, lo: int, hi: int, seed: int):
        self.rng, self.lo, self.hi = random.Random(seed), lo, hi

    def make(self, n: int, length: int) -> list[dict]:
        return [{"prompt_token_ids": [self.rng.randrange(self.lo, self.hi) for _ in range(length)]} for _ in range(n)]


def timed(llm, prompts, sp) -> float:
    t = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--model", required=True)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pc = cfg["profile"]
    mc = pc["models"][args.model]
    out_dir = Path(args.out_dir).resolve()
    trace_root = out_dir / "traces" / args.model
    trace_root.mkdir(parents=True, exist_ok=True)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    llm = LLM(model=mc["name"], **pc["engine"],
              profiler_config={"profiler": "torch", "torch_profiler_dir": str(trace_root / "_tmp"),
                               "torch_profiler_with_stack": False})
    vc = llm.llm_engine.vllm_config
    cache = vc.cache_config
    capacity = cache.num_gpu_blocks * cache.block_size
    engine_info = {"block_size": cache.block_size, "num_gpu_blocks": cache.num_gpu_blocks,
                   "capacity_tokens": capacity, "enable_prefix_caching": cache.enable_prefix_caching,
                   "mamba_cache_mode": getattr(cache, "mamba_cache_mode", None),
                   "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
                   "max_num_seqs": vc.scheduler_config.max_num_seqs,
                   "cudagraph_mode": str(vc.compilation_config.cudagraph_mode),
                   "kv_cache_dtype": cache.cache_dtype}
    print("engine:", engine_info, flush=True)
    P = Prompts(*pc["prompt_token_range"], seed=pc["engine"]["seed"])
    one = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
    llm.generate(P.make(4, 1024), SamplingParams(max_tokens=16, ignore_eos=True, temperature=0), use_tqdm=False)

    res = {"engine": engine_info, "prefill": [], "decode": [], "profiles": []}
    if not args.skip_sweep:
        for L in pc["prefill_lengths"]:
            ts = [timed(llm, P.make(1, L), one) for _ in range(pc["repeats"])]
            res["prefill"].append({"prompt_tokens": L, "seconds": ts, "median_s": statistics.median(ts),
                                   "tokens_per_s": L / statistics.median(ts)})
            print(f"prefill {L}: {statistics.median(ts) * 1e3:.1f} ms", flush=True)
        N = pc["decode_steps"]
        gen = SamplingParams(max_tokens=1 + N, ignore_eos=True, temperature=0)
        for C in pc["decode_contexts"]:
            for b in pc["decode_batches"]:
                if b * (C + N + cache.block_size) > pc["capacity_share"] * capacity:
                    print(f"decode C={C} b={b}: skipped (capacity)", flush=True)
                    continue
                prompts = P.make(b, C)
                t_cold = timed(llm, prompts, one)
                t_hit = timed(llm, prompts, one)
                t_gen = timed(llm, prompts, gen)
                step = (t_gen - t_hit) / N
                res["decode"].append({"context": C, "batch": b, "steps": N, "cold_s": t_cold, "hit_s": t_hit,
                                      "gen_s": t_gen, "step_ms": step * 1e3, "tokens_per_s": b / step})
                print(f"decode C={C} b={b}: step {step * 1e3:.2f} ms ({b / step:.0f} tok/s; cold {t_cold:.2f}s hit {t_hit:.2f}s)", flush=True)

    if not args.skip_profile:
        def profile(name: str, fn):
            tmp = trace_root / "_tmp"
            for f in tmp.glob("*"):
                f.unlink()
            llm.start_profile()
            t = time.perf_counter()
            fn()
            wall = time.perf_counter() - t
            llm.stop_profile()
            dest = trace_root / name
            dest.mkdir(exist_ok=True)
            files = []
            for f in sorted(tmp.glob("*")):
                f.rename(dest / f.name)
                files.append(str((dest / f.name).relative_to(out_dir)))
            res["profiles"].append({"name": name, "wall_s": wall, "files": files})
            print(f"profiled {name}: {wall:.2f}s", flush=True)

        for L in pc["profile_prefill"]:
            prompts = P.make(1, L)
            profile(f"prefill_{L}", lambda: llm.generate(prompts, one, use_tqdm=False))
        n = pc["profile_decode_steps"]
        gen = SamplingParams(max_tokens=1 + n, ignore_eos=True, temperature=0)
        for C, b in pc["profile_decode"]:
            if b * (C + n + cache.block_size) > pc["capacity_share"] * capacity:
                continue
            prompts = P.make(b, C)
            llm.generate(prompts, one, use_tqdm=False)
            profile(f"decode_{C}_b{b}", lambda: llm.generate(prompts, gen, use_tqdm=False))

    prov = provenance(cfg, model=mc["name"], context_length=None, script=str(Path(__file__).resolve()),
                      settings=pc, model_key=args.model, vllm=vllm.__version__,
                      cuda=torch.version.cuda, load_avg=list(__import__("os").getloadavg()),
                      python=platform.python_version())
    json.dump({"provenance": prov, **res}, open(out_dir / f"{args.model}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
