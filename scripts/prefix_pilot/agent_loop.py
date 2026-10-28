"""How much does an agent loop recompute under vLLM's checkpoint retention?

Each turn's prompt is the previous prompt + its generated output + a "tool result", as in a tool-calling
loop that keeps the reasoning of the current turn. Per turn: a max_tokens=1 call times the prefill and
reports cached tokens, then the same prompt is generated (greedy, ignore_eos) to build the next turn.
Retention is set by VLLM_PREFIX_CACHE_RETENTION_INTERVAL (unset = dense, the fork's default; 0 = only the
prompt replay boundary and shared-prefix junctions, upstream main's default since its config default is 0).
Usage: agent_loop.py OUT.json REPO_DIR
"""
import json, os, socket, subprocess, sys, time

import numpy as np

SETTINGS = {
    "model": "/share2/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "max_model_len": 40960,
    "gpu_memory_utilization": 0.85,
    "first_prompt_tokens": 8000,
    "turns": 5,
    "output_tokens": 2048,
    "tool_tokens": 300,
    "token_range": [1000, 100000],
    "seed": 0,
}


def main():
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    s = SETTINGS
    llm = LLM(model=s["model"], max_model_len=s["max_model_len"], gpu_memory_utilization=s["gpu_memory_utilization"],
              enable_prefix_caching=True, language_model_only=True, max_num_seqs=4, disable_log_stats=True)
    bs = llm.llm_engine.vllm_config.cache_config.block_size
    rng = np.random.default_rng(s["seed"])
    one = SamplingParams(temperature=0.0, max_tokens=1)
    gen = SamplingParams(temperature=0.0, max_tokens=s["output_tokens"], ignore_eos=True)

    def run(ids, sp):
        torch.cuda.synchronize(); t = time.perf_counter()
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e3, o

    run(rng.integers(*s["token_range"], size=2048).tolist(), one)        # warm-up
    ids = rng.integers(*s["token_range"], size=s["first_prompt_tokens"]).tolist()
    rows = []
    for turn in range(s["turns"]):
        ms, o = run(ids, one)
        _, g = run(ids, gen)
        rows.append({"turn": turn, "prompt_tokens": len(ids), "cached_tokens": o.num_cached_tokens,
                     "recomputed": len(ids) - o.num_cached_tokens, "prefill_ms": ms})
        print(rows[-1], flush=True)
        ids = ids + list(g.outputs[0].token_ids) + rng.integers(*s["token_range"], size=s["tool_tokens"]).tolist()
    prov = {"settings": s, "block_size": bs, "retention_env": os.environ.get("VLLM_PREFIX_CACHE_RETENTION_INTERVAL"),
            "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "host": socket.gethostname(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "replayssm_commit": subprocess.run(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip()}
    json.dump({"provenance": prov, "rows": rows}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
