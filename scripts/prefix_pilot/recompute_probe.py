"""How much does a hybrid prefix-cache hit pay for landing between GDN checkpoints?

vLLM (align mode) keeps a GDN checkpoint every block_size tokens, so a request whose shared prefix ends
r tokens past a block boundary recomputes those r tokens. This probe caches one long prompt, then sends
prefixes of it ending r tokens past a boundary and times prefill (max_tokens=1). It also times a cold
prefill of the same length for contrast. Output: JSON with provenance.
"""
import json
import socket
import subprocess
import sys
import time

import numpy as np

SETTINGS = {
    "model": "/share2/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "max_model_len": 40960,
    "gpu_memory_utilization": 0.85,
    "contexts": [8192, 32768],          # approximate cached-prefix lengths
    "offsets": [1, 32, 128, 264, 527],  # tokens past the last checkpoint
    "repeats": 3,
    "token_range": [1000, 100000],      # random ordinary token ids
    "seed": 0,
}


def main():
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    out_path = sys.argv[1]
    s = SETTINGS
    llm = LLM(model=s["model"], max_model_len=s["max_model_len"], gpu_memory_utilization=s["gpu_memory_utilization"],
              enable_prefix_caching=True, language_model_only=True, max_num_seqs=4, disable_log_stats=True)
    bs = llm.llm_engine.vllm_config.cache_config.block_size
    rng = np.random.default_rng(s["seed"])
    base = rng.integers(*s["token_range"], size=s["max_model_len"] - 64).tolist()
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    def prefill(ids):
        torch.cuda.synchronize()
        t = time.perf_counter()
        o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e3, o.num_cached_tokens

    prefill(base[:2048])                                   # warm-up (compile paths)
    warm_ms, _ = prefill(base)                             # cache the whole prompt: checkpoints every bs tokens
    rows = []
    for ctx in s["contexts"]:
        k = ctx // bs
        for rep in range(s["repeats"]):
            for r in s["offsets"]:
                k += 1                                     # a fresh boundary each probe: no reuse of probe tails
                L = k * bs + r
                ms, cached = prefill(base[:L])
                rows.append({"context": ctx, "offset": r, "length": L, "ms": ms, "cached_tokens": cached,
                             "recomputed": L - cached, "repeat": rep})
                print(rows[-1], flush=True)
        cold = rng.integers(*s["token_range"], size=k * bs).tolist()
        ms, cached = prefill(cold)
        rows.append({"context": ctx, "offset": "cold", "length": len(cold), "ms": ms, "cached_tokens": cached,
                     "recomputed": len(cold) - cached})
        print(rows[-1], flush=True)
    prov = {"settings": s, "block_size": bs, "warm_prefill_ms": warm_ms, "vllm": vllm.__version__,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "host": socket.gethostname(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "replayssm_commit": subprocess.run(["git", "-C", sys.argv[2], "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip()}
    json.dump({"provenance": prov, "rows": rows}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
