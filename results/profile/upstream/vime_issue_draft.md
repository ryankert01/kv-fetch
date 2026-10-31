<!-- Posted as https://github.com/vllm-project/vime/issues/508. Edited 2026-10-31 to this text: retitled [Perf] (not a bug), shortened, no links to kv-fetch. -->

**Title:** [Perf] Default --vllm-watermark 0.05 for rollouts (hybrid models re-preempt under KV pressure)

**Proposal:** default `--vllm-watermark` to 0.05 for rollouts. One line in `vime/backends/vllm_utils/arguments.py`, next to the router defaults:

```python
parser.set_defaults(vllm_watermark=0.05)
```

**Why:** once a rollout batch outgrows the KV cache, vLLM preempts the newest running request. At its default watermark of 0, hybrid models (Qwen3.5, Qwen3-Next) often readmit a request and evict it again within a few steps, because their large blocks (528+ tokens) make free memory run out in big jumps. Each readmission recomputes most of the sequence. With 0.05, vLLM readmits only when 5% of the cache stays free.

vLLM keeps 0 because the margin delays new requests (vllm-project/vllm#44594: TTFT p50 6.6 s → 11.2 s). A rollout step ends with its last sample, so TTFT doesn't matter there.

**Results:** one GRPO rollout step in vime (d7e40fe8, `--debug-rollout-only`), 64 MATH-500 problems × 8 samples, 16K cap, `--vllm-gpu-memory-utilization 0.5`, RTX 6000 Ada, one run each:

| | watermark 0 | watermark 0.05 | step time |
|---|---|---|---|
| Qwen3.5-4B, 1 GPU | 4,755 s, 5,531 preemptions | 3,960 s, 530 | −17% |
| Qwen3.5-4B + MTP, 1 GPU | 4,327 s, 2,589 | 4,275 s, 176 | −1% |
| Qwen3.5-4B + MTP, TP2 | 2,240 s, 1,797 | 2,090 s, 72 | −7% |
| Qwen3-4B (dense), 1 GPU | 3,951 s, 3,319 | 3,900 s, 1,451 | −1% |

It was never slower. The shipped hybrid recipes (8 prompts × 8 samples) fit in the KV cache, so they are unaffected; it matters once users scale the batch up.

**Repro with vLLM alone** (0.32.0, about 15 min per run; Qwen3.5-4B goes from 703 s to 586 s):

```bash
vllm bench throughput --model Qwen/Qwen3.5-4B --language-model-only \
  --dataset-name random --random-input-len 512 --random-output-len 8192 --random-range-ratio 0.5 \
  --num-prompts 16 --n 8 --max-model-len 16384 --gpu-memory-utilization 0.5 \
  --max-num-seqs 256 --max-num-batched-tokens 8192 --seed 0 --watermark 0   # then 0.05
```

I can send the PR: either the one-line default, or the flag in the two hybrid recipes plus a line in `usage.md`.
