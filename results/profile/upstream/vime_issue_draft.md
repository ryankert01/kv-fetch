<!-- Draft for github.com/vllm-project/vime, "Bug Report" form. Not posted. Each "##" is one form field. -->

**Title:** [Bug] Hybrid models re-preempt the same requests during rollout; `--vllm-watermark 0.05` fixes it

## Bug Description

When a rollout batch does not fit in the KV cache, hybrid models (Qwen3.5, Qwen3-Next) keep preempting the request that was just readmitted. This happens at vLLM's default `--watermark 0`, which vime's hybrid recipes use.

A hybrid model caches its linear-attention state only at block boundaries (528 tokens for Qwen3.5-4B, up to 1,056 for larger ones). So a readmitted request recomputes almost its whole sequence, gets preempted again, and repeats. Dense models (16-token blocks) lose little to this.

`--vllm-watermark 0.05` makes vLLM admit a waiting or preempted request only if 5% of the KV cache stays free. That breaks the loop and changes nothing when the batch fits. vime already forwards the flag to vLLM.

## Steps to Reproduce

vLLM only, since vime passes `--vllm-watermark` straight through:

```bash
vllm bench throughput --model Qwen/Qwen3.5-4B --language-model-only \
  --dataset-name random --random-input-len 512 --random-output-len 8192 --random-range-ratio 0.5 \
  --num-prompts 16 --n 8 --max-model-len 16384 --gpu-memory-utilization 0.5 \
  --max-num-seqs 256 --max-num-batched-tokens 8192 --seed 0 \
  --watermark 0        # then --watermark 0.05; dense control: --model Qwen/Qwen3-4B
```

This is one GRPO-style batch: 16 prompts x 8 samples, 4K-12K output tokens each.

| 1x RTX 6000 Ada, 128 sequences, one run each | `--watermark 0` | `--watermark 0.05` | change |
|---|---|---|---|
| Qwen3.5-4B (hybrid) | 703 s, 1,577 tok/s | 586 s, 1,892 tok/s | -17% time |
| Qwen3-4B (dense) | 1,486 s, 746 tok/s | 1,502 s, 738 tok/s | +1% time |

## Expected Behavior

A preempted request is not readmitted only to be preempted again, and the batch runs about as fast as with a watermark.

## Actual Behavior

See the table above. On a full GRPO rollout step (64 MATH-500 problems x 8 samples, 16K cap, each model's real response lengths replayed exactly), `--watermark 0.05` cuts the hybrid's step time by 21-29%:

| Qwen3.5-4B, 1x RTX 6000 Ada | watermark 0 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|
| step time, 0.5 memory | 4,697 s | 3,688 s | 3,731 s | 3,673 s |
| step time, 0.7 memory | 4,559 s | 3,215 s | 3,320 s | 3,314 s |
| preemptions, 0.5 memory | 5,901 | 567 | 410 | 265 |

- At watermark 0, 93.5% of readmissions get no prefix-cache hit. A preempted request is preempted 9 times at the median, 70 at most. Recompute is 12.0M tokens, against 5.7M decoded.
- The dense Qwen3-4B on the same step gains only 1.5-4.1% from 0.05.
- Run on the dense model's response lengths (same work), the hybrid still gains 22-26%.
- With any watermark > 0, the hybrid is within 6.4-7.7% of a perfect-packing lower bound. A new scheduling policy would not gain much more.

## Environment

- vime version: not used. Measured on vLLM directly; vime forwards `--vllm-watermark` unchanged (checked against `vime/backends/vllm_utils/arguments.py` at d7e40fe8).
- Python version: 3.12.11
- PyTorch version: 2.13.0+cu129
- CUDA/ROCm version: CUDA 12.9, driver 565.57.01
- GPU type and count: 1x NVIDIA RTX 6000 Ada (48 GB)
- OS: Ubuntu 22.04.5 LTS
- vLLM version: 0.32.0
- vllm-router version: n/a
- Megatron-LM version (if relevant): n/a

## Additional Context

- The same loop appears when serving larger hybrids at 64 concurrent requests on one GPU. Qwen3.5-27B-FP8 goes from 1,006 to 46 preemptions and gains 28% output throughput with `--watermark 0.1`. Qwen3.6-35B-A3B-FP8 goes from 645 to 28 and gains 10%.
- vLLM added the watermark in vllm-project/vllm#44594 and benchmarked it on a dense model only (+5% at 0.05).
- I'll open a small PR that adds the flag to `scripts/run-qwen3.5-27B.sh` and `scripts/run-qwen3-next-80B-A3B.sh` and documents it in `usage.md`.
- Scripts and raw results: https://github.com/ryankert01/kv-fetch/tree/phase0/scripts/rl (rollout replay and bound)
