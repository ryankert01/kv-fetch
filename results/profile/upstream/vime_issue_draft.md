<!-- Posted 2026-10-31 as https://github.com/vllm-project/vime/issues/508 (the form fields as ### headings, checklist ticked). Kept as the source text. -->

**Title:** [Bug] Hybrid-model rollouts keep re-preempting requests once the batch outgrows the KV cache; proposal: default `--vllm-watermark 0.05`

## Bug Description

When a rollout batch does not fit in the KV cache, hybrid models (Qwen3.5, Qwen3-Next) keep preempting requests right after readmitting them. This happens at vLLM's default watermark of 0, which vime uses.

vime's hybrid recipes as shipped (8 prompts x 8 samples) fit in the KV cache, so they are not affected. The problem shows once samples per engine outgrow it: the 32-128 prompts per step of vime's other recipes, longer responses, or a smaller memory share.

Hybrid models cache linear-attention state only at block boundaries: 528 tokens for Qwen3.5-4B, up to 1,056 for larger models. So a readmitted request recomputes almost its whole sequence, gets preempted again, and repeats. Dense models (16-token blocks) lose little to this.

`--vllm-watermark 0.05`, which vime already forwards to vLLM, makes vLLM admit a waiting or preempted request only if 5% of the KV cache stays free. In four vime rollout-step comparisons it was never slower and cut preemptions 2-25x. Step time fell 17% for Qwen3.5-4B with plain decoding, and 1-7% with MTP speculative decoding as in the recipes.

**Proposal:** make 0.05 vime's default. vLLM keeps 0 because the watermark delays new requests: in vllm-project/vllm#44594, TTFT p50 went from 6.6 s to 11.2 s at 0.05. A rollout step ends only when its last sample does, so TTFT doesn't matter there. It is one line in `add_vllm_arguments` (`vime/backends/vllm_utils/arguments.py`), next to the existing router defaults:

```python
parser.set_defaults(vllm_watermark=0.05)
```

I checked that this parses to 0.05 without the flag and to 0.0 with `--vllm-watermark 0`.

## Steps to Reproduce

One GRPO rollout step in vime, rollout only: 64 MATH-500 problems (levels 3-5) x 8 samples, 16K response cap. The prompts file: [prompts.jsonl](https://github.com/ryankert01/kv-fetch/blob/phase0/results/profile/runs/vime_qwen35_4b_wm0/prompts.jsonl).

With a Ray head running, as in `scripts/run-qwen3-4B.sh`:

```bash
source scripts/models/qwen3.5-4B.sh      # dense control: scripts/models/qwen3-4B.sh and Qwen3-4B
python3 train.py --debug-rollout-only \
  --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --num-gpus-per-node 1 \
  --hf-checkpoint /root/Qwen3.5-4B ${MODEL_ARGS[@]} \
  --prompt-data prompts.jsonl --input-key prompt --label-key label --apply-chat-template --rm-type deepscaler \
  --num-rollout 1 --rollout-batch-size 64 --n-samples-per-prompt 8 --global-batch-size 512 \
  --rollout-max-response-len 16384 --rollout-temperature 1.0 --rollout-top-p 1.0 \
  --vllm-gpu-memory-utilization 0.5 \
  --vllm-watermark 0                     # then 0.05
# MTP rows: add --vllm-speculative-config '{"method": "mtp", "num_speculative_tokens": 4}'
# TP2 row: set the three GPU counts to 2
```

Each 1-GPU run takes 70-85 minutes on this GPU. A faster check with vLLM alone (10-25 minutes per run):

```bash
vllm bench throughput --model Qwen/Qwen3.5-4B --language-model-only \
  --dataset-name random --random-input-len 512 --random-output-len 8192 --random-range-ratio 0.5 \
  --num-prompts 16 --n 8 --max-model-len 16384 --gpu-memory-utilization 0.5 \
  --max-num-seqs 256 --max-num-batched-tokens 8192 --seed 0 \
  --watermark 0                          # then 0.05
```

## Expected Behavior

A preempted request is not readmitted only to be preempted again, and the step takes about as long as it does with a watermark.

## Actual Behavior

| vime rollout step, RTX 6000 Ada, one run each | `--vllm-watermark 0` | `--vllm-watermark 0.05` | step time |
|---|---|---|---|
| Qwen3.5-4B (hybrid), 1 GPU | 4,755 s, 5,531 preemptions | 3,960 s, 530 preemptions | -17% |
| Qwen3.5-4B + MTP, 1 GPU | 4,327 s, 2,589 preemptions | 4,275 s, 176 preemptions | -1% |
| Qwen3.5-4B + MTP, TP2 | 2,240 s, 1,797 preemptions | 2,090 s, 72 preemptions | -7% |
| Qwen3-4B (dense), 1 GPU | 3,951 s, 3,319 preemptions | 3,900 s, 1,451 preemptions | -1% |

The step time is vime's `perf/rollout_time`. Preemptions come from vLLM's `vllm:num_preemptions_total`. Samples differ between runs: output tokens vary by -1.4% to +1.8% between the paired runs, so the 1% rows are within sampling noise.

The vLLM-only check gives the same picture. Qwen3.5-4B goes from 703 s to 586 s (-17%). Qwen3-4B goes from 1,486 s to 1,502 s (+1%).

## Environment

- vime version: d7e40fe8 (current main)
- Python version: 3.12.11
- PyTorch version: 2.13.0+cu129
- CUDA/ROCm version: CUDA 12.9, driver 565.57.01
- GPU type and count: NVIDIA RTX 6000 Ada (48 GB), 1 per engine (2 for TP2)
- OS: Ubuntu 22.04.5 LTS
- vLLM version: vime's pin 43b4aaea with `docker/patch/latest/vllm*.patch` (vLLM-only check: 0.32.0)
- vllm-router version: 0.1.16
- Megatron-LM version (if relevant): 4b4acac9 with `megatron.patch` (no training step in rollout-only mode)
- Install: vime's Docker stack rebuilt in a venv, because the image needs CUDA 13 and this driver is 565

## Logs

```
# Qwen3.5-4B, 1 GPU, --vllm-watermark 0
perf 0: {... 'perf/rollout_time': 4755.41, 'perf/tokens_per_gpu_per_sec': 1168.03, ...}
vllm:num_preemptions_total 5531
# Qwen3.5-4B, 1 GPU, --vllm-watermark 0.05
perf 0: {... 'perf/rollout_time': 3959.93, 'perf/tokens_per_gpu_per_sec': 1428.00, ...}
vllm:num_preemptions_total 530
# Qwen3.5-4B + MTP, TP2, --vllm-watermark 0
perf 0: {... 'perf/rollout_time': 2240.14, 'perf/tokens_per_gpu_per_sec': 1263.78, ...}
vllm:num_preemptions_total 1797
# Qwen3.5-4B + MTP, TP2, --vllm-watermark 0.05
perf 0: {... 'perf/rollout_time': 2089.93, 'perf/tokens_per_gpu_per_sec': 1336.16, ...}
vllm:num_preemptions_total 72
```

## Additional Context

- I also replayed the step on vLLM 0.32.0, with identical response lengths at each setting. `--watermark 0.05` cuts the hybrid's step time by 21-29% (at 0.5 and 0.7 memory) and the dense model's by 1.5-4.1%. At watermark 0, a preempted request is preempted 9 times at the median, and recompute is 12.0M tokens against 5.7M decoded.
- Larger hybrids show the same loop when serving 64 concurrent requests on one GPU, with `--watermark 0.1`: Qwen3.5-27B-FP8 goes from 1,006 to 46 preemptions (+28% output throughput), Qwen3.6-35B-A3B-FP8 from 645 to 28 (+10%).
- With MTP the engine runs about half as many requests at the same KV usage (22 vs 47 while requests wait), because vLLM reserves 4 extra state blocks per linear-attention layer group for each request. That may be why the MTP rows gain less.
- I can send a PR either way: the one-line default above, or `--vllm-watermark 0.05` in `scripts/run-qwen3.5-27B.sh` and `scripts/run-qwen3-next-80B-A3B.sh` plus one line in `usage.md`. Which do you prefer?
- Scripts and raw results: https://github.com/ryankert01/kv-fetch/tree/phase0/scripts/rl
