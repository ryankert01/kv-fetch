<!-- Draft for github.com/vllm-project/vime. Not posted. Blank issues are off, so use the "Bug Report" form: each "##" below is one form field; paste its text into that field. The Logs field is rendered as code: paste its lines without the ``` fences. -->

**Title:** [Bug] Hybrid-model rollouts keep re-preempting the same requests; `--vllm-watermark 0.05` cuts step time 17%

## Bug Description

When a rollout batch does not fit in the KV cache, hybrid models (Qwen3.5, Qwen3-Next) keep preempting requests right after readmitting them. This happens at vLLM's default watermark of 0, which vime's hybrid recipes use.

Hybrid models cache linear-attention state only at block boundaries: 528 tokens for Qwen3.5-4B, up to 1,056 for larger models. So a readmitted request recomputes almost its whole sequence, gets preempted again, and repeats. Dense models (16-token blocks) lose little to this.

`--vllm-watermark 0.05`, which vime already forwards to vLLM, makes vLLM admit a waiting or preempted request only if 5% of the KV cache stays free. On one vime rollout step it cuts Qwen3.5-4B's step time by 17% and its preemptions by 10x. It changes nothing when the batch fits.

## Steps to Reproduce

One GRPO rollout step in vime, rollout only, on 1 GPU: 64 MATH-500 problems (levels 3-5) x 8 samples, 16K response cap. The prompts file: [prompts.jsonl](https://github.com/ryankert01/kv-fetch/blob/phase0/results/profile/runs/vime_qwen35_4b_wm0/prompts.jsonl).

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
```

Each run takes 70-85 minutes on this GPU. A faster check with vLLM alone (10-25 minutes per run):

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

| vime rollout step, 1x RTX 6000 Ada, one run each | `--vllm-watermark 0` | `--vllm-watermark 0.05` | change |
|---|---|---|---|
| Qwen3.5-4B (hybrid) | 4,755 s, 5,531 preemptions | 3,960 s, 530 preemptions | -17% step time |
| Qwen3-4B (dense) | 3,951 s, 3,319 preemptions | 3,900 s, 1,451 preemptions | -1% step time |

The step time is vime's `perf/rollout_time`. Preemptions come from vLLM's `vllm:num_preemptions_total`. The samples differ between runs: the 0.05 runs produced 2% (hybrid) and 0.5% (dense) more tokens.

The vLLM-only check gives the same picture. Qwen3.5-4B goes from 703 s to 586 s (-17%). Qwen3-4B goes from 1,486 s to 1,502 s (+1%).

## Environment

- vime version: d7e40fe8 (current main)
- Python version: 3.12.11
- PyTorch version: 2.13.0+cu129
- CUDA/ROCm version: CUDA 12.9, driver 565.57.01
- GPU type and count: 1x NVIDIA RTX 6000 Ada (48 GB)
- OS: Ubuntu 22.04.5 LTS
- vLLM version: vime's pin 43b4aaea with `docker/patch/latest/vllm*.patch` (vLLM-only check: 0.32.0)
- vllm-router version: 0.1.16
- Megatron-LM version (if relevant): 4b4acac9 with `megatron.patch` (no training step in rollout-only mode)
- Install: vime's Docker stack rebuilt in a venv, because the image needs CUDA 13 and this driver is 565

## Logs

```
# --vllm-watermark 0, Qwen3.5-4B
perf 0: {... 'perf/rollout_time': 4755.41, 'perf/tokens_per_gpu_per_sec': 1168.03, ...}
vllm:num_preemptions_total 5531
# --vllm-watermark 0.05, Qwen3.5-4B
perf 0: {... 'perf/rollout_time': 3959.93, 'perf/tokens_per_gpu_per_sec': 1428.00, ...}
vllm:num_preemptions_total 530
```

## Additional Context

- I also replayed the same step on vLLM 0.32.0, with identical response lengths at each setting. `--watermark 0.05` cuts the hybrid's step time by 21-29% (at 0.5 and 0.7 memory), and the dense model's by 1.5-4.1%.
  - At watermark 0, a preempted request is preempted 9 times at the median.
  - Recompute is 12.0M tokens, against 5.7M decoded.
  - With any watermark > 0, the hybrid is within 6.4-7.7% of a no-preemption lower bound.
- Larger hybrids show the same loop when serving 64 concurrent requests on one GPU, with `--watermark 0.1`:
  - Qwen3.5-27B-FP8 goes from 1,006 to 46 preemptions, with 28% more output throughput.
  - Qwen3.6-35B-A3B-FP8 goes from 645 to 28, with 10% more.
- vLLM added the watermark in vllm-project/vllm#44594 and benchmarked it only on a dense model.
- I'll send a small PR. It adds `--vllm-watermark 0.05` to `scripts/run-qwen3.5-27B.sh` and `scripts/run-qwen3-next-80B-A3B.sh`, and one line to `usage.md`.
- Scripts and raw results: https://github.com/ryankert01/kv-fetch/tree/phase0/scripts/rl
