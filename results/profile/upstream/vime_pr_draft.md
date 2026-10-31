<!-- Draft PR for github.com/vllm-project/vime. Not opened. Branch example/hybrid-vllm-watermark, 1 commit, 4 lines. -->

**Title:** [Example][Doc] Set `--vllm-watermark 0.05` for hybrid-model rollouts

## What

- Add `--vllm-watermark 0.05` to the two hybrid long-response recipes: `scripts/run-qwen3.5-27B.sh` and `scripts/run-qwen3-next-80B-A3B.sh`.
- Add one line on the flag to the vLLM arguments section of `docs/en/get_started/usage.md` and `docs/zh/get_started/usage.md`.

No code changes. vime already forwards `--vllm-watermark` to vLLM.

If you'd rather make 0.05 vime's default, it is one line, `parser.set_defaults(vllm_watermark=0.05)` in `vime/backends/vllm_utils/arguments.py`, and I'll switch this PR to that.

## Why

Fixes #508. With vLLM's default watermark of 0, a hybrid model under KV pressure keeps preempting the request it just readmitted. Each readmission recomputes almost the whole sequence, because hybrid state is cached only at block boundaries (528+ tokens).

The recipes' batch as shipped (8 prompts x 8 samples) fits in the KV cache, so this changes nothing for it. It is a safe setting for users who scale the batch up, as vime's other recipes do (32-128 prompts per step).

One GRPO rollout step in vime (`train.py --debug-rollout-only`, 64 MATH-500 problems x 8 samples, 16K cap, `--vllm-gpu-memory-utilization 0.5`, RTX 6000 Ada, one run each):

| | `--vllm-watermark 0` | `--vllm-watermark 0.05` | step time |
|---|---|---|---|
| Qwen3.5-4B (hybrid), 1 GPU | 4,755 s, 5,531 preemptions | 3,960 s, 530 preemptions | -17% |
| Qwen3.5-4B + MTP, 1 GPU | 4,327 s, 2,589 preemptions | 4,275 s, 176 preemptions | -1% |
| Qwen3.5-4B + MTP, TP2 | 2,240 s, 1,797 preemptions | 2,090 s, 72 preemptions | -7% |
| Qwen3-4B (dense), 1 GPU | 3,951 s, 3,319 preemptions | 3,900 s, 1,451 preemptions | -1% |

MTP uses 4 speculative tokens, as the recipes do. Sampling is real, so the paired runs differ in output tokens by -1.4% to +1.8%; the 1% rows are within that noise.

The watermark only delays admitting a request when less than 5% of the KV cache would stay free. It was never slower in these runs.

## How I verified

- The vime runs above: this branch on d7e40fe8, vime's pinned vLLM (43b4aaea) with all of `docker/patch/latest/vllm*.patch`, Megatron-LM at the Dockerfile pin. vLLM's `/server_info` reports the scheduler watermark (0.05 with the flag, 0 without) and the tensor-parallel size, so both reach the engine.
- Rollout only, outside Docker: the change doesn't touch training, the image needs CUDA 13 (this machine's driver is 565), and the training stack targets sm90+.
- On vLLM 0.32.0 directly: the repro in the issue, and a rollout-step replay with identical response lengths at each setting (hybrid -21 to -29% step time, dense -1 to -4%).
- `bash -n` on both scripts. I did not run the 27B and 80B recipes themselves (no H100s).

## AI assistance

Measurements and this change were done with Claude Code. I reviewed every changed line and the numbers. The commit carries a `Co-Authored-By` trailer.
