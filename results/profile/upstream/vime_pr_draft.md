<!-- Draft PR for github.com/vllm-project/vime. Not opened. Branch example/hybrid-vllm-watermark, 1 commit, 4 lines. -->

**Title:** [Example][Doc] Set `--vllm-watermark 0.05` for hybrid-model rollouts

## What

- Add `--vllm-watermark 0.05` to the two hybrid long-response recipes: `scripts/run-qwen3.5-27B.sh` and `scripts/run-qwen3-next-80B-A3B.sh`.
- Add one line on the flag to the vLLM arguments section of `docs/en/get_started/usage.md` and `docs/zh/get_started/usage.md`.

No code changes. vime already forwards `--vllm-watermark` to vLLM.

## Why

Fixes #{{ISSUE}}. With vLLM's default watermark of 0, a hybrid model under KV pressure keeps preempting the request it just readmitted. Each readmission recomputes almost the whole sequence, because hybrid state is cached only at block boundaries (528+ tokens).

Repro from the issue (`vllm bench throughput`, one GRPO-style batch of 16 prompts x 8 samples):

| 1x RTX 6000 Ada, 128 sequences, one run each | `--watermark 0` | `--watermark 0.05` | change |
|---|---|---|---|
| Qwen3.5-4B (hybrid) | 703 s, 1,577 tok/s | 586 s, 1,892 tok/s | -17% time |
| Qwen3-4B (dense) | 1,486 s, 746 tok/s | 1,502 s, 738 tok/s | +1% time |

On a full GRPO rollout step (Qwen3.5-4B, 64 prompts x 8 samples, 16K cap), 0.05 cuts step time by 21-29%; for the dense Qwen3-4B it changes by 1-4%. Details are in the issue.

The watermark only delays admitting a request when less than 5% of the KV cache would stay free. It changes nothing when the rollout batch fits.

## How I verified

- vLLM 0.32.0 on 1x RTX 6000 Ada: the `vllm bench throughput` repro in the issue, and the rollout-step replay.
- vime's parser: `--vllm-watermark 0.05` gives `vllm_watermark=0.05`, and `watermark` is an `AsyncEngineArgs` field, so `vllm_engine.py` forwards it (default 0.0).
- `bash -n` on both scripts.
- Not run end to end in vime: I don't have the GPUs for these recipes. Happy to run a smaller hybrid recipe if you'd like.

## AI assistance

Measurements and this change were done with Claude Code. I reviewed every changed line and the numbers. The commit carries a `Co-Authored-By` trailer.
