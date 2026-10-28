# Profiling notebook (agent-written; newest last)

Harness: scripts/profile/harness.py (queue in results/profile/harness/, runs in results/profile/runs/),
analysis scripts/profile/findings.py -> results/profile/findings.md after every job. GPU 6 only while
GPU 0 is in use by others. Selection rule: EXPERIMENTS.md, "Profiling hybrid models on vLLM".

## 2026-10-29 00:30 -- start of night 1
Pilot (results/profile/pilot0, vision tower loaded, superseded by the harness base runs):
- Qwen3.5-4B decode C=2048: b=128 38.9 ms/step, b=256 348 ms/step (9x). Hypothesis: preemption or
  block-pool exhaustion from per-request Mamba state blocks. Job probe_b256_qwen35_4b counts preemptions.
- Hybrid vs dense at long context: C=32K b=4 15.1 vs 30.2 ms (Qwen3.5-4B vs Qwen3-4B); prefill 32K
  2.95 vs 4.63 s. KV capacity 955K vs 227K tokens.
- Decode GEMMs at b=16 run near the weight-read bound (9.7 ms vs ~9.1 ms ideal): little headroom there.
Queued: configs/harness_plan_night1.yaml (24 jobs: base sweeps, serving, 8 ablations, scaling).
