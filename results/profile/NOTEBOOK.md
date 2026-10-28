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

## 00:50 -- base_qwen35_4b (Ada, text-only)
- b=256 collapse confirmed as preemption: C=2048 b=256 323 ms/step with 39 preemptions during generation,
  although 256 x 2.7K tokens = 692K < the reported 955K-token capacity. Hypothesis: the Mamba groups'
  state/checkpoint blocks share the pool (each GDN state page = one 528-token attention page per layer),
  so a hybrid request needs about twice the blocks its tokens suggest. probe_b256 scans b=128..256.
- Decode roofline: GEMMs 0.89-1.12 of the weight-read bound, attention 0.99-1.02 of the KV-read bound:
  no kernel headroom in the big classes. GDN recurrent looks faster than read+write of fp32 state (its
  writes drain from L2 during later kernels); share 6% at b=16, 16% at b=64. State I/O is ReplaySSM's
  topic (upstream 0.32 has ReplaySSM for Mamba2 only), so it fails rule (c) unless the problem differs.
- GPU idle between decode steps ~1.5-1.9 ms regardless of batch: 14% of the step at b=1 (qualifies as
  overhead), 9% at b=16, 5% at b=64. Host CPU is shared (load ~127 on 96 cores), so part may be the machine.
  Queued nsys captures (configs/harness_plan_night1_f1.yaml): hybrid b=1 and b=16, dense b=1.
- Prefill: GEMMs at 0.70-0.72 of the bf16 peak; attention 24% of a 32K prefill; GDN chunk kernels 7-8%.
- Whole decode step vs the memory roofline (weights + KV + state read and write, at 926 GB/s): 0.89 at
  b=1 and b=16, 0.92 at b=64 (C=8192). Decode is close to bandwidth-optimal for the bytes it moves; the
  headroom left is mostly the inter-step gap. Bigger decode wins need fewer bytes (quantisation, KV or
  state compression, speculative decoding), not faster kernels.
