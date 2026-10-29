# RL post-training systems for LLMs and the rollout share of each RL step

Scope: end-to-end RL post-training systems (PPO/GRPO/DAPO-style, reasoning and agentic), 2024-2026, with measured time breakdowns. Research date 2026-10-29. "Measured" means the number comes from a table or profile in the cited paper. "Claim" means a sentence in the paper with no data behind it in that paper. "Computed" means I derived it from the paper's own table. "Estimate" means my own arithmetic, labeled as such.

## Q1. What share of RL step time does rollout take, and how does it change with response length, model size, and n?

### Takeaway
In synchronous or colocated long-CoT reasoning RL, measured rollout share is about 63-91% of step time. Seven independent measurements fall in that range, and it exceeds 95% at 32K or more max length (RhymeRL). The share falls to about 30-40% only when outputs are short: a video task averaging 166 output tokens, or 2024-era RLHF with 512-1K responses. So the student's worry holds only for short-output workloads. It does not hold for 8K-64K reasoning traces. Measured end-to-end gains still follow Amdahl's law: a 1.54-1.77x faster generation gave a 1.35-1.41x faster step (NVIDIA, NeMo-RL), and in a one-step-async pipeline the same idea gave only 1.24x.

### Cited Findings

**Summary table of measured rollout shares (synchronous unless noted)**

| Source | Model / task | Lengths | Samples/prompt (n) | HW | Rollout share | Other phases |
|---|---|---|---|---|---|---|
| Seer, Table 1 (Moonshot AI + Tsinghua) | Moonlight (32 GB) | max 65,536; avg 22,386 | 8 | 32 H800 | **84%** | train 14%, weight update 2% |
| Seer, Table 1 | Qwen2-VL-72B (LLM-as-judge reward) | max 40,960; avg 7,615 | 16 | 128 H800 | **63%** | train 31%, weight update 6% |
| Seer, Table 1 | Kimi-K2 (1 TB) | max 98,304; avg 38,959 | 8 | 256 H800 | **87%** | train 10%, weight update 3% |
| RollPacker, Table 1 (HKUST + Alibaba; NSDI'26) | Qwen2.5-14B, GRPO on veRL | max 16K | 8 (128 prompts/step) | 32 H800 | **72% math, 66% code, 71% LLM-judge** | reward 5/13/7%, train 23/21/22% |
| RhymeRL (SJTU IPADS + ByteDance) | Qwen3-8B/14B-Base, Qwen2.5-32B | max 16K | 16 | 64 GPUs (trace) | **84% (code) to 91% (math)**; **>95% at max length ≥32K** | n/a |
| TLT (MIT/NVIDIA/ETH; ASPLOS'26), Fig. 1a | reasoning RL (model unnamed for this figure) | max 32K | n/a | self-collected traces | **~85%** | n/a |
| Laminar (HKU + ByteDance Seed; EuroSys'26) | reasoning tasks | max output 16K | n/a | H800 | **up to 83.1%** | experience preparation 7.3% |
| NVIDIA NeMo-RL spec-decoding paper, Table 1 (computed) | Qwen3-8B "RL-Think" | DAPO-Math-17K, lengths not given | n/a | 32 GB200 (NVL72) | **72.1%** (133.6 s of 185.3 s) | logprob 17.9 s, train 31.4 s, prepare/weight sync 2.1 s |
| same | Qwen3-8B-Base "RL-Zero" | same | n/a | same | **66.1%** (100.0 s of 151.2 s) | logprob 17.8 s, train 31.3 s, prepare 1.9 s |
| RL in the Wild (USTC/Shanghai AI Lab et al.) | 235B MoE, coding | avg output 9,329 tokens | 16 | A800/H800 clusters | **>80%** (decode alone 85.4%) | n/a |
| same | video understanding | avg output **166.5** tokens, avg input 3,123 | 16 | same | **~30-40%** (decode 11.7%) | training communication 32.5% of training time |
| HybridFlow (EuroSys'25) | Llama 7B-70B PPO RLHF | 1,024 prompt + 1,024 response | 1 | 128 A100 | NeMo-Aligner: generation **up to 81.2%** of iteration; HybridFlow: actor training + generation = 58.9% combined | n/a |

Sources for the table rows: [Seer](https://arxiv.org/html/2511.14617), [RollPacker arXiv](https://arxiv.org/html/2509.21009) / [NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-gao-wei.pdf), [RhymeRL](https://arxiv.org/html/2508.18588), [TLT](https://arxiv.org/html/2511.16665), [Laminar](https://arxiv.org/html/2510.12633v1), [NeMo-RL speculative decoding](https://arxiv.org/html/2604.26779), [RL in the Wild](https://arxiv.org/html/2509.25279v1), [HybridFlow](https://arxiv.org/html/2409.19256)

**Headline claims with no measurement behind them in the paper itself**
- OpenRLHF's introduction says the inference phase "often accounts for over 90% of the total RLHF (or RLVR) runtime". The paper gives no breakdown supporting it — [OpenRLHF](https://arxiv.org/html/2405.11143)
- APRIL (AMD/CMU/LMSYS/UCLA) repeats ">90% of total runtime" and attributes it to the OpenRLHF paper. It is a cited claim, not a measurement — [APRIL v3](https://arxiv.org/html/2509.18521v3)
- ROLL Flash (Alibaba): rollout is "over 70% of total training time", citing prior work — [ROLL Flash](https://arxiv.org/html/2510.11345)
- Seer's introduction rounds its own Table 1 to "approximately 80%" — [Seer](https://arxiv.org/html/2511.14617)
- ROSE (agentic RL; arXiv May 2026) says "end-to-end training time is dominated by rollouts" without a number in the abstract. A search snippet put the figure at ">70%", which I could not verify — [ROSE](https://arxiv.org/abs/2605.06534)
- MiniMax-M1 says rollout computation and latency are "often the main RL bottlenecks" but gives no breakdown — [MiniMax-M1](https://arxiv.org/html/2506.13585)
- A secondary blog synthesis gives a median rollout time share of 68.5% across 10 configurations (range 22-87%). It also says rollout is only ~20% of FLOPs. These are blog-level numbers and I did not verify them against primary data — [zenn.dev blog](https://zenn.dev/tosshi/articles/99375dd5ac013f?locale=en)

**Absolute step times, to show scale**
- TLT, 32B reasoning RL: 11 days, 385 steps on 128 GPUs. Steps averaged about 40 minutes, and an evaluation every 5 steps took about 20 minutes — [TLT](https://arxiv.org/html/2511.16665)
- LlamaRL, synchronous baseline step times (global batch 2048, H100): 8B 22.45 s (256 GPUs), 70B 82.32 s (256 GPUs), 405B 635.8 s (1024 GPUs). LlamaRL best: 8.90 s, 20.67 s, 59.5 s (2.52x / 3.98x / 10.7x) — [LlamaRL](https://arxiv.org/html/2505.24034)
- Seer, Qwen2-VL-72B on veRL: instances idled 1,580 s on average, 37% of total rollout time — [Seer](https://arxiv.org/html/2511.14617)
- DeepSeek-R1-Zero (660B): 64×8 H800 for about 198 h (101K GPU-hours, $202K at $2/GPU-h). The DeepSeek-R1 RL stage took about 80 h (41K GPU-hours). Total including SFT data was 147K GPU-hours, $294K. The R1-Zero-Qwen-32B recipe uses 16 samples per question, max length 32,768, 512 samples per step, and 8,192 outputs per rollout split into 16 minibatches — [DeepSeek-R1 Nature supplementary](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf)
- MiniMax-M1 RL: 512 H800 for about 3 weeks, $534.7K rental. Output limit was 40K, extended in stages to 80K — [MiniMax-M1](https://arxiv.org/html/2506.13585)
- DeepSeek-V3.2: the post-training compute budget exceeds 10% of pre-training cost (from a search snippet of the report) — [DeepSeek-V3.2](https://arxiv.org/html/2512.02556)

**Long-tail statistics, the mechanism behind high rollout share**
- Seer defines the tail as the last 10% of requests. For memory-constrained tasks the tail can take up to 50% of total time. On Qwen2-VL-72B/veRL the run saw 13,686 preemptions, and the last 5% of requests started only at 42% of total time — [Seer](https://arxiv.org/html/2511.14617)
- RLHFuse: long-tailed samples are <1% of samples but take >50% of generation time, and P99.9 output length is >10x the median (2048-token-max RLHF) — [RLHFuse](https://arxiv.org/html/2409.13221)
- RollPacker: the longest response is 25-32x the median, and the 75th percentile is about 755-1.1K tokens against a 16K max — [RollPacker](https://arxiv.org/html/2509.21009)
- RhymeRL: the earliest-finishing GPU idles about 76% of rollout duration, and veRL shows >46% GPU idleness — [RhymeRL](https://arxiv.org/html/2508.18588)
- DynaRL (OSDI'26), 1.5B math GRPO across 64 engines: about 95% of queries finish in under 150 s against a 259.5 s max. Long-tail idling "can waste up to 60% of available computation" — [DynaRL OSDI'26](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing)
- Magistral: the longest completions take up to 5x longer than the shortest — [Magistral](https://arxiv.org/html/2506.10910)

**How the share varies with response length**
- RhymeRL: 84-91% at 16K max, >95% at max ≥32K — [RhymeRL](https://arxiv.org/html/2508.18588)
- Seer's three workloads rise monotonically with average generation length: 7.6K average gives 63%, 22.4K gives 84%, 39K gives 87%. This is confounded with model and task — [Seer](https://arxiv.org/html/2511.14617)
- RL in the Wild: decode share follows output length. Coding (avg 9.3K) decode is 85.4%; video (avg 166 tokens) decode is 11.7%. Max response length correlates with iteration time for coding and math but not for video or search — [RL in the Wild](https://arxiv.org/html/2509.25279v1)
- StreamRL says generation's share of iteration time rises with output length (5K→10K→20K max; Figure 2, numbers not in text) — [StreamRL](https://arxiv.org/html/2504.15930)
- RollPacker: tail batching's benefit grows with length (1.30x at 7B/8K, 1.48x at 14B/16K, 2.21x at 32B/32K). Overlap tricks such as reward scheduling and the stream trainer matter less as rollout becomes more dominant at long lengths — [RollPacker](https://arxiv.org/html/2509.21009)

**How the share varies with model size**
- No paper isolates model size with length held fixed. RollPacker reports 7B/14B/32B breakdowns only graphically (Fig. 10c). End-to-end speedups from rollout-side fixes grow from 2.03x (7B/8K) to 2.56x (32B/32K) over veRL, with size and length co-varying — [RollPacker](https://arxiv.org/html/2509.21009)
- TLT's end-to-end speedup is larger for 32B (1.83-2.12x) than 7B (1.21-1.76x) — [TLT](https://arxiv.org/html/2511.16665)

**How it varies with n (samples per prompt)**
- The n values used: 8-16 is "typical for GRPO" (Seer); 16 in RhymeRL, RL in the Wild, and DeepSeek-R1-Zero-Qwen-32B; 8 in RollPacker — [Seer](https://arxiv.org/html/2511.14617), [RhymeRL](https://arxiv.org/html/2508.18588), [DeepSeek-R1 supp.](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf)
- ROLL Flash "prompt replication" spreads the n samples of a prompt across instances. Per-step generation time fell from 116 s to 89 s (32 prompts × 16 samples), 149 s to 81 s (64×16), and 162 s to 83 s (16×32). The more samples per prompt, the more co-locating a group hurts the tail — [ROLL Flash](https://arxiv.org/html/2510.11345)
- PODS (CMU; TMLR 2026) frames rollout as "embarrassingly parallel and memory-light" and the policy update as communication- and memory-heavy. It generates many rollouts but trains on a max-variance subset, reaching GRPO's peak accuracy ≥1.7x faster. So with large n the training side can become the cost — [PODS](https://arxiv.org/abs/2504.13818)

**Amdahl check: measured end-to-end effect of faster rollout**
- NeMo-RL, synchronous 8B: generation 1.54x (Think) / 1.77x (Zero) faster with speculative decoding gave a step 1.35x / 1.41x faster. Logprob plus training stayed at about 49 s. With speculation, generation drops to 63% / 53% of the step — [NeMo-RL spec decoding](https://arxiv.org/html/2604.26779)
- Same paper, async with policy lag 1 (12 generation nodes, 4 training nodes): exposed generation fell from 10.4 s to 0.6 s, but the effective step went only from 75.0 s to 60.5 s (1.24x). The authors say the gain is smaller because async overlap already hides much of the rollout — [NeMo-RL spec decoding](https://arxiv.org/html/2604.26779)
- Same paper, **simulator projection, not measured**: Qwen3-235B-A22B on 512 GB200 gives 2.72-4.07x rollout and 1.70-1.96x end-to-end. At 2048 GPUs with lag 2, about 3.5x rollout gives about 2.5x end-to-end — [NeMo-RL spec decoding](https://arxiv.org/html/2604.26779)
- Seer: rollout throughput is 1.44-2.04x over veRL, and long-tail latency drops 72-94% — [Seer](https://arxiv.org/html/2511.14617)
- RhymeRL: up to 2.6x end-to-end over veRL — [RhymeRL](https://arxiv.org/html/2508.18588)
- TLT: 1.7-2.1x end-to-end over VeRL — [TLT](https://arxiv.org/html/2511.16665)

### Inferences
- Estimate (Amdahl's law): end-to-end speedup = 1/((1−f) + f/s) for rollout share f and rollout speedup s. At s=2: f=0.70 gives 1.54x, f=0.85 gives 1.74x, f=0.35 (short-output video) gives 1.21x. NeMo-RL's measured 1.41x from f=0.661 and s=1.77 matches the formula, which predicts 1.40x. Rollout is the dominant term for long-CoT RL, but the training-plus-logprob residual (about 10-35% in the table) caps gains at about 3-10x even with infinitely fast rollout.
- Response length is the strongest predictor of rollout share in the data. The share is well above 60% for 8K-100K reasoning traces, and the long tail rather than average throughput drives much of it: tails are up to 50% of rollout time in Seer and over 50% in RLHFuse. Rollout acceleration that helps the tail, such as speculative decoding at low batch, rebalancing, or KV-memory headroom against preemption, is worth more than raising peak throughput.
- For a KV-centric project: Seer's 13,686 preemptions and its statement that memory-constrained tasks have the worst tails suggest KV-cache capacity and management directly affect RL rollout time. Seer frames rollout as KV-cache-memory-bound, not just compute-bound.
- In async/disaggregated systems "rollout share of step time" is not the right metric, because rollout is overlapped. There the relevant quantity is the fraction of GPUs devoted to rollout. Computed: LlamaRL 405B uses 512 of 1024 GPUs for generation (50%), 70B uses 120 of 256 (47%); PipelineRL 48 of 128 (37.5%); ROLL Flash's best split 24 of 40 (60%); NeMo-RL async 12 of 16 nodes (75%). Faster rollout then cuts GPU cost or staleness rather than the critical path, until training becomes the bottleneck. NeMo-RL's 1.24x async result shows this. Sources: [LlamaRL](https://arxiv.org/html/2505.24034), [PipelineRL](https://arxiv.org/html/2509.19128v2), [ROLL Flash](https://arxiv.org/html/2510.11345), [NeMo-RL](https://arxiv.org/html/2604.26779).

### Gaps
- No paper sweeps n (samples/prompt) while holding everything else fixed and reports rollout share. The effect of n is only indirect (ROLL Flash prompt replication, PODS).
- No clean model-size sweep at fixed length with numeric shares. RollPacker's 7B/32B and StreamRL's Figure 2 values exist only as figures.
- PPO with a critic versus GRPO/DAPO without one: no source compared rollout share directly. The older PPO-RLHF numbers (HybridFlow, RLHFuse) use 512-2K responses, so length and algorithm are confounded.
- The secondary claim that rollout is ~20% of FLOPs but ~68% of time is unverified against primary sources.
- Industry reports (Kimi K1.5/K2, MiniMax-M1, Seed1.5-Thinking, GLM-4.5, DeepSeek-R1) give no rollout/train time split. Seer's Kimi-K2 row (87%) is the only production-scale number from a frontier lab.

## Q2. Main system architectures (colocated, disaggregated, async/off-policy) and what each says rollout costs

### Takeaway
There are three families. (1) Colocated or time-shared systems (verl/HybridFlow, Kimi K1.5/K2, DeepSeek-R1, Seer, slime in sync mode, GLM-4.5 reasoning RL) put rollout on the critical path, so rollout share equals its share of step time (63-87% in Seer). (2) Disaggregated synchronous or streaming systems (StreamRL, RollPacker's stream trainer, RhymeRL) overlap stages and let hardware specialize. (3) Asynchronous or off-policy systems (AReaL, Laminar, LlamaRL, ROLL Flash, PipelineRL, Magistral, Seed1.5-Thinking's SRS, GLM-4.5 agentic RL, verl fully-async) hide rollout behind training at the price of bounded staleness. Async speedups over sync are consistently about 1.5-5.5x, and they come from removing rollout's long-tail idle time. Every family names rollout as the dominant cost.

### Cited Findings

**Colocated / time-sharing**
- HybridFlow/verl (HKU + ByteDance; EuroSys'25): hybrid single-controller/multi-controller design. 3D-HybridEngine reshards the actor between training and generation on the same GPUs. Reported 1.53-20.57x over DeepSpeed-Chat, OpenRLHF, and NeMo-Aligner, on 128 A100 with vLLM 0.3.1 — [HybridFlow](https://arxiv.org/html/2409.19256)
- Kimi K1.5 (Moonshot): iterative synchronous RL. Megatron and vLLM run in separate containers in one pod (Kubernetes sidecar) sharing all GPUs, linked by "checkpoint-engine". Switching takes under 1 minute from training to inference and about 10 s the other way. vLLM is killed and restarted rather than offloaded, because CUDA graphs, NCCL buffers, and drivers make full memory release hard. Partial rollouts carry unfinished long trajectories to the next iteration. RL context scaled to 128K — [Kimi K1.5](https://arxiv.org/html/2501.12599)
- Kimi K2 (Moonshot): "hybrid colocated architecture", with training parameters offloaded to DRAM during rollout. A distributed checkpoint engine broadcasts full parameters and each inference worker pulls its shard. A full 1T parameter update takes under 30 s. Partial rollout handles long-tail trajectories, and sandboxes run more than 10,000 concurrent instances — [Kimi K2](https://arxiv.org/html/2507.20534)
- DeepSeek-R1: decoupled four-module framework (Rollout on vLLM workers, Inference for reward/ref, rule-based Reward, Training). Each module's models are offloaded from VRAM to host memory or disk after use. Rule-based rewards run asynchronously, overlapped with rollout. Rollout uses cross-node expert parallelism, redundant hot experts, and MTP self-speculative decoding to "minimize the completion time for the longest samples". Training uses DualPipe and best-fit packing — [DeepSeek-R1 supp.](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf)
- Seer (Moonshot + Tsinghua): a "synchronous, colocated RL system" with Megatron, in-house vLLM, and Checkpoint Engine. It keeps on-policy RL and attacks the tail with divided rollout, context-aware scheduling, and grouped speculative decoding — [Seer](https://arxiv.org/html/2511.14617)
- Seed1.5-Thinking (ByteDance Seed; 20B active / 200B total MoE): built on HybridFlow/Ray with a colocated hybrid engine. Its Streaming Rollout System generates a fraction α of samples on-policy, and the rest (1−α) come from versioned older snapshots on standalone resources. It uses FP8 policy during rollout and claims "3x faster iteration cycles than synchronous frameworks" (α value and baseline not given) — [Seed1.5-Thinking](https://arxiv.org/html/2504.13914)
- GLM-4.5 / slime (Zhipu + Tsinghua): colocated synchronous mode for general and reasoning RL (math/code), and a disaggregated asynchronous mode for agentic RL (SWE). Training uses Megatron; rollout uses SGLang plus a router with FP8 block-wise quantized weights (BF16 training). Reasoning RL ran single-stage at 64K output. No timing numbers — [GLM-4.5](https://arxiv.org/html/2508.06471)
- RL in the Wild: both production 235B-MoE configs are colocated (256 and 128 GPUs) with FP8 rollout. No task shows strong scaling, and some show negative scaling: search rollout took 585.5 s on 128 GPUs against 492.3 s on 64 — [RL in the Wild](https://arxiv.org/html/2509.25279v1)

**Disaggregated / streaming (still mostly on-policy)**
- StreamRL (PKU + StepFun; arXiv Apr 2025): argues colocation causes "resource coupling", since generation is memory-bandwidth-bound and training is compute-bound. It disaggregates, allowing heterogeneous hardware (H20 for generation, H800 for training) and cross-datacenter deployment. Stream generation, skewness-aware dispatch, and one-step async give 1.30-2.66x throughput over prior systems and up to 1.33x cost-effectiveness on heterogeneous hardware. Ablation on 72B/20K: colocation 1.00, plus skew scheduling 1.08, plus disaggregation/streaming 1.23, plus async 1.48 — [StreamRL](https://arxiv.org/html/2504.15930)
- RLHFuse (PKU + StepFun; NSDI'25): fuses generation and inference at sample level by migrating long-tail samples once about 20% remain, and fuses training micro-batches. Gains are 2.5-3.7x over DeepSpeed-Chat and 1.4-2.4x over ReaLHF. Migration overhead is under 3% of the iteration — [RLHFuse](https://arxiv.org/html/2409.13221), [NSDI'25 page](https://www.usenix.org/conference/nsdi25/presentation/zhong)
- RollPacker (HKUST + Alibaba; NSDI'26): synchronous, on ROLL with vLLM 0.8.4 and Megatron 0.12.2. Tail batching, reward colocation via MPS, and a stream trainer give 2.03-2.56x over veRL on up to 128 H800 — [RollPacker NSDI'26](https://www.usenix.org/system/files/nsdi26-gao-wei.pdf)
- RhymeRL (SJTU + ByteDance): built on veRL with disaggregated rollout, reward, and train workers in a streaming pipeline. HistoSpec reuses the previous epoch's rollouts as speculative drafts: about 93% of math tokens are acceptable, with 65-79% acceptance. Up to 2.6x over veRL, and up to 2.1x over AReaL at staleness 1 — [RhymeRL](https://arxiv.org/html/2508.18588)
- DynaRL (PKU + Infinigence AI et al.; OSDI'26): dynamic reallocation of GPUs across RL components via a hypergraph scheduler. Up to 1.98x throughput on math and agentic RL (128 H100) — [DynaRL](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing)

**Asynchronous / off-policy**
- AReaL (Tsinghua IIIS + Ant Group + HKUST; NeurIPS'25): fully async. Rollout workers never wait, and trainers update when a batch is ready. Staleness η ≤ 8 has minimal impact, and a staleness-aware PPO handles old samples. Interruptible rollout gives +12-17%. End-to-end hours versus sync AReaL: 1.5B 41.0→14.8 (2.77x), 7B 57.7→25.4, 14B 48.8→21.9, 32B 51.1→31.1. Uses SGLang 0.4.6 and Megatron-Core 0.11 — [AReaL](https://arxiv.org/html/2505.24298), [NeurIPS'25](https://neurips.cc/virtual/2025/poster/117538)
- Laminar (HKU + ByteDance Seed; EuroSys'26): trajectory-level asynchrony with relay workers as a distributed parameter service, plus dynamic repack of long-tail trajectories (+26% generation throughput, 0.69 s overhead). Up to 1,024 H800. Average 2.56x over verl (up to 5.49x), 1.98x over one-step staleness, 1.93x over stream generation, and 1.39x over AReaL. Staleness typically under 3, max 4 — [Laminar](https://arxiv.org/html/2510.12633v1), [HKU EuroSys'26 record](https://hub.hku.hk/handle/10722/375626)
- LlamaRL (Meta GenAI; arXiv): distributed async with separate generator and trainer, different model-parallel sizes per side, an FP8 generator, and DDMA weight sync. 2.52x / 3.98x / 10.7x step-time speedups at 8B / 70B / 405B. Most of the 405B gain is attributed to decoupled model-parallel sizes — [LlamaRL](https://arxiv.org/html/2505.24034)
- ROLL Flash (Alibaba + HKUST + SJTU; arXiv): async, with up to 2.24x on RLVR and 2.72x on ALFWorld agentic. An async ratio of 2 gives +64% throughput. Uses truncated importance sampling (C=5), SGLang 0.4.6 and vLLM 0.8.4 — [ROLL Flash](https://arxiv.org/html/2510.11345)
- PipelineRL (ServiceNow + Mila): in-flight weight updates after every optimizer step without recomputing the KV cache. About 2x faster learning than conventional RL with G=32. Qwen2.5-7B, with 48 generation and 80 training H100. Max lag is about ⌈t_gen/t_train⌉ — [PipelineRL](https://arxiv.org/html/2509.19128v2)
- Magistral (Mistral): generators never wait. Weights are broadcast over NCCL mid-generation in under 5 s, and the KV cache is not recomputed ("not necessary"), so one completion can span several policy versions — [Magistral](https://arxiv.org/html/2506.10910)
- verl recipes: one-step-off-policy overlaps generation for batch t+1 with training on batch t, and the docs say shrinking generation resources reduces long-tail idle. The fully-async recipe exposes a `staleness_threshold`, plus streaming and partial rollout (from search snippets of the docs) — [verl one-step-off](https://verl.readthedocs.io/en/latest/advance/one_step_off.html), [verl fully async](https://verl.readthedocs.io/en/latest/advance/fully_async.html)
- RL in the Wild: async training gives about a 60% end-to-end gain over sync (52h14m at staleness 0 vs 33h50m at staleness 8, 1.5B) — [RL in the Wild](https://arxiv.org/html/2509.25279v1)
- Partial / over-provisioned rollouts as a middle ground: APRIL over-provisions requests and carries unfinished ones forward (in slime), giving +22.5% average and up to 49.5% throughput on Qwen3-4B/8B at 16K. Kimi K1.5/K2 use partial rollouts — [APRIL](https://arxiv.org/html/2509.18521v3), [Kimi K1.5](https://arxiv.org/html/2501.12599)
- Algorithm-side caveat: the NeurIPS 2025 work by Feng Yao, Liyuan Liu et al. ("Your Efficient RL Framework Secretly Brings You Off-Policy RL Training", later "On the Rollout-Training Mismatch in Modern RL Systems") shows vLLM vs FSDP token probabilities differ even with identical weights, so "on-policy" RL is already slightly off-policy. Truncated importance sampling fixes it, including under aggressive rollout quantization — [Feng Yao et al. blog](https://fengyao.notion.site/off-policy-rl), [NeurIPS listing](https://neurips.cc/virtual/2025/124259)

### Inferences
- A sync-vs-async split appears in production. Frontier labs that publish details (Moonshot, DeepSeek, Zhipu for reasoning) keep reasoning RL synchronous or colocated and attack the tail with partial rollout, speculation, and scheduling. Agentic RL (GLM-4.5 SWE, Kimi K2 agentic, ROSE, ROLL Flash agentic) moves to disaggregated async because environment latency dominates. So rollout acceleration is on the critical path for the most common reasoning setup.
- The 2-5x async speedups (AReaL, Laminar, LlamaRL) are largely a measure of how much long-tail idle the sync baseline wastes. That idle is a rollout-phase cost, so the existence and size of async gains is indirect evidence that rollout dominates.
- On-policy-preserving rollout acceleration (Seer, RollPacker, TLT, RhymeRL, NeMo-RL spec decoding) reports 1.4-2.6x end-to-end. That is comparable to async gains without the staleness trade-off, which is why several 2025-26 papers frame "fast synchronous RL" as their contribution.

### Gaps
- StreamRL's and Seer's peer-reviewed venues were not confirmed; I saw only arXiv. LlamaRL, ROLL Flash, RhymeRL, PipelineRL, and APRIL appear arXiv-only as of what I could check; an "ICLR 2026 submission" label for APRIL was in a search snippet only.
- Qwen3 tech report RL infrastructure was not examined. I found no breakdown and did not verify whether one exists.
- DeepSeek-V3.2 RL infrastructure beyond the ">10% of pre-training cost" budget was not examined.
- MiniMax-M1 does not say whether its RL pipeline is sync or async.

## Q3. Weight synchronization / resharding: how long it takes, which systems address it, is it a bottleneck?

### Takeaway
Weight sync was a recognized bottleneck in 2024. It could take up to 36.4% of iteration time for a 70B model when trainer and generator are on different devices (OpenRLHF figure cited by HybridFlow), and OpenRLHF took 111.65 s for 70B. By 2025-2026 it was largely engineered down to about 1-6% of iteration time: Seer 2-6%, NeMo-RL about 1.1-1.3%, StreamRL under 2%. Absolute times are about 1-20 s even at 1T parameters, via CUDA-IPC (colocated), NCCL broadcast trees, RDMA P2P (Mooncake, fabric-lib, Awex), and relay or parameter-service designs. It remains an active engineering topic at 1T-scale MoE, for elastic or dynamic instances, and across datacenters. It is not the main lever compared with rollout.

### Cited Findings

**Measured sync times (chronological / by scale)**

| System | Model | Setup | Time | Source |
|---|---|---|---|---|
| OpenRLHF (baseline in LlamaRL Table 4) | 7B / 70B | H100 | 4.32 s / 111.65 s | [LlamaRL](https://arxiv.org/html/2505.24034) |
| LlamaRL DDMA (zero-copy GPU-direct over NVLink/IB) | 7B / 70B / 405B | H100 | 0.04 s / 1.15 s / 2.31 s | [LlamaRL](https://arxiv.org/html/2505.24034) |
| HybridFlow 3D-HybridEngine | 70B | 128 A100 colocated | transition overhead cut by up to 89.1% (78.2 s) vs OpenRLHF; 55.2% (11.7 s) on average | [HybridFlow](https://arxiv.org/html/2409.19256) |
| OpenRLHF figure cited by HybridFlow | 70B (140 GB) | separate devices | up to 36.4% of iteration time | [HybridFlow](https://arxiv.org/html/2409.19256) |
| StreamRL (multi-root broadcast trees; GPUDirect RDMA, TCP fallback) | 72B | over 80 Gbps cross-DC link | <10 s, <2% of iteration | [StreamRL](https://arxiv.org/html/2504.15930) |
| Laminar relay workers | 72B | master to 127 relays | <1.6 s broadcast; actor stall 0.64 s (32B) / 1.40 s (72B); storage-based baseline 8 s serialize + 10-20 s TCP per 4 GB shard | [Laminar](https://arxiv.org/html/2510.12633v1) |
| Magistral NCCL broadcast | Magistral Medium | large world sizes | <5 s | [Magistral](https://arxiv.org/html/2506.10910) |
| Kimi K1.5 checkpoint-engine (Mooncake RDMA, vLLM restart) | Kimi K1.5 | colocated pod | train→infer <1 min; infer→train ~10 s | [Kimi K1.5](https://arxiv.org/html/2501.12599) |
| Kimi K2 checkpoint engine (full broadcast, pipelined per parameter) | 1T | colocated | <30 s full update | [Kimi K2](https://arxiv.org/html/2507.20534) |
| MoonshotAI checkpoint-engine (open source, vLLM 0.10.2) | Kimi-K2 FP8 | 16 / 256 H20 | 14.36 s / 16.04 s broadcast; 17.49 s / 16.75 s P2P; README claims ~20 s for 1T "across thousands of GPUs" | [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) |
| same | Qwen3-235B-A22B BF16 / GLM-4.5-Air | 8 H800 | 6.22 s / 3.47 s broadcast | [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) |
| Perplexity fabric-lib (one-sided RDMA WRITE, static schedule, pipelined H2D/prep/transfer) | Kimi-K2 1T | 256 BF16 train GPUs to 128 FP8 inference GPUs | ~1.3 s (LMSYS table lists ~1.2 s); 1.2-2 s in DeepSeek-V3/Qwen3-235B logs (search-derived) | [Perplexity blog](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds), [fabric-lib paper](https://arxiv.org/pdf/2510.27656) |
| LMSYS/SGLang RDMA P2P via Mooncake TransferEngine (Apr 2026) | Kimi-K2 FP8 | 32/32 nodes H100 | 53.3 s (NCCL) to 7.2 s (7.37x) | [LMSYS P2P blog](https://lmsys.org/blog/2026-04-29-p2p-update/) |
| same | GLM-5 744B / Qwen3-235B / GLM-4.5-Air / 30B-A3B | 16/8/4/1 nodes | 58.3→8.5 s; 10.8→3.2 s; 5.0→2.6 s; 2.5→4.2 s (P2P slower at small scale) | [LMSYS P2P blog](https://lmsys.org/blog/2026-04-29-p2p-update/) |
| Ant Group Awex (with Megatron and SGLang) | 1T | thousand-GPU cluster | 6 s RDMA / ~20 s NCCL (README headline "within 10 s") | [Awex repo](https://github.com/inclusionAI/asystem-awex), [Pandaily](https://pandaily.com/ant-group-open-sources-awex-framework-for-second-level-tb-scale-rl-parameter-swapping) |
| slime colocated (CUDA IPC handles, async gather, 512 MB buckets, flattening) | Qwen3-30B-A3B | 8 H100 | ~60 s to ~7 s (author says the sequence was simulated from PRs) | [slime weight-sync blog](https://hebiao064.github.io/rl-weight-sync) |

**Weight-sync share of the step**
- Seer: weight update is 2% (Moonlight), 6% (Qwen2-VL-72B), and 3% (Kimi-K2) of iteration — [Seer](https://arxiv.org/html/2511.14617)
- NeMo-RL, 8B on GB200: "Prepare" (weight sync plus backend prep) takes 1.6-2.1 s of a 107-185 s step, about 1-2% (computed) — [NeMo-RL spec decoding](https://arxiv.org/html/2604.26779)
- LMSYS: during weight transfer "the entire RL training comes to a halt". NCCL broadcast is limited by a single source rank, all-gather redundancy, and lock-step stalls — [LMSYS P2P blog](https://lmsys.org/blog/2026-04-29-p2p-update/)
- ROSE (agentic, 2026) lists a cross-cluster weight transfer engine with shard-aware routing and weight sparsity among its three core components. Elastic, cross-cluster rollout reopens the problem — [ROSE](https://arxiv.org/abs/2605.06534)

### Inferences
- Weight sync matters most in three regimes: (a) async or elastic systems where new instances join (checkpoint-engine P2P mode, ROSE), (b) cross-datacenter or heterogeneous deployments (StreamRL), and (c) 1T-scale MoE with naive NCCL, where 50-60 s per step would be about 5-15% of a 6-10-minute step (estimate). At state-of-the-art 1-7 s it is a small fraction next to rollout's 63-87%.
- Systems that keep stale KV across weight updates (PipelineRL, Magistral) and those that discard and recompute it (AReaL) make different trade-offs. A KV-fetch or KV-reuse mechanism that is cheap on weight update could matter for in-flight-update designs.

### Gaps
- Several numbers above (Perplexity 1.3 s, Awex 6 s) come from vendor blogs and READMEs, not peer-reviewed measurement. The slime 60→7 s sequence is explicitly illustrative.
- No source gave weight-sync share of step time for a 1T model under NCCL broadcast in a real run.

## Q4. Which rollout engines are used, and what engine-level features do RL frameworks depend on?

### Takeaway
vLLM and SGLang dominate. vLLM is used by verl/HybridFlow, OpenRLHF, NeMo-RL, PipelineRL, Laminar, RollPacker/ROLL, DeepSeek-R1, Kimi (in-house fork in Seer), and TLT. SGLang is used by slime/GLM-4.5, AReaL, APRIL, and ROLL Flash. Some labs use in-house engines (StreamRL's C++/CUDA engine; Meta's internal library in LlamaRL). RL frameworks depend on six engine features: (1) memory release and restore (vLLM sleep levels 1/2, SGLang memory release, or simply killing the process as Kimi does); (2) in-place weight-update APIs (collective_rpc/reload_weights, update_weights_from_tensor/distributed, CUDA-IPC, dummy-load plus external push); (3) interruptible or pausable generation and partial rollout; (4) faithful sampling logprobs and numerics matching the trainer; (5) low-precision (FP8) rollout; and (6) long-tail aids such as speculative decoding (MTP/EAGLE/n-gram/history), EP load balancing, and prefix sharing across a group's n samples.

### Cited Findings
- **Engine versions in papers:** HybridFlow used vLLM 0.3.1 with a distributed KV manager in place of the centralized one — [HybridFlow](https://arxiv.org/html/2409.19256). Laminar used vLLM 0.9.0, with SGLang for its AReaL baseline — [Laminar](https://arxiv.org/html/2510.12633v1). RollPacker used vLLM 0.8.4 — [RollPacker](https://arxiv.org/html/2509.21009). AReaL used SGLang 0.4.6 — [AReaL](https://arxiv.org/html/2505.24298). ROLL Flash used SGLang 0.4.6 and vLLM 0.8.4 — [ROLL Flash](https://arxiv.org/html/2510.11345). RL in the Wild used verl 0.3.1 with SGLang 0.4.6 and vLLM 0.8.5 — [RL in the Wild](https://arxiv.org/html/2509.25279v1). DeepSeek-R1 used vLLM workers — [DeepSeek-R1 supp.](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf). Seer used in-house vLLM — [Seer](https://arxiv.org/html/2511.14617). NeMo-RL used a vLLM backend — [NeMo-RL](https://arxiv.org/html/2604.26779). GLM-4.5/slime used SGLang plus a router — [GLM-4.5](https://arxiv.org/html/2508.06471). StreamRL used an in-house C++/CUDA engine with continuous batching and prefix sharing — [StreamRL](https://arxiv.org/html/2504.15930). LlamaRL used Meta's internal inference library — [LlamaRL](https://arxiv.org/html/2505.24034).
- **Sleep/wake (vLLM):** Level 1 offloads weights to CPU and discards the KV cache. Level 2 discards both, followed by `collective_rpc("reload_weights")`. `wake_up(tags=["weights"])` restores weights before a weight update, then `tags=["kv_cache"]` restores the KV cache, to avoid OOM. HTTP `/sleep`, `/wake_up`, `/collective_rpc` endpoints exist in dev mode, and the docs say "up to 90%+" of GPU memory can be released, with RLHF as the stated use case — [vLLM sleep mode docs](https://docs.vllm.ai/en/latest/features/sleep_mode.html)
- **Kimi's counterexample:** vLLM is terminated and restarted instead of offloaded, because CUDA graphs, NCCL buffers, and drivers make full memory release difficult — [Kimi K1.5](https://arxiv.org/html/2501.12599)
- **Weight-update hooks:** checkpoint-engine needs vLLM's `/collective_rpc` endpoint (v0.10.2 recommended), `--load-format dummy`, and a worker extension class. SGLang integration uses `--wait-for-initial-weights` — [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine). slime's colocated sync uses SGLang `update_weights_from_tensor` with CUDA IPC handles; disaggregated mode uses a distributed endpoint over NVLink/IB — [slime weight-sync blog](https://hebiao064.github.io/rl-weight-sync). LMSYS's disk/NCCL/RDMA P2P options are SGLang update paths (`update_weights_from_disk` named explicitly) — [LMSYS P2P blog](https://lmsys.org/blog/2026-04-29-p2p-update/)
- **Interruptible generation:** AReaL's `update_weights` interrupts ongoing generation, discards old-weight KV caches, and recomputes them — [AReaL](https://arxiv.org/html/2505.24298). PipelineRL and Magistral pause briefly, swap weights, and continue with stale KV — [PipelineRL](https://arxiv.org/html/2509.19128v2), [Magistral](https://arxiv.org/html/2506.10910)
- **Logprobs and numerics:** vLLM v1 did not directly return the adjusted sampling probabilities, so Feng Yao et al. patched it. Rollout and trainer probabilities differ even with identical weights, and truncated importance sampling corrects for it — [Feng Yao et al.](https://fengyao.notion.site/off-policy-rl). MiniMax-M1 raised the LM head to FP32 so training/inference probability correlation went from about 0.9 to 0.99 — [MiniMax-M1](https://arxiv.org/html/2506.13585). verl recomputes logprobs in the trainer, with tuning knobs `log_prob_micro_batch_size_per_gpu` and `log_prob_max_token_len_per_gpu` — [verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html). In NeMo-RL the logprob pass costs about 18 s per step, about 10-12% of step time (computed) — [NeMo-RL](https://arxiv.org/html/2604.26779)
- **FP8 rollout:** GLM-4.5 does online block-wise FP8 quantization each policy update — [GLM-4.5](https://arxiv.org/html/2508.06471). LlamaRL runs an FP8 generator at 70B and 405B — [LlamaRL](https://arxiv.org/html/2505.24034). Seed1.5-Thinking uses FP8 policy in rollout — [Seed1.5-Thinking](https://arxiv.org/html/2504.13914). Both RL in the Wild production configs use FP8 rollout — [RL in the Wild](https://arxiv.org/html/2509.25279v1). Perplexity transfers BF16 training weights to FP8 inference weights — [Perplexity](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds)
- **Speculative decoding for the tail:** DeepSeek-R1 uses MTP self-speculation — [DeepSeek-R1 supp.](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf). Seer uses grouped speculative decoding that exploits similarity among a group's n samples, up to 1.3x over vanilla SD — [Seer](https://arxiv.org/html/2511.14617). TLT trains an adaptive EAGLE drafter on idle GPUs and enables SD only when fewer than 32 requests remain — [TLT](https://arxiv.org/html/2511.16665). RhymeRL's HistoSpec drafts from the previous epoch's rollouts — [RhymeRL](https://arxiv.org/html/2508.18588). NeMo-RL supports MTP, draft models, and Eagle3 in vLLM — [NeMo-RL](https://arxiv.org/html/2604.26779)
- **Scheduling and KV:** RL in the Wild finds FCFS scheduling poor, prefill/decode disaggregation not beneficial for RL, and coarse static offloading plus load-unaware KV preemption inefficient — [RL in the Wild](https://arxiv.org/html/2509.25279v1). DynaRL says FIFO scheduling in SGLang/vLLM lets a finished turn's KV cache be evicted before the next agentic turn, wasting prefix reuse — [DynaRL](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing). verl recommends `gpu_memory_utilization` of 0.5-0.7 and raising `max_num_seqs`/`max_num_batched_tokens` when KV utilization is low — [verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)
- **Prefix sharing across n samples:** StreamRL's engine explicitly supports prefix sharing — [StreamRL](https://arxiv.org/html/2504.15930). ROLL Flash instead splits a prompt's n samples across instances to cut the tail — [ROLL Flash](https://arxiv.org/html/2510.11345)

### Inferences
- Short reasoning prompts make prefix sharing across n samples low-value: RL in the Wild math averages 126.5 input tokens against 9,839 output tokens. Multimodal (8.57x larger inputs) and multi-turn agentic workloads (tool use p90 input 2,802 tokens, max 50K) are where prefix and KV reuse can pay off — [RL in the Wild](https://arxiv.org/html/2509.25279v1). That is the more promising target for a KV-fetch-style idea in RL.
- Engine features RL needs that serving does not (fast weight swap, memory release, interruptibility, faithful logprobs, batch-invariant numerics) are being upstreamed into vLLM and SGLang by the frameworks (slime, checkpoint-engine, AReaL). Any new rollout-acceleration mechanism must coexist with frequent weight swaps that invalidate cached state.

### Gaps
- SGLang's RL docs page (docs.sglang.io/advanced_features/rl.html) returned 404, so I could not confirm exact SGLang API names (release/resume memory occupation, update_weights_from_distributed, pause/continue generation, deterministic inference) from primary docs. Only `update_weights_from_tensor` and `update_weights_from_disk` are confirmed above via the slime and LMSYS blogs.
- I found no measured wake-up or sleep latency for vLLM sleep mode in the docs.
- I found no measured benefit of prefix caching across n samples in an RL setting.

## Q5. System catalog: venue, date, and affiliation

### Takeaway
The peer-reviewed systems record runs from 2024 PPO-RLHF systems (HybridFlow EuroSys'25, RLHFuse NSDI'25, ReaL MLSys'25) to 2025-26 long-CoT systems (AReaL NeurIPS'25, TLT ASPLOS'26, Laminar EuroSys'26, RollPacker NSDI'26, DynaRL OSDI'26). Many influential systems and all industry reports are arXiv-only.

### Cited Findings

| System / report | Affiliation | Date / venue | Type | Key rollout number | Source |
|---|---|---|---|---|---|
| OpenRLHF | OpenRLHF team (affiliations not listed on page) | arXiv 2405.11143 (May 2024; v6 Oct 2025) | Ray, vLLM, separate or hybrid placement | claim ">90%" | [arXiv](https://arxiv.org/html/2405.11143) |
| ReaL / ReaLHF (background) | Tsinghua / Ant (ReaLHF lineage) | MLSys 2025 | parameter reallocation | none extracted | [MLSys'25](https://mlsys.org/virtual/2025/3348) |
| HybridFlow / verl (background) | HKU + ByteDance | EuroSys 2025 (arXiv Sep 2024) | colocated hybrid engine | NeMo-Aligner gen up to 81.2% | [arXiv](https://arxiv.org/html/2409.19256) |
| RLHFuse (background) | PKU + StepFun | NSDI 2025 | stage fusion | long tail >50% of gen time | [NSDI'25](https://www.usenix.org/conference/nsdi25/presentation/zhong) |
| Kimi K1.5 | Moonshot AI | arXiv Jan 2025 | colocated, partial rollout, checkpoint-engine | switch <1 min / ~10 s | [arXiv](https://arxiv.org/html/2501.12599) |
| DeepSeek-R1 | DeepSeek | arXiv Jan 2025; Nature 2025 | decoupled modules, offload, vLLM + MTP | 147K H800-hours total | [Nature supp.](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf) |
| Seed1.5-Thinking | ByteDance Seed | arXiv Apr 2025 | HybridFlow + streaming rollout | claim 3x vs sync | [arXiv](https://arxiv.org/html/2504.13914) |
| StreamRL | PKU + StepFun | arXiv Apr 2025 (venue unconfirmed) | disaggregated stream generation | weight sync <2% | [arXiv](https://arxiv.org/html/2504.15930) |
| PODS | CMU | arXiv Apr 2025; TMLR 2026 | algorithmic down-sampling | rollout "memory-light" | [arXiv](https://arxiv.org/abs/2504.13818) |
| AReaL | Tsinghua IIIS + Ant Group + HKUST | NeurIPS 2025 (arXiv May 2025) | fully async, SGLang | 2.77x vs sync | [NeurIPS](https://neurips.cc/virtual/2025/poster/117538) |
| LlamaRL | Meta GenAI | arXiv May/Jun 2025 | distributed async, DDMA | 405B 10.7x | [arXiv](https://arxiv.org/html/2505.24034) |
| MiniMax-M1 | MiniMax | arXiv Jun 2025 | CISPO, lightning attention | 512 H800 × 3 wk | [arXiv](https://arxiv.org/html/2506.13585) |
| Magistral | Mistral AI | arXiv Jun 2025 | async, in-flight NCCL updates | sync <5 s | [arXiv](https://arxiv.org/html/2506.10910) |
| ROLL / ROLL Flash | Alibaba (+ HKUST, SJTU) | arXiv Jun 2025 / Oct 2025 | async, queue scheduling | >70% (cited) | [ROLL Flash](https://arxiv.org/html/2510.11345) |
| slime | Zhipu AI / THUDM (with LMSYS) | blog Jul 2025 | SGLang + Megatron, colocated or decoupled | colocated sync 60→7 s | [LMSYS blog](https://lmsys.org/blog/2025-07-09-slime) |
| Kimi K2 | Moonshot AI | arXiv Jul 2025 | colocated, checkpoint engine | 1T update <30 s | [arXiv](https://arxiv.org/html/2507.20534) |
| GLM-4.5 | Zhipu AI + Tsinghua | arXiv Aug 2025 | slime: sync colocated (reasoning) / async disaggregated (agentic) | none | [arXiv](https://arxiv.org/html/2508.06471) |
| RhymeRL | SJTU IPADS + ByteDance | arXiv Aug 2025 | veRL-based streaming; history-based speculative decoding | 84-91% at 16K; >95% at ≥32K | [arXiv](https://arxiv.org/html/2508.18588) |
| PipelineRL | ServiceNow + Mila | arXiv Sep 2025 | in-flight weight updates | ~2x | [arXiv](https://arxiv.org/html/2509.19128v2) |
| APRIL | AMD + CMU + LMSYS + UCLA | arXiv Sep 2025 | over-provisioned partial rollout in slime | +22.5% avg | [arXiv](https://arxiv.org/html/2509.18521v3) |
| RL in the Wild | USTC + Shanghai AI Lab + NTU + Tsinghua + SJTU | arXiv Sep 2025 | characterization | coding >80%, video 30-40% | [arXiv](https://arxiv.org/html/2509.25279v1) |
| RollPacker | HKUST + Alibaba | NSDI 2026 (arXiv Sep 2025) | sync tail batching | ~70% (14B/16K) | [NSDI'26](https://www.usenix.org/system/files/nsdi26-gao-wei.pdf) |
| Laminar | HKU + ByteDance Seed | EuroSys 2026 (arXiv Oct 2025) | trajectory-level async, relay workers | up to 83.1% | [arXiv](https://arxiv.org/html/2510.12633v1), [HKU record](https://hub.hku.hk/handle/10722/375626) |
| fabric-lib / Perplexity weight transfer | Perplexity | arXiv Oct 2025 + blog | RDMA P2P weight sync | 1T in ~1.3 s | [blog](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds) |
| Seer | Moonshot AI + Tsinghua | arXiv Nov 2025 (v3 Apr 2026; venue unconfirmed) | sync colocated; divided rollout + grouped SD | 63-87% | [arXiv](https://arxiv.org/html/2511.14617) |
| TLT | MIT + NVIDIA + ETH + UMass/MIT-IBM | ASPLOS 2026 (arXiv Nov 2025) | adaptive drafter for long tail | ~85% | [arXiv](https://arxiv.org/html/2511.16665), [DOI](https://doi.org/10.1145/3779212.3790231) |
| Awex | Ant Group (inclusionAI) | open source, Nov 2025 | weight exchange | 1T in 6 s (RDMA) | [repo](https://github.com/inclusionAI/asystem-awex) |
| DeepSeek-V3.2 | DeepSeek | arXiv Dec 2025 | GRPO, unified RL stage | RL budget >10% of pre-training | [arXiv](https://arxiv.org/html/2512.02556) |
| LMSYS P2P weight update | LMSYS / SGLang | blog Apr 2026 | Mooncake RDMA P2P | Kimi-K2 53→7.2 s | [blog](https://lmsys.org/blog/2026-04-29-p2p-update/) |
| NeMo-RL speculative decoding | NVIDIA (inferred from NeMo-RL authorship; affiliations not on page) | arXiv Apr 2026 | sync and async spec-decoding in rollout | gen 66-72%; step 1.35-1.41x | [arXiv](https://arxiv.org/html/2604.26779) |
| ROSE | HKUST + Alibaba (inferred from author overlap with RollPacker; not confirmed on page) | arXiv May 2026 | rollout on serving GPUs, agentic | 1.3-3.3x e2e | [arXiv](https://arxiv.org/abs/2605.06534) |
| DynaRL | PKU + Infinigence AI + Tsinghua + SJTU et al. | OSDI 2026 | dynamic resource reallocation | up to 60% compute wasted by tail | [OSDI'26](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing) |

### Inferences
- Top systems venues accepted RL-rollout papers in 2025-2026: NSDI'25 and '26, EuroSys'25 and '26, ASPLOS'26, OSDI'26, MLSys'25. Each of them states that rollout or long-tail generation is the bottleneck. That suggests reviewers at these venues already treat "rollout dominates RL step time" as accepted background, and a new rollout paper will be judged on end-to-end gains over these baselines (verl, RLHFuse, AReaL, StreamRL, RollPacker, Laminar).

### Gaps
- Affiliations for NeMo-RL spec-decoding and ROSE are inferred, not read from the page.
- ReaL's own time breakdown was not extracted.
- Venue status for StreamRL, Seer, LlamaRL, RhymeRL, ROLL Flash, PipelineRL, and APRIL beyond arXiv was not confirmed.
- I did not fetch the NeurIPS proceedings to check whether the Feng Yao et al. NeurIPS 2025 item is main track or a workshop.
