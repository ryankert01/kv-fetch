# Rollout (generation-phase) acceleration for LLM RL post-training: technique catalog and open problems

Scope note: covers 2024-2026 work (search date 2026-10-29). Sources were arXiv abstracts (via the arXiv API), arXiv HTML full text for the core papers, conference pages (USENIX OSDI '26 / NSDI '25, NeurIPS 2025, MLSys 2026), and vLLM / ServiceNow / NVIDIA blog posts. "arXiv-only" means I found no peer-reviewed venue. A venue marked "per curated list" comes only from a GitHub paper list maintained by a RollPacker co-author ([mental2008/awesome-papers](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md)), not from proceedings, so it should be re-checked. Affiliations marked "(inferred)" come from author names, not the paper. Every speedup is the paper's own number against the paper's own baseline.

## Q0. Catalog: venue, date, affiliation, status (reference table for all later sections)

### Takeaway
The peer-reviewed core is small: NSDI '25 (RLHFuse), EuroSys '25 (HybridFlow/verl), NeurIPS '25 (AReaL), ASPLOS '26 (TLT; RhymeRL per curated list), NSDI '26 (RollPacker per curated list), MLSys '26 (DAS; ReSpec and HetRL per curated list), OSDI '26 (Seer, RollArt, RLinf, Weave, DynaRL, RobustRL), ICLR '26 (QuRL), ICML '26 (DARTS), ATC '26 (PrefixFlow) and NeurIPS '26 (EfficientRollout). Most of the 2026 work, especially on speculative rollouts and quantized rollouts, is still arXiv-only.

### Cited Findings
| Technique / system | Date (arXiv v1) | Venue status | Affiliation | Source |
|---|---|---|---|---|
| Kimi k1.5 (partial rollouts) | 2025-01-22 | arXiv tech report | Kimi Team (Moonshot AI) | [arXiv 2501.12599](https://arxiv.org/abs/2501.12599) |
| RLHFuse (stage fusion) | 2024-09-20 | NSDI '25 | Peking Univ. + StepFun | [USENIX NSDI '25](https://www.usenix.org/conference/nsdi25/presentation/zhong) |
| HybridFlow / verl | 2024-09-28 | EuroSys '25 | ByteDance (volcengine) | [verl README](https://github.com/volcengine/verl); [arXiv 2409.19256](https://arxiv.org/abs/2409.19256) |
| StreamRL | 2025-04-22 | arXiv-only (no venue found) | authors overlap with RLHFuse team; affiliation not verified | [arXiv 2504.15930](https://arxiv.org/abs/2504.15930) |
| AReaL | 2025-05-30 | NeurIPS 2025 (main track) | Tsinghua IIIS, Ant Group, HKUST | [NeurIPS 2025 proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/33c00862bfa29ac72ecf630a41e19352-Abstract-Conference.html); [arXiv HTML](https://arxiv.org/html/2505.24298v5) |
| LlamaRL | 2025-05-29 | arXiv-only | Meta (Llama 3 post-training) | [arXiv 2505.24034](https://arxiv.org/abs/2505.24034) |
| UloRL (segment rollout) | 2025-07-26 | arXiv-only | not verified | [arXiv 2507.19766](https://arxiv.org/abs/2507.19766) |
| RhymeRL (HistoSpec, HistoPipe) | 2025-08-26 | ASPLOS '26 per curated list (retitled "History Doesn't Repeat Itself but Rollouts Rhyme") | SJTU IPADS + ByteDance | [arXiv HTML](https://arxiv.org/html/2508.18588v1); [curated list](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md) |
| APRIL (active partial rollouts) | 2025-09-23 | arXiv-only | AMD, CMU, LMSYS, UCLA | [arXiv HTML](https://arxiv.org/html/2509.18521v3) |
| PipelineRL (in-flight weight updates) | 2025-09-23 | arXiv-only | ServiceNow | [arXiv 2509.19128](https://arxiv.org/abs/2509.19128) |
| RollPacker (tail batching) | 2025-09-25 | NSDI '26 per curated list (retitled "Taming Long-Tail Rollouts for RL Post-Training with Tail Batching") | HKUST + Alibaba | [arXiv HTML](https://arxiv.org/html/2509.21009v1); [curated list](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md) |
| SPEC-RL | 2025-09-27 | arXiv-only | Xiamen Univ., Tsinghua, Shopee | [arXiv HTML](https://arxiv.org/html/2509.23232v3) |
| FastGRPO | 2025-09-26 | "Submitted to ICLR 2026" (outcome not found) | not verified | [arXiv 2509.21792](https://arxiv.org/abs/2509.21792) |
| RLinf | 2025-09-19 | OSDI '26 | Tsinghua, Infinigence AI, PKU, others | [OSDI '26 program](https://www.usenix.org/conference/osdi26/technical-sessions); [arXiv 2509.15965](https://arxiv.org/abs/2509.15965) |
| QeRL (NVFP4 + LoRA) | 2025-10-13 | arXiv-only | NVIDIA/NVlabs, MIT, HKU and others | [arXiv 2510.11696](https://arxiv.org/abs/2510.11696) |
| ROLL Flash | 2025-10-13 | arXiv-only | Alibaba ROLL team (inferred) | [arXiv 2510.11345](https://arxiv.org/abs/2510.11345) |
| Laminar | 2025-10-14 | arXiv-only (template placeholder, no venue) | HKU + ByteDance Seed | [arXiv HTML](https://arxiv.org/html/2510.12633v1) |
| RLBoost (preemptible/spot GPUs) | 2025-10-22 | arXiv-only | UC Berkeley, UMich, Google, CMU, UW | [arXiv HTML](https://arxiv.org/html/2510.19225v3) |
| ReSpec | 2025-10-30 | MLSys '26 per curated list | NTU/Shanghai AI Lab (inferred) | [arXiv 2510.26475](https://arxiv.org/abs/2510.26475) |
| FP16 vs BF16 mismatch | 2025-10-30 | arXiv-only (no venue found) | Sea AI Lab / NUS (inferred) | [arXiv 2510.26788](https://arxiv.org/abs/2510.26788) |
| CoPRIS | 2025-11-05 | arXiv-only | not verified | [arXiv 2511.05589](https://arxiv.org/abs/2511.05589) |
| DAS (distribution-aware SD) | 2025-11-17 | MLSys '26 (oral) | UIUC, Together AI, UCSD, Prime Intellect | [MLSys 2026 oral page](https://mlsys.org/virtual/2026/oral/3766); [arXiv HTML](https://arxiv.org/html/2511.13841v1) |
| Seer | 2025-11-18 | OSDI '26 | Moonshot AI + Tsinghua | [USENIX OSDI '26](https://www.usenix.org/conference/osdi26/presentation/qin) |
| TLT (adaptive drafter) | 2025-11-20 | ASPLOS '26 (stated in paper) | MIT, NVIDIA, ETH Zurich, MIT-IBM/UMass | [arXiv HTML](https://arxiv.org/html/2511.16665v3) |
| SpecActor | 2025-11-20 | arXiv-only | SJTU IPADS + ByteDance (inferred) | [arXiv 2511.16193](https://arxiv.org/abs/2511.16193) |
| Periodic Asynchrony | 2025-11-24 | arXiv-only | not verified | [arXiv 2511.18871](https://arxiv.org/abs/2511.18871) |
| RollMux | 2025-12-12 | arXiv-only | Alibaba/HKUST group (inferred) | [arXiv 2512.11306](https://arxiv.org/abs/2512.11306) |
| RollArt | 2025-12-27 | OSDI '26 | HKUST, Alibaba, Tongyi Lab | [OSDI '26 program](https://www.usenix.org/conference/osdi26/technical-sessions) |
| SRT (tree-cache speculative rollout) | 2026-01-14 | arXiv-only | not verified | [arXiv 2601.09083](https://arxiv.org/abs/2601.09083) |
| Jet-RL (unified FP8) | 2026-01-20 | arXiv-only | MIT Han lab / NVIDIA (inferred) | [arXiv 2601.14243](https://arxiv.org/abs/2601.14243) |
| FP8-RL (verl FP8 stack) | 2026-01-26 | arXiv-only | NVIDIA (inferred) | [arXiv 2601.18150](https://arxiv.org/abs/2601.18150) |
| QuRL | 2026-02-15 | ICLR 2026 (arXiv comment) | NVIDIA / Yale (inferred) | [arXiv 2602.13953](https://arxiv.org/abs/2602.13953) |
| SortedRL | 2026-03-24 | arXiv-only | not verified | [arXiv 2603.23414](https://arxiv.org/abs/2603.23414) |
| Heddle (agentic rollout) | 2026-03-30 | arXiv-only | PKU (inferred) | [arXiv 2603.28101](https://arxiv.org/abs/2603.28101) |
| NeMo-RL speculative decoding | 2026-04-29 | arXiv-only | NVIDIA | [arXiv 2604.26779](https://arxiv.org/abs/2604.26779) |
| DORA (multi-version streaming) | 2026-04-29 | arXiv-only | Meituan LongCat (inferred) | [arXiv 2604.26256](https://arxiv.org/abs/2604.26256) |
| BubbleSpec | 2026-05-09 | arXiv-only | not verified | [arXiv 2605.08862](https://arxiv.org/abs/2605.08862) |
| DARTS | 2026-05-29 | ICML 2026 (arXiv comment) | PKU group (inferred) | [arXiv 2605.30859](https://arxiv.org/abs/2605.30859) |
| PrefixFlow | 2026-05-31 | ATC 2026 (arXiv comment) | not verified | [arXiv 2606.01143](https://arxiv.org/abs/2606.01143) |
| Bebop (MTP + rejection sampling) | 2026-06-10 | arXiv-only | Qwen team (inferred) | [arXiv 2606.12370](https://arxiv.org/abs/2606.12370) |
| EfficientRollout | 2026-06-17 | NeurIPS 2026 (arXiv comment) | FuriosaAI and others | [arXiv 2606.18967](https://arxiv.org/abs/2606.18967) |
| WAR | 2026-07-19 | arXiv-only | not verified | [arXiv 2607.17299](https://arxiv.org/abs/2607.17299) |
| TailSieve | 2026-08-24 | arXiv-only | not verified | [arXiv 2608.22788](https://arxiv.org/abs/2608.22788) |
| Online Draft Co-Training | 2026-09-07 | arXiv tech report | NVIDIA NeMo-RL | [arXiv 2609.07108](https://arxiv.org/abs/2609.07108) |
| GrowMTP | 2026-09-15 | arXiv-only | not verified | [arXiv 2609.16648](https://arxiv.org/abs/2609.16648) |
| RollVerify | 2026-10-07 | arXiv-only | not verified | [arXiv 2610.09914](https://arxiv.org/abs/2610.09914) |
| KV4-RL | 2026-10-10 | arXiv-only | KAIST (inferred) | [arXiv 2610.14448](https://arxiv.org/abs/2610.14448) |
| ReFlow (replay) | 2026-10-15 | arXiv-only | not verified | [arXiv 2610.21640](https://arxiv.org/abs/2610.21640) |
| NgramRL | 2026-10-23 | arXiv-only | UCSD (inferred) | [arXiv 2610.32881](https://arxiv.org/abs/2610.32881) |

- Other OSDI '26 RL-post-training papers, known by title only: "Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training" (HKUST, UIUC, Alibaba), "DynaRL: Flexible and Dynamic Scheduling of Large-Scale RL Training" (PKU/Infinigence AI and others) and "RobustRL: Role-Based Fault Tolerance System for RL Post-Training" (Zhejiang Univ.) — [OSDI '26 program](https://www.usenix.org/conference/osdi26/technical-sessions)
- HetRL ("Efficient RL for LLMs in Heterogeneous Environments") is listed as MLSys '26 — [curated list](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md)
- Truncated Importance Sampling (Feng Yao, Liyuan Liu et al.) appeared as a blog in Aug 2025 and as a poster at the OPT 2025 workshop (NeurIPS 2025 workshop, not main track) under the title "On the Rollout-Training Mismatch in Modern RL Systems" — [NeurIPS virtual page](https://neurips.cc/virtual/2025/124259); [author page](https://liyuanlucasliu.github.io/publication/feng-off-2025)
- Seer's numbers changed between arXiv versions. v1 reported 74–97% throughput gains and 75–93% tail reduction; v3 and the OSDI page report up to 2.04× and 72–94% — [search summary of OSDI page and arXiv versions](https://www.usenix.org/conference/osdi26/presentation/qin); [arXiv v3 abstract](https://arxiv.org/abs/2511.14617)
- One survey of rollout strategies exists: "Generate, Filter, Control, Replay" (GFCR taxonomy, Apr 2026, arXiv-only) — [arXiv 2605.02913](https://arxiv.org/abs/2605.02913)

### Inferences
- Systems venues (OSDI/NSDI/ASPLOS/MLSys 2026) have accepted the first generation of rollout-acceleration systems: tail scheduling (RollPacker), history-based speculation and scheduling (RhymeRL, DAS), adaptive drafters (TLT, ReSpec) and context-aware scheduling plus grouped SD (Seer). A new paper reusing these ideas will need a clearly different angle.

### Gaps
- I could not confirm venues for StreamRL, Laminar, RLBoost, SpecActor, SPEC-RL, FastGRPO, QeRL, the FP16 paper, GRESO, or APRIL. The OSDI '26 program page was only partly read (about 200K of 303K characters), and SOSP '26, EuroSys '26 and ATC '26 programs were not checked systematically.
- RhymeRL (ASPLOS '26), RollPacker (NSDI '26), ReSpec and HetRL (MLSys '26) venues come only from the curated list, not from proceedings.

## Q1. Long-tail mitigation: partial rollouts, over-provisioning, early termination, length-aware scheduling, tail batching, asynchrony

### Takeaway
Long-tail work falls into three families: (a) relax synchrony (partial rollouts, async, staleness-bounded), (b) stay synchronous but reorder or isolate long requests (tail batching, length-aware or context-aware scheduling, repack, tail routing), and (c) shrink the tail itself (distribution shaping, pruning, segment rollout). Reported end-to-end gains cluster at about 1.4–2.8× over veRL-style synchronous baselines. The largest number, 5.48× (Laminar), was measured at 1024-GPU scale.

### Cited Findings
**Characterization numbers (why the tail matters)**
- APRIL: rollout generation takes "more than 90% of total runtime" in its setting — [APRIL abstract](https://arxiv.org/abs/2509.18521)
- TLT: rollout is about 85% of RL step time. A 32B reasoning RL run took 11 days for 385 steps on 128 GPUs, about 40 minutes per step — [TLT HTML](https://arxiv.org/html/2511.16665v3)
- RollPacker: rollout is 66–72% of step time across math, code and LLM-judge tasks. The longest response is 25–32× the median, P75 is 755–1.1k tokens and the cap is 16k — [RollPacker HTML](https://arxiv.org/html/2509.21009v1)
- Seer, Qwen2-VL-72B on veRL, one iteration: the last 5% of requests started at 42% of total time on average, instances sat idle 1,580 s (37% of time) on average, and the long-tail phase was nearly 50% of rollout time — [Seer HTML](https://arxiv.org/html/2511.14617v3)
- DORA: rollout is 50–80% of step time — [DORA abstract](https://arxiv.org/abs/2604.26256)

**Partial rollouts / over-provisioning / early termination**
- Kimi k1.5 partial rollouts: each rollout has a fixed output-token budget. Unfinished trajectories go to a replay buffer and continue in later iterations, so long responses span several iterations as segments. Only the current iteration's segment needs on-policy computation, and some segments can be excluded from the loss. A repeat detector ends degenerate generations early. The report gives no quantitative speedup for partial rollouts. Infrastructure is Megatron and vLLM sidecars linked by "checkpoint-engine", with weights moved via Mooncake — [Kimi k1.5 HTML](https://arxiv.org/html/2501.12599v4)
- APRIL over-provisions 2× (over_sampling_batch_size 64 vs rollout_batch_size 32), stops once 256 rollouts finish, and puts unfinished rollouts in a continuation buffer that gets priority next step. It applies no importance-sampling correction. About 40% of tokens in later steps come from carried-over instances, and up to 5 policy versions mix in one trajectory. Results: throughput +22.5% average (max 44% in the abstract; text lists up to +49.5% for Qwen3-8B GRPO) and +2.1% average final accuracy (max 8%) vs standard synchronous rollout. Setup: single node, 8×H100 or 8×AMD MI300, Qwen3-4B/8B, integrated in slime. The fetch noted that Table 1's "Avg Imp." column does not match the text — [APRIL HTML](https://arxiv.org/html/2509.18521v3)
- CoPRIS keeps a fixed number of concurrent rollouts, terminates early once enough samples are collected, and reuses unfinished trajectories. It adds "cross-stage importance sampling correction" by concatenating buffered old-policy logprobs with recomputed current ones. Up to 1.94× faster training vs synchronous RL — [CoPRIS abstract](https://arxiv.org/abs/2511.05589)
- RollVerify (Oct 2026) builds on partial rollout. It defines an off-policy shift metric (OPS) and verifies or truncates invalid suffixes of stale partial trajectories before training, instead of reweighting them. It claims accuracy comparable to on-policy at lower cost, with no single speedup number in the abstract — [RollVerify abstract](https://arxiv.org/abs/2610.09914)
- UloRL splits ultra-long decoding (up to 128k-token outputs) into short segments. Segment rollout gave 2.06× training speed on Qwen3-30B-A3B — [UloRL abstract](https://arxiv.org/abs/2507.19766)
- Seer's evaluation found Seer had 43% higher throughput than Partial Rollout. Partial rollout's doubled request volume made preemption worse — [Seer HTML](https://arxiv.org/html/2511.14617v3)

**Tail batching / synchronous reordering**
- RollPacker tail batching: a short round launches η·P0 prompts × η·R0 responses (η=1.25; e.g., 160 prompts × 10 responses for P0=128, R0=8). It keeps the first P0 prompts to finish and the first R0 responses each, and aborts the rest through an extended vLLM abort_request. Aborted prompts queue up and run together as a "long round" with speculation off (e.g., 4 short rounds + 1 long round). The authors argue this only reorders prompts, so training stays on-policy; validation curves are nearly identical to veRL. End-to-end: 2.03× (7B), 2.22× (14B), 2.56× (32B) vs veRL, and 1.14×, 1.68×, 2.24× vs RLHFuse (the authors' augmented version). Hardware: 16×8 H800 (128 GPUs), Qwen2.5 7B/14B/32B with max lengths 8k/16k/32k. Tail batching alone cuts max short-round length by up to 8.9× and average rollout time by up to 3.9× — [RollPacker HTML](https://arxiv.org/html/2509.21009v1)
- Seer (OSDI '26): (1) divided rollout splits each GRPO group into individual requests and each request into chunks, rescheduled at chunk boundaries onto the instance with the most free memory and compute; (2) context-aware scheduling runs one "speculative" request per group first (shortest-first) and uses the longest finished sibling as the group's length estimate, approximating longest-first; (3) grouped SD (see Q2). Testbed: 32 nodes × 8 H800 (256 GPUs). Workloads: Moonlight (32 GPUs), Qwen2-VL-72B (128 GPUs), Kimi-K2 (256 GPUs), average generation lengths 22k/7.6k/39k tokens. Throughput is +44–104% over veRL (up to 2.04×) and tail latency is 72–94% lower. Ablation for divided rollout / +context scheduling / +grouped SD: 1.41/1.47/1.90× (Moonlight), 1.42/1.56/2.04× (Qwen2-VL-72B), 1.16/1.27/1.53× (Kimi-K2). Context scheduling reached 96% of oracle throughput — [Seer HTML](https://arxiv.org/html/2511.14617v3)
- RhymeRL HistoPipe ranks prompts by historical median response length and splits them into ranking groups. Tier 1 divides prompts equally; tier 2 allocates GPUs per group by binary search over a profiled time model. Adjacent steps alternate ascending and descending assignment so long and short work complement each other. Long outliers migrate within a step or defer to the next step. HistoPipe is off in epoch 1. Only 2–4% of responses show significant rank changes across epochs — [RhymeRL HTML](https://arxiv.org/html/2508.18588v1)
- StreamRL ranks outputs by length to find long-tail samples and uses "skewness-aware dispatching and scheduling" in a disaggregated design. Up to 2.66× throughput vs existing systems, and 1.33× cost-effectiveness in a heterogeneous cross-datacenter setting — [StreamRL abstract](https://arxiv.org/abs/2504.15930)
- SortedRL reorders rollouts by output length, updates early on groups of short samples, and bounds off-policyness with a cache. More than 50% lower bubble ratio and 3.9–18.4% better performance at equal data (LLaMA-3.1-8B, Qwen-2.5-32B) — [SortedRL abstract](https://arxiv.org/abs/2603.23414)
- TailSieve (Aug 2026): shows that makespan-optimal routing with known lengths combines tail isolation with load balancing, and that a top-k policy approximates it. It uses partial rollouts as a training-free tail signal, because long-tail prompts stay long-tailed across updates. Routing alone is up to 1.67× faster than uniform group routing, and up to 2.59× when combined with MTP/DFlash SD in the low-concurrency tail pool. Selected prompts are regenerated under the current policy so generation stays on-policy — [TailSieve abstract](https://arxiv.org/abs/2608.22788)
- RLHFuse (NSDI '25) splits generation and inference into sample-level subtasks and fuses stages to overlap long-tail generation with inference. Up to 3.7× training throughput — [USENIX page](https://www.usenix.org/conference/nsdi25/presentation/zhong)
- RolloutPipe (Jun 2026) pipelines complete GRPO groups to the trainer while later groups are still generating, keeping on-policy correctness. Rollout-to-train-end time is 30.7–42.3% shorter vs slime (Qwen3-1.7B) — [RolloutPipe abstract](https://arxiv.org/abs/2606.26997)

**Asynchronous / off-policy approaches**
- AReaL (NeurIPS '25) is fully asynchronous. On a weight update, rollout workers interrupt generation, "discard KV caches computed by old weights, and re-compute them using the new weights", then keep decoding, so trajectories can mix policy versions. A staleness bound η rate-limits generation (η=4 for code, η=8 for math), and a decoupled PPO objective separates behavior and proximal policies. Up to 2.77× training speedup vs synchronous systems on the same GPUs. Table 1 vs verl: 1.5B 33.6→14.8 h, 7B 52.1→25.4 h, 14B 44.4→21.9 h, 32B 46.4→31.1 h. Hardware: H800 cluster (64×8 GPUs), R1-Distilled-Qwen 1.5B–32B — [AReaL HTML](https://arxiv.org/html/2505.24298v5)
- Laminar uses trajectory-level asynchrony with CPU relay workers as a distributed parameter service, so rollouts pull the latest weights at any time. Dynamic repack uses best-fit bin packing on KV-cache utilization to consolidate long-tail trajectories onto fewer rollouts (0.69 s overhead, about 26% higher generation throughput, KV utilization 82.2% vs 71.6%). On 1024 H800 GPUs with Qwen2.5 7B/32B/72B (math): 2.56× average over verl (up to 5.48×; Section 8.1 says 5.49×) and 1.39× over AReaL. Staleness is not configured and was at most 4 — [Laminar HTML](https://arxiv.org/html/2510.12633v1)
- PipelineRL uses in-flight weight updates. About 2× faster learning vs conventional RL baselines on 128 H100s — [PipelineRL abstract](https://arxiv.org/abs/2509.19128)
- ROLL Flash: up to 2.24× (RLVR) and 2.72× (agentic) vs synchronous baselines at the same GPU budget — [ROLL Flash abstract](https://arxiv.org/abs/2510.11345)
- LlamaRL: up to 10.7× vs DeepSpeed-Chat-like systems on a 405B policy — [LlamaRL abstract](https://arxiv.org/abs/2505.24034)
- DORA (multi-version streaming rollout): 2–3× throughput vs SOTA on open benchmarks, and 2–4× vs synchronous training in industrial runs on "tens of thousands of accelerators" (LongCat-Flash-Thinking) — [DORA abstract](https://arxiv.org/abs/2604.26256)
- Periodic Asynchrony syncs weights at each iteration start, so it stays on-policy. About 2× throughput on NPUs and up to 3× on GPUs vs mainstream frameworks — [Periodic Asynchrony abstract](https://arxiv.org/abs/2511.18871)
- RhymeRL compared against AReaL: up to 2.1× vs AReaL at off-policyness 1 and up to 1.6× at off-policyness 8. Against veRL v0.4.1 it was up to 2.6×. Hardware: 128 GPUs (model not named), Qwen3-8B/14B and Qwen2.5-32B — [RhymeRL HTML](https://arxiv.org/html/2508.18588v1)

**Shrinking the tail itself**
- DARTS (ICML '26) samples redundant trajectories per prompt and selects concise, certain ones ("active distribution shaping") after finding that intra-prompt long tails are often "ineffective verbosity". Up to 1.77× over SOTA systems — [DARTS abstract](https://arxiv.org/abs/2605.30859)
- arrol prunes rollouts during generation with an online quality head. Up to 1.7× training speedup and +2.30 to +2.99 average accuracy (1B–8B) — [arrol abstract](https://arxiv.org/abs/2603.24840)

### Inferences
- Over-provisioning (APRIL, RollPacker's η=1.25, Kimi partial rollouts) raises concurrent KV demand. Seer and RollPacker both measured more preemptions as a result (Q6). The balance between over-provisioning and memory looks under-explored.
- "Exactly on-policy" designs (RollPacker, Seer, TailSieve, RolloutPipe, BubbleSpec) are a separate design point from partial or async ones (APRIL, Kimi, AReaL, Laminar, CoPRIS). Several 2026 papers (RollVerify, "Missing Old Logits") are now repairing the off-policy side effects of partial rollouts.

### Gaps
- Kimi k1.5 gives no quantitative benefit for partial rollouts, and I found no public number for Kimi's token budget.
- No paper I found compares partial rollout, tail batching, context-aware scheduling and async head-to-head on one testbed, apart from pairwise comparisons (Seer vs Partial Rollout, RhymeRL vs AReaL, Laminar vs AReaL).

## Q2. Speculative decoding for rollouts (drafters, n-gram/suffix from prior rollouts, online drafters) and exactness

### Takeaway
This is the most crowded sub-area: at least 17 papers from Aug 2025 to Oct 2026. They split into model-free history drafters (RhymeRL HistoSpec, DAS, SRT, NgramRL, Seer's grouped suffix trees, WAR, BubbleSpec), learned drafters trained online (TLT, ReSpec, FastGRPO, GrowMTP, NVIDIA co-training, Bebop/MTP), self-speculation (EfficientRollout) and lossy prefix reuse (SPEC-RL with lenience ℓ>1). Most claim losslessness through standard target-model verification. SPEC-RL with ℓ>1 deliberately breaks exactness, and ReSpec's reward-weighted drafter updates do not change the sampled distribution. Common constraints in nearly every paper: SD helps only at low concurrency (the long tail), and drafters go stale as the policy changes.

### Cited Findings
**Model-free / history-based drafters**
- RhymeRL HistoSpec drafts from the same prompt's previous-epoch responses through a per-prompt suffix tree whose nodes carry reward-sum priorities. The draft window grows AIMD-style (start 2, +2 up to 32, reset on rejection). Speculation turns off above a batch-size threshold. Acceptance was 65–79%, and 75–95% of tokens are reusable (93% math, about 75% code over 8 epochs). HistoSpec gave up to 1.86× rollout throughput, and 1.80–1.86× over Eagle3 at very large batch sizes (Eagle3 degrades beyond batch 64 per the paper). The authors say concurrently trained draft models "often failed" to give consistently accurate drafts — [RhymeRL HTML](https://arxiv.org/html/2508.18588v1)
- DAS (MLSys '26) builds a nonparametric suffix tree (incremental Ukkonen) over a sliding window of recent rollouts, per problem or request (a global tree did worse). A length-aware policy gives long requests larger draft budgets, sorted into Long/Medium/Short classes, using a latency model of base cost per forward plus per-token cost (about 12% mean relative error). Math rollout time dropped by up to 50% on 8×H100 (DeepSeek-R1-Distill-Qwen-7B). Code rollout time dropped about 25% (Section 5.2) or "over 30%" (conclusion), which the paper itself reports inconsistently. Training curves are identical. The paper calls the method lossless but the fetched text does not spell out the acceptance rule — [DAS HTML](https://arxiv.org/html/2511.13841v1)
- SRT keeps a per-prompt tree cache of earlier continuations as the draft, updates it online, and does "run-ahead generation during idle GPU bubbles". Up to 2.08× wall-clock rollout speedup, with "distributional correctness" claimed — [SRT abstract](https://arxiv.org/abs/2601.09083)
- Seer's adaptive grouped SD: a distributed grouped draft server keeps a compressed suffix tree per GRPO group, built from all sibling requests' tokens (SuffixDecoding-style, with multi-path beam). Draft budget is split by marginal benefit (γmax=8, λ=2). Up to 1.3× over vanilla SD, and +0.22 mean acceptance length over CST-based SD. Losslessness is asserted in general SD terms; the fetched text gives no specific argument for multi-path drafts — [Seer HTML](https://arxiv.org/html/2511.14617v3)
- BubbleSpec (May 2026) uses idle time on fast data-parallel ranks to pre-generate rollouts for later steps and uses them as drafts. It needs no history or warm-up and keeps "strict synchronous" exactness. 50% fewer decoding steps and up to 1.8× rollout throughput — [BubbleSpec abstract](https://arxiv.org/abs/2605.08862)
- NgramRL (Oct 2026) uses a per-prompt n-gram drafter from bounded recent history, with CPU-side lookup hidden by an adaptive fan-out prefetcher. It commits 3.60–7.77 tokens per target forward in the warm-cache regime: up to 1.99× more than SuffixDecoding and 7.0× more than prompt-lookup. 1.38× rollout throughput over synchronous NgramRL on Spider (Qwen3-8B) — [NgramRL abstract](https://arxiv.org/abs/2610.32881)
- WAR (agentic, synchronous) uses SuffixDecoding over earlier trajectories under low load and switches to cache-aware scheduling under high load. 1.4× rollout throughput under low load, up to 1.6× under high load — [WAR abstract](https://arxiv.org/abs/2607.17299)

**Learned / adaptive drafters**
- TLT (ASPLOS '26) uses a one-layer EAGLE-style drafter that reuses the target's frozen embedding and LM head. A "Spot Trainer" trains it on GPUs that go idle during the long tail, from cached hidden states. Accuracy dips after each policy update and recovers within a few drafter iterations, and training every 10 steps is enough. The adaptive rollout engine pre-captures CUDA graphs per batch-size bucket (peak graph memory 30.39 GB naive → 10.69 GB) and picks the SD config with a bucketed epsilon-greedy bandit. SD turns on only when fewer than 32 requests remain (default), with an n-gram fallback. End-to-end 1.7–2.1× vs VeRL on 64 H100s (Qwen2.5-7B/32B, DeepSeek-R1-Distill-Qwen-7B, Llama-3.3-70B), and 2.44× rollout in a Qwen-32B, 128-request case study. Switching to SD costs about 3 s of re-prefill. HASS/EAGLE-3 drafters cost 3× and 7× more to train per step, so EAGLE was chosen — [TLT HTML](https://arxiv.org/html/2511.16665v3)
- ReSpec (MLSys '26 per curated list) names three gaps for SD in RL: diminishing speedup at large batch, drafter staleness, and drafter-induced policy degradation. It addresses them with dynamic SD config tuning, a drafter evolved by knowledge distillation, and updates weighted by rollout reward. Up to 4.5× speedup on Qwen 3B–14B — [ReSpec abstract](https://arxiv.org/abs/2510.26475)
- FastGRPO uses concurrency-aware SD (drafting and verification adjusted to live concurrency) plus online draft learning. End-to-end 2.35–2.72× — [FastGRPO abstract](https://arxiv.org/abs/2509.21792)
- SpecActor: "decoupled speculation" for large per-worker batches plus "Fastest-of-N" draft selection by rollout progress. Mean rollout 2.0–2.4× faster (up to 2.7×) on production traces, 1.1–2.6× faster than vanilla speculative rollout, and 1.4–2.3× faster end-to-end — [SpecActor abstract](https://arxiv.org/abs/2511.16193)
- NVIDIA NeMo-RL with a vLLM backend supports MTP heads, small draft models and Eagle3 in sync and async pipelines. 1.8× rollout throughput at 8B (synchronous). Up to 2.5× end-to-end at 235B when combined with async RL, but this is a simulator projection, not a measurement — [arXiv 2604.26779](https://arxiv.org/abs/2604.26779)
- NVIDIA online draft co-training (Sep 2026) extends zigzag ring attention for branch attention under context parallelism and adds "TapChannel" to carry target features across pipeline stages. It reports "substantial" speedups up to 122B models and strong scaling at 256K tokens, with no single number in the abstract — [arXiv 2609.07108](https://arxiv.org/abs/2609.07108)
- Bebop (Qwen team, Jun 2026): MTP acceptance falls roughly linearly as entropy rises during RL. Probabilistic rejection sampling handles this better than greedy draft acceptance. An end-to-end TV loss raises acceptance about 10% (up to 95%). Pre-RL MTP training removes the need for online MTP updates. Up to 1.8× end-to-end in async RL of Qwen3.5/3.6/3.7 — [Bebop abstract](https://arxiv.org/abs/2606.12370)
- GrowMTP trains a draft head from scratch inside the RL loop using verification as supervision, detached from the backbone. Rollout 2.13× / 1.93× / 1.36× and end-to-end 1.60× / 1.41× / 1.20× on Qwen3-4B (no head), MiMo-7B-SFT (weak head) and Qwen3.5-4B-Base (strong head) — [GrowMTP abstract](https://arxiv.org/abs/2609.16648)
- EfficientRollout (NeurIPS '26) builds a quantized self-drafter from the target, so the drafter follows the evolving policy without separate training. A system-aware toggle runs SD only in memory-bound regimes. Up to 24.2% lower rollout latency and 15.6% lower end-to-end latency vs an accelerated AR rollout baseline — [EfficientRollout abstract](https://arxiv.org/abs/2606.18967)
- verl ships a pre-release "verl-SpeCo" co-training framework for speculative decoding — [verl README](https://github.com/volcengine/verl)

**Lossy speculative reuse**
- SPEC-RL drafts from the previous epoch's rollout for the same prompt. Each token is accepted with probability min(1, ℓ·π_curr/π_prev); the verified prefix runs up to the first rejection and the current policy generates the rest. ℓ=1 is standard and exact. ℓ>1 deliberately relaxes acceptance, so the distribution is no longer exact; the paper gives no bias bound and reports instability at very large ℓ. Rollout speedups 1.94–2.88× (average 2.31×, about 66% fewer generated tokens) on one 8×H100 node with veRL+vLLM (Qwen3-1.7B/8B, LLaMA-3.2-1B; GRPO/PPO/DAPO), and 1.66–2.46× on Qwen3-14B. End-to-end time drops less than rollout time — [SPEC-RL HTML](https://arxiv.org/html/2509.23232v3)

### Inferences
- Exactness has three tiers: (i) standard rejection-sampling verification, exact at any temperature (TLT, NeMo-RL, Bebop, BubbleSpec); (ii) model-free tree or multi-path drafts claimed lossless, where verification under sampling (not greedy) for multi-path drafts is rarely spelled out (Seer, DAS); (iii) intentionally lossy reuse (SPEC-RL ℓ>1). A rigorous treatment of multi-path, history-based drafts under high-temperature sampling would fill a real gap.
- Nearly every paper makes SD depend on concurrency (TLT under 32 requests, RhymeRL batch threshold, FastGRPO, EfficientRollout toggle, SpecActor's decoupled speculation, WAR's load switch, TailSieve's low-concurrency tail pool). SD in RL is effectively a long-tail-phase technique.

### Gaps
- I found no paper that measures SD speedup for hybrid linear-attention or SSM policies during RL, where verifying drafts requires rolling back recurrent state. The vLLM consistency blog mentions a Mamba-2 hybrid only for numerical consistency (see Q5).
- Head-to-head comparisons are scarce. RhymeRL vs Eagle3 and Seer vs vanilla SD are the main ones found.

## Q3. Request migration, load rebalancing, KV-cache migration cost, dynamic batch sizes, elastic scaling

### Takeaway
Most RL systems migrate requests by moving the generated tokens and recomputing KV with a prefill (RhymeRL, RollPacker, RLBoost; AReaL recomputes after weight updates). RhymeRL explicitly found KV migration costlier than recompute. Seer is the exception: it moves KV through a global Mooncake-based KV pool (DRAM/SSD over RDMA) because chunk-level rescheduling would otherwise force full recompute. Elastic scaling is now its own sub-area (spot GPUs, borrowed serving GPUs, borrowed idle trainer GPUs, cluster-level multiplexing).

### Cited Findings
**Migration and KV handling**
- Seer migrates requests between instances at chunk boundaries using a global KVCache pool adapted from Mooncake (DRAM + SSD, RDMA), so migration needs no re-prefill. The authors argue that without the pool, chunk-level rescheduling forces full KV recomputation and can cancel the load-balancing gains, and that local DRAM offload is not enough because one iteration can produce "tens of terabytes of KVCache" — [Seer HTML](https://arxiv.org/html/2511.14617v3)
- RhymeRL: "Generated tokens are kept, and the KV cache is recomputed by prefill rather than migrated, because the authors found migration overhead exceeded recomputation." 2.2–5.5% of math samples (1.6–4.6% of code) migrated, decreasing over training — [RhymeRL HTML](https://arxiv.org/html/2508.18588v1)
- RollPacker moves stream-trainer GPUs between roles by recomputation (tokens kept, KV rebuilt). Migration overhead is under 3 s and decoding throughput drops under 1% after scaling. Scaling triggers only if the projected peak KV demand (from historical length distributions) still fits after migration — [RollPacker HTML](https://arxiv.org/html/2509.21009v1)
- RLBoost collects responses token by token. On spot preemption, partial responses migrate to a healthy instance, which prefills prompt plus generated tokens. With 3 of 6 instances preempted, this cut step-time overhead vs full regeneration by 33% (preemption at 100 s) and 76% (at 200 s). End-to-end: 1.51–1.97× throughput and 28–49% better cost efficiency vs on-demand-only veRL (on-demand 8×H100 training, spot 2×H100 rollout; Qwen3 8B/14B/32B) — [RLBoost HTML](https://arxiv.org/html/2510.19225v3)
- Laminar's dynamic repack moves unfinished trajectories to destination rollouts (0.69 s overhead on 32B/128 GPUs). The fetched text does not say whether KV is transferred or re-prefilled — [Laminar HTML](https://arxiv.org/html/2510.12633v1)
- AReaL discards and recomputes KV for in-flight sequences after each weight update — [AReaL HTML](https://arxiv.org/html/2505.24298v5)
- Heddle (agentic) does "opportunistic migration during idle tool call intervals" and trajectory-adaptive model-parallelism tuning to speed up per-token time on long-tail trajectories. Up to 2.5× end-to-end rollout throughput — [Heddle abstract](https://arxiv.org/abs/2603.28101)
- TideRL keeps rollout state across tool pauses ("Continuous Task Batching"): 1.58× KV cache hit rate, up to 5.6× goodput vs synchronous baselines and over 33% vs async baselines — [TideRL abstract](https://arxiv.org/abs/2608.10402)

**Dynamic batch / parallelism adaptation**
- RollPacker's planner changes rollout TP per step from online preemption counts: preemptions rising above 1.05× doubles TP, and zero preemptions for 4 steps halves it. Up to 21.9% lower rollout latency, 1.9× vs fixed TP=1 in a microbenchmark. Only TP is covered, not EP — [RollPacker HTML](https://arxiv.org/html/2509.21009v1)
- SD-on/off by batch size: TLT (fewer than 32 remaining requests) — [TLT HTML](https://arxiv.org/html/2511.16665v3); RhymeRL (profiled threshold) — [RhymeRL HTML](https://arxiv.org/html/2508.18588v1); EfficientRollout (system-aware toggle) — [abstract](https://arxiv.org/abs/2606.18967); FastGRPO (concurrency-aware) — [abstract](https://arxiv.org/abs/2509.21792)
- Libra routes requests across "execution buckets with different parallelism configurations" and moves workers between rollout and training without blocking. Up to 4.2× throughput on 48×A800 and 160×Ascend 910B3 — [Libra abstract](https://arxiv.org/abs/2606.03077)

**Elastic scaling and resource harvesting**
- ROSE co-locates rollouts on already-deployed serving GPUs while keeping serving SLOs. 1.3–3.3× end-to-end throughput vs resource-fixed baselines, 1.2–1.5× shorter rollout vs resource-elastic baselines, no SLO violations — [ROSE abstract](https://arxiv.org/abs/2605.06534)
- PEARL coordinates external elasticity, borrowed idle trainer GPUs, and adaptive prefill-decode colocation vs disaggregation and PD ratio. 2.17–2.79× throughput vs fixed-resource ROLL, and up to about 26.9% (Qwen3-8B) and 36.3% (Qwen3-30B-A3B) over RLBoost+ — [PEARL abstract](https://arxiv.org/abs/2609.35158)
- TensorHub (ByteDance production) uses reference-oriented storage for weights. Up to 6.7× less GPU stall for standalone rollouts, 4.8× faster weight updates for elastic rollouts, and 19× less cross-datacenter stall — [TensorHub abstract](https://arxiv.org/abs/2604.09107)
- Cluster-level multiplexing: RollMux, 1.84× cost efficiency vs standard disaggregation and 1.38× vs co-located baselines on 328 H20 + 328 H800 — [RollMux abstract](https://arxiv.org/abs/2512.11306); PlexRL, up to 37.58% lower user GPU-hour cost — [PlexRL abstract](https://arxiv.org/abs/2605.20863)
- SeamlessFlow reassigns idle training nodes to rollout ("spatiotemporal multiplexing") and supports partial rollout through a trajectory manager — [SeamlessFlow abstract](https://arxiv.org/abs/2508.11553)
- QwenGyre (rollouts up to about 700K tokens, Qwen 3.8 2.4T) reallocates GPUs between rollout and training "without interrupting live executions". Up to 1.85× vs Colocate and 1.78× vs Async — [QwenGyre abstract](https://arxiv.org/abs/2609.33848)

### Inferences
- Whether to transfer KV or recompute it for a migrated rollout is decided case by case. Seer (with RDMA plus a global pool) chose transfer; RhymeRL and RLBoost (without one) chose recompute. No paper sweeps the crossover point (context length × model size × interconnect × hybrid-state size) for RL rollouts. That parameterized crossover study looks open, especially for hybrid models whose per-request state is a fixed-size recurrent state plus a smaller full-attention KV.

### Gaps
- Laminar's KV handling during repack is not specified in the fetched text.
- I found no RL paper measuring KV-transfer bandwidth or latency for migrating decoding requests, beyond Seer's design argument and RhymeRL's qualitative finding.

## Q4. Prefix sharing across the n samples of a GRPO group (fork / copy-on-write, prefix caching, memory)

### Takeaway
On the rollout side, sharing the prompt across n samples is a standard engine feature (vLLM PagedAttention copy-on-write, automatic prefix caching). Its only rigorous measurement I found is from serving (SOSP '23): 6.1–9.8% KV memory saved for parallel sampling on Alpaca and 16.2–30.5% on ShareGPT. Recent RL work on shared prefixes targets the training side instead: Prefix Grouper, DualKV, PrefixFlow (ATC '26), Periodic Asynchrony's shared-prompt attention. Prefix caching across weight updates is a correctness hazard.

### Cited Findings
- vLLM (SOSP '23): in parallel sampling, all sequences of a request share the prompt's KV blocks through copy-on-write. Memory saving (blocks saved ÷ blocks without sharing) is 6.1–9.8% for parallel sampling and 37.6–55.2% for beam search on Alpaca, and 16.2–30.5% and 44.3–66.3% on ShareGPT. Sequences in a group are gang-scheduled and "always preempted or rescheduled together due to potential memory sharing" — [vLLM paper, arXiv 2309.06180](https://arxiv.org/abs/2309.06180)
- Prefix caching can reuse KV "computed before a weight update" if the cache ignores update boundaries. ServiceNow's PipelineRL port to vLLM V1 disabled prefix caching (`enable-prefix-caching: false`) for correctness (May 2026) — [ServiceNow/HF blog](https://hface.pages.dev/blog/ServiceNow-AI/correctness-before-corrections)
- Seer schedules a GRPO group's requests separately, but uses the group as the unit for length estimation and for its suffix-tree draft source. Going from group size 8 to 16 gave Seer about 5% average improvement — [Seer HTML](https://arxiv.org/html/2511.14617v3)
- MISA-T (mixed RLVR/RLHF/agentic rollouts on one async service): prefix-aware routing alone does not control how sessions compete for KV capacity. Adaptive admission plus residency-time-aware KV accounting beat a sweep-tuned cache-aware vLLM Router by 53.3% (Step3.7) and 43.6% (Qwen3.6-35B-A3B) in rollout throughput, keeping high prefix-cache hit rates — [MISA-T abstract](https://arxiv.org/abs/2608.11152)
- Training side: Prefix Grouper encodes the shared prefix once with restructured self-attention and is training-equivalent to GRPO — [arXiv 2506.05433](https://arxiv.org/abs/2506.05433). DualKV repacks N(P+R) tokens as P+NR, giving a 1.63–2.09× policy-update speedup on Qwen3-8B (N=32, 8K context, 8×H100) and 3.82× policy-update / 3.38× end-to-end at 30B MoE on 16×H100 — [arXiv 2605.15422](https://arxiv.org/abs/2605.15422). PrefixFlow (ATC '26) gives 1.90× end-to-end on a long-prefix RAG workload in verl, up to 4.4× actor-side at an 83% prefix ratio, and 59% lower peak HBM; "suffix-dominated splits gain little" — [arXiv 2606.01143](https://arxiv.org/abs/2606.01143). Periodic Asynchrony uses a "shared-prompt attention mechanism" — [arXiv 2511.18871](https://arxiv.org/abs/2511.18871)
- Tree-structured rollouts share KV beyond the prompt. TreePO's segment-wise tree sampling "alleviates the KV cache burden" and saves 22–43% GPU hours of the sampling design, with up to 40% less trajectory-level and 35% less token-level sampling compute — [TreePO abstract](https://arxiv.org/abs/2508.17445)
- VenusRL applies copy-on-write to agent sandboxes, not KV: template-keyed page sharing across sandboxes launched from the same prompt. Up to 89% lower environment cost and up to 4.24× end-to-end — [VenusRL abstract](https://arxiv.org/abs/2610.03286)

### Inferences
- In long-reasoning RLVR, responses (8k–40k tokens; e.g., Seer's 22k–39k averages) are far longer than prompts, so sharing the prompt across n samples should save only a small fraction of rollout KV. This explains why the gains show up on the training side (PrefixFlow: "suffix-dominated splits gain little") and in long-prefix RAG, multimodal or multi-turn agentic settings. This is my inference from the cited numbers; no RL paper measures it directly.
- Rollout-side prefix sharing matters more for multi-turn agents (repeated prefill of growing contexts; TideRL, WAR, MISA-T), where KV residency and cache-hit rate are the metric.

### Gaps
- I found no RL paper that measures rollout-side KV memory saved by sharing a GRPO group's prompt (fork/COW), or how group-level gang preemption interacts with long reasoning outputs.

## Q5. Rollout reuse / history-based acceleration and approximate methods (quantized/FP8 rollouts, KV quantization, truncated importance sampling)

### Takeaway
History reuse works because consecutive epochs' rollouts for a prompt are highly similar (RhymeRL: 75–95% of tokens reusable). It is used for exact speculation (RhymeRL, DAS, SRT, NgramRL), lossy prefix reuse (SPEC-RL), prompt skipping (GRESO), scheduling (RhymeRL, TailSieve), and experience replay (ReFlow, RePO). Low-precision rollouts give about 1.2–2× rollout speedups (FP8/INT8/NVFP4/INT4-KV), but they open a training-inference mismatch. That has spawned a correction literature (TIS, MIS, adaptive IS, FP16, unified-precision training, bitwise-consistent kernels), and several 2026 papers argue IS corrections alone are not enough.

### Cited Findings
**History / reuse**
- SPEC-RL (lossy for ℓ>1), RhymeRL HistoSpec, DAS, SRT, NgramRL: see Q2.
- GRESO skips prompts predicted to be uninformative, based on temporal consistency of prompt value across epochs. Up to 2.4× rollout and 2.0× total training wall-clock speedup without accuracy loss (Qwen2.5-Math-1.5B/7B, R1-Distill-Qwen-1.5B) — [GRESO abstract](https://arxiv.org/abs/2506.02177)
- PODS trains on a max-variance subset of rollouts and reaches vanilla GRPO's peak accuracy at least 1.7× faster (TMLR 2026) — [PODS abstract](https://arxiv.org/abs/2504.13818)
- ReFlow replays recent trajectories (a "dual replay data cycle") to amortize rollout cost. Up to 2.06× training throughput vs SOTA sync and async systems — [ReFlow abstract](https://arxiv.org/abs/2610.21640)
- RePO retrieves off-policy samples from a replay buffer. +15% compute and +48% effective optimization steps (Qwen3-1.7B) — [RePO abstract](https://arxiv.org/abs/2506.09340)
- Selective Rollout (agentic) stops a group mid-trajectory when partial action prefixes converge, predicting zero variance. About 40% of groups are zero-variance in practice. 10.7% faster wall-clock on ALFWorld with Qwen2.5-7B — [arXiv 2605.05802](https://arxiv.org/abs/2605.05802)

**Quantized / low-precision rollouts**
- FlashRL / TIS (Feng Yao, Liyuan Liu et al., Aug 2025): rollout engines (vLLM) and trainers (FSDP) assign different token probabilities even with identical weights, which "implicitly turns on-policy RL into off-policy". Truncated importance sampling corrects this and allows INT8/FP8 rollouts. The FlashRL README gives no explicit speedup number, recommends 32B+ models with long CoT, and supports only VeRL+vLLM — [NeurIPS OPT 2025 workshop page](https://neurips.cc/virtual/2025/124259); [FlashRL GitHub](https://github.com/yaof20/Flash-RL)
- QuRL (ICLR '26): quantized actor for rollout, with Adaptive Clipping Range keyed to the full-precision/quantized policy ratio, and "invariant scaling" because per-step weight changes are too small for quantization to capture. 20–80% faster rollout with INT8/FP8 (DeepScaleR, DAPO) — [QuRL abstract](https://arxiv.org/abs/2602.13953)
- FP8-RL (verl, NVIDIA): blockwise FP8 W8A8 rollout, FP8 KV cache with per-step QKV scale recalibration, and token-level TIS/MIS correction. Up to 44% rollout throughput gain, with learning behavior comparable to BF16 — [FP8-RL abstract](https://arxiv.org/abs/2601.18150)
- Jet-RL: the common BF16-train + FP8-rollout recipe "suffers from severe training instability and catastrophic accuracy collapse under long-horizon rollouts". A unified FP8 flow for both training and rollout gives up to 33% faster rollout, 41% faster training and 16% faster end-to-end vs BF16 — [Jet-RL abstract](https://arxiv.org/abs/2601.14243)
- QeRL: NVFP4 + LoRA gives over 1.5× rollout speedup and trains 32B RL on a single H100 80GB. It argues quantization noise raises entropy and helps exploration (Adaptive Quantization Noise) — [QeRL abstract](https://arxiv.org/abs/2510.11696)
- AIS (adaptive importance sampling): the FP8 mismatch is non-stationary, an exploration bonus early and destabilizing bias later. AIS matches BF16 on most tasks and keeps FP8's 1.5–2.76× rollout speedup (Qwen3-8B, Qwen3.5-9B, LLaDA-8B) — [AIS abstract](https://arxiv.org/abs/2605.13907)
- QaRL aligns the training forward with the quantized rollout and adds TBPO. It notes that quantized long-form responses produce "repetitive, garbled tokens" and reports +5.5 over quantized-rollout training on Qwen3-30B-A3B — [QaRL abstract](https://arxiv.org/abs/2604.07853)
- KV4-RL: under async rollout, "KV concentration" makes KV quantization deliver more acceleration than weight-only or W-A quantization. INT4 KV degrades learning "even with token-level TIS". Protecting attention sinks and recent context at INT8 fixes this. 1.80× cumulative rollout and 1.27× full-iteration speedup over 400 iterations (Qwen3-8B-Base GRPO), within 0.5 pp of BF16-rollout accuracy — [KV4-RL abstract](https://arxiv.org/abs/2610.14448)
- MXFP8 end-to-end on B200: MXFP8 training pays off only above about 4.5k tokens per micro-batch (dense). It cut MoE step time 10.7% and generation 24%, but slowed a short-sequence dense workload by 5%. Without a quantized KV cache, BF16 end to end is 2.5–10× more consistent. The author lists 7 weight-sync correctness conditions, 5 of which failed silently in practice. Single-seed, at most 60 steps — [arXiv 2610.34915](https://arxiv.org/abs/2610.34915)
- Sparse-RL (sparse/compressed KV rollouts): naive KV compression in RL causes "catastrophic performance collapse". Sparsity-aware rejection sampling and importance reweighting fix it — [Sparse-RL abstract](https://arxiv.org/abs/2601.10079). A related paper on KV compression in RL ("Shadow Mask Distillation", 2605.06850) was withdrawn by its authors for experimental errors — [arXiv 2605.06850](https://arxiv.org/abs/2605.06850)

**Mismatch root causes and non-IS fixes**
- FP16 instead of BF16 "effectively eliminates" the mismatch; BF16's rounding errors are the root cause — [arXiv 2510.26788](https://arxiv.org/abs/2510.26788)
- TBIK (tree-based invariant kernels) gives bitwise-identical vLLM vs FSDP results across TP sizes — [arXiv 2511.17826](https://arxiv.org/abs/2511.17826)
- vLLM blog (Oct 14, 2026): the training forward calls vLLM's own inference kernels, modified for bitwise-identical logprobs. Rollout throughput was 91.3% of the non-consistent baseline on DeepSeek-V4-Flash (GB200), with about 10% end-to-end overhead in a verl recipe on 32 GB200. The non-consistent baseline's train-inference KL rose from 2e-3 to 29.9. The post argues TIS, MIS, R3 and QAT "reduce the effect of the mismatch but do not remove it" — [vLLM blog](https://vllm.ai/blog/2026-10-14-train-inference-consistency)
- "When Speed Kills Stability" (Yingru Li, Jiacai Liu blog, 2025): token-level IS is biased, while sequence-level masked IS (MIS) drops sequences whose ratio exceeds a threshold. Reportedly adopted in DeepSeek-V3.2 as geometric sequence masking — [blog](https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda) (details via search summary, not full read)
- Mismatch grows with training and can be suppressed by an LR schedule triggered by response length — [arXiv 2602.01826](https://arxiv.org/abs/2602.01826). Small numerical disagreements can cause collapse independently of staleness (VeXact zero-mismatch reference) — [arXiv 2605.14220](https://arxiv.org/abs/2605.14220)
- Practical async pipelines with partial rollouts often lose the "old logits" needed for decoupled correction. Exact recovery (snapshot versioning, dedicated old-logit model, interrupt-to-sync) and approximate PPO-EWMA routes are proposed — [arXiv 2605.12070](https://arxiv.org/abs/2605.12070)
- ServiceNow (PipelineRL vLLM V0→V1): processed vs raw logprobs, prefix caching across weight updates, async scheduling and an fp32 lm_head all caused mismatch. Advice: fix backend correctness before adding objective corrections like TIS, because corrections can mask inference bugs — [ServiceNow/HF blog](https://hface.pages.dev/blog/ServiceNow-AI/correctness-before-corrections)

### Inferences
- Precision approaches have shifted from "quantize rollout, correct with TIS" (2025) to "align both sides" (Jet-RL, QaRL, MXFP8 matching, bitwise-consistent kernels; 2026). KV-cache precision is the newest frontier (FP8-RL KV, KV4-RL) because long outputs make attention and KV traffic dominate.

### Gaps
- FlashRL's own speedup numbers could not be extracted (the Notion page did not render and the README gives no number).
- I could not confirm the IcePop (Ant/Ring) details or the DeepSeek-V3.2 masking claim from primary sources.

## Q6. Preemption and memory pressure during rollouts (KV exhaustion with long reasoning; recompute vs swap)

### Takeaway
KV exhaustion in long-reasoning RL is common and measured. Seer counted 13,686 preemption events in one veRL rollout iteration (Qwen2-VL-72B), and RollPacker saw 1.79× more preemptions in over-provisioned short rounds. Engines default to recompute: vLLM V1 recomputes rather than swaps, and SOSP '23 found recompute overhead never above 20% of swap latency. Mitigations so far are scheduling (Seer divided rollout, RollPacker TP scaling, Laminar KV-aware repack, MISA-T KV accounting) and KV compression or quantization (FP8 KV, INT4 KV, sparse KV). Each compression method adds mismatch that must be corrected.

### Cited Findings
- Seer, Qwen2-VL-72B on veRL: 13,686 preemption events in one rollout iteration. Per-request KV grows from a few hundred MB to tens of GB during decoding (or "several GB per request in the final stage"; the paper's sections disagree). For memory-constrained tasks the last 10% of requests take up to 50% of total time. Partial rollout's doubled request volume makes preemption worse. Seer's stated aim is proactive, mobility-aware placement rather than reactive eviction — [Seer HTML](https://arxiv.org/html/2511.14617v3)
- RollPacker: over-provisioned short rounds had up to 1.79× more preemptions than long rounds (Qwen3-8B/32k). Raising TP cut preemptions 21.1% (short rounds) and 10.3% (long rounds). The planner lowered short-round preemptions 13.8% on average — [RollPacker HTML](https://arxiv.org/html/2509.21009v1)
- vLLM docs: when KV space is insufficient, vLLM preempts requests and recomputes them later. V1 defaults to RECOMPUTE rather than SWAP "because recomputation has lower overhead in the V1 architecture". Listed mitigations: raise gpu_memory_utilization, lower max_num_seqs or max_num_batched_tokens, raise TP or PP — [vLLM optimization docs](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- vLLM/PagedAttention (SOSP '23): all-or-nothing eviction per sequence and gang preemption of sequence groups. Swapping is costly with small blocks (many small PCIe transfers). Recompute overhead is constant across block sizes and "never higher than 20% of swapping's latency", and the two are comparable for block sizes 16–64 — [vLLM paper](https://arxiv.org/abs/2309.06180)
- Laminar uses KV-cache utilization to decide repack (thresholded by Cmax and a roofline batch bound). Average KV utilization rose to 82.2% from 71.6% — [Laminar HTML](https://arxiv.org/html/2510.12633v1)
- TLT's multi-strategy CUDA-graph pool competes for memory with KV. Bucketed capture cut peak graph memory from 30.39 GB to 10.69 GB — [TLT HTML](https://arxiv.org/html/2511.16665v3)
- FP8-RL extends FP8 to the KV cache "to remove long-context memory bottlenecks" — [FP8-RL abstract](https://arxiv.org/abs/2601.18150); KV4-RL uses INT4 KV with INT8 protection of sinks and recent context — [KV4-RL abstract](https://arxiv.org/abs/2610.14448); Sparse-RL targets the "memory wall" of KV in long-horizon rollouts — [Sparse-RL abstract](https://arxiv.org/abs/2601.10079)
- MISA-T: heterogeneous rollouts (RLVR/RLHF/agentic) sharing an async service compete for KV capacity, and residency-time-aware KV accounting gives 35.6–53.3% throughput gains — [MISA-T abstract](https://arxiv.org/abs/2608.11152)

### Inferences
- The decoding batch shrinks during the long tail (EfficientRollout, TLT). So memory pressure peaks early in a rollout step and compute under-utilization peaks late. Systems that move KV off-GPU early and fetch it back for the tail (Seer's global pool is the only one found) exploit this asymmetry; RL-specific KV offload, prefetch or migration policies keyed to predicted length look under-explored.

### Gaps
- No paper found compares swap, recompute and KV-pool fetch for RL rollouts at reasoning lengths (16k–128k).
- Preemption counts are reported only by Seer and RollPacker. Most papers don't report preemption at all (RhymeRL and the ServiceNow blog explicitly don't discuss it).

## Q7. Open problems and limitations the papers list explicitly

### Takeaway
Stated limitations recur across papers: (1) staleness or off-policy bias from partial and async rollouts is unbounded or unproven; (2) drafters go stale and cost too much to retrain, and SD fails at high concurrency; (3) history-based methods cold-start in epoch 1 and depend on cache diversity; (4) single-turn math/code only, with multi-turn agentic RL left as future work; (5) heuristic length prediction and resource splits; (6) mismatch corrections reduce but don't remove the gap. Few papers have a dedicated limitations section, so many of these points come from the body text.

### Cited Findings
- AReaL (Appendix E): the 75/25 inference-to-training device split is a heuristic that "might benefit from dynamic adjustment during training". Evaluation covers only single-step math and code. "We leave the exploration of multi-turn interactions and agentic scenarios to future work." Unbounded staleness underperforms zero staleness even with the decoupled objective — [AReaL HTML](https://arxiv.org/html/2505.24298v5)
- APRIL calls the effect of mixing many earlier policies an "open question", and warns that excessive off-policy influence may hurt. Its stability benefit is observed, not proven. Scope is math only, two Qwen3 models, GRPO/DAPO — [APRIL HTML](https://arxiv.org/html/2509.18521v3)
- Kimi k1.5: improving efficiency and scalability of long-context RL "remains an important question". Releasing vLLM GPU memory is hard (CUDA graphs, NCCL buffers, drivers), so they restart vLLM instead — [Kimi k1.5 HTML](https://arxiv.org/html/2501.12599v4)
- RollPacker: the parallelism planner covers TP only, not EP. MPS gives no error isolation (Green Contexts are future work). Extension to async/off-policy and DAPO-style sampling is not evaluated — [RollPacker HTML](https://arxiv.org/html/2509.21009v1)
- Seer (no limitations section): length-prediction error costs performance (96% of oracle). Small-draft-model SD has overhead. Adaptation to async settings is suggested but not tested — [Seer HTML](https://arxiv.org/html/2511.14617v3)
- TLT: drafter training is limited by the idle window (HASS/EAGLE-3 cost 3–7× more). SD switch-in costs about 3 s of re-prefill. Future work: uniformly long responses, multi-turn tool-calling RL, combining with async RL — [TLT HTML](https://arxiv.org/html/2511.16665v3)
- RhymeRL: long-tailed rollouts prevent perfect complementarity across steps, an anomalously long rollout can disrupt the pipeline, HistoPipe is off in epoch 1, and co-trained draft models were unreliable — [RhymeRL HTML](https://arxiv.org/html/2508.18588v1)
- SPEC-RL: depends on cache quality and diversity, lenience is fixed and hand-tuned (adaptive lenience unexplored), there is no speedup in epoch 1 (cold start), and multi-turn, multimodal and agentic RL are future work — [SPEC-RL HTML](https://arxiv.org/html/2509.23232v3)
- DAS: window size is a bias-stability trade-off with no principled setting, the latency model has about 12% error, length prediction is heuristic, and prefix routing's CPU overhead can outweigh its benefit for small models — [DAS HTML](https://arxiv.org/html/2511.13841v1)
- ReSpec names three gaps: "diminishing speedups at large batch sizes, drafter staleness under continual actor updates, and drafter-induced policy degradation" — [ReSpec abstract](https://arxiv.org/abs/2510.26475)
- EfficientRollout: "the evolving target policy makes any fixed drafter increasingly mismatched", and shrinking active batch sizes shift decoding between compute-bound and memory-bound regimes — [EfficientRollout abstract](https://arxiv.org/abs/2606.18967)
- GrowMTP: existing draft heads "must be pretrained or warmed up before RL, introducing substantial training cost outside the RL run" — [GrowMTP abstract](https://arxiv.org/abs/2609.16648)
- Laminar: experience sampling over the large volume of data generated is "a promising direction". Several baselines were reimplemented because the originals are closed. Scaling efficiency is still below 60% — [Laminar HTML](https://arxiv.org/html/2510.12633v1)
- RLBoost: gains shrink under frequent preemption (37% for 14B in the high-preemption trace segment). Future work: dynamic broadcast trees, RDMA transports (Mooncake, NIXL), weight-delta compression, heterogeneous rollout hardware, async RL — [RLBoost HTML](https://arxiv.org/html/2510.19225v3)
- Jet-RL: BF16-train + FP8-rollout suffers "catastrophic accuracy collapse under long-horizon rollouts" — [Jet-RL abstract](https://arxiv.org/abs/2601.14243). KV4-RL: INT4 KV "can degrade learning even with token-level TIS" — [KV4-RL abstract](https://arxiv.org/abs/2610.14448)
- vLLM consistency blog: correction methods "reduce the effect of the mismatch but do not remove it", and wall-clock gains from fewer crashes are "not yet measured" — [vLLM blog](https://vllm.ai/blog/2026-10-14-train-inference-consistency)
- MXFP8 study: single-seed, at most 60 steps, "we make no claim about final model quality". 5 of 7 weight-sync failures were silent — [arXiv 2610.34915](https://arxiv.org/abs/2610.34915)
- GFCR survey (Sec. 9): no standardized protocols for verifier error rates. Compute accounting often omits tool calls, verifier runtime, branch/prune overhead and replay refresh. "Early stopping and partial rollouts improve average efficiency but reduce accuracy on difficult instances" — [arXiv HTML 2605.02913](https://arxiv.org/html/2605.02913v1)

### Inferences (candidate gaps for a PhD student, based on the above)
- Crowded: suffix/n-gram history drafters (RhymeRL, DAS, SRT, NgramRL, Seer, WAR), online-adapted drafters (TLT, ReSpec, FastGRPO, GrowMTP, NVIDIA co-training, Bebop), tail batching and length-aware scheduling (RollPacker, SortedRL, StreamRL, TailSieve), partial rollouts (Kimi, APRIL, CoPRIS, RollVerify), FP8 rollouts with IS correction.
- Less crowded:
  - KV/state migration vs recompute for RL rollouts as a function of context length, interconnect and model architecture. Only Seer moves KV; others recompute.
  - Rollout acceleration for hybrid (linear-attention/SSM + attention) policies: SD state rollback, state migration, prefix/state caching under weight updates. Only numerical-consistency work touches hybrids (vLLM blog Nemotron Mamba-2, IsoExec).
  - Interaction between over-provisioning and KV pressure (preemption storms).
  - Rollout-side prefix and KV sharing for multi-turn agents under weight updates (correctness hazard per ServiceNow).
  - Formal exactness of multi-path, history-based SD under sampling.
  - Standardized compute accounting and benchmarks for rollout systems; most comparisons reimplement closed baselines.

### Gaps
- Several papers' limitations sections fell outside the parts I read (DAS, Seer and TLT HTML were read to about 100K characters; the final 10–30K went unread). Their appendices may hold more explicit future-work items.
- I did not read the GFCR survey's Section 6.8 (Systems to Accelerate Rollout Throughput), so its list of throughput systems is not captured here.
