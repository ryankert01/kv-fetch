# RL post-training / rollout acceleration systems for LLMs: accepted papers at top venues (2024-2026), crowdedness, and publishability

Scope note: research done 2026-10-29. Each entry is marked by how its acceptance was checked:
- **[PR-official]**: confirmed on the official program or proceedings page.
- **[PR-secondary]**: venue tag comes from a community paper list, an author or university page, or a camera-ready header. Likely correct, but I did not check it against the official program.
- **[arXiv-only]**: no peer-reviewed venue found.

## Q1. Which RL-for-LLM systems papers were accepted at top venues in 2024-2026?

### Takeaway
Yes, this is an accepted research topic, and it got much bigger in 2026. Through 2025 there were only a handful of papers (HybridFlow at EuroSys'25, RLHFuse at NSDI'25, ReaL at MLSys'25, AReaL and GRESO at NeurIPS'25, Async RLHF at ICLR'25). In 2026 I count at least 15 systems-venue papers. OSDI'26 gave the topic its own "RL Training at Scale" session with 5 papers, plus a 6th RL paper (RobustRL) in its reliability session. NSDI'26 had 3, MLSys'26 had 3, ASPLOS'26 had 2 and EuroSys'26 had at least 1. ICML'26 and ICLR'26 added several algorithm-system co-design papers. SOSP'25 had none.

### Cited Findings

**2024-2025 (early wave)**

| Title | Venue | Status | Affiliation | One-line contribution | Source |
|---|---|---|---|---|---|
| HybridFlow: A Flexible and Efficient RLHF Framework (verl) | EuroSys 2025 | PR-secondary (EuroSys'25 per a web-search summary; not checked against the EuroSys'25 program) | ByteDance + HKU (affiliation not checked here) | Hybrid single/multi-controller design; reshards actor weights between training and generation. This is the de-facto baseline framework. | [arXiv 2409.19256](https://arxiv.org/abs/2409.19256); [Awesome-Agentic-RL-Infra](https://github.com/kissshhot/Awesome-Agentic-RL-Infra) |
| RLHFuse: Efficient RLHF Training for LLMs with Inter- and Intra-Stage Fusion | NSDI 2025 | PR-secondary | not verified | Fuses RLHF stages (inter- and intra-stage) to cut bubbles | [AmberLJC LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList) |
| ReaL (ReaLHF): Efficient RLHF Training of LLMs with Parameter Reallocation | MLSys 2025 | PR-secondary | THU | Reallocates parameters and parallelism across RLHF stages | [mental2008 awesome-papers](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md) |
| Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models | ICLR 2025 | PR-official | Noukhovitch et al. (affiliation not verified here) | Decouples generation from learning (off-policy). About 40% faster at Llama-3.1-8B, about 70% faster for Rho-1B on GSM8k. | [ICLR 2025 poster](https://www.iclr.cc/virtual/2025/poster/30332); [arXiv 2410.18252](https://arxiv.org/abs/2410.18252v3) |
| AReaL: A Large-Scale Asynchronous RL System for Language Reasoning | NeurIPS 2025 (main track) | PR-official | Ant Group (inclusionAI) + collaborators | Fully asynchronous rollout and training with staleness control; up to 2.77x over synchronous systems | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/33c00862bfa29ac72ecf630a41e19352-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2505.24298v4) |
| Act Only When It Pays (GRESO): Efficient RL for LLM Reasoning via Selective Rollouts | NeurIPS 2025 (main track) | PR-official | Infini-AI-Lab GitHub org | Pre-rollout prompt filter based on reward dynamics; up to 2.4x rollout and 2.0x end-to-end speedup, 1.5B-32B models | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b42fb82fc3ffb88b872c4714f093875d-Abstract-Conference.html) |

- SOSP 2025 accepted 66 papers. Judging by titles, **none is about RL, RLHF or rollouts**. The LLM-training papers it did take were ByteDance's "Robust LLM Training Infrastructure", DCP, TrainVerify, Mycroft and Sailor. — [SOSP'25 accepted list](https://sigops.org/s/conferences/sosp/2025/accepted.html)

**2026 systems venues (main wave)**

| Title | Venue | Status | Affiliation | One-line contribution | Source |
|---|---|---|---|---|---|
| Weave: Efficient Co-Scheduling for Disaggregated RL Post-Training | OSDI 2026 ("RL Training at Scale" session) | PR-official | HKUST, UIUC, Alibaba | Recovers idle time caused by rollout/train dependencies in disaggregated RL | [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions); [mental2008](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md) |
| RLinf: Flexible and Efficient Large-Scale RL via Macro-to-Micro Flow Transformation | OSDI 2026 | PR-official | THU, Infinigence AI, PKU, Beihang, SJTU, UC Berkeley and others | Turns the RL workflow into fine-grained flows for better scheduling and utilization | [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions) |
| DynaRL: Flexible and Dynamic Scheduling of Large-Scale RL Training | OSDI 2026 | PR-official | PKU, Infinigence AI, ICT-CAS, Beihang, THU, SJTU | Shifts compute, memory and network resources among RL components at runtime | [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions) |
| RollArt: Disaggregated Multi-Task Agentic RL Training at Scale | OSDI 2026 | PR-official | HKUST, Alibaba, Tongyi Lab | Maps agentic RL stages (prefill, decode, env, reward) onto suitable hardware; serverless reward; bounded-staleness weight sync | [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions) |
| Seer: Online Context Learning for Fast Synchronous LLM RL | OSDI 2026 | PR-official | Moonshot AI + Tsinghua | Prompt-level context learning to predict rollout lengths; up to 2.04x rollout throughput, 72-94% lower long-tail latency, on production workloads | [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions); [arXiv 2511.14617](https://arxiv.org/abs/2511.14617) |
| RobustRL: Role-Based Fault Tolerance System for RL Post-Training | OSDI 2026 ("Training Reliability and Silent Errors" session) | PR-official | Zhejiang Univ. + SKLMEAC | Isolates failures by role (trainer, rollout, manager) | [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions) |
| RollPacker: Taming Long-Tail Rollouts for RL Post-Training with Tail Batching | NSDI 2026 | PR-official | HKUST + Alibaba | Packs long-response prompts into dedicated "tail" rounds; 2.03-2.56x over veRL, up to 2.24x over RLHFuse; 128 H800, Qwen2.5 7B-32B | [NSDI'26 program](https://www.usenix.org/conference/nsdi26/technical-sessions); [author page](https://wutianyuan1.github.io/publication/rollpacker) |
| RLBoost: Harvesting Preemptible Cloud Resources for Cost-Efficient RL on LLMs | NSDI 2026 | PR-official | UC Berkeley, UMich, CMU, Google, UW | Uses preemptible or spot instances for rollouts | [NSDI'26 program](https://www.usenix.org/conference/nsdi26/technical-sessions) |
| DistRS: Disaggregated Reward Service for RLVR with Batch-Level Constraint | NSDI 2026 | PR-official | PKU + ByteDance Seed | Runs reward computation as a disaggregated service | [NSDI'26 program](https://www.usenix.org/conference/nsdi26/technical-sessions) |
| Laminar: A Scalable Asynchronous RL Post-Training Framework | EuroSys 2026 | PR-secondary | HKU + ByteDance Seed | Relay workers act as a distributed parameter service, so there is no global weight-sync barrier; up to 5.48x throughput on 1024 GPUs | [HKU repository record](https://hub.hku.hk/handle/10722/375626); [AmberLJC](https://github.com/AmberLJC/LLMSys-PaperList); [arXiv](https://arxiv.org/html/2510.12633v1) |
| Taming the Long-Tail (TLT): Efficient Reasoning RL Training with Adaptive Drafter | ASPLOS 2026 | PR-secondary (ASPLOS'26 camera-ready header) | MIT, NVIDIA, ETH Zurich, MIT-IBM, UMass | Trains a drafter on idle GPUs during the long tail and picks spec-decoding strategies with a bandit; more than 1.7x end-to-end | [arXiv HTML](https://arxiv.org/html/2511.16665); [abs](https://arxiv.org/abs/2511.16665) |
| History Doesn't Repeat Itself but Rollouts Rhyme (RhymeRL) | ASPLOS 2026 | PR-secondary | SJTU + ByteDance | Reuses similarity across epochs of rollout history (speculative reuse) to cut redundant generation | [mental2008](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md) |
| Beat the Long Tail: Distribution-Aware Speculative Decoding for RL Training (DAS) | MLSys 2026 | PR-official (in proceedings) | Together AI, Stanford, UCSD | Distribution-aware spec decoding for long-tail rollouts; up to 50% shorter rollout time with identical training curves | [MLSys'26 proceedings PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/cbc4ab80cd77aa0eb87da062fbcddb46-Paper-Conference.pdf); [mental2008](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md) |
| ReSpec: Towards Optimizing Speculative Decoding in RL Systems | MLSys 2026 | PR-official (MLSys virtual poster) | NTU + HPC-AI Tech | Tunes spec-decoding configs, updates the drafter via distillation, weights updates by reward; up to 4.5x on Qwen 3B-14B | [MLSys poster page](https://mlsys.org/virtual/2026/poster/3613) |
| HetRL: Efficient RL for LLMs in Heterogeneous Environments | MLSys 2026 | PR-secondary (Amazon Science listing) | Amazon | RL training across clusters that mix GPU generations | [Amazon Science MLSys'26](https://www.amazon.science/conferences-and-events/mlsys-2026) |

- The OSDI'26 program put Weave and RLinf (plus, by the page ordering, DynaRL, RollArt and Seer) in a dedicated Tuesday session, "RL Training at Scale", chaired by Shadi Noghabi (Google). RobustRL appears in "Training Reliability and Silent Errors". — [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions)
- A Chinese-language OSDI'26 review lists the RL session as Weave, **RLux, DynaRL, MARS, GroupMind**. These names do not match the official program (RLinf, RollArt, Seer), so they are probably pre-camera-ready or anonymized titles. The same review calls it a "brand-new direction, 5-paper dedicated session". — [chenzongyao200127 issue](https://github.com/chenzongyao200127/chenzongyao200127.github.io/issues/19); contradicted on titles by the [official program](https://www.usenix.org/conference/osdi26/technical-sessions)
- The OSDI'26 total paper count is uncertain: one GitHub issue says 134 papers and the Chinese review says "about 80+". The official program's Monday sessions alone list 42 papers. — [InferMatrixCopilot issue](https://github.com/JiusiServe/InferMatrixCopilot/issues/5); [chenzongyao200127 issue](https://github.com/chenzongyao200127/chenzongyao200127.github.io/issues/19); [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions)
- EuroSys'26 also accepted LoRAFusion (LoRA fine-tuning, not RL). I could not find an official EuroSys'26 accepted-papers page (2026.eurosys.org/accepted-papers.html returned 404). — [AmberLJC](https://github.com/AmberLJC/LLMSys-PaperList)

**2026 ML venues (ICLR / ICML), mostly algorithm-system co-design**

| Title | Venue | Status | Contribution | Source |
|---|---|---|---|---|
| QuRL | ICLR 2026 | PR-official (proceedings.iclr.cc) | Quantized-inference rollouts with fixes for clipping and mismatch; 20-80% throughput over BF16 rollout | [ICLR'26 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/6b4880ced785077159b340710ceb7681-Paper-Conference.pdf) |
| DistillSparse (sparse attention for efficient LLM RL) | ICLR 2026 | PR-secondary (iclr.cc virtual page; main track vs workshop not verified) | Sparse attention in rollouts; 1.72x rollout speedup on H200 at 16K generation | [ICLR'26 page](https://iclr.cc/virtual/2026/10013424) |
| ASearcher (long-horizon search-agent RL with fully async RL) | ICLR 2026 | PR-official ("Published as a conference paper at ICLR 2026") | Agent paper that relies on a fully asynchronous RL system | [OpenReview PDF](https://openreview.net/pdf/c9a8c02cc5a055201027e0ac7e7bee0a60444366.pdf) |
| DARTS: Distribution-Aware Active Rollout Trajectory Shaping | ICML 2026 | PR-official (poster) | Shapes the rollout distribution toward shorter outputs to cut intra-prompt long tails; up to 1.77x over SOTA systems | [ICML'26 poster](https://icml.cc/virtual/2026/poster/61634) |
| Stable Asynchrony (VCPO): Variance-Controlled Off-Policy RL for LLMs | ICML 2026 | PR-official (poster) | Scales the learning rate by effective sample size for stale rollouts; matches sync at 2.5x speed on a tool-use task | [ICML'26 poster](https://icml.cc/virtual/2026/poster/62012) |
| BubbleSpec: Turning Long-Tail Bubbles into Speculative Rollout Drafts | ICML 2026 | PR-secondary (listed in ICML'26 Poster Session 5) | Uses straggler-induced idle time across DP ranks for spec drafts; up to 1.8x rollout throughput | [ICML'26 session](https://icml.cc/virtual/2026/session/68689); [arXiv](https://arxiv.org/pdf/2605.08862) |
| EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts | ICML 2026 | PR-secondary (icml.cc page; main track vs workshop not verified) | Self-speculative decoding; up to 19.6% rollout and 12.7% end-to-end latency reduction | [ICML'26 page](https://icml.cc/virtual/2026/75105) |
| SortedRL: Online Length-Aware Scheduling | ICML 2025 page (likely workshop); ICLR 2026 submission status unclear | Uncertain | Reorders rollouts by length for early updates; more than 50% fewer bubbles on Llama-3.1-8B and Qwen-2.5-32B (Microsoft Research) | [ICML'25 page](https://icml.cc/virtual/2025/51798); [OpenReview forum](https://openreview.net/forum?id=5v3Gzuic8i); [MSR page](https://www.microsoft.com/en-us/research/publication/sortedrl-accelerating-rl-training-for-llms-through-online-length-aware-scheduling/) |

- GRESO also has an ICML 2025 virtual page (likely a workshop version) in addition to its NeurIPS'25 main-track paper. — [ICML'25 page](https://icml.cc/virtual/2025/51878)

### Inferences
- Peer-reviewed count by venue: **2024: ~0 at top systems venues** that I could verify (HybridFlow and ReaLHF were arXiv in 2024). **2025: about 6** (EuroSys 1, NSDI 1, MLSys 1, NeurIPS 2, ICLR 1). **2026: about 15 at systems venues** (OSDI 6, NSDI 3, MLSys 3, ASPLOS 2, EuroSys at least 1) plus about 6-7 at ICLR/ICML. That is roughly a 3-4x jump. The topic is clearly accepted, and OSDI now gives it a named session.
- Acceptance is not closed off, but systems venues reward a new *system abstraction* at scale (disaggregation, co-scheduling, flow transformation, fault tolerance, reward service), not one more trick.
- SOSP'25 having zero RL papers while OSDI'26 has six suggests the wave of submissions written in early-to-mid 2025 landed in the 2026 cycle (OSDI'26 / NSDI'26 / ASPLOS'26 / MLSys'26).

### Gaps
- I could not read the official EuroSys 2026 or ASPLOS 2026 programs. RhymeRL (ASPLOS'26) and Laminar (EuroSys'26) rest on secondary sources. EuroSys'26 may have more RL papers.
- HybridFlow's EuroSys'25 and RLHFuse's NSDI'25 status, and their affiliations, were not checked against the official programs this session.
- I did not check PUZZLE (RLHF context switching, often cited as USENIX ATC'24), ATC 2025, SC 2024/2025 or MLSys 2026 beyond the 3 papers above. I found no reliable source for an SC paper in this area.
- NeurIPS 2025 likely has more efficiency-for-RL papers (algorithmic rollout pruning and similar) than the two I confirmed; I did not scan the full list.
- OSDI'26 total paper count is unresolved (134 vs ~80+).

## Q2. How many RL-systems arXiv preprints appeared in 2025-2026, and how many came from large industry labs?

### Takeaway
The preprint flow is large and speeding up. A Sept-2026 survey screened 150 records and kept 86 rollout-efficiency works from Jan 2024 to Sep 2026. A narrow arXiv query (abstract must contain "rollout" + "reinforcement learning" + "throughput") returns 72 CS papers submitted in 2026 alone, about 60% of them LLM-RL systems. Many headline frameworks and systems are arXiv-only industry tech reports or system papers from ByteDance, Alibaba, Meta, Ant, Moonshot, Huawei and others. Industry co-authors are also on most accepted systems papers.

### Cited Findings
- The survey "Rollout Efficiency in RL for Reasoning LLMs: A Taxonomy and Future Directions" (arXiv 2609.25463, Sep 21 2026; ÉTS / U. Melbourne / Huawei) screened 150 candidate records and included 86 works published Jan 2024 to Sep 2026. 80 of them get method-level rows. — [arXiv abs](https://arxiv.org/abs/2609.25463); [HTML](https://arxiv.org/html/2609.25463)
- arXiv advanced search (abstract: rollout AND "reinforcement learning" AND throughput; CS; submitted in 2026) returned **72 results**. Most of the first 50 are LLM-RL systems papers, e.g. NgramRL, FlashNext-RL, ReFlow, S³R, RollVerify, VenusRL, PEARL, AReaL-TIK, HARTS, TailSieve, RoutePack, TideRL, WAR, Libra, SparseRL-Sync, ROSE, DORA, Relax, QaRL, BubbleSpec. — [arXiv search, 2026](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=rollout&terms-0-field=abstract&terms-1-operator=AND&terms-1-term=%22reinforcement+learning%22&terms-1-field=abstract&terms-2-operator=AND&terms-2-term=throughput&terms-2-field=abstract&classification-computer_science=y&classification-physics_archives=all&date-filter_by=specific_year&date-year=2026&date-date_type=submitted_date&abstracts=hide&size=50&order=-announced_date_first)
- The same query for 2025 returned only 6 results, and a "speedup" variant returned 10 (7 of them RL-systems: SkyRL-Agent, ReSpec, Laminar, ROLL Flash, RollPacker, VerlTool, GRESO). These numbers are surprisingly low next to 2026, so arXiv's keyword-and-date search is a weak counter and should be treated as a lower bound. — [arXiv search, 2025 "throughput"](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=rollout&terms-0-field=abstract&terms-1-operator=AND&terms-1-term=%22reinforcement+learning%22&terms-1-field=abstract&terms-2-operator=AND&terms-2-term=throughput&terms-2-field=abstract&classification-computer_science=y&classification-physics_archives=all&date-filter_by=specific_year&date-year=2025&date-date_type=submitted_date&abstracts=hide&size=50&order=-announced_date_first); [arXiv search, 2025 "speedup"](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=rollout&terms-0-field=abstract&terms-1-operator=AND&terms-1-term=%22reinforcement+learning%22&terms-1-field=abstract&terms-2-operator=AND&terms-2-term=speedup&terms-2-field=abstract&classification-computer_science=y&classification-physics_archives=all&date-filter_by=date_range&date-from_date=2025-01-01&date-to_date=2025-12-31&date-date_type=submitted_date&abstracts=hide&size=200&order=-announced_date_first)
- A curated agentic-RL-infra reading list has 34 entries, all linked to arXiv. They include StreamRL, AReaL, AsyncFlow, RollPacker, ROLL Flash, RollArt, DORA, WAR, VenusRL, OpenRLHF, ReaL, HybridFlow, ROLL, RLinf, Agent Lightning, AgentRL, SkyRL-Agent and Molt. — [Awesome-Agentic-RL-Infra](https://github.com/kissshhot/Awesome-Agentic-RL-Infra)
- Examples of arXiv-only (no venue found) RL systems from large labs:
  - LlamaRL (Meta; evaluated at 1,024 H100s, 8B-405B) — [survey Table 2](https://arxiv.org/html/2609.25463)
  - StreamRL (disaggregated stream generation) — [AmberLJC](https://github.com/AmberLJC/LLMSys-PaperList)
  - ROLL / ROLL Flash (Alibaba) — [Awesome-Agentic-RL-Infra](https://github.com/kissshhot/Awesome-Agentic-RL-Infra)
  - NeMo-Aligner (NVIDIA) — [AmberLJC](https://github.com/AmberLJC/LLMSys-PaperList)
  - NeMo-RL speculative decoding (NVIDIA; 1.8x at 8B, projected 2.5x at 235B) — [arXiv 2604.26779](https://arxiv.org/html/2604.26779)
  - RLAX (Apple, TPUs) — [arXiv 2512.06392](https://arxiv.org/html/2512.06392v1)
  - DORA / LongCat-Flash-Thinking (Meituan) — [arXiv 2601.16725](https://arxiv.org/pdf/2601.16725)
  - Agent Lightning (Microsoft) — [AmberLJC](https://github.com/AmberLJC/LLMSys-PaperList)
  - AsyncFlow, APRIL, SeamlessFlow, Relax, AstraFlow (CMU) — [AmberLJC](https://github.com/AmberLJC/LLMSys-PaperList); [arXiv search](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=rollout&terms-0-field=abstract&terms-1-operator=AND&terms-1-term=%22reinforcement+learning%22&terms-1-field=abstract&terms-2-operator=AND&terms-2-term=throughput&terms-2-field=abstract&classification-computer_science=y&classification-physics_archives=all&date-filter_by=specific_year&date-year=2025&date-date_type=submitted_date&abstracts=hide&size=50&order=-announced_date_first)
- Model tech reports also publish RL-infra details without peer review, e.g. MiMo-V2.6 and the FlashNext-RL "system optimization study" for a Qwen model. — [arXiv search 2026](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=rollout&terms-0-field=abstract&terms-1-operator=AND&terms-1-term=%22reinforcement+learning%22&terms-1-field=abstract&terms-2-operator=AND&terms-2-term=throughput&terms-2-field=abstract&classification-computer_science=y&classification-physics_archives=all&date-filter_by=specific_year&date-year=2026&date-date_type=submitted_date&abstracts=hide&size=50&order=-announced_date_first)
- An HKUST PhD qualifying exam, "A Survey on Systems for RL Post-Training of LLMs" (Apr 2026), names three system challenges: orchestration complexity, the rollout bottleneck, and the synchronization barrier. — [HKUST defense page](https://cse.hkust.edu.hk/pg/defenses/S26/lcaoar-15-04-2026.html)

### Inferences
- My classification of the first 50 of the 72 hits for 2026: about 31 are LLM-RL systems papers, about 8 are efficiency-motivated RL algorithm papers, and about 8 are unrelated (robotics, simulators). That suggests on the order of 40-50 systems-flavored RL preprints in 2026 from this one narrow query. Counting papers that don't use the word "throughput", the true number is probably 100+ RL-for-LLM systems or efficiency preprints across 2025-2026, consistent with the survey's 86 included works. This estimate comes from my own title classification.
- **Industry share of the 15 peer-reviewed 2026 systems-venue papers** (from the affiliation columns in Q1):
  - About 9 have a large-lab co-author: Laminar, DistRS, RhymeRL (ByteDance); RollPacker, Weave, RollArt (Alibaba); Seer (Moonshot); TLT (NVIDIA); HetRL (Amazon).
  - About 4 come from AI-infra startups or mid-size firms: RLinf and DynaRL (Infinigence AI), DAS (Together AI), ReSpec (HPC-AI Tech).
  - Only about 2 are mostly academic: RobustRL, and RLBoost (which has one Google author).
- The academia-industry pairing is very concentrated: a single HKUST group (Wei Wang) with Alibaba has 3 of the 15 (RollPacker, Weave, RollArt). Tsinghua/PKU with Infinigence has 2 (RLinf, DynaRL).
- The ratio of accepted to preprinted work is low. Many frameworks with large user bases (OpenRLHF, ROLL, slime, SkyRL, StreamRL, LlamaRL) have no top-venue paper I could find. This suggests industry often doesn't bother submitting, so arXiv volume overstates the competition at venues, but it raises the novelty bar because reviewers know these systems.

### Gaps
- I could not get exact per-month or per-year arXiv counts. The arXiv API (export.arxiv.org) failed from this environment (HTTP 500 or no response), and the web search's date filters gave inconsistent 2025 vs 2026 numbers.
- Affiliations for most 2026 arXiv-only titles were not checked, so the industry share of preprints is a qualitative judgment, not a count.

## Q3. What contribution types got accepted vs. likely seen as incremental? What do reviewers expect?

### Takeaway
Systems venues accepted three kinds of work:
- **New end-to-end architectures or abstractions**: HybridFlow, RLinf, DynaRL, Laminar, RollArt, Weave.
- **Long-tail mitigation**, through scheduling (RollPacker, Seer) or RL-specific speculative decoding (TLT, DAS, ReSpec, RhymeRL).
- **New resource angles**: preemptible or spot capacity (RLBoost), heterogeneous GPUs (HetRL), a disaggregated reward service (DistRS), fault tolerance (RobustRL).

ML venues accepted **algorithm-system co-design with correctness arguments**: async/off-policy correction (Async RLHF, AReaL, VCPO), rollout or prompt selection (GRESO), quantized and sparse rollouts (QuRL, DistillSparse), and distribution shaping (DARTS).

Long-tail plus speculative decoding is now the most crowded niche: at least 4 peer-reviewed papers in 2026 and many more on arXiv. I could not get public OpenReview review text, because OpenReview served a bot challenge. The best proxy for reviewer expectations is the 2026 survey's critique of evaluation practice: report end-to-end time-to-quality, not just rollout throughput.

### Cited Findings
- **Survey taxonomy** (7 mechanism families, 4 bottlenecks) with representative systems — [survey HTML](https://arxiv.org/html/2609.25463):
  - pipeline decoupling: AReaL, LlamaRL, Laminar, ProRL Agent
  - resource-aware execution: RLBoost, AReaL-Hex, Libra, ROSE, BiDiRL
  - scheduling and load balancing: RollPacker, SortedRL, StreamRL, Heddle
  - partial and early-stop rollout: APRIL, ESPO, DeepScaleR, Kimi k1.5
  - speculative decoding: BubbleSpec, TLT, RhymeRL, DAS
  - rollout selection: PODS, POPO
  - prompt filtering: GRESO, MoPPS, VCRL, AERO
- **The same survey's evaluation critique** — [survey HTML](https://arxiv.org/html/2609.25463):
  - Only 12 of 80 methods report gains on both system and algorithmic levers.
  - Papers conflate "cost of collecting trajectories" with "end-to-end cost to reach target quality". The authors call this a principal source of incomparability.
  - System papers report throughput and utilization, while algorithm papers report accuracy and samples-to-target.
  - The authors propose accelerator-hours-to-target-quality as a common metric, plus a reporting checklist.
  - Open problems they list: the unmapped staleness-quality tradeoff (most systems report one staleness bound), length predictors never compared on a common workload, and online rollout/train repartitioning on top of async pipelines not yet measured.
- **veRL is the standard baseline.** The survey says veRL is the baseline for many rows of its comparison table. TLT compares against VeRL and Open-R1. RollPacker compares against veRL and RLHFuse. — [survey HTML](https://arxiv.org/html/2609.25463); [TLT HTML](https://arxiv.org/html/2511.16665); [RollPacker author page](https://wutianyuan1.github.io/publication/rollpacker)
- **Accuracy preservation is part of the claim in every accepted spec-decoding or async paper** — [TLT](https://arxiv.org/abs/2511.16665); [WukLab / DAS](https://mlsys.wuklab.io/); [MLSys ReSpec](https://mlsys.org/virtual/2026/poster/3613); [ICML VCPO](https://icml.cc/virtual/2026/poster/62012):
  - TLT: "preserving model accuracy"
  - DAS: "identical training curves"
  - ReSpec: "preserving reward convergence and training stability"
  - VCPO: "matches synchronous performance"
- **Speculative decoding for RL, saturation count.** Peer-reviewed: TLT (ASPLOS'26), DAS (MLSys'26), ReSpec (MLSys'26), BubbleSpec (ICML'26), EfficientRollout (ICML'26 page). arXiv: NgramRL, S³R, NeMo-RL spec decoding, "Fast LLM Post-training via Decoupled and Fastest-of-N Speculation", Aurora, GrowMTP, RLHFSpec, "Breaking Entropy Bounds via MTP". — [mental2008](https://github.com/mental2008/awesome-papers/blob/develop/paper-list/systems-for-ml/llm.md); [ICML session](https://icml.cc/virtual/2026/session/68689); [ICML'26 page](https://icml.cc/virtual/2026/75105); [arXiv 2604.26779](https://arxiv.org/html/2604.26779); [arXiv 2511.16193](https://arxiv.org/pdf/2511.16193); [AmberLJC](https://github.com/AmberLJC/LLMSys-PaperList); [arXiv 2609.16648](https://arxiv.org/pdf/2609.16648); [arXiv search 2026](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=rollout&terms-0-field=abstract&terms-1-operator=AND&terms-1-term=%22reinforcement+learning%22&terms-1-field=abstract&terms-2-operator=AND&terms-2-term=throughput&terms-2-field=abstract&classification-computer_science=y&classification-physics_archives=all&date-filter_by=specific_year&date-year=2026&date-date_type=submitted_date&abstracts=hide&size=50&order=-announced_date_first)
- **Long-tail scheduling or partial-rollout, besides the spec-decoding papers.** Peer-reviewed: RollPacker (NSDI'26), Seer (OSDI'26), DARTS (ICML'26). arXiv: APRIL, SortedRL (status unclear), CoPRIS, TailSieve, RollVerify, WAR. — [NSDI'26 program](https://www.usenix.org/conference/nsdi26/technical-sessions); [OSDI'26 program](https://www.usenix.org/conference/osdi26/technical-sessions); [ICML DARTS](https://icml.cc/virtual/2026/poster/61634); [arXiv CoPRIS](https://arxiv.org/pdf/2511.05589); [arXiv TailSieve](https://arxiv.org/pdf/2608.22788)
- **Emerging, thinner sub-areas in 2026 arXiv titles** — [arXiv search 2026](https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term=rollout&terms-0-field=abstract&terms-1-operator=AND&terms-1-term=%22reinforcement+learning%22&terms-1-field=abstract&terms-2-operator=AND&terms-2-term=throughput&terms-2-field=abstract&classification-computer_science=y&classification-physics_archives=all&date-filter_by=specific_year&date-year=2026&date-date_type=submitted_date&abstracts=hide&size=50&order=-announced_date_first):
  - low-precision RL: TRIAGE, QUADS, AIS, QaRL, "FP4 Explore, BF16 Train"
  - weight sync: SparseRL-Sync, about 100x less communication
  - MoE-RL load balancing: RoutePack, ReLibra
  - prefix sharing with hybrid-attention models: HARTS
  - agent sandbox and environment cost: CUA-Sandbox; "Rollout Infrastructure Tax" ([Awesome-Agentic-RL-Infra](https://github.com/kissshhot/Awesome-Agentic-RL-Infra))
  - rollout on serving GPUs: ROSE
- **Review-process context.** ICLR 2026 issued 13,763 decisions from 76,139 reviews. An OpenReview API exploit leaked author, reviewer and AC identities for a large subset of submissions, and desk rejections were relatively high, linked to hallucinated references. — [ICLR 2026 retrospective](https://blog.iclr.cc/2026/03/31/a-retrospective-on-the-iclr-2026-review-process/)

### Inferences
- **Likely judged incremental now:**
  - another long-tail scheduler or partial-rollout heuristic, or another "spec decoding for RL" paper, unless it shows a clearly new mechanism
  - "async RL with bounded staleness" without new correctness analysis
  - speedups reported only as rollout throughput, not time-to-target-accuracy
  - evaluations without a veRL (or slime/AReaL) baseline at realistic response lengths (16K-32K+)
- **Still open, per the survey's open-problems list plus thin coverage:**
  - mapping the staleness-quality frontier
  - benchmarking length predictors
  - online repartitioning on async pipelines
  - low-precision rollouts with mismatch correction
  - memory, KV, prefix-sharing and state management for hybrid-attention or agentic multi-turn rollouts
  - weight-sync compression
  - a standardized evaluation methodology (time-to-quality)
- Relevance to the user's own project: HARTS ("A Prefix-Sharing Training System for Agentic RL with Hybrid-Attention Models", arXiv 2026) is directly next to KV/state-caching work on hybrid models. Treat it as a must-read competitor before choosing a niche.

### Gaps
- **I could not read any OpenReview reviews.** The forum page and the api2 JSON endpoint both returned a browser-verification challenge. So I have no direct quotes of reviewer demands (scale, verl baselines) for ICLR 2026 or NeurIPS 2025 RL-systems submissions. Whether SortedRL and APRIL were accepted, rejected or withdrawn at ICLR 2026 is also unknown.
- I found no public reviews or meta-reviews for systems venues (OSDI/NSDI/EuroSys don't publish them), and no blog posts summarizing reviewer feedback on RL-systems papers.

## Q4. Evaluation scale in accepted papers, and whether small or single-GPU academic work gets in

### Takeaway
Systems-venue RL papers evaluate on multi-node clusters: from **64 H100s** (TLT, the smallest I found) through 128 H800s (RollPacker) to 1,024 GPUs (Laminar), often on 7B-70B models and sometimes on production workloads (Seer). I found **no systems-venue RL paper evaluated on a single GPU**. The ML venues (ICLR/NeurIPS/ICML) accept small-scale algorithm-system co-design with 1B-8B models and per-GPU throughput results (Async RLHF, GRESO, QuRL, DistillSparse), and ICML workshops seem to host early versions (SortedRL, GRESO).

### Cited Findings
- **Systems venues:**
  - Laminar (EuroSys'26): 1,024-GPU cluster, up to 5.48x throughput. — [Laminar](https://arxiv.org/html/2510.12633v1)
  - RollPacker (NSDI'26): up to 128 H800 GPUs, Qwen2.5 7B-32B. — [RollPacker page](https://wutianyuan1.github.io/publication/rollpacker); [survey Table 2](https://arxiv.org/html/2609.25463)
  - TLT (ASPLOS'26): 8 DGX H100 nodes (64 GPUs) plus a separate A100 cluster, with Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-32B and Llama-3.3-70B. — [TLT HTML](https://arxiv.org/html/2511.16665)
  - Seer (OSDI'26): "production-grade RL workloads" at Moonshot. — [arXiv 2511.14617](https://arxiv.org/abs/2511.14617)
  - ReSpec (MLSys'26): Qwen 3B-14B. — [MLSys poster](https://mlsys.org/virtual/2026/poster/3613)
- **Arxiv-only industry for comparison:** LlamaRL at 1,024 H100s, 8B-405B; DeepScaleR at 8-32 A100s with a 1.5B model; AReaL-DTE at 8-16 H200s with 8B and 30B-A3B models. — [survey Table 2](https://arxiv.org/html/2609.25463)
- **ML venues at small scale:**
  - Async RLHF (ICLR'25) used Rho-1B, 2.8B and Llama-3.1-8B. — [arXiv 2410.18252](https://arxiv.org/abs/2410.18252v3)
  - GRESO (NeurIPS'25) used 1.5B-32B Qwen models. — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b42fb82fc3ffb88b872c4714f093875d-Abstract-Conference.html)
  - DistillSparse (ICLR'26) reports rollout speedup on an H200 at 16K. — [ICLR page](https://iclr.cc/virtual/2026/10013424)
  - QuRL (ICLR'26) reports throughput over BF16 rollouts. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/6b4880ced785077159b340710ceb7681-Paper-Conference.pdf)
- APRIL (arXiv-only; affiliation not checked) evaluated 4B-8B models. — [survey Table 2](https://arxiv.org/html/2609.25463)
- An ICML 2026 workshop focused on RL from world feedback (RLxF) had an accepted-papers page, so ICML 2026 had RL-specific workshops. — [RLxF ICML 2026](https://sites.google.com/view/rlxf-icml2026/accepted-papers)

### Inferences
- With one 48 GB GPU, a systems-venue paper (OSDI/NSDI/EuroSys/ASPLOS) on RL rollout acceleration is a long shot. Even the most "academic" accepted papers used 64+ GPUs or had an industry partner (TLT: MIT+NVIDIA; RLBoost: Berkeley with cloud spot instances).
- **Realistic routes for single-GPU work:**
  - **ML venues (ICLR/ICML/NeurIPS)** with an algorithm-system co-design angle where per-GPU effects are the point: quantized or sparse rollouts, KV/state management, spec decoding with correctness or variance analysis, staleness-quality studies. The paper needs time-to-accuracy curves on 1.5B-8B models.
  - **MLSys**, if the mechanism is clearly per-device (e.g., memory or kernel-level rollout efficiency), with simulation or extrapolation for scale. ReSpec at 3B-14B is the smallest-scale MLSys example I found.
  - **Workshops** (ICML/NeurIPS efficiency or RL workshops, EuroMLSys) for an early version.
- An industry collaboration or cluster-time grant would substantially raise systems-venue odds. 13 of 15 accepted 2026 systems papers had industry co-authors.

### Gaps
- GPU counts for DAS (MLSys'26), HetRL, RLinf, DynaRL, Weave, RollArt, RhymeRL, AReaL (NeurIPS'25), QuRL and DistillSparse were not extracted. pdftotext was unavailable for the MLSys PDF, and I had limited fetch budget.
- I found no confirmed accepted paper in this area evaluated on a single GPU. Whether one exists (e.g., in a workshop) is unverified.

## Q5. Upcoming deadlines as of late October 2026 (for 2027 venues)

### Takeaway
The most urgent deadline is **MLSys 2027: Oct 30, 2026**, per trackers; mlsys.org still shows a placeholder. **OSDI 2027 abstracts are due Dec 1, 2026 and papers Dec 8, 2026.** ICML 2027 is likely late January 2027 (not yet announced), and ATC 2027 (now under ACM SIGOPS) returns to a January deadline. EuroSys 2027, NSDI 2027 and ASPLOS 2027 deadlines have all passed, so the next chances there are the 2028 cycles in spring 2027.

### Cited Findings
- **MLSys 2027.** Paper deadline Oct 30, 2026 at 20:00 UTC; notification Feb 28, 2027; conference June 21-25, 2027, Bellevue, WA. Another tracker estimates Oct 31. The official mlsys.org 2027 CFP page says "Details for MLSys 2027 have not been announced yet." — [mldeadlines MLSys 2027](https://mldeadlines.com/conference/mlsys-2027/); [mlsys.org 2027 CFP placeholder](https://mlsys.org/Conferences/2027/CallForPapers)
- **OSDI 2027** (July 7-9, 2027, Baltimore). Abstract registration Dec 1, 2026 and full papers Dec 8, 2026 (5:59 pm EST). Notification Mar 16, 2027. The CFP is preliminary and warns that submission rates are "in great flux": initial review might cover only the first few pages, to be announced at least a month ahead. — [OSDI '27 preliminary CFP](https://www.usenix.org/node/321091); [OSDI '27 page](https://usenix.org/conference/osdi27)
- **NSDI 2027** (May 11-13, 2027, Providence). The spring deadline (Apr 16, 2026) passed. A tracker lists the fall deadline as Sep 18, 2026, which has also passed. Notification Dec 8, 2026. — [NSDI '27 CFP](https://www.usenix.org/conference/nsdi27/call-for-papers)
- **EuroSys 2027** (April 2027, Rabat, Morocco). Fall-cycle abstract deadline Sep 17, 2026 and paper deadline Sep 24, 2026 (AoE); notification Jan 29, 2027. These dates are from third-party trackers only. — [beri.net EuroSys 2027](https://www.beri.net/events/eurosys-2027); [PaperPilot EuroSys 2027](https://www.getpaperpilot.com/deadlines/eurosys-2027.html)
- **ASPLOS 2027.** Two cycles: April (registration closed; notification Dec 21, 2026) and September (tracker gives Sep 9, 2026). Both have passed. — [ASPLOS'27 CFP](https://www.asplos-conference.org/?p=8512); [asplos27-apr HotCRP](https://asplos27-apr.hotcrp.com/)
- **ATC.** USENIX discontinued ATC after 2025. ACM SIGOPS revived it as the "ACM SIGOPS Annual Technical Conference": ATC 2026 is in Hong Kong (June 2026 deadline, November conference) and ATC 2027 in France. A Wikipedia mirror says that from ATC 2027 the conference returns to a January deadline and July conference. — [Wikipedia: ACM SIGOPS ATC](https://en.wikipedia.org/wiki/ACM_SIGOPS_Annual_Technical_Conference); [Cantrill "RIP USENIX ATC"](https://bcantrill.dtrace.org/2025/05/11/rip-usenix-atc/)
- **ICML 2027.** Listed as "TBD" by a deadline tracker as of Aug 2026. — [trybibby deadlines](https://trybibby.com/conference-deadlines)

### Inferences
- **Practical calendar for a single-GPU student:**
  - MLSys 2027 (Oct 30, 2026) is too soon unless a draft already exists.
  - **OSDI 2027 (Dec 8, 2026)** is feasible only with a strong, at-scale system.
  - **ICML 2027 (likely late Jan 2027)** and **ATC 2027 (January 2027)** are the nearest realistic targets.
  - Then NeurIPS 2027 (likely May 2027), the EuroSys 2028 spring cycle, the NSDI 2028 spring deadline (~April 2027), ASPLOS 2028, and MLSys 2028 (~Oct 2027).
- These timings for ICML 2027, NeurIPS 2027 and the 2028 cycles are inferred from past patterns, not confirmed.

### Gaps
- No official MLSys 2027 or EuroSys 2027 CFP was retrieved; those dates are from trackers.
- I did not check SOSP 2027 (if held), SC 2027, the ICLR 2027 deadline (very likely already passed in fall 2026) or the NeurIPS 2027 dates.
