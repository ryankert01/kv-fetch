# Systems for LLM fine-tuning, training and evaluation (everything except inference serving)

Notes compiled 2026-11-11. Venue status legend used throughout:
- **venue confirmed** = seen on the venue's own program/proceedings page.
- **venue from paper header / DOI only** = the paper's own header or a DOI names the venue, but the venue program page was not reachable.
- **unverified** = only a secondary source (curated list, blog, search snippet) or my own recollection.

Hardware target used for the reproducibility verdicts: ONE node, about 4 idle 48 GB GPUs (RTX A6000 / RTX 6000 Ada, PCIe, no NVLink, no H100/A100), models up to about 30B parameters. The verdicts ("yes" / "scaled down" / "no") are my reading of each paper's reported hardware against that target; they are inferences, not statements by the authors.

## Which top-venue papers (2024-2026) address fine-tuning, training and evaluation systems for LLMs?

### Takeaway
Twenty papers were catalogued: 16 with a top-venue acceptance that could be tied to a venue page, paper header or DOI (EuroSys '26 x2, PVLDB 2025 x2, NeurIPS 2025, NSDI '26, NSDI '25, MLSys 2026 x2, MLSys 2025, USENIX ATC '25, ASPLOS '26, ASPLOS '25, OSDI '26, ICML 2026, ICML 2024), one DAC 2025 paper (outside the requested venue list), and three arXiv-only or venue-unverified papers (PLoRA, LoHan, GoCkpt) kept because they sit directly in these lines of work; further arXiv-only follow-ups (tLoRA, GreedySnake, MegaTrain) are named but not given full records. The densest cluster is multi-LoRA / multi-tenant fine-tuning (mLoRA, LobRA, LoRAFusion, PLoRA, tLoRA, FlexLLM, Loquetier); the next are memory-efficient training/offload, checkpointing, and offline/batch inference.

### Cited Findings

#### PEFT / multi-LoRA / fine-tuning-as-a-service
- **mLoRA** — PVLDB Vol. 18 No. 6 (2025), pp. 1948-1961; venue confirmed on vldb.org with an "Artifact available" badge — [VLDB volume page](https://vldb.org/pvldb/volumes/18/paper/mLoRA%3A%20Fine-Tuning%20LoRA%20Adapters%20via%20Highly-Efficient%20Pipeline%20Parallelism%20in%20Multiple%20GPUs), [PDF](https://www.vldb.org/pvldb/vol18/p1948-tang.pdf)
- **LobRA: Multi-tenant Fine-tuning over Heterogeneous Data** — PVLDB 18(8): 2616-2625, 2025; venue confirmed on vldb.org — [VLDB volume page](https://vldb.org/pvldb/volumes/18/paper/LobRA%3A%20Multi-tenant%20Fine-tuning%20over%20Heterogeneous%20Data), [PDF](https://www.vldb.org/pvldb/vol18/p2616-fu.pdf)
- **LoRAFusion: Efficient LoRA Fine-Tuning for LLMs** — EuroSys '26 (Edinburgh, April 27-30, 2026); venue from paper header and ACM DOI only (the EuroSys 2026 accepted-papers page returned 404 when fetched) — [arXiv HTML](https://arxiv.org/html/2510.00206v1), [ACM DOI](https://dl.acm.org/doi/10.1145/3767295.3769331)
- **FlexLLM: Token-Level Co-Serving of LLM Inference and Finetuning with SLO Guarantees** — NSDI '26; venue confirmed on usenix.org — [USENIX NSDI '26 page](https://www.usenix.org/conference/nsdi26/presentation/oliaro)
- **Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving** — NeurIPS 2025 Main Conference Track; venue confirmed on proceedings.neurips.cc — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5ebc22174f6dff6fa7c2447d0dfa31ea-Abstract-Conference.html)
- **PLoRA** (LoRA hyperparameter-sweep / concurrent LoRA training) — arXiv only; v1 Aug 2025, v2 dated July 2026 with a changed title; no venue confirmed (unverified) — [arXiv 2508.02932](https://arxiv.org/html/2508.02932)
- **tLoRA** — arXiv only (Feb 2026); follow-up to the multi-LoRA line — [arXiv 2602.07263](https://arxiv.org/abs/2602.07263)

#### Memory-efficient training (offload, checkpointing of activations, allocator, long context)
- **ProTrain: Efficient LLM Training via Automatic Memory Management** — MLSys 2026 (poster and oral); venue confirmed on mlsys.org — [MLSys 2026 poster page](https://mlsys.org/virtual/2026/poster/3577), [MLSys 2026 oral page](https://mlsys.org/virtual/2026/oral/3800)
- **Jenga: Enhancing LLM Long-Context Fine-tuning with Contextual Token Sparsity** — USENIX ATC '25; venue confirmed on usenix.org (artifact badges Available / Functional / Reproduced) — [USENIX ATC '25 page](https://www.usenix.org/conference/atc25/presentation/wang-tuowei)
- **STAlloc: Enhancing Memory Efficiency in Large-Scale Model Training with Spatio-Temporal Planning** — EuroSys '26; venue from paper header only (program page not reachable) — [arXiv HTML](https://arxiv.org/html/2507.16274v2)
- **SSDTrain** (SSD-based activation offloading) — DAC 2025 per the Google Research publication page; DAC is not in the requested venue list — [Google Research page](https://research.google/pubs/ssdtrain-faster-large-language-model-training-using-ssd-based-activation-offloading/)
- **LoHan: Low-Cost High-Performance Framework to Fine-Tune 100B Model on a Consumer GPU** — arXiv v2 (Dec 2024); peer-reviewed venue unverified — [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- **GreedySnake** (follow-up to SSD-offloaded training) — arXiv only — [arXiv 2512.17570](https://arxiv.org/abs/2512.17570)

#### Checkpointing / fault tolerance
- **ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development** — NSDI '25; venue confirmed on usenix.org — [USENIX NSDI '25 page](https://www.usenix.org/conference/nsdi25/presentation/wan-borui)
- **PCcheck: Persistent Concurrent Checkpointing for ML** — ASPLOS '25 according to the authors' own artifact repository README (the ASPLOS program page was not fetched) — [eth-easl/pccheck](https://github.com/eth-easl/pccheck)
- **GoCkpt: Gradient-Assisted Multi-Step Overlapped Checkpointing for Efficient LLM Training** — arXiv Nov 2025; venue unverified and conflicting (see Gaps) — [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)

#### Offline / batch LLM inference (synthetic data, distillation, evaluation, analytics)
- **BatchGen: An Architecture for Scalable and Efficient Batch Inference** — OSDI '26; venue confirmed on usenix.org — [USENIX OSDI '26 page](https://www.usenix.org/conference/osdi26/presentation/xu-tairan)
- **BlendServe: Optimizing Offline Inference for Auto-regressive Large Models with Resource-aware Batching** — ASPLOS '26; venue from paper header and DOI only — [arXiv HTML](https://arxiv.org/html/2411.16102v2), [ACM DOI](https://dl.acm.org/doi/abs/10.1145/3779212.3790133)
- **BatchLLM: Optimizing Large Batched LLM Inference with Global Prefix Sharing and Throughput-oriented Token Batching** — MLSys 2026 Industry Track; venue confirmed via the MLSys proceedings PDF — [MLSys proceedings PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/5b7ae1758452854dee4e962207d38304-Paper-Conference.pdf)
- **ConServe: Fine-Grained GPU Harvesting for LLM Online and Offline Co-Serving** — ICML 2026; venue confirmed on icml.cc — [ICML 2026 poster page](https://icml.cc/virtual/2026/poster/62686)
- **Optimizing LLM Queries in Relational Data Analytics Workloads** — MLSys 2025 per the paper header; mlsys.org program page not checked — [arXiv HTML](https://arxiv.org/html/2403.05821)

#### Evaluation cost
- **tinyBenchmarks: evaluating LLMs with fewer examples** — ICML 2024, PMLR 235, pp. 34303-34326; venue confirmed on proceedings.mlr.press — [PMLR page](https://proceedings.mlr.press/v235/maia-polo24a.html)
- **Efficient multi-prompt evaluation of LLMs (PromptEval)** — NeurIPS 2024 poster; venue confirmed on neurips.cc — [NeurIPS 2024 poster page](https://neurips.cc/virtual/2024/poster/93925)

#### Listed in a curated index but not individually verified or read (venue as the index states it)
- The curated LLMSys paper list names, for training systems: BOOST low-rank LLM training (MLSys '26), MoEvement sparse checkpointing (NSDI '26), TrainMover (OSDI '26), Maya GPU runtime emulation (EuroSys '26), "Reducing the GPU Memory Bottleneck with Lossless Compression for ML" (EuroSys '26), MegaScale-Data dataloader (EuroSys '26), "Teaching the Old Dog New Tricks: Building Efficient Data Pipelines for Large-Scale LLM Pre-Training" (OSDI '26), SuperOffload (ASPLOS '26), Charon simulator (MLSys '26), Lumos (MLSys '25), "Understanding Stragglers in Large Model Training Using What-if Analysis" (OSDI '25), Ymir fine-tuning scheduler (ICS '24), Mixtera (no venue listed), BitSnap (arXiv). All unverified here — [AmberLJC/LLMSys-PaperList](https://github.com/AmberLJC/LLMSys-PaperList)
- The PPoPP 2026 accepted-papers page lists "Elastor: Elastic and Efficient Model Partitioning and Checkpointing for Fault-Tolerant Distributed Training" (not read) — [PPoPP 2026 papers](https://ppopp26.sigplan.org/track/PPoPP-2026-papers)

### Inferences
- Within what was found, 2025-2026 fine-tuning-systems papers concentrate on LoRA (many adapters over one frozen base model) rather than full fine-tuning; five of the catalogued papers share the mLoRA/LobRA/LoRAFusion problem framing.
- Papers that treat LLM evaluation cost were found only in ML main tracks (ICML/NeurIPS) and frame it as a statistics problem (which examples to run), not as a systems problem (how to execute a large suite cheaply). In systems venues, evaluation appears only as one motivating workload for offline/batch inference papers.

### Gaps
- EuroSys 2026 accepted-papers page (`https://2026.eurosys.org/accepted-papers.html`) returned 404; LoRAFusion and STAlloc venue status rests on the papers' own headers plus ACM DOIs.
- GoCkpt venue is conflicting: the curated list labels it PPoPP '26, the PPoPP 2026 accepted-papers page did not show it when fetched, and the arXiv header carries an unfilled template with "March 22-26, 2026; Pittsburgh, USA" (the ASPLOS '26 dates and location). Treat as unverified.
- PLoRA: no venue could be confirmed. LoHan: a claim that it appeared at ICDE 2025 (possibly under the name "Ratel") could not be confirmed.
- SIGMOD, SC, HPCA, MICRO and ISCA programs were not searched individually; the papers above came from a curated list plus targeted searches, so coverage of those venues is incomplete.
- No top-venue paper was found on hyperparameter/experiment scheduling for LLM fine-tuning other than PLoRA (arXiv) and the unverified Ymir (ICS '24) entry; none found that frames distillation or synthetic-data generation pipelines as a system beyond the offline/batch inference papers.
- mlsys.org's 2026 paper list loads dynamically and could not be scanned for further in-scope titles.

## For each paper: problem, evaluation setup, single-node reproducibility, stated limitations, follow-ups

### Takeaway
Several accepted papers ran their main experiments on hardware at or below the target node: mLoRA (4x/8x RTX A6000 48 GB over PCIe), ProTrain (4x RTX 3090), Jenga (1x A40 48 GB), PCcheck (1x A100-40GB), tinyBenchmarks (CPU only), and the inference-only half of Loquetier (4x A6000 48G). Most others report at least one single-GPU or 4-GPU configuration with 7B-32B models next to larger H100/A100 runs. Records for the five offline/batch inference papers are in the next section.

### Cited Findings

#### 1. mLoRA (PVLDB 18(6), 2025)
- Links: paper [PDF](https://www.vldb.org/pvldb/vol18/p1948-tang.pdf), [arXiv HTML](https://arxiv.org/html/2312.02515); the VLDB page carries an "Artifact available" badge — [VLDB page](https://vldb.org/pvldb/volumes/18/paper/mLoRA%3A%20Fine-Tuning%20LoRA%20Adapters%20via%20Highly-Efficient%20Pipeline%20Parallelism%20in%20Multiple%20GPUs). Code URL was not visible on the fetched pages (see Gaps).
- Problem: training many LoRA adapters across GPUs or machines with existing model-parallelism schemes suffers "high communication overhead and inefficient GPU utilization"; the stated user is anyone training multiple LoRA tasks on one base model (an AntGroup production deployment is reported) — [arXiv HTML](https://arxiv.org/html/2312.02515)
- Models and workload: Llama2-13B, Llama2-7B, TinyLlama-1.1B; GSM8K; Alpaca-LoRA default hyperparameters (batch 8, sequence length 512, 10 epochs, rank 16 on q/k/v/o_proj) — [arXiv HTML](https://arxiv.org/html/2312.02515)
- Hardware: single machine with four (or eight) NVIDIA RTX A6000 48 GB connected by PCIe 4.0 x16; single-GPU mode on one A6000; multi-machine mode on eight RTX 3090 24 GB across eight machines on 1 Gbps Ethernet; Xeon Silver 4314, 256 GB RAM — [arXiv HTML](https://arxiv.org/html/2312.02515)
- Baselines: HF PEFT (single GPU, with 8-bit quantization and activation checkpointing); 1F1B pipeline (PipeDream-Flush), tensor parallelism (Megatron-LM), FSDP — [arXiv HTML](https://arxiv.org/html/2312.02515)
- Headline: average task completion time reduced 30-45% on single-machine multi-GPU (up to 45% for Llama-2-7B fp32); about 30% vs 1F1B in the multi-machine setting; throughput +75% vs FSDP and +86% vs TP on 1.1B, +35% / +58% on 7B, +46% vs TP on 13B (FSDP out of memory); BatchLoRA on a single GPU gives up to 8% (about 2% for 13B); hyperparameter selection time reduced 30% in AntGroup production — [arXiv HTML](https://arxiv.org/html/2312.02515)
- Reproducible on target node: **yes**. The paper's primary testbed is 4x/8x RTX A6000 48 GB over PCIe with models up to 13B — [arXiv HTML](https://arxiv.org/html/2312.02515)
- Stated limitations (no dedicated section): running LoRA without model parallelism "is impractical in our evaluation due to significant memory limitations"; "the 13B model benefits less due to much higher computational overhead"; authors say "most techniques can be easily applied to other LoRA-based fine-tuning methods" — [arXiv HTML](https://arxiv.org/html/2312.02515)
- Follow-ups: LoRAFusion (EuroSys '26) uses mLoRA as a baseline and reports up to 1.46x (1.29x average) end-to-end speedup over it, and a pipeline bubble ratio of 11.09% vs mLoRA's 34.11% with four adapters — [LoRAFusion arXiv HTML](https://arxiv.org/html/2510.00206v1)

#### 2. LobRA (PVLDB 18(8), 2025)
- Links: [VLDB page](https://vldb.org/pvldb/volumes/18/paper/LobRA%3A%20Multi-tenant%20Fine-tuning%20over%20Heterogeneous%20Data), [PDF](https://www.vldb.org/pvldb/vol18/p2616-fu.pdf), [arXiv HTML](https://arxiv.org/html/2509.01193v1); code in [PKU-DAIR/Hetu](https://github.com/PKU-DAIR/Hetu), full version with appendix at [ccchengff/LobRA](https://github.com/ccchengff/LobRA)
- Problem: for fine-tuning service providers, joint fine-tuning of multiple LoRA adapters over one base model is "dampened by two heterogeneity issues in the training data": sequence-length variation and skewness — [arXiv HTML](https://arxiv.org/html/2509.01193v1)
- Models and workload: Llama2-7B, Qwen2.5-32B, Llama2-70B; 12 fine-tuning datasets treated as tenants' tasks (6 tasks for 7B, 12 for the larger models; dataset names are in the paper's Appendix B.1) — [arXiv HTML](https://arxiv.org/html/2509.01193v1)
- Hardware: environment 1 = 2 servers x 8 A100-40GB (16 GPUs), NVLink plus 100 Gb InfiniBand, used for 7B; environment 2 = 8 servers x 8 A800-80G (64 GPUs), used for 32B and 70B — [arXiv HTML](https://arxiv.org/html/2509.01193v1)
- Frameworks and baselines: built on Hetu; SCIP solver for the MINLP/ILP, FlashAttention, NCCL. Baselines: Task-Fused, Task-Sequential, LobRA-Sequential — [arXiv HTML](https://arxiv.org/html/2509.01193v1)
- Headline: GPU-seconds per joint step reduced 45.03%-60.67% vs Task-Fused; 1.32-1.44x vs LobRA-Sequential; 7B ablation steps 18.94% -> 36.65% -> 45.03% — [arXiv HTML](https://arxiv.org/html/2509.01193v1)
- Reproducible on target node: **scaled down**. Smallest reported config is 16 A100-40GB; on 4x 48 GB one would keep the 7B model and reduce the number of heterogeneous replicas and tasks — hardware facts from [arXiv HTML](https://arxiv.org/html/2509.01193v1)
- Stated limitations (no dedicated section): "some tasks would even experience an efficiency drop when employing LobRA-Sequential"; "In this work, we assume padding is employed."; the cost model assumes time cost is linear in the number of dispatched sequences — [arXiv HTML](https://arxiv.org/html/2509.01193v1)

#### 3. LoRAFusion (EuroSys '26, venue from header/DOI)
- Links: [arXiv HTML](https://arxiv.org/html/2510.00206v1), [ACM DOI](https://dl.acm.org/doi/10.1145/3767295.3769331); code [CentML/lorafusion](https://github.com/CentML/lorafusion), artifact [Zenodo 17051801](https://zenodo.org/records/17051801)
- Problem: existing LoRA fine-tuning systems reuse full-model fine-tuning optimizations, spend time on redundant memory accesses to large activation tensors, and do not jointly fine-tune multiple adapters that share a base model — [arXiv HTML](https://arxiv.org/html/2510.00206v1)
- Models and workload: LLaMa-3.1-8B, Qwen-2.5-32B, LLaMa-3.1-70B; datasets XSum, CNN/DailyMail, WikiSum, plus "Mixed" and "Heterogeneous" settings; 4 LoRA adapters trained in parallel — [arXiv HTML](https://arxiv.org/html/2510.00206v1)
- Hardware: primarily NVIDIA H100 80 GB (8-GPU NVLink nodes, 208 vCPUs, InfiniBand between nodes); additional results on L40S 48 GB servers with 4 GPUs over PCIe; most experiments use 1, 2 or 4 GPUs; scalability study on 4/8/16 H100 — [arXiv HTML](https://arxiv.org/html/2510.00206v1)
- Frameworks and baselines: Triton kernels, Megatron-Core 0.11.0, HF Transformers, PEFT, PyTorch 2.6, CUDA 12.4. Baselines: Megatron-LM with FSDP, Megatron-LM pipeline parallelism, mLoRA (authors state they reimplemented mLoRA's communication with faster primitives) — [arXiv HTML](https://arxiv.org/html/2510.00206v1)
- Headline: up to 1.96x (1.47x average) end-to-end over Megatron-LM; up to 1.46x (1.29x average) over mLoRA; fused kernel up to 1.39x (1.27x average); memory traffic reduced 34-37% — [arXiv HTML](https://arxiv.org/html/2510.00206v1)
- Reproducible on target node: **yes for 8B and 32B, scaled down for 70B**. The paper itself reports 4x L40S 48 GB PCIe results and single-GPU kernel/layer benchmarks — [arXiv HTML](https://arxiv.org/html/2510.00206v1)
- Stated limitations / future work: "we plan to leverage torch.compile by adding compiler annotations as hints"; on one pipeline imbalance case, "This limitation is not solvable by our scheduler and is out of scope for this work."; "Tensor parallelism is not evaluated due to the lack of efficient support in existing LoRA frameworks."; extension to DoRA / VeRA / 4-bit QLoRA is described as possible, with the note that fusing dequantization with the LoRA path is not always faster at large token counts — [arXiv HTML](https://arxiv.org/html/2510.00206v1)
- Follow-ups: tLoRA (arXiv, Feb 2026) reports 1.2-1.8x training throughput, 2.3-5.4x job completion time and +37% GPU utilization on cluster traces — [arXiv 2602.07263](https://arxiv.org/abs/2602.07263)

#### 4. PLoRA (arXiv only; venue unverified)
- Links: [arXiv HTML](https://arxiv.org/html/2508.02932); v1 title "PLoRA: Efficient LoRA Hyperparameter Tuning for Large Models", v2 title "PLoRA: Efficient Concurrent LoRA Training for Large Language Models". No public code found.
- Problem: when sweeping many LoRA hyperparameter configurations, existing LoRA training paradigms "do not efficiently utilize hardware resources" — [arXiv HTML](https://arxiv.org/html/2508.02932)
- Models and workload: Qwen-2.5 3B/7B/14B/32B, LLaMa-3.2-3B, LLaMa-3.1-8B; GSM8K and GLUE MRPC, CoLA, WNLI; sequence length 1024; a search space of 120 LoRA configurations (learning rate, batch size, rank, alpha) — [arXiv HTML](https://arxiv.org/html/2508.02932)
- Hardware: AWS p4d.24xlarge (8x A100 40 GB, NVLink) and g5 (8x A10 24 GB, PCIe Gen4) — [arXiv HTML](https://arxiv.org/html/2508.02932)
- Frameworks and baselines: prototype on torchtune (about 5,000 lines Python + 800 lines CUDA), cvxpy, PyTorch DTensor, CUTLASS; Unsloth default configs as the quality reference. Baselines: Min GPU, Max GPU (TP=8), Sequential PLoRA — [arXiv HTML](https://arxiv.org/html/2508.02932)
- Headline: makespan reduced up to 7.52x; throughput up to 12.8x at batch size 1; the best adapter found beats the default Unsloth configuration by up to 23.4% (CoLA, Qwen-2.5-3B) — [arXiv HTML](https://arxiv.org/html/2508.02932)
- Reproducible on target node: **yes / scaled down**. One of its two testbeds is a single node of 8x A10 24 GB over PCIe; with 4 GPUs the packing space is smaller — hardware facts from [arXiv HTML](https://arxiv.org/html/2508.02932)
- Stated limitations (no dedicated section): as batch size grows "the performance gain diminishes since the Min GPU strategy can better utilize the hardware"; "Further increasing batch sizes harms model quality" — [arXiv HTML](https://arxiv.org/html/2508.02932)

#### 5. FlexLLM (NSDI '26)
- Links: [USENIX page](https://www.usenix.org/conference/nsdi26/presentation/oliaro), [arXiv HTML v3](https://arxiv.org/html/2402.18789v3); project [flexllm.github.io](https://flexllm.github.io), artifact [FlexLLM/artifact](https://github.com/FlexLLM/artifact)
- Problem: inference and fine-tuning run on separate GPU clusters, which wastes resources; FlexLLM co-serves inference and PEFT fine-tuning on shared GPUs by fusing them at token level. Stated user: providers that offer both inference and fine-tuning APIs — [arXiv HTML v3](https://arxiv.org/html/2402.18789v3)
- Models and workload: LLaMA-3.1-8B, Qwen-2.5-14B, Qwen-2.5-32B; LoRA rank 16 on the MLP down-projection. Inference requests use ShareGPT lengths with arrival times from the Azure / BurstGPT trace; fine-tuning data is Sky-T1_data_17k truncated to 8192 tokens; a 10-minute BurstGPT case study with Qwen-2.5-14B; TPOT SLO 50 ms (8B) / 75 ms (14B, 32B), max TTFT 5 s — [arXiv HTML v3](https://arxiv.org/html/2402.18789v3)
- Hardware: Perlmutter nodes with 4x A100-SXM4-80GB each, AMD EPYC 7763, 256 GB DRAM; end-to-end runs use 4, 8 and 16 A100s for the three models — [arXiv HTML v3](https://arxiv.org/html/2402.18789v3)
- Frameworks and baselines: built on FlexFlow Serve (about 9K lines C++/CUDA). Baselines: separate clusters (vLLM + LlamaFactory at 25/75, 50/50, 75/25 GPU splits), temporal sharing, spatial sharing — [arXiv HTML v3](https://arxiv.org/html/2402.18789v3)
- Headline: holds inference SLO up to 20 req/s; fine-tuning throughput 1.9-4.8x under heavy inference load and 2.5-6.8x under light load vs separate clusters; activation memory savings up to 80% end to end — [arXiv HTML v3](https://arxiv.org/html/2402.18789v3)
- Reproducible on target node: **scaled down**. Smallest config is 4x A100 80 GB for the 8B model; on 48 GB PCIe cards the 8B (possibly 14B) config with smaller batch/sequence budgets is the candidate — hardware facts from [arXiv HTML v3](https://arxiv.org/html/2402.18789v3)
- Stated limitations / future work: "extending our implementation beyond offline finetuning scenarios where sequences are processed individually (batch size 1)"; "developing adaptive strategies for workloads with consistently high inference demand"; under stricter SLOs "co-serving introduces service time variance that can affect tail latencies", and the system is described as "most effective in moderate SLO environments"; only base models and LoRA-based PEFT are supported — [arXiv HTML v3](https://arxiv.org/html/2402.18789v3)
- Follow-ups: Loquetier (NeurIPS 2025) uses FlexLLM as a baseline and reports up to 3.0x its throughput on inference-only tasks — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5ebc22174f6dff6fa7c2447d0dfa31ea-Abstract-Conference.html)

#### 6. Loquetier (NeurIPS 2025)
- Links: [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5ebc22174f6dff6fa7c2447d0dfa31ea-Abstract-Conference.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf); code [NJUDeepEngine/Loquetier](https://github.com/NJUDeepEngine/Loquetier)
- Problem: no existing runtime unifies LoRA fine-tuning and LoRA inference over one shared base model — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5ebc22174f6dff6fa7c2447d0dfa31ea-Abstract-Conference.html)
- Models and workload: base model Llama3-8B; ShareGPT for inference inputs; Alpaca and GSM8K for fine-tuning; a real workload trace "comprising over ten million traces collected from Azure OpenAI GPT services" cut into 20-minute slices — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf)
- Hardware: "We test the inference-only tasks on a server with 4 NVIDIA A6000 48G GPUs. We test the fine-tuning-only tasks and the unified fine-tuning and inference tasks on servers with 4 NVIDIA H800 80G GPUs. Each test process has at least 128G of host memory available." — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf)
- Baselines: FlexLLM (docker images), S-LoRA + PEFT, HF Transformers + PEFT. The paper reports difficulties with baselines: FlexLLM supports LoRA only on the up/gate/down MLP modules and lacks backward for other modules; S-LoRA's kernel showed NaN/Inf instability — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf)
- Headline: up to 3.0x the throughput of the state-of-the-art co-serving system on inference-only tasks and 46.4x higher SLO attainment than PEFT on unified fine-tuning and inference tasks — [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5ebc22174f6dff6fa7c2447d0dfa31ea-Abstract-Conference.html)
- Reproducible on target node: **yes for the inference-only experiments (same 4x A6000 48G hardware); scaled down for fine-tuning and unified experiments (run on 4x H800 80G)** — hardware facts from [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf)
- Stated limitations (Appendix A): "Loquetier can be improved by combination with other training and fine-tuning optimization methods, and by applying other cluster management systems to provide clusterized services. We plan to provide a backward propagation kernel operating in concert with the SMLM kernel to accelerate fine-tuning. ... We also consider providing support for other LoRA-like PEFT methods." — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf)

#### 7. ProTrain (MLSys 2026)
- Links: [MLSys 2026 poster page](https://mlsys.org/virtual/2026/poster/3577), [arXiv HTML](https://arxiv.org/html/2406.08334) (v1, June 2024, was titled "ProTrain: Efficient LLM Training via Memory-Aware Techniques"). No code link found.
- Problem: memory pressure limits LLM training on a given GPU budget, and existing frameworks "often expose low-level configuration knobs that require manual tuning and specialized system expertise" — [arXiv HTML](https://arxiv.org/html/2406.08334)
- Models and workload: Mistral 7B; GPT-2 configurations from 1B to 40B; OPT 13B and 30B; LLaMA 13B and 34B; a 175B GPT for multi-node; sequence length 1024; metrics are throughput (tokens/s) and maximum trainable model size, with no named datasets — [arXiv HTML](https://arxiv.org/html/2406.08334)
- Hardware: 4x RTX 3090 24 GB (PCIe 3.0, no NVLink, 384 GB DRAM); 4x A100 80 GB (NVLink, 1 TB DRAM); multi-node 16 A100 across 4 nodes with 100 Gb InfiniBand; single-GPU results as well — [arXiv HTML](https://arxiv.org/html/2406.08334)
- Baselines: DeepSpeed v0.12.1 (ZeRO-3 with offload), Colossal-AI 0.3.3 (Gemini), PyTorch FSDP 2.0.1 — [arXiv HTML](https://arxiv.org/html/2406.08334)
- Headline: throughput 1.43x-2.71x over the baselines; largest trainable GPT-2 on one RTX 3090 is 34B vs 15B (DeepSpeed), 25B (Colossal-AI), 3B (FSDP); on 4x A100, 87B vs 37B / 53B / 55B; the memory/runtime estimator is within about 4-5% runtime error — [arXiv HTML](https://arxiv.org/html/2406.08334)
- Reproducible on target node: **yes**. The 4x RTX 3090 PCIe testbed is weaker than 4x 48 GB PCIe; host DRAM (384 GB in the paper) is the resource to check — hardware facts from [arXiv HTML](https://arxiv.org/html/2406.08334)
- Stated limitations: "ProTrain focuses on data-parallel training with ZeRO."; the user still chooses the parallelism configuration (the paper contrasts this with Mist); "ProTrain takes a narrower but deeper approach." — [arXiv HTML](https://arxiv.org/html/2406.08334)
- Note on an internal inconsistency: the abstract gives 1.43x-2.71x while the conclusion text says "up to 5x" — [arXiv HTML](https://arxiv.org/html/2406.08334)

#### 8. SSDTrain (DAC 2025; outside the requested venue list)
- Links: [Google Research page](https://research.google/pubs/ssdtrain-faster-large-language-model-training-using-ssd-based-activation-offloading/), [arXiv HTML](https://arxiv.org/html/2408.10013); code [K-Wu/FlashTrain](https://github.com/K-Wu/FlashTrain)
- Problem: GPU memory growth does not keep up with LLM size and activations dominate training memory; SSDTrain offloads activations to NVMe SSDs — [arXiv HTML](https://arxiv.org/html/2408.10013)
- Models and workload: BERT, GPT and T5 with (hidden size, layers) = (8192, 4), (12288, 3), (16384, 2); OSCAR dataset; batch 16, sequence length 1024 — [arXiv HTML](https://arxiv.org/html/2408.10013)
- Hardware: one machine with 2x A100 40 GB, 2x EPYC 7702, 1 TB RAM, 7x Intel Optane P5800X 1.6 TB in two RAID0 arrays — [arXiv HTML](https://arxiv.org/html/2408.10013)
- Frameworks and baselines: PyTorch 2.2.2, DeepSpeed 0.14.2, Megatron-DeepSpeed, kvikio 24.08; two-GPU tensor parallelism; FP16 and SGD used to fit 40 GB. Baselines: no offloading; layerwise recomputation (activation checkpointing) — [arXiv HTML](https://arxiv.org/html/2408.10013)
- Headline: activation peak memory reduced by up to 47% (arXiv) / 45% (DAC abstract) with negligible overhead — [arXiv HTML](https://arxiv.org/html/2408.10013); [Google Research page](https://research.google/pubs/ssdtrain-faster-large-language-model-training-using-ssd-based-activation-offloading/)
- Reproducible on target node: **scaled down**. The GPUs fit (2x 40 GB in the paper); the dependency is a fast NVMe array (the paper used 7 Optane drives in RAID0) — hardware facts from [arXiv HTML](https://arxiv.org/html/2408.10013)
- Stated limitations: "The CPU offloader is for future work on clusters with massive remote SSD storage."; the SSD-lifespan analysis rests on write-amplification assumptions — [arXiv HTML](https://arxiv.org/html/2408.10013)

#### 9. LoHan (arXiv v2 Dec 2024; peer-reviewed venue unverified)
- Links: [arXiv HTML](https://arxiv.org/html/2403.06504v2); code [RC4ML/LoHan](https://github.com/RC4ML/LoHan)
- Problem: fine-tuning very large models "on a single consumer-grade GPU in a commodity server with limited main memory capacity", using SSD and host-memory offload — [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- Models and workload: decoder-only models from 6B to 412B (GPT-3 / OPT hyperparameters), sequence length 1024, mixed precision, random data; DiT-XL/2-based diffusion models 0.67B-40B — [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- Hardware: dual Xeon Gold 5320, 768 GB DDR4, one RTX 3090 / 4080 / 4090 per test, 12x 3.84 TB Intel P5510 SSDs; a multi-GPU test with 4x RTX 4090 — [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- Baselines: ZeRO-Infinity and ZeRO-Offload (DeepSpeed 0.9.3), Colossal-AI 0.3.5, a FlashNeuron prototype; Megatron-LM on DGX-A100 for the cost comparison — [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- Headline: first to fine-tune a 175B model on an RTX 4090 with 256 GB host RAM; for 13B on a 4090, 2.32x ZeRO-Offload, 3.46x ZeRO-Infinity, 8.02x Colossal-AI; on 4 GPUs, 2.21x (13B) and 1.69x (70B) vs ZeRO-Infinity — [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- Reproducible on target node: **yes for the GPU side; depends on host RAM and the number of NVMe SSDs** (the paper uses up to 12 SSDs and 768 GB RAM) — hardware facts from [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- Stated limitations: "LoHan maintains a relatively small 53% of peak FLOPS when fine-tuning a 175B model."; asynchronous optimizers are "beyond the scope of this paper"; cost-effectiveness drops when going from 6 to 12 SSDs — [arXiv HTML](https://arxiv.org/html/2403.06504v2)
- Follow-ups: GreedySnake (arXiv only) reports, vs ZeRO-Infinity on A100, 1.96x on 1 GPU and 1.93x on 4 GPUs for GPT-65B, and 2.53x on 1 GPU for GPT-175B — [arXiv 2512.17570](https://arxiv.org/abs/2512.17570); MegaTrain (arXiv only) reports 1.84x over ZeRO-3 Offload at 14B using host memory on GH200 — [arXiv 2604.05091](https://arxiv.org/abs/2604.05091)

#### 10. Jenga (USENIX ATC '25)
- Links: [USENIX page](https://www.usenix.org/conference/atc25/presentation/wang-tuowei), [PDF](https://www.usenix.org/system/files/atc25-wang-tuowei.pdf); arXiv preprint under the name "LeMo": [arXiv HTML](https://arxiv.org/html/2501.09767v1). Artifact repository name seen only in a search result (Pairshoe/Jenga-AE), not opened.
- Problem: in long-context fine-tuning, "high memory footprints, especially for activations, present a critical practical limitation" — [arXiv HTML](https://arxiv.org/html/2501.09767v1)
- Models and workload: OPT 350M / 1.3B / 2.7B / 6.7B, Llama2-7B, Llama3-8B; RedPajama fine-tuning following the LongLoRA setup; PG19 and proof-pile perplexity; LongBench after LongAlign-10k; sequence lengths 2K-64K — [arXiv HTML](https://arxiv.org/html/2501.09767v1)
- Hardware: platform A = 1x A800 80 GB; platform B = 1x A40 48 GB; platform C = 4x RTX 4090 24 GB — [arXiv HTML](https://arxiv.org/html/2501.09767v1)
- Baselines: LoRA, LongLoRA — [arXiv HTML](https://arxiv.org/html/2501.09767v1)
- Headline: memory reduced up to 1.93x and speed improved up to 1.36x — [USENIX page](https://www.usenix.org/conference/atc25/presentation/wang-tuowei)
- Reproducible on target node: **yes**. One of its three platforms is a single A40 48 GB — hardware facts from [arXiv HTML](https://arxiv.org/html/2501.09767v1)
- Stated limitations (no dedicated section): a slight perplexity increase is reported (7.11 vs 6.95 at 8K on PG19); "We envision LeMo inspiring broader exploration of sparsity for advancing LLMs." — [arXiv HTML](https://arxiv.org/html/2501.09767v1)

#### 11. STAlloc (EuroSys '26, venue from paper header)
- Links: [arXiv HTML](https://arxiv.org/html/2507.16274v2) (v1 was named STWeaver); code [infinigence/STAlloc](https://github.com/infinigence/STAlloc)
- Problem: PyTorch's online caching allocator ignores tensor lifespans; the paper reports fragmentation wasting up to 43% of memory and causing out-of-memory failures in large-model training — [arXiv HTML](https://arxiv.org/html/2507.16274v2)
- Models and workload: GPT-2 345M, Llama2-7B, Qwen1.5-MoE-A2.7B, Qwen2.5 7B / 14B / 32B / 72B; measured over training iterations, no named datasets — [arXiv HTML](https://arxiv.org/html/2507.16274v2)
- Hardware: one node with 8x A800-80GB; up to 16 nodes x 8 H200-141GB; 8 nodes x 8 MI210-64GB — [arXiv HTML](https://arxiv.org/html/2507.16274v2)
- Frameworks and baselines: Megatron-LM and Colossal-AI as host frameworks. Baselines: PyTorch caching allocator (2.0 / 2.3 / 2.6), PyTorch `expandable_segments`, GMLake — [arXiv HTML](https://arxiv.org/html/2507.16274v2)
- Headline: fragmentation reduced 85.1% on average (up to 100%); up to 56.3 GB saved; throughput up to +32.5% (Qwen2.5-14B on 16 GPUs); plan synthesis about 2 minutes (up to 6) — [arXiv HTML](https://arxiv.org/html/2507.16274v2)
- Reproducible on target node: **scaled down**. Smallest testbed is 8x A800-80GB; the allocator is a PyTorch plug-in, so 7B-14B runs on 4x 48 GB would change model/batch sizes — hardware facts from [arXiv HTML](https://arxiv.org/html/2507.16274v2)
- Stated limitations: no dedicated section; related approaches are called "complementary and orthogonal" — [arXiv HTML](https://arxiv.org/html/2507.16274v2)

#### 12. ByteCheckpoint (NSDI '25)
- Links: [USENIX page](https://www.usenix.org/conference/nsdi25/presentation/wan-borui), [arXiv HTML](https://arxiv.org/html/2407.20143v4); code [ByteDance-Seed/ByteCheckpoint](https://github.com/ByteDance-Seed/ByteCheckpoint)
- Problem: checkpoint save/load for large foundation-model training is slow and tied to one framework and one parallelism layout; ByteCheckpoint adds a unified format with load-time resharding. Stated user: industrial-scale training teams (ByteDance) — [arXiv HTML](https://arxiv.org/html/2407.20143v4)
- Models and workload: a 4B video DiT (FSDP) on A100 80 GB, 32 -> 64/128 GPUs; a 70B GPT (Megatron-LM) on H800, 2,400 -> 4,800 GPUs; 13B / 30B GPT microbenchmarks on 32 GPUs; HDFS storage — [arXiv HTML](https://arxiv.org/html/2407.20143v4)
- Baselines: PyTorch Distributed Checkpoint (DCP) and Megatron MCP — [arXiv HTML](https://arxiv.org/html/2407.20143v4)
- Headline: checkpoint stalls reduced 54.20x on average (12.13x-161.50x); saving up to 9.96x and loading up to 8.80x faster — [arXiv HTML](https://arxiv.org/html/2407.20143v4)
- Reproducible on target node: **no at paper scale** (smallest configuration is 32 A100-80GB with HDFS); the library itself is open source — hardware facts from [arXiv HTML](https://arxiv.org/html/2407.20143v4)
- Stated limitations: "We leave the design of a more efficient dataloader to our future works."; dataloader state increases resharding time; stragglers during save/load are discussed — [arXiv HTML](https://arxiv.org/html/2407.20143v4)

#### 13. PCcheck (ASPLOS '25 per authors' repository)
- Links: code and artifact [eth-easl/pccheck](https://github.com/eth-easl/pccheck). The paper text was not fetched; results below marked "secondary" come from a third-party slide summary.
- Problem: existing DNN checkpointing mechanisms allow only one in-flight checkpoint, so checkpointing frequently stalls training; PCcheck allows concurrent persistent checkpoints — [eth-easl/pccheck](https://github.com/eth-easl/pccheck)
- Artifact setup: one NVIDIA A100-40GB on a GCP a2-highgpu-1g VM, 12 vCPUs, 85 GB DRAM, 1 TB pd-ssd, Ubuntu 20.04; model directories include bert, opt, transformer, vision and a distributed LLM case; integrated baselines CheckFreq, DeepSpeed, Gemini, GPM; scripts reproduce Figures 8, 9, 11, 12 — [eth-easl/pccheck](https://github.com/eth-easl/pccheck)
- Headline (secondary, unverified against the paper): about 3% overhead when checkpointing every 10 iterations; goodput under spot-preemption traces up to 2.86x vs CheckFreq, 1.75x vs GPM, 2.75x vs Gemini; models include OPT-1.3B, BERT, VGG16 — [third-party slide summary](https://senbon-sakura.github.io/slide_html/pccheck-asplos-2025/)
- Reproducible on target node: **yes** (single GPU plus one SSD in the artifact) — hardware facts from [eth-easl/pccheck](https://github.com/eth-easl/pccheck)

#### 14. GoCkpt (arXiv Nov 2025; venue unverified)
- Links: [arXiv 2511.07035](https://arxiv.org/abs/2511.07035). No code found.
- Problem: existing checkpointing "often pause[s] or delay[s] GPU computation during checkpoint saving"; GoCkpt overlaps checkpoint transfer across several training steps and reconstructs a consistent state from gradients — [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)
- Models and workload: LLaMA3.2-1B, Qwen3-0.6B, OPT-350M on a single GPU; LLaMA3-8B on multiple GPUs; Wikidata — [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)
- Hardware: one Tesla V100S (96 cores, 128 GB RAM, 3.8 TB NVMe); an Alibaba Cloud server with 8x H100-80GB (4-GPU runs); integrated into DeepSpeed — [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)
- Baselines: DeepSpeed, PyTorch DCP (synchronous), DCP-Async, Async, Async-O, and an "Ideal" bound — [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)
- Headline: training throughput +38.4% vs traditional asynchronous checkpointing; model interruption time reduced 86.7% and throughput +4.8% vs state-of-the-art checkpoint-transfer methods — [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)
- Reproducible on target node: **yes** (single-GPU experiments with models of 1B or less on a V100S; the 4-GPU LLaMA3-8B run used H100s) — hardware facts from [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)
- Stated limitations: no dedicated section; the design falls back to blocking transfer for checkpoints not sent within its multi-step window — [arXiv 2511.07035](https://arxiv.org/abs/2511.07035)

#### 15. tinyBenchmarks (ICML 2024)
- Links: [PMLR page](https://proceedings.mlr.press/v235/maia-polo24a.html), [arXiv HTML](https://arxiv.org/html/2402.14992); code [felipemaiapolo/tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks), reproduction code [felipemaiapolo/efficbench](https://github.com/felipemaiapolo/efficbench), datasets [huggingface.co/tinyBenchmarks](https://huggingface.co/tinyBenchmarks)
- Problem: evaluating an LLM on full benchmarks costs many inference calls; the paper asks how few examples are needed to estimate benchmark scores. Stated users: practitioners evaluating many models or checkpoints — [arXiv HTML](https://arxiv.org/html/2402.14992)
- Data and workload: per-example correctness records from the Open LLM Leaderboard (6 scenarios, about 29K examples, about 400 models), MMLU (about 14K examples), HELM Lite v1.0.0 (about 10K examples, 30 models), AlpacaEval 2.0 (805 examples, 100 models); IRT fitting runs on a CPU — [arXiv HTML](https://arxiv.org/html/2402.14992)
- Baselines: stratified random sampling, correctness-based clustering (anchor points), IRT-based clustering, each with IRT-estimator variants — [arXiv HTML](https://arxiv.org/html/2402.14992)
- Headline: 100 curated examples per scenario estimate performance within about 2% error on average; MMLU reduced from 14K to 100 examples — [PMLR page](https://proceedings.mlr.press/v235/maia-polo24a.html)
- Reproducible on target node: **yes** (CPU-only analysis over public per-example results; GPUs only needed to add new models) — setup facts from [arXiv HTML](https://arxiv.org/html/2402.14992)
- Stated limitations: "The main limitations of the methods described in this paper are related to potential severe distribution shifts."; "A rapid increase in LLM capabilities may also cause extrapolation errors."; "we recommend periodically updating the curated examples and IRT parameter estimates using data from more modern LLMs." — [arXiv HTML](https://arxiv.org/html/2402.14992)
- Follow-ups: PromptEval, "Efficient multi-prompt evaluation of LLMs" (NeurIPS 2024), extends the approach to estimating performance across many prompt templates — [NeurIPS 2024 poster page](https://neurips.cc/virtual/2024/poster/93925); "Towards Optimal Evaluation Efficiency for Large Language Models" (EMNLP 2025) — [ACL Anthology preview](https://preview.aclanthology.org/setup/2025.emnlp-main.716); Scales++ (arXiv preprint) — [arXiv 2510.26384](https://arxiv.org/abs/2510.26384)

### Inferences
- Papers whose own testbed already matches or undercuts the target node (mLoRA, ProTrain's RTX 3090 testbed, Jenga's A40 platform, PCcheck, GoCkpt's single-GPU runs, LoHan's consumer-GPU runs, tinyBenchmarks) need no hardware substitution; the others (LobRA, FlexLLM, STAlloc, Loquetier's fine-tuning half, LoRAFusion's 70B runs) would have absolute numbers change and need smaller models or fewer replicas.
- Crowdedness as visible from citations between the catalogued papers: the multi-LoRA line has at least five systems in two years, each using the previous one as a baseline (mLoRA -> LoRAFusion -> tLoRA; FlexLLM -> Loquetier). SSD/host-memory offloaded training has LoHan followed by GreedySnake and MegaTrain (both arXiv only). No follow-up systems paper to Jenga, STAlloc, or PLoRA was found in this search.
- Many of these papers lack a dedicated limitations section (mLoRA, LobRA, PLoRA, Jenga, STAlloc, GoCkpt); the limitation quotes above for those were pulled from evaluation and discussion text.
- Several training-memory papers report throughput on synthetic or unnamed data (ProTrain, LoHan, STAlloc), so reproducing them needs only model configs, not datasets.

### Gaps
- mLoRA code URL: I recall it as github.com/TUDB-Labs/mLoRA but did not see it on any fetched page; unverified.
- LobRA's 12 dataset names (Appendix B.1 of the full version) were not captured.
- PCcheck: the ASPLOS '25 paper itself was not read; headline numbers come from a third-party slide summary and the authors' own limitations text is unknown.
- Jenga artifact repository (Pairshoe/Jenga-AE) was seen only in a search snippet.
- ProTrain: no public code located; the "1.43-2.71x" vs "up to 5x" discrepancy inside the paper was not resolved.
- tLoRA baselines and hardware were not captured (abstract only). An "ALTO" multi-LoRA system (2026) appeared only in a secondary source and is not recorded as a finding.
- Follow-up/crowdedness lists are based on what surfaced in searches, not on a forward-citation crawl (no Google Scholar / Semantic Scholar "cited by" pass was done), so they are lower bounds.
- Host RAM and NVMe configuration of the student's node are unknown to me; LoHan, SSDTrain and ProTrain verdicts depend on them.

## Which papers treat offline/batch LLM inference (synthetic data, distillation, evaluation) as its own workload, and what setups do they use?

### Takeaway
Five papers frame offline/batch inference as distinct from online serving: BatchGen (OSDI '26), BlendServe (ASPLOS '26), BatchLLM (MLSys '26 Industry), ConServe (ICML '26, offline co-served with online) and "Optimizing LLM Queries in Relational Data Analytics Workloads" (MLSys '25). All replace TTFT/TPOT with throughput or whole-batch completion time and exploit knowing the full request set in advance; three of them have single-GPU experiments with 7B-14B models.

### Cited Findings

#### How each paper defines the offline workload
- BatchGen: "Batch inference systems focus on minimizing the completion time of the entire batch" (batch completion time, BCT), contrasted with TTFT/TPOT for interactive serving; its workloads include benchmark evaluation (LongBench), Recursive Self-Aggregation and RL rollouts — [arXiv HTML](https://arxiv.org/html/2606.21712)
- BlendServe: "As the latency objective is highly relaxed, maximizing throughput becomes the primary goal in offline inference." — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- BatchLLM: existing engines are built for streaming requests; implicit LRU prefix caching can evict KV that later requests could reuse, and per-request scheduling limits GPU saturation, whereas in offline batch jobs the prompts and their prefix structure are known ahead of time — [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- ConServe: co-serves "latency-tolerant offline tasks such as model benchmarking" with online requests on the same GPUs; "offline requests can be freely preempted within the engine" — [arXiv HTML](https://arxiv.org/html/2410.01228)
- Optimizing LLM Queries: "we identify and present solutions to optimize relational data analytics workloads for offline LLM inference"; "Since the full table structure and content are known in advance, we can reorder these requests." — [arXiv HTML](https://arxiv.org/html/2403.05821)

#### 16. BatchGen (OSDI '26)
- Links: [USENIX page](https://www.usenix.org/conference/osdi26/presentation/xu-tairan), [PDF](https://www.usenix.org/system/files/osdi26-xu-tairan.pdf), [arXiv HTML](https://arxiv.org/html/2606.21712). The paper says "We will open-source BatchGen at" [batchgen-project/batchgen](https://github.com/batchgen-project/batchgen); release status not checked.
- Problem: large batch jobs on large MoE models are run through engines designed for interactive serving; who has it: teams running evaluation, data generation and RL-rollout batches (Tencent co-authors) — [arXiv HTML](https://arxiv.org/html/2606.21712)
- Models: Mixtral-8x7B (47B), Mixtral-8x22B (141B), DeepSeek-R1 (671B), Kimi-K2 (1T) — [arXiv HTML](https://arxiv.org/html/2606.21712)
- Workloads: LongBench, 6K sequences (8K-in/2K-out and 2K-in/8K-out); Recursive Self-Aggregation on DeepSeek-R1; VeRL RL rollouts (256 prompts per iteration); 10K-request workloads derived from production traces; GSM8K (8.5K) and ChatBotArena (36K) on a single A5000 — [arXiv HTML](https://arxiv.org/html/2606.21712)
- Hardware: H20 and H200 servers, 8-128 GPUs, 2 TB host memory, 200 Gb InfiniBand; memory-constrained setting: one RTX A5000 24 GB with a 28-core AMD 7453 and 1 TB host memory — [arXiv HTML](https://arxiv.org/html/2606.21712)
- Baselines: SGLang v0.5.5.post3 (default and tuned), vLLM v0.11.2, TensorRT-LLM, DeepSpeed, FlexGen (reproduced), MoE-Lightning (reproduced), SGLang prefill/decode disaggregation, VeRL — [arXiv HTML](https://arxiv.org/html/2606.21712)
- Headline: BCT up to 2.3x lower on 128 GPUs; up to 9.6x over the strongest offloading baseline on one A5000 (Mixtral-8x22B on GSM8K: 5.1 h vs 32.5 h for MoE-Lightning) — [arXiv HTML](https://arxiv.org/html/2606.21712)
- Reproducible on target node: **scaled down**. The single-A5000 offloading experiment fits the GPU budget but used 1 TB host memory; the 8-128 GPU cluster results do not fit — hardware facts from [arXiv HTML](https://arxiv.org/html/2606.21712)
- Stated limitations: "In our implementation, gains plateau beyond 64 GPUs"; "Our current implementation targets transformer MoE models"; "Extending BatchGen to other models and tasks is future work." — [arXiv HTML](https://arxiv.org/html/2606.21712)

#### 17. BlendServe (ASPLOS '26, venue from header/DOI)
- Links: [arXiv HTML](https://arxiv.org/html/2411.16102v2), [author PDF](https://homes.cs.washington.edu/~baris/public/blendserve.pdf), [ACM DOI](https://dl.acm.org/doi/abs/10.1145/3779212.3790133). Artifact is distributed as Docker images on Google Drive ([GPU image](https://drive.google.com/file/d/1fVe8C1_tSAjiTW9UZ0Fsz7udHuCHiy2U/view?usp=sharing)); no GitHub link in the paper.
- Problem: offline batch APIs receive mixed requests (text, multimodal, different input/output lengths); ordering them to overlap compute-heavy and memory-heavy requests while keeping prefix sharing — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Models: Llama-3-8B, Llama-3-70B, Qwen-2.5-7B, Llama-2-7B, Qwen-2.5-72B, DeepSeek-67B — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Workloads: WildChat, ShareGPT, Azure-Trace, BurstGPT, OpenVid-1M, MMLU, combined into four synthesized mixes (Trace#1-#4) — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Hardware: A100-SXM4-80GB; 1 GPU for 7B/8B models, 8 GPUs (TP=8) for 70B-class; the artifact appendix estimates about 50 A100-hours and about 50 GB disk, and provides a non-GPU image that runs the simulator — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Frameworks and baselines: built on SGLang and NanoFlow. Baselines: vLLM v0.6.3.post2.dev102, SGLang v0.3.4.post1, NanoFlow-DFS, NanoFlow-Balance, DistServe — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Headline: up to 1.44x throughput over vLLM and SGLang; +20.84% on average over NanoFlow (8B); 86.55% of the practical optimal (8B) — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Reproducible on target node: **yes for the 7B/8B single-GPU experiments (A100 80 GB -> 48 GB changes batch capacity and absolute numbers); scaled down for the 70B-class TP=8 runs** — hardware facts from [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Stated limitations: "Searching for an optimal request order is NP-hard."; "predicting output length may not always be accurate due to the dynamic nature of decoding"; benefits "tend to shrink with smaller compute density"; on ShareGPT/WildChat with large output-length variance "the sampling strategy works less effectively, leading to sub-optimal performance" — [arXiv HTML](https://arxiv.org/html/2411.16102v2)
- Note on an internal inconsistency: the main text says traces have "at least 400,000" requests while Appendix A.3 says 40,000 — [arXiv HTML](https://arxiv.org/html/2411.16102v2)

#### 18. BatchLLM (MLSys 2026 Industry Track)
- Links: [MLSys proceedings PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/5b7ae1758452854dee4e962207d38304-Paper-Conference.pdf), [arXiv HTML](https://arxiv.org/html/2412.03594v3); code [microsoft/MixLLM, branch batchllm_vllm_064](https://github.com/microsoft/MixLLM/tree/batchllm_vllm_064)
- Problem: large batched jobs with shared prompt prefixes (Microsoft production snippet generation) lose prefix reuse and GPU saturation under streaming-oriented engines — [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- Models: Llama-3.1-8B, Qwen-2.5-14B, Llama-3.1-70B (4-bit GPTQ), Qwen-2.5-7B — [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- Workloads: synthetic microbenchmarks (shared-prefix/distinct lengths 2000/200, 2000/1000, 1000/1000; 100 generated tokens; 6400 requests, 800 for 70B); an industry snippet-generation workload (8000 queries, sharing degree about 3, prefix about 1570 tokens); a Pareto heavy-tail workload (320 requests); no public datasets — [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- Hardware: 1x A100 80 GB and 2x A100 (TP=2); AMD MI200 — [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- Frameworks and baselines: built on vLLM v0.6.4. Baselines: vLLM v0.6.4 with prefix caching and chunked prefill, SGLang, and a January 2026 vLLM commit — [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- Headline: 1.3x-10.8x over vLLM and SGLang; 1.30x on the industry workload — [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- Reproducible on target node: **yes** (single-GPU experiments; 8B and 14B fit in 48 GB; the workloads must be re-synthesized from the stated parameters since no dataset is released) — setup facts from [arXiv HTML](https://arxiv.org/html/2412.03594v3)
- Stated limitations: the design is "less suitable for latency-critical workloads"; "the advantages of BatchLLM are diminished in environments lacking common prefixes"; on switching between streaming and batching: "We leave the implementation of such a hybrid system for future work." — [arXiv HTML](https://arxiv.org/html/2412.03594v3)

#### 19. ConServe (ICML 2026)
- Links: [ICML poster page](https://icml.cc/virtual/2026/poster/62686), [author PDF](https://web.cs.ucla.edu/~harryxu/papers/conserve-icml26.pdf), [arXiv HTML](https://arxiv.org/html/2410.01228). No code link found.
- Problem: GPUs provisioned for peak online load sit idle off-peak; harvesting them for offline jobs without hurting online latency. Who has it: providers running both interactive and batch APIs — [arXiv HTML](https://arxiv.org/html/2410.01228)
- Models: Llama-3.1 8B, Qwen-2.5 14B, Llama-3.1 70B (FP16) — [arXiv HTML](https://arxiv.org/html/2410.01228)
- Workloads: online = BurstGPT trace plus synthetic Gamma arrivals; offline = DuReader, MultiNews, VCSUM (LongBench document tasks) and a mix — [arXiv HTML](https://arxiv.org/html/2410.01228)
- Hardware: one DGX with 8x H100; 8B and 14B on one H100, 70B on four H100 with tensor parallelism; built on vLLM 0.8.4 with chunked prefill — [arXiv HTML](https://arxiv.org/html/2410.01228)
- Baselines: Online-Only, Non-Preemptive co-serving, Sarathi-Preemptive, DistServe-Preemptive — [arXiv HTML](https://arxiv.org/html/2410.01228)
- Headline: 2.2x higher offline throughput and 2.9x lower online tail latency on average vs state-of-the-art; 82.3% of optimal offline throughput — [arXiv HTML](https://arxiv.org/html/2410.01228)
- Reproducible on target node: **scaled down** (the single-GPU 8B and 14B configurations fit in 48 GB; H100 -> A6000 changes absolute latency/throughput and SLO thresholds) — hardware facts from [arXiv HTML](https://arxiv.org/html/2410.01228)
- Stated limitations: no dedicated section; the paper states compatibility with BlendServe-style offline ordering and that it "can gracefully degrade and adapt to the available host memory size" for its checkpointed KV state — [arXiv HTML](https://arxiv.org/html/2410.01228)

#### 20. Optimizing LLM Queries in Relational Data Analytics Workloads (MLSys 2025 per paper header)
- Links: [arXiv HTML](https://arxiv.org/html/2403.05821). No code link found.
- Problem: SQL-style analytics that call an LLM per row issue many near-duplicate prompts; reordering rows and fields raises prefix-cache hit rate. Who has it: analytics users running LLM operators over tables — [arXiv HTML](https://arxiv.org/html/2403.05821)
- Models: Llama-3-8B-Instruct (main), Llama-3-70B-Instruct, Llama-3.2-1B; GPT-4o-mini, GPT-4o and Claude 3.5 Sonnet for the API-cost study — [arXiv HTML](https://arxiv.org/html/2403.05821)
- Workloads: 7 datasets (Rotten Tomatoes Movies, Amazon Products, BIRD, PDMX, RateBeer Beer, SQuAD, FEVER) and 16 queries over 5 query types — [arXiv HTML](https://arxiv.org/html/2403.05821)
- Hardware: a single NVIDIA L4 24 GB (GCP g2-standard-4) for 8B; 8x L4 (g2-standard-48) with tensor parallelism for 70B; PySpark + vLLM — [arXiv HTML](https://arxiv.org/html/2403.05821)
- Baselines: No Cache; Cache (Original order) — [arXiv HTML](https://arxiv.org/html/2403.05821)
- Headline: 1.5-3.4x faster than Cache (Original); 32% cost savings under OpenAI and Anthropic pricing — [arXiv HTML](https://arxiv.org/html/2403.05821)
- Reproducible on target node: **yes** (the paper's hardware is L4 24 GB-class) — hardware facts from [arXiv HTML](https://arxiv.org/html/2403.05821)
- Stated limitations: the authors built their own benchmark "Due to the lack of standard workloads in this area"; the greedy reordering "may be suboptimal" on ties; one dataset (PDMX) shows lower speedup because of long inputs and a 43% cache-miss rate — [arXiv HTML](https://arxiv.org/html/2403.05821)

#### Shared elements of the setups
- Traces reused across these papers: BurstGPT (BlendServe, ConServe; also FlexLLM), ShareGPT (BlendServe; also FlexLLM and Loquetier), LongBench or its subsets (BatchGen, ConServe; also Jenga), MMLU (BlendServe; also tinyBenchmarks), GSM8K (BatchGen; also mLoRA, PLoRA, Loquetier) — [BlendServe](https://arxiv.org/html/2411.16102v2), [ConServe](https://arxiv.org/html/2410.01228), [BatchGen](https://arxiv.org/html/2606.21712), [FlexLLM](https://arxiv.org/html/2402.18789v3)
- Baseline engines: vLLM and SGLang appear as baselines in BatchGen, BlendServe and BatchLLM; ConServe and the relational-analytics paper are built on vLLM — [BatchGen](https://arxiv.org/html/2606.21712), [BlendServe](https://arxiv.org/html/2411.16102v2), [BatchLLM](https://arxiv.org/html/2412.03594v3), [ConServe](https://arxiv.org/html/2410.01228), [Optimizing LLM Queries](https://arxiv.org/html/2403.05821)

### Inferences
- The offline papers differ in what "known in advance" is exploited: prefix structure (BatchLLM, relational analytics), per-request compute/memory demand (BlendServe), sequence-level scheduling freedom and host-memory offload (BatchGen), and preemptibility (ConServe).
- None of the five was evaluated on a named distillation or synthetic-data-generation job; evaluation suites (LongBench, MMLU), RL rollouts, analytics queries and production-derived traces stand in for the offline workload.
- Engine versions used as baselines are already old relative to today (vLLM 0.6.x in BlendServe and BatchLLM; BatchLLM added a January 2026 vLLM commit as an extra baseline), so a reproduction would have to decide which engine version to compare against.

### Gaps
- BatchGen code release status and ConServe code availability were not confirmed.
- ConServe details come from arXiv v2 (Sep 2025); the camera-ready ICML PDF could not be text-extracted, so any changes in the final version are not reflected.
- BatchLLM's production workload is not public; BlendServe's mixed traces are synthesized by the authors and distributed only inside the Docker artifact (not inspected).
- I found no top-venue paper that benchmarks an end-to-end distillation or synthetic-data pipeline (generation + filtering + training) as a system; this may be a search gap rather than an absence.
- Searches did not specifically cover batch-inference work in data-management venues (SIGMOD/VLDB LLM-operator papers beyond the one above).

## Which problems do several papers name as still open?

### Takeaway
The recurring self-stated open items are: output-length unpredictability and hybrid online/offline operation in batch inference; unfinished backward-pass/fine-tuning support and missing tensor-parallel and non-LoRA PEFT support in multi-LoRA and co-serving systems; shrinking gains at larger batch sizes and models; host-memory/SSD-bound offload efficiency; and a lack of standard public workloads.

### Cited Findings
- Output-length uncertainty limits offline scheduling: BlendServe states "predicting output length may not always be accurate due to the dynamic nature of decoding" and that with large output-length variance its "sampling strategy works less effectively, leading to sub-optimal performance" — [BlendServe](https://arxiv.org/html/2411.16102v2)
- Mixing latency-sensitive and batch operation is left open: BatchLLM says it is "less suitable for latency-critical workloads" and "We leave the implementation of such a hybrid system for future work." — [BatchLLM](https://arxiv.org/html/2412.03594v3); FlexLLM lists "developing adaptive strategies for workloads with consistently high inference demand" and says co-serving under strict SLOs "introduces service time variance that can affect tail latencies" — [FlexLLM](https://arxiv.org/html/2402.18789v3)
- Fine-tuning path in co-serving systems is less developed than the inference path: Loquetier plans "a backward propagation kernel operating in concert with the SMLM kernel to accelerate fine-tuning" — [Loquetier PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf); FlexLLM lists "extending our implementation beyond offline finetuning scenarios where sequences are processed individually (batch size 1)" — [FlexLLM](https://arxiv.org/html/2402.18789v3)
- PEFT coverage beyond plain LoRA: Loquetier "consider[s] providing support for other LoRA-like PEFT methods" — [Loquetier PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5ebc22174f6dff6fa7c2447d0dfa31ea-Paper-Conference.pdf); FlexLLM supports only base models and LoRA-based PEFT — [FlexLLM](https://arxiv.org/html/2402.18789v3); LoRAFusion describes DoRA/VeRA/QLoRA as extensions and notes fused dequantization is not always faster — [LoRAFusion](https://arxiv.org/html/2510.00206v1)
- Tensor parallelism for LoRA training: "Tensor parallelism is not evaluated due to the lack of efficient support in existing LoRA frameworks." — [LoRAFusion](https://arxiv.org/html/2510.00206v1)
- Gains shrink with larger batch sizes or larger models: PLoRA, "the performance gain diminishes since the Min GPU strategy can better utilize the hardware" — [PLoRA](https://arxiv.org/html/2508.02932); mLoRA, "the 13B model benefits less due to much higher computational overhead" — [mLoRA](https://arxiv.org/html/2312.02515); BlendServe, benefits "tend to shrink with smaller compute density" — [BlendServe](https://arxiv.org/html/2411.16102v2)
- Data heterogeneity assumptions in multi-tenant LoRA: LobRA states "In this work, we assume padding is employed." and that "some tasks would even experience an efficiency drop" under one of its configurations — [LobRA](https://arxiv.org/html/2509.01193v1); LoRAFusion reports a pipeline imbalance case that "is not solvable by our scheduler and is out of scope for this work." — [LoRAFusion](https://arxiv.org/html/2510.00206v1)
- Offload-based training efficiency at the large end: "LoHan maintains a relatively small 53% of peak FLOPS when fine-tuning a 175B model." — [LoHan](https://arxiv.org/html/2403.06504v2); SSDTrain leaves remote-SSD clusters to future work ("The CPU offloader is for future work on clusters with massive remote SSD storage.") — [SSDTrain](https://arxiv.org/html/2408.10013); ProTrain limits itself to data-parallel training with ZeRO and manual choice of parallelism configuration — [ProTrain](https://arxiv.org/html/2406.08334)
- Scope limited to one model family or task: BatchGen, "Our current implementation targets transformer MoE models" and "Extending BatchGen to other models and tasks is future work."; gains "plateau beyond 64 GPUs" — [BatchGen](https://arxiv.org/html/2606.21712)
- Checkpointing leaves non-model state and I/O as bottlenecks: ByteCheckpoint, "We leave the design of a more efficient dataloader to our future works." — [ByteCheckpoint](https://arxiv.org/html/2407.20143v4)
- Lack of standard workloads: the relational-analytics paper built its own benchmark "Due to the lack of standard workloads in this area" — [Optimizing LLM Queries](https://arxiv.org/html/2403.05821); BatchLLM evaluates on synthetic and private industry workloads with no public dataset — [BatchLLM](https://arxiv.org/html/2412.03594v3); BlendServe synthesizes its four mixed traces from public single-modality traces — [BlendServe](https://arxiv.org/html/2411.16102v2)
- Evaluation subsampling under distribution shift: "The main limitations of the methods described in this paper are related to potential severe distribution shifts."; the authors recommend "periodically updating the curated examples and IRT parameter estimates using data from more modern LLMs." — [tinyBenchmarks](https://arxiv.org/html/2402.14992)

### Inferences
- Items named by two or more catalogued papers: (a) hybrid/mixed online-offline or inference-finetuning operation under tight latency targets (BatchLLM, FlexLLM); (b) fine-tuning-side kernels and batching in unified runtimes (Loquetier, FlexLLM); (c) PEFT variants beyond LoRA (Loquetier, FlexLLM, LoRAFusion); (d) diminishing returns with batch size / model size (PLoRA, mLoRA, BlendServe); (e) absence of shared public offline workloads (relational analytics, BatchLLM, BlendServe); (f) offload efficiency bounded by host memory or storage (LoHan, SSDTrain, BatchGen's 1 TB-host-memory setting).
- Items named by a single paper only: tensor parallelism for LoRA (LoRAFusion), dataloader state in checkpoints (ByteCheckpoint), distribution shift in eval subsampling (tinyBenchmarks), output-length prediction (BlendServe).
- These are the authors' own statements at publication time; whether later work has closed them was not checked beyond the follow-ups listed in the per-paper records.

### Gaps
- Papers without a limitations section (mLoRA, LobRA, PLoRA, Jenga, STAlloc, GoCkpt, ConServe) contribute fewer open-problem statements, so this list over-represents papers that wrote one.
- PCcheck's and the curated-list-only papers' (MoEvement, TrainMover, MegaScale-Data, SuperOffload, BOOST, etc.) stated open problems were not read.
- No survey or workshop report that aggregates open problems for training/fine-tuning systems was located; the list above is assembled paper by paper.
