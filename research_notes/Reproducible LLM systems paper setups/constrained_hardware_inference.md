# LLM inference under tight hardware limits (single GPU, consumer/workstation GPUs, CPU+GPU, edge): problems and evaluation setups, 2024-2026

Notes on conventions used below. "Venue confirmed" means the title was found on the venue's own program, accepted-paper or proceedings page in this research session. "Venue from paper header / secondary" means only the paper's own header, a DOI, or a bibliographic aggregator was seen. "Reproducible on the student's node" refers to one node with about 4 idle 48 GB GPUs (RTX A6000 / RTX 6000 Ada, PCIe) plus host CPU/RAM; that verdict is my reading of the hardware each paper lists and is repeated under Inferences, not a claim made by the papers.

## Q1. Which top-venue papers (2024-2026) address LLM inference under tight hardware limits?

### Takeaway
Twenty-three peer-reviewed papers were catalogued across SOSP, OSDI, ASPLOS, ISCA, HPCA, ATC, EuroSys, MLSys, MobiSys, MobiCom, NeurIPS, ICML and ICLR; 18 had their venue confirmed on the venue's own page and 5 only from the paper header or a secondary listing. They fall into five groups: GPU-CPU hybrid/offloading on one GPU, multi-model sharing of GPUs, mobile/edge runtimes, energy, and benchmarks.

### Cited Findings

Group A: model larger than GPU memory, CPU+GPU hybrid on a single GPU
- PowerInfer, "Fast Large Language Model Serving with a Consumer-grade GPU", SOSP 2024, venue confirmed — [SOSP'24 accepted papers](https://sigops.org/s/conferences/sosp/2024/accepted.html)
- Kairox, "Adaptive GPU-CPU Hybrid LLM Inference via Online Neuron Balancing", OSDI 2026, venue confirmed — [USENIX OSDI'26](https://www.usenix.org/conference/osdi26/presentation/jiang-yapeng)
- DynamicInfer, "Runtime-Aware Sparse Offloading for LLMs Inference on a Consumer-Grade GPU", ICLR 2026, venue confirmed — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b9762d1f15c68057275fa18d384c1a8-Abstract-Conference.html)
- Fiddler, "CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models", ICLR 2025, venue confirmed — [ICLR 2025 papers](https://iclr.cc/virtual/2025/papers.html)
- MoE-Lightning, "High-Throughput MoE Inference on Memory-constrained GPUs", ASPLOS 2025 Vol. 1, pp. 715-730; venue from a secondary bibliographic listing only (the ASPLOS program page did not yield the title in this session) — [researchr listing](https://researchr.org/publication/CaoLGSL0GZS25)
- NEO, "Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference", MLSys 2025, venue confirmed — [MLSys 2025 papers](https://mlsys.org/virtual/2025/papers.html)
- FlexInfer, "Flexible LLM Inference with CPU Computations", MLSys 2025, venue confirmed — [MLSys 2025 papers](https://mlsys.org/virtual/2025/papers.html)
- LIA, "A Single-GPU LLM Inference Acceleration with Cooperative AMX-Enabled CPU-GPU Computation and CXL Offloading", ISCA 2025, venue confirmed — [ISCA 2025 program](https://iscaconf.org/isca2025/program/). A co-author's publication page lists the same work at ISCA 2025 under a different title, "LIA: Cost-efficient LLM Inference Acceleration with Intel Advanced Matrix Extensions and CXL" — [Google DeepMind publication page](https://deepmind.google/research/publications/81986/)
- SpecExec, "Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices", NeurIPS 2024, venue confirmed — [NeurIPS 2024 papers](https://neurips.cc/virtual/2024/papers.html)

Group B: several models or tenants sharing GPUs
- Nixie, "Efficient, Transparent Temporal Multiplexing for Consumer GPUs", OSDI 2026, venue confirmed — [USENIX OSDI'26](https://www.usenix.org/conference/osdi26/presentation/xu-yechen)
- MuxServe, "Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving", ICML 2024, venue confirmed — [PMLR v235](https://proceedings.mlr.press/v235/)
- Aegaeon, "Effective GPU Pooling for Concurrent LLM Serving on the Market", SOSP 2025, venue confirmed — [SOSP'25 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html)
- Prism, "Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning", OSDI 2026, venue confirmed — [USENIX OSDI'26](https://www.usenix.org/conference/osdi26/presentation/yu-shan)
- WarmServe, "Enabling One-for-Many GPU Prewarming for Multi-LLM Serving", ICML 2026, venue confirmed — [ICML 2026 poster page](https://icml.cc/virtual/2026/poster/65453)
- KunServe, "Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving", EuroSys 2026 per the paper's own header (DOI 10.1145/3767295.3769348); the EuroSys 2026 accepted-papers page did not yield the title in this session, so venue is from the paper header only — [arXiv 2412.18169](https://arxiv.org/html/2412.18169)

Group C: mobile, on-device and edge
- llm.npu, "Fast On-device LLM Inference with NPUs", ASPLOS 2025 Vol. 1 per the paper's own header (DOI 10.1145/3669940.3707239); venue from paper header only — [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2)
- HeteroInfer, "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference", SOSP 2025, venue confirmed — [SOSP'25 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html)
- ElastiLM, "Elastic On-Device LLM Service", MobiCom 2025, venue confirmed — [MobiCom 2025 accepted papers](https://www.sigmobile.org/mobicom/2025/accepted.html)
- EdgeLoRA, "An Efficient Multi-Tenant LLM Serving System on Edge Devices", MobiSys 2025, venue confirmed — [MobiSys 2025 accepted papers](https://www.sigmobile.org/mobisys/2025/accepted_papers/)
- CLONE, "Customizing LLMs for Efficient Latency-Aware Inference at the Edge", USENIX ATC 2025, venue confirmed — [ATC'25 technical sessions](https://www.usenix.org/conference/atc25/technical-sessions)

Group D: energy and power
- throttLL'eM, "Predictive GPU Throttling for Energy Efficient LLM Inference Serving", HPCA 2025 according to an aggregator surfaced by search; the HPCA 2025 program page could not be downloaded, so venue is from a secondary source only. The first arXiv version carried a different title, "SLO-aware GPU Frequency Scaling for Energy Efficient LLM Inference Serving" — [arXiv 2408.05235v2](https://arxiv.org/html/2408.05235v2)
- The ML.ENERGY Benchmark, "Toward Automated Inference Energy Measurement and Optimization", NeurIPS 2025, title confirmed on the venue page (which track was not confirmed) — [NeurIPS 2025 papers](https://neurips.cc/virtual/2025/papers.html)

Group E: benchmark and characterization of local engines
- MELT, "MELTing point: Mobile Evaluation of Language Transformers", MobiCom 2024: the arXiv record's comment field states "Accepted at the 30th Annual International Conference On Mobile Computing And Networking (MobiCom 2024)" — [arXiv 2403.12844](https://arxiv.org/abs/2403.12844v3); a bibliographic listing gives pp. 890-907 of the MobiCom 2024 proceedings — [researchr listing](https://researchr.org/publication/LaskaridisKMH24). The MobiCom 2024 accepted page did not yield the title in this session.

Title-level leads, not read (venue pages only)
- The MobiSys 2026 accepted list includes "Act Before It's Too Late: Power-Efficient LLM Inference on Mobile Device" and "KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference" — [MobiSys 2026 accepted papers](https://sigmobile.org/mobisys/2026/accepted_papers/)
- An MLSys 2026 paper on DVFS for mobile LLM inference (CORE) is in the MLSys 2026 proceedings — [MLSys 2026 proceedings](https://proceedings.mlsys.org/paper_files/paper/2026/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)
- The ATC'25 program also lists "Weaver: Efficient Multi-LLM Serving with Attention Offloading" and "Resource Multiplexing in Tuning and Serving Large Language Models" — [ATC'25 technical sessions](https://www.usenix.org/conference/atc25/technical-sessions)

### Inferences
- By venue year: 2024 has 4 papers (PowerInfer, SpecExec, MuxServe, MELT), 2025 has 13, 2026 has 6 (Kairox, DynamicInfer, Nixie, Prism, WarmServe, KunServe).
- The papers evaluated on consumer or workstation GPUs of the student's class (RTX 2080 Ti to RTX 5090, RTX 6000 Ada, RTX A5000) are PowerInfer, Kairox, DynamicInfer, Fiddler, SpecExec and Nixie. The multi-model sharing papers other than Nixie (MuxServe, Aegaeon, Prism, WarmServe, KunServe) use 8-GPU datacenter nodes with 80 GB cards.
- Aegaeon, Prism, WarmServe, KunServe and MuxServe sit at the border of the "datacenter cluster scheduling" area assigned to another researcher; they are kept here because their mechanism is per-GPU memory sharing.

### Gaps
- Venue could not be confirmed on the venue's own page for MoE-Lightning and llm.npu (ASPLOS 2025), MELT (MobiCom 2024), throttLL'eM (HPCA 2025) and KunServe (EuroSys 2026); the pages were unreachable or did not contain the titles in the form fetched.
- No peer-reviewed top-venue paper was verified for two parts of the scope: compiler-level optimization for commodity GPUs, and a benchmark of llama.cpp / MLC-LLM / ExecuTorch / vLLM on a single workstation GPU. The only engine benchmark verified is MELT, which targets phones and Jetson boards. Benchmarks seen on desktop-class hardware (Bench360, arXiv 2511.16682; an Apple Silicon engine comparison, arXiv 2511.05502) are arXiv-only and were not read.
- SC, PPoPP, CGO, MICRO, NSDI and SenSys programs were not searched systematically; absence from this list does not mean those venues have no relevant papers.
- The MobiSys 2026 and MLSys 2026 leads above were not read beyond their titles.

## Q2. For each paper: problem, evaluation setup, reproducibility on one node of 48 GB workstation GPUs, stated limitations, and follow-ups

### Takeaway
Six papers (PowerInfer, Kairox, DynamicInfer, Fiddler, SpecExec, Nixie) were evaluated on single consumer or workstation GPUs with batch-1 or few-request workloads and publish enough hardware, model and prompt detail to be re-run on a 48 GB workstation GPU node; the multi-model sharing papers use 16-32 datacenter GPUs and the mobile papers need Snapdragon or Jetson devices. The most common shared evaluation ingredients are llama.cpp as baseline, ShareGPT / Alpaca / Azure-trace prompts, and tokens/s, TTFT and TPOT as metrics.

### Cited Findings

#### Group A: single GPU, model larger than GPU memory (CPU+GPU hybrid)

##### A1. PowerInfer (SOSP 2024)
- Links: paper [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2); venue [SOSP'24 accepted papers](https://sigops.org/s/conferences/sosp/2024/accepted.html). Code: the paper says the system is open sourced; the repository URL was not in the text fetched (commonly cited as github.com/SJTU-IPADS/PowerInfer, unverified in this session).
- Problem: a single user running an LLM locally on a PC with one consumer GPU; the model does not fit in GPU memory, and layer-wise CPU offload (llama.cpp style) is slow because of PCIe transfers and the weak CPU. The paper exploits power-law neuron activation: hot neurons preloaded on GPU, cold neurons computed on CPU, chosen by per-layer online predictors — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Hardware: PC-High = Intel i9-13900K, 192 GB RAM, one RTX 4090 24 GB, PCIe 4.0; PC-Low = Intel i7-12700K, 64 GB RAM, one RTX 2080 Ti 11 GB, PCIe 3.0 — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Models (FP16 and INT4): OPT-7B/13B/30B/66B/175B, Bamboo-7B, Falcon(ReLU)-40B, LLaMA2(ReGLU)-7B/13B/70B, Qwen1.5-4B, LLaMA2(SwiGLU)-13B, Yi-34B — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Workloads: prompts from Chatbot Arena, ChatGPT prompts and Alpaca, 8-128 token prompts, batch size 1; accuracy checked on PIQA, Winogrande, ARC-Challenge, MMLU, GSM8K — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Framework and baselines: built by extending llama.cpp (about 4,200 lines C++/CUDA) plus 400 lines Python on transformers; baselines are llama.cpp and SpecInfer, with vLLM on an A100-80GB as an upper reference — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Headline: metric is generation speed in tokens/s. FP16 on PC-High averages 8.32 tokens/s (peak 16.06), 7.23x over llama.cpp on average and up to 11.69x (Falcon-40B); INT4 on PC-High averages 13.20 tokens/s (2.89x); PC-Low INT4 averages 5.01x; OPT-30B on the RTX 4090 is only 18% slower than on an A100 — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Limitations stated by the authors: "The effectiveness of PowerInfer is directly correlated with the model's activation sparsity."; gains are limited for long inputs with short outputs, where "the CPU becomes the primary bottleneck"; speedup falls as batch size grows because joint sparsity drops; SiLU-family models gain only 1.47-1.7x; predictors "may occasionally miss some active neurons"; predictor training is "time-consuming (often several hours)"; and "integrating speculative inference into PowerInfer could further boost LLM inference speed." — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Follow-ups: Kairox (OSDI 2026) and DynamicInfer (ICLR 2026) both use PowerInfer as a baseline and report beating it (see A2, A3). Kairox additionally compares against Neuralink and Q-Infer, two other sparse hybrid systems — [Kairox, OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf). PowerInfer-2 (mobile) appears as a baseline in llm.npu and HeteroInfer — [arXiv 2501.14794](https://arxiv.org/html/2501.14794). arXiv-only follow-ups found by search and not read: ATSInfer [arXiv 2607.10183](https://arxiv.org/abs/2607.10183), EPIC [arXiv 2504.03664](https://arxiv.org/abs/2504.03664).

##### A2. Kairox (OSDI 2026)
- Links: [USENIX page](https://www.usenix.org/conference/osdi26/presentation/jiang-yapeng), [PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf). Code: [github.com/Anhelor/kairox.cpp](https://github.com/Anhelor/kairox.cpp) (artifact appendix; Docker, CUDA 12.8; model GGUFs from the authors' model release).
- Problem: same user as PowerInfer (LLM larger than GPU memory on a consumer PC). Static, offline hot/cold neuron placement cannot follow runtime changes in which neurons activate, leaving GPU and CPU unbalanced; Kairox rebalances neurons between GPU and CPU online — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Hardware: PC-Low = RTX 3080 Ti 12 GB, Xeon E5-2680 v4 capped at 12 threads, 64 GB RAM, PCIe 3.0; PC-High = RTX 4090 24 GB, AMD EPYC 7542 capped at 16 threads, 128 GB RAM, PCIe 4.0 — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Models: OPT-6.7B/13B, OPT-30B (Q8 and Q4), OPT-66B-Q4, ProSparse-LLaMA2-7B/13B, Bamboo-7B, SparseQwen2-7B, ReLU-Falcon-40B-Q8, plus non-ReLU models in a separate table — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Workload: a fixed subset of ShareGPT prompts, batch size 1, maximum output 512 tokens, maximum context 1024; predictors trained on 400,000 activation patterns from C4; accuracy on PIQA, Winogrande, ARC, MMLU, GSM8K (under 0.5% average degradation) — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Framework and baselines: llama.cpp plus about 5,700 lines C++/CUDA; baselines llama.cpp, PowerInfer (ported to the same backend), Neuralink (reimplemented), Q-Infer — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Headline: tokens/s; up to 7.57x / 3.70x / 6.35x / 3.76x over llama.cpp / PowerInfer / Neuralink / Q-Infer; geometric-mean 3.15x and 3.93x over llama.cpp on the two PCs — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Authors' statements on limits: results on non-ReLU models "highlight potential opportunities for future research into inference optimization for less sparse or non-sparse model architectures on edge devices"; the artifact appendix warns "Different CPUs, memory bandwidth, PCIe generations, drivers, or model subsets may change absolute throughput", and the PowerInfer and Q-Infer comparisons are outside the required artifact path — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Follow-ups: none found (published 2026).

##### A3. DynamicInfer (ICLR 2026)
- Links: [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b9762d1f15c68057275fa18d384c1a8-Abstract-Conference.html), [PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf), [OpenReview](https://openreview.net/forum?id=CvjmvjlczZ). Code: no code link found in the paper text.
- Problem: sparse CPU-GPU offloading on a consumer GPU where the offloading decision is fixed offline; DynamicInfer makes the neuron offloading decision at runtime — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf)
- Hardware: PC-High = RTX 4090 24 GB, Intel Xeon Platinum 8358P, PCIe 4.0, 450 GB RAM; PC-Low = RTX 2080 Ti, AMD Ryzen 9 5950X, PCIe 3.0, 120 GB RAM. The paper gives the 2080 Ti memory as 12 GB in the setup text and 11 GB elsewhere (internal inconsistency in the source) — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf)
- Models: ReluLLaMA and ProSparse variants, 7B/13B on PC-Low and 13B/70B on PC-High (70B in INT4, others FP16); SiLU Llama-2 in the appendix — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf)
- Workload: 100 samples from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), default 64 input / 128 output tokens, batch size 1, metric tokens/s — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf)
- Baselines and headline: Transformers+Accelerate, llama.cpp, PowerInfer; up to 253% faster than llama.cpp and 59% faster than PowerInfer; on PC-High 13B is 18-26% and 70B is 53-59% over PowerInfer; on PC-Low 7B is 20-35% and 13B is 11-13%, which the authors attribute "mainly due to the limited 11GB GPU memory" — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf)
- Limitations stated: the profiling "assumes a dedicated execution environment. In scenarios with resource contention from concurrent applications, a more sophisticated online measurement mechanism would be beneficial, which represents a direction for future work."; CPU memory use is higher than PowerInfer (ReluLlama-13B on the 2080 Ti: 36.6 GB vs 21.3 GB) — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf)
- Follow-ups: none found (published 2026).

##### A4. Fiddler (ICLR 2025)
- Links: paper [arXiv 2402.07033](https://arxiv.org/html/2402.07033); code [github.com/efeslab/fiddler](https://github.com/efeslab/fiddler); venue [ICLR 2025 papers](https://iclr.cc/virtual/2025/papers.html).
- Problem: individuals or groups without GPU clusters who want to run a large Mixture-of-Experts model (Mixtral-8x7B at 16-bit needs more than 90 GB) on one GPU; existing offloading moves weights over PCIe every time, so Fiddler instead runs cold experts on the CPU and moves only activations — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- Hardware: Environment 1 = Quadro RTX 6000 24 GB, PCIe Gen3, Xeon Gold 6126 (48 cores); Environment 2 = RTX 6000 Ada 48 GB, PCIe Gen4, Xeon Platinum 8480+ (112 cores). Host RAM size is not stated in the text read — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- Models and placement: Mixtral-8x7B 16-bit, with 56 of 256 experts on GPU in Environment 1 and 125 of 256 in Environment 2; Phi-3.5-MoE in the appendix — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- Workloads: ShareGPT prompts (LMSYS-Chat-1M for sensitivity); three scenarios: single request (input 32-256, output 64-512 tokens; tokens/s), long prefill (512-4096 input tokens; TTFT), beam search (width 4-16) — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- Framework and baselines: PyTorch plus a custom AVX512_BF16 CPU expert kernel; baselines DeepSpeed-MII v0.2.3, Mixtral-Offloading, llama.cpp (build b2956) — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- Headline: versus llama.cpp, 1.26x faster on average for single-batch, 1.30x for long prefill, 11.57x for beam search — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- Limitations stated: non-expert layers are assumed to fit on the GPU; expert placement is popularity-based ("We assume this method is enough"); evaluation covers only "a small number of concurrent queries"; exploiting sparsity "remains an intriguing direction for future research" — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- Follow-ups: MoE-Lightning (ASPLOS 2025, see A5). arXiv-only follow-ups found by search and not read: FreeToken [arXiv 2608.16157](https://arxiv.org/abs/2608.16157), CoX-MoE [arXiv 2605.17889](https://arxiv.org/abs/2605.17889), TriMoE [arXiv 2603.01058](https://arxiv.org/abs/2603.01058).

##### A5. MoE-Lightning (ASPLOS 2025; venue from secondary listing)
- Links: paper [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1); venue listing [researchr](https://researchr.org/publication/CaoLGSL0GZS25). Code: no code URL in the paper text read.
- Problem: batch (offline, throughput-oriented) inference of MoE models on GPUs whose memory is far smaller than the model; existing offloading systems overlap CPU, GPU and I/O poorly. The paper adds a CPU-GPU-I/O pipelining schedule (CGOPipe) and a Hierarchical Roofline performance model (HRM) to pick policies — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- Hardware: 1x T4 16 GB (24-core Xeon, 192 GB RAM); 1x L4 24 GB; 2x and 4x T4 (32-core Xeon, 416 GB RAM) — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- Models and workloads: Mixtral 8x7B, Mixtral 8x22B, DBRX; MTBench, HELM Synthetic Reasoning, HELM Summarization — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- Baselines and headline: FlexGen, FlexGen with CPU attention, DeepSpeed Zero-Inference; metric is end-to-end throughput (tokens/s); up to 10.3x over FlexGen for Mixtral 8x7B on one T4 without request padding and 3.5x with padding — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- Limitations stated: HRM "is limited to hardware within a single node"; the work "focuses on scenarios where GPU memory is limited but sufficient CPU memory is available"; "when CPU memory is insufficient to hold the entire model, disk offloading becomes essential" (not covered); "the CPU attention will eventually become the bottleneck, which calls for higher CPU memory bandwidth" — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- Follow-ups: arXiv-only MoE offloading papers listed under A4 (not read).

##### A6. NEO (MLSys 2025)
- Links: paper [arXiv 2411.01142](https://arxiv.org/html/2411.01142); venue [MLSys 2025 papers](https://mlsys.org/virtual/2025/papers.html). Code: the paper says "We will open source Neo codebase"; a released repository was not located.
- Problem: online serving operators whose batch size is capped by GPU memory (KV cache). NEO offloads part of decoding attention and KV cache to the host CPU so the GPU can batch more requests — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)
- Hardware: AWS g5.2xlarge to g5.16xlarge (one A10G 24 GB, EPYC 7R32), g4dn.4xlarge (one T4 16 GB, Xeon Platinum 8259CL 8 cores, 64 GB RAM), and an 8x H100 server (Xeon 8462Y+, 2 TB RAM) — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)
- Models and workloads: LLaMa-3.1-8B, LLaMa-3.1-70B, LLaMa-2-7B; Azure LLM inference trace (code), OpenAI summarization comparison dataset, synthetic workloads — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)
- Framework and baselines: built on SwiftLLM; baselines vLLM, FastDecode+, SwiftLLM (GPU-only) — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)
- Headline: throughput under latency limits; up to 7.5x on T4, 26% on A10G, 14% on H100 over GPU-only. The fetched text gave two different T4 figures in different sections (563% and 750%); source inconsistency noted, not resolved — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)
- Limitations stated: "We leave as future work integrating Neo into vLLM to measure our 4-GPU and 8-GPU performance gains"; scheduling is affected by the "inevitable inaccuracy of the offline performance profiling"; integration with chunked prefill is left for future work; with enough GPU memory NEO falls back to no offloading — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)
- Follow-ups: none verified.

##### A7. FlexInfer (MLSys 2025)
- Links: paper [author-hosted PDF](https://seonjinna.github.io/assets/pdf/MLSys25_FlexInfer.pdf); venue [MLSys 2025 papers](https://mlsys.org/virtual/2025/papers.html). Code: no link found.
- Problem: offloading-based inference on single-GPU servers with large CPU memory is bottlenecked by PCIe weight transfers; FlexInfer chooses per phase (prefill vs decode) whether to compute on CPU or GPU — [FlexInfer PDF](https://seonjinna.github.io/assets/pdf/MLSys25_FlexInfer.pdf)
- Hardware: Server 1 = Ice Lake Xeon (no AMX) + A100 40 GB, PCIe 4.0; Server 2 = Sapphire Rapids Xeon (AMX) + H100 80 GB, PCIe 5.0 — [FlexInfer PDF](https://seonjinna.github.io/assets/pdf/MLSys25_FlexInfer.pdf)
- Models and workload: OPT-30B, OPT-66B, LLaMA-2-70B, LLaMA-3-70B; input 512 tokens, output 32 tokens, batch 1-32 — [FlexInfer PDF](https://seonjinna.github.io/assets/pdf/MLSys25_FlexInfer.pdf)
- Framework, baselines, headline: CPU kernels from Intel Extension for PyTorch; baselines CPU-only, FlexGen, an optimized FlexGen, SplitGen; metrics end-to-end latency, TTFT, TPOT; 75% and 76% average end-to-end latency reduction versus FlexGen on the two servers — [FlexInfer PDF](https://seonjinna.github.io/assets/pdf/MLSys25_FlexInfer.pdf)
- Limitations: not extracted (see Gaps).

##### A8. LIA (ISCA 2025)
- Links: venue [ISCA 2025 program](https://iscaconf.org/isca2025/program/); abstract [Google DeepMind publication page](https://deepmind.google/research/publications/81986/); code mirror [25ISCA-LIA_AMXGPU](https://github.com/MachineLearningSystem/25ISCA-LIA_AMXGPU) (README: requires a 4th-generation-or-newer Xeon with AMX, an A100 or H100, Docker; scripts reproduce Sapphire Rapids + A100 results on OPT-30B and OPT-175B with dummy weights, baseline IPEX).
- Problem: single-GPU LLM inference where the alternatives are costly multi-GPU setups or slow CPU-GPU transfers over PCIe; the authors measure AMX matrix-multiply throughput of 20 TFLOPS (Sapphire Rapids) and 40 TFLOPS (Granite Rapids) and offload more computation to the CPU, plus CXL memory for throughput workloads — [DeepMind publication page](https://deepmind.google/research/publications/81986/)
- Headline (abstract level): with one H100, up to 5.1x lower latency and 3.7x higher throughput on Sapphire Rapids, and up to 19x lower latency and 5.1x higher throughput on Granite Rapids, versus the latest single-GPU offloading framework; CXL offloading adds 1.5x throughput and raises maximum batch size from 900 to 1.6K — [DeepMind publication page](https://deepmind.google/research/publications/81986/)
- Full evaluation tables and limitations: not obtained (see Gaps).

##### A9. SpecExec (NeurIPS 2024)
- Links: paper [arXiv 2406.02532](https://arxiv.org/html/2406.02532); venue [NeurIPS 2024 papers](https://neurips.cc/virtual/2024/papers.html). Code: github.com/yandex-research/specexec per the paper.
- Problem: interactive use of 50B+ models on consumer GPUs with RAM/SSD offloading, where each target-model forward pass is very slow; SpecExec uses a small draft model to build a large speculative token tree so each slow target pass verifies many tokens — [arXiv 2406.02532](https://arxiv.org/html/2406.02532)
- Hardware: A100 80 GB used to emulate offloading, plus RTX 4090, RTX 4060, RTX 3090, RTX 2080 Ti; at least 140 GB host RAM for the 16-bit 70B model — [arXiv 2406.02532](https://arxiv.org/html/2406.02532)
- Models and datasets: targets Llama 2-70B (16-bit and GPTQ 4-bit), Mixtral 8x7B, Llama 3-70B; drafts Llama 2-7B, TinyLlama-1.1B and others; OASST1, MTBench, C4, WikiText-2 — [arXiv 2406.02532](https://arxiv.org/html/2406.02532)
- Baselines and headline: SpecInfer, Sequoia, sequential decoding; 10-18x speedup over sequential offloaded inference; 4-6 tokens/s with 4-bit and 2-3 tokens/s with 16-bit 70B targets; 5.66 tokens/s on the RTX 4090 — [arXiv 2406.02532](https://arxiv.org/html/2406.02532)
- Limitations stated: "the goal of our study was not to create the fastest possible implementation of local LLM inference"; lower per-token acceptance rate than other methods; inference "may become increasingly more constrained by the memory bandwidth" — [arXiv 2406.02532](https://arxiv.org/html/2406.02532)
- Follow-ups: none verified.

#### Group B: several models or tenants sharing GPUs

##### B1. Nixie (OSDI 2026)
- Links: [USENIX page](https://www.usenix.org/conference/osdi26/presentation/xu-yechen), [PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf). Code: [github.com/XOR-op/Nixie](https://github.com/XOR-op/Nixie), evaluation scripts [github.com/XOR-op/nixie-eval](https://github.com/XOR-op/nixie-eval).
- Problem: a single user's consumer GPU runs several large ML applications (LLM chat, code completion, text-to-image) whose working sets each nearly fill VRAM; relying on NVIDIA unified memory (UVM) thrashes and wastes pinned host memory. Nixie time-multiplexes the GPU transparently (unmodified applications) — [Nixie PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf)
- Hardware: AMD Ryzen 9 9950X, 96 GB DDR5, two RTX 5090 32 GB on PCIe 5.0 x8 (one GPU used, the second only for a "2 GPUs" upper-bound baseline), Debian 12, CUDA 12.9; second testbed: two EPYC 7352, 1 TB DDR4, eight RTX A5000 24 GB on PCIe 4.0 — [Nixie PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf)
- Software and models: Ollama 0.12.11, SGLang 0.5.4.post1, llama.cpp b7027, ComfyUI; Qwen3 family (including Qwen3-MoE 30B-Q6; on the A5000 testbed Qwen3 14B-Q4, Qwen3-VL 32B-Q4, Qwen3-Coder 30B-Q4), Gemma3, Z-Image 6B, Qwen-Image 20B — [Nixie PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf)
- Workloads: four case studies (prompt-expanded image generation, multi-agent orchestration, concurrent interactive + long-running workloads), with the DreamBench++ dataset for image prompts — [Nixie PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf)
- Baselines and headline: Ollama, UVM, TGS, nvshare; up to 3.8x lower latency for interactive code completion co-located with a long-running LLM versus UVM; pinned host memory reduced 66.8%; 1.3-1.6x on multi-application workloads; on the A5000 testbed 3.4x faster than nvshare and 73% of two-GPU performance — [Nixie PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf)
- Limitations and future work stated: Linux-only prototype; "incorporating such semantic hints represents an interesting direction for future work" (for example dropping immutable weights instead of migrating them); policies are "intentionally conservative; refining them is a promising direction for future work, for example by learning scheduling decisions from recent execution history or by treating different GPU memory regions (e.g., model weights, activations, KV caches) differently"; "How to integrate spatial multiplexing into Nixie is a promising future direction"; single-user threat model. The artifact appendix notes the models were "targeted specifically for the 32 GB RTX 5090" and for other VRAM sizes suggests "models that fit into the GPU while maximizing VRAM usage" — [Nixie PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf)
- Follow-ups: none found (published 2026).

##### B2. MuxServe (ICML 2024)
- Links: paper [arXiv 2404.02015](https://arxiv.org/html/2404.02015); code [github.com/hao-ai-lab/MuxServe](https://github.com/hao-ai-lab/MuxServe); venue [PMLR v235](https://proceedings.mlr.press/v235/).
- Problem: providers serving many LLMs with different popularity waste GPUs when each model gets dedicated cards; MuxServe colocates models on shared GPUs with spatial (MPS) and temporal multiplexing and a placement algorithm — [arXiv 2404.02015](https://arxiv.org/html/2404.02015)
- Setup: 4 nodes x 8 A100 80 GB (32 GPUs); 19 LLaMA-family LLMs in size buckets; synthetic power-law request rates with ShareGPT prompts, plus a ChatLMSYS trace; built on vLLM with NVIDIA MPS; baselines spatial partitioning and temporal multiplexing — [arXiv 2404.02015](https://arxiv.org/html/2404.02015)
- Headline: up to 1.8x throughput, or 2.9x more requests within 99% SLO attainment — [arXiv 2404.02015](https://arxiv.org/html/2404.02015)
- Limitations stated: colocation interference raises P99 TPOT slightly; the placement algorithm is not optimal; it assumes known, slowly changing workloads and needs profiling — [arXiv 2404.02015](https://arxiv.org/html/2404.02015)
- Follow-ups (all use MuxServe as a baseline): Aegaeon reports MuxServe refuses to place more than two models per GPU — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf); Prism re-implements it as "MuxServe++" because the original supports only Llama-2 on an old vLLM — [arXiv 2505.04021v3](https://arxiv.org/html/2505.04021v3); WarmServe reimplements it on its own vLLM version — [arXiv 2512.09472](https://arxiv.org/abs/2512.09472).

##### B3. Aegaeon (SOSP 2025)
- Links: [PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf); venue [SOSP'25 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html); DOI [10.1145/3731569.3764815](https://doi.org/10.1145/3731569.3764815). Code: no public code link found in the paper.
- Problem: model marketplaces host many sporadically used models; existing GPU pooling fits at most two or three models per GPU. Aegaeon switches models at token granularity (token-level autoscaling) — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Setup: 2 nodes x 8 H800 80 GB (16 GPUs; 6 for prefill, 10 for decoding), 2 TB DDR5, Xeon Platinum 8469C; sensitivity runs on a 4x A10 24 GB node with 6-7B models and on 8x H800 with 72B models at tensor-parallel 4; Qwen, Llama, InternLM, Yi models of 6-14B; ShareGPT and length-scaled variants with Poisson arrivals; SLOs TTFT 10 s and TBT 100 ms; vLLM backend; baselines ServerlessLLM, ServerlessLLM+ (shortest-job-first oracle), MuxServe — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Headline: sustains 2-2.5x higher arrival rates or 1.5-9x goodput; autoscaling overhead cut 97%; in production 213 H20 GPUs replace 1,192 (82% saving) — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Authors' statements on limits: the token-level approach is "more effective with high-end GPUs"; on the A10 node model prefetching is disabled because 24 GB cannot hold two models' weights — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)

##### B4. Prism (OSDI 2026)
- Links: [USENIX page](https://www.usenix.org/conference/osdi26/presentation/yu-shan); paper [arXiv 2505.04021v3](https://arxiv.org/html/2505.04021v3). Code: balloon driver [github.com/ovg-project/kvcached](https://github.com/ovg-project/kvcached).
- Problem: multi-LLM serving with long-tail model popularity; static GPU partitions strand memory. Prism lets colocated models grow and shrink GPU memory on demand (ballooning via CUDA virtual memory) with cross-model placement — [arXiv 2505.04021v3](https://arxiv.org/html/2505.04021v3)
- Setup: 4 nodes x 8 H100 80 GB, two Xeon Platinum 8480+, 1.7 TB RAM, with some two-GPU experiments; 58 LLMs (43 of 1-3B, 8 of 4-8B, 3 of 9-30B, 4 of 31-70B); production traces from Hyperbolic and Arena-Chat; built on SGLang; baselines static partition, MuxServe++, QLM, ServerlessLLM — [arXiv 2505.04021v3](https://arxiv.org/html/2505.04021v3)
- Headline: up to 3.3x TTFT SLO attainment and 2x TPOT attainment, more than 2x cost reduction; on a two-GPU setup with the Hyperbolic trace, 2.3x / 3.5x more requests than MuxServe++ / static partition — [arXiv 2505.04021v3](https://arxiv.org/html/2505.04021v3)
- Authors' statements: "general GPU-sharing systems target application-agnostic workloads and cannot be directly compared"; eviction is "based on an empirical threshold" — [arXiv 2505.04021v3](https://arxiv.org/html/2505.04021v3)

##### B5. WarmServe (ICML 2026)
- Links: [ICML 2026 poster page](https://icml.cc/virtual/2026/poster/65453); paper [arXiv 2512.09472](https://arxiv.org/abs/2512.09472); code [github.com/LLMServe/WarmServe](https://github.com/LLMServe/WarmServe) (vLLM 0.6.3.post1, patched PyTorch 2.4.0, CUDA 12.1, Ray; trace generators for Azure, ServeGen and BurstGPT traces).
- Problem: multi-LLM serving where autoscaling suffers cold starts and GPU sharing suffers interference; WarmServe prewarms one GPU for many candidate models — [arXiv 2512.09472](https://arxiv.org/abs/2512.09472)
- Setup: two servers with eight GPUs each (NVLink, RDMA, PCIe 5.0; the GPU model is not named in the text read); Llama2 7B (duplicated), 13B, 70B; Azure conversation trace with power-law per-model rates; baselines ServerlessLLM (GPU) and MuxServe; BlitzScale is not compared because its "documentation is incomplete and we are unable to install it" — [arXiv 2512.09472](https://arxiv.org/abs/2512.09472)
- Headline: tail TTFT up to 50.8x lower than the autoscaling baseline and up to 2.5x higher throughput than the GPU-sharing baseline — [arXiv 2512.09472](https://arxiv.org/abs/2512.09472)
- Limitations: not extracted (see Gaps).

##### B6. KunServe (EuroSys 2026; venue from paper header)
- Links: paper [arXiv 2412.18169](https://arxiv.org/html/2412.18169); code [github.com/SJTU-IPADS/kunserve](https://github.com/SJTU-IPADS/kunserve).
- Problem: bursts overload GPU memory (KV cache) in serving, causing long queuing; KunServe frees memory by dropping replicated model parameters and running pipelined across instances — [arXiv 2412.18169](https://arxiv.org/html/2412.18169)
- Setup: Cluster A = 8 servers each with one A800 80 GB; Cluster B = 2 servers x 8 H800 80 GB; Qwen-2.5-14B and 72B; BurstGPT trace with ShareGPT and LongBench prompts; baselines vLLM v0.6.3, InferCept, Llumnix — [arXiv 2412.18169](https://arxiv.org/html/2412.18169)
- Headline: up to 72.2x tail TTFT reduction — [arXiv 2412.18169](https://arxiv.org/html/2412.18169)
- Limitations stated: "the memory that can be freed by dropping is limited"; "Our current implementation focuses on dense models"; lookahead batching "may add additional latency, which we leave possible solutions as a future work"; median TPOT increases — [arXiv 2412.18169](https://arxiv.org/html/2412.18169)

#### Group C: mobile, on-device and edge

##### C1. llm.npu (ASPLOS 2025; venue from paper header)
- Links: paper [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2); code [github.com/UbiquitousLearning/mllm](https://github.com/UbiquitousLearning/mllm); artifact [Zenodo 10.5281/zenodo.14392760](https://doi.org/10.5281/zenodo.14392760).
- Problem: on-device LLM prefill for long prompts is slow on phone CPU/GPU; the paper offloads prefill to the mobile NPU by restructuring prompt, model and scheduling for NPU constraints — [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2)
- Setup: Redmi K70 Pro (Snapdragon 8 Gen 3, 24 GB) and Redmi K60 Pro (Snapdragon 8 Gen 2, 16 GB); Qwen1.5-1.8B, Gemma-2B, Phi2-2.7B, LLaMA-2-7B, Mistral-7B; LongBench (2wikimqa, TriviaQA), DroidTask, Persona-Chat; baselines llama.cpp, TFLite, MNN, MLC-LLM, PowerInfer-v2 — [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2)
- Headline: 22.4x faster prefill and 30.7x energy savings on average; over 1,000 tokens/s prefill — [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2)
- Limitations stated: "llm.npu does not consider resource detection and contention in the current implementation"; it "currently supports only CPU-based decoding"; short prompts must be padded; GPU-NPU coordination is not implemented — [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2)
- Follow-ups: HeteroInfer (SOSP 2025) uses llm.npu as a baseline — [arXiv 2501.14794](https://arxiv.org/html/2501.14794)

##### C2. HeteroInfer (SOSP 2025)
- Links: paper [arXiv 2501.14794](https://arxiv.org/html/2501.14794); venue [SOSP'25 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html). Code: no link found.
- Problem: phone LLM engines use one processor at a time; the paper characterizes mobile GPU and NPU and partitions work across both at layer and tensor level — [arXiv 2501.14794](https://arxiv.org/html/2501.14794)
- Setup: Snapdragon 8 Gen 3 and 8 Elite devices; Llama-8B/7B/3B and InternLM-1.8B (W4A16); BELLE multi-turn chat, GSM8K, LongBench TriviaQA; baselines llama.cpp, MLC, MNN-OpenCL, llm.npu, PowerInfer-2 — [arXiv 2501.14794](https://arxiv.org/html/2501.14794)
- Headline: 1.34x-6.02x end-to-end speedup; 247.9 tokens/s prefill and 14.01 tokens/s decode for Llama-8B — [arXiv 2501.14794](https://arxiv.org/html/2501.14794)
- Authors' statements: "the absence of a unified scheduling mechanism may lead to situations where tasks running on one processor simultaneously block the execution of tasks on both processors"; "A unified API and memory management layer across CPU, GPU, and NPU would greatly simplify development"; the NPU lacks W4A16 support for decoding — [arXiv 2501.14794](https://arxiv.org/html/2501.14794)

##### C3. ElastiLM (MobiCom 2025)
- Links: paper [arXiv 2409.09071v2](https://arxiv.org/html/2409.09071v2); venue [MobiCom 2025 accepted papers](https://www.sigmobile.org/mobicom/2025/accepted.html). Code: no ElastiLM repository found (the on-device part uses mllm).
- Problem: an on-device LLM is a shared OS service whose callers have different latency SLOs; one fixed model cannot meet them. ElastiLM builds an elastic model (sub-models plus prompt compression) that switches per request — [arXiv 2409.09071v2](https://arxiv.org/html/2409.09071v2)
- Setup: offline stage on a server with 8x NVIDIA A40 (45 GB), 64-core CPU, 750 GB RAM; devices Redmi K60 Champion (Snapdragon 8 Gen 2), Mi 14 and Redmi K70 Pro (8 Gen 3); LLaMA-7B, Llama3-8B, Vicuna-7B, Llama3-instruct-8B, Orca-mini-3B; ARC-Easy, OBQA, Octopus, PIQA, SciQ, LlamaTouch and six synthesized app traces with SLOs; baselines include LLMPruner, LaCo, ShortGPT, LLMLingua2-based variants — [arXiv 2409.09071v2](https://arxiv.org/html/2409.09071v2)
- Headline: 6.60-14.83 points higher absolute accuracy at the same SLO; up to 5x TTFT and 2x TPOT; 0.31 s switching; offline cost 68.3 GPU hours — [arXiv 2409.09071v2](https://arxiv.org/html/2409.09071v2)
- Authors' statements: "none of mobile DSAs can seamlessly run prefill and decode of ElastiLM and all its baselines"; "Performance gains are minimal in some high-SLO scenarios" — [arXiv 2409.09071v2](https://arxiv.org/html/2409.09071v2)

##### C4. EdgeLoRA (MobiSys 2025)
- Links: paper [arXiv 2507.01438](https://arxiv.org/html/2507.01438); venue [MobiSys 2025 accepted papers](https://www.sigmobile.org/mobisys/2025/accepted_papers/). Code: URL not in the text fetched (the paper has an artifact appendix).
- Problem: an edge device serving many tenants, each with its own LoRA adapter, runs out of memory and thrashes swapping adapters; EdgeLoRA adds adaptive adapter selection, an adapter cache and batched multi-adapter inference — [arXiv 2507.01438](https://arxiv.org/html/2507.01438)
- Setup: Jetson AGX Orin, Jetson Orin Nano, Raspberry Pi 5 (router fine-tuned on two workstation GPUs the paper calls "RTX A6000 Ada"); Llama3.1-8B (Q8_0), Llama3.2-3B (Q4_0), OpenELM-1.1B; up to 2,000 adapters; synthetic Gamma arrivals with power-law adapter popularity; built on llama.cpp, baseline llama.cpp — [arXiv 2507.01438](https://arxiv.org/html/2507.01438)
- Headline: 2-4x throughput; first-token latency from 206.28 s to 0.51 s in one Orin Nano setting with 20 adapters — [arXiv 2507.01438](https://arxiv.org/html/2507.01438)

##### C5. CLONE (USENIX ATC 2025)
- Links: paper [arXiv 2506.02847](https://arxiv.org/html/2506.02847); venue [ATC'25 technical sessions](https://www.usenix.org/conference/atc25/technical-sessions). Code: no URL found.
- Problem: latency- and energy-aware LLM inference on edge devices via algorithm-hardware co-design (pruning, LoRA routing, DVFS, and a custom accelerator) — [arXiv 2506.02847](https://arxiv.org/html/2506.02847)
- Setup: Jetson Orin Nano 8 GB and Orin NX 16 GB; a 28 nm accelerator evaluated by post-layout simulation only; Llama-7B, Llama2-7B/13B, Vicuna-7B; WikiText2, PTB, BBH, MMLU, Flan v2; baselines include LLMPruner, ShortGPT, SliceGPT, FlexGen — [arXiv 2506.02847](https://arxiv.org/html/2506.02847)
- Headline: up to 11.92x speedup and 7.36x energy saving — [arXiv 2506.02847](https://arxiv.org/html/2506.02847)

#### Group D: energy and power

##### D1. throttLL'eM (HPCA 2025; venue from secondary source)
- Links: paper [arXiv 2408.05235v2](https://arxiv.org/html/2408.05235v2). Code: a repository surfaced by search, [github.com/WilliamBlaskowicz/throttLL-eM](https://github.com/WilliamBlaskowicz/throttLL-eM), not opened in this session (unverified).
- Problem: serving operators run GPUs at maximum frequency although load and SLO slack vary; throttLL'eM predicts KV-cache usage and batch size, then picks the lowest GPU frequency (plus instance autoscaling) that still meets SLOs — [arXiv 2408.05235v2](https://arxiv.org/html/2408.05235v2)
- Setup: AWS p4d.24xlarge with 8x A100 40 GB; Llama2-13B (tensor-parallel 1/2/4), Llama3-8B, Llama3-70B (tensor-parallel 8); a 60-minute Azure LLM inference trace and synthetic queries; Triton + TensorRT-LLM; baseline default Triton — [arXiv 2408.05235v1](https://arxiv.org/html/2408.05235v1)
- Headline: up to 43.8% lower energy and at least 1.71x energy efficiency under SLOs versus Triton — [arXiv 2408.05235v2](https://arxiv.org/html/2408.05235v2)
- Limitations stated: "throttLL'eM does not explicitly optimize the prefill phase" (frequency switching takes about 200 ms, longer than a roughly 175 ms prefill); "optimal GPU frequency is architecture- and LLM-dependent"; the trace has no prompt content — [arXiv 2408.05235v1](https://arxiv.org/html/2408.05235v1)
- Follow-ups: arXiv-only papers found by search and not read: GreenLLM [arXiv 2508.16449](https://arxiv.org/abs/2508.16449) and a phase-decoupled power control paper [arXiv 2609.11133](https://arxiv.org/abs/2609.11133).

##### D2. The ML.ENERGY Benchmark (NeurIPS 2025)
- Links: paper [arXiv 2505.06371](https://arxiv.org/html/2505.06371); code [github.com/ml-energy/benchmark](https://github.com/ml-energy/benchmark), [leaderboard code](https://github.com/ml-energy/leaderboard), [leaderboard](https://ml.energy/leaderboard), measurement library [Zeus](https://github.com/ml-energy/zeus); venue [NeurIPS 2025 papers](https://neurips.cc/virtual/2025/papers.html).
- Problem: no standard way to measure and compare inference energy of generative models under realistic serving configurations — [arXiv 2505.06371](https://arxiv.org/html/2505.06371)
- Setup: A100 40 GB and H100 80 GB on AWS p4d/p5; 40 model architectures over 6 tasks; request datasets ShareGPT, EvalPlus, LLaVA instruction data, PartiPrompts, ShareGPT4Video; vLLM and Diffusers — [arXiv 2505.06371](https://arxiv.org/html/2505.06371)
- Headline findings: estimates based on GPU TDP overestimate energy by up to 4.1x; automated configuration tuning cuts energy 44% at a 100 ms TPOT target — [arXiv 2505.06371](https://arxiv.org/html/2505.06371)
- Limitations stated: "We were not able to run the benchmark multiple times due to the high monetary cost of even a single run."; the authors call for "automated cross-layer energy optimizations" — [arXiv 2505.06371](https://arxiv.org/html/2505.06371)

#### Group E: benchmark and characterization

##### E1. MELT (MobiCom 2024)
- Links: paper [arXiv 2403.12844](https://arxiv.org/html/2403.12844); code [github.com/brave-experiments/MELT-public](https://github.com/brave-experiments/MELT-public).
- Problem: no systematic measurement of on-device LLM execution; MELT is an automation harness that runs LLMs headlessly on phones and Jetson boards and measures performance, energy and accuracy — [arXiv 2403.12844](https://arxiv.org/html/2403.12844)
- Setup: Galaxy S23, Pixel 6a, iPhone 14 Pro, iPhone SE, Jetson Orin AGX 64 GB, Jetson Orin Nano 8 GB; engines MLC-LLM, llama.cpp (b2202), LLMFarm; TinyLlama, Zephyr-3B, Mistral-7B, Gemma-2B/7B, Llama-2-7B/13B at 3-4 bit; 50 conversations from OpenAssistant oasst1; accuracy on HellaSwag, Winogrande, TruthfulQA, ARC — [arXiv 2403.12844](https://arxiv.org/html/2403.12844)
- Headline findings: generation is memory-bound; 0.16-0.21 mWh per token for Zephyr-3B; throughput degrades under continuous use (thermal) — [arXiv 2403.12844](https://arxiv.org/html/2403.12844)
- Limitations stated: "our analysis has been limited to chat fine-tuned models of 1-13B parameter size"; "we only studied quantization as a way of reducing model footprint"; energy was analysed "at a device-centric level"; the authors expect "NPU acceleration, and framework-hardware co-design to be the biggest bet" — [arXiv 2403.12844](https://arxiv.org/html/2403.12844)

### Inferences

Reproducibility on one node with about 4 idle 48 GB GPUs (RTX A6000 / RTX 6000 Ada, PCIe) plus host CPU/RAM. These verdicts are derived by comparing each paper's listed hardware with the student's node; none of the papers states them, and none was tested.

| Paper | Verdict | What changes / why |
|---|---|---|
| A1 PowerInfer | scaled down | Original GPUs are 24 GB and 11 GB. On a 48 GB card the smaller models fit entirely, so the "larger than GPU memory" condition needs either a GPU memory cap or only the largest models (OPT-66B/175B, LLaMA2-70B FP16, Falcon-40B). Needs up to 192 GB host RAM for the largest. Absolute tokens/s depend on CPU and PCIe generation. |
| A2 Kairox | scaled down | Same as PowerInfer (12 GB / 24 GB originals); public artifact with Docker; the artifact itself warns absolute throughput changes with CPU, memory bandwidth and PCIe. PowerInfer and Q-Infer comparisons are not in the required artifact path. |
| A3 DynamicInfer | scaled down | Same hardware class as A1; no code found, so the system would have to be re-implemented; baselines (llama.cpp, PowerInfer, Transformers) are public. |
| A4 Fiddler | yes | Environment 2 is one RTX 6000 Ada 48 GB, the same card class as the student's node; code is public. CPU must support AVX512_BF16 for the authors' CPU kernel; host RAM requirement not stated. |
| A5 MoE-Lightning | scaled down | Original GPUs are T4 16 GB / L4 24 GB; a 48 GB card needs a memory cap or the larger models (Mixtral 8x22B, DBRX) to stay memory-constrained. No code found. Needs 192-416 GB host RAM. |
| A6 NEO | scaled down | T4 / A10G experiments map to one capped GPU; the 8x H100 experiment does not fit. Code release not located; SwiftLLM base is public. |
| A7 FlexInfer | scaled down | Needs an AMX-capable Xeon for the Server 2 results and large host RAM; A100/H100 replaced by a 48 GB card. No code found. |
| A8 LIA | no, unless the host has an AMX Xeon | README requires a 4th-generation-or-newer Xeon with AMX and an A100/H100; the CXL part needs CXL memory hardware. |
| A9 SpecExec | yes | Evaluated on single consumer GPUs (RTX 4090, 3090, 2080 Ti, 4060) with offloading; code public; needs at least 140 GB host RAM for the 16-bit 70B target. |
| B1 Nixie | yes / scaled down | Public code and evaluation repo; evaluated on RTX 5090 32 GB and RTX A5000 24 GB. For a 48 GB card the artifact advises choosing models that fill VRAM, so model sizes change. How it is installed and what privileges it needs on a shared machine were not checked. |
| B2 MuxServe | scaled down | 32 A100-80GB in the paper; on 4x 48 GB GPUs fewer and smaller models. Code public but tied to an old vLLM and Llama-2 according to Prism. |
| B3 Aegaeon | scaled down | 16 H800 in the paper; the authors' own A10 24 GB sensitivity run (6-7B models, 4 GPUs) is the closest configuration. No public code found. |
| B4 Prism | scaled down | 32 H100 in the paper; the two-GPU experiments are the closest configuration; kvcached is public; traces (Hyperbolic, Arena-Chat) availability not checked. |
| B5 WarmServe | scaled down | 16 GPUs in the paper; code public; Llama2-70B at full precision would not fit one 48 GB card. |
| B6 KunServe | scaled down | Multi-server clusters of 80 GB GPUs; code public; models of 14B fit, 72B needs several cards. |
| C1 llm.npu | no | Needs Snapdragon NPU phones and Qualcomm tooling. |
| C2 HeteroInfer | no | Needs Snapdragon 8 Gen 3 / 8 Elite devices; no code. |
| C3 ElastiLM | scaled down | The offline stage ran on 8x A40 (same memory class as A6000) and could run on 4 GPUs more slowly; the on-device stage needs phones. |
| C4 EdgeLoRA | scaled down | Built on llama.cpp, so the multi-adapter serving experiments could run on a GPU node, but the paper's devices are Jetson and Raspberry Pi boards and results would not be comparable. |
| C5 CLONE | no | Custom 28 nm accelerator evaluated only in simulation; Jetson boards for the rest. |
| D1 throttLL'eM | scaled down | Needs GPU frequency control on the cards (whether RTX A6000 / 6000 Ada expose the same frequency controls as A100 was not verified) and tensor-parallel 8 for Llama3-70B is not possible on 4 GPUs. |
| D2 ML.ENERGY | scaled down | Benchmark code public; paper hardware is A100/H100; results on 48 GB workstation cards would be new measurements, not a reproduction of the paper's numbers. |
| E1 MELT | scaled down | Harness and engines (llama.cpp, MLC-LLM) are public and Jetson is CUDA-based, but phone and Jetson devices are required for the paper's numbers. |

- Common evaluation ingredients across Group A: llama.cpp is a baseline in PowerInfer, Kairox, DynamicInfer and Fiddler; FlexGen is the baseline in MoE-Lightning and FlexInfer; all the consumer-GPU papers report batch size 1 tokens/s on short prompts (ShareGPT, Alpaca or Chatbot Arena prompts).
- Common ingredients across Group B: ServerlessLLM and MuxServe are the recurring baselines (Aegaeon, Prism, WarmServe); metrics are TTFT/TPOT SLO attainment or goodput; request prompts are ShareGPT and arrival patterns are synthetic power-law or Azure/production traces.
- The newest paper in each chain re-implements or ports its predecessor rather than using the released code as is (Kairox ports PowerInfer; Prism builds MuxServe++; WarmServe reimplements MuxServe), which suggests the predecessors' released code does not run unchanged on current stacks.

### Gaps
- LIA: only the abstract and the code README were obtained; exact evaluation models, baselines by name, batch sizes and the authors' limitations are missing (no arXiv version was found and the ISCA PDF was not fetched).
- FlexInfer: limitations/future-work text and any code link were not extracted. WarmServe: GPU model and limitations were not extracted.
- Code links not verified from a primary page: PowerInfer repository URL, SpecExec repository (taken from the paper text), throttLL'eM repository (from a search result only), EdgeLoRA artifact URL, NEO code release.
- Host RAM for Fiddler's two environments is not stated in the text read.
- Whether the Hyperbolic / Novita / Arena traces used by Prism and the production traces used by Aegaeon are publicly downloadable was not checked.
- Follow-up lists rely on the baselines sections of later papers and on search results; a forward-citation search (Google Scholar / Semantic Scholar) was not run, so follow-ups are likely incomplete. The arXiv-only follow-ups named were not read.
- Dataset links are given only where a URL was seen in this session (Alpaca); ShareGPT, LMSYS-Chat-1M, Azure LLM trace, BurstGPT, LongBench, MTBench and HELM were named by the papers but their download URLs were not collected.

## Q3. Which problems do several papers name as still open?

### Takeaway
Nine open problems are named by two or more of the catalogued papers; the ones with the most independent mentions are the dependence of hybrid GPU-CPU speedups on activation sparsity, the CPU (compute and memory bandwidth) as the remaining bottleneck, no handling of resource contention from co-running applications, and reliance on offline profiling.

### Cited Findings

1. Hybrid GPU-CPU gains depend on activation sparsity and shrink on non-ReLU models
- PowerInfer: "The effectiveness of PowerInfer is directly correlated with the model's activation sparsity."; SiLU-family models gain only 1.47-1.7x — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Kairox: non-ReLU results "highlight potential opportunities for future research into inference optimization for less sparse or non-sparse model architectures on edge devices" — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- Fiddler: exploiting sparsity for MoE "remains an intriguing direction for future research" — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)

2. The CPU side (compute and memory bandwidth) becomes the bottleneck
- PowerInfer: for long inputs with short outputs "the CPU becomes the primary bottleneck" — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- MoE-Lightning: "the CPU attention will eventually become the bottleneck, which calls for higher CPU memory bandwidth" — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- SpecExec: inference "may become increasingly more constrained by the memory bandwidth" — [arXiv 2406.02532](https://arxiv.org/html/2406.02532)
- MELT: generation on device is memory-bound — [arXiv 2403.12844](https://arxiv.org/html/2403.12844)

3. Batch size above 1, concurrency and long prompts are weakly covered by single-GPU hybrid systems
- PowerInfer: speedup decreases as batch size grows because joint activation sparsity drops — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- Fiddler: evaluated only for "a small number of concurrent queries" — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- NEO: multi-GPU (4- and 8-GPU) integration and chunked-prefill integration are left as future work — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)

4. Resource contention from other applications on the same machine is not handled
- DynamicInfer: profiling "assumes a dedicated execution environment. In scenarios with resource contention from concurrent applications, a more sophisticated online measurement mechanism would be beneficial, which represents a direction for future work." — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b9762d1f15c68057275fa18d384c1a8-Paper-Conference.pdf)
- llm.npu: "llm.npu does not consider resource detection and contention in the current implementation" — [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2)
- HeteroInfer: without unified scheduling, tasks on one processor can "simultaneously block the execution of tasks on both processors" — [arXiv 2501.14794](https://arxiv.org/html/2501.14794)

5. Offline profiling and predictors are costly, inaccurate or hardware-specific
- PowerInfer: predictor training is "time-consuming (often several hours)" and predictors "may occasionally miss some active neurons" — [arXiv 2312.12456v2](https://arxiv.org/html/2312.12456v2)
- NEO: scheduling suffers from the "inevitable inaccuracy of the offline performance profiling" — [arXiv 2411.01142](https://arxiv.org/html/2411.01142)
- MuxServe: assumes known, slowly changing workloads and requires profiling — [arXiv 2404.02015](https://arxiv.org/html/2404.02015)
- Kairox artifact: "Different CPUs, memory bandwidth, PCIe generations, drivers, or model subsets may change absolute throughput" — [Kairox PDF](https://www.usenix.org/system/files/osdi26-jiang-yapeng.pdf)
- throttLL'eM: "optimal GPU frequency is architecture- and LLM-dependent" — [arXiv 2408.05235v1](https://arxiv.org/html/2408.05235v1)

6. Host RAM is assumed sufficient; disk/SSD offloading is left out
- MoE-Lightning: "when CPU memory is insufficient to hold the entire model, disk offloading becomes essential", which the paper does not address — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- Fiddler: assumes all non-expert layers fit on the GPU — [arXiv 2402.07033](https://arxiv.org/html/2402.07033)
- SpecExec: needs at least 140 GB host RAM for the 16-bit 70B target — [arXiv 2406.02532](https://arxiv.org/html/2406.02532)

7. No unified scheduling / memory layer across CPU, GPU and NPU on mobile SoCs
- HeteroInfer: "A unified API and memory management layer across CPU, GPU, and NPU would greatly simplify development" — [arXiv 2501.14794](https://arxiv.org/html/2501.14794)
- llm.npu: decoding is CPU-only and GPU-NPU coordination is not implemented — [arXiv 2407.05858v2](https://arxiv.org/html/2407.05858v2)
- ElastiLM: "none of mobile DSAs can seamlessly run prefill and decode of ElastiLM and all its baselines" — [arXiv 2409.09071v2](https://arxiv.org/html/2409.09071v2)
- MELT: expects "NPU acceleration, and framework-hardware co-design to be the biggest bet" — [arXiv 2403.12844](https://arxiv.org/html/2403.12844)

8. GPU sharing: combining spatial and temporal multiplexing, model-aware memory policies, and small-memory GPUs
- Nixie: "How to integrate spatial multiplexing into Nixie is a promising future direction"; policies could treat "different GPU memory regions (e.g., model weights, activations, KV caches) differently"; semantic hints are "an interesting direction for future work" — [Nixie PDF](https://www.usenix.org/system/files/osdi26-xu-yechen.pdf)
- MuxServe: colocation interference raises tail TPOT and placement is not optimal — [arXiv 2404.02015](https://arxiv.org/html/2404.02015)
- Aegaeon: the approach is "more effective with high-end GPUs"; on 24 GB A10 GPUs prefetching is disabled because two models' weights do not fit — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Prism: eviction is "based on an empirical threshold" — [arXiv 2505.04021v3](https://arxiv.org/html/2505.04021v3)
- KunServe: "the memory that can be freed by dropping is limited" and the implementation "focuses on dense models" — [arXiv 2412.18169](https://arxiv.org/html/2412.18169)

9. Energy: prefill phase and cross-layer optimization
- throttLL'eM: "throttLL'eM does not explicitly optimize the prefill phase" because frequency switching (about 200 ms) is slower than prefill — [arXiv 2408.05235v1](https://arxiv.org/html/2408.05235v1)
- ML.ENERGY: calls for "automated cross-layer energy optimizations"; the benchmark could not be repeated because of cost — [arXiv 2505.06371](https://arxiv.org/html/2505.06371)
- MELT: energy was analysed only "at a device-centric level", and sustained use is limited by thermal throttling — [arXiv 2403.12844](https://arxiv.org/html/2403.12844)

### Inferences
- Problems 1, 2, 3, 5 and 6 are all stated by papers whose testbed is a single consumer or workstation GPU (PowerInfer, Kairox, DynamicInfer, Fiddler, SpecExec) or a single small cloud GPU (MoE-Lightning, NEO), i.e. hardware of the student's class or smaller.
- Problem 4 (contention) is named by papers on both PCs and phones, and Nixie (OSDI 2026) is the one catalogued system whose evaluation is built around co-running applications on a consumer GPU; it does so for whole-model GPU residency, not for hybrid GPU-CPU inference.
- Problem 8 is stated only by papers evaluated on 80 GB datacenter GPUs except Nixie; Aegaeon's own 24 GB sensitivity run is the only data point in that group on a GPU smaller than the student's.
- This grouping is my clustering of the authors' statements; the papers do not cross-reference each other's open problems, and the count of mentions reflects which limitation sections were extracted (FlexInfer, LIA, WarmServe, EdgeLoRA and CLONE limitations are missing).

### Gaps
- Limitations text was not extracted for FlexInfer, LIA, WarmServe, EdgeLoRA and CLONE, so their open problems are absent from the tally.
- Kairox and Prism have no dedicated limitations section in the text read; the statements quoted come from evaluation and artifact text.
- No catalogued paper was found that states compiler/kernel-level optimization for commodity GPUs, or SSD/mmap-backed inference, as its main problem; whether those are named as open elsewhere could not be established from this set.
