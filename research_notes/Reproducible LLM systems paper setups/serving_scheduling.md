# LLM serving at the request-scheduling and cluster level: top-venue papers 2024-2026, their problems and evaluation setups

Notes on conventions used below.
- "Venue verified" means the title was seen on the venue's own program/proceedings page (URL given). "Venue from paper footer" means the ACM/USENIX footer of the PDF. "Search-level" means only a search-engine summary of the venue page was seen; treat as unverified.
- The field "Fits 1 node, 4x 48 GB (A6000/6000 Ada, PCIe), <=30B" is this note-taker's reading of the cited hardware and model facts, not a claim made by the papers.
- Quotes are verbatim from the paper text (arXiv HTML or PDF) unless marked as paraphrase.
- Date of these notes: 2026-11-11.

## Key Question 1: Which papers at OSDI, SOSP, NSDI, EuroSys, ASPLOS, ATC, MLSys, HPCA/MICRO/ISCA, NeurIPS/ICML/ICLR (2024-2026) address request scheduling and cluster-level LLM serving?

### Takeaway
At least 28 in-scope papers were located on venue program pages for 2024-2026, concentrated at OSDI'24 (two full sessions), NSDI'26, ASPLOS'26, OSDI'25/'26, ATC'25 and SOSP'24/'25. The 2026 programs (NSDI'26, ASPLOS'26, OSDI'26) contain more in-scope titles than the 2025 ones.

### Cited Findings

Venue-verified, by venue (titles as they appear on the program page):

- OSDI'24, session "Low-Latency LLM Serving": Sarathi-Serve ("Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"), ServerlessLLM, Llumnix, DistServe (plus InfiniGen, which is KV-cache and out of scope). Session "ML Scheduling": dLoRA, "Fairness in Serving Large Language Models" (VTC), Parrot (LLM applications, out of scope), USHER, MonoNN — [OSDI'24 technical sessions](https://www.usenix.org/conference/osdi24/technical-sessions)
- OSDI'25: "BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching"; NanoFlow (intra-device parallelism; engine-level) — [OSDI'25 technical sessions](https://www.usenix.org/conference/osdi25/technical-sessions)
- OSDI'26: "Simple Is Better: Multiplication May Be All You Need for LLM Request Scheduling" (LMetric); "Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning"; StriaTrace (tracing/diagnosis for online LLM inference); BatchGen (batch inference); "Efficient LLM Serving on Commodity GPU Clusters with Data-Reduced Cross-Instance Orchestration"; "Revisiting Pipeline Parallelism for LLM Serving"; OpenTela; Kairox — [OSDI'26 technical sessions](https://www.usenix.org/conference/osdi26/technical-sessions)
- NSDI'26: "FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference"; "HydraServe: Minimizing Cold Start Latency for Serverless LLM Serving in Public Clouds"; "JITServe: SLO-aware LLM Serving with Imprecise Request Information"; "Libra: Flexible Request Partitioning and Scheduling for Serving Unbalanced and Dynamic LLM Workloads"; "FlexLLM: Token-Level Co-Serving of LLM Inference and Finetuning with SLO Guarantees"; "ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production"; SYMPHONY; PlanetServe — [NSDI'26 technical sessions](https://www.usenix.org/conference/nsdi26/technical-sessions)
- NSDI'25: only SuperServe (not LLM-specific) was found in the portions of the page that were read — [NSDI'25 technical sessions](https://www.usenix.org/conference/nsdi25/technical-sessions)
- USENIX ATC'24: "Power-aware Deep Learning Model Serving with μ-Serve" — [ATC'24 technical sessions](https://www.usenix.org/conference/atc24/technical-sessions)
- USENIX ATC'25: "DEEPSERVE: Serverless Large Language Model Serving at Scale"; "Toppings: CPU-Assisted, Rank-Aware Adapter Serving for LLM Inference"; "Weaver: Efficient Multi-LLM Serving with Attention Offloading"; "Torpor: GPU-Enabled Serverless Computing for Low-Latency, Resource-Efficient Inference" — [ATC'25 technical sessions](https://www.usenix.org/conference/atc25/technical-sessions)
- SOSP'24: LoongServe ("Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism") — [SOSP'24 accepted papers](https://sigops.org/s/conferences/sosp/2024/accepted.html)
- SOSP'25: "Aegaeon: Effective GPU Pooling for Concurrent LLM Serving on the Market"; PrefillOnly; Pie; Jenga; IC-Cache (the last four are engine/KV/agent topics covered by other researchers) — [SOSP'25 accepted papers](https://sigops.org/s/conferences/sosp/2025/accepted.html)
- EuroSys'25: "SpotHedge: Serving AI Models on Spot Instances" (arXiv title: SkyServe); "DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs" — [EuroSys'25 accepted papers](https://2025.eurosys.org/accepted-papers.html)
- ASPLOS'26: "Towards High-Goodput LLM Serving with Prefill-decode Multiplexing" (MuxWise); "Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration"; "QoServe: Breaking the Silos of LLM Inference Serving"; "BlendServe: Optimizing Offline Inference with Resource-Aware Batching"; "Shift Parallelism"; "XY-Serve" — [ASPLOS'26 program](https://www.asplos-conference.org/asplos2026/program/)
- MLSys'24: "VIDUR: A Large-Scale Simulation Framework for LLM Inference" (also listed there: Punica, S-LoRA — multi-tenant LoRA serving) — [MLSys 2024 proceedings](https://proceedings.mlsys.org/paper_files/paper/2024)
- ISCA'24 (Splitwise) and HPCA'25 (DynamoLLM): venue stated in the Azure trace repository README that points to the two papers — [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset)

Venue from paper footer only (program page not fetched):

- SkyWalker ("a multi-region load balancer for LLM inference"): footer reads "EUROSYS '26, Edinburgh, Scotland Uk", doi 10.1145/3767295.3769353 — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095)
- MuxWise: footer "ASPLOS '26, March 22-26, 2026, Pittsburgh, PA, USA", doi 10.1145/3779212.3790236 (consistent with the program page above) — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489)

Search-level only (treat venue as unverified):

- "Efficient LLM Scheduling by Learning to Rank": a search summary of the NeurIPS proceedings page reports Main Conference Track, NeurIPS 2024 (NeurIPS 37); the proceedings page itself was not opened — [NeurIPS 2024 proceedings entry](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6c8985579293e0209bdaa4f21bb1d237-Abstract.html)
- Chameleon ("Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments"): a search summary of the arXiv v2 HTML reports MICRO 2025 (58th IEEE/ACM MICRO, Oct 18-22 2025, Seoul), doi 10.1145/3725843.3756083 — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741)
- BurstGPT: a search summary of university publication pages reports KDD 2025 (31st ACM SIGKDD, Toronto, published 3 Aug 2025); the ACM page was not opened — [arXiv 2401.17644](https://arxiv.org/abs/2401.17644)

### Inferences
- By count of in-scope titles on program pages, the densest single sessions are OSDI'24 "Low-Latency LLM Serving" (4 in scope) and the NSDI'26 program (6 in scope); several papers that circulated on arXiv in 2023-2025 under other names appear at 2025-2026 venues (FastServe arXiv 2023 -> NSDI'26; SkyServe -> SpotHedge EuroSys'25; Niyama -> QoServe ASPLOS'26; CaraServe -> Toppings ATC'25; Tempo -> JITServe NSDI'26), so arXiv names and venue names must be cross-checked when searching for follow-ups.

### Gaps
- EuroSys'26 accepted-papers page returned HTTP 404; KunServe, FlexPipe and AdaServe at EuroSys'26 are search-level only (unverified). SkyWalker's EuroSys'26 venue rests on the paper footer.
- The ASPLOS'25 program URL returned the same list as ASPLOS'26 (apparent redirect), so ASPLOS'25 was NOT surveyed.
- Not surveyed at all: SIGCOMM, FAST (beyond the Mooncake trace release), VLDB, SC, ICML, ICLR, MLSys'25/'26, HPCA'26, ISCA'25/'26, MICRO'24/'26, ATC'26, SOSP'26, EuroSys'24. Preble (prefix-aware distributed routing, [arXiv 2407.00023](https://arxiv.org/abs/2407.00023)) is commonly cited as ICLR'25; that venue was not checked.
- Tails of several long USENIX program pages were not read (NSDI'26 last ~40k characters, ATC'25 last ~31k, ATC'24 last ~81k, part of NSDI'25), so those lists may be incomplete.
- For the OSDI'26 titles "Efficient LLM Serving on Commodity GPU Clusters with Data-Reduced Cross-Instance Orchestration", "Revisiting Pipeline Parallelism for LLM Serving", Libra, FlexLLM (NSDI'26), Weaver, Torpor (ATC'25), only the titles were confirmed; their evaluation setups were not extracted.

## Key Question 2: For each paper, what are the problem, the evaluation setup, the fit to one node with 4x 48 GB GPUs, the stated limitations, and the follow-ups?

### Takeaway
Twenty-six papers are recorded below with models, workloads, hardware, baselines and headline numbers. Most request-scheduling papers (Sarathi-Serve, VTC, Learning-to-Rank, QoServe, Chameleon, Toppings, FastServe's single-GPU experiments) were evaluated on 1-8 GPUs with 7B-34B models and public datasets, whereas cluster-level papers (DistServe, dLoRA, BlitzScale, Prism, LMetric, Aegaeon, DynamoLLM) used 16-32 A100/A800/H100/H800/H20 GPUs with NVLink or RDMA.

### Cited Findings

#### A. Request scheduling, batching, SLO and fairness inside/around one engine

**A1. Sarathi-Serve — "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (OSDI'24, venue verified)**
- Links: paper [arXiv 2403.02310](https://arxiv.org/abs/2403.02310); code [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve).
- Problem: prefill-prioritizing schedulers (vLLM, Orca) stall ongoing decodes and produce high tail time-between-tokens (TBT); decode-prioritizing schedulers lose throughput. Affects anyone running interactive serving on a shared replica — [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
- Setup: Mistral-7B on 1x A100-80GB; Yi-34B on 2x A100 (TP2); LLaMA2-70B on 8x A40 48 GB (TP4-PP2); Falcon-180B on 8x A100 across 2 nodes. Workloads: openchat_sharegpt4 and arxiv_summarization lengths with Poisson arrivals. Baselines: vLLM, Orca. Metric: capacity (max QPS) under a P99 TBT SLO; 2.6x (Mistral-7B) and up to 3.7x (Yi-34B) over vLLM, up to 5.6x for Falcon-180B with pipeline parallelism — [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
- Fits 1 node, 4x 48 GB: yes for Mistral-7B and Yi-34B (the paper itself ran LLaMA2-70B on 48 GB A40s, but with 8 of them); the 70B and 180B configurations need 8 GPUs, so scaled down = drop those two models.
- Stated limitations: "The system performance can be further enhanced by dynamically varying the token budget based on workload characteristics... We leave this exploration for future work."; "We leave a quantitative comparison between Sarathi-Serve and disaggregation-based solutions for future work."; the prototype "doesn't have complete feature parity with open-source vLLM"; a small token budget "can lead to higher overhead due to lower arithmetic intensity" — [arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
- Follow-ups that use it as a baseline or build on it: QoServe uses Sarathi-FCFS/EDF/SRPF/Silo as all four baselines — [arXiv 2503.22562](https://arxiv.org/abs/2503.22562); JITServe compares against Sarathi-Serve — [arXiv 2504.20068](https://arxiv.org/abs/2504.20068); MuxWise and Bullet compare against chunked prefill in SGLang/vLLM — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489), [arXiv 2504.19516](https://arxiv.org/abs/2504.19516); FastServe reports up to 1.5x over vLLM with chunked prefill — [NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf); Vidur simulates it — [arXiv 2405.05465](https://arxiv.org/abs/2405.05465).

**A2. FastServe — "Iteration-Level Preemptive Scheduling for Large Language Model Inference" (NSDI'26, venue verified; arXiv since 2023)**
- Links: paper [NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf), [arXiv 2305.05920](https://arxiv.org/abs/2305.05920); code: no repository link found in the paper text (no public code, unverified).
- Problem: run-to-completion FCFS causes head-of-line blocking; FastServe preempts at token granularity with a skip-join MLFQ and proactively offloads KV state to host memory — [NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf)
- Setup: testbed "two AWS EC2 p4d.24xlarge instances" with "eight NVIDIA A100 40GB GPUs" each; design-choice experiments on 1x A100 40GB. Models OPT-13B (1 GPU), OPT-66B (4 GPUs), OPT-175B (16 GPUs), Llama3-8B, FP16. Workloads: ShareGPT and Alpaca lengths with Poisson arrivals. Baselines: FasterTransformer v5.3, vLLM v0.6.1, FastServe-FCFS, and vLLM with chunked prefill. Headline: "improves the throughput by up to 6.1x" over vLLM — [NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf)
- Fits 1 node, 4x 48 GB: scaled down — OPT-13B and Llama3-8B on one GPU match the paper's single-GPU experiments; OPT-66B/175B (4 and 16 A100s) are out.
- Stated limitations: no explicit limitations section was found by text search of the NSDI PDF.
- Follow-ups: Learning-to-Rank uses MLFQ (FastServe) as a baseline — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792); JITServe's related work groups it with Orca, dLoRA and VTC — [JITServe NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf).

**A3. "Efficient LLM Scheduling by Learning to Rank" (NeurIPS'24 main track — search-level venue confirmation)**
- Links: paper [arXiv 2408.15792](https://arxiv.org/abs/2408.15792); code [hao-ai-lab/vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr).
- Problem: FCFS head-of-line blocking; exact output-length prediction is hard, so rank requests by predicted relative length to approximate SJF — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792)
- Setup: Llama-3 8B and 70B; predictors OPT-125M/350M; datasets ShareGPT, LMSYS-Chat-1M (10k serving + 10k training prompts), Alpaca; hardware one DGX with 8x A100 40GB (some ablations 1x A100 80GB); baselines FCFS, MLFQ (FastServe), Perception Only, Classification (S3), on vLLM v0.4.1. Headline: 2.8x lower latency in chatbot serving and 6.5x higher throughput in synthetic data generation; Kendall's Tau 0.54/0.62 — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792)
- Fits 1 node, 4x 48 GB: yes for Llama-3-8B; Llama-3-70B (above the ~30B limit) dropped or replaced.
- Stated limitations: "The ranking metric Kendall's Tau still has limitations in reflecting the performance of end-to-end tasks."; the scheduler "currently works with standard LLM serving techniques"; integration with newer optimizations "has still not been fully studied... We will leave them as future work."; a gap to the Oracle remains — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792)
- Follow-ups: JITServe uses LTR as a baseline — [arXiv 2504.20068](https://arxiv.org/abs/2504.20068); Chameleon cites it and μ-Serve as SJF-style schedulers — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741).

**A4. VTC — "Fairness in Serving Large Language Models" (OSDI'24, venue verified)**
- Links: paper [arXiv 2401.00588](https://arxiv.org/abs/2401.00588); code [Ying1123/VTC-artifact](https://github.com/Ying1123/VTC-artifact).
- Problem: with FCFS one heavy client degrades all others, and requests-per-minute limits waste capacity; defines token-based fair service and a Virtual Token Counter scheduler with a proven 2x bound on service difference between backlogged clients — [arXiv 2401.00588](https://arxiv.org/abs/2401.00588)
- Setup: Llama-2-7b on A10G 24 GB and Llama-2-13b on A100 80 GB, on S-LoRA/LightLLM; workloads synthetic (constant, ON/OFF, Poisson, distribution shift) plus an LMSYS Chatbot Arena trace (27 clients, 10 min, 210 req/min); baselines FCFS, RPM, LCF, VTC(predict), VTC(oracle). Result on the real trace: max service difference 759.97 (FCFS) vs 368.40 (VTC); throughput 777 (FCFS) vs 779 (VTC) vs 340 (RPM(5)) — [arXiv 2401.00588](https://arxiv.org/abs/2401.00588)
- Fits 1 node, 4x 48 GB: yes (single GPU, 7B/13B).
- Stated limitations: "we leave an investigation on preemption as an orthogonal future work"; "Our algorithm does not consider preemption."; the cost formulations "cannot accurately reflect the actual LLM serving cost"; length-predictor effectiveness is "contingent upon both the workload and the accuracy of predictions" — [arXiv 2401.00588](https://arxiv.org/abs/2401.00588)
- Follow-ups: SkyWalker's related work describes DLPM as a scheduler that "improves upon Preble in both latency and throughput while also offering fairness guarantees to clients" — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095).

**A5. QoServe — "Breaking the Silos of LLM Inference Serving" (ASPLOS'26, venue verified; arXiv name "Niyama")**
- Links: paper [arXiv 2503.22562](https://arxiv.org/abs/2503.22562) (arXiv version was read; the camera-ready may differ); code: none stated in the arXiv text.
- Problem: providers run separate (siloed) fleets for interactive and batch traffic, wasting capacity; QoServe co-schedules fine-grained QoS classes on shared replicas — [arXiv 2503.22562](https://arxiv.org/abs/2503.22562)
- Setup: Llama3-8B on 1x A100-80GB; Qwen-7B on 2x A100 (TP2); workloads ShareGPT plus Azure Conversation and Azure Code traces (2024) with Poisson arrivals and 3 QoS tiers; baselines Sarathi-Silo, Sarathi-FCFS, Sarathi-EDF, Sarathi-SRPF. Headline: 32% more serving capacity than siloed deployment; 12-32% fewer GPUs at 50 QPS; under overload, violations of 8.64% vs 81.88% (FCFS) and 84.12% (EDF) — [arXiv 2503.22562](https://arxiv.org/abs/2503.22562)
- Fits 1 node, 4x 48 GB: yes (8B/7B on 1-2 GPUs; absolute numbers will differ from A100).
- Stated limitations: no limitations section in the arXiv version; the text notes that decode length is unknown in advance and that the α parameter needs tuning with load (paraphrase) — [arXiv 2503.22562](https://arxiv.org/abs/2503.22562)
- Follow-ups: none identified.

**A6. JITServe — "SLO-aware LLM Serving with Imprecise Request Information" (NSDI'26, venue verified; arXiv v1 titled "Tempo")**
- Links: paper [NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf), [arXiv 2504.20068](https://arxiv.org/abs/2504.20068); code [UIUC-MLSys/JITServe](https://github.com/UIUC-MLSys/JITServe).
- Problem: SLO-aware scheduling when output lengths and request dependencies (multi-step/agentic) are not known up front — [arXiv 2504.20068](https://arxiv.org/abs/2504.20068)
- Setup: Llama-3.1-8B, Qwen2.5-14B, Qwen3-30B-MoE-A3B, Llama3.1-70B on a "cluster of 16 NVIDIA A100 GPUs"; workloads: chatbot (Alpaca, LMSYS-chat), long-context math reasoning, deep research (Search Arena), agentic code generation; arrivals follow "Microsoft's real-world LLM serving trace, scaled" plus a Poisson ablation; more than 10K requests over at least 1 hour; SLOs derived from 1K DeepSeek API calls (about 2 s TTFT, 100 ms TBT, 20 s end-to-end). Baselines Autellix, LTR, Sarathi-Serve, vLLM. Headline: 1.4x-6.3x goodput or 28.5%-83.2% resource savings — [arXiv 2504.20068](https://arxiv.org/abs/2504.20068)
- Fits 1 node, 4x 48 GB: scaled down — the 8B, 14B and 30B-MoE models fit; Llama3.1-70B and the 16-GPU cluster scale do not.
- Stated limitations: "JITServe adopts an all-or-nothing goodput definition... does not capture scenarios where near-miss completions still provide partial utility"; "Under significant workload distribution shifts or poor pattern matching, early scheduling decisions may be suboptimal, particularly in multi-tenant settings with conflicting SLO priorities... Persistent shifts may require retraining predictors or incorporating explicit fallback policies."; "For more complex deployments, such as multi-node, highly parallel, multi-replica, or mixed-GPU environments, runtime profiling may become necessary" — [JITServe NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf)
- Follow-ups: none identified (published 2026).

#### B. Prefill/decode disaggregation and multiplexing

**B1. DistServe — "Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" (OSDI'24, venue verified)**
- Links: paper [arXiv 2401.09670](https://arxiv.org/abs/2401.09670); code [LLMServe/DistServe](https://github.com/LLMServe/DistServe).
- Problem: colocating prefill and decode on one instance causes interference and couples their resource allocation, so providers over-provision to meet both TTFT and TPOT SLOs — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
- Setup: OPT-13B/66B/175B (FP16); workloads ShareGPT (chatbot), HumanEval (code completion), LongBench (summarization) with Poisson arrivals; hardware 4 nodes x 8 A100-80GB (32 GPUs), NVLink inside a node, 25 Gbps across nodes; baselines vLLM, DeepSpeed-MII. Headline: up to 7.4x more requests or 12.6x tighter SLO while keeping >90% SLO attainment — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
- Fits 1 node, 4x 48 GB: scaled down — OPT-13B with e.g. prefill and decode instances split across the 4 GPUs; no NVLink, so KV transfer goes over PCIe/host; OPT-66B/175B and the 32-GPU placement search are out.
- Stated limitations: "DistServe does not implement advanced runtime policies like preemption"; "the dependency between prefill and decoding instances introduces the risk of fault propagation"; "We leave both as future work."; FCFS "can lead to a 'convoy effect'"; the authors note the approach may be less effective in resource-constrained or small-scale settings, and that KV transfer overhead grows with prompt length (paraphrase) — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
- Follow-ups: BlitzScale uses DistServe (full/half) as a baseline — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246); MuxWise compares with disaggregated SGLang-PD and LoongServe — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489); LoongServe reports 5.81x over prefill-decoding disaggregation — [arXiv 2404.09526](https://arxiv.org/abs/2404.09526); DeepServe schedules across PD-disaggregated and PD-colocated instances — [arXiv 2501.14417](https://arxiv.org/abs/2501.14417); ServeGen's case study runs PD-disaggregated Qwen2.5-72B — [arXiv 2505.09999](https://arxiv.org/abs/2505.09999).

**B2. Splitwise — "Efficient Generative LLM Inference Using Phase Splitting" (ISCA'24 per the Azure trace README)**
- Links: paper [arXiv 2311.18677](https://arxiv.org/abs/2311.18677); simulator [Mutinifni/splitwise-sim](https://github.com/Mutinifni/splitwise-sim); traces [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset).
- Problem: prompt and token phases have different compute/power profiles; splitting them onto separate (possibly different-generation) machines lets a provider optimize a cluster for throughput, cost or power — [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)
- Setup: models Llama2-70B and BLOOM-176B; KV-cache transfer implemented on vLLM and run on "two DGX-A100 and two DGX-H100 virtual machines (VMs) on Microsoft Azure" connected with InfiniBand; cluster-level results from a simulator driven by Azure coding and conversation traces; baselines Baseline-A100 and Baseline-H100 clusters with mixed continuous batching. Headline: "up to 1.4x higher throughput at 20% lower cost" or "2.35x more throughput under the same power and cost budgets" — [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)
- Fits 1 node, 4x 48 GB: no for the hardware experiments (70B/176B models on DGX A100/H100 with InfiniBand); the simulator and the traces are usable on CPU.
- Stated limitations: "Designing safe and efficient failure recovery is out of scope for our paper."; "Due to customer privacy requirements (e.g., GDPR), we do not have visibility into the content of the prompt" — [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)
- Follow-ups: DynamoLLM (same group; uses the 2024 trace) — [arXiv 2408.00741](https://arxiv.org/abs/2408.00741); Chameleon takes its request lengths from the Splitwise trace — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741); MuxWise describes Splitwise as "static disaggregation" — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489).

**B3. MuxWise — "Towards High-Goodput LLM Serving with Prefill-decode Multiplexing" (ASPLOS'26, venue verified and in paper footer)**
- Links: paper [arXiv 2504.14489](https://arxiv.org/abs/2504.14489); code: an SGLang fork is referenced (github.com/ykcombat/sglang) and the paper says "We plan to open-source MuxWise after publication."; blog [LMSYS PD-multiplexing post](https://www.lmsys.org/blog/2025-09-28-pdmux).
- Problem: disaggregation statically splits GPUs and duplicates KV pools, chunked prefill couples the phases; MuxWise runs prefill and decode concurrently inside one GPU with spatial sharing — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489)
- Setup: built on SGLang; servers with 8x A100-80GB (NVLink), 8x H100 and 8x H200; models Llama-8B, Llama-70B, Qwen3-235B MoE; baselines chunked prefill in SGLang, SGLang-PD (disaggregated), LoongServe. Headline: "improves peak throughput under SLO guarantees by an average of 2.20x (up to 3.06x)" — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489)
- Fits 1 node, 4x 48 GB: scaled down — Llama-8B only; relies on intra-process GPU spatial sharing, whose behaviour on A6000-class cards was not evaluated in the paper.
- Stated limitations: "When the SLO target is loose or absent, such as in offline serving, MuxWise has no opportunity to outperform baselines such as chunked-prefill or NanoFlow."; "MuxWise is a single-instance optimization"; when prefill instances consistently handle long requests "or decode multiplexing violates TTFT SLO, MuxWise offers limited benefit" — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489)
- Follow-ups / same-venue neighbours: Bullet (ASPLOS'26) addresses the same concurrent prefill/decode execution — [arXiv 2504.19516](https://arxiv.org/abs/2504.19516); MuxWise's related work names WindServe as prior prefill/decode multiplexing — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489).

**B4. Bullet — "Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration" (ASPLOS'26, venue verified; arXiv version read)**
- Links: paper [arXiv 2504.19516](https://arxiv.org/abs/2504.19516); code [zejia-lin/Bullet](https://github.com/zejia-lin/Bullet).
- Problem: chunked-prefill ("hybrid") batching under-uses the GPU and favours latency over throughput; Bullet runs prefill and decode concurrently and provisions GPU compute (SM share) dynamically from a performance model — [arXiv 2504.19516](https://arxiv.org/abs/2504.19516)
- Setup: Llama3.1-8B/70B and Qwen3-235B-A22B-FP8; three servers (8x A100-80GB, 8x H100, 8x H20), multi-GPU via tensor parallelism; workloads ShareGPT, Azure-Code, arXiv-Summary with Poisson arrivals; SLOs per workload (TPOT 150/200/175 ms); baselines vLLM v0.8.5, SGLang v0.4.6 (1024/2048 chunk), NanoFlow. Headline: "1.26x average throughput gains (up to 1.55x)" — [arXiv 2504.19516](https://arxiv.org/abs/2504.19516)
- Fits 1 node, 4x 48 GB: scaled down — Llama3.1-8B on one GPU; 70B and 235B are out.
- Stated limitations: "on low-compute GPUs running dense models, the limited SM budget obliges the decode phase to claim a larger ratio of SMs to saturate memory bandwidth... slightly tempering concurrent-execution benefits"; "wave quantization remains an open issue"; "Extending the latency model to cover new attention variants or LoRA adapters is straightforward and left to future work." — [arXiv 2504.19516](https://arxiv.org/abs/2504.19516)
- Follow-ups: none identified.

**B5. LoongServe — "Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism" (SOSP'24, venue verified)**
- Links: paper [arXiv 2404.09526](https://arxiv.org/abs/2404.09526); code [LoongServe/LoongServe](https://github.com/LoongServe/LoongServe).
- Problem: request lengths vary by orders of magnitude and the two phases need different degrees of parallelism; static parallelism (chunked prefill or PD disaggregation) wastes resources — [arXiv 2404.09526](https://arxiv.org/abs/2404.09526)
- Setup: model LWM-1M-Text (same architecture as Llama-2-7B, 1M-token context); servers "each with eight NVIDIA A800 80GB GPUs", four 200 Gbps InfiniBand NICs, most experiments on one server; workloads ShareGPT (4-2.3K tokens), L-Eval (2.7K-210.5K), LV-Eval (15.1K-497.3K) and an equal-probability Mixed set with Poisson arrivals; baselines vLLM (TP8), DeepSpeed-MII (Dynamic SplitFuse), LightLLM with SplitFuse, DistServe. Headline: throughput "up to 3.85x compared to chunked prefill and 5.81x compared to prefill-decoding disaggregation" — [arXiv 2404.09526](https://arxiv.org/abs/2404.09526)
- Fits 1 node, 4x 48 GB: scaled down — the model is 7B-class so 4 GPUs can host 4 elastic instances, but the paper used 8x 80 GB GPUs with NVLink (400 GB/s) and the longest requests (hundreds of thousands of tokens) need the larger memory.
- Stated limitations: no explicit limitations section was found by text search.
- Follow-ups: MuxWise uses LoongServe as a baseline ("dynamic disaggregation") — [arXiv 2504.14489](https://arxiv.org/abs/2504.14489).

#### C. Load balancing and routing across replicas

**C1. Llumnix — "Dynamic Scheduling for Large Language Model Serving" (OSDI'24, venue verified)**
- Links: paper [arXiv 2406.03243](https://arxiv.org/abs/2406.03243); code [AlibabaPAI/llumnix](https://github.com/AlibabaPAI/llumnix).
- Problem: heterogeneous, unpredictable requests cause queuing, memory fragmentation, tail latency and SLO violations across multi-instance deployments; Llumnix reschedules running requests with live migration — [arXiv 2406.03243](https://arxiv.org/abs/2406.03243)
- Setup: LLaMA-7B (1 GPU/instance) and LLaMA-30B (4 GPUs, TP) on vLLM; workloads ShareGPT (GPT4) and BurstGPT (GPT4-Conversation) lengths, generated power-law lengths, Poisson/Gamma arrivals, 10,000 requests; hardware 16 GPUs = 4 Alibaba Cloud VMs each with 4x A10 24 GB (PCIe 4.0); baselines round-robin, INFaaS++, Llumnix-base. Headline: up to 15x better P99 prefill latency than INFaaS++, high-priority requests up to 1.5x faster, up to 36% cost saving — [arXiv 2406.03243](https://arxiv.org/abs/2406.03243)
- Fits 1 node, 4x 48 GB: scaled down — same GPU class or better (A10 24 GB, PCIe) but 4 instances instead of 16; 48 GB allows 2 small instances per GPU as a substitute.
- Stated limitations: the vLLM version used "only supports the original LLaMA with a maximum sequence length of 2k"; "Currently, Llumnix supports two priority classes, high and normal"; "There could be a lot of heuristics to explore for setting the virtual usage"; "We plan to extend Llumnix to support multiple model types in future work"; the scalability test replaced GPU execution with sleep (paraphrase) — [arXiv 2406.03243](https://arxiv.org/abs/2406.03243)
- Follow-ups: LMetric (OSDI'26) revisits cross-instance request scheduling with KV-cache awareness — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf); Chameleon cites Llumnix as cross-replica rescheduling — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741).

**C2. LMetric — "Simple Is Better: Multiplication May Be All You Need for LLM Request Scheduling" (OSDI'26, venue verified)**
- Links: paper [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf), [arXiv 2603.15202](https://arxiv.org/abs/2603.15202); code [blitz-serving/blitz-router](https://github.com/blitz-serving/blitz-router); traces [qwen-bailian-usagetraces-anon](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon).
- Problem: a router in front of many engine instances must balance KV-cache (prefix) hits against load; linear-combination scores need per-workload tuning. LMetric scores an instance by (new prefill tokens) x (batch size) — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf)
- Setup: "a testbed with 16 NVIDIA H20 GPUs", 96 GB each, one vLLM-v1 instance per GPU; router on a 160-core Xeon server; models Qwen2-7B (dense) and Qwen3-30B (MoE); traces ChatBot and Agent (Qwen, public), Coder (Bailian, not released), ToolAgent (Mooncake FAST'25 release); baselines the Bailian production scheduler, vLLM router, ai-Dynamo, llm-d. Headline (OSDI abstract): TTFT reduced "by 92% and 39%, and TPOT by 24% and 51%, compared to vLLM-v1 and an in-production scheduler" — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf). The arXiv v1 abstract gave different numbers (TTFT -92%/-52%, TPOT -21%/-20%) — [arXiv 2603.15202](https://arxiv.org/abs/2603.15202).
- Fits 1 node, 4x 48 GB: scaled down — 4 instances (or 8 with two 7B instances per GPU) instead of 16; both models are within 30B; the public traces must be rate-scaled.
- Stated limitations: "there do exist cases where multiplication may fail due to extremely skewed KV$ accesses"; "Efficient scheduling for PD-disaggregated serving is beyond the scope of this paper."; "we cannot afford to sweep all possible configurations"; on ToolAgent the mean TTFT is about 10% higher than llm-d (paraphrase) — [arXiv 2603.15202](https://arxiv.org/abs/2603.15202)
- Follow-ups: none identified (published 2026).

**C3. SkyWalker — multi-region load balancer for LLM inference (EuroSys'26 per paper footer; program page not verified)**
- Links: paper [arXiv 2505.24095](https://arxiv.org/abs/2505.24095); code: no repository link found in the paper text.
- Problem: providers reserve GPUs for each region's peak; diurnal patterns differ across regions, so cross-region routing that preserves KV-cache locality can cut reserved capacity — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095)
- Setup: "All replicas use one L4 GPU, hosting the meta-llama/Llama-3.1-8B-Instruct model via SGLang"; replicas across US, Europe and Asia on AWS (GKE Gateway runs on GCP); workloads ChatBot Arena multi-turn (80 conversations per region), WildChat (40 US / 30 EU / 30 Asia clients), Tree of Thoughts on GSM8K; baselines GKE Gateway, Round Robin, Least Load, Consistent Hashing, SGLang Router. Headline: "1.12-2.06x higher throughput and 1.74-6.30x lower latency compared to existing load balancers, while reducing total serving cost by 25%" — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095)
- Fits 1 node, 4x 48 GB: scaled down — the replica unit (one 24 GB GPU, 8B model) fits, but the multi-region network would have to be emulated on one node.
- Stated limitations: "While SkyWalker currently focuses on homogeneous replicas, it can be extended to support heterogeneous accelerators"; for that case "the prefix-aware routing and overall load balancing policies remain as an open question"; "Request characteristics, such as prompt length, can influence ideal routing strategies... SkyWalker can be extended to incorporate request-characteristic aware routing strategies"; GDPR/regulatory constraints on cross-region routing are discussed as future work — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095)
- Follow-ups: none identified.

#### D. Autoscaling, cold start, serverless and multi-model serving

**D1. ServerlessLLM — "Low-Latency Serverless Inference for Large Language Models" (OSDI'24, venue verified)**
- Links: paper [arXiv 2401.14351](https://arxiv.org/abs/2401.14351); code [ServerlessLLM/ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM).
- Problem: serverless LLM endpoints suffer long cold starts because checkpoints are large; proposes a loading-optimized checkpoint format, multi-tier local storage, live migration and locality-aware scheduling — [arXiv 2401.14351](https://arxiv.org/abs/2401.14351)
- Setup: models OPT 2.7B/6.7B/13B/30B, LLaMA-2 7B/13B/70B, Falcon; datasets GSM8K and ShareGPT; arrivals from the Azure Serverless (Functions) trace modelled as Gamma with CV=8; hardware (i) one server with 8x A5000, NVMe RAID0, MinIO over 1 Gbps; (ii) 4 servers x 4 A40 (48 GB), 10 Gbps; baselines PyTorch and Safetensors loaders, Ray Serve (with and without cache), KServe, Shepherd*. Headline: 10-200x latency reduction; checkpoint loading 3.6-8.2x faster — [arXiv 2401.14351](https://arxiv.org/abs/2401.14351)
- Fits 1 node, 4x 48 GB: yes for the single-server experiments (loader, 8x A5000 testbed is weaker per GPU); scaled down for the cluster scheduler (the paper used 4 servers x 4 A40, i.e. the same GPU class but 16 GPUs and a 10 Gbps network). Needs fast NVMe.
- Stated limitations: "Since there are no publicly available LLM serverless inference workloads, we use Azure Serverless Trace"; "Optimization of checkpoint placement is considered a separate issue and is not addressed in this paper."; "exploring concurrent loading on servers with a fairness guarantee is planned for future work."; "We believe our work can be extended to ensure fairness of resources across the cluster" — [arXiv 2401.14351](https://arxiv.org/abs/2401.14351)
- Follow-ups that use it as a baseline: BlitzScale (OSDI'25) — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246); HydraServe (NSDI'26) — [arXiv 2502.15524](https://arxiv.org/abs/2502.15524); Aegaeon (SOSP'25) — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf); Prism (OSDI'26) — [arXiv 2505.04021](https://arxiv.org/abs/2505.04021). Four later top-venue papers compare against it.

**D2. BlitzScale — "Fast and Live Large Model Autoscaling with O(1) Host Caching" (OSDI'25, venue verified)**
- Links: paper [arXiv 2412.17246](https://arxiv.org/abs/2412.17246); code [blitz-serving/blitz-scale](https://github.com/blitz-serving/blitz-scale).
- Problem: in model-as-a-service, autoscaling speed is limited by parameter loading; host caching every model is expensive. BlitzScale loads parameters over the network from already-running instances and serves live while loading — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246)
- Setup: Llama3-8B, Mistral-24B, Qwen2.5-72B; traces BurstGPT, AzureCode, AzureConv, scaled with TraceUpscaler; Cluster A 4 hosts x 8 A800-80GB (NVLink, 100 Gbps RDMA), Cluster B 2 hosts x 8 A100-80GB (PCIe, 100 Gbps RDMA); baselines ServerlessLLM, ServerlessLLM AllCache, DistServe (full/half), vLLM. Headline (abstract): up to 94% lower tail latency than ServerlessLLM, 47-75% shorter TTFT, 49% less GPU time than DistServe/vLLM at the same SLO — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246). Conflict to note: the evaluation text gives a 19.46% GPU-time figure versus ServerlessLLM, which differs from the abstract's 49% (different baselines) — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246)
- Fits 1 node, 4x 48 GB: no as designed (the mechanism is multi-host parameter transfer over 100 Gbps RDMA); only a single-host GPU-to-GPU variant with 8B/24B models could be emulated.
- Stated limitations: the scaling "policy depends heavily on workload characteristics, which we leave as future work."; "BlitzScale currently focuses on instance-level scaling"; "Live scaling a decode instance in PD disaggregation without interference is impossible"; the spine network is not modelled; less effective on BurstGPT "due to the limitations of our cluster" — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246)
- Follow-ups: none identified beyond same-group work (LMetric's router is published under the same GitHub organization) — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf).

**D3. HydraServe — "Minimizing Cold Start Latency for Serverless LLM Serving in Public Clouds" (NSDI'26, venue verified)**
- Links: paper [arXiv 2502.15524](https://arxiv.org/abs/2502.15524); code [LLMServe/hydraserve](https://github.com/LLMServe/hydraserve).
- Problem: in public clouds (no local model cache, limited fetch bandwidth) cold starts dominate TTFT; HydraServe starts a model as pipeline-parallel pieces across servers and later consolidates — [arXiv 2502.15524](https://arxiv.org/abs/2502.15524)
- Setup: Llama2-7B/13B, OPT-6.7B; datasets ShareGPT, HumanEval, LongBench; arrivals from Azure Function Trace (Gamma, varied CV and RPS); testbeds (i) 4 A10 servers (1 GPU each) + 4 V100 servers (4 GPUs each), 16 Gbps, (ii) 2 A10 servers (4 GPUs) + 4 V100 servers; production run on A10; baselines serverless vLLM, ServerlessLLM (with/without cache). Headline: cold-start latency 1.7-4.7x lower; TTFT SLO attainment 1.43-1.74x higher — [arXiv 2502.15524](https://arxiv.org/abs/2502.15524)
- Fits 1 node, 4x 48 GB: scaled down — models and GPU class fit (A10/V100 are weaker), but the multi-server fetch bandwidth and cross-server pipeline must be emulated on one node.
- Stated limitations: pipeline parallelism "may lead to worker colocations across different models", which "can degrade performance"; "Due to the lack of high-speed SSDs in our testbeds..."; "Due to the lack of NVLink, all evaluated models can reside in a single GPU"; PCIe contention is not considered (paraphrase) — [arXiv 2502.15524](https://arxiv.org/abs/2502.15524)
- Follow-ups: none identified (published 2026).

**D4. Aegaeon — "Effective GPU Pooling for Concurrent LLM Serving on the Market" (SOSP'25, venue verified)**
- Links: paper [author-hosted PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf), doi 10.1145/3731569.3764815; code: no repository link found in the paper text.
- Problem: model marketplaces host many sporadically invoked models, each pinning GPUs; Aegaeon auto-scales (switches models) at token granularity so several models share a GPU pool — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Setup: testbed "two nodes with 16 GPUs in total, where each node is equipped with eight NVIDIA H800 80GB GPUs connected via NVLINK"; models from the Qwen, Llama, InternLM and Yi families, mainly 6B-14B, plus a 72B (TP=4) experiment and a sensitivity run on a 4x A10 node with 6B/7B models; workloads ShareGPT and variants with doubled input or output length, Poisson arrivals; SLO TTFT 10 s, TBT 100 ms; baselines ServerlessLLM, ServerlessLLM+ (oracle SJF), MuxServe. Headline: "2-2.5x higher request arrival rates or 1.5-9x more goodput"; auto-scaling overhead reduced by 97%; production deployment reduced GPUs "from 1,192 to 213" (82%) — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Fits 1 node, 4x 48 GB: scaled down — the paper's own 4x A10 run (2 prefill + 2 decode instances, 6B/7B models) is the matching configuration; the 16x H800 main results and the 72B run are out.
- Stated limitations: no dedicated limitations/discussion section was found by text search; the paper states MuxServe "refuses to place more than two models on the same GPU due to insufficient GPU memory capacity" as the limit of multiplexing — [Aegaeon PDF](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf)
- Follow-ups: Prism (OSDI'26) addresses the same multi-LLM sharing problem with a different mechanism (GPU memory ballooning) — [arXiv 2505.04021](https://arxiv.org/abs/2505.04021).

**D5. Prism — "Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning" (OSDI'26, venue verified)**
- Links: paper [arXiv 2505.04021](https://arxiv.org/abs/2505.04021); code [ovg-project/kvcached](https://github.com/ovg-project/kvcached).
- Problem: dedicated GPUs are under-used for long-tail models in multi-LLM services; Prism shares GPU memory across models on demand — [arXiv 2505.04021](https://arxiv.org/abs/2505.04021)
- Setup: 58 LLMs (43 of 1-3B, 8 of 4-8B, 3 of 9-30B, 4 of 31-70B); traces Hyperbolic and Arena-Chat (public availability not stated in the paper), Novita AI data in the motivation; hardware 4 nodes x 8 H100-80GB (32 GPUs), NVLink, 100 Gbps; baselines static partition, MuxServe++, QLM, ServerlessLLM. Headline: up to 3.3x TTFT and 2x TPOT SLO attainment, more than 2x cost reduction or 3.5x more requests — [arXiv 2505.04021](https://arxiv.org/abs/2505.04021)
- Fits 1 node, 4x 48 GB: scaled down — 51 of the 58 models are <=8B and fit; 32 H100s reduce to 4 GPUs; the two traces may not be obtainable.
- Stated limitations: no dedicated section; caveats in text: eviction uses an "empirical threshold", placement is greedy, the non-NVLink fallback "takes at most a few seconds for <70B models" — [arXiv 2505.04021](https://arxiv.org/abs/2505.04021)
- Follow-ups: none identified (published 2026).

**D6. DeepServe — "Serverless Large Language Model Serving at Scale" (USENIX ATC'25, venue verified)**
- Links: paper [arXiv 2501.14417](https://arxiv.org/abs/2501.14417); code: none (in-house FlowServe engine).
- Problem: an industry description of Huawei Cloud's serverless LLM platform: a request-job-task abstraction, scheduling across PD-disaggregated and PD-colocated instances, and fast scaling (pre-warmed pods, DRAM pre-loading, NPU-fork) "to scale up to 64 instances in seconds" — [arXiv 2501.14417](https://arxiv.org/abs/2501.14417)
- Setup: runs "on a large Ascend NPU cluster"; PD experiments use "an internal trace (roughly 2K input with 200 output)" with TP=4 and RPS 0.2-1.2 — [arXiv 2501.14417](https://arxiv.org/abs/2501.14417)
- Fits 1 node, 4x 48 GB: no (Ascend NPUs, in-house engine, internal traces).
- Stated limitations: "Accurately predicting decode lengths with minimal overhead remains a challenging research problem."; "we leave further accuracy improvements to future work" — [arXiv 2501.14417](https://arxiv.org/abs/2501.14417)

**D7. SpotHedge / SkyServe — "Serving AI Models on Spot Instances" (EuroSys'25, venue verified)**
- Links: paper [arXiv 2411.01438](https://arxiv.org/abs/2411.01438); code [skypilot-org/skypilot](https://github.com/skypilot-org/skypilot).
- Problem: spot GPU replicas are cheap but preempted; mix spot and on-demand replicas across regions and clouds — [arXiv 2411.01438](https://arxiv.org/abs/2411.01438)
- Setup: Llama-2-70B on vLLM (g5.48xlarge, 8x A10G per replica) and OPT-6.7B on SpotServe (g4dn.12xlarge, 4x T4); workloads Chatbot Arena trace, Poisson, MAF; spot availability traces from AWS/GCP; baselines AWS ASG, MArk, AWSSpot, SpotServe. Headline: 43% lower cost than on-demand; P50/P90/P99 latency 2.3x/2.1x/2.1x better; about 22 hours of experiments costing $4.1k — [arXiv 2411.01438](https://arxiv.org/abs/2411.01438)
- Fits 1 node, 4x 48 GB: no for the real experiments (needs cloud spot instances in several regions); the trace-driven simulation part is hardware-independent.
- Stated future work: locality-aware load balancing and policies for heterogeneous accelerators (paraphrase) — [arXiv 2411.01438](https://arxiv.org/abs/2411.01438)
- Follow-ups: SkyWalker modifies SkyServe to implement its baselines — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095).

#### E. LoRA-adapter serving

**E1. dLoRA — "Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving" (OSDI'24, venue verified)**
- Links: paper [OSDI'24 PDF](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf); code: no repository link found in the paper text.
- Problem: merged-adapter inference is fast but forces per-adapter queuing; unmerged inference batches across adapters but is slower; replicas become load-imbalanced under skewed adapter popularity — [OSDI'24 PDF](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
- Setup: "a four-node GPU cluster, each with eight NVIDIA A800 80GB GPUs, i.e., 32 GPUs in total", 200 Gbps InfiniBand; Llama-2-7B/13B/70B with rank-8 adapters; ShareGPT lengths with arrival patterns from the Microsoft Azure function traces 2019 (MAF1) and 2021 (MAF2), described as "widely used as the proxy of the LLM inference traces"; baselines vLLM, HuggingFace PEFT, S-LoRA. Headline: throughput up to 57.9x over vLLM and 26.0x over PEFT; up to 1.8x lower average latency than S-LoRA — [OSDI'24 PDF](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
- Fits 1 node, 4x 48 GB: scaled down — Llama-2-7B/13B replicas on 4 GPUs instead of 32; the cross-replica migration part shrinks accordingly.
- Stated limitations: no limitations or future-work section was found by text search.
- Follow-ups: Toppings (ATC'25) and Chameleon (MICRO'25) both target multi-adapter serving and compare against S-LoRA — [arXiv 2401.11240](https://arxiv.org/abs/2401.11240), [arXiv 2411.17741](https://arxiv.org/abs/2411.17741).

**E2. Toppings — "CPU-Assisted, Rank-Aware Adapter Serving for LLM Inference" (USENIX ATC'25, venue verified; arXiv name "CaraServe", arXiv version read)**
- Links: paper [arXiv 2401.11240](https://arxiv.org/abs/2401.11240); code: no repository link found in the arXiv text.
- Problem: loading an adapter onto the GPU on demand adds a cold start (the paper measures 10%, 16% and 20% of serving time at 3, 6 and 9 RPS), and adapters of different rank batched together slow each other; uses CPU prefill during loading and a rank-aware scheduler — [arXiv 2401.11240](https://arxiv.org/abs/2401.11240)
- Setup: Llama2-7B on A10 (24 GB), Llama2-13B on 2x A10, Llama2-70B on 4x A100 (80 GB); synthetic workloads and scaled production workloads following the Microsoft Azure Function (MAF) trace with Poisson arrivals; baselines Cached (oracle), OnDmd, S-LoRA. Headline: average request latency sped up "by up to 1.4x" and "SLO attainment of up to 99%" — [arXiv 2401.11240](https://arxiv.org/abs/2401.11240)
- Fits 1 node, 4x 48 GB: yes for 7B/13B (needs many CPU cores; the paper mentions instances with "one A10 GPU and 128 vCPU cores"); 70B dropped.
- Stated limitations: "In future work, we plan to leverage resource disaggregation to address the scalability issue." (the CPU-assisted design depends on CPU cores per GPU) — [arXiv 2401.11240](https://arxiv.org/abs/2401.11240)
- Follow-ups: Chameleon — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741).

**E3. Chameleon — "Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments" (MICRO'25 — search-level venue confirmation)**
- Links: paper [arXiv 2411.17741](https://arxiv.org/abs/2411.17741); code: no repository link found in the paper text (implemented "on top of the open-source S-LoRA").
- Problem: many-adapter serving pays adapter loading cost on the critical path and suffers head-of-line blocking; Chameleon caches popular adapters in otherwise idle GPU memory and uses a non-preemptive multi-queue scheduler — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741)
- Setup: "most of our experiments on a server equipped with an A40 NVIDIA GPU" (48 GB) and an AMD EPYC 9454; scalability runs on an A100 configured with 24/48/80 GB; multi-GPU runs on four A100-80GB; models mainly Llama-7B, also Llama-13B and Llama-30B; request lengths from the Azure production trace (Splitwise), Poisson arrivals; additional traces WildChat-1M and LMSYS-Chat-1M; baselines S-LoRA and μ-Serve. Headline: under high load "reduces the P99 and P50 TTFT latencies by 80.7% and 48.1%" over the baseline "while improving the throughput by 1.5x" — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741)
- Fits 1 node, 4x 48 GB: yes (the main testbed is a single 48 GB A40).
- Stated limitations: none extracted (no limitations section found by text search).
- Follow-ups: none identified.

#### F. Cluster management and simulation

**F1. DynamoLLM — "Designing LLM Inference Clusters for Performance and Energy Efficiency" (HPCA'25 per the Azure trace README)**
- Links: paper [arXiv 2408.00741](https://arxiv.org/abs/2408.00741); code: no link found.
- Problem: energy-efficient cluster management (instance count, model parallelism, GPU frequency) under latency SLOs — [arXiv 2408.00741](https://arxiv.org/abs/2408.00741)
- Setup: Llama2-70B primarily (also Mixtral, Falcon, BLOOM, Llama3-70B); one-week Azure Coding and Conversation traces (2024 dataset); H100 servers (8 GPUs each; 11-12 servers in cluster runs) plus large-scale simulation; baselines SinglePool, MultiPool, ScaleInst, ScaleShard, ScaleFreq. Headline: 53% energy, 38% operational carbon and 61% cost savings — [arXiv 2408.00741](https://arxiv.org/abs/2408.00741)
- Fits 1 node, 4x 48 GB: no (70B-class models on ~90 H100s); only the frequency/parallelism profiling idea could be repeated at small scale.
- Stated caveats: output-length misprediction can cause SLO misses; savings shrink at high load; only tensor parallelism is considered (paraphrase) — [arXiv 2408.00741](https://arxiv.org/abs/2408.00741)

**F2. Vidur — "A Large-Scale Simulation Framework for LLM Inference" (MLSys'24, venue verified)**
- Links: paper [arXiv 2405.05465](https://arxiv.org/abs/2405.05465); code [microsoft/vidur](https://github.com/microsoft/vidur).
- Problem: finding the cheapest deployment configuration (scheduler, batch size, parallelism, replica count) by real runs costs tens of thousands of GPU hours — [arXiv 2405.05465](https://arxiv.org/abs/2405.05465)
- Setup: LLaMA2-7B/70B, InternLM-20B, Qwen-72B; workloads LMSYS-Chat-1M, Arxiv-Summarization, Bilingual-Web-Book; operator profiling on A100-80GB and H100; simulation on a 96-core CPU; validated against a vLLM fork with vLLM, Orca+ and Sarathi-Serve schedulers. Headline: <9% latency error; configuration search in about 1 CPU hour versus an estimated 42K GPU hours (about $218K) — [arXiv 2405.05465](https://arxiv.org/abs/2405.05465)
- Fits 1 node, 4x 48 GB: yes for simulation (CPU-only); profiling a new GPU type (A6000) is required and fidelity on it was not evaluated in the paper.
- Stated caveats: higher error for the 7B model (CPU overhead), error grows near capacity, only synchronous pipeline parallelism supported (paraphrase) — [arXiv 2405.05465](https://arxiv.org/abs/2405.05465)
- Follow-ups: QoServe's baselines are Sarathi-Serve scheduler variants — [arXiv 2503.22562](https://arxiv.org/abs/2503.22562).

### Inferences
- Papers whose main testbed matches or is weaker than one node with 4x 48 GB PCIe GPUs: VTC (single A10G / A100), Chameleon (single A40 48 GB), Toppings 7B/13B (A10), QoServe (1-2 A100, 8B/7B), Learning-to-Rank with Llama-3-8B, Sarathi-Serve with Mistral-7B/Yi-34B, FastServe single-GPU experiments, Llumnix (A10 24 GB PCIe, but 16 of them), ServerlessLLM (A5000 / A40), Aegaeon's 4x A10 sensitivity run, SkyWalker's replica unit (one L4).
- Papers whose mechanism depends on hardware the target node lacks: BlitzScale (multi-host RDMA), DistServe/LoongServe/MuxWise main results (8-32 A100/A800 with NVLink), Prism and LMetric (16-32 H100/H20), DynamoLLM, Splitwise hardware runs, DeepServe (Ascend NPU), SpotHedge (cloud spot markets).
- Baseline reuse indicates which setups later papers treat as reference points: vLLM appears as a baseline in nearly every paper; ServerlessLLM is a baseline in four later top-venue papers; Sarathi-Serve/chunked prefill in five; S-LoRA in three LoRA papers; DistServe or PD-disaggregated SGLang in three.
- The 80B+ models in 2024 papers (OPT-175B, Falcon-180B, BLOOM-176B) are replaced in 2026 papers by MoE models (Qwen3-30B-A3B, Qwen3-235B-A22B) and Llama-3.x 8B/70B.

### Gaps
- Not extracted (title confirmed only): Libra and FlexLLM (NSDI'26), "Efficient LLM Serving on Commodity GPU Clusters with Data-Reduced Cross-Instance Orchestration" and "Revisiting Pipeline Parallelism for LLM Serving" (OSDI'26), BatchGen, StriaTrace, Weaver, Torpor, μ-Serve, DeltaZip, Shift Parallelism, XY-Serve. BlendServe (ASPLOS'26, offline batch) was only skimmed: it synthesizes workloads from WildChat, ShareGPT, Azure-Trace, BurstGPT, OpenVid and MMLU because "there is no open-sourced trace available for offline batch inference" — [arXiv 2411.16102](https://arxiv.org/abs/2411.16102).
- "No repository link found in the paper text" (FastServe, dLoRA, Aegaeon, QoServe, Toppings, Chameleon, SkyWalker, DynamoLLM) does not prove no code exists; GitHub was not searched separately.
- MuxWise's workload traces (its Table 1) were not extracted. QoServe and Toppings and Bullet were read in their arXiv versions; camera-ready numbers may differ.
- Follow-up/crowdedness lists are built only from baselines and related-work sections of the papers read here; no citation-graph search (Semantic Scholar/Google Scholar) was run, so arXiv-only 2025-2026 follow-ups are largely missing.
- For several papers no limitations section exists; "none found by text search" means a keyword search over extracted PDF text, not a full read.

## Key Question 3: Which public workload traces exist, what does each contain, and which papers use them?

### Takeaway
Public LLM-serving traces carry timestamps and token counts but no prompt text (Azure 2023/2024, BurstGPT, Mooncake FAST'25, Alibaba Bailian/Qwen), while the conversation datasets that carry text (ShareGPT, LMSYS-Chat-1M, WildChat) are used only for length distributions and are paired with Poisson/Gamma arrivals or with Azure Functions (MAF) arrival patterns as a proxy. Several papers state that no public trace exists for their specific setting (serverless LLM, offline batch).

### Cited Findings

Traces with arrival timestamps:

- Azure LLM inference traces: AzureLLMInferenceDataset2023 is "a sample of two LLM inference services in Azure containing the input and output tokens" (released with Splitwise, ISCA'24); AzureLLMInferenceDataset2024 is "a longer one week sample of two LLM inference services" (released with DynamoLLM, HPCA'25); the repository also lists a 2025 multimodal trace — [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset). The two services are coding and conversation; prompt content is not included ("we do not have visibility into the content of the prompt") — [Splitwise, arXiv 2311.18677](https://arxiv.org/abs/2311.18677)
  - Used by: Splitwise and DynamoLLM (origin) — [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset); BlitzScale (AzureCode, AzureConv) — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246); QoServe — [arXiv 2503.22562](https://arxiv.org/abs/2503.22562); Bullet (Azure-Code) — [arXiv 2504.19516](https://arxiv.org/abs/2504.19516); Chameleon (lengths) — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741); BlendServe — [arXiv 2411.16102](https://arxiv.org/abs/2411.16102); JITServe says arrivals follow "Microsoft's real-world LLM serving trace, scaled" — [arXiv 2504.20068](https://arxiv.org/abs/2504.20068)
- BurstGPT (KDD'25, venue search-level): traces of Azure OpenAI GPT services at a regional provider, 213 days, 10.31M traces; fields are timestamp, request and response token counts, service type (API vs Conversation) and model (ChatGPT vs GPT-4); no prompt text; failure records included in the raw version — [arXiv 2401.17644](https://arxiv.org/abs/2401.17644); data at [HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT). Findings: conversation traffic is periodic, API traffic bursty and aperiodic, burstiness fits a Gamma process, ChatGPT response lengths are bimodal; the paper's own serving experiments use Llama-2-13b/7b on vLLM on A800 and A6000 servers — [arXiv 2401.17644](https://arxiv.org/abs/2401.17644). Note: per-category counts in the paper sum to about 10.10M, not 10.31M (internal inconsistency) — [arXiv 2401.17644](https://arxiv.org/abs/2401.17644)
  - Used by: Llumnix — [arXiv 2406.03243](https://arxiv.org/abs/2406.03243); BlitzScale — [arXiv 2412.17246](https://arxiv.org/abs/2412.17246); BlendServe — [arXiv 2411.16102](https://arxiv.org/abs/2411.16102)
- Mooncake FAST'25 traces (Moonshot AI / Kimi): conversation_trace.jsonl (12,031 requests), toolagent_trace.jsonl (23,608), synthetic_trace.jsonl (3,993), about one hour each; fields timestamp (ms), input_length, output_length, hash_ids (hashes of 512-token prefix blocks, which allow prefix-cache hit simulation); no text — [Mooncake FAST25-release](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release)
  - Used by: LMetric (ToolAgent trace) — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf)
- Alibaba Bailian / Qwen usage traces (ChatBot and Agent), released with LMetric: "all our traces can be found at" the repository; a third trace (Coder) used in the paper is not released — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf); data at [qwen-bailian-usagetraces-anon](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)
- ServeGen (NSDI'26): characterization of Alibaba Cloud Model Studio over four months, 3.54B requests, 12 models (language, multimodal, reasoning); releases a workload generator with "parameterized and sanitized data instead of full data samples" — [arXiv 2505.09999](https://arxiv.org/abs/2505.09999); code [alibaba/ServeGen](https://github.com/alibaba/ServeGen). Reported findings include: arrivals are bursty (CV>1) and not fit by a single process; rates shift over time; input lengths fit Pareto+LogNormal mixtures and outputs Exponential; a few clients dominate (29 of 2,412); reasoning workloads are longer and bimodal. Case study: with Qwen2.5-14B on 2x A100-80GB vLLM instances, provisioning from a naive ShareGPT+Poisson workload under-provisions by 50% — [arXiv 2505.09999](https://arxiv.org/abs/2505.09999)
- Azure Functions traces 2019/2021 (MAF1/MAF2; not LLM traces) are used for arrival patterns: dLoRA calls them "widely used as the proxy of the LLM inference traces" — [OSDI'24 PDF](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf); also ServerlessLLM — [arXiv 2401.14351](https://arxiv.org/abs/2401.14351), HydraServe — [arXiv 2502.15524](https://arxiv.org/abs/2502.15524), Toppings — [arXiv 2401.11240](https://arxiv.org/abs/2401.11240), SpotHedge — [arXiv 2411.01438](https://arxiv.org/abs/2411.01438)
- Not public or availability not stated: Hyperbolic and Arena-Chat multi-model traces (Prism) — [arXiv 2505.04021](https://arxiv.org/abs/2505.04021); Bailian Coder trace (LMetric) — [OSDI'26 PDF](https://www.usenix.org/system/files/osdi26-zhang-dingyan.pdf); Huawei internal trace (DeepServe) — [arXiv 2501.14417](https://arxiv.org/abs/2501.14417)

Conversation datasets with text, used for length distributions (no usable arrival process):

- LMSYS-Chat-1M: 1,000,000 conversations with 25 LLMs from about 210K unique IPs, collected April-August 2023 from the Vicuna demo and Chatbot Arena; fields are conversation ID, model name, conversation text (OpenAI JSON format), language tag, moderation tag; the dataset card does not mention timestamps; access is gated behind a license agreement that forbids redistribution — [HF dataset card](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
  - Used by: Learning-to-Rank — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792); Vidur — [arXiv 2405.05465](https://arxiv.org/abs/2405.05465); JITServe — [arXiv 2504.20068](https://arxiv.org/abs/2504.20068); Chameleon — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741). Chatbot Arena traces (with client IDs) are used by VTC (27 clients, 10 minutes) — [arXiv 2401.00588](https://arxiv.org/abs/2401.00588), SpotHedge — [arXiv 2411.01438](https://arxiv.org/abs/2411.01438) and SkyWalker — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095)
- ShareGPT (e.g. [shibing624/sharegpt_gpt4](https://huggingface.co/datasets/shibing624/sharegpt_gpt4), [RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)): used for chatbot length distributions by Sarathi-Serve, DistServe, Llumnix, ServerlessLLM, dLoRA, FastServe, Learning-to-Rank, HydraServe, QoServe, Aegaeon, LoongServe, Bullet — see each paper's record in Key Question 2; LoongServe gives its sequence-length range as 4-2.3K tokens — [arXiv 2404.09526](https://arxiv.org/abs/2404.09526)
- WildChat (1M): described by SkyWalker as "one million multi-turn conversations with demographic metadata such as state, country, and hashed IP address" — [arXiv 2505.24095](https://arxiv.org/abs/2505.24095); also used by Chameleon — [arXiv 2411.17741](https://arxiv.org/abs/2411.17741) and BlendServe — [arXiv 2411.16102](https://arxiv.org/abs/2411.16102)
- Task datasets used as workload classes: HumanEval (code) and LongBench (summarization) in DistServe and HydraServe — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670), [arXiv 2502.15524](https://arxiv.org/abs/2502.15524); arxiv_summarization in Sarathi-Serve, Vidur, Bullet — [arXiv 2403.02310](https://arxiv.org/abs/2403.02310); Alpaca in FastServe, Learning-to-Rank, JITServe — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792); L-Eval and LV-Eval (up to 497.3K tokens) in LoongServe — [arXiv 2404.09526](https://arxiv.org/abs/2404.09526)

Statements about missing traces:

- "Since there are no publicly available LLM serverless inference workloads, we use Azure Serverless Trace" — [ServerlessLLM, arXiv 2401.14351](https://arxiv.org/abs/2401.14351)
- "To the best of our knowledge, there is no open-sourced trace available for offline batch inference." — [BlendServe, arXiv 2411.16102](https://arxiv.org/abs/2411.16102)
- "characterizing prefix caching requires access to the content of requests which we currently opt out of"; the analysis "does not yet span long enough for longitudinal studies" — [ServeGen, arXiv 2505.09999](https://arxiv.org/abs/2505.09999)

### Inferences
- The common evaluation recipe in 2024 papers is "ShareGPT (or LMSYS) lengths + Poisson/Gamma arrivals"; ServeGen's case study reports that this recipe under-provisions by 50% relative to production-like workloads, and 2025-2026 papers (BlitzScale, QoServe, LMetric, JITServe, Bullet) shift to Azure 2024, BurstGPT, Mooncake or Bailian traces.
- Only the Mooncake and (per LMetric) the Bailian traces carry prefix-hash information that lets a router/caching evaluation be replayed without text; the Azure and BurstGPT traces cannot reproduce prefix-sharing structure.
- No public trace located here carries per-request SLO class, tenant identity at scale (beyond Chatbot Arena user IDs and ServeGen's parameterized clients), multi-model popularity (Prism's traces are not stated as public), or adapter popularity (LoRA papers synthesize it).

### Gaps
- The field layout and size of the Alibaba Bailian/Qwen trace repository were not read from the repository itself.
- Whether Prism's Hyperbolic/Arena-Chat traces and SkyWalker's code are released was not checked beyond the paper text.
- The Azure 2025 multimodal trace's contents and the papers using it were not examined.
- BurstGPT's KDD'25 venue and LMSYS-Chat-1M's absence of timestamps are each supported by one indirect source (search summary; dataset card silence) and should be treated as unverified.

## Key Question 4: Which problems do several papers name as still open?

### Takeaway
Problems that at least two papers explicitly leave open are: scheduling without knowing output length (prediction accuracy), policy parameters that must be tuned to the workload, scheduling and scaling for prefill/decode-disaggregated deployments (including fault tolerance and live scaling), preemption, fairness beyond a single engine, heterogeneous or multi-model replicas, and the lack of public traces for specific settings.

### Cited Findings

- Unknown output length / predictor accuracy:
  - "Accurately predicting decode lengths with minimal overhead remains a challenging research problem." — [DeepServe, arXiv 2501.14417](https://arxiv.org/abs/2501.14417)
  - "The ranking metric Kendall's Tau still has limitations in reflecting the performance of end-to-end tasks."; a gap to the Oracle remains — [Learning to Rank, arXiv 2408.15792](https://arxiv.org/abs/2408.15792)
  - Length-predictor effectiveness is "contingent upon both the workload and the accuracy of predictions" — [VTC, arXiv 2401.00588](https://arxiv.org/abs/2401.00588)
  - "Under significant workload distribution shifts or poor pattern matching, early scheduling decisions may be suboptimal... Persistent shifts may require retraining predictors or incorporating explicit fallback policies." — [JITServe, NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf)
  - Output-length misprediction can cause SLO misses (paraphrase) — [DynamoLLM, arXiv 2408.00741](https://arxiv.org/abs/2408.00741)
- Workload-dependent parameters and policies left untuned or heuristic:
  - Dynamically varying the token budget "based on workload characteristics... We leave this exploration for future work." — [Sarathi-Serve, arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
  - The scaling "policy depends heavily on workload characteristics, which we leave as future work." — [BlitzScale, arXiv 2412.17246](https://arxiv.org/abs/2412.17246)
  - "There could be a lot of heuristics to explore for setting the virtual usage" — [Llumnix, arXiv 2406.03243](https://arxiv.org/abs/2406.03243)
  - "we cannot afford to sweep all possible configurations"; "there do exist cases where multiplication may fail due to extremely skewed KV$ accesses" — [LMetric, arXiv 2603.15202](https://arxiv.org/abs/2603.15202)
  - Request-characteristic-aware routing (e.g. by prompt length) is listed as an extension — [SkyWalker, arXiv 2505.24095](https://arxiv.org/abs/2505.24095)
- Prefill/decode disaggregation: comparison, scheduling, scaling, fault tolerance:
  - "We leave a quantitative comparison between Sarathi-Serve and disaggregation-based solutions for future work." — [Sarathi-Serve, arXiv 2403.02310](https://arxiv.org/abs/2403.02310)
  - "the dependency between prefill and decoding instances introduces the risk of fault propagation"; "We leave both as future work." — [DistServe, arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
  - "Designing safe and efficient failure recovery is out of scope for our paper." — [Splitwise, arXiv 2311.18677](https://arxiv.org/abs/2311.18677)
  - "Efficient scheduling for PD-disaggregated serving is beyond the scope of this paper." — [LMetric, arXiv 2603.15202](https://arxiv.org/abs/2603.15202)
  - "Live scaling a decode instance in PD disaggregation without interference is impossible" — [BlitzScale, arXiv 2412.17246](https://arxiv.org/abs/2412.17246)
  - "When the SLO target is loose or absent, such as in offline serving, MuxWise has no opportunity to outperform baselines"; "MuxWise is a single-instance optimization" — [MuxWise, arXiv 2504.14489](https://arxiv.org/abs/2504.14489)
- Preemption:
  - "we leave an investigation on preemption as an orthogonal future work" — [VTC, arXiv 2401.00588](https://arxiv.org/abs/2401.00588)
  - "DistServe does not implement advanced runtime policies like preemption"; FCFS "can lead to a 'convoy effect'" — [DistServe, arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
  - (FastServe, NSDI'26, is the paper in this set that takes token-level preemption as its topic — [NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf))
- Fairness and multi-tenant priorities beyond one engine:
  - "We believe our work can be extended to ensure fairness of resources across the cluster"; concurrent loading "with a fairness guarantee is planned for future work" — [ServerlessLLM, arXiv 2401.14351](https://arxiv.org/abs/2401.14351)
  - Cost formulations "cannot accurately reflect the actual LLM serving cost" — [VTC, arXiv 2401.00588](https://arxiv.org/abs/2401.00588)
  - "Currently, Llumnix supports two priority classes, high and normal" — [Llumnix, arXiv 2406.03243](https://arxiv.org/abs/2406.03243)
  - "multi-tenant settings with conflicting SLO priorities" named as a case where early decisions may be suboptimal; all-or-nothing goodput "does not capture scenarios where near-miss completions still provide partial utility" — [JITServe, NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf)
- Heterogeneous hardware, multiple model types, global-vs-local scheduling:
  - "We plan to extend Llumnix to support multiple model types in future work"; "We also plan to explore the interplay between the global scheduling across instances" and local scheduling — [Llumnix, arXiv 2406.03243](https://arxiv.org/abs/2406.03243)
  - Heterogeneous accelerators: "the prefix-aware routing and overall load balancing policies remain as an open question" — [SkyWalker, arXiv 2505.24095](https://arxiv.org/abs/2505.24095); a policy for heterogeneous accelerators is listed as future work (paraphrase) — [SpotHedge, arXiv 2411.01438](https://arxiv.org/abs/2411.01438)
  - "For more complex deployments, such as multi-node, highly parallel, multi-replica, or mixed-GPU environments, runtime profiling may become necessary" — [JITServe, NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf)
  - "BlitzScale currently focuses on instance-level scaling" — [BlitzScale, arXiv 2412.17246](https://arxiv.org/abs/2412.17246)
- Placement of models/checkpoints and colocation interference:
  - "Optimization of checkpoint placement is considered a separate issue and is not addressed in this paper." — [ServerlessLLM, arXiv 2401.14351](https://arxiv.org/abs/2401.14351)
  - Pipeline pieces "may lead to worker colocations across different models", which "can degrade performance" — [HydraServe, arXiv 2502.15524](https://arxiv.org/abs/2502.15524)
  - Placement is greedy and eviction uses an "empirical threshold" — [Prism, arXiv 2505.04021](https://arxiv.org/abs/2505.04021)
- Missing or inadequate public workloads: see the three quoted statements at the end of Key Question 3 (ServerlessLLM, BlendServe, ServeGen) and dLoRA's use of Azure Functions traces as "the proxy of the LLM inference traces" — [OSDI'24 PDF](https://www.usenix.org/system/files/osdi24-wu-bingyang.pdf)
- Integration with newer engine optimizations: LTR's integration with the latest serving optimizations "has still not been fully studied... We will leave them as future work." — [arXiv 2408.15792](https://arxiv.org/abs/2408.15792); the Sarathi-Serve prototype "doesn't have complete feature parity with open-source vLLM" — [arXiv 2403.02310](https://arxiv.org/abs/2403.02310); Bullet leaves extending its latency model to "new attention variants or LoRA adapters" to future work — [arXiv 2504.19516](https://arxiv.org/abs/2504.19516)

### Inferences
- The open items named by the largest number of papers in this set are (i) output-length uncertainty (5 papers), (ii) PD-disaggregation scheduling/scaling/fault tolerance (6 papers), and (iii) workload-dependent parameter tuning (5 papers). This is a count of statements in the papers read, not an assessment of the directions.
- Some 2024 "future work" items were taken up by later papers in the set: Sarathi-Serve's deferred comparison with disaggregation is carried out by MuxWise and Bullet (chunked prefill vs disaggregated vs multiplexed); VTC's and DistServe's deferred preemption is FastServe's topic; ServerlessLLM's remark about missing serverless LLM traces precedes ServeGen and the Azure 2024 release.

### Gaps
- Open problems were collected only from the papers read in Key Question 2; the unread NSDI'26/OSDI'26 papers (Libra, FlexLLM, the commodity-GPU-cluster and pipeline-parallelism papers) may state further ones.
- No survey paper was read in full; the survey "Taming the Titans" ([arXiv 2504.19720](https://arxiv.org/abs/2504.19720)) and its list ([Awesome-LLM-Inference-Serving](https://github.com/zenrran4nlp/Awesome-LLM-Inference-Serving)) were located but their own open-problem lists were not extracted.
- Papers without a limitations section (dLoRA, FastServe, LoongServe, Aegaeon, Chameleon, QoServe) contribute no quotes here.
