# 大模型推理优化与 AI Infra 自学路线图 (LLM Inference & AI Infra Roadmap)

> 本路线图旨在帮助开发者、研究者以及系统工程师，从零开始系统性掌握大模型推理优化与 AI Infra（LLM Inference / AI Infra / MLSys）的核心理论、前沿算法、工程架构与底层算子开发——涵盖显存与算力推导、量化、注意力架构、投机解码、PD 分离与调度、多租户 LoRA 服务、底层算子与硬件等专题。

---

## 目录
1. [阶段一：理论基础与显存/算力推导](#阶段一理论基础与显存算力推导)
   - [1.1 核心概念与推导](#11-核心概念与推导)
   - [1.2 必读文档与入门参考](#12-必读文档与入门参考)
   - [1.3 注意力变体与 KV Cache 的关系](#13-注意力变体与-kv-cache-的关系)
2. [阶段二：核心论文与突破性算法](#阶段二核心论文与突破性算法)
   - [2.1 显存分页与 IO 优化](#21-显存分页与-io-优化)
   - [2.2 量化（Quantization）](#22-量化quantization)
   - [2.3 注意力架构与长上下文](#23-注意力架构与长上下文)
   - [2.4 投机解码（Speculative Decoding）](#24-投机解码speculative-decoding)
   - [2.5 PD 分离与调度架构](#25-pd-分离与调度架构)
   - [2.6 多租户 LoRA 服务](#26-多租户-lora-服务)
   - [2.7 KV Cache 全景深入](#27-kv-cache-全景深入)
3. [阶段三：生产级推理引擎与源码剖析](#阶段三生产级推理引擎与源码剖析)
   - [3.1 核心开源框架](#31-核心开源框架)
4. [阶段四：底层算子与硬件压榨](#阶段四底层算子与硬件压榨)
   - [4.1 Triton 算子编程](#41-triton-算子编程首选进阶路径)
   - [4.2 前沿工业级算子库](#42-前沿工业级算子库deepseek-开源精粹)
   - [4.3 性能分析与 Diagnostics](#43-性能分析与-diagnostics)
   - [4.4 算子学习清单（补全核心算子）](#44-算子学习清单补全核心算子)
   - [4.5 模型编译](#45-模型编译)
   - [4.6 昇腾 CANN 算子](#46-昇腾-cann-算子)
5. [阶段五：核心实战项目与工程级实现](#阶段五核心实战项目与工程级实现)
   - [5.1 项目一：EAGLE 风格 Tree Attention 投机解码器](#51-项目一eagle-风格-tree-attention-投机解码器)
   - [5.2 项目二：nano-vllm 最小推理引擎](#52-项目二nano-vllm-最小推理引擎)
   - [5.3 项目三：MaaS 高并发场景压测与 nsys Profiling 管线](#53-项目三maas-高并发场景压测与-nsys-profiling-管线)
   - [5.4 项目四（选做）：多 LoRA 混批 / INT4 反量化 GEMM](#54-项目四选做多-lora-混批服务-或-int4-在线反量化-gemm)
   - [5.5 项目五：模型编译加速](#55-项目五模型编译加速)
6. [阶段六：训练系统 infra](#阶段六训练系统-infra)
   - [6.1 分布式训练并行](#61-分布式训练并行)
   - [6.2 通信与集群](#62-通信与集群)
   - [6.3 训练框架与显存](#63-训练框架与显存)
   - [6.4 RL 后训练（深入）](#64-rl-后训练深入)
   - [6.5 实战项目：小规模 RL 后训练](#65-实战项目小规模-rl-后训练)
7. [阶段七：前沿选读](#阶段七前沿选读)
   - [7.1 硬件多样性](#71-硬件多样性)
   - [7.2 推理时扩展](#72-推理时扩展)
   - [7.3 多模态推理](#73-多模态推理)
   - [7.4 Agentic Infra](#74-agentic-infra)
   - [7.5 实战项目：最小 Agent Harness](#75-实战项目最小-agent-harness)

---

## 阶段一：理论基础与显存/算力推导

在深入具体框架和算子之前，首先需要建立对硬件瓶颈（Memory-Bound vs Compute-Bound）的量化认知。

### 1.1 核心概念与推导

**入门直觉**：把 GPU 想象成两个工人——一个算得飞快（**算力**）、一个搬货慢吞吞（**带宽**）。Prefill 一次性送来一大摞货（N 个 token），算力忙不过来 → 算力受限（Compute-Bound）；Decode 每次只搬一个 token、却要把整套参数和 KV 都从显存搬一遍，工人大部分时间在等货 → 带宽受限（Memory-Bound）。本节就是把这种“感觉”量化成可算的数字。

* **Roofline Model（算力瓶颈判定）**：
  $$\text{Arithmetic Intensity (算术强度)} = \frac{\text{计算量 (FLOPs)}}{\text{内存访问量 (Bytes)}}$$
  * **Prefill 阶段**：处理 $N$ 个输入 Token。注意力部分计算量约 $O(N^2 \cdot d)$、线性投影约 $O(N \cdot d^2)$，总体前向 FLOPs $\approx 2 \times \text{params} \times N$（2 为矩阵乘的乘累加系数，与 Token 数线性相关）；算术强度高，通常属于 **Compute-Bound（计算密集型）**。
  * **Decode 阶段**：逐字自回归生成，每次仅输入 1 个 Token，参数与 KV Cache 必须逐字节从 HBM/显存搬运至 SRAM，算术强度极低（通常 $<10$），属于 **Memory-Bound（访存受限型）**。
* **KV Cache 显存精准计算公式**（以 FP16/BF16 2 字节为例）：
  $$\text{KV Cache Size (Bytes)} = 2 \times \text{layers} \times (2 \times \text{kv\_heads} \times \text{head\_dim}) \times \text{seq\_len} \times \text{batch\_size}$$

> 算一算：Llama-2-7B（32 层、32 头 MHA、head_dim=128），seq=4K、batch=1、FP16 → KV Cache = 2(字节) × 32(层) × (2×32×128) × 4096 ≈ **2 GB**；到 seq=32K 时膨胀到 ~16 GB，反超模型权重本身（~14 GB）——这就是 decode 阶段显存/带宽成为命门的原因。

> 📝 作业（1.1）：① 不看公式，口算 FP16 下“每层每 token 的 KV 字节 = 2(字节)×2(K,V)×kv_heads×head_dim”，并推广到 GQA；② 解释为何 seq 翻倍时 KV 翻倍、但 decode 单步算术强度几乎不变（仍是带宽瓶颈）。答清即完成。

### 1.2 必读文档与入门参考

**建议顺序**：先跑通 nanoGPT 的最小自回归生成 → 再读 kipply 博文手推显存/FLOPs → 最后用 Wikipedia 补 Roofline 背景。

* [Transformer Inference Arithmetic (kipply's blog)](https://kipp.ly/transformer-inference-arithmetic/)：神级博文，手把手推导 KV Cache 显存占用、FLOPs 以及带宽限制。
* [Roofline Model Concepts (Wikipedia)](https://en.wikipedia.org/wiki/Roofline_model)：了解 Roofline 分析模型的官方理论背景。
* [nanoGPT (by Andrej Karpathy)](https://github.com/karpathy/nanoGPT)：建议先读懂极简自回归生成逻辑，再尝试在上面自行添加 KV Cache 缓存。
* 📺 **配套入门视频**（中文 / 动画，补直觉）：
  * 3Blue1Brown — [How word vectors encode meaning](https://www.youtube.com/watch?v=FJtFZwbvkI4) / [How might LLMs store facts (Deep Learning Ch.7)](https://www.youtube.com/watch?v=9-Jl0dxWQs8)
  * Andrej Karpathy — [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)
  * 李宏毅 — [生成式AI時代下的機器學習(2025) 课程页](https://speech.ee.ntu.edu.tw/~hylee/ml/2025-spring.php)（B站搜“李宏毅”可得；第一講：[YouTube](https://www.youtube.com/watch?v=QLiKmca4kzI)）

> 📝 作业（1.2）：在 nanoGPT 上加最小 KV Cache（decode 不重算历史），跑通生成；并用 kipply 博文方法推一遍其 FLOPs / 显存。跑通 + 推对即完成。

### 1.3 注意力变体与 KV Cache 的关系
KV Cache 公式中的 `kv_heads` 直接对应注意力头的组织方式，不同变体对显存影响巨大：

* **MHA（标准多头注意力）**：`kv_heads = num_heads`，每个头独立存 K/V，KV Cache 最大。
* **MQA（Multi-Query Attention）**：`kv_heads = 1`，所有 query 头共享一组 K/V，KV Cache 最小但质量略损。
* **GQA（Grouped-Query Attention）**：`kv_heads` 介于 1 与 num_heads 之间，在质量与显存间折中——**Llama-2/3、Qwen 等主流模型均采用**，是当前 KV 公式最常见的取值。
* **MLA（Multi-head Latent Attention，DeepSeek-V2/V3）**：将 K/V 压缩进低秩隐向量缓存，相比 MHA 可将 KV Cache 降低约 93%；但结构与标准 FlashAttention / PagedAttention 不直接兼容（详见阶段二），需专用 kernel（如 FlashInfer）配合。

> 推导练习：给定 Llama-3-8B（32 层、8 KV 头、head_dim=128、GQA），分别计算 seq_len=8K / 32K、batch=1 时 FP16 下的 KV Cache 大小，体会 GQA 对长上下文显存的缓解。

> 📝 作业（1.3）：① 一句话说清 MHA / MQA / GQA 在 KV 显存上的取舍；② MLA 凭什么砍掉约 93% KV、却又与标准 FlashAttn 不兼容？答清即完成。

---

## 阶段二：核心论文与突破性算法

掌握近年大模型推理领域的经典与前沿论文，重点理解其“为了解决什么硬件/系统矛盾”而提出。按主题分组如下。

> 注：本节链接已于 2026-08-21 核验（arXiv ID 经 arXiv API、GitHub 仓库经 GitHub API、ORCA 经 DBLP+USENIX、kipp.ly 博文经直连）。ORCA 为 USENIX OSDI '22 会议论文、未上 arxiv，故用其官方会议链接。

### 2.1 显存分页与 IO 优化
| 论文名称与链接 | 核心突破与要点 |
| :--- | :--- |
| [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) | 借鉴 OS 虚拟内存思想，解决 KV Cache 内存碎片化与预分配虚高问题。 |
| [FlashAttention-1](https://arxiv.org/abs/2205.14135) / [2](https://arxiv.org/abs/2307.08691) / [3](https://arxiv.org/abs/2407.08608) | SRAM 分块（Tiling）+ Online Softmax，避免频繁往返 HBM，大幅降低 IO；v3 利用 Hopper 异步数据流。 |

> 📝 作业（2.1）：① PagedAttention 解决的“碎片化”与“预分配虚高”分别指什么？② FlashAttention 的 Tiling 为何能减少 HBM 读写（和算术强度有何关系）？答清即完成。

### 2.2 量化（Quantization）

**入门**：量化＝用更少比特（INT8/INT4/FP8）存权重、激活或 KV，显存和带宽按比例降，代价是些许精度。三条线先记牢：只压权重（GPTQ/AWQ）、权重+激活一起压（LLM.int8/SmoothQuant/FP8）、单独压 KV Cache（KIVI/KVQuant）。

| 论文名称与链接 | 核心突破与要点 |
| :--- | :--- |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | 发现激活中少量 outlier，混合精度分解（outlier 用 FP16、其余 INT8）保精度。 |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 基于二阶信息的逐层权重量化（W4A16），少量校准样本即可。 |
| [AWQ](https://arxiv.org/abs/2306.00978) | 发现“显著权重”通道，按激活尺度缩放保护重要权重，INT4 近乎无损。 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 把激活难度迁移到权重（平滑），实现 W8A8 激活+权重同时量化。 |
| FP8 量化（选读） | Hopper/Blackwell 的 FP8（E4M3/E5M2 格式）权重+激活量化；DeepSeek-V3 报告 §3.3「FP8 Training」给出大规模 FP8 训练→推理实践（[DeepSeek-V3](https://arxiv.org/abs/2412.19437)），衔接阶段四 DeepGEMM。 |
| [KIVI](https://arxiv.org/abs/2402.02750) / [KVQuant](https://arxiv.org/abs/2401.18079) | KV Cache 专用量化，K/V 沿不同轴（K 按通道、V 按 token），直接缓解 Decode 阶段带宽瓶颈（呼应阶段一公式）。 |
| 旋转辅助量化（QuaRot / SpinQuant，选读） | 通过 Hadamard / 旋转打散 outlier，使更低比特量化成为可能。 |

> 📝 作业（2.2）：① W4A16 / W8A8 / FP8 各压的是权重还是激活？精度损失主要来自哪？② 为何 KV Cache 量化能直接缓解 decode 带宽瓶颈？答清即完成。

### 2.3 注意力架构与长上下文

**入门**：注意力的“形状”直接决定 KV 多大——头共享得越多越省显存但质量越折损（MQA < GQA < MHA），MLA 则用低秩压缩另辟蹊径。长上下文靠 RoPE 外推（YaRN）和跨卡分片（Ring Attention）把窗口拉长。

| 论文名称与链接 | 核心突破与要点 |
| :--- | :--- |
| [DeepSeek-V2（MLA）](https://arxiv.org/abs/2405.04434) / [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | **MLA** 将 K/V 压入低秩隐向量，KV Cache 相比 MHA 降 93.3%；但与标准 FlashAttn/PagedAttn 不兼容，催生专用 kernel（FlashInfer）。 |
| [YaRN](https://arxiv.org/abs/2309.00071) | NTK-aware RoPE 外推，低成本扩展上下文长度（配合 PI / NTK 系列）。 |
| Ring Attention / Context Parallelism（选读） | 跨 GPU 分片计算长序列注意力，突破单卡显存上限。 |

> 📝 作业（2.3）：① 头数减半改 GQA，KV 缩多少、质量为何不线性恶化？② YaRN 与朴素 RoPE 外推在“远处衰减”上差在哪？答清即完成。

### 2.4 投机解码（Speculative Decoding）

**入门**：常规 decode 一次只出 1 个 token；投机解码先用“草稿”（小模型 / 草稿头 / 检索）一次猜几个，再让大模型一次并行验证——猜中的直接采纳，省掉多次串行 forward。关键是结果无损、且接受率越高越快。

| 论文名称与链接 | 核心突破与要点 |
| :--- | :--- |
| [Speculative Decoding (Leviathan)](https://arxiv.org/abs/2211.17192) | Draft-Target 拒绝采样，数学证明与大模型分布无损一致。 |
| [EAGLE](https://arxiv.org/abs/2401.15077) / [EAGLE-2](https://arxiv.org/abs/2406.16858) | 放弃独立小模型，在大模型特征空间加极轻量 Draft Head；v2 引入上下文感知动态草稿树。 |
| [Medusa](https://arxiv.org/abs/2401.10774) | 多头并行草稿（每个头预测一个未来 token），无需自回归草稿模型，树形验证。 |
| [LayerSkip](https://arxiv.org/abs/2404.16710) / 自投机 (self-speculative) | 草稿与验证用同一模型（早退出 / 跳层做草稿、全模型验证），无 draft/target 分布失配。 |
| MTP（Multi-Token Prediction，DeepSeek-V3） | 模型自带多 token 预测头，可直接作自投机草稿源——与 DeepSeek 主题高度契合。 |
| [DFlash](https://arxiv.org/abs/2602.06036) / [DSpark](https://arxiv.org/abs/2607.05147) | DFlash 用扩散模型做块级并行草稿；DSpark 用置信度调度的半自回归生成 + 自适应验证——均解决串行草稿在 Decode 阶段的延迟瓶颈。 |
| [MagicDec](https://arxiv.org/abs/2408.11049) / [REST](https://arxiv.org/abs/2311.08252)（选读） | MagicDec 证明中长序列、大 batch 高吞吐下投机解码仍能加速、打破延迟-吞吐权衡；REST 基于语料库检索提供免模型草稿。 |

> 工程要点：① 接受率（acceptance rate）是核心指标，需实测调参；② 拒绝采样的“无损”依赖 draft/target tokenizer 一致与采样实现正确，并非任意接线即无损；③ EAGLE-3 放弃 EAGLE 的特征级预测、改靠扩大训练规模提升接受率，具体实现以 [EAGLE 官方仓库](https://github.com/SafeAILab/EAGLE) 最新版为准（论文 [EAGLE-3](https://arxiv.org/abs/2503.01840)）。

> 📝 作业（2.4）：① 拒绝采样为何“无损”？依赖哪两个前提？② 接受率=0.6 时每步期望采纳约几个 token（a/(1−a)）？③ EAGLE 相比 Medusa 的关键改进？答对 2/3 即完成。

### 2.5 PD 分离与调度架构

**入门**：服务端两件事——把 prefill（算力密集）和 decode（带宽密集）拆到不同机器互不拖累（PD 分离），以及把多请求的 token 级别交错打包（连续批处理 / chunked prefill）把 GPU 喂满。

| 论文名称与链接 | 核心突破与要点 |
| :--- | :--- |
| [DistServe](https://arxiv.org/abs/2401.09670) / [Splitwise](https://arxiv.org/abs/2311.18677) | 将 Compute-Bound 的 Prefill 与 Memory-Bound 的 Decode 解耦部署，消除互相干扰。 |
| [Mooncake（Moonshot/Kimi）](https://arxiv.org/abs/2407.00079) | 生产级 PD 分离 + 全局 KVCache 池，跨节点高效迁移 / 复用 KV（RDMA / NVLink）。 |
| [ORCA（OSDI '22）](https://www.usenix.org/conference/osdi22/presentation/yu) | 迭代级调度（iteration-level scheduling），连续批处理的理论基石。 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) / [Sarathi](https://arxiv.org/abs/2308.16369) | Chunked Prefill（分块预填充），Prefill 与 Decode 混批以打满吞吐、降低 TTFT 抖动。 |
| [Llumnix](https://arxiv.org/abs/2406.03243)（选读） | Decode 请求跨节点迁移做负载均衡，缓解长尾延迟。 |

> 📝 作业（2.5）：① 为什么 prefill / decode 拆机后能同时改善 TTFT 与吞吐？② chunked prefill 比朴素连续批处理额外解决了什么？答清即完成。

### 2.6 多租户 LoRA 服务

**入门**：成百上千个用户各要自己的微调模型，全装进显存不现实——共享一个基座、按请求动态切换/融合各自的 LoRA 适配器，是省显存又保吞吐的关键。

| 论文名称与链接 | 核心突破与要点 |
| :--- | :--- |
| [S-LoRA](https://arxiv.org/abs/2311.03285) | 多 LoRA 混批 + 定制 MBGMM/MBGMV kernel（Triton），单基座服务大量适配器、近乎无损吞吐。 |
| [Punica](https://arxiv.org/abs/2310.18547) / [dLoRA](https://arxiv.org/abs/2404.05182)（选读） | Punica 为多租户 LoRA 服务（批处理不同 adapter 的 kernel）；dLoRA 实为分布式 PEFT 训练（联邦 / 隐私），非服务场景，列此仅作区分。 |

> 📝 作业（2.6）：① 多 LoRA 共享基座时，瓶颈为何在“按 adapter 切换的批处理 kernel”而非显存？② S-LoRA 的 MBGMM / MBGMV 在做什么？答清即完成。

### 2.7 KV Cache 全景深入

KV Cache 是 LLM 推理显存与带宽的枢纽——既是显存大头（长上下文 / 长生成下线性爆炸），又是 decode 带宽瓶颈（每步要读全部 KV）。几乎所有推理优化都围着它转。

* **本质与规模**：自回归 decode 每步只需新 K/V，但注意力要用历史全部 K/V → 缓存它们避免每步重算 prefill。规模见 1.1 公式，长上下文 / 长生成下线性爆炸（长 CoT 模型尤甚）。
* **分页管理 [PagedAttention](https://arxiv.org/abs/2309.06180)**：物理块(block) + 每请求逻辑块表(block table) + 按需分配，借鉴 OS 虚拟内存——消除碎片化与预分配虚高，vLLM 的核心。
* **复用 / 前缀缓存**：多请求共享公共前缀（system prompt / few-shot）的 KV → 省 prefill。[SGLang RadixAttention](https://arxiv.org/abs/2312.07104)（基数树）+ vLLM 自动前缀缓存(APC)。
* **量化**：KV 用更少比特存——[KIVI](https://arxiv.org/abs/2402.02750)（K 按通道 / V 按 token 非对称 2bit）、FP8 KV——直接减 decode 带宽（呼应 2.2）。
* **offload 与跨节点迁移**：显存放不下 → 分层 offload 到 CPU/SSD（[FlexGen](https://arxiv.org/abs/2303.06865) 的 GPU-CPU-SSD 调度）；PD 分离下 KV pool + 跨节点迁移（[Mooncake](https://arxiv.org/abs/2407.00079)，RDMA/NVLink），权衡“重算 vs 搬运”。
* **压缩 [MLA](https://arxiv.org/abs/2405.04434)**：把 K/V 压成低秩 latent 向量缓存，KV 降约 93%（DeepSeek），代价是与标准 attention 不兼容、需专用 kernel(FlashInfer)。
* **kernel 视角**：decode 每步 1 query 读全部 KV → memory-bound 元凶；FlashAttention decode path / FlashInfer decode kernel + paged KV 访问优化此 kernel。
* **长生成的压缩 / 驱逐**：长链式思考使 KV 无界膨胀 → 滑窗 / 注意力 sink / 重要度驱逐（[StreamingLLM](https://arxiv.org/abs/2309.17453)、[H2O](https://arxiv.org/abs/2306.14048)）取舍——牺牲部分精度换显存可控。

> 📝 作业（2.7）：① 给 seq=128K 的请求算 KV 占多少显存；offload 到 CPU 后每步 decode 多一次 KV 搬运，算带宽代价；② MLA 降 KV 93%，“不兼容标准 attention”具体指哪一步要改？③ 长生成下选 StreamingLLM(丢历史) vs 重算，各在什么场景更优？答清即完成。

---

## 阶段三：生产级推理引擎与源码剖析

不要泛泛阅读代码，建议选择主干引擎（优先推荐 vLLM），重点对请求调度、显存块管理和模型执行打断点追踪。

### 3.1 核心开源框架

**怎么选**：入门/通用首选 **vLLM**；结构化输出与复杂控制流看 **SGLang**；生产严苛 SLA 上 **TensorRT-LLM**；CPU/Mac/端侧用 **llama.cpp**；服务 DeepSeek(MLA) 配 **FlashInfer**。

* **vLLM**：生态最健全、最规范的开源推理引擎。
  * [vLLM GitHub 仓库](https://github.com/vllm-project/vllm) | [vLLM 官方文档](https://docs.vllm.ai/)
  * **核心源码文件与模块**：
    * 调度与 Continuous Batching：[`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
    * 显存分页管理：[`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py) / [`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
    * Worker 与 CUDA Graph：[`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py)
    * ⚠️ 注：以上为 V1 引擎路径（2026-08-21 经 GitHub API 核验存在）；vLLM 迭代极快，后续仍可能再迁移，请以仓库当前目录结构为准。
* **SGLang**：复杂场景与结构化推理性能首选。
  * [SGLang GitHub 仓库](https://github.com/sgl-project/sglang) | [SGLang 官方文档](https://docs.sglang.ai/)
  * **重点研读**：RadixAttention（基于基数树的前缀 KV Cache 共享与复用）。
* **EAGLE Official Repo**：
  * [EAGLE GitHub 仓库](https://github.com/SafeAILab/EAGLE)：了解官方 Tree Attention 以及训练/加载 Draft Head 的细节。
* **TensorRT-LLM**：NVIDIA 生产级引擎，与 vLLM 是两条主流路线，极致 kernel 融合与 in-flight batching；适合对延迟/吞吐有严苛 SLA 的生产部署。
  * [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
* **llama.cpp / GGUF / MLX**：覆盖 CPU / Mac / 边缘这一整条部署层级（路线图其余部分偏 GPU 数据中心，此处补齐）。k-quants 与极低显存占用是其强项。
  * [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) | [MLX（Apple Silicon）](https://github.com/ml-explore/mlx)
* **FlashInfer**：生产级注意力 kernel 库，块稀疏注意力 + 专门优化的 MLA kernel，是高效服务 DeepSeek（MLA）的关键依赖，已被 vLLM/SGLang 集成。
  * [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)

**📖 入门路径（每个引擎怎么起步，新手向）**：

* **vLLM**：`pip install vllm` → `vllm serve <model>` 起一个服务 → 打断点跟一个请求 `vllm/v1/core/sched/scheduler.py`→`block_pool.py`→`gpu_model_runner.py` → 改 batch size 看吞吐变化。
* **SGLang**：`pip install sglang` → `python -m sglang.launch_server` 起服务 → 读 RadixAttention 的前缀命中 → 试 regex/JSON 结构化生成。
* **TensorRT-LLM**：用 `trtllm-build` 把 HF 模型编译成 engine → 跑 benchmark 看吞吐 → 读 `cpp/tensorrt_llm/` 下的 attention / MLP plugin。
* **llama.cpp**：`make` 编译 + 下一个 GGUF 量化模型 → `./llama-cli -m model.gguf -p '你好'` 跑推理 → 读 `llama_decode` / KV 管理看极简实现。
* **FlashInfer**：`pip install flashinfer` → 在 vLLM 里设 attention backend=FLASHINFER 跑一下 → 读 `flashinfer/decode.py` 的 paged decode kernel。

  🎧 **延伸播客**：Latent Space — [The Inference Engineering Masterclass (Baseten，推理工程实战)](https://www.latent.space/p/inference-eng)；Practical AI — [Serverless GPUs (ep211)](https://changelog.com/practicalai/211)。

**vLLM / SGLang 内部结构（深入）**：

* **vLLM 架构**（[PagedAttention 论文](https://arxiv.org/abs/2309.06180)）：Engine / Scheduler / Worker / ModelRunner 三层——Scheduler 决定每步跑哪些请求（[`v1/core/sched`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)）；KV 管理 = PagedAttention + [`block_pool` / `kv_cache_manager`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)；调度技巧 = 连续批处理 / 抢占 / chunked prefill / APC；加速 = CUDA Graph / 投机解码 / 多模态 / LoRA。
* **SGLang 架构**（[论文](https://arxiv.org/abs/2312.07104)）：RadixAttention（基数树前缀 KV 复用）、压缩 FSM 结构化生成、overlap 调度、内置投机解码。
* **怎么读源码**：vLLM 从 `vllm/v1/core/sched/scheduler.py` 打断点，跟踪一个请求 waiting→running→释放 KV 块；SGLang 从 RadixAttention 前缀命中看起。
* **易用可扩展 API**：两引擎都对外暴露统一推理 API + 多后端 / 多模态 / LoRA 扩展点——JD 的“易用可扩展 API、支持多框架接入”即此（vLLM-Ascend 就是把后端扩展到昇腾）。

> 📝 作业（3.1）：克隆 vLLM，在 `vllm/v1/core/sched/scheduler.py` 与 `block_pool.py` 打断点跑一次 serve，跟踪一个请求从 waiting→running→释放 KV 块的过程；再用一句话说清 vLLM / SGLang / TRT-LLM 各自适合什么。跑通 + 说清即完成。

---

## 阶段四：底层算子与硬件压榨

如果希望深入至 CUDA / Triton 算子层，直接压榨 GPU Tensor Core 与显存带宽能力。

### 4.1 Triton 算子编程（首选进阶路径）

**入门**：Triton 让你用 Python 写 GPU kernel，不必碰 CUDA C，性能接近手写。建议从向量加、融合 softmax 起步，再挑战 FlashAttention 极简版。

* [OpenAI Triton GitHub](https://github.com/openai/triton) | [Triton 官方教程文档](https://triton-lang.org/main/getting-started/tutorials/index.html)
* **进阶实战顺序**：
  1. Vector Addition（向量加法）
  2. Fused Softmax（融合 Softmax）
  3. FlashAttention-2 极简实现

> 📝 作业（4.1）：用 Triton 写一个 fused softmax kernel，对比 PyTorch baseline 的正确性与耗时；再读 FlashAttention-2 极简实现。跑通且对齐即完成。

### 4.2 前沿工业级算子库（DeepSeek 开源精粹）
* [DeepSeek DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)：极轻量高性能 FP8 GEMM 矩阵乘法算子库。
* [DeepSeek DeepEP](https://github.com/deepseek-ai/DeepEP)：专为 MoE 架构设计的节点间全连接通信库。

> 衔接：DeepGEMM 的 FP8 GEMM 属“反量化-融合 GEMM”，需先理解阶段二的 FP8 量化（E4M3/E5M2）与 DeepSeek-V3 的 FP8 实践，再回到这里看它如何把反量化融进矩阵乘。

> 📝 作业（4.2）：① DeepGEMM 的“反量化-融合”指把哪一步融进了 GEMM？② DeepEP 解决 MoE 的哪个通信瓶颈（与 TP / EP 何关）？答清即完成。

### 4.3 性能分析与 Diagnostics
* [NVIDIA Nsight Systems (nsys) 官网](https://developer.nvidia.com/nsight-systems)
* [vLLM Performance Profiling Guide](https://docs.vllm.ai/en/latest/dev/profiling.html)：教你如何抓取 Profiling 轨迹并定位算子开销。

> 📝 作业（4.3）：用 nsys 抓一次 vLLM serve 的 trace，定位 decode 阶段占比最大的 kernel；解释它为何是访存瓶颈。抓到 + 说清即完成。

### 4.4 算子学习清单（补全核心算子）
阶段四原清单偏 GEMM / 通信。推理中以下算子同样关键，建议逐个吃透：

* **解码态单 token 注意力 kernel**：Decode 阶段每步仅 1 个 query token 却要遍历全部 KV——这正是 Memory-Bound 的元凶，是优化的主战场（FlashAttention decode path、FlashInfer decode kernel）。
* **反量化-融合 GEMM（W4A16 / W8A8 / FP8）**：权重在线反量化并融进 GEMM，避免单独反量化 pass，是低比特推理的核心 kernel。
* **MoE 分组 GEMM / 专家路由 kernel**：配合阶段二 EP 与 DeepEP 理解 all-to-all 路由 + 不均衡分组 GEMM。
* **融合元素算子**：RMSNorm、SiLU×mul（SwiGLU）、RoPE 常融合成单 kernel，减少 kernel launch 与访存。
* **采样 kernel（top-k / top-p / 温度）**：在 GPU 侧完成采样，避免 GPU↔CPU 往返拖慢 Decode。
* **MMA / Tensor Core 基础**：理解 wgmma/mma 指令与 TMA（Tensor Memory Access），是写出高效 GEMM 的前提（Hopper / Blackwell）。

> 📝 作业（4.4）：① decode 单 token 注意力为何是 Memory-Bound 的“元凶”？② 反量化-融合 GEMM 比先反量化再 GEMM 省在哪？答清即完成。

### 4.5 模型编译

**入门**：手写算子太重，编译器替你做融合、autotune、图捕获——把 eager 的零散 op 编成高效执行图。

* [torch.compile / Inductor](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)：PyTorch 原生 JIT——**Dynamo** 抓 Python 帧做图捕获（遇动态会 graph break 退回 eager）；**AOTAutograd** 做融合规划；**Inductor** 生成 Triton / C++ 代码。
* **CUDA Graph**：把一串 kernel 录成图、单次 launch，消除 CPU launch 开销——decode 每步几十个 kernel，收益最大（vLLM 关键加速，呼应 3.1）。局限：动态 shape 受限，需静态张量池。
* [XLA / OpenXLA](https://github.com/openxla/xla)：**HLO** IR 上做图级融合 + 算子重排，JAX 原生 / PyTorch-XLA；阿里 JD 的 XLA / MLIR，异构编译的中间表示基础。
* **MLIR**：多级 IR，XLA / Triton / IREE 等编译栈的共同底座，让“一份 IR 多后端”。
* **kernel autotune**：Triton `@autotune` 扫 block / tile / num_warps 配置择优，自动适配不同 GPU（呼应 4.1）。
* [Cutlass](https://github.com/NVIDIA/cutlass)：NVIDIA 的 CUDA GEMM 模板库（CuTe 表达式模板），手写 split-K / grouped / FP8 GEMM 的底座——DeepGEMM、vLLM kernel 都基于它（补 4.4 算子清单里缺的）。

> 📝 作业（4.5）：① torch.compile 相比 eager 主要省在哪（融合 / launch / autotune）？② CUDA Graph 为何对 decode 阶段尤其有用？答清即完成。

### 4.6 昇腾 CANN 算子

国产昇腾(Ascend) NPU 的算子栈与 NVIDIA/CUDA 平行但独立——CANN(Compute Architecture for Neural Networks) 是其软件栈，理解它补齐“非 CUDA”算子视角（呼应 7.1）。

* **CANN 软件栈**：对标 CUDA + cuDNN + cuBLAS——含 ATC(图编译)、AscendCL(运行时 API)、算子库(Ascend C / TE / TBE)。[CANN 文档](https://www.hiascend.com/document/detail/en/CANNcommercial)。
* **算子开发**：Ascend C（C++ 算子 API，对标 CUTLASS / Triton）、Tile-based Engine(TE / TBE) 分块算子编程——NPU 的“Tile / Cube”编程模型。
* **图编译**：GE(Graph Engine) 做图级融合 / 编排，对标 XLA / Inductor（呼应 4.5）。
* **推理生态**：MindIE（昇腾推理引擎，JD 高频）、[vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)（vLLM 的昇腾后端）、[Ascend Samples](https://github.com/Ascend)（CANN 算子样例）。
* **与 CUDA 的差异**：NPU 的 AI Core（Cube / Vector 协同）vs GPU Tensor Core；显存层次与编程模型不同——同一算法常需重写 / 重调优。

> 📝 作业（4.6）：① 一句话说清 CANN 分别对标 CUDA 的哪几层；② vLLM-Ascend 要把 vLLM 的哪些 CUDA 依赖替换 / 适配为昇腾？答清即完成。

---

## 阶段五：核心实战项目与工程级实现

本阶段抛弃玩具 Demo，直接构建工业级引擎的核心关键模块——覆盖投机解码、显存调度、压测、多租户 / 量化、模型编译等推理链路（呼应 2.4 / 2.1 / 4.3 / 2.6·2.2 / 4.5）；RL 与 agent 的实战项目放在 6.5 / 7.5，紧跟各自理论之后。

### 5.1 项目一：EAGLE 风格 Tree Attention 投机解码器

**目标**：掌握现代投机解码（Tree Speculation）的树状注意力掩码（Tree Attention Mask）构造与路径校验。

```python
import torch

def build_tree_attention_mask(tree_structure, prefix_len):
    """
    构造 Tree Attention Mask
    :param tree_structure: 树分支拓扑节点依赖, 如 [[0], [1, 2], [3]]
    :param prefix_len: 原 Prefix 的 Token 长度
    """
    num_draft_tokens = len(tree_structure)
    total_len = prefix_len + num_draft_tokens
    
    # 1. 初始化全遮蔽矩阵 (-inf)
    mask = torch.full((total_len, total_len), fill_value=torch.finfo(torch.float16).min)
    
    # 2. Prefix 区域：标准因果下三角 Mask
    mask[:prefix_len, :prefix_len] = torch.triu(
        torch.full((prefix_len, prefix_len), fill_value=torch.finfo(torch.float16).min), diagonal=1
    )
    
    # 3. Draft Tokens 对 Prefix 全部可见
    mask[prefix_len:, :prefix_len] = 0.0
    
    # 4. Draft Tokens 仅对其祖先节点 (Ancestors) 可见
    for node_idx, parent_nodes in enumerate(tree_structure):
        curr_pos = prefix_len + node_idx
        mask[curr_pos, curr_pos] = 0.0  # 对自身可见
        for p in parent_nodes:
            mask[curr_pos, prefix_len + p] = 0.0  # 对树上的父节点可见
            
    return mask  # 可直接作为 PyTorch SDPA 或 Custom Kernel 的 attn_mask 传入
```

> 易踩坑点：Tree Attention Mask 正确还不够，draft tokens 必须各自分配正确的 **position ids**（通常沿用其树上祖先的位置序列），position 错位会导致输出异常——这是新手最常忽略的一环。

> 📝 作业（5.1）：
> [问答] draft tokens 为何必须各自赋 position id？错位会导致什么？
> [代码] 在 build_tree_attention_mask 上：① 支持二叉树草稿（每节点 2 子）；② 为每个 draft token 正确赋 position id；③ 写 1 个断言验证“某 token 只可见 prefix + 祖先”。跑通即完成。

---

### 5.2 项目二：nano-vllm 最小推理引擎

**目标**：从零搭一个最小推理引擎，把 PagedAttention（分页 KV）+ 连续批处理 + decode loop 串成端到端——即一个 nano-vllm。参考实现：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（PyTorch 极简 vLLM）、[karpathy/llama2.c](https://github.com/karpathy/llama2.c)（单文件 C 推理引擎）。

```python
class BlockAllocator:
    """物理显存块管理器 (类似 vLLM BlockSpaceManager)"""
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks = set(range(num_blocks))
        self.block_tables = {}  # req_id -> List[int] (映射到物理 Block ID)

    def allocate(self, req_id: str, num_tokens: int):
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < needed_blocks:
            raise MemoryError("Out of KV Cache Memory! Triggering Preemption...")
        
        allocated = [self.free_blocks.pop() for _ in range(needed_blocks)]
        self.block_tables[req_id] = allocated
        return allocated

    def free(self, req_id: str):
        if req_id in self.block_tables:
            for b in self.block_tables[req_id]:
                self.free_blocks.add(b)
            del self.block_tables[req_id]

class ContinuousBatchScheduler:
    """连续批处理调度器"""
    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.waiting_queue = []
        self.running_queue = []

    def step(self):
        # 1. 检查 Waiting 队列，将新 Request 调度至 Running 批次 (Prefill 阶段)
        # 2. 对 Running 队列中的 Request 分配新 Token 的 KV 空间 (Decode 阶段)
        # 3. 剔除遇到 <EOS> 的 Request 并归还显存 Block
        pass
```

```python
class Engine:
    """最小推理引擎：调度器 + 分页 KV + 模型 forward，跑连续批处理 decode"""
    def __init__(self, model, allocator: BlockAllocator):
        self.model = model                              # 任意 nn.Module（或 forward stub）
        self.scheduler = ContinuousBatchScheduler(allocator)

    def step(self):
        # 1. 调度：waiting→running、为新 token 申请 KV 块(decode)、遇 EOS 释放
        batch = self.scheduler.step()
        # 2. 一次 forward（prefill+decode 混批）：读 block_tables 里的 paged KV
        logits = self.model(batch.token_ids, kv_block_tables=self.scheduler.allocator.block_tables)
        # 3. GPU 侧采样下一 token，追加进对应请求
        next_tokens = self.model.sample(logits)
        for req, tok in zip(batch.reqs, next_tokens):
            req.append_token(tok)
        return batch

# 用法：3 个并发请求 → 连续批处理直到全部 <EOS>
# engine = Engine(model, BlockAllocator(num_blocks=1024, block_size=16))
# while engine.scheduler.has_active():
#     engine.step()
```

> 参考：[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 把这套做成可跑的最小实现（paged KV + continuous batching + prefill/decode 分离），可对照阅读；[llama2.c](https://github.com/karpathy/llama2.c) 是单文件 C 的极简推理引擎。

> 📝 作业（5.2）：① 给 `ContinuousBatchScheduler.step()` 补全三步（waiting→running 调度、decode 分配 KV 块、遇 EOS 释放）+ 最简 LRU 抢占；② 在 `Engine.step()` 上接一个 forward stub + 采样，跑通 3 个并发请求直到全部 EOS。跑通即完成。

---

### 5.3 项目三：MaaS 高并发场景压测与 `nsys` Profiling 管线

**目标**：掌握吞吐量、首字延迟（TTFT）、Token 间延迟（ITL）评估及性能瓶颈分析。

1. **流量模型建立**：使用泊松分布模拟真实并发请求到达。
2. **性能基准与 Profiling 指令**：
   ```bash
   # 抓取 GPU 侧的 CUDA Kernel 执行时间与 CPU Launch 开销
   nsys profile        --trace=cuda,nvtx,osrt        --output=vllm_eager_vs_cudagraph        --export=sqlite        python benchmark_serving.py --backend vllm --dataset sharegpt --qps 20
   ```

   📺 李宏毅 — [第九講：大型語言模型評估](https://www.youtube.com/watch?v=s266BzGNKKc)（呼应本节评测）；🎧 Practical AI — [Stellar inference speed via AutoNAS (ep148)](https://changelog.com/practicalai/148)（推理加速）。

> 📝 作业（5.3）：用泊松到达跑一次 benchmark_serving，读出 TTFT / TPOT / 吞吐三条曲线，判断当前瓶颈是 prefill 还是 decode；再用 nsys 验证。跑通 + 判断一致即完成。

### 5.4 项目四（选做）：多 LoRA 混批服务 或 INT4 在线反量化 GEMM

**目标**：二选一，补齐生产关键链路。

* **方向 A — 多 LoRA 混批**：单基座 + 多个 LoRA 适配器，实现按请求路由 adapter 的连续批处理（参考 S-LoRA 的 MBGMM/MBGMV 融合 kernel），体会多租户共享基座的吞吐优势。
* **方向 B — INT4 / FP8 在线反量化 GEMM**：用 Triton 实现权重在线反量化并融进 GEMM 的 kernel，对比离线反量化 baseline 的带宽节省，并体会阶段二量化与阶段四算子的衔接。

> 📝 作业（5.4）：从方向 A / B 选一个完成最小可用版（多 LoRA 路由 + 混批，或 INT4 在线反量化 GEMM），并给出相对 baseline 的吞吐 / 带宽对比数。跑出数即完成。

### 5.5 项目五：模型编译加速

**目标**：用编译 / 图捕获把一个 decode 循环加速并量化收益——呼应 4.5。

1. 起一个最小 LLM 的 decode 循环（eager 版，逐 token 生成），记录 token/s。
2. 套 `torch.compile`（Inductor），对比 token/s 与首字延迟；注意 graph break。
3. 再用 CUDA Graph 捕获 decode 单步，对比 CPU launch 开销。
4. 用 nsys 抓 trace，看 kernel 密度提升。

> 📝 作业（5.5）：给出 eager / +compile / +CUDA Graph 三档的 token/s 与 trace 对比，指出收益主要来自融合还是 launch 消除。跑通即完成。

---

## 阶段六：训练系统 infra

> 推理 infra（1~5 阶段）是“跑一个模型出 token”；训练 infra 是“把大模型在万卡上训起来、对齐起来”。本阶段补训练半边——并行、通信、框架、显存、RL 后训练与性能，构成完整 AI Infra。选学，按需进入。
>
> 🧑‍💻 个人可行性：本节真实场景是千卡-万卡集群，个人无法复现；自学模式 = 单 / 双卡小模型缩放 demo（FSDP / ZeRO / TP / NCCL 都能在 1-2 卡跑通）+ 读框架源码 + 看论文 / trace。能读懂、能跑通 demo 即达标。

**关键论文与项目（速查）**

| 论文 / 项目 | 链接 | 核心要点 |
| :--- | :--- | :--- |
| [ZeRO](https://arxiv.org/abs/1910.02054) / [DeepSpeed](https://github.com/microsoft/DeepSpeed) | 显存 | 优化器态 / 梯度 / 参数分片到多卡（ZeRO-1/2/3），用通信换显存，万亿参数可训。 |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) / [Megatron-3D](https://arxiv.org/abs/2104.04473) | 并行 | TP+PP 奠基实现；3D 并行（DP / TP / PP）在 GPU 集群上的工程化。 |
| [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198) | SP / 显存 | Megatron 序列并行 + 选择性重算，进一步压激活显存。 |
| [GPipe](https://arxiv.org/abs/1811.06965) | PP | 流水并行 + micro-batch，把层切到不同卡、用流水掩盖。 |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | 精度 | FP16 / BF16 训练 + loss scaling，省显存提速（呼应 2.2 量化）。 |
| [verl](https://github.com/volcengine/verl) / [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | RL | 大规模 PPO / GRPO 后训练，rollout 复用推理引擎。 |

### 6.1 分布式训练并行

**入门**：单卡放不下大模型也喂不饱算力，于是把模型 / 数据 / 序列切片到多卡——并行策略就是“切哪、怎么切、怎么同步”。

* **数据并行 DP / FSDP / ZeRO**：切数据、各卡算一份梯度再同步；FSDP/ZeRO 把优化器状态 / 梯度 / 参数也切片省显存。[DeepSpeed](https://github.com/microsoft/DeepSpeed)（ZeRO）、[PyTorch 激活检查点文档](https://pytorch.org/docs/stable/checkpoint.html)。
* **张量并行 TP / 流水并行 PP**：切权重矩阵(TP)或切层(PP)；TP 通信密、PP 有气泡。[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。
* **序列并行 SP / 上下文并行 CP**：长序列切多卡算注意力（呼应 2.3 Ring Attention）。
* **专家并行 EP**：MoE 按专家分卡（呼应阶段四 DeepEP 的 all-to-all）。
* **5D 并行**：DP / TP / PP / EP / SP 组合——阿里 / 字节 JD 的高频词。

> 算例：70B 参数 FP16 训练态 ≈ 70B × (2 参数 + 2 梯度 + 4 master + 4 Adam-m + 4 Adam-v) ≈ **1.1 TB**（含 fp32 优化器态）；8×80G=640G 也放不下整个训练态 → 必须 **TP(切权重)+ PP(切层)+ ZeRO-3/FSDP(分片参数/优化器态/梯度)** 组合才塞得下、跑得动。万卡就是这套 ×N。

> 📝 作业（6.1）：① 给 70B 模型 + 8×80G，设计一种放得下的并行组合（哪些切数据、哪些切权重）；② PP 的“气泡”为何降低 MFU、怎么 overlap 掉？答清即完成。

### 6.2 通信与集群

**入门**：并行越多、卡间通信越成瓶颈；通信库、拓扑、overlap 决定扩展性。

* [NCCL](https://github.com/NVIDIA/nccl)：NVIDIA 集合通信库（all-reduce / all-gather 等），分布式训练通信事实标准。
* **计算-通信 overlap**：算下一层的同时通信上一层，掩盖延迟、提 MFU（阿里 JD“overlap 通信掩盖”）。
* **拓扑感知放置**：NVLink / NVSwitch（节点内）、RDMA / IB（节点间），层级带宽决定并行上限。
* **容错**：万卡训练频繁掉卡，需 checkpoint 恢复、弹性 reshard、故障检测（Anthropic JD“fault-tolerant accelerator fleets”）。

> 📝 作业（6.2）：① all-reduce 通信量随卡数怎么变？为何 ring 拓扑常用？② “通信与计算 overlap”为何直接提 MFU？答清即完成。

### 6.3 训练框架与显存

**入门**：手写 5D 并行太重，框架替你管切分与显存。

* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)：NVIDIA 的 TP / PP / EP / SP 训练参考实现，工业级并行事实标准。
* [DeepSpeed](https://github.com/microsoft/DeepSpeed)：ZeRO 系列显存优化 + 3D 并行 + 通信优化。
* **PyTorch FSDP / JAX pjit**：原生分片并行（FSDP 与[激活检查点](https://pytorch.org/docs/stable/checkpoint.html)配合）。
* **显存技巧 / 以存代算**：activation checkpointing（算力换显存）、ZeRO 分片、CPU offload、混合精度（BF16 / FP8，呼应 2.2）——“以存代算”（用算力 / 带宽换显存）的工程统称，JD 高频词。

> 📝 作业（6.3）：① activation checkpointing 为何“算力换显存”？省的是哪部分、重算的是哪部分？② ZeRO-1 / 2 / 3 各切什么、显存换算力代价递增在哪？答清即完成。

### 6.4 RL 后训练（深入）

后训练把“会接话”的预训练模型变成“对齐 + 会推理”的——从 SFT 到 RLHF/PPO 到 DPO/GRPO，再到 R1 式 RL with verifiable rewards。infra 极重，是训练与推理的“交汇点”。

* **算法谱系**：
  * **SFT**：监督微调（指令跟随），起点。
  * **RLHF / PPO**（[PPO](https://arxiv.org/abs/1707.06347)）：奖励模型(RM) + 策略(PPO) + 参考模型(KL 约束) + 价值模型(critic)——**4 模型同驻**显存极重；advantage = reward − value。
  * **DPO**（[2305.18290](https://arxiv.org/abs/2305.18290)）：无 RM / 无在线采样，离线偏好对直接训，简单但牺牲在线探索。
  * **GRPO**（[DeepSeekMath 2402.03300](https://arxiv.org/abs/2402.03300)）：组内相对优势，**去掉 critic**（省价值模型），算力友好——DeepSeek 的选择。
  * **RLVR / R1**（[2501.12948](https://arxiv.org/abs/2501.12948)）：可验证奖励（数学 / 代码有标准答案），无需 RM，长 CoT 涌现推理。
* **infra 挑战（核心）**：
  * **rollout = 在线推理**：每步采样大量生成 → 复用推理引擎(vLLM)做 rollout（不是训练框架）——训练×推理交汇点。
  * **多模型显存墙**：PPO 4 模型同驻爆炸 → offload/reload 切换、colocation vs disaggregation。
  * **rollout-train 交错**：生成(推理)与更新(训练)交替，引擎在两种模式切换；[verl](https://github.com/volcengine/verl) / [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 是“rollout engine + trainer + 参数同步”的胶水。
  * **长生成**：reasoning RL 单条极长 → KV 膨胀、rollout 吞吐下降（呼应 2.7 KV 深入）。
  * **奖励 / 评测**：RM serving + verifiable reward + 评测环境。
* **性能**：rollout 吞吐(在线推理) + 训练 MFU（因在线生成，低于预训练）；rollout:train 时长比是关键调度参数。

> 📝 作业（6.4）：① PPO 为什么需要 4 个模型？GRPO 省掉的是哪个、靠什么替代？② 为什么 rollout 复用 vLLM 而非 Megatron？③ R1 用“可验证奖励”为什么能省掉 RM？答对 2/3 即完成。

### 6.5 实战项目：小规模 RL 后训练

**目标**：在小模型上跑通“训练×推理交汇”的 RL 后训练（紧跟 6.4 理论）。

1. 选 0.5B-1B 小模型 + 一个可验证奖励任务（数学题 / 代码单测，有标准答案）。
2. 用 [verl](https://github.com/volcengine/verl) / [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 跑 GRPO：rollout 用 vLLM、训练用 FSDP 小规模。
3. 观察奖励曲线上升、KL 不发散。

> 📝 作业（6.5）：跑通一次 GRPO，贴奖励曲线；说明 rollout 为何复用 vLLM、rollout:train 时长比大致多少。跑通即完成。
>
> 🧑‍💻 个人可行性：0.5B 小模型 + 单卡即可跑通 toy 版；真实 R1 级别需集群（见 6.4 caveat）。

---

## 阶段七：前沿选读

> 本阶段为前沿与交叉方向，按兴趣选学，不纳入主时间线；内容更新快，以官方资料为准。

### 7.1 硬件多样性
前五阶段偏 NVIDIA / CUDA 数据中心。实际部署生态更广，理解异构硬件的取舍有助于因地制宜：

* **AMD Instinct + ROCm/HIP**：软件栈与 CUDA 高度平行（PyTorch / Triton / CK 均有 ROCm 版）；vLLM、SGLang 支持 ROCm 后端。[AMD ROCm 文档](https://docs.amd.com)
* **Apple Silicon + MLX**：统一内存架构，适合端侧 / 研究（见阶段三 [MLX](https://github.com/ml-explore/mlx)）。
* **Groq LPU**：确定性数据流架构，单流延迟极低（吞吐非其强项），适合延迟敏感的轻并发场景。[Groq](https://groq.com)
* **Cerebras / SambaNova**：晶圆级 / RDU 等非主流架构，走 weight-streaming 或数据流路线。[Cerebras](https://www.cerebras.net) | [SambaNova](https://sambanovasystems.com)
* **NVIDIA Blackwell（下一代 kernel 目标）**：FP4、第二代 Transformer Engine、增强 TMA 与 NVLink 带宽，是后续算子优化的新硬件基线。[NVIDIA 开发者站](https://developer.nvidia.com)

> 📝 作业（7.1）：用一句话说清 Groq LPU、Cerebras、NVIDIA Blackwell 各自押注的硬件优势；并指出 vLLM 上 ROCm 后端的取舍。答清即完成。

### 7.2 推理时扩展
与“压低成本”相反的另一条轴：在推理时投入更多算力换取质量，已成为 2024–2025 的重要方向。

* [Self-Consistency（Wang et al.）](https://arxiv.org/abs/2203.11171)：采样多条 CoT、多数投票选答案，简单有效。
* [Let's Verify Step by Step（Lightman et al.）](https://arxiv.org/abs/2305.20050)：用过程奖励模型（PRM）对推理步骤打分，best-of-N + verifier 显著提升推理质量。
* [Scaling LLM Test-Time Compute（Snell et al.）](https://arxiv.org/abs/2408.03314)：系统研究“何时、以何种方式增加推理算力比加参数更划算”。

> 服务侧影响：o1 / R1 式长 CoT 推理模型单次生成极长——KV Cache 持续膨胀、TPOT 与长程吞吐成主战场、投机解码收益在长生成中复利放大、长 prompt 的 prefill 成本不可忽视。这把前五阶段的议题（显存、KV、PD 分离、投机解码）在“长生成”尺度上又压了一遍。

> 📝 作业（7.2）：① best-of-N + verifier 比朴素采样好在哪（算力-质量曲线）？② 长 CoT 推理模型（o1 / R1）对服务侧 KV / TPOT 提出什么新要求？答清即完成。

### 7.3 多模态推理
* **视觉编码器 prefill 成本**：VLM 通常先用 ViT 等编码图像，经投影层喂入 LLM；一张图常等价于数百~数千 token，prefill FLOPs 与 KV Cache 随图像 token 数线性膨胀。
* **多模态连续批处理**：文本 / 图像请求混合、图像 token 数差异大，使 batching 与 padding 比纯文本更复杂。
* **跨模态 KV**：图像特征经 projector 接入 LLM 的 KV 序列，需与文本 KV 统一管理（PagedAttention 等机制仍适用，但 token 来源更杂）。
* **预处理并行**：ViT 编码、图像解码 / resize / 归一化、projector 前处理可流水并行 + 多图 batch，并与 LLM prefill 重叠——多模态 JD 的「预处理并行」提速点。

> 📝 作业（7.3）：① 一张 1024×1024 图经 ViT + projector 约变成多少 token？它如何同时抬高 prefill 成本与 KV？② 多模态连续批处理的 padding 难点比纯文本多在哪？答清即完成。

### 7.4 Agentic Infra

**入门**：跑一个 agent = LLM 多轮调工具 + 长程执行 + 需要轨迹 / 评测 / 复盘；支撑这套的“agent infra”是新兴方向（字节 JD“Agentic Infra”）。

* **Agent loop**：observe → plan → act(调工具) → observe 循环；长程 = 多轮 + 状态 / 记忆 + 工具失败重试——比单轮推理难“稳”的根源。
* **Tool-calling**：function calling / 工具 API + 结果校验；[ReAct](https://arxiv.org/abs/2210.03629)（推理+行动）、[Toolformer](https://arxiv.org/abs/2302.04761)（自学习用工具）是范式起点。
* **Agent Harness**：运行时知识库 + Skills&CLI + 规模化验证，让 agent 稳定长程执行（字节 JD）。
* **Trace 机制**：记录每步决策与工具调用，用于 debug、归因、效率优化。
* **框架**：[LangGraph](https://github.com/langchain-ai/langgraph)（状态机式 agent 图）、[aider](https://github.com/Aider-AI/aider)（coding agent）。
* **Auto Research / 评测**：[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 做 benchmark；仿真 + 真实环境支撑大规模 eval。
* **RL + 工具调用**：大规模 RL（呼应 6.4 verl / OpenRLHF）+ 工具调用数据 + 仿真环境，训练会做工具的 agent（agentic 后训练）。

> 📝 作业（7.4）：① agent 长程执行为何比单轮推理难“稳”（trace、状态、工具失败）？② agent 后训练比普通 RLHF 多了“工具调用”维度，infra 上多了什么？答清即完成。

### 7.5 实战项目：最小 Agent Harness

**目标**：搭一个能多轮调工具 + 有 trace 的最小 agent（紧跟 7.4 理论）。

1. LLM loop（observe → plan → act 调工具 → observe）+ 一个工具（计算器 / 本地搜索 / 代码执行）。
2. 加 trace 记录每步决策与工具调用。
3. 加一个简单评测（任务完成率 / 步数）。

> 📝 作业（7.5）：跑通一个 3-5 步的工具调用任务，贴 trace；说明长程执行比单轮推理难“稳”在哪。跑通即完成。

---

## 总结与建议学习路线

1. **第 1~2 周（理论）**：理解 Roofline 模型，推导 KV Cache 显存公式，搞清 MQA/GQA/MLA 对显存的影响；精读 *PagedAttention* 与 *FlashAttention-1/2*。
2. **第 3~4 周（投机解码 + 量化）**：实现树状 Attention 掩码与投机校验（注意 position ids），读 *EAGLE* / *Medusa* / *LayerSkip* / MTP，搞懂拒绝采样与自投机；同时过一遍 *GPTQ* / *AWQ* / *SmoothQuant* 与 KV Cache 量化。
3. **第 5~6 周（服务架构）**：精读 *DistServe* / *Mooncake*（PD 分离）、ORCA / Sarathi-Serve（连续批处理与 chunked prefill）、*S-LoRA*（多租户 LoRA）。
4. **第 7~10 周（引擎源码）**：克隆 vLLM，打断点阅读 `vllm/v1/core/sched/scheduler.py` 与 `kv_cache_manager.py`（见 3.1）；横向对比 TensorRT-LLM、FlashInfer（尤其 MLA kernel）、llama.cpp 的取舍。
5. **第 11~14 周（算子压榨）**：学 Triton 官方教程，手写解码态注意力 kernel / 反量化融合 GEMM / 融合 RMSNorm，用 `nsys` 定位算子开销，向开源项目提 PR。

> 阶段六（训练系统 infra）与阶段七（前沿选读：硬件多样性 / 推理时扩展 / 多模态 / Agentic Infra）按兴趣穿插选学，不计入主时间线；KV Cache / RL / 昇腾 CANN / vLLM-SGLang 的深入已并入 2.7 / 6.4 / 4.6 / 3.1 等相关章节，不单列。
>
> 🧑‍💻 个人自学硬件门槛（贯穿全文）：① 纯阅读（论文 / 架构）→ 无需硬件；② 单卡可练（算子 / 编译 / 小模型推理调度）→ 一张 GPU 或 Colab 即可；③ 需多卡 / 集群（5D 并行 / NCCL 万卡 / 容错 / 真实压测）→ 个人只能缩放 demo + 读源码 + 看 trace，达标以“懂 + 跑得通 demo”为准，不要求复现真实规模。
