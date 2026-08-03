---
id: transformer_evolution
type: synthesis
tags: ["machine-learning", "NLP", "theoretical"]
aliases: ["Transformer演进", "Transformer发展路线", "注意力架构演进"]
related_nodes: ["transformer_architecture", "attention_mechanism", "flash_attention", "state_space_model", "kv_cache"]
last_verified: 2026-08-03
---

# Transformer 演进路线（2017-2025）

## 概述

从 2017 年提出到 2025 年，Transformer 沿**效率、推理、长度、替代**四条主线演进：注意力计算越来越快、KV cache 越来越小、上下文越来越长、并接受 SSM/记忆架构的挑战与融合。

## 阶段 0：提出与奠基（2017-2021）

- [[vaswani_2017_transformer]] — "Attention Is All You Need"：纯注意力架构（[[transformer_architecture]]）
- [[dai_2019_transformer_xl]] — 片段级循环 + 相对位置编码（[[segment_level_recurrence]] / [[relative_position_encoding]]）
- [[roformer_2021]] — RoPE 旋转位置编码（[[rotary_position_embedding]]），现代 LLM 标配
- [[alibi_2021]] — ALiBi 线性偏置（[[alibi_position_encoding]]），零参数长度外推
- [[dao_2022_flashattention]] — FlashAttention：IO 感知算法（[[flash_attention]] / [[io_aware_attention]]）

## 主线一：效率（2023-2024）——让注意力更快

- [[dao_2023_flashattention2]] — FA-2：并行与工作划分优化，再快 2 倍
- [[liu_2023_ringattention]] — Ring Attention（[[sequence_parallelism]]）：序列分块环形并行，近无限上下文训练
- [[dao_2024_mamba2]] — SSD 理论：注意力与 SSM 的统一（见主线四）

## 主线二：推理（2023-2024）——让 KV cache 更小、解码更快

- [[ainslie_2023_gqa]] — GQA 分组查询注意力（[[grouped_query_attention]]）：KV cache 缩减 1/组数
- [[deepseek_2024_v2]] — MLA（[[multi_head_latent_attention]]）：KV 压缩为潜在向量，压缩 93.3%
- [[kwon_2023_pagedattention]] — PagedAttention（[[kv_cache]]）：分页管理消除碎片，vLLM 基础
- [[xiao_2023_streamingllm]] — 注意力汇点：sink + 滑动窗口的 KV 淘汰策略
- [[leviathan_2023_speculative]] — 投机解码（[[speculative_decoding]]）：草稿-验证加速 2-3 倍
- [[wang_2024_bitnet]] — BitNet b1.58：1.58-bit 三值权重，能耗降低 71-91%

## 主线三：长度（2023-2025）——上下文不断扩展

- [[peng_2023_yarn]] — YaRN（[[length_extrapolation]]）：NTK 插值 + 温度修正，4K→128K
- [[longnet_2023]] — 扩张注意力（[[dilated_attention]]）：线性复杂度超长程建模
- [[gemini_2024_15]] — Gemini 1.5：百万 token 上下文工程化，长上下文成为旗舰标配
- [[fu_2025_moba]] — MoBA（[[sparse_attention]]）：MoE 式键块路由
- [[deepseek_2025_v32]] — DSA：Lightning Indexer 稀疏选择，工业级稀疏注意力

## 主线四：替代与融合（2023-2025）——Transformer 之外

- [[gu_2023_mamba]] — Mamba 选择性状态空间（[[state_space_model]]）：线性复杂度匹敌注意力
- [[dao_2024_mamba2]] — SSD 对偶性：SSM 与注意力是同一矩阵族的两个极端
- [[retnet_2023]] — RetNet（[[retention_mechanism]]）：线性递推 + 并行训练
- [[behrouz_2024_titans]] — Titans（[[neural_long_term_memory]]）：测试时学习的神经记忆
- [[cheng_2026_engram]] — Engram（[[conditional_memory]]）：O(1) 静态查表与 MoE 互补

## 演进时间线

| 年份 | 里程碑 | 主线 |
|:---|:---|:---|
| 2017 | Transformer 提出 | 奠基 |
| 2019-2021 | 位置编码定型（相对/RoPE/ALiBi） | 长度 |
| 2022 | FlashAttention | 效率 |
| 2023 | FA-2 / GQA / PagedAttention / 投机解码 / YaRN / Mamba | 全主线 |
| 2024 | MLA / Mamba-2 (SSD) / BitNet / Titans / Gemini 1.5 | 全主线 |
| 2025 | MoBA / DSA 稀疏注意力 / DeepSeek-V3 系 | 长度+效率 |

## 关键洞察

- **注意力本身没有变**：2017 的缩放点积注意力公式至今未改，演进发生在工程（FA、分页）、压缩（GQA/MLA）、选择（稀疏路由）与替代（SSM/记忆）四个层面
- **两条边界竞争**：稠密全注意力 vs 线性替代（SSM/稀疏）——SSD 理论表明两者同源，混合架构（注意力 + SSM/记忆）是收敛方向
- **KV cache 是第一瓶颈**：长上下文的成本从"计算"转移到"存储"，压缩与淘汰策略（[[kv_cache]]）成为核心战场

## 相关概念

- [[transformer_architecture]] — 架构总览
- [[attention_mechanism]] / [[attention_variants]] — 注意力机制与变体
- [[flash_attention]] / [[kv_cache]] — 效率与推理
- [[positional_encoding]] / [[length_extrapolation]] — 位置与长度
- [[sparse_attention]] / [[state_space_model]] — 稀疏与替代
- [[multi_head_latent_attention]] — MLA 潜在压缩
