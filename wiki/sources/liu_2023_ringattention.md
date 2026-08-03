---
id: liu_2023_ringattention
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["Ring Attention", "环注意力", "2310.01889"]
related_nodes: [sequence_parallelism, flash_attention]
authors: Hao Liu et al.
authors_institution: UC Berkeley
arxiv_id: 2310.01889
last_verified: 2026-08-03
---

# Ring Attention with Blockwise Transformers for Near-Infinite Context

- **元数据**: arXiv 2310.01889 | 2023 | **作者**: Hao Liu et al. | **机构**: UC Berkeley
- **概述**: 提出环注意力：将序列分块分布在多设备上环形传递（ring），配合分块注意力（blockwise）实现近无限上下文训练，无需序列级内存聚合。
- **新颖概念**: [[sequence_parallelism]]
- **关键要点**: 1. 设备间环形传递 KV 块，各设备轮流计算分块注意力 2. 通信与计算重叠 3. 突破单设备序列长度上限
- **方法/发现**: 在 64 卡上训练 400 万 token 序列；与 FA 分块技术结合，构成超长上下文训练的基础并行方案
- **局限/意义**: 注意力系统级并行的重要工作；后续 Striped Attention 等进一步改进负载均衡

## 引用
- **原始论文**: [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
- **相关概念**: [[sequence_parallelism]], [[flash_attention]]
