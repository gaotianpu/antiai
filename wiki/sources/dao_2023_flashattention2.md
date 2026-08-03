---
id: dao_2023_flashattention2
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["FlashAttention-2", "2307.08691"]
related_nodes: [flash_attention, io_aware_attention]
authors: Tri Dao
authors_institution: Stanford
arxiv_id: 2307.08691
last_verified: 2026-08-03
---

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

- **元数据**: arXiv 2307.08691 | 2023 | **作者**: Tri Dao | **机构**: Stanford
- **概述**: FlashAttention 的改进版：优化 GPU 线程块并行与工作划分，减少非矩阵乘运算，前向/反向各提速约 2 倍，达理论计算上限 50-73%。
- **新颖概念**: [[flash_attention]]
- **关键要点**: 1. 减少重计算与非矩阵乘开销 2. 序列长度维度并行 + 头/块间负载均衡 3. 反向传播无需重算注意力矩阵
- **方法/发现**: 相比 FA-1 吞吐提升 2 倍，比 PyTorch 标准注意力快 5-9 倍；训练 GPT-3 级模型时端到端提速显著
- **局限/意义**: 仍是 H100 前 GPU 设计；FA-3（2024）针对 Hopper 架构的异步并行继续优化——FA 系成为 LLM 训练/推理的默认注意力算子

## 引用
- **原始论文**: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- **相关概念**: [[flash_attention]], [[io_aware_attention]]
