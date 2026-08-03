---
id: fu_2025_moba
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["MoBA", "块注意力", "Mixture of Block Attention", "2505.05955"]
related_nodes: [sparse_attention, mixture_of_experts]
authors: Enzhe Lu et al.
authors_institution: Moonshot AI
arxiv_id: 2505.05955
last_verified: 2026-08-03
---

# MoBA: Mixture of Block Attention for Long-Context LLMs

- **元数据**: arXiv 2505.05955 | 2025 | **作者**: Enzhe Lu et al. | **机构**: Moonshot AI
- **概述**: 提出 MoBA（混合块注意力）：借鉴 MoE 的路由思想，将注意力改为"键块路由"——每个查询只关注被选中的键块，长上下文下计算量线性化。
- **新颖概念**: [[sparse_attention]]
- **关键要点**: 1. 查询-键块路由替代全量注意力 2. 局部窗口 + 全局路由结合（GQA 式分组共享路由）3. 训练时可渐进从全注意力切换到 MoBA
- **方法/发现**: 10M token 上下文训练/推理可行；长序列任务与全注意力相当，短序列可用全注意力保证质量
- **局限/意义**: 将 MoE 的"稀疏路由"思想迁移到注意力，与 DeepSeek 的 DSA 并列代表 2025 年稀疏注意力主线

## 引用
- **原始论文**: [arXiv:2505.05955](https://arxiv.org/abs/2505.05955)
- **相关概念**: [[sparse_attention]], [[mixture_of_experts]]
