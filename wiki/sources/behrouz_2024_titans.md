---
id: behrouz_2024_titans
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["Titans", "神经长期记忆", "2412.08762"]
related_nodes: [neural_long_term_memory, conditional_memory, transformer_architecture]
authors: Ali Behrouz et al.
authors_institution: Google
arxiv_id: 2412.08762
last_verified: 2026-08-03
---

# Titans: Learning to Memorize at Test Time

- **元数据**: arXiv 2412.08762 | 2024 | **作者**: Ali Behrouz et al. | **机构**: Google
- **概述**: 提出 Titans 架构：将"测试时学习"的神经长期记忆模块与注意力结合，模型在推理中持续更新记忆，长上下文任务超越 Transformer 与线性注意力基线。
- **新颖概念**: [[neural_long_term_memory]]
- **关键要点**: 1. 神经记忆：以梯度下降为记忆更新机制（test-time training） 2. 记忆-注意力三种融合模式（门控/并行/级联） 3. 百万级上下文建模
- **方法/发现**: 在语言建模与长上下文基准上超过 Mamba 与 Transformer；核心洞察：记忆应"在线学习"而非固定
- **局限/意义**: 与 Engram（静态条件记忆）互补，代表"记忆增强架构"的两极；测试时训练的计算成本是主要开放问题

## 引用
- **原始论文**: [arXiv:2412.08762](https://arxiv.org/abs/2412.08762)
- **相关概念**: [[neural_long_term_memory]], [[conditional_memory]]
