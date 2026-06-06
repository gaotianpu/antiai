---
id: cheng_2026_engram
type: source
tags: [NLP, machine-learning, empirical-study]
aliases: [Engram, 条件记忆, Conditional Memory via Scalable Lookup, 2601.07372]
related_nodes: [conditional_memory, sparsity_allocation, deepseek_ai, mixture_of_experts]
arxiv_id: 2601.07372
authors: Xin Cheng, Wangding Zeng, Damai Dai et al.
authors_institution: DeepSeek, Peking University
last_verified: 2026-06-06
---

# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

- **元数据**: arXiv | 2026-01 | **作者**: Xin Cheng, Wangding Zeng, Damai Dai et al. | **机构**: DeepSeek / Peking University | 相关: [[conditional_memory]], [[mixture_of_experts]], [[deepseek_ai]]
- **概述**: 提出条件记忆（conditional memory）作为稀疏性的新轴心，实例化为 Engram 模块——将经典 N-gram 嵌入改造为 O(1) 查表，与 MoE 互补。发现 U 形 Scaling Law 指导 MoE 与静态记忆间的最优分配。
- **关键要点**:
  1. 语言建模包含两个本质不同的子任务：组合推理（需动态计算）和知识检索（可静态查表），但 Transformer 缺少原生查表原语
  2. Engram 模块：N-gram 哈希索引 → 多头哈希 → 上下文门控 → 多分支融合，确定性寻址支持运行时预取
  3. U 形 Sparsity Allocation 定律：MoE 与 Engram 存在最优配比，过少或过多静态记忆均次优
  4. 27B 模型在 iso-parameter & iso-FLOPs 下全面超越纯 MoE 基线，不仅在知识任务（MMLU +3.4）更在推理（BBH +5.0）和代码数学（HumanEval +3.0）上显著提升
  5. 机制分析发现 Engram 减轻了 backbone 早期层对静态知识的重构，等效增加了网络有效深度
  6. 长上下文检索突破：Multi-Query NIAH 从 84.2 提升至 97.0
- **方法/发现**: 形式化 Sparsity Allocation 问题 → 实验发现 U 形定律 → 按最优配比缩放至 27B → Logit Lens / CKA 机制分析
- **局限/意义**: N-gram 记忆的有效性依赖于语言中大量局部静态模式的存在；确定性访问虽利于预取，但牺牲了动态检索的灵活性

## 引用
- **原始论文**: [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) | [清理后 Markdown](../../raw/2601.07372.md)
- **相关概念**: [[conditional_memory]] | [[sparsity_allocation]]
- **相关实体**: [[deepseek_ai]]
