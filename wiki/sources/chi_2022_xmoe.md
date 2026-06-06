---
id: chi_2022_xmoe
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["X-MoE", "Representation Collapse in MoE", "2204.09179"]
related_nodes: ["fedus_2021_switch"]
arxiv_id: 2204.09179
authors: Zewen Chi et al.
authors_institution: Microsoft
last_verified: 2026-06-06
---

# On the Representation Collapse of Sparse Mixture of Experts

- **元数据**: arXiv | 2022 | **作者**: Zewen Chi et al. | **机构**: Microsoft | 相关: [[fedus_2021_switch]]
- **概述**: 分析 MoE 的表示坍塌问题（token 向专家质心聚类），提出低维超球路由缓解坍塌。
- **新颖概念**: —
- **关键要点**: 1. 表示坍塌现象 2. 低维超球路由 3. 多语言预训练一致提升
- **方法/发现**: 路由分数在低维超球面上估计，促进更均匀的专家利用
- **局限/意义**: 揭示了 MoE 训练的核心缺陷并给出有效缓解方案

## 引用
- **原始论文**: [arXiv:2204.09179](https://arxiv.org/abs/2204.09179) | [阅读笔记](../../raw/X-MoE.md)
- **相关概念**: [[fedus_2021_switch]]
