---
id: dao_2024_mamba2
type: source
tags: ["machine-learning", "theoretical"]
aliases: ["Mamba-2", "SSD", "2405.21060"]
related_nodes: [state_space_model, multi_head_attention]
authors: Tri Dao, Albert Gu
authors_institution: Stanford / CMU
arxiv_id: 2405.21060
last_verified: 2026-08-03
---

# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

- **元数据**: arXiv 2405.21060 | 2024 | **作者**: Tri Dao, Albert Gu | **机构**: Stanford / CMU
- **概述**: 提出状态空间对偶性（SSD）：证明 SSM 与注意力存在理论等价（半可分矩阵视角），统一两大家族；Mamba-2 据此获得类注意力并行效率。
- **新颖概念**: [[state_space_model]]
- **关键要点**: 1. SSD 框架：SSM 对应结构化（半可分）矩阵，注意力是其特例 2. 块分解算法同时获得 SSM 的线性效率与注意力的并行性 3. Mamba-2 训练吞吐超 Mamba-1 数倍
- **方法/发现**: 统一了"线性注意力 vs 标准注意力"的争论——两者是同一矩阵族的两个极端；为混合架构提供理论基石
- **局限/意义**: 理论贡献深远：解释为何 SSM 能匹敌 Transformer，并指引下一代高效注意力设计

## 引用
- **原始论文**: [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **相关概念**: [[state_space_model]], [[multi_head_attention]]
