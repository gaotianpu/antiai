---
id: gu_2023_mamba
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["Mamba", "状态空间模型", "2312.00752"]
related_nodes: [state_space_model, retention_mechanism]
authors: Albert Gu, Tri Dao
authors_institution: CMU / Princeton
arxiv_id: 2312.00752
last_verified: 2026-08-03
---

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- **元数据**: arXiv 2312.00752 | 2023 | **作者**: Albert Gu, Tri Dao | **机构**: CMU / Princeton
- **概述**: 提出 Mamba：选择性状态空间模型（S6），输入依赖的时变参数使 SSM 具备内容感知的选择能力，以线性复杂度在长序列任务上匹敌甚至超越 Transformer。
- **新颖概念**: [[state_space_model]]
- **关键要点**: 1. 选择性机制：按输入动态决定信息保留/遗忘 2. 硬件感知的并行扫描算法 3. 线性时间、常数内存推理
- **方法/发现**: 语言建模在 1M token 上下文上优于同规模 Transformer；序列长度线性缩放（对比注意力的平方）
- **局限/意义**: 引发"后 Transformer"架构浪潮（Jamba、Zamba、RWKV 等混合路线）；与注意力融合的混合架构成为 2024-2025 主流探索方向

## 引用
- **原始论文**: [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- **相关概念**: [[state_space_model]], [[retention_mechanism]]
