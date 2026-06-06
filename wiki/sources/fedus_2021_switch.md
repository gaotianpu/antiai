---
id: fedus_2021_switch
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["Switch Transformer", "开关Transformer", "2101.03961"]
related_nodes: []
arxiv_id: 2101.03961
authors: William Fedus et al.
authors_institution: Google
last_verified: 2026-06-06
---

# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- **元数据**: arXiv | 2021 | **作者**: William Fedus et al. | **机构**: Google
- **概述**: 提出简化稀疏路由的 Switch Transformer，将 MoE 扩展到 1.6T 参数，训练速度提升 7 倍。
- **关键要点**: 1. 简化的 Top-1 路由 2. 1.6T 参数 3. 7× 训练加速 4. 负载均衡损失
- **方法/发现**: 每个 token 仅路由到一个专家，大幅简化 MoE 训练并提升效率
- **局限/意义**: 推动 MoE 在大规模语言模型中的广泛应用，后续 MoE 模型的基础

## 引用
- **原始论文**: [arXiv:2101.03961](https://arxiv.org/abs/2101.03961) | [阅读笔记](../../raw/Switch_Transformers.md)
