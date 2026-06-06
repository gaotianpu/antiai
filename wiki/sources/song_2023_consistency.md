---
id: song_2023_consistency
type: source
tags: ["machine-learning", "computer-vision", "theoretical"]
aliases: ["Consistency Models", "一致性模型", "2303.01469"]
related_nodes: ["ho_2020_ddpm", "rombach_2021_latentdiffusion"]
arxiv_id: 2303.01469
authors: Yang Song et al.
authors_institution: OpenAI
last_verified: 2026-06-06
---

# Consistency Models

- **元数据**: arXiv | 2023 | **作者**: Yang Song et al. | **机构**: OpenAI | 相关: [[ho_2020_ddpm]], [[rombach_2021_latentdiffusion]]
- **概述**: 提出一致性模型，通过概率流 ODE 将任意噪声点直接映射到数据分布，实现单步生成，无需对抗训练。
- **关键要点**: 1. 单步生成替代迭代采样 2. 可蒸馏预训练扩散模型或独立训练 3. CIFAR-10 单步 FID 3.55
- **方法/发现**: 学习 ODE 轨迹到起点的映射，支持零样本数据编辑
- **局限/意义**: 大幅降低扩散模型推理成本，为实时应用铺平道路

## 引用
- **原始论文**: [arXiv:2303.01469](https://arxiv.org/abs/2303.01469) | [阅读笔记](../../raw/Generative/Consistency_models.md)
- **相关概念**: [[ho_2020_ddpm]], [[rombach_2021_latentdiffusion]]
