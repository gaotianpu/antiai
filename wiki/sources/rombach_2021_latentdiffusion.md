---
id: rombach_2021_latentdiffusion
type: source
tags: ["machine-learning", "computer-vision"]
aliases: ["Latent Diffusion", "LDM", "Stable Diffusion", "2112.10752"]
related_nodes: ["ho_2020_ddpm"]
arxiv_id: 2112.10752
authors: Robin Rombach et al.
authors_institution: Ludwig Maximilian University of Munich
last_verified: 2026-06-06
---

# High-Resolution Image Synthesis with Latent Diffusion Models

- **元数据**: arXiv | 2021 | **作者**: Robin Rombach et al. | **机构**: LMU Munich | 相关: [[ho_2020_ddpm]]
- **概述**: 将扩散模型引入预训练自编码器的潜在空间，大幅降低计算成本，引入交叉注意力实现文本/条件控制，即 Stable Diffusion。
- **关键要点**: 1. 潜在空间扩散显著降低计算量 2. 交叉注意力实现文本/边框条件控制 3. 成为 Stable Diffusion 的基础架构
- **方法/发现**: 在 VAE 潜在空间中训练扩散模型，首次在复杂度降低和细节保留之间达到近最优平衡
- **局限/意义**: 开源生态的核心底座，催生了整个 Stable Diffusion 社区

## 引用
- **原始论文**: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | [阅读笔记](../../raw/Generative/LatentDiffusion.md)
- **相关概念**: [[ho_2020_ddpm]]
