---
id: ho_2020_ddpm
type: source
tags: ["machine-learning", "theoretical"]
aliases: ["DDPM", "Denoising Diffusion Probabilistic Models", "2006.11239"]
related_nodes: [rombach_2021_latentdiffusion, song_2023_consistency]
arxiv_id: 2006.11239
authors: Jonathan Ho et al.
authors_institution: UC Berkeley
last_verified: 2026-06-06
---

# Denoising Diffusion Probabilistic Models

- **元数据**: arXiv | 2020 | **作者**: Jonathan Ho et al. | **机构**: UC Berkeley
- **概述**: 提出去噪扩散概率模型（DDPM），将扩散模型与去噪分数匹配建立联系，实现高质量图像生成。
- **新颖概念**: [[diffusion_model]]
- **关键要点**: 1. 加权变分边界训练 2. 渐进式有损解压缩 3. CIFAR-10 上 FID 3.17 SOTA
- **方法/发现**: 扩散模型 = 去噪分数匹配 + Langevin 动力学，训练加权变分下界
- **局限/意义**: 奠定了现代扩散模型的基础，后续 Stable Diffusion、DALL-E 的基石

## 引用
- **原始论文**: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | [阅读笔记](../../raw/Generative/DDPM.md)
