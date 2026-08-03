---
id: latent_diffusion
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [潜在扩散, 潜空间扩散, LDM]
related_nodes: [rombach_2021_latentdiffusion, diffusion_model]
last_verified: 2026-08-03
---

# Latent Diffusion

## 定义
将扩散过程从像素空间转移到预训练自编码器的低维潜在空间，大幅降低计算成本，配合交叉注意力实现文本条件控制，即 Stable Diffusion。

## 关键要点
- **两阶段**：自编码器压缩到潜空间 + 潜空间扩散生成
- **效率**：计算量比像素空间扩散降低数量级
- **条件控制**：交叉注意力注入文本/布局等条件

## 来源
- [[rombach_2021_latentdiffusion]] — Latent Diffusion / Stable Diffusion
