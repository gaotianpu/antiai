---
id: generative_model
type: concept
tags: [machine-learning, theoretical, empirical-study]
aliases: [生成模型, generative]
related_nodes: [ho_2020_ddpm, rombach_2021_latentdiffusion, imagen, dall_e_v2, diffusion_model]
last_verified: 2026-06-06
---

# Generative Model

## 定义
生成模型（Generative Model）是一类学习数据分布 $p(x)$ 或条件分布 $p(x|y)$ 的模型，能够从学习到的分布中采样新样本。与判别模型 $p(y|x)$ 不同，生成模型建模的是数据的产生过程。

## 主要范式
| 范式 | 基本原理 | 代表模型 |
|:---|:---|:---|
| GAN | 生成器 vs 判别器对抗训练 | StyleGAN, BigGAN |
| VAE | 变分下界 + 重参数化 | VQ-VAE, NVAE |
| **扩散模型** | 正向加噪 → 反向去噪 | [[ho_2020_ddpm]], [[imagen]] |
| 自回归模型 | 逐个 token 生成序列 | GPT, PixelCNN |
| 流模型 | 可逆变换 + 精确对数似然 | Glow, RealNVP |

## 来源
- [[ho_2020_ddpm]] — 去噪扩散概率模型
- [[rombach_2021_latentdiffusion]] — 潜在扩散模型
- [[imagen]] — 文本到图像扩散
- [[dall_e_v2]] — 两阶段文本到图像生成
