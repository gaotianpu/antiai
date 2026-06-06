---
id: diffusion_model
type: concept
tags: [machine-learning, theoretical, empirical-study]
aliases: [扩散模型, diffusion, DM]
related_nodes: [ho_2020_ddpm, rombach_2021_latentdiffusion, song_2023_consistency, imagen, glide, generative_model]
last_verified: 2026-06-06
---

# Diffusion Model

## 定义
扩散模型（Diffusion Model）是一种受非平衡热力学启发的生成模型：通过马尔可夫链逐步向数据添加高斯噪声（前向过程），然后训练神经网络反向去噪，从纯噪声中恢复出数据。

## 核心进展
- **DDPM**（[[ho_2020_ddpm]]）— 奠定扩散模型训练框架，证明其可生成高质量图像
- **LDM**（[[rombach_2021_latentdiffusion]]）— 在潜在空间执行扩散，大幅降低计算成本
- **Consistency Models**（[[song_2023_consistency]]）— 一次性采样替代多步去噪
- **文本条件扩散** — [[imagen]]、[[dall_e_v2]]、[[glide]] 推动文本到图像生成

## 关键特性
- **逐步生成**：多步迭代去噪，可权衡速度与质量
- **多样性**：从随机噪声出发，每次采样结果不同
- **可控性**：支持条件生成（文本、类别、图像引导）

## 来源
- [[ho_2020_ddpm]] — 去噪扩散概率模型奠基
- [[rombach_2021_latentdiffusion]] — 潜在空间扩散
- [[song_2023_consistency]] — 一致性模型快速采样
