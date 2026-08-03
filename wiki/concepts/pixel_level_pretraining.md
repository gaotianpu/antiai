---
id: pixel_level_pretraining
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [像素级预训练, 像素生成预训练]
related_nodes: [igpt, generative_pretraining, masked_image_modeling]
last_verified: 2026-08-03
---

# Pixel-Level Pretraining

## 定义
直接在原始像素序列上做生成式预训练（如 iGPT 将图像像素当作 token 训练自回归/自编码目标），探索"像素即语言"的路线。

## 关键要点
- **像素序列化**：按行列展开像素，套用语言模型目标
- **结果**：特征质量低于卷积/对比方法，但证明生成目标可学
- **意义**：与 ViT、MAE 共同构成"视觉 token 化"谱系

## 来源
- [[igpt]] — Image GPT
