---
id: hierarchical_image_generation
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [层次化图像生成, 分层生成]
related_nodes: [dall_e_v2, generative_model]
last_verified: 2026-08-03
---

# Hierarchical Image Generation

## 定义
分阶段生成图像的范式：先由先验模型生成 CLIP 图像嵌入（粗语义），再以嵌入为条件生成高分辨率图像（细细节），如 DALL·E 2。

## 关键要点
- **两阶段**：先验（文本→图像嵌入）+ 解码器（嵌入→图像）
- **CLIP 隐空间**：语义在 CLIP 空间对齐文本与图像
- **可控性**：中间嵌入可编辑，支持图像变体生成

## 来源
- [[dall_e_v2]] — DALL·E 2 层次化生成
