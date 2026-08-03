---
id: guided_diffusion
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [引导扩散, 条件扩散]
related_nodes: [glide, diffusion_model, cascaded_diffusion]
last_verified: 2026-08-03
---

# Guided Diffusion

## 定义
在扩散采样过程中注入条件信号（类别标签、文本嵌入）控制生成内容的机制，如 GLIDE 用文本编码器条件化 + CLIP 引导。

## 关键要点
- **条件化方式**：交叉注意力/adaLN 注入条件嵌入
- **引导强度**：classifier guidance 或 classifier-free guidance 控制服从度
- **应用**：文生图、图像编辑（GLIDE 首次实现高质量文本引导）

## 来源
- [[glide]] — GLIDE 文本引导扩散
