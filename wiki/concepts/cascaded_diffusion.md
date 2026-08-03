---
id: cascaded_diffusion
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [级联扩散, 级联生成]
related_nodes: [imagen, diffusion_model]
last_verified: 2026-08-03
---

# Cascaded Diffusion

## 定义
用多个串联的扩散模型逐级生成：基础模型生成低分辨率图像，超分辨率模型逐级放大细化，降低单模型生成高分辨率图像的难度。

## 关键要点
- **分而治之**：每级模型专注一个任务（生成/上采样）
- **噪声调节**：Imagen 用 conditioning augmentation 缓解级联误差累积
- **应用**：高分辨率文生图、视频生成的常见管线

## 来源
- [[imagen]] — Imagen 级联扩散
