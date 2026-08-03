---
id: vision_transformer
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [ViT, 视觉Transformer]
related_nodes: [vit, transformer_architecture, shifted_window_attention]
last_verified: 2026-08-03
---

# Vision Transformer (ViT)

## 定义
将图像切分为 patch 序列直接输入标准 Transformer 编码器进行分类的架构，证明纯注意力可取代卷积成为 CV 基础架构。

## 关键要点
- **Patch 化**：16×16 patch 线性投影为 token + 位置编码
- **数据依赖**：大规模预训练（如 JFT-300M）下超越 CNN，中小数据不如
- **影响**：开启 ViT 家族（Swin、DeiT、DINO 等）成为 CV 主流范式

## 来源
- [[vit]] — ViT 原始论文
