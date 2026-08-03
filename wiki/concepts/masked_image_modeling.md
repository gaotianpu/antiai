---
id: masked_image_modeling
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [掩码图像建模, MIM]
related_nodes: [mae, beit, ibot, self_supervised_learning]
last_verified: 2026-08-03
---

# Masked Image Modeling (MIM)

## 定义
图像版 MLM：掩码部分图像块/像素，训练模型重建被掩码内容（MAE 重建像素，BEiT 预测离散 token），是视觉自监督预训练的主流范式。

## 关键要点
- **MAE**：高掩码率（75%）+ 轻量解码器重建像素，效率高
- **BEiT**：预测 dVAE 离散 token，对标 BERT
- **iBOT**：在线 tokenizer 提供动态目标
- **地位**：与对比学习并列的视觉自监督两大路线

## 来源
- [[mae]] — MAE
- [[beit]] — BEiT
- [[ibot]] — iBOT
