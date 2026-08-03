---
id: multimodal_mixture_of_encoders_decoders
type: concept
tags: [NLP, computer-vision, machine-learning, empirical-study]
aliases: [多模态编解码混合, MoE-D]
related_nodes: [blip, encoder_decoder_architecture]
last_verified: 2026-08-03
---

# Multimodal Mixture of Encoders-Decoders (MoE-D)

## 定义
BLIP 提出的统一架构：一个共享 Transformer 通过注意力掩码切换三种角色——单模态编码器、图像-文本编码器、图像-文本解码器，统一理解与生成。

## 关键要点
- **一网三用**：掩码切换角色，参数共享，无需三个独立模型
- **训练目标**：ITC（对比）+ ITM（匹配）+ LM（生成）三目标联合
- **地位**：BLIP 系列统一图文理解-生成的基础架构

## 来源
- [[blip]] — BLIP 引导式预训练
