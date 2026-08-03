---
id: vector_quantized_tokenizer
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [向量量化分词器, VQ tokenizer]
related_nodes: [beit_2, online_tokenizer, masked_image_modeling]
last_verified: 2026-08-03
---

# Vector-Quantized Tokenizer

## 定义
将图像连续特征映射为离散码本索引的模型（VQ-VAE/dVAE 类）：把视觉内容 token 化，使 BERT 式离散预测目标可以迁移到图像。

## 关键要点
- **离散化**：特征 → 最近码本向量 → 码索引
- **BEiT-2 应用**：VQ-KD 用蒸馏提升 tokenizer 质量，优于 dVAE
- **地位**：BEiT/MAGVIT 等"视觉 BERT"路线的关键组件

## 来源
- [[beit_2]] — BEiT-2 向量量化 tokenizer
