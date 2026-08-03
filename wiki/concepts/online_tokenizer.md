---
id: online_tokenizer
type: concept
tags: [machine-learning, computer-vision, empirical-study]
aliases: [在线分词器, 在线标记器]
related_nodes: [ibot, masked_image_modeling, vector_quantized_tokenizer]
last_verified: 2026-08-03
---

# Online Tokenizer

## 定义
与主网络联合训练、动态更新的视觉 tokenizer（对比固定预训练 tokenizer）：tokenizer 与表示学习相互促进，目标随训练演化。

## 关键要点
- **静态 vs 动态**：固定 dVAE 目标不变；在线 tokenizer 与网络共同进化
- **iBOT 方案**：在线 tokenizer + EMA 教师提供稳定目标
- **意义**：缓解 MIM 对预训练 tokenizer 的依赖

## 来源
- [[ibot]] — iBOT 在线 tokenizer
