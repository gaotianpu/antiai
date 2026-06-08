---
id: multi_token_prediction
type: concept
tags: [empirical-study, NLP]
aliases: ["MTP", "Multi-Token Prediction", "多 Token 预测"]
related_nodes: [transformer_architecture, loss_function, deepseek_2024_v3]
---

# Multi-Token Prediction (MTP)

## 概述
MTP 是一种训练目标，要求模型同时预测未来多个位置的 token，而非标准的仅预测下一个 token。

## 核心机制
- 标准语言模型训练：预测下一个 token
- MTP：在模型顶部添加多个预测头，每个头负责预测不同位置的未来 token
- 训练时对各头的损失加权求和

## 优势
- 更强的表示学习信号
- 在 HumanEval 等代码生成基准上提升约 12%（Meta 同期工作）
- DeepSeek-V3 将其作为核心训练目标之一

## 来源
- [[deepseek_2024_v3]]
