---
id: gated_recurrent_unit
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [门控循环单元, GRU]
related_nodes: [gru_2014, gating_mechanism, recurrent_neural_network]
last_verified: 2026-08-03
---

# Gated Recurrent Unit (GRU)

## 定义
LSTM 的简化变体：合并遗忘门与输入门为更新门，去掉细胞状态，参数更少、计算更高效，性能与 LSTM 相当。

## 关键要点
- **两门结构**：更新门（融合遗忘+输入）、重置门
- **优势**：参数量约为 LSTM 的 3/4，训练更快
- **应用**：机器翻译、语音等序列任务的常用选择

## 来源
- [[gru_2014]] — 提出 GRU
