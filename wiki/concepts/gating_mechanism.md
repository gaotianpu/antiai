---
id: gating_mechanism
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [门控机制, 门控]
related_nodes: [hochreiter_1997_lstm, long_short_term_memory, gated_recurrent_unit]
last_verified: 2026-08-03
---

# Gating Mechanism

## 定义
用可学习的门（0-1 区间值）控制信息流通过比例的机制，决定"记住什么、遗忘什么、输出什么"，是 LSTM/GRU 的核心设计。

## 关键要点
- **门的形态**：sigmoid 输出 0-1，与候选值逐元素相乘
- **作用**：缓解梯度消失、提供可微分的软记忆开关
- **泛化**：门控思想也出现在 GLU、Gated Attention 等现代架构中

## 来源
- [[hochreiter_1997_lstm]] — LSTM 三门结构
