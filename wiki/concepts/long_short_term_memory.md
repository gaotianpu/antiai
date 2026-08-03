---
id: long_short_term_memory
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [长短期记忆, LSTM]
related_nodes: [hochreiter_1997_lstm, gating_mechanism, recurrent_neural_network]
last_verified: 2026-08-03
---

# Long Short-Term Memory (LSTM)

## 定义
通过门控机制（遗忘门/输入门/输出门）控制信息流的循环网络，解决 RNN 长序列梯度消失问题，是深度学习早期最成功的序列模型。

## 关键要点
- **细胞状态**：一条贯穿时间的"传送带"，梯度可无损流动
- **三门结构**：遗忘门决定丢弃什么，输入门决定写入什么，输出门决定读出什么
- **历史地位**：2010 年代语音/机器翻译/时序预测的标准模型

## 来源
- [[hochreiter_1997_lstm]] — 提出 LSTM
