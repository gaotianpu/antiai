---
id: recurrent_neural_network
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [循环神经网络, RNN]
related_nodes: [elman_1990_rnn, long_short_term_memory, gated_recurrent_unit]
last_verified: 2026-08-03
---

# Recurrent Neural Network (RNN)

## 定义
按时间步递归处理序列的神经网络：每个时刻的隐藏状态由当前输入与上一时刻状态共同更新，天然建模时序依赖。

## 关键要点
- **递归结构**：$h_t = f(h_{t-1}, x_t)$，参数跨时间共享
- **局限**：梯度消失/爆炸、无法并行、长依赖捕获困难
- **演进**：LSTM/GRU 缓解梯度问题；Transformer 取代其主流地位

## 来源
- [[elman_1990_rnn]] — 提出 Elman RNN
