---
id: retention_mechanism
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [保留机制, RetNet Retention]
related_nodes: [retnet_2023, transformer_architecture, parallel_training_sequential_inference]
last_verified: 2026-08-03
---

# Retention Mechanism

## 定义
RetNet 提出的序列建模机制：用线性递推形式表达注意力，支持并行训练（类似注意力）与 O(1) 增量推理（类似 RNN）。

## 关键要点
- **三形态**：并行（训练）/ 循环（推理）/ 分块（混合）三种等价计算形式
- **线性复杂度**：推理时状态大小固定，不随序列长度增长
- **目标**：兼具 Transformer 的训练并行性与 RNN 的推理效率

## 来源
- [[retnet_2023]] — 提出保留网络 RetNet
