---
id: sparse_connection
type: concept
tags: [machine-learning, theoretical]
aliases: [稀疏连接, 稀疏结构]
related_nodes: [zhu_2018_sparsenet, dense_connection, model_pruning]
last_verified: 2026-08-03
---

# Sparse Connection

## 定义
网络中仅保留部分连接（边）的结构设计：SparseNet 将 DenseNet 的密集连接稀疏化，把连接复杂度从 O(L²) 降至 O(L)，性能不减反升。

## 关键要点
- **复杂度控制**：连接数随深度线性而非平方增长
- **规律发现**：深层网络每层仅需少数稀疏连接即可充分训练
- **意义**：为"连接稀疏性"作为正则与效率手段提供证据

## 来源
- [[zhu_2018_sparsenet]] — SparseNet
