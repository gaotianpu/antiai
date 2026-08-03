---
id: parallel_training_sequential_inference
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [并行训练顺序推理, 训练推理一致性]
related_nodes: [retnet_2023, retention_mechanism]
last_verified: 2026-08-03
---

# Parallel Training & Sequential Inference

## 定义
架构设计目标：训练阶段可像 Transformer 一样并行计算，推理阶段退化为顺序增量计算，兼顾训练效率与推理成本。

## 关键要点
- **训练并行性**：充分利用 GPU 大规模并行
- **推理顺序性**：O(1) 常数状态，内存不随上下文增长
- **意义**：长上下文场景（如百万 token）的实用化前提

## 来源
- [[retnet_2023]] — RetNet 的核心设计目标
