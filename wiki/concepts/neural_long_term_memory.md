---
id: neural_long_term_memory
type: concept
tags: [machine-learning, empirical-study]
aliases: [神经长期记忆, 测试时记忆]
related_nodes: [behrouz_2024_titans, conditional_memory, engram]
last_verified: 2026-08-03
---

# Neural Long-Term Memory

## 定义
Titans 提出的记忆范式：以"测试时学习"（梯度更新）为记忆写入机制，模型在推理中持续更新神经记忆模块，与注意力/门控并行融合，支撑百万级上下文。

## 关键要点
- **在线学习**：记忆随推理动态更新（对比固定权重）
- **与静态记忆互补**：Engram（条件记忆）是 O(1) 静态查表，神经记忆是可学习的动态存储
- **融合模式**：门控、并行、级联三种注意力-记忆结合方式

## 来源
- [[behrouz_2024_titans]] — Titans 测试时记忆
