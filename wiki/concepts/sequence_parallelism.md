---
id: sequence_parallelism
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [序列并行, 序列分块并行]
related_nodes: [liu_2023_ringattention, flash_attention]
last_verified: 2026-08-03
---

# Sequence Parallelism

## 定义
将序列维度分块分布到多设备并行计算的策略：Ring Attention 以环形传递 KV 块 + 分块注意力，使上下文长度随设备数线性扩展，突破单设备上限。

## 关键要点
- **通信-计算重叠**：环形传递与注意力计算流水并行
- **与 FA 互补**：分块注意力（tiling）是序列并行的计算基础
- **意义**：百万级上下文训练的基础并行方案

## 来源
- [[liu_2023_ringattention]] — Ring Attention
