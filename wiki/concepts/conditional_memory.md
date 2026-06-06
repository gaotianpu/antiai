---
id: conditional_memory
type: concept
tags: [machine-learning, NLP, empirical-study]
aliases: [条件记忆, static lookup, 静态查表]
related_nodes: [cheng_2026_engram, sparsity_allocation]
last_verified: 2026-06-06
---

# Conditional Memory

## 定义
条件记忆（Conditional Memory）是一种通过稀疏查表（sparse lookup）操作检索静态嵌入的建模原语，与条件计算（Conditional Computation, 即 MoE）形成互补。两者的区别在于：

| 维度 | 条件计算 (MoE) | 条件记忆 (Engram) |
|:---|:---|:---|
| 操作类型 | 动态前向计算 | 静态嵌入查表 |
| 计算模式 | 稀疏激活参数 | 确定性哈希索引 |
| 典型耗时 | O(FLOPs) | O(1) |
| 适用任务 | 组合推理 | 知识/模式检索 |

## 核心思想
语言信号存在异质性：大部分文本（命名实体、固定搭配、格式模板）是局部、静态、高度刻板的，适合用低成本查表替代计算式重构。将这类操作从神经网络中剥离，可释放计算资源用于更高层次的推理。

## 来源
- [[cheng_2026_engram]] — 首次提出条件记忆轴，实例化为 Engram 模块
