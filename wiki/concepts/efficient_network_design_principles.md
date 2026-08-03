---
id: efficient_network_design_principles
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [高效网络设计准则, 4条设计原则]
related_nodes: [ma_2018_shufflenet_v2, neural_architecture_search]
last_verified: 2026-08-03
---

# Efficient Network Design Principles

## 定义
ShuffleNetV2 提出的高效网络设计四准则：1) 输入输出同通道最省内存；2) 过量分组卷积增加 MAC；3) 网络碎片化降低并行度；4) 元素级操作不可忽略。

## 关键要点
- **核心洞察**：FLOPs 不能准确衡量实际速度，直接以内存访问成本（MAC）为设计指标
- **直接后果**：同一 FLOPs 下，ShuffleNetV2 比 v1 更快
- **意义**：将"效率设计"从经验转向可度量原则

## 来源
- [[ma_2018_shufflenet_v2]] — ShuffleNetV2 设计准则
