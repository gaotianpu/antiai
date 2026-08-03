---
id: adaptive_budget_allocation
type: concept
tags: [machine-learning, empirical-study]
aliases: [自适应预算分配, 动态参数分配]
related_nodes: [zhang_2023_adalora, low_rank_adaptation]
last_verified: 2026-08-03
---

# Adaptive Budget Allocation

## 定义
AdaLoRA 提出的参数预算分配策略：按模块重要性动态分配 LoRA 参数预算，通过 SVD 参数化 + 剪枝不重要奇异值实现自适应调节。

## 关键要点
- **重要性度量**：奇异值大小反映参数重要性，剪枝小奇异值
- **动态调节**：训练中持续调整各模块预算
- **效果**：同等参数下优于均匀分配（标准 LoRA）

## 来源
- [[zhang_2023_adalora]] — AdaLoRA
