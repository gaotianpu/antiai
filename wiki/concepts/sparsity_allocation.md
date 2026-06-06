---
id: sparsity_allocation
type: concept
tags: [machine-learning, theoretical, empirical-study]
aliases: [稀疏分配, 稀疏性分配, U-shaped scaling law]
related_nodes: [cheng_2026_engram, conditional_memory, mixture_of_experts]
last_verified: 2026-06-06
---

# Sparsity Allocation

## 定义
Sparsity Allocation（稀疏分配）是指在固定参数预算下，如何将容量在神经计算（MoE 专家）和静态记忆（Engram）之间最优分配的问题。

## U 形 Scaling Law
Cheng et al. (2026) 通过实验发现，模型性能与记忆分配比例呈 U 形关系：
- **记忆过少**：模型仍依赖计算模拟查表，浪费有效深度
- **记忆过多**：计算容量不足，无法处理需深度推理的任务
- **最优配比**：处于 U 形底部，此时 MoE 与 Engram 互补最佳

## 来源
- [[cheng_2026_engram]] — 首次形式化此问题并发现 U 形定律
