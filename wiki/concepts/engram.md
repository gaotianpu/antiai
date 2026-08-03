---
id: engram
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [条件记忆模块, Engram]
related_nodes: [cheng_2026_engram, conditional_memory, mixture_of_experts]
last_verified: 2026-08-03
---

# Engram

## 定义
条件记忆（conditional memory）的实例化模块：将经典 N-gram 嵌入改造为 O(1) 查表，作为稀疏性的新轴心，与 MoE 互补。

## 关键要点
- **O(1) 查表**：静态知识以查表而非前向计算获取
- **与 MoE 互补**：MoE 做动态计算路由，Engram 做静态知识检索
- **U 形 Scaling Law**：指导 MoE 与静态记忆间的最优容量配比

## 来源
- [[cheng_2026_engram]] — 条件记忆 / Engram
