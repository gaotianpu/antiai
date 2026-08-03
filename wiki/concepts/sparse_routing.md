---
id: sparse_routing
type: concept
tags: [machine-learning, empirical-study]
aliases: [稀疏路由, 稀疏专家路由]
related_nodes: [fedus_2021_switch, mixture_of_experts]
last_verified: 2026-08-03
---

# Sparse Routing

## 定义
MoE 的核心机制：每个 token 仅路由到少量专家（如 Top-1/2）而非全部，实现条件计算——总参数量巨大但单次前向只激活一部分。

## 关键要点
- **激活率**：Switch 的 Top-1 路由激活率约 1/专家数
- **负载均衡**：需辅助损失防止 token 聚集到少数专家
- **规模**：Switch Transformer 借此扩展到 1.6T 参数，训练提速 7 倍

## 来源
- [[fedus_2021_switch]] — Switch Transformer
