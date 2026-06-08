---
id: mixture_of_experts
type: concept
tags: [machine-learning, theoretical]
aliases: [MoE, 混合专家, 稀疏专家, Sparse MoE]
related_nodes: [sparse_expert_review_2022, cheng_2026_engram, conditional_memory, sparsity_allocation, fedus_2021_switch, chi_2022_xmoe, deepseek_2024_v2, deepseek_2024_v3]
last_verified: 2026-06-06
---

# Mixture-of-Experts (MoE)

## 定义
混合专家（MoE）是一种条件计算范式：每个 token 仅激活模型参数的一个稀疏子集，从而在不显著增加计算量的前提下大幅扩展模型总参数量。

## 核心思想
- **稀疏激活**：前馈网络替换为多个并行"专家"FFN，路由网络选择 top-k 专家
- **容量扩展**：总参数量可增加数倍，计算量仅线性增长
- **负载均衡**：辅助损失确保各专家被均匀使用

## 关键发展
- Shazeer et al. (2017) — 首次将 MoE 引入神经网络
- GShard (Lepikhin 2020) — 分布式 MoE 训练框架
- Switch Transformer (Fedus 2021) — 简化为单专家路由
- X-MoE (Chi 2022) — 发现 MoE 表示坍塌问题并提出专家多样化正则化
- DeepSeek MoE (Dai 2024) — 细粒度专家分割 + 共享专家

## 来源
- [[sparse_expert_review_2022]] — 稀疏专家模型综述
- [[fedus_2021_switch]] — Switch Transformer：简化 MoE 路由为单专家
- [[chi_2022_xmoe]] — X-MoE：MoE 表示坍塌分析与专家多样化
- [[cheng_2026_engram]] — 条件记忆作为 MoE 的互补轴
