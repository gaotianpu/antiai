---
id: goodhart_law
type: concept
tags: [machine-learning, theoretical]
aliases: [古德哈特定律]
related_nodes: [gao_2022_rmoveroptimization, reward_overoptimization, reward_modeling]
last_verified: 2026-08-03
---

# Goodhart's Law

## 定义
"当指标成为目标，它就不再是好指标"（When a measure becomes a target, it ceases to be a good measure）——优化代理指标会导致其与真实目标脱节。

## 关键要点
- **RLHF 场景**：优化奖励模型分数 → 奖励过优化，真实质量先升后降
- **普遍性**：教育、金融、AI 对齐中的通用现象
- **应对**：限制优化强度、奖励模型校准、过程监督

## 来源
- [[gao_2022_rmoveroptimization]] — 奖励过优化的实证研究
