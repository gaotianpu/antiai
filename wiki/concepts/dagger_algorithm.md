---
id: dagger_algorithm
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [DAgger, 数据集聚合]
related_nodes: [prakash_2010_dagger, imitation_learning]
last_verified: 2026-08-03
---

# DAgger

## 定义
Dataset Aggregation：在策略自身遇到的轨迹上反复查询专家标注并聚合训练，缓解行为克隆的分布偏移，实现策略的在线改进。

## 关键要点
- **核心循环**：执行策略 → 收集轨迹 → 专家标注 → 聚合重训
- **解决 covariate shift**：训练分布 = 部署分布
- **代价**：需要可随时查询的专家（在线人类/最优控制器）

## 来源
- [[prakash_2010_dagger]] — DAgger 及其自动驾驶分析
