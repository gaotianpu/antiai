---
id: imitative_models
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [模仿模型, Imitative Models]
related_nodes: [rhinehart_2020_imitative, imitation_learning]
last_verified: 2026-08-03
---

# Imitative Models

## 定义
Rhinehart 等提出的驾驶策略框架：以专家轨迹定义分布（模仿），结合目标导向规划（优化），融合模仿学习的稳健与规划的目标可达。

## 关键要点
- **分布建模**：学习专家轨迹的分布而非仅均值动作
- **目标条件**：给定目标状态，选择高概率且可达的轨迹
- **优势**：兼顾安全（不偏离示范）与目标达成（长程规划）

## 来源
- [[rhinehart_2020_imitative]] — Imitative Models
