---
id: reward_modeling
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [奖励建模, 奖励模型]
related_nodes: [christiano_2017_rlhf, reinforcement_learning_from_human_feedback, reward_overoptimization]
last_verified: 2026-08-03
---

# Reward Modeling

## 定义
从人类偏好数据中学习奖励函数的技术：让人类比较模型输出对（A vs B），训练奖励模型预测偏好，再用于 RL 优化策略。

## 关键要点
- **偏好而非分数**：人类标注相对偏好更可靠，转化为 Bradley-Terry 排序损失
- **流程**：收集偏好 → 训练 RM → RL 阶段用 RM 打分
- **风险**：RM 是近似，过度优化会过拟合（见 reward overoptimization）

## 来源
- [[christiano_2017_rlhf]] — 从偏好学习奖励
