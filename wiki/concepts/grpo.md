---
id: grpo
type: concept
tags: [empirical-study, reinforcement_learning]
aliases: ["GRPO", "Group Relative Policy Optimization", "组相对策略优化"]
related_nodes: [reinforcement_learning, policy_gradient, deepseek_2025_r1]
---

# Group Relative Policy Optimization (GRPO)

## 概述
GRPO 是 DeepSeek-R1 使用的强化学习算法，替代 PPO 中的 critic 网络，通过组内采样结果的相对比较来估计优势函数。

## 核心机制
- PPO 需要价值网络（critic）估计状态价值
- GRPO 对同一 prompt 采样多个输出，将组内输出的奖励归一化后作为优势估计
- 无需维护 critic 网络，降低训练复杂度

## 优势
- 简化 RL 训练管线（无需 critic）
- 天然适配可验证任务（数学、代码等有明确正确性判断的任务）

## 来源
- [[deepseek_2025_r1]]
