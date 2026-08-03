---
id: ppo
type: concept
tags: [NLP, machine-learning, empirical-study, RL]
aliases: [PPO, 近端策略优化]
related_nodes: [ouyang_2022_instructgpt, proximal_policy_optimization, reinforcement_learning_from_human_feedback]
last_verified: 2026-08-03
---

# PPO (RLHF 应用)

## 定义
PPO 在 LLM 对齐中的关键应用：InstructGPT 等用 PPO 优化语言模型策略，使其输出获得高奖励模型分数，同时用 KL 惩罚防止偏离初始模型。

## 关键要点
- **策略优化**：语言模型作为 Actor，奖励模型提供信号
- **KL 约束**：惩罚与 SFT 模型的 KL 散度，保持输出自然性
- **地位**：RLHF 的事实标准组件

## 来源
- [[ouyang_2022_instructgpt]] — InstructGPT 用 PPO 对齐
