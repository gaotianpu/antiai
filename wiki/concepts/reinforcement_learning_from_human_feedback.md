---
id: reinforcement_learning_from_human_feedback
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [RLHF, 人类反馈强化学习, 对齐]
related_nodes: [ouyang_2022_instructgpt, post_training]
last_verified: 2026-06-06
---

# Reinforcement Learning from Human Feedback (RLHF)

## 定义
RLHF 是一种使用人类偏好数据微调语言模型的对齐技术，使模型输出更符合人类期望。

## 三阶段流程
1. **SFT（监督微调）**: 在人类撰写的指令-回复数据上微调
2. **RM（奖励模型训练）**: 训练奖励模型预测人类偏好
3. **PPO 强化学习**: 使用 PPO 算法优化策略模型以最大化奖励

## 来源
- [[ouyang_2022_instructgpt]] — 系统提出 RLHF 在 LLM 中的应用
