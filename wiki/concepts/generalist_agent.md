---
id: generalist_agent
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [通用智能体, 通才智能体]
related_nodes: [reed_2022_gato, foundation_model, imitation_learning]
last_verified: 2026-08-03
---

# Generalist Agent

## 定义
以单一模型跨多任务、多模态、多环境执行决策的智能体范式：Gato 将文本/图像/机器人动作统一 token 化，单模型在 604 个任务上达到专家水平 50%+。

## 关键要点
- **统一表示**：多模态输入输出 token 化，与 LLM 同构
- **训练方式**：离线 RL + 行为克隆式监督的混合
- **与基础模型的关系**：通用智能体是 foundation model 思想在决策/具身侧的延伸（RT-1/RT-2 后续发展）

## 来源
- [[reed_2022_gato]] — Gato 通用智能体
