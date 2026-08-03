---
id: world_model
type: concept
tags: [machine-learning, theoretical, RL]
aliases: [世界模型]
related_nodes: [hafner_2023_dreamerv3, model_based_rl, generative_model]
last_verified: 2026-08-03
---

# World Model

## 定义
学习环境动态的模型：预测状态转移与奖励，使智能体可以在"想象"中规划与训练，而非仅依赖真实交互。DreamerV3 以潜在空间世界模型在 150+ 任务达到 SOTA。

## 关键要点
- **潜在空间预测**：压缩观测为潜在状态，预测转移与奖励
- **想象训练**：策略在模型生成的想象轨迹上学习（model-based RL 的核心）
- **扩展应用**：视频生成（Genie）、世界模拟等延续此范式

## 来源
- [[hafner_2023_dreamerv3]] — DreamerV3
