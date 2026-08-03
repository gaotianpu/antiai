---
id: rl_evolution
type: synthesis
tags: ["machine-learning", "theoretical", "RL"]
aliases: ["强化学习演进", "RL演进路线", "深度强化学习发展"]
related_nodes: ["reinforcement_learning", "mnih_2013_dqn", "schulman_2017_ppo", "deepseek_2025_r1", "post_training"]
last_verified: 2026-08-03
---

# RL 演进路线（2013-2025）

## 概述

从 DQN 深度化突破到 LLM 推理能力涌现，强化学习经历五个阶段：**价值学习深度化 → 策略优化与连续控制 → 实用化标准 → 人类反馈对齐 → LLM 推理时代**，另有模仿学习旁支（自动驾驶）。

## 演进路线

### ① 深度化突破（2013-2015）：价值学习

- [[mnih_2013_dqn]] — DQN 将 CNN 与 Q-learning 结合，经验回放打破数据相关性 + Target Network 稳定训练，49 个 Atari 游戏达到人类水平
- **范式意义**：确立"深度网络 + 经验回放"的深度 RL 基本盘

### ② 策略优化与连续控制（2015-2016）

- [[schulman_2015_trpo]] — TRPO 用 KL 散度约束新旧策略距离，保证单调改进，但需二阶优化
- [[lillicrap_2015_ddpg]] — DDPG 将 DQN 思路扩展到连续动作空间（确定性策略梯度 + 经验回放 + 软更新）
- [[mnih_2016_a2c]] — A3C/A2C 用异步并行 Actor 的自然去相关替代经验回放
- **范式意义**：从"只学价值"走向"策略 + 价值"双线并行，覆盖连续控制

### ③ 实用化标准（2017）

- [[schulman_2017_ppo]] — PPO 以裁剪代理目标（clip）实现 TRPO 的稳定性 + 一阶优化的简单性
- **范式意义**：成为 RLHF、游戏 AI、机器人控制的工业界事实标准

### ④ 人类反馈对齐（2017-2022）

- [[christiano_2017_rlhf]] — 从人类偏好对（A vs B）学习奖励模型（[[reward_modeling]]）
- [[ouyang_2022_instructgpt]] — InstructGPT 将 RLHF 用于 LLM 指令遵循（[[instruction_tuning]] + [[reinforcement_learning_from_human_feedback]]）
- [[gao_2022_rmoveroptimization]] — 发现奖励过优化（[[goodhart_law]]），提出 scaling law 预测最优优化强度（[[reward_overoptimization]]）
- **范式意义**：RL 从"打游戏"转向"对齐人类价值观"，成为 LLM 后训练核心（见 [[post_training]]）

### ⑤ LLM 推理时代（2024-2025）

- [[deepseek_2025_r1]] — GRPO（[[grpo]]）无 critic 的组相对策略优化，纯 RL 激发推理能力，涌现自我反思/验证，Nature 封面报道
- **范式意义**：RL 从"对齐偏好"进一步走向"激发能力"，RL 训练信号从人类反馈扩展到可验证奖励

### 旁支：模仿学习与自动驾驶（2010-2020）

- [[prakash_2010_dagger]] — DAgger（[[dagger_algorithm]]）交互式聚合专家数据，缓解行为克隆分布偏移（[[imitation_learning]]）
- [[rhinehart_2020_imitative]] — Imitative Models（[[imitative_models]]）融合模仿分布与目标导向规划
- **范式意义**：免奖励信号的学习路线，与基于奖励的 RL 互补

## 关键转折点

| 转折 | 时间 | 意义 |
|:---|:---|:---|
| DQN 超越人类 | 2013 | 深度 RL 可行性证明 |
| PPO 出现 | 2017 | 稳定易用，RL 工业化的前提 |
| RLHF 成型 | 2017-2022 | RL 从控制任务走向人类对齐 |
| GRPO/纯 RL 推理 | 2025 | 无 critic 简化 + 能力涌现 |

## 相关概念

- [[reinforcement_learning]] — 强化学习总纲
- [[policy_gradient]] — 策略梯度家族
- [[actor_critic]] — 策略-价值双网络框架
- [[reward_modeling]] / [[reward_overoptimization]] — 奖励的建模与风险
- [[grpo]] — 新一代 LLM RL 算法
- [[post_training]] — RLHF 在后训练中的位置
