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
- [[bai_2022_constitutional_ai]] — 宪法 AI：以原则集 + AI 反馈（RLAIF）替代人类偏好标注（[[constitutional_ai]]）
- [[rafailov_2023_dpo]] — DPO 将奖励建模与 RL 优化合并为一步，无 RL 的偏好优化（[[direct_preference_optimization]]）
- **范式意义**：RL 从"打游戏"转向"对齐人类价值观"，成为 LLM 后训练核心（见 [[post_training]]）；DPO 开启低成本对齐路线

### ⑤ LLM 推理时代（2024-2025）

- [[shao_2024_deepseekmath]] — DeepSeekMath 提出 GRPO（[[grpo]]）：组内相对奖励替代 critic，无 critic RL 训练
- [[openai_2024_o1]] — o1：以大规模 RL 训练模型内部思维链，推理时计算可扩展（[[test_time_compute]]），开启推理模型时代
- [[deepseek_2025_r1]] — DeepSeek-R1 纯 RL 激发推理能力，涌现自我反思/验证，Nature 封面报道
- [[kimi_2025_k2]] — Kimi K2：多轮 RL（可验证奖励 + 偏好）训练 1T MoE，开源对标 o3
- **范式意义**：RL 从"对齐偏好"进一步走向"激发能力"，可验证奖励（RLVR）成为主流训练信号

### 旁支二：通用智能体与世界模型（2022-2023）

- [[reed_2022_gato]] — Gato：单模型统一文本/图像/机器人动作，604 任务达到专家 50%+（[[generalist_agent]]）
- [[hafner_2023_dreamerv3]] — DreamerV3：潜在世界模型 + 想象训练，固定超参跨 150+ 任务 SOTA（[[world_model]] / [[model_based_rl]]）
- **范式意义**：探索 RL 的"通用性"（单模型多任务）与"效率"（世界模型替代真实交互）两个方向

### 旁支三：模仿学习与自动驾驶（2010-2020）

- [[prakash_2010_dagger]] — DAgger（[[dagger_algorithm]]）交互式聚合专家数据，缓解行为克隆分布偏移（[[imitation_learning]]）
- [[rhinehart_2020_imitative]] — Imitative Models（[[imitative_models]]）融合模仿分布与目标导向规划
- **范式意义**：免奖励信号的学习路线，与基于奖励的 RL 互补

## 关键转折点

| 转折 | 时间 | 意义 |
|:---|:---|:---|
| DQN 超越人类 | 2013 | 深度 RL 可行性证明 |
| PPO 出现 | 2017 | 稳定易用，RL 工业化的前提 |
| RLHF 成型 | 2017-2022 | RL 从控制任务走向人类对齐 |
| DPO 简化对齐 | 2023 | 无 RL 的偏好优化，开源对齐事实标准 |
| o1/GRPO | 2024-2025 | 推理时扩展 + 无 critic RL，能力涌现 |

## 相关概念

- [[reinforcement_learning]] — 强化学习总纲
- [[policy_gradient]] — 策略梯度家族
- [[actor_critic]] — 策略-价值双网络框架
- [[reward_modeling]] / [[reward_overoptimization]] — 奖励的建模与风险
- [[constitutional_ai]] / [[direct_preference_optimization]] — RLHF 的两条替代路线
- [[grpo]] / [[test_time_compute]] — 新一代 LLM RL 算法与推理时扩展
- [[world_model]] / [[model_based_rl]] — 世界模型路线
- [[generalist_agent]] — 通用智能体路线
- [[post_training]] — RLHF 在后训练中的位置
