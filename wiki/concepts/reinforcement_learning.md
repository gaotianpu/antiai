---
id: reinforcement_learning
type: concept
tags: [machine-learning, theoretical, empirical-study]
aliases: [RL, 强化学习]
related_nodes: [mnih_2013_dqn, lillicrap_2015_ddpg, schulman_2017_ppo, mnih_2016_a2c, policy_gradient, deepseek_2025_r1, grpo]
last_verified: 2026-06-06
---

# Reinforcement Learning

## 定义
Reinforcement Learning（强化学习）是机器学习的三大范式之一。智能体（agent）通过与环境的交互获得奖励信号，学习最大化累积奖励的策略。其核心框架为马尔可夫决策过程（MDP）。

## 核心要素
- **状态（State）$s$** — 环境的表示
- **动作（Action）$a$** — 智能体的决策
- **奖励（Reward）$r$** — 反馈信号
- **策略（Policy）$\pi$** — 状态到动作的映射
- **值函数（Value Function）$V(s)$ / $Q(s,a)$** — 期望累积奖励

## 主要分支
| 分支 | 特点 | 代表算法 |
|:---|:---|:---|
| 基于价值 | 学习 $Q$ 函数，隐式策略 | DQN（[[mnih_2013_dqn]]） |
| 基于策略 | 直接优化策略参数 | PPO（[[schulman_2017_ppo]]） |
| Actor-Critic | 策略 + 价值联合学习 | A2C（[[mnih_2016_a2c]]）, DDPG（[[lillicrap_2015_ddpg]]） |

## 来源
- [[mnih_2013_dqn]] — 深度 Q 网络突破 Atari 游戏
- [[lillicrap_2015_ddpg]] — 确定性策略梯度用于连续控制
- [[schulman_2017_ppo]] — 稳定策略优化的实用算法
