---
id: imitation_learning
type: concept
tags: [machine-learning, empirical-study, RL]
aliases: [模仿学习, 行为克隆]
related_nodes: [prakash_2010_dagger, dagger_algorithm, end_to_end_driving]
last_verified: 2026-08-03
---

# Imitation Learning

## 定义
从专家示范中学习策略的范式：行为克隆（监督学习式）直接拟合专家动作，无需奖励信号，是自动驾驶等领域的常用方法。

## 关键要点
- **行为克隆**：$(s,a^*)$ 监督回归，简单但存在分布偏移（covariate shift）
- **DAgger 改进**：交互式收集专家纠正数据
- **局限**：无法超越专家、对 OOD 状态脆弱

## 来源
- [[prakash_2010_dagger]] — DAgger 分析
