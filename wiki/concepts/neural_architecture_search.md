---
id: neural_architecture_search
type: concept
tags: [machine-learning, empirical-study]
aliases: [神经架构搜索, NAS]
related_nodes: [lin_2020_mcunet, neural_architecture_search_applied]
last_verified: 2026-08-03
---

# Neural Architecture Search (NAS)

## 定义
用算法自动搜索网络架构（层数、算子、连接）替代人工设计的范式，搜索空间 + 搜索策略 + 性能评估三要素构成。

## 关键要点
- **搜索空间**：从算子序列到单元（cell）级再到层次化空间
- **搜索策略**：强化学习、进化算法、梯度（DARTS）、随机（TinyNAS）
- **代价**：传统 NAS 计算昂贵，催生权重共享、零成本代理等加速手段

## 来源
- [[lin_2020_mcunet]] — TinyNAS：面向 MCU 的架构搜索
