---
id: momentum_contrast
type: concept
tags: [machine-learning, empirical-study]
aliases: [动量对比, MoCo]
related_nodes: [he_2019_moco, contrastive_learning]
last_verified: 2026-08-03
---

# Momentum Contrast (MoCo)

## 定义
将对比学习视为字典查找：用队列存储历史样本编码（大而一致的字典）+ 动量编码器平滑更新，摆脱对大批量的依赖。

## 关键要点
- **字典大小**：队列可容纳数千样本，不受 batch size 限制
- **动量更新**：$θ_k ← mθ_k + (1-m)θ_q$，保证字典特征一致性
- **成果**：无监督表示接近有监督水平，成为对比学习经典框架

## 来源
- [[he_2019_moco]] — MoCo
