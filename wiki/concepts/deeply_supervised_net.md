---
id: deeply_supervised_net
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [深度监督, 深度监督网络, DSN]
related_nodes: [lee_2014_dsn, gradient_descent]
last_verified: 2026-08-03
---

# Deeply Supervised Net (DSN)

## 定义
在深层网络的隐藏层引入辅助监督信号（companion objective）的训练方式，使中间层也能直接接收梯度，缓解梯度消失并提升特征质量。

## 关键要点
- **辅助分类器**：中间层分支接分类损失，与主损失联合训练
- **双重收益**：训练更稳定 + 中间特征判别性更强
- **应用**：GoogLeNet 辅助头、深度监督分割网络等沿用

## 来源
- [[lee_2014_dsn]] — 深度监督网络
