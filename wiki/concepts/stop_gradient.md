---
id: stop_gradient
type: concept
tags: [machine-learning, empirical-study]
aliases: [停止梯度, 梯度阻断, stop-grad]
related_nodes: [simsiam_2020, siamese_network, contrastive_learning]
last_verified: 2026-08-03
---

# Stop Gradient

## 定义
阻断梯度反向传播的操作（sg()）：SimSiam 等孪生方法中一侧分支不接收梯度，防止表示坍塌，是"不对称设计"的关键机制。

## 关键要点
- **防坍塌**：没有 stop-grad 时网络退化为常数输出（坍塌解）
- **不对称性**：预测头 + stop-grad 构成隐式 EM 优化
- **普适性**：BYOL、DINO 等均依赖此机制

## 来源
- [[simsiam_2020]] — SimSiam 证明 stop-grad 的充分性
