---
id: state_space_model
type: concept
tags: [machine-learning, theoretical]
aliases: [状态空间模型, SSM, 选择性状态空间]
related_nodes: [gu_2023_mamba, dao_2024_mamba2, retention_mechanism, recurrent_neural_network]
last_verified: 2026-08-03
---

# State Space Model (SSM)

## 定义
以线性时变状态方程建模序列的架构：$h_t = A_t h_{t-1} + B_t x_t$，以线性复杂度（训练可并行、推理 O(1) 状态）处理长序列，Mamba 的选择性机制使其匹敌 Transformer。

## 关键要点
- **选择性**：参数随输入变化，内容感知地决定保留/遗忘（Mamba S6）
- **SSD 对偶性**：Mamba-2 证明 SSM 与注意力是同一矩阵族的两个极端
- **混合架构**：SSM + 注意力混合（Jamba/Zamba）是主流实用路线

## 来源
- [[gu_2023_mamba]] — Mamba：选择性状态空间
- [[dao_2024_mamba2]] — SSD 统一框架
