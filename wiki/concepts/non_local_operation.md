---
id: non_local_operation
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [非局部操作, Non-local]
related_nodes: [wang_2017_nonlocal, attention_mechanism, self_attention]
last_verified: 2026-08-03
---

# Non-local Operation

## 定义
通用构建块：以所有位置加权聚合的方式捕获长距离依赖，形式为 $y_i = \frac{1}{C(x)}\sum_j f(x_i,x_j)g(x_j)$，是自注意力的广义形式。

## 关键要点
- **通用性**：适用于视频、图像、序列任意模态
- **与自注意力的关系**：非局部块 ≈ 自注意力（嵌入高斯版本）
- **应用**：视频分类（Non-local Net）、检测/分割中增强上下文

## 来源
- [[wang_2017_nonlocal]] — Non-local Neural Networks
