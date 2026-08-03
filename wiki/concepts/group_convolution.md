---
id: group_convolution
type: concept
tags: [computer-vision, machine-learning, theoretical]
aliases: [分组卷积, Group Conv]
related_nodes: [xie_2016_resnext, zhang_2017_shufflenet, convolutional_neural_network]
last_verified: 2026-08-03
---

# Group Convolution

## 定义
将输入通道分为若干组，每组独立卷积后拼接输出，参数与计算量按组数等比例下降，同时引入组间稀疏结构。

## 关键要点
- **正则化效应**：组间无信息交换，可视为结构化稀疏约束
- **基数维度**：ResNeXt 证明增大组数（基数）比加深/加宽更高效
- **注意**：组间无通信导致信息瓶颈，需配合通道重排

## 来源
- [[xie_2016_resnext]] — ResNeXt 基数维度
- [[zhang_2017_shufflenet]] — ShuffleNet 组卷积 + 通道重排
