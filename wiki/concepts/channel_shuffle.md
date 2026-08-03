---
id: channel_shuffle
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [通道重排, 通道洗牌]
related_nodes: [zhang_2017_shufflenet, group_convolution]
last_verified: 2026-08-03
---

# Channel Shuffle

## 定义
对分组卷积的输出通道做均匀重排后再输入下一层分组卷积，使组间信息可以流动，消除分组卷积的信息隔离。

## 关键要点
- **操作**：reshape → transpose → flatten，无额外参数与计算
- **解决瓶颈**：纯分组卷积堆叠会切断组间信息通路
- **应用**：ShuffleNet 系列的核心设计

## 来源
- [[zhang_2017_shufflenet]] — ShuffleNet 提出通道重排
