---
id: channel_attention
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [通道注意力, 通道注意力机制]
related_nodes: [hu_2017_senet, attention_mechanism]
last_verified: 2026-08-03
---

# Channel Attention

## 定义
显式建模通道间依赖关系的机制：通过全局池化 + 两层全连接（Squeeze-and-Excitation）学习每个通道的重要性权重，对特征图做通道级重校准。

## 关键要点
- **SE 块**：Squeeze（全局平均池化）→ Excitation（门控）→ Scale（重校准）
- **成本极低**：参数量约为全网络的 1-2%，收益显著
- **演进**：ECA（一维卷积替代全连接）、CBAM（通道+空间）等变体

## 来源
- [[hu_2017_senet]] — SENet：ILSVRC 2017 冠军
