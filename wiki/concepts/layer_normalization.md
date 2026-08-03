---
id: layer_normalization
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [层归一化, LayerNorm, LN]
related_nodes: [layer_norm_2016, normalization, transformer_architecture]
last_verified: 2026-08-03
---

# Layer Normalization

## 定义
对单个样本的所有特征维度做归一化（均值 0 方差 1，再仿射变换），与 batch 无关，适用于 RNN/Transformer 等变长序列场景。

## 关键要点
- **与 BN 的区别**：BN 沿 batch 维归一化，LN 沿特征维归一化
- **优点**：不受 batch size 影响，训练/推理行为一致
- **地位**：Transformer 的标准组件（Pre-LN / Post-LN 两种摆放）

## 来源
- [[layer_norm_2016]] — 提出 Layer Normalization
