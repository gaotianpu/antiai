---
id: deepnorm
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [DeepNorm, 深度归一化]
related_nodes: [deepnet, transformer_architecture, residual_connection]
last_verified: 2026-08-03
---

# DeepNorm

## 定义
DeepNet 提出的残差缩放方案：在残差连接中引入缩放因子，稳定深层 Transformer 训练，可将模型扩展到 1000 层。

## 关键要点
- **缩放残差**：$x_{l+1} = LayerNorm(\alpha x_l + f(x_l))$，α 随层数调整
- **解决退化**：抑制深层残差中信号放大导致的训练不稳定
- **意义**：证明 Transformer 可像 CNN（ResNet）一样堆叠极深

## 来源
- [[deepnet]] — DeepNet：1000 层 Transformer
