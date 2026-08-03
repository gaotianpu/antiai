---
id: gelu_activation
type: concept
tags: [machine-learning, empirical-study]
aliases: [GELU激活, 高斯误差线性单元]
related_nodes: [hendrycks_2016_gelu, activation_function]
last_verified: 2026-08-03
---

# GELU Activation

## 定义
Gaussian Error Linear Unit：$GELU(x) = x\Phi(x)$（Φ 为标准高斯 CDF），按输入值而非符号加权输入，融合 ReLU 与 Dropout 的思路。

## 关键要点
- **概率加权**：以输入值的累积概率为权重，平滑过渡
- **性能**：在 CV、NLP、语音任务上一致优于 ReLU 与 ELU
- **地位**：BERT/GPT 等 Transformer 模型的标准激活

## 来源
- [[hendrycks_2016_gelu]] — GELU 原始论文
