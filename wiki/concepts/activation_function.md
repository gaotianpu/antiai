---
id: activation_function
type: concept
tags: ["machine-learning", "theoretical"]
aliases: ["激活函数", "activation", "非线性激活"]
related_nodes: ["convolutional_neural_network", "hendrycks_2016_gelu", "krizhevsky_2012_alexnet"]
last_verified: 2026-06-06
---

# Activation Function（激活函数）

## 定义
激活函数是神经网络中的非线性变换层，使网络能学习复杂模式。没有激活函数，多层网络等价于单层线性变换。

## 常见激活函数

| 函数 | 公式 | 特点 | 出现时间 |
|:---|:---|:---|:---|
| 函数 | 公式 | 特点 | 出处 |
|:---|:---|:---|:---|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | 输出 (0,1)，易饱和梯度消失 | Rumelhart et al., 1986 |
| **Tanh** | tanh(x) | 输出 (-1,1)，零中心，仍易饱和 | LeCun et al., 1991 |
| **ReLU** | max(0, x) | 计算简单，缓解梯度消失，但 Dead ReLU | [[krizhevsky_2012_alexnet\|AlexNet 2012]] |
| **Leaky ReLU** | max(αx, x) | 解决 Dead ReLU | Maas et al., 2013 |
| **ELU** | x≥0: x, x<0: α(eˣ-1) | 平滑负值部分 | Clevert et al., 2015 |
| **GELU** | x·Φ(x) | 高斯误差线性单元，BERT/Transformer 标配 | [[hendrycks_2016_gelu]] |
| **Swish / SiLU** | x·σ(x) | 无上界有下界，平滑 | Ramachandran et al., 2017 |
| **Mish** | x·tanh(ln(1+eˣ)) | 比 Swish 更平滑 | Misra, 2019 |

## 演进趋势
- **Sigmoid/Tanh** → 梯度消失严重，主要用于输出门
- **ReLU** → 计算高效，但 Dead ReLU 问题
- **GELU/Swish** → 平滑激活，兼顾 ReLU 优势和梯度流动
- 现代 LLM 普遍使用 **GELU**（BERT、GPT）或 **Swish**（EfficientNet）

## 关键发现
- ReLU 使深层网络可训练（AlexNet 2012）
- GELU 在 Transformer 上显著优于 ReLU（[[hendrycks_2016_gelu]]）

## 相关页面
- [[convolutional_neural_network]] — CNN 中的激活函数
- [[hendrycks_2016_gelu]] — GELU 原始论文
