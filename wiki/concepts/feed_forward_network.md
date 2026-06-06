---
id: feed_forward_network
type: concept
tags: [machine-learning, theoretical]
aliases: [前馈网络, FFN, MLP层, position-wise FFN]
related_nodes: [transformer_architecture, activation_function]
---

# Feed-Forward Network (FFN)

## 概述

Feed-Forward Network（前馈网络），在 Transformer 中特指 **position-wise FFN** — 对序列中每个位置独立应用相同的一组全连接层。它是 Transformer 每个子层中紧跟在 [[attention_mechanism]] 之后的关键组件。

## 数学形式

### 标准 FFN（ReLU 版本）

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, $b_1 \in \mathbb{R}^{d_{\text{ff}}}$
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $b_2 \in \mathbb{R}^{d_{\text{model}}}$
- 隐藏层维度 $d_{\text{ff}}$ 通常为 $4 \times d_{\text{model}}$（如 Transformer base: $d_{\text{model}}=512, d_{\text{ff}}=2048$）

### SwiGLU 变体

LLaMA、PaLM 等现代 LLM 采用 SwiGLU 替代 ReLU：

$$\text{FFN}_{\text{SwiGLU}}(x) = (W_3 \cdot x) \odot \text{Swish}(W_1 \cdot x) \cdot W_2$$

其中 $\odot$ 为逐元素乘，Swish 激活函数 $\text{Swish}(x) = x \cdot \sigma(x)$。SwiGLU 比 ReLU 版本通常需要多一个权重矩阵 $W_3$，但效果更优。

## 关键特性

| 特性 | 说明 |
|:---|:---|
| Position-wise | 同一层内所有位置共享参数（类似 1×1 卷积） |
| 维度扩展 | 内部维度扩张 4×，提供模型容量 |
| 逐点非线性 | 多头注意力是线性变换的加权和，FFN 引入必要非线性 |

## FFN 的角色：“记忆存储”

有多篇研究（Geva et al. 2021, Meng et al. 2022）认为 FFN 在 Transformer 中扮演了 **键值记忆（Key-Value Memory）** 的角色：

- 第一层将输入映射到高维（$d_{\text{ff}}$），相当于查询记忆中的模式
- [[activation_function#GELU\|GELU]]/ReLU 门控激活相当于只激活与输入匹配的"记忆槽"
- 第二层将激活的模式投影回模型维度

这一视角解释了为什么 FFN 在知识存储和事实召回中起关键作用，也是模型编辑（Model Editing）和知识定位工作的理论基础。

## 模型维度对比

| 模型 | $d_{\text{model}}$ | $d_{\text{ff}}$ | 扩展比 | 激活函数 |
|:---|:---:|:---:|:---:|:---:|
| Transformer base | 512 | 2048 | 4× | ReLU |
| BERT Base | 768 | 3072 | 4× | GELU |
| GPT-3 | 12288 | 49152 | 4× | GELU |
| LLaMA-7B | 4096 | 11008 | 2.7× | SwiGLU |
| PaLM-540B | 18432 | 49152 | 2.7× | SwiGLU |

## 相关概念网络

- [[transformer_architecture]] — FFN 作为 Transformer 子层的上下文
- [[activation_function]] — FFN 中使用的激活函数

## 引用资料

1. [[vaswani_2017_transformer]] — 原始 Transformer 中的 FFN
2. [[hendrycks_2016_gelu]] — GELU 激活函数
