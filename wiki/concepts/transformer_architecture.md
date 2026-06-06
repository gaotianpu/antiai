---
id: transformer_architecture
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [Transformer, Transformers, Transformer 架构]
related_nodes: [attention_mechanism, attention_variants, positional_encoding, residual_connection, normalization, feed_forward_network, encoder_decoder_architecture]
---

# Transformer Architecture

## 概述

Transformer 是一种完全基于 [[attention_mechanism]] 的序列到序列架构，摒弃了 RNN 的循环结构和 CNN 的卷积结构，仅依赖自注意力（Self-Attention）和前馈网络（[[feed_forward_network]]）构建。

## 核心结构

### Encoder-Decoder 堆叠

Transformer 采用 [[encoder_decoder_architecture]]：

- **Encoder** (N=6层): 每层包含 Multi-Head Self-Attention + FFN，使用 [[residual_connection]] 和 [[normalization#Layer Normalization\|LayerNorm]] 包裹每个子层
- **Decoder** (N=6层): 每层包含 Masked Multi-Head Self-Attention + Cross-Attention（以 Encoder 输出为 K/V）+ FFN，同样使用残差连接和 LayerNorm

### 每个子层的公式

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

即 Pre-LN（或 Post-LN）残差结构。所有子层（注意力、FFN）的输出维度均为 $d_{\text{model}}=512$。

## 关键创新

| 创新 | 描述 | 意义 |
|:---|:---|:---|
| 自注意力 | 序列内任意位置直接交互 | 捕获长程依赖，不受距离限制 |
| 多头注意力 | 并行计算 h=8 个注意力头 | 捕捉不同子空间的表示模式 |
| 位置编码 | Sinusoidal [[positional_encoding]] | 为置换不变的自注意力注入顺序信息 |
| 并行化 | 序列内所有位置同时计算 | 训练速度远超 RNN |
| 残差连接 + LN | 每个子层后包裹 | 稳定深度网络训练 |

## 优势与局限

### 优势

- **并行计算**: 摆脱 RNN 的时序依赖，GPU 利用率高
- **长程依赖**: 任意两位置直接连接，路径长度为 1
- **可扩展性**: 增加层数和宽度可稳定提升性能（Scaling Law）

### 局限

- **$O(n^2)$ 复杂度**: 自注意力的计算量随序列长度平方增长，制约长序列处理
- **缺乏局部偏置**: 对比 CNN，Transformer 需要更多数据和更大模型才能学习局部模式
- **位置编码需求**: 必须外挂位置信号，增加了设计复杂度

## 影响与变体

Transformer 已成为现代深度学习的基石架构，衍生出三大主流方向：

| 方向 | 代表模型 | 核心改动 |
|:---|:---|:---|
| Encoder-only | BERT [[devlin_2018_bert]] | 仅保留 Encoder，双向自注意力 |
| Decoder-only | GPT [[radford_2018_gpt]] | 仅保留 Decoder，因果注意力 |
| Encoder-Decoder | T5 [[raffel_2019_t5]] | 完整 Encoder-Decoder，前缀注意力 |

## 相关概念网络

- [[attention_mechanism]] — Transformer 的核心计算原语
- [[attention_variants]] — 多头/因果/高效注意力变体
- [[positional_encoding]] — 为注意力提供顺序感知
- [[residual_connection]] — 残差连接使深层训练成为可能
- [[normalization]] — LayerNorm 稳定训练过程
- [[feed_forward_network]] — 每层中的 FFN 子层
- [[encoder_decoder_architecture]] — Transformer 的宏观结构模式

## 引用资料

1. [[vaswani_2017_transformer]] — 原始 Transformer 论文
