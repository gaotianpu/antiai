---
id: positional_encoding
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [位置编码, position embedding, positional embedding]
related_nodes: [attention_mechanism, transformer_architecture]
---

# Positional Encoding

## 概述

Positional Encoding（位置编码）是为序列模型注入 token 位置或相对位置信息的技术。由于 [[attention_mechanism]] 本身是置换不变的（permutation-invariant），必须通过额外输入区分 token 顺序。

## 详细阐述

### 定义与内涵

位置编码向模型提供每个 token 在序列中的位置信号，使自注意力能够感知顺序关系。按注入方式分为 **绝对位置编码**（每个位置对应固定向量）和 **相对位置编码**（建模 token 间距离/偏移）。

### 主要方案

| 方案 | 类型 | 核心思路 | 出处 |
|:---|:---|:---|:---|
| Sinusoidal | 绝对 | 正弦/余弦固定频率编码 | [[vaswani_2017_transformer]] |
| RoPE | 绝对-相对混合 | 旋转矩阵实现位置敏感点积 | [[roformer_2021]] |
| ALiBi | 相对 | 注意力 logits 加上线性偏置 | [[alibi_2021]] |
| T5 Bias | 相对 | 可学习的相对位置偏置表 | [[raffel_2019_t5]] |
| Transformer-XL | 相对 | 片段级循环 + 相对位置编码 | [[dai_2019_transformer_xl]] |
| xPos | 相对 | RoPE 改进，更好外推 | [[xpos_2022]] |
| No Positional Encoding | — | 某些架构（如 RetNet）通过设计消除位置编码 | [[retnet_2023]] |

### 关键维度

- **外推能力（Extrapolation）**：模型能否泛化到比训练时更长的序列。RoPE、ALiBi 外推性优于 Sinusoidal
- **计算效率**：ALiBi 几乎零开销；Transformer-XL 相对偏置需额外存储
- **与 Attention 交互**：相对位置编码常需修改注意力计算逻辑

### 应用场景

- Transformers 的标准组件
- 长序列建模中的外推问题
- 跨片段上下文融合

## 相关概念网络

- [[attention_mechanism]] — 上游概念，位置编码为注意力提供顺序感知
- [[normalization]] — 常与位置编码配合使用
- [[feature_pyramid]] — 多尺度表征中的位置信息

## 引用资料

1. [[vaswani_2017_transformer]] — 提出 Sinusoidal 位置编码
2. [[roformer_2021]] — 提出旋转位置编码 RoPE
3. [[alibi_2021]] — 提出 ALiBi 线性偏置
4. [[dai_2019_transformer_xl]] — 提出相对位置编码
5. [[xpos_2022]] — RoPE 外推改进
6. [[raffel_2019_t5]] — T5 相对位置偏置

## 更新记录

- **2026-06-06**: 初始创建
