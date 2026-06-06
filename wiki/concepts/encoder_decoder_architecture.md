---
id: encoder_decoder_architecture
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [Encoder-Decoder, 编码器-解码器, seq2seq]
related_nodes: [transformer_architecture, attention_variants]
---

# Encoder-Decoder Architecture

## 概述

Encoder-Decoder（编解码器架构）是一种将输入序列编码为隐表示、再解码为目标序列的宏观架构模式。在 Transformer 时代，这一架构通过自注意力机制被彻底重塑。

## Encoder：双向上下文编码

Encoder 由 N 层堆叠，每层包含 **Bidirectional Self-Attention**（双向自注意力）+ [[feed_forward_network]]。

- **双向注意力**：每个 token 可以关注输入序列中的所有 token（包括左和右），捕获全局上下文
- **输出**：一组上下文感知的向量表示，维度为 $d_{\text{model}}$，长度与输入序列一致
- **代表模型**：BERT [[devlin_2018_bert]] 使用纯 Encoder，在 NLU 任务上表现优异

## Decoder：自回归生成

Decoder 也是 N 层堆叠，但自注意力层是 **Causal (Masked) Attention**（因果注意力），且额外增加一个 **Cross-Attention** 子层。

### 因果自注意力

每个 token 只能关注自身及左侧的 token，通过上三角掩码实现。保证自回归生成时不会泄露未来信息。

### 交叉注意力（Cross-Attention）

- **Query**：来自 Decoder 当前层表示
- **Key/Value**：来自 Encoder 最后一层输出
- **作用**：让 Decoder 动态聚焦于输入序列的不同部分，实现条件生成

## 原始 Transformer 的 Encoder-Decoder

[[transformer_architecture]] 的完整版：

| 组件 | Encoder (6x) | Decoder (6x) |
|:---|:---|:---|
| 自注意力 | 双向 | 因果（Masked） |
| 交叉注意力 | 无 | 有（K/V 来自 Encoder） |
| FFN | 有 | 有 |
| 残差连接 + LN | 有 | 有 |
| 典型任务 | 理解/分类 | 生成 |

## 三大衍生变体

### Encoder-only（BERT 路线）

- **结构**：仅保留 Encoder 堆叠
- **注意力**：双向
- **任务**：文本分类、NER、QA、句子对匹配
- **代表**：BERT [[devlin_2018_bert]]、RoBERTa、ALBERT

### Decoder-only（GPT 路线）

- **结构**：仅保留 Decoder（去掉交叉注意力）
- **注意力**：因果
- **任务**：文本生成、对话、代码生成
- **代表**：GPT 系列 [[radford_2018_gpt]]、LLaMA、PaLM

### Encoder-Decoder（T5 路线）

- **结构**：完整 Encoder + Decoder
- **注意力**：双向编码 + 因果（带交叉注意力）解码
- **任务**：翻译、摘要、多模态生成
- **代表**：T5 [[raffel_2019_t5]]、BART、Flan-T5

### 选型指南

| 场景 | 推荐 | 原因 |
|:---|:---|:---|
| 文本理解、分类 | Encoder-only | 双向上下文最充分 |
| 自由文本生成 | Decoder-only | 自回归生成最自然，效率高 |
| 条件生成（翻译/摘要） | Encoder-Decoder | 输入理解 + 输出生成解耦 |
| 多模态输入 | Encoder-Decoder | Cross-Attention 融合效果好 |

## 相关概念网络

- [[transformer_architecture]] — 完整的 Encoder-Decoder 实现
- [[attention_variants]] — 双向/因果/交叉注意力详解
- [[attention_mechanism]] — 注意力机制基础

## 引用资料

1. [[vaswani_2017_transformer]] — 原始 Encoder-Decoder Transformer
2. [[devlin_2018_bert]] — Encoder-only 代表 BERT
3. [[radford_2018_gpt]] — Decoder-only 代表 GPT
