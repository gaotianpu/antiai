---
id: attention_variants
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [注意力变体, multi-head attention, cross-attention, self-attention]
related_nodes: [attention_mechanism]
---

# Attention Variants

## 概述

Attention Variants（注意力变体）是对 [[attention_mechanism]] 基本公式在不同维度上的扩展和改造。这些变体在注意力作用域、结构组织、计算效率和掩码模式上各有不同，适应不同任务和架构需求。

## 详细阐述

### 定义与内涵

基础注意力公式 $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$ 可通过三个维度分化：**作用对象**（Self vs Cross）、**组织方式**（单头 vs 多头）、**访问模式**（双向 vs 因果 vs 稀疏）。

### 按作用对象分类

#### Self-Attention（自注意力）

$Q=K=V$ 来自同一序列。捕捉序列内部依赖，是 Transformer 编码器的基本构件 [[vaswani_2017_transformer]]。

#### Cross-Attention（交叉注意力）

$Q$ 来自一个序列，$K,V$ 来自另一个序列。用于序列间交互，如 Transformer 解码器的编码器-解码器注意力、多模态融合。

### 按组织结构分类

#### 单头注意力（Single-Head Attention）

单组 QKV 投影。早期机器翻译应用 [[bahdanau_2014_attention]]。

#### 多头注意力（Multi-Head Attention, MHA）

将输入投影到 $h$ 个低维子空间，分别计算注意力后拼接。多头使模型同时关注不同位置的表示子空间 [[vaswani_2017_transformer]]。
- **MHA**：标准多头，每个头各自投影 QKV
- **GQA（Grouped Query Attention）**：分组共享 Key/Value 头，减少 KV 缓存
- **MQA（Multi-Query Attention）**：所有 Query 头共享一组 K/V，进一步减少缓存

### 按访问模式分类

| 模式 | 遮罩 | 典型用例 |
|:---|:---|:---|
| 双向（Full/Bidirectional） | 无遮罩 | 编码器（BERT、ViT） |
| 因果（Causal/Masked） | 禁止关注未来 token | 自回归解码器（GPT） |
| 前缀（Prefix LM） | 前缀双向 + 后缀因果 | T5、GLM |
| 滑动窗口（Sliding Window） | 只关注局部窗口 | Longformer、Mistral |
| 稀疏（Sparse） | 固定稀疏模式 | BigBird、Longformer |

### 高效注意力变体

标准注意力的 $O(n^2)$ 复杂度制约长序列。以下变体旨在降低复杂度：

| 方法 | 核心思路 | 复杂度 | 出处 |
|:---|:---|:---|:---|
| FlashAttention | IO-aware tiling + 重计算，不改变结果 | $O(n^2)$ 但常数极小 | [[dao_2022_flashattention]] |
| Linear Attention | $QK^T$ 替换为核函数 $\\phi(Q)\\phi(K)^T$，利用矩阵结合律 | $O(n)$ | — |
| Sparse Attention | 只计算部分位置对的注意力 | $O(n\\sqrt{n})$ ~ $O(n)$ | — |
| Reformer (LSH) | 局部敏感哈希近似最近邻注意力 | $O(n\\log n)$ | — |
| Rotary Embedding | 通过 RoPE 实现相对位置编码，不降低复杂度但提升外推 | 不变 | [[roformer_2021]] |

### 应用场景

- **Self-Attention**：Transformer 编码器中捕捉双向上下文
- **Cross-Attention**：图像到文本生成、视频-语言对齐
- **Causal Attention**：所有自回归语言模型（GPT、LLaMA）
- **FlashAttention**：长序列训练和推理（64k+ tokens）

## 相关概念网络

- [[attention_mechanism]] — 基础注意力概念页
- [[positional_encoding]] — 注意力需要位置信号
- [[chain_of_thought]] — 注意力支持长程推理

## 引用资料

1. [[vaswani_2017_transformer]] — Transformer 中的 Scaled Dot-Product + Multi-Head
2. [[dao_2022_flashattention]] — FlashAttention IO 优化
3. [[roformer_2021]] — RoPE 旋转位置嵌入

## 更新记录

- **2026-06-06**: 初始创建
