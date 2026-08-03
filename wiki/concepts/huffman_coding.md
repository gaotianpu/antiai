---
id: huffman_coding
type: concept
tags: [information-theory, machine-learning, empirical-study]
aliases: [霍夫曼编码, Huffman编码]
related_nodes: [han_2015_deepcompression, model_quantization]
last_verified: 2026-08-03
---

# Huffman Coding

## 定义
最优前缀编码算法：按符号频率构造二叉树，高频符号用短码，低频用长码，实现无损数据压缩。

## 关键要点
- **最优性**：给定频率分布下平均码长最短（熵编码）
- **Deep Compression 应用**：对量化后的权重进行霍夫曼编码，再省 20-30% 存储
- **硬件友好**：查表解码，适合存储带宽受限场景

## 来源
- [[han_2015_deepcompression]] — 压缩流程第三阶段
