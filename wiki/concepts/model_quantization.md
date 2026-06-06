---
id: model_quantization
type: concept
tags: [machine-learning, empirical-study, parameter-optimization]
aliases: [量化, quantization]
related_nodes: [han_2015_deepcompression, dettmers_2023_qlora]
last_verified: 2026-06-06
---

# Model Quantization

## 定义
Model Quantization（模型量化）是将神经网络参数（权重、激活值）从高精度浮点（如 FP32）映射到低位宽表示（如 INT8、INT4、NF4）的技术，旨在减少模型存储和加速推理。

## 量化方式
- **训练后量化（PTQ）** — 在训练完成后直接量化，无需微调
- **量化感知训练（QAT）** — 在训练中模拟量化效果，精度更高
- **混合精度量化** — 不同层使用不同位宽

## 关键进展
- [[han_2015_deepcompression]] — 量化+剪枝联合压缩，6 倍压缩无精度损失
- [[dettmers_2023_qlora]] — NF4 精度量化，使大模型单 GPU 微调成为可能

## 来源
- [[han_2015_deepcompression]] — 深度压缩框架中的量化组件
- [[dettmers_2023_qlora]] — QLoRA：4-bit NormalFloat 量化
