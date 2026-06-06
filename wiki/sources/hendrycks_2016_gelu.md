---
id: hendrycks_2016_gelu
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["GELU", "Gaussian Error Linear Unit", "1606.08415"]
related_nodes: []
arxiv_id: 1606.08415
authors: Dan Hendrycks et al.
authors_institution: UC Berkeley
last_verified: 2026-06-06
---

# Gaussian Error Linear Units (GELUs)

- **元数据**: arXiv | 2016 | **作者**: Dan Hendrycks, Kevin Gimpel | **机构**: UC Berkeley
- **概述**: 提出 GELU 激活函数 $x\Phi(x)$（$\Phi$ 为标准高斯 CDF），按输入值而非符号进行加权，在 CV、NLP、语音任务上一致优于 ReLU 和 ELU。
- **新颖概念**: [[gelu_activation]]
- **关键要点**: 1. GELU 是 ReLU 的平滑版本，结合了 dropout 的随机正则化与 zoneout 的确定性门控 2. 在 Transformer 架构中成为默认激活函数（BERT、GPT 等） 3. 相比于 ReLU，GELU 在深层网络中梯度传播更平稳
- **方法/发现**: 在 MNIST、CIFAR-10/100、TIMIT 等数据集上对 GELU/ReLU/ELU 进行了系统性对比实验。
- **局限/意义**: GELU 计算开销略高于 ReLU（涉及高斯误差函数），但现代框架通过近似实现已将其开销降至可忽略。

## 引用
- **原始论文**: [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | [阅读笔记](../../raw/cnn/GELUs.md)
