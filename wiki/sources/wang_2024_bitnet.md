---
id: wang_2024_bitnet
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["BitNet", "b1.58", "1-bit LLM", "2410.16144"]
related_nodes: [model_quantization, transformer_architecture]
authors: Shuming Ma et al.
authors_institution: Microsoft
arxiv_id: 2410.16144
last_verified: 2026-08-03
---

# The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

- **元数据**: arXiv 2410.16144 | 2024 | **作者**: Shuming Ma et al. | **机构**: Microsoft
- **概述**: 提出 BitNet b1.58：权重三值化（{-1, 0, +1}），以 1.58 bit/权重 存储，同等能耗下延迟与吞吐显著优于 FP16 基线，精度损失可控。
- **新颖概念**: 无（归入 [[model_quantization]]）
- **关键要点**: 1. 三值权重 + 缩放因子，乘加变为加减 2. 推理能耗降低 71-91% 3. 从头训练而非后训练量化
- **方法/发现**: 3B 模型在部分任务匹敌 FP16 LLaMA；探索"极低比特原生训练"路线（BitNet 后续与 MoE 结合）
- **局限/意义**: 挑战"精度换效率"的底线；若规模化验证成功，可能重构 LLM 部署硬件需求

## 引用
- **原始论文**: [arXiv:2410.16144](https://arxiv.org/abs/2410.16144)
- **相关概念**: [[model_quantization]]
