---
id: leviathan_2023_speculative
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["Speculative Decoding", "投机解码", "2211.17192"]
related_nodes: [speculative_decoding, transformer_architecture]
authors: Yaniv Leviathan et al.
authors_institution: Google
arxiv_id: 2211.17192
last_verified: 2026-08-03
---

# Fast Inference from Transformers via Speculative Decoding

- **元数据**: arXiv 2211.17192 | 2022-11 | **作者**: Yaniv Leviathan et al. | **机构**: Google
- **概述**: 提出投机解码：小模型（草稿模型）快速生成候选 token 序列，大模型并行验证并接受合理前缀，解码加速 2-3 倍且分布无偏。
- **新颖概念**: [[speculative_decoding]]
- **关键要点**: 1. 草稿-验证两模型配合 2. 拒绝采样保证输出分布与目标模型一致 3. 无需修改模型权重/训练
- **方法/发现**: 与 Medusa（多头草稿）、EAGLE 等构成"投机解码"家族；成为 LLM 推理加速的标准技术
- **局限/意义**: 依赖草稿模型质量与接受率；代表"用并行换序列依赖"的推理加速新范式

## 引用
- **原始论文**: [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- **相关概念**: [[speculative_decoding]]
