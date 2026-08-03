---
id: xiao_2023_streamingllm
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["StreamingLLM", "流式LLM", "2309.17453"]
related_nodes: [kv_cache, attention_mechanism]
authors: Guangxuan Xiao et al.
authors_institution: MIT
arxiv_id: 2309.17453
last_verified: 2026-08-03
---

# Efficient Streaming Language Models with Attention Sinks

- **元数据**: arXiv 2309.17453 | 2023 | **作者**: Guangxuan Xiao et al. | **机构**: MIT
- **概述**: 发现注意力"汇点"现象：初始 token 吸收大量注意力分数；保留初始 token + 滑动窗口 KV 即可实现无限流式生成，无需重训练。
- **新颖概念**: [[kv_cache]]
- **关键要点**: 1. 注意力汇点（attention sink）：首 token 是关键锚点 2. 滑动窗口 + sink 组合的 KV 丢弃策略 3. 流式生成稳定，困惑度与全缓存相当
- **方法/发现**: StreamingLLM 让模型在 400 万 token 以上稳定生成；为 KV cache 压缩（H2O、SnapKV 等）提供理论基础
- **局限/意义**: 揭示自注意力对"固定锚点"的隐性依赖；KV 复用策略成为长上下文推理的标配技巧

## 引用
- **原始论文**: [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- **相关概念**: [[kv_cache]]
