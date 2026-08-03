---
id: deepseek_2025_v32
type: source
tags: ["NLP", "machine-learning", "empirical-study"]
aliases: ["DeepSeek-V3.2", "DSA", "稀疏注意力"]
related_nodes: [sparse_attention, multi_head_latent_attention]
authors: DeepSeek-AI
authors_institution: DeepSeek
last_verified: 2026-08-03
---

# DeepSeek-V3.2: Pushing the Frontier of Sparse Attention and Inference Efficiency

- **元数据**: DeepSeek-AI 技术报告 | 2025 | **作者**: DeepSeek-AI | **机构**: DeepSeek
- **概述**: 提出 DeepSeek Sparse Attention（DSA）：轻量 Lightning Indexer 预选关键 token（Top-k + 局部窗口 + 全局 token），在不重训练前提下实现高效稀疏注意力，性能比肩全注意力。
- **新颖概念**: [[sparse_attention]]
- **关键要点**: 1. Lightning Indexer：低开销索引器预测 token 重要性 2. 稀疏选择：局部窗口 + 全局锚点 + Top-k 跳跃选择 3. 同时支持 MLA/GQA 两种 KV 压缩
- **方法/发现**: 长上下文任务与全注意力持平，推理效率大幅提升；提供从 Dense 到 Sparse 的平滑切换
- **局限/意义**: 与 MoBA 共同确立"可学习索引 + 稀疏选择"的 2025 稀疏注意力范式；工业级长上下文推理的重要一步

## 引用
- **原始论文**: [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2)
- **相关概念**: [[sparse_attention]], [[multi_head_latent_attention]]
