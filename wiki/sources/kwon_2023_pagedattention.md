---
id: kwon_2023_pagedattention
type: source
tags: ["machine-learning", "empirical-study"]
aliases: ["PagedAttention", "vLLM", "2309.06180"]
related_nodes: [kv_cache, attention_mechanism]
authors: Woosuk Kwon et al.
authors_institution: UC Berkeley
arxiv_id: 2309.06180
last_verified: 2026-08-03
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention

- **元数据**: arXiv 2309.06180 | 2023 | **作者**: Woosuk Kwon et al. | **机构**: UC Berkeley
- **概述**: 提出 PagedAttention：借鉴操作系统的虚拟内存分页，将 KV cache 分块存储于非连续物理页，消除显存碎片，支撑高吞吐推理服务（vLLM）。
- **新颖概念**: [[kv_cache]]
- **关键要点**: 1. KV cache 按块管理，物理页非连续、按需分配 2. 同请求共享 KV 块（并行采样）3. 显存利用率接近 100%，吞吐提升 2-4 倍
- **方法/发现**: 与传统连续缓存相比，vLLM 在相同显存下服务更多请求；成为 LLM 推理服务（vLLM/SGLang 等）的基础
- **局限/意义**: 系统层优化而非算法改进；凸显 KV cache 是长上下文/高并发推理的第一瓶颈

## 引用
- **原始论文**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **相关概念**: [[kv_cache]]
