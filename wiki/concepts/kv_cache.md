---
id: kv_cache
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [键值缓存, KV缓存]
related_nodes: [kwon_2023_pagedattention, xiao_2023_streamingllm, grouped_query_attention, multi_head_latent_attention]
last_verified: 2026-08-03
---

# KV Cache

## 定义
自回归解码时为避免重复计算而缓存的键值张量：每生成一个 token 只需计算新 K/V 并追加到缓存，推理速度 O(1)/token，但内存随上下文线性增长。

## 关键要点
- **瓶颈**：长上下文/高并发下 KV cache 是显存第一消耗者
- **缩减路线**：GQA 分组共享、MLA 潜在压缩、KV 淘汰（StreamingLLM）
- **系统优化**：PagedAttention 分页管理消除碎片

## 来源
- [[kwon_2023_pagedattention]] — 分页 KV 管理
- [[xiao_2023_streamingllm]] — 注意力汇点与 KV 淘汰
- [[gemini_2024_15]] — 百万上下文工程
