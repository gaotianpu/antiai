---
id: flash_attention
type: concept
tags: [NLP, machine-learning, empirical-study]
aliases: [FlashAttention, 闪速注意力]
related_nodes: [dao_2022_flashattention, self_attention, io_aware_attention]
last_verified: 2026-08-03
---

# FlashAttention

## 定义
IO 感知的精确注意力实现：通过分块计算（tiling）与在线 softmax 重计算，避免将完整注意力矩阵写入 HBM，显著提速并节省显存。

## 关键要点
- **核心思想**：GPU 上 HBM↔SRAM 带宽是瓶颈，减少内存读写比减少 FLOPs 更重要
- **成果**：2-4 倍加速、显存从 O(n²) 降至 O(n)
- **影响**：成为 LLM 训练/推理的标准算子，催生 FlashAttention-2/3

## 来源
- [[dao_2022_flashattention]] — 提出 FlashAttention
