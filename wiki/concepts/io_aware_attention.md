---
id: io_aware_attention
type: concept
tags: [NLP, machine-learning, theoretical]
aliases: [IO感知注意力, 内存感知算法]
related_nodes: [dao_2022_flashattention, flash_attention]
last_verified: 2026-08-03
---

# IO-Aware Attention

## 定义
以内存访问模式（而非仅 FLOPs）为导向的注意力算法设计原则：通过分块与融合，最小化 HBM 读写次数。

## 关键要点
- **瓶颈转移**：现代 GPU 上内存带宽主导运行时间
- **设计手段**：tiling（分块）、kernel fusion（算子融合）、重计算
- **通用性**：原则适用于任何访存密集算子

## 来源
- [[dao_2022_flashattention]] — IO 感知算法设计的代表工作
