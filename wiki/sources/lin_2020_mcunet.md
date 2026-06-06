---
id: lin_2020_mcunet
type: source
tags: ["computer-vision", "machine-learning", "empirical-study"]
aliases: ["MCUNet", "2007.10319"]
related_nodes: []
arxiv_id: 2007.10319
authors: Ji Lin et al.
authors_institution: Massachusetts Institute of Technology
last_verified: 2026-06-06
---

# MCUNet: Tiny Deep Learning on IoT Devices

- **元数据**: arXiv | 2020 | **作者**: Ji Lin, Wei-Ming Chen, Yujun Lin, John Cohn, Chuang Gan, Song Han | **机构**: Massachusetts Institute of Technology
- **概述**: 提出 MCUNet 系统-模型协同设计框架，联合优化 TinyNAS 神经架构搜索和 TinyEngine 轻量推理引擎，首次在商用 MCU 上实现 >70% ImageNet 精度。
- **关键要点**: 1. TinyNAS 两阶段 NAS：自动优化搜索空间 + 资源约束下模型特化 2. TinyEngine 通过代码生成和模型自适应内存调度减少 3.4× 内存 3. 在 STM32 MCU（320kB SRAM）上 ImageNet 达 70.7% Top-1
- **方法/发现**: 系统-算法协同设计 + 自动搜索空间优化 + 代码生成推理引擎 + 原地深度可分离卷积
- **局限/意义**: 开启物联网设备上的 TinyML 时代，为资源极度受限场景提供可行方案。

## 引用
- **原始论文**: [arXiv](https://arxiv.org/abs/2007.10319) | [阅读笔记](../../raw/cnn/MCUNet.md)
