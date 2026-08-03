---
id: end_to_end_driving
type: concept
tags: [computer-vision, machine-learning, empirical-study]
aliases: [端到端驾驶, 直接感知控制]
related_nodes: [bojarski_2016_dave2, imitation_learning, transfuser]
last_verified: 2026-08-03
---

# End-to-End Driving

## 定义
从传感器原始输入（相机像素）直接学习驾驶命令（转向/油门）的范式，跳过手工设计的感知-规划-控制流水线。

## 关键要点
- **单网络映射**：CNN 像素 → 转向命令，如 DAVE-2
- **优点**：无手工特征，感知-控制联合优化
- **局限**：可解释性差、安全验证难、分布外脆弱；现代方案（TransFuser 等）转向多模态融合缓解

## 来源
- [[bojarski_2016_dave2]] — DAVE-2 端到端驾驶
