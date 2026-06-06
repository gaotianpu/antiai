---
id: li_2022_time3d
type: source
tags: ["3d-detection", "3d-tracking", "monocular"]
aliases: ["Time3D", "2205.14882"]
related_nodes: []
arxiv_id: 2205.14882
authors: Peixuan Li et al.
authors_institution: CVPR 2022
last_verified: 2026-06-06
---

# Time3D: End-to-End Joint Monocular 3D Object Detection and Tracking for Autonomous Driving

- **元数据**: CVPR 2022 | 2022 | **作者**: Peixuan Li et al. | **机构**: CVPR 2022
- **概述**: 以端到端方式从单目视频联合训练 3D 检测和 3D 跟踪
- **关键要点**: 1. 时空信息流模块聚合几何和外观特征预测相似性 2. Transformer 自注意力和交叉注意力分别处理空间和时序 3. 时间一致性损失使 3D 轨迹在世界坐标系中平滑
- **方法/发现**: 端到端联合训练 + 时空 Transformer + 时间一致性损失
- **局限/意义**: 在 nuScenes 3D 跟踪上达 SOTA（21.4% AMOTA），38 FPS

## 引用
- **原始论文**: [arXiv:2205.14882](https://arxiv.org/abs/2205.14882) | [阅读笔记](../../raw/Autonomous_Robot/Time3D.md)
