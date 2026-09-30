---
id: kaiming_he
type: entity
tags: [person, computer-vision, machine-learning]
aliases: [何恺明, Kaiming He]
related_nodes: [resnet_2015, mask_rcnn_2017, sppnet_2014, mae, vitdet, moco_v3, focal_loss_2017, fpn_2016]
last_verified: 2026-09-30
---

# 何恺明 (Kaiming He)

## 概述
计算机视觉领域最具影响力的学者之一，以残差网络（ResNet）等系列工作推动了深度学习在视觉领域的突破；2024 年起任职于 MIT，研究重心扩展至生成模型与自监督学习。

## 关键贡献
- [[sppnet_2014]]（2014）：空间金字塔池化
- [[resnet_2015]]（2015）：残差学习 ResNet，152 层可训练网络
- [[fpn_2016]]（2016）：特征金字塔网络 FPN
- [[mask_rcnn_2017]]（2017）：Mask R-CNN 实例分割
- [[focal_loss_2017]]（2017）：Focal Loss（合作者）
- [[moco_v3]]（2020）：MoCo v3 自监督 ViT
- [[mae]]（2021）：掩码自编码器 MAE
- [[vitdet]]（2022）：ViTDet 检测骨架

> 以下为 2023 年至今的工作（尚未创建 source 页，链接指向 arXiv）：

- [RAGE: Return of Unconditional Generation](https://arxiv.org/abs/2312.03701)（2023）：自监督表征生成，使无条件生成模型逼近条件生成
- [Deconstructing Denoising Diffusion Models for Self-Supervised Learning](https://arxiv.org/abs/2401.14404)（2024）：拆解扩散模型用作自监督表征学习
- [A Decade's Battle on Dataset Bias: Are We There Yet?](https://arxiv.org/abs/2403.08632)（2024）：用合成数据检验数据集偏差问题十年进展
- [Dynamic Inhomogeneous Quantum Resource Scheduling with Reinforcement Learning](https://arxiv.org/abs/2405.16380)（2024）：Transformer+RL 量子资源调度（MIT 跨界合作）
- [TetSphere Splatting](https://arxiv.org/abs/2405.20283)（2024）：拉格朗日体网格的高质量 3D 几何表示
- [Physically Compatible 3D Object Modeling from a Single Image](https://arxiv.org/abs/2405.20510)（2024）：单图生成物理兼容 3D 物体
- [MAR: Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)（2024）：扩散损失建模连续 token，摆脱向量量化依赖
- [HPT: Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers](https://arxiv.org/abs/2409.20537)（2024）：异构预训练 Transformer 的机器人学习
- [Fluid: Scaling Autoregressive Text-to-image Generation with Continuous Tokens](https://arxiv.org/abs/2410.13863)（2024）：连续 token 自回归文生图的缩放研究
- [Is Noise Conditioning Necessary for Denoising Generative Models?](https://arxiv.org/abs/2502.13129)（2025）：挑战去噪生成模型依赖噪声条件的共识
- [Fractal Generative Models](https://arxiv.org/abs/2502.17437)（2025）：递归调用原子生成模块的分形架构
- [Denoising Hamiltonian Network for Physical Reasoning](https://arxiv.org/abs/2503.07596)（2025）：哈密顿算子约束的物理推理网络
- [Transformers without Normalization (Derf)](https://arxiv.org/abs/2503.10622)（2025）：Dynamic Tanh 替代归一化层
- [MeanFlow: Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)（2025）：平均速度场实现一步生成
- [Disperse: Diffuse and Disperse](https://arxiv.org/abs/2506.09027)（2025）：Dispersive Loss 正则化扩散模型
- [JiT: Back to Basics — Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)（2025）：直接预测干净数据而非噪声
- [ARC Is a Vision Problem!](https://arxiv.org/abs/2511.14761)（2025）：把 ARC 抽象推理当作视觉问题处理
- [Improved Mean Flows](https://arxiv.org/abs/2512.02012)（2025）：改进 MeanFlow 的训练目标与引导机制
- [Bidirectional Normalizing Flow](https://arxiv.org/abs/2512.10953)（2025）：数据与噪声互转的双向归一化流
- [Pixel Mean Flows](https://arxiv.org/abs/2601.22158)（2026）：融合一步采样与无隐空间的像素级生成
- [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)（2026）：Drifting Models——训练中演化推前分布，天然一步推理
- [GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training](https://arxiv.org/abs/2602.20399)（2026）：几何预训练加速物理仿真
- [Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329)（2026）：图像生成器零样本视觉理解能力的实证
- [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938)（2026）：连续扩散语言模型的有效实现
- [Video Generation Models are General-Purpose Vision Learners](https://arxiv.org/abs/2607.09024)（2026）：文生视频作为计算机视觉通用预训练范式
