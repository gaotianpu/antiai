---
id: he_2021_alpha_iou
type: source
tags: ["computer-vision", "machine-learning", "theoretical"]
aliases: ["Alpha-IoU", "α-IoU", "Power IoU Loss", "幂次交并比损失", "2110.13675"]
related_nodes: ['iou_loss', 'iou_loss_survey']
arxiv_id: 2110.13675
authors: Jiabo He et al.
last_verified: 2026-09-30
---

# Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression

- **元数据**: NeurIPS 2021 | **作者**: Jiabo He, Sarah Erfani, Xingjun Ma, James Bailey, Ying Chi, Xian-Sheng Hua | 相关: [[iou_loss]]
- **概述**: 用 Box-Cox 变换把现有 IoU 系损失统一为幂次族：$L_{\alpha-IoU}=(1-IoU^\alpha)/\alpha$，单参数 $\alpha$ 同时调节回归精度、梯度加权与鲁棒性。
- **新颖概念**: [[iou_loss]]
- **关键要点**: 1. $\alpha\to0$ 时退化为 $-\log(IoU)$，$\alpha=1$ 为标准 IoU 损失，$\alpha=2$ 得平方 IoU 项 2. 可把 GIoU/DIoU/CIoU 等正则项一起幂次推广 3. 保持 IoU 的序关系，同时重新分配损失与梯度权重 4. 在小数据集和带噪声 bbox 场景更鲁棒
- **方法/发现**: 从 IoU 损失出发做幂变换，给出统一表达式，并在多个检测器与基准上验证不同 α 可换取不同精度/鲁棒性水平。
- **局限/意义**: α 是新增超参数，不同数据集/检测器的最优值不同；理论与实验表明它更适合作为统一分析框架而非“万能替代”。

## 引用
- **原始论文**: [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html) | [arXiv:2110.13675](https://arxiv.org/abs/2110.13675)
- **相关页面**: [[iou_loss]], [[iou_loss_survey]]
