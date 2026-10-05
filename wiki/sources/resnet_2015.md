---
id: resnet_2015
type: source
tags: [computer-vision, machine-learning, empirical-study]
aliases: [Deep Residual Learning for Image Recognition, 残差网络, ResNet, 1512.03385]
related_nodes: [microsoft, kaiming_he, residual_connection, bottleneck_architecture, convolutional_neural_network, batch_normalization, ioffe_2015_batchnorm, hu_2017_senet, gao_2019_res2net, ding_2021_repvgg, wang_2017_nonlocal, chen_2017_deeplabv3, zhao_2016_pspnet]
arxiv_id: 1512.03385
authors: Kaiming He et al.
authors_institution: Microsoft
last_verified: 2026-10-05
---

# Deep Residual Learning for Image Recognition

- **元数据**: CVPR 2016 | arXiv 2015 | **作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | **机构**: Microsoft | 相关: [[residual_connection]]
- **概述**: 提出残差学习框架，让堆叠层拟合残差映射 $F(x)=H(x)-x$，以无参数的恒等捷径解决深层网络的退化问题，首次稳定训练 152 层网络，夺得 ILSVRC 2015 分类冠军。
- **新颖概念**: [[residual_connection]]、[[bottleneck_architecture]]
- **关键要点**: 1. 退化问题的本质：34 层 plain 网训练误差**高于** 18 层，尽管其解空间更大；BN 下前向/反向信号均不消失，故非梯度消失，而是深层收敛率过低 2. 恒等捷径不增参数与计算量；投影捷径的 A/B/C 三方案差异极小，恒等映射已足以解决退化 3. [[bottleneck_architecture]]「1×1 降维 → 3×3 → 1×1 升维」使 152 层复杂度（11.3 GFLOPs）仍低于 VGG-19（19.6 GFLOPs） 4. ImageNet：ResNet-152 单模型 top-5 验证错误 4.49%，优于此前全部集成结果；6 模型集成测试集 3.57%，夺 ILSVRC 2015 冠军 5. CIFAR-10：ResNet-110 达 6.43%；1202 层训练误差 <0.1% 但测试 7.93%，归因于过拟合
- **方法/发现**: ImageNet 训练配置——SGD、batch 256、lr 0.1（平台期 ÷10）、weight decay 1e-4、momentum 0.9、≤60 万次迭代、卷积后激活前接 [[batch_normalization]]、不用 dropout；测试用 10-crop，最佳结果取全卷积多尺度（短边 {224,256,384,480,640}）。层响应分析显示残差分支响应普遍小于 plain 网，且越深单层改动越小，支持恒等映射起预处理作用的假设。COCO 检测以 ResNet-101 替换 VGG-16，mAP@[.5,.95] 相对提升 28%。
- **局限/意义**: 1202 层在 CIFAR-10 上过拟合（未引入 maxout/dropout 等强正则化）；极深网络的优化困难成因留待后续研究。残差连接此后成为 CNN 与 Transformer 的默认组件，本页是该范式的奠基出处。

## 引用
- **原始论文**: [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | [阅读笔记](../../raw/cnn/resnet.md)
- **相关概念**: [[residual_connection]] | [[bottleneck_architecture]] | [[convolutional_neural_network]]
