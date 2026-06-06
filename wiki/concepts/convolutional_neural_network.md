---
id: convolutional_neural_network
type: concept
tags: [machine-learning, empirical-study, theoretical]
aliases: [CNN, 卷积神经网络, ConvNet]
related_nodes: [lecun_1998_lenet, krizhevsky_2012_alexnet, simonyan_2014_vgg, resnet_2015, cnn_summary, deepnet]
last_verified: 2026-06-06
---

# Convolutional Neural Network

## 定义
卷积神经网络（CNN）是一种利用卷积操作捕捉局部空间相关性的前馈神经网络，通过权值共享和局部连接大幅降低参数量，特别适用于图像等具有网格结构的数据。

## 核心组件
- **卷积层** — 学习局部特征检测器（kernel/filter）
- **池化层** — 下采样降维（最大池化、平均池化）
- **激活函数** — 引入非线性（ReLU、GELU）
- **全连接层** — 组合高层特征进行分类

## 标志性架构
| 架构 | 创新 | 来源 |
|:---|:---|:---|
| LeNet-5 | 奠定 CNN 基础结构 | [[lecun_1998_lenet]] |
| AlexNet | 深度 CNN + GPU 加速 + ReLU | [[krizhevsky_2012_alexnet]] |
| VGG | 小卷积核堆叠 | [[simonyan_2014_vgg]] |
| ResNet | 残差连接突破深度瓶颈 | [[resnet_2015]] |

## 来源
- [[lecun_1998_lenet]] — 卷积网络奠基
- [[krizhevsky_2012_alexnet]] — 深度 CNN 在 ImageNet 上的突破
- [[simonyan_2014_vgg]] — 小卷积核深度结构
- [[resnet_2015]] — 残差连接革命
- [[cnn_summary]] — CNN 架构演变综述
