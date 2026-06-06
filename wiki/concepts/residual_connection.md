---
id: residual_connection
type: concept
tags: [machine-learning, empirical-study, theoretical]
aliases: [Skip Connection, 跳跃连接, 残差连接]
related_nodes: [resnet_2015, kaiming_he, transformer_architecture]
last_verified: 2026-06-06
---

# Residual Connection

## 定义
残差连接（Residual Connection / Skip Connection）将层的输入直接加到其输出上，形成恒等捷径（identity shortcut），使梯度可直接流入浅层，从而缓解深层网络的退化问题。

## 核心机制
$$y = \mathcal{F}(x) + x$$

- **梯度高速公路**：梯度经恒等路径无损回传，缓解梯度消失
- **隐式集成**：每个残差块可视为浅层网络的集成（Veit et al. 2016）
- **代表性架构**：ResNet（[[resnet_2015]]）是首个大规模使用残差连接的架构

## 影响
残差连接是现代深度网络的必备组件，从 CNN（ResNet、DenseNet）到 Transformer（[[vaswani_2017_transformer]]）均依赖此机制。

## 来源
- [[resnet_2015]] — 残差连接的奠基性工作
