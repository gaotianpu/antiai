# Deeply-Supervised Nets
深度监督网(DSN) 2014.09.18 https://arxiv.org/abs/1409.5185

## Abstract
Our proposed deeply-supervised nets (DSN) method simultaneously minimizes classification error while making the learning process of hidden layers direct and transparent. We make an attempt to boost the classification performance by studying a new formulation in deep networks. Three aspects in convolutional neural networks (CNN) style architectures are being looked at: (1) transparency of the intermediate layers to the overall classification; (2) discriminativeness and robustness of learned features, especially in the early layers; (3) effectiveness in training due to the presence of the exploding and vanishing gradients. We introduce "companion objective" to the individual hidden layers, in addition to the overall objective at the output layer (a different strategy to layer-wise pre-training). We extend techniques from stochastic gradient methods to analyze our algorithm. The advantage of our method is evident and our experimental result on benchmark datasets shows significant performance gain over existing methods (e.g. all state-of-the-art results on MNIST, CIFAR-10, CIFAR-100, and SVHN).

我们提出的深度监督网(DSN)方法最小化分类错误的同时使隐藏层的学习过程直接透明。我们试图通过研究深度网络中的新公式来提高分类性能。卷积神经网络(CNN)风格架构的三个方面正在被研究：(1)中间层对整体分类的透明度; (2) 学习特征的辨别力和稳健性，尤其是在早期阶段; (3) 由于存在爆炸和消失的梯度，训练的有效性。除了输出层的总体目标(与分层预训练不同的策略)之外，我们还将“同伴目标”引入各个隐藏层。我们扩展了随机梯度方法的技术来分析我们的算法。我们的方法的优势是显而易见的，我们在基准数据集上的实验结果表明，与现有方法相比，我们的性能显著提高(例如，在MNIST、CIFAR-10、CIFAR-100和SVHN上的所有最新结果)。