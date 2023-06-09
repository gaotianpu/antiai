# A Survey of Model Compression and Acceleration for Deep Neural Networks
深度神经网络模型压缩与加速研究综述  https://arxiv.org/abs/1710.09282

深度神经网络(DNN)最近在许多视觉识别任务中取得了巨大成功。然而，现有的深度神经网络模型计算成本高且内存密集，阻碍了它们在低内存资源的设备或具有严格延迟要求的应用中的部署。因此，自然的想法是在深度网络中执行模型压缩和加速，而不会显著降低模型性能。在过去五年中，这一领域取得了巨大进展。在本文中，我们回顾了最近用于压缩和加速DNN模型的技术。一般来说，这些技术分为四类：参数修剪和量化、低阶因子分解、迁移/紧凑卷积滤波器和知识蒸馏。首先描述了参数修剪和量化的方法，然后介绍了其他技术。对于每个类别，我们还提供了关于性能、相关应用程序、优点和缺点的深入分析。然后我们介绍了一些最近成功的方法，例如动态容量网络和随机深度网络。之后，我们调查了评估矩阵、用于评估模型性能的主要数据集以及最近的基准工作。最后，我们总结了本文，讨论了剩余的挑战和未来工作的可能方向。

