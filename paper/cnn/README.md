# 计算机视觉 CV

## 趋势&前沿
* 目标检测、语义分割、实例分割等，如何实现多任务无监督的预训练大模型 + 提示学习?  [DetCo](./DetCo.md), [CLIP](../Multimodal/CLIP.md)
* 对比学习 + ViT + 多模态 + 单阶段检测？
* 输入：一张图片 + 一段提示文本(例如：图中人物标记/框出来; 图中有{猫}吗,{猫}在{树上}吗？) CLIP?

## 一、主要任务
1. 图像分类
2. [目标检测](./Object_Detection.md)
3. [语义分割](./Semantic_Segmentation.md)
4. [实例分割](./Instance_Segmentation.md)
5. [深度估计](#)
5. 文字识别(OCR,Optical character recognition 光学字符识别) ~ 以下为具体应用
6. 人体关键点检测
7. 人脸识别
8. 视频: 视频分类、目标追踪、轨迹预测(学习物理常识？)
9. 3D重建

## 二、数据
### 数据集
* MNIST
* Fashion--MNIST
* ImageNet
* MS-COCO
* CIFAR-10
* VOC
* LVIS
* JFT
* Oxford-IIIT Pets
* Oxford Flowers-102
* VTAB
* Open Images 
* VisualQA
* CelebFaces,MegaFace,MS-Celeb-1M
* Cityscapes
* 街景房屋号码(SVHN)

### 数据标注工具
* 数据标注工具 https://blog.51cto.com/u_13984132/5622715 https://github.com/tzutalin/labelImg

### 大规模样本挖掘、数据清理
* 互联网数据清洗，大量内容良莠不齐，数据分布均衡？ 降低高频分布，覆盖长尾分布; 
* CDP半监督聚类 识驱动传播(Consensus-Driven Propagation，CDP)
* 主动学习
    * Settles, B. 2009. Active Learning Literature Survey
    * https://www.mindspore.cn/file/course/document/d004fe29952746b1917b88884b6f4367.pdf
    * https://mp.weixin.qq.com/s/Y8GQnj3bHzwq2LHCNMf6Cg
    * https://github.com/baifanxxx/awesome-active-learning

### 图像增广
* 图像加载、归一化
* 空间/几何变换：裁剪和调整大小（水平翻转）、旋转 和剪切
* 外观变换：颜色失真（包括颜色下降、亮度、对比度、饱和度、色调）、高斯模糊、 Sobel滤波


## 三、卷积网络 CNN 
分辨率、卷积核大小、宽度(卷积核数)、深度(层数)、基数(跳跃连接，深度可分离，分组)，跨阶段部分连接，本质是增加各种特征的组能能力。

1. 卷积 Convolution
* 标准卷积(正方形), 非对称卷积(长方形)
* 池化 pooling (max,avg,GlobalAvg Pooling,overlap-重叠池化,向上池化) ，一种特殊的卷积
    * [Pooling is neither necessary nor sufficient for appropriate deformation stability in CNN](./Pooling.md) 2018-4-12
* 深度卷积 (Depthwise)
    * depthwise separable convolution 深度可分离卷积
* 分组卷积 (Group)
    * pointwise group convolution
    * Channel shuffle ?
* 空洞卷积 (Dilated)
* 转置卷积 (Transposed)
* 可变形卷积 (deformable)

2. 1*1 卷积，主要用于增加/减少通道数，卷积压缩提效的重要工具 https://zhuanlan.zhihu.com/p/40050371

3. 层与层之间的连接方式
    * 跳跃连接，残差, add操作, 通道数不变 Identity Mappings 恒等映射
    * 增加基数, 多分支concat操作，通道数改变 https://juejin.cn/post/7064940823146102820
    * CSP, Cross-Stage-Partial-connections, 跨阶段部分连接

4. 推理部署
    * 评估指标：模型参数量, FLOPs, 内存, 特定设备的实际速度测试 https://zhenhuaw.me/blog/2019/neural-network-quantization-resources.html
    * 深度可分离卷积 depthwise separable convolutions 可有效降低计算量
    * 模型缩放(不同设备不同规模) + NAS, Neural Architecture Search, 神经结构搜索
    * Model re-parameterization 模型重新参数化, 模块级重参数化


### 质量精度提升
1. LeNet 1998 [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)
2. AlexNet 2012-9-30 [ImageNet Classification with Deep Convolutional](./alexnet.md)
3. NIN 2013-12-16 [Network in Network](https://arxiv.org/abs/1312.4400)  
4. VGG: 2014-9-4 [Very Deep Convolutional Networks for Large-Scale Image Recognition](./vgg.md)
5. [Inception](https://juejin.cn/post/7064940823146102820) 多分支卷积网络
    * Inception v1(GoogLeNet, 2014-9-17) [Going deeper with convolutions](./inception_v1.md)
    * Inception v2(BN-Inception 2015-02) [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](./BatchNorm.md)
    * Inception v3 2015.12 [Rethinking the Inception Architecture for Computer Vision](./inception_v3.md)
    * Inception v4 (Inception-ResNet 2016.02)  [Inception-ResNet and the Impact of Residual Connections on Learning](./inception_v4.md)
    * Xception 2016-11-7 [Deep Learning with Depthwise Separable Convolutions](./xception.md)
6. ResNet 2015-12-10  [Deep Residual Learning for Image Recognition](./resnet.md)
    * ResNeXt 2016-11-16 [Aggregated Residual Transformations for Deep Neural Networks](./resnext.md)
7. DenseNet 2016-8-25 [Densely Connected Convolutional Networks](./densenet.md)
    * SparseNet: 2018-4-15 [A Sparse DenseNet for Image Classification](./sparsenet.md) 引入注意力机制
8. CSPNet 2019-11-27 [A New Backbone that can Enhance Learning Capability of CNN](./cspnet.md) Cross Stage Partial 跨阶段部分连接 

### 计算效率提升
1. SqueezeNet 2016-2-24 [AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](./SqueezeNet.md) 早期的一些模型压缩
2. MobileNets 深度可分离卷积 depthwise separable convolutions
    * v1 2017-4-17 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](./MobileNet_v1.md) 
    * v2 2018-1-13 [MobileNetV2: Inverted Residuals and Linear Bottlenecks](./MobileNet_v2.md)
    * v3 2019-5-6 [Searching for MobileNetV3](./MobileNet_v3.md)
3. ShuffleNet 逐点组卷积,通道混洗 pointwise group convolution and channel shuffle
    * v1 2017-7-4 [An Extremely Efficient Convolutional Neural Network for Mobile Devices](./ShuffleNet.md)
    * v2 2018-7-30 [Practical Guidelines for Efficient CNN Architecture Design](./ShuffleNet_v2.md) 输入/输出通道比率、架构的分支数和元素操作对网络推理速度的影响
4. SENet: 2017-9-5 [Squeeze-and-Excitation Networks](./senet.md) 挤压和激励
5. 模型缩放,不同设备不同规模 + NAS, Neural Architecture Search, 神经结构搜索
* Elastic 2018-12-13  [Improving CNNs with Dynamic Scaling Policies](https://arxiv.org/abs/1812.05262) 
* Res2Net 2019-4-2 [A New Multi-scale Backbone Architecture](./Res2Net.md)  
* EfficientNet 2019-5-28 [Rethinking Model Scaling for Convolutional Neural Networks](./EfficientNet.md)
* 2021.3.11 [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877)
https://zhuanlan.zhihu.com/p/320447427
https://www.cvmart.net/community/detail/1506
6. HarDNet 2019-9-3, [A Low Memory Traffic Network](./HarDNet.md)
7. GhostNet 2019-11-27 [More Features from Cheap Operations](./GhostNet.md)
8. MCUNet 2020.7.20 [Tiny Deep Learning on IoT Devices](./MCUNet.md)
9. RepVGG 2021-1-11, [Making VGG-style ConvNets Great Again](./repvgg.md) 
10. 不好放入时间线的
* Shift 2017.8.22 [Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions](https://arxiv.org/abs/1711.08141)
* DiracNet 2017-6-1 [Training Very Deep Neural Networks Without Skip-Connections](https://arxiv.org/abs/1706.00388) 很水啊
* HRNet

### 受ViT启发，重新设计CNN
* 2022.1.10 ConvNeXt [A ConvNet for the 2020s](./ConvNeXt.md)
* 2023.1.2 [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](./ConvNeXt_v2.md)

### 推理速度优化总结 - 工业应用
1. 知识蒸馏 teacher-student
* [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550) Feature-Based 
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) Response-Based
* [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068) Relation-Based,Distance-wise损失函数,angle-wise损失，用于计算三个样本间的角度差异, huber损失函数,当预测偏差小于1时，采用平方误差; 当预测偏差大于δ.
2. 权重量化 Quantization 32float -> 8int(0~255) https://zhuanlan.zhihu.com/p/64744154
    1. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
    4. [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)
    1. [A Survey of Model Compression and Acceleration for Deep Neural Networks](https://arxiv.org/abs/1710.09282)
    模型量化
    Quantize/Dequantize, 码本min,max值
    https://bbs.huaweicloud.com/blogs/293961
    R 表示真实的浮点值，Q 表示量化后的定点值，Z 表示浮点值对应的量化定点值，S 则为定点量化后可表示的最小刻度
    Q = R/S + Z
    R = (Q-Z) * S

    S = (R_max - R_min) / (Q_max - Q_min)
    Z = Q_max - R_max / S

    S最大和最小的浮点数，Z最大和最小的定点数
3. 网络剪枝:训练时剪枝，训练后剪枝
4. 低秩近似


## 四、[目标检测](./Object_Detection.md)

## 五、[语义分割](./Semantic_Segmentation.md)

## 六、[实例分割](./Instance_Segmentation.md)

## 七、视觉Transformer ViT
1. ViT 2020.11.22 [Vision Transformer(ViT): An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](../vit/vit.md)
    * [Vision Transformer: Vit and its Derivatives](../vit/vit_derivatives.md) 2022-5-12
    * DeiT 2012.12.23 [Training data-efficient image transformers & distillation through attention.](https://arxiv.org/abs/2012.12877) 更好的训练策略? 数据增广和正则化操作
    * 2021.3  Transformer in Transformer：TNT  充分挖掘patch内部信息 
    * CaiT 2021.3.31 [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239) LayerScale,Class-Attention layers
    * Tokens-to-token vit: Training vision transformers from scratch on imagenet. arXiv:2101.11986, 2021
    * Deep high-resolution representation learning for visual recognition.
    * [Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882)
    * Do We Really Need Explicit Position Encodings for Vision Transformers?  21.02 位置编码的必要性
2. MAE 2021.11.11 [Masked autoencoders are scalable vision learners](../vit/MAE.md) 类似NLP里BERT+GPT 牛！
    * BEiT 2021.06.15 [BERT pre-training of image transformers](../vit/BEiT.md) 预测离散令牌
    * iBOT 2021.11.15 [Emerging properties in self-supervised vision transformers](../vit/iBOT.md)
    * iGPT 2020.6.17 [Generative Pretraining from Pixels](../vit/iGPT.md) 生成式的
3. ViTDet 2022.3.30 [Exploring Plain Vision Transformer Backbones for Object Detection](../vit/ViTDet.md) 目标检测,干掉了颈部的横向连接
    * 2021.11.22 [Benchmarking Detection Transfer Learning with Vision Transformers](https://arxiv.org/abs/2111.11429)

## 八、其他架构探索
1. MLP-Mixer 2021-5-4, [An all-MLP Architecture for Vision](./mlp-mixier.md)
    * https://mp.weixin.qq.com/s/WwEgHv4b_kkO3b-aP0ovfQ
    * RepMLP 2021-5-5, [Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition](./repmlp.md)     
    * ResMLP 2021-5-7, [Feedforward networks for image classification with data-efficient training](./ResMLP.md)
2. 对比cnn,transfomer,mlp
    * Container: 2021-6-2 [Container: Context Aggregation Network](https://arxiv.org/abs/2106.01401) https://www.cvmart.net/community/detail/4897   
    * SPACH 2021-8-30 [A Battle of Network Structures: An Empirical Study of CNN, Transformer, and MLP](https://arxiv.org/abs/2108.13002) 
    https://www.cnblogs.com/pprp/p/15726282.html
3. NLP,CV的统一
    * 2021.12.14 [Towards a Unified Foundation Model: Jointly Pre-Training Transformers on Unpaired Images and Text](https://arxiv.org/abs/2112.07074)  

## 人脸识别
* CDP [Consensus-Driven Propagation in Massive Unlabeled Data for FaceRecognition](https://arxiv.org/abs/1809.01407) https://github.com/XiaohangZhan/cdp https://mp.weixin.qq.com/s/Ubcw9KOmPBPxAUqaRmd3cA  affinity graph
