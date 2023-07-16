# 语义分割

* 数据集: Pascal Voc, Cityscapes, MSCOCO
* 评估指标: 像素精度(pixel accuracy), 平均像素精度(mean pixel accuracy), 平均交并比(Mean Intersection over Union), 权频交并比(Frequency Weight Intersection over Union)
* 思路: 滑窗、候选区域、编码器-解码器、
[语义分割](https://blog.csdn.net/Mind_programmonkey/article/details/120846094) 
https://zhuanlan.zhihu.com/p/538050231
* FCN 2014-11-14 [Fully Convolutional Networks for Semantic Segmentation](./FCN.md)
* U-Net 2015-5-18 [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* SegNet 2015-11-2 [A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561) 
* Fully Convolutional DenseNets 2016-11-28 [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326) 
* DeepLab
    * v1 2014-12-22 [Semantic Image Segmentation With Deep Convolutional Nets and Fully Connected CRFS](https://arxiv.org/abs/1412.7062)
    * v2 2016-6-2 [Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)
    * v3 2017-6-17 [Rethinking Atrous Convolution for Semantic Image Segmentation](./DeepLab_v3.md)
* RefineNet 2016-11-20 [Multi-Path Refinement Networks for High-Resolution Semantic Segmentation](https://arxiv.org/abs/1611.06612)
* PSPNet 2016-12-4 [Pyramid Scene Parsing Network](./pspnet.md)  PPM(Pyramid Pooling Module) 基于不同区域的上下文聚合来利用全局上下文信息的能力
* Large Kernel Matters 2017-5-8 [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719)
* DANet 2018-9-9 [Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983)
<!--
* Non-Local
* CCNet
* Gated-SCNN: Gated Shape CNNs for Semantic Segmentation
* SETR, TransUNet
* SegFormer
* PVT 
* Swin Transformer v1,v2
* Shunted Transformer
* Segmenter
* MaskFormer
* MagNet


https://blog.csdn.net/weixin_46142822/article/details/123969164
强监督：
* ReSTR: Convolution-free Referring Image Segmentation Using Transformers
* Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation
* Deep Hierarchical Semantic Segmentation
* Semantic Segmentation by Early Region Proxy
* SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation
* Rethinking Semantic Segmentation: A Prototype View
弱监督：
* Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation
* Multi-class Token Transformer for Weakly Supervised Semantic Segmentation
* Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers
* Self-supervised Image-specific Prototype Exploration for Weakly Supervised Semantic Segmentation
* Cross Language Image Matching for Weakly Supervised Semantic Segmentation
* Weakly Supervised Semantic Segmentation using Out-of-Distribution Data
* Threshold Matters in WSSS: Manipulating the Activation for the Robust and Accurate Segmentation Model Against Thresholds
半监督:
* ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation
* Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels
无监督：
* GroupViT: Semantic Segmentation Emerges from Text Supervision
-->