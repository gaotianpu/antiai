# 归一化 Normalization  


## 一、为何要归一化
统一单位量, ICS(Internal Covariate Shift) 内部协变量迁移 ?

梯度消失梯度爆炸 https://www.cnblogs.com/shine-lee/p/11989612.html

输入前
各隐藏层

## 二、方法
* 标准归一化
* 最大最小值归一化

## 三、方法
* BN [Batch normalization](./cnn/BatchNorm.md) 批归一化 
    * Cross mini-Batch Normalization (CmBN) 
    * https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html  #1D,2D,3D
* LN [Layer Normalization](./LayerNorm.md) , 层归一化 transformer里
    * https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    * [Foundation Transformers](./Multimodal/MAGNETO.md) 比较了post-LN, pre-LN
* GN [Group Normalization](./cnn/GroupNorm.md)
    * https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
* LRN(local response normalization) 局部响应归一
    * https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html
    * local contrast normalization 局部对比归一
* Instance Normalization(IN)
    * https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html #1D,2D,3D
* Switchable Normalization(SN)
* Weight Normalization(WN)
*  Cosine Norm

RMSNorm, Zhang and Sennrich (2019)

https://bbs.huaweicloud.com/blogs/187196