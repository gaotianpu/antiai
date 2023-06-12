# 归一化 Normalization  
具有多个维度的向量，每个维度所表征的含义不同，其度量单位可能会不同，因此取值范围也会不同。例如：描述人的向量，身高、体重这两个维度，身高的单位是m，取值范围可能在0.5~2.3m之间，而体重单位是kg，可能在10~80kg之间。

## 一、为何要归一化

我们知道深度神经学习网络是将输入的向量经过堆叠的线性和非线性函数后得到最终的输出。根据线性函数的公式：

$f(X) = w_0*x_0 + w_1*x_1 + ... + w_n*x_n + b $

当输入向量每个维度取值范围不一致时，W和b这些不同维度上的参数更新时就比较难以把握。

因此，我们需要将向量不同维度的值区间统一起来，这种操作就叫做归一化。


## 二、归一化函数
### 1. 最大最小归一化 (Min-Max Normalization)

公式： $ Min-Max-Norm(x) = \frac{x-min(x)}{max(x)-min(x)}$ ， 

Norm(x)的输出范围[0,1]。

给定一个向量集合，分别计算向量每个维度的最大值和最小值，将每个向量对应维度上的值代入上式，得到归一化后的值。

这种方法有一个较典型的应用: 图像的每个RGB的像素值是[0,255]之间，应用最大最小化后，得到[0,1]之间的浮点值。

该方法会受到数据集中异常值的影响，若果max,min值比大多数正常值差异很大，将会导致正常值之间的区分度很小。不利于后续的模型处理。


### 2. 标准归一化
公式 : $STD_Normal(x) = \frac{x-μ}{σ}$, 

μ代表向量集中该维度值的均值，σ代表标准差。实际上这个公式，也是计算z-score.

<!-- ### 3. log对数函数归一化

### 4. 反正切函数

### 5. L2范数 -->

## 三、具体应用
上面讲到的是归一化函数，根据选择的数值集合不同，又有以下几种不同的归一化策略。

### 1. [Batch normalization](../paper/cnn/BatchNorm.md) 批归一化 
Cross mini-Batch Normalization (CmBN) 

https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html  #1D,2D,3D

### 2. [Layer Normalization](../paper/LayerNorm.md) , 层归一化
    * https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    * [Foundation Transformers](../paper/Multimodal/MAGNETO.md) 比较了post-LN, pre-LN
### 3. GN [Group Normalization](../paper/cnn/GroupNorm.md)
    * https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
### 4. LRN(local response normalization) 局部响应归一
    * https://pytorch.org/docs/stable/generated/torch.nn.LocalResponseNorm.html
    * local contrast normalization 局部对比归一
### 5. Instance Normalization(IN)
    * https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html #1D,2D,3D

* Switchable Normalization(SN)
* Weight Normalization(WN)
*  Cosine Norm

RMSNorm, Zhang and Sennrich (2019)

https://bbs.huaweicloud.com/blogs/187196

统一单位量, ICS(Internal Covariate Shift) 内部协变量迁移 ?