# 激活函数 Activation functions

如果只是线性函数嵌套，最终的复合函数仍然还是线性函数，而通过线性+非线性函数的深度堆叠，则可以模拟各种多项式函数。

激活函数和归一化策略有一定的关联关系，

做一个激活函数直观的交互展示页面，将所有激活函数集成在一起，允许用户选择关心的激活函数进行对比。

## From [yolo_v4](../paper/cnn/yolo_v4.md):
A good activation function can make the gradient more efficiently propagated, and at the same time it will not cause too much extra computational cost. In 2010, Nair and Hinton [56] propose ReLU to substantially solve the gradient vanish problem which is frequently encountered in traditional tanh and sigmoid activation function. Subsequently, LReLU [54], PReLU [24], ReLU6 [28], Scaled Exponential Linear Unit (SELU) [35], Swish [59], hard-Swish [27], and Mish [55], etc., which are also used to solve the gradient vanish problem, have been proposed. The main purpose of LReLU and PReLU is to solve the problem that the gradient of ReLU is zero when the output is less than zero. As for ReLU6 and hard-Swish, they are specially designed for quantization networks. For self-normalizing a neural network, the SELU activation function is proposed to satisfy the goal. One thing to be noted is that both Swish and Mish are continuously differentiable activation function.

一个好的激活函数可以使梯度更有效地传播，同时不会造成太多额外的计算成本。 2010年，Nair和Hinton [56] 提出 ReLU 来大幅解决传统 tanh 和 sigmoid 激活函数中经常遇到的梯度消失问题。随后，LReLU [54]、PReLU [24]、ReLU6 [28]、Scaled Exponential Linear Unit (SELU) [35]、Swish [59]、hard-Swish [27]和Mish [55]等已经被提出用于解决梯度消失问题。 LReLU和PReLU的主要目的是解决输出小于零时ReLU的梯度为零的问题。至于 ReLU6 和 hard-Swish，它们是专门为量化网络设计的。对于神经网络的自归一化，提出了 SELU 激活函数来满足目标。需要注意的一点是 Swish 和 Mish 都是连续可微的激活函数。

* Sigmod
    * https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
    * LogSigmoid
        * https://pytorch.org/docs/stable/generated/torch.nn.LogSigmoid.html
    * SiLu, Sigmoid Linear Unit : 输出小于0 silu(x)=x∗σ(x)
        * https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
* Tanh
    * https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
    * Tanhshrink
* ReLU  2010.4 [Rectified linear units improve restricted boltzmann machines](#56)
    * https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    * rectified linear unit, 整流线性单元 $ReLU(x)=max(0,x)$
* LReLU 2013.4 [Rectifier nonlinearities improve neural network acoustic models.](#54)
    * https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.LeakyReLU.html
    * $LeakyReLU(x)=max(0,x) + negative_slope * min(0,x) $, 适合边缘GPU?
* PReLU 2015.2.6 [Delving deep into rectifiers:Surpassing human-level performance on imagenet classification](https://arxiv.org/abs/1502.01852)
    * https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
* GELU, 2016.6.27 [GELU, Gaussian Error Linerar Units](./GELUs.md) 
    * https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
* ReLU6 2017.4.17 [MobileNets: Efficient convolutional neural networks for mobile vision applications](./MobileNet_v1.md)
    * https://pytorch.org/docs/stable/generated/torch.nn.ReLU6.html
    * $min(max(0,x),6)$, 专为量化网络设计
* SELU  2017.4  [Self-normalizing neural networks](#35)
    * Scaled Exponential Linear Unit 
    * https://pytorch.org/docs/stable/generated/torch.nn.SELU.html
* Swish 2017.4 [Searching for activation functions](#59) 
    * sigmoid + Relu, 无上界有下界、平滑、非单调
* hard-Swish 2019.5.6 [Searching for MobileNetV3](./MobileNet_v3.md)
    * https://pytorch.org/docs/stable/generated/torch.nn.Hardswish.html
    * 专为量化网络设计
* Mish  2019.4 [Mish: A self regularized nonmonotonic neural activation function](#55)
    * https://pytorch.org/docs/stable/generated/torch.nn.Mish.html

softsign $\frac{x}{(1 + |x|)}$

SwiGLU, Shazeer(2020) LLaMA


* ReLU, 
    * ELU, $ELU(x)=max(0,x) + min(0,a*(exp(x)-1)) $    
    * CELU











https://zhuanlan.zhihu.com/p/99401264 
https://cloud.tencent.com/developer/article/1521839

![Sigmod and Tanh](./images/Activation/sigmoid_tanh.png)

![S](./images/Activation/sigmoid_tanh_relu_softplus.png)


![PReLU, 2015.2.6](./images/Activation/relu_prelu.png)

![GLUE,ReLU,ELU](./images/GELUs/fig_1.png)