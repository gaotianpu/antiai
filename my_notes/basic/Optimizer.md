# 优化器 Optimizer

![optim](./images/optim.png)<br/>
图源：https://www.paddlepaddle.org.cn/tutorials/projectdetail/4459568


* BGD(Batch gradient descent), 采用整个训练集的数据
* SGD(Stochastic Gradient Descent) – 随机梯度下降法
    * https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
* AdaGrad(Adaptive gradient algorithm), 
    * 对低频的参数做较大的更新，对高频的做较小的更新
    * https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
* Adadelta,对 Adagrad 的改进
    * https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
* Adam = Momentum + AdaGrad
    * https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
* AdamW
    * https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
* NAdam
    * https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html
* RMSprop, AdaGrad的基础上，增加一个衰减系数ρ控制历史信息的获取量
    * https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html


* Momentum, 动量
* Nesterov, 对Momentum的改进和优化

* MBGD(Mini-Batch Gradient Descent), 小批量数据  
* NAG(Nesterov Accelerated Gradient),

https://www.cnblogs.com/guoyaohua/p/8542554.html

## 学习率衰减：
https://zhuanlan.zhihu.com/p/93624972

1. 指数衰减
2. 固定步长衰减
3. 多步长衰减
4. 余弦退火衰减,啥道理？

余弦学习率, cosine learning rate schedule

weight decay 权重衰减 0.1 
gradient clipping 梯度剪裁 1.0

* 模拟退火算法，逐渐缩小随机范围？
* 最小二乘法？
