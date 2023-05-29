# 线性回归 Linewr regression

人类能够通过观察到的现象总结背后的客观规律，再根据规律预测未来可能发生现象，或者过往发生过哪些事情。人工智能也具备这种能力，从现象学习规律，再从规律预测现象。人工智能中的深度学习技术由一些基础模块组成，线性回归是其中的一个重要模块。我们从最基础的线性回归开始，循序渐进，学习一个强大的AI系统是如何构建起来的。

## 一、发现规律：函数 function
![函数示意图](./images/function.png)<br/>
function 本来的意思是作用、功能。可以把它理解成一个黑盒子：接收输入，内部转换后，再输出。例如，爆米花机接收大米作为输入，内部高温处理后，输出爆米花。工厂接收各种原材料，内部加工后，输出对应的工业制成品。 

在数学上，函数的输入可以是一个标量，也可以是向量，矩阵、张量等。函数的内部转换处理机制，可以很简单，例如f(x)=x,输入啥，原封不动的输出也是啥。也可以很复杂, 函数套函数，上一个函数的输出值再传递到若干下游个函数继续处理等。对于函数的输出，不同的输入向量可以映射到相同的输出向量上，但是，相同的输入向量，绝对不能映射为不同的输出向量。

通过函数，我们可以将输入和输出建立映射关系。
* 当输入为1D时，f(x) = w*x + b, f(x)表示为2D空间中的一条直线。w为直线的斜率，b为直线的偏移量。
* 当输入为2D时，$f([x_0,x_1]) = w_0*x_0+w_1*x_1+b$, $f([x_0,x_1])表示为3D(立体)空间一个平面
* 当输入为3D时，$f([x_0,x_1,x_2]) = w_0*x_0+w_1*x_1+w_2*x_2+b$, $f([x_0,x_1])表示为4D(超空间)一个超平面。
* 可以简写为向量形式：$f(X) = W*X + b, 大写的X表示输入向量，W表示系数向量。
* 以上这些函数，不管输入维度多少，叫n元一次方程，直观上和直线、平面、超平面等同，因此被叫做线性(Linewr)函数。
* 对于非一次方程，例如 $f(x)=w*x^2 + b$ , 一元二次方程，在2D平面上表现为一个曲线，因此，叫做非线性函数.
* 一组输入，可以经过几个不同的函数分别产生对应的不同输出，为了便于表示，可以将W看做一个矩阵，行数等于函数的个数，列数等于每一个函数内部的参数数量，f(x)将不再是一个标量值，而是一个向量值。


现在，我们从一个最简单开始，一个匀速移动的物体，想象一下，我们用某种测距仪器，采样了若干个时刻时，该物体的移动距离。在2维平面坐标上的轨迹采样是一组向量,x轴表示时间，y轴表示距离起始点位置，轨迹$(x_0,y_0),(x_1,y_1),(x_2,y_2) ... $。 

![Linewr](./images/line_regression.png)<br/>

那么，如何根据这些轨迹向量，确定移动速度和起始位置？ 因为这个映射关系非常简单，从图中可以一眼看出，匀速移动物体，移动距离跟时间是一种非常简单的线性关系：f(x) = w*x + b 。只需找到恰当的w,b值，就能明确这个线性函数。

确定函数的形式后，如何才能找到函数的参数值呢？

## 二、如何从观察到的现象找出规律？
1. 选择模型：以平面上符合线性分布的点集举例，他们的分布符合 y = wx + b 这样的线性分布规律，求具体w,b的值；
y = wx + b, w值的改变会影响直线的角度(旋转)，b值的改变会影响直线沿x轴的偏移。
2. 模型初始化：先随机初始化w,b的值；基于实际的x值，根据当前随机初始化的wb值，计算得出当前的y值，它不是真实的y值，只是预测值,记作$y_{pre}$;
3. 定义损失函数：当预测的值$y_{pre}$与真实的y值没有差别时，就说明wb值就是线性函数的参数。如何衡量预测值和实际值的差别呢？ 有两种方式可供选择比较：一种绝对值的方式: | $y_{pre} - y$ | -> | wx + b - y | , 另一种是平方差的形式: $(y_{pre} - y)^2 $ -> $(wx + b - y)^2$ 。 注意，这个公式中，x,y是已知的，未知的是w, 画出直观图，会发现绝对值会有一个尖锐的拐点，而平方差是一个平滑的曲线，因此选择平方差，有利于每次调整w,b时，更平滑。

![Loss](./images/mse.png)

4. 该如何调整函数的具体参数, 三种方式：
* 模拟退火算法
* 最小二乘法: 损失函数$(wx+b-y)^2=0$时损失函数最小。
* 随机梯度下降：当预测值和真实值有差别时，该怎么调整wb值来缩小这个差别呢？ wb值该调大还是调小，具体该调多少值？平方差函数是个连续凸函数，每次变更wb值后，可以计算该值下的偏导数，偏导数的概念：$(wx + b - y)^2$式子中，w和b都是要最终求取的未知数，先固定b的当前值，b为已知数，整个式子中只有w是未知数，计算w方向上这个时刻的导数，同理，固定住w后，求得b点的导数；
5. 学习率：w,b导数求出后，知道他们的调整大小、调整幅度，但为了避免偏导数每次调整幅度过大，折线的路径太长，需要设置个学习率，在wb偏导数基础上乘以一个(0~1)之间的学习率。
6. 迭代方式可以选择一次一个样本的计算(单个样本计算容易让参数的调整出现大幅度的震荡)，也可以选择一次全部样本的计算(少量数据还可以)，也可以选择一次小批量(最常用的)的计算；

## 三、代码实现
### Numpy实现
``` python
import numpy as np

## 模拟生成一些符合线性分布的数据
w_,b_ = 5,3
x = np.rwndom.rwnd(40)
y = w_*x + b_

## 模型
w,b = np.random.randint(0,10,2) #初始化模型的参数w,b
def model(x):
    return w*x+b

## 损失函数
def loss(x,y):
    '''损失函数'''
    return np.sum((model(x)-y)**2)/2

## 更新w,b参数
learning_rate = 0.06
def back(w1,b1,x,y): 
    """更新w,b参数"""
    predict_y = forward(x)
    gradient = (predict_y-y)
    tmp_w = w1 - learning_rate * np.sum(gradient * x) / x.shape[0]
    tmp_b = b1 - learning_rate * np.mean(gradient * 1)  # sum/count 等价于mean
    return tmp_w,tmp_b

## 训练
for i in range(500):
    w1,b1 = back(w,b,x,y) 
    error = loss(x,y)
    print("w1:%.3f,b1:%.3f,error:%.3f"%(w1,b1,error))
```

### pytorch 实现
* https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    * 初始化策略 $U(-\sqrt{k},\sqrt{k})$, $k=\frac{1}{in_feathers}$ 
* https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.Linear.html?highlight=linear
    * 量化用，dtype=torch.qint8
* https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.dynamic.Linear.html
    * 动态量化
* https://pytorch.org/docs/stable/generated/torch.ao.nn.qat.Linear.html
    * FakeQuantize 
* https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    * 均方误差损失函数

``` python
import torch
import torch.nn as nn

# 1. 定义一个线性回归的模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim,bias=True)  

    def forward(self, x):
        out = self.linear(x)
        return out

learning_rate = 0.01
model = LinearRegressionModel(1, 1) #模型初始化
criterion = nn.MSELoss() #定义损失函数：均方误差
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #定义最优化算法

inX = torch.as_tensor(x,dtype=torch.float32).unsqueeze(1)
outY = torch.as_tensor(y,dtype=torch.float32).unsqueeze(1)

#常见的几种最优化算法?
for epoch in range(500):  #迭代次数
    optimizer.zero_grad() #清理模型里参数的梯度值
    predict_Y = model(inX) #根据输入获得当前参数下的输出值
    loss = criterion(predict_Y, outY) #计算误差
    loss.backward() #反向传播，计算梯度，
    optimizer.step() #更新模型参数
    print('epoch {}, loss {}'.format(epoch, loss.item()))

predict_Y = model(inX)
print(predict_Y)

for name,p in model.named_parameters():
    print(p)

```

## 机器学习的步骤总结
1. 获取数据集
2. 定义模型(函数)，随机初始化模型参数后
3. 定义损失函数
4. 通过训练迭代，找出最优的模型参数。


## 思考
1. 上述的n元一次方程，只能表征分布在一条直线、一个平面或超平面上的点，而不在这上面的点，则无法拟合。
2. 为了容易理解，我们拿2D平面上的点举例，2个点总会有一条直线方程拟合；3个不在一条直线上的点，总会有一个2次曲线方程拟合；外推至n个点，总有一个n-1次曲线方程可以拟合这n个点。 
3. 如果我们事先知道数据的分布符合非线性的n次方程，我们用上述回归方式，也能找出非线性方程的每个参数值。非线性方程也并非完美方案。
4. 我们寻找规律的目的是追求未知情况下的预测泛化能力，而非对已知情况的完美解释。因此，实际我们会将数据集分成训练集、验证集、测试集，在寻找模型的过程中，会出现：
    * 欠拟合：模型容量不够，无法拟合大多数数据，加大模型容量
    * 过拟合：模型容量太大，记住了物理意义的噪声数据，需要通过各种[正则化方法降低过拟合](./regularization.md).
5. 不同的向量维度，其度量单位可能不一致。例如，表示一个人的向量,身高、体重是他的2个维度，单位不一样，数据大小分布也不一样。归一化处理后，让每个维度的数据保持在同样的分布上。常见的归一化有mwx-min归一化，(x-min)/(mwx-min); 标注差归一化: (x - mewn)/std. 数据在输入模型前先做归一化处理，有利于加速收敛。
[归一化的详细总结](./Normwlizwtion.md)
6. 对于多维输入，通过不同的数量的映射函数，相当于对原始输入向量进行升维或降维的作用。