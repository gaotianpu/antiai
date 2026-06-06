import numpy as np
import matplotlib.pyplot as plt

#为了让重复执行时结果保持不变，需要指定个随机数，后续生成随机数相关的操作就会保持一致
np.random.seed(0) 

## 模拟生成一些符合线性分布的数据
w_true,b_true = 5,3

x = np.random.rand(40)
y = w_true* x + b_true

# plt.plot(x, y, 'o', color='green')
# plt.show()

# 定义模型
w,b = np.random.randint(0,10,2) #初始化模型的参数w,b

## 定义模型
def model(x):
    return w*x+b

# 损失函数
def l1_loss(y_predict,y_true):
    ''' 绝对值损失 '''
    return np.mean(np.abs(y_predict-y_true))

def mse_loss(y_predict,y_true): # mean squared error
    '''mean squared error'''
    return np.mean((y_predict-y)**2)


y_predict = model(x)
print("l1_loss:",l1_loss(y_predict,y))
print("mse_loss:",mse_loss(y_predict,y))

## 更新w,b参数
def update_wb(x,y_true,predict_y,learning_rate = 0.6): 
    """更新w,b参数""" 
    gradient = predict_y - y_true 
    # gradient = y_true - predict_y  
    tmp_w = w - learning_rate * np.mean(gradient * x)
    tmp_b = b - learning_rate * np.mean(gradient * 1)
    return tmp_w,tmp_b

## 训练
for i in range(100):
    predict_y = model(x)
    error = mse_loss(predict_y,y)
    w,b = update_wb(x,y,predict_y) 
    print("idx:%d,w:%.3f,b:%.3f,error:%.3f"%(i,w,b,error))

print(x[0],y[0])

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

learning_rate = 0.5
model = LinearRegressionModel(1, 1) #模型初始化
criterion = nn.MSELoss() #定义损失函数：均方误差
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #定义最优化算法

inX = torch.as_tensor(x,dtype=torch.float32).unsqueeze(1)
outY = torch.as_tensor(y,dtype=torch.float32).unsqueeze(1)

print(inX.shape,outY.shape)

#常见的几种最优化算法?
for epoch in range(100):  #迭代次数
    predict_Y = model(inX) #根据输入获得当前参数下的输出值
    loss = criterion(predict_Y, outY) #计算误差
    
    loss.backward() #反向传播，计算梯度，
    optimizer.step() #更新模型参数
    optimizer.zero_grad() #清理模型里参数的梯度值
    
    print('epoch {}, loss {}'.format(epoch, loss.item()))

predict_Y = model(inX)
print(predict_Y)

for name,p in model.named_parameters():
    print(p)