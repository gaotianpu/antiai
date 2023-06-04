import numpy as np
import matplotlib.pyplot as plt

plt.plot([0,1], [1,0], '*', color='green')
plt.plot([0,1],[0,1], 'x', color='blue')
# plt.show()


import torch 
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)
    
# 创建一个Xor的模型
class XOrModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(XOrModel, self).__init__()
        hidden_dim = 2
        self.fc1 = nn.Linear(input_dim, hidden_dim,bias=True)  
        self.fc2 = nn.Linear(hidden_dim, output_dim,bias=True)  

    def forward(self, x):
        out = torch.sigmoid(self.fc1(x))  
        out = self.fc2(out) 
        out = torch.sigmoid(out)
        return out 

    def output(self,x):
        out = torch.sigmoid(self.fc1(x)) 
        print("first:",x,out)
        
        out1 = self.fc2(out) 
        out1 = torch.sigmoid(out1)  
        print("second:",out1)
        return out,out1 

model = XOrModel(2, 1) #模型初始化
criterion = nn.BCELoss() #定义损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9) #定义优化算法
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.5) 

features = torch.Tensor([[0,1],[1,0],[0,0],[1,1]])
labels = torch.Tensor([0,0,1,1]).unsqueeze(1)
print(labels.shape) #特别注意labels.shpe是[4,1], 而非[4]

# 开始训练
for epoch in range(550):  #迭代次数  
    predict_Y = model(features) #根据输入获得当前参数下的输出值
    loss = criterion(predict_Y, labels) #计算误差
    
    loss.backward() #反向传播，计算梯度，
    optimizer.step() #更新模型参数
    optimizer.zero_grad() #清理模型里参数的梯度值
    # if epoch % 50 ==0:
    print('epoch {}, loss {}'.format(epoch, loss.item()))


# 第一层的两个线性函数，用于将原始的向量转为新的向量，
model_params = list(model.parameters())
print(model_params)

model_weights = model_params[0].data.numpy()
model_bias = model_params[1].data.numpy()

# 第一层的两个维度的投影函数
x_1 = np.arange(-0.1, 1.1, 0.1)
y_1 = ((x_1 * model_weights[0,0]) + model_bias[0]) / (-model_weights[0,1])
plt.plot(x_1, y_1, color='green')

x_11 = np.arange(-0.1, 1.1, 0.1)
y_11 = ((x_11 * model_weights[1,0]) + model_bias[1]) / (-model_weights[1,1])
plt.plot(x_11, y_11, color='green')

# no_grad设置，冻结模型的参数，不再变化
with torch.no_grad():
    # 经第一层线性函数+sigmod后，将原始输入投影到新位置
    out = model.output(features)
    tmp = out[0].detach().numpy()
    plt.plot(tmp[0],tmp[1], '*', color='black')
    plt.plot(tmp[2],tmp[3], 'x', color='orange')

# 基于投影后的位置，执行分割
model_weights1 = model_params[2].data.numpy()
model_bias1 = model_params[3].data.numpy()

x_2 = np.arange(-0.1, 1.1, 0.1)
y_2 = ((x_2 * model_weights1[0,0]) + model_bias1[0]) / (-model_weights1[0,1])
plt.plot(x_2, y_2, color='red')
plt.show()