import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)

def generate_data():
    data_count = 50
    w,b = 5,3
    x1 = np.random.rand(data_count)
    y1 = w*x1 + b + np.random.rand(data_count)*5

    x2 = np.random.rand(data_count)
    y2 = w*x2 + b - np.random.rand(data_count)*5

    # 绘制散点图
    plt.plot(x1, y1, '*', color='green')
    plt.plot(x2, y2, 'x', color='blue')
    # plt.show()

    def generate_data(x,y,data_count,labels=0):
        label = np.ones_like(y) if labels==1 else np.zeros_like(y) 
        return np.concatenate((label,np.reshape(x,(-1, 1)),y),axis=1)

    #生成数据，第一列为label，值0,1, 第二列x0, 第三列x1
    data_0 = generate_data(x1,y1.reshape((-1, 1)),data_count,0)
    data_1 = generate_data(x2,y2.reshape((-1, 1)),data_count,1)
    data_all = np.concatenate((data_0,data_1),axis=0)
    np.random.shuffle(data_all)

    labels = data_all[:,0:1]
    features = data_all[:,1:3]
    return labels,features


labels,features = generate_data()
print(labels.shape,features.shape)


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

class BinClassModel(nn.Module):
    def __init__(self, input_dim):
        super(BinClassModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)  

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

class MultiClassModel(nn.Module):
    def __init__(self, input_dim, class_count):
        super(MultiClassModel, self).__init__()
        self.linear = nn.Linear(input_dim, class_count,bias=True)  
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.linear(x)
        # out = self.softmax(out)
        return out

inX = torch.tensor(features, dtype=torch.float32) 
outY = torch.tensor(labels, dtype=torch.long) 

outY = outY.squeeze() 
print(inX.shape,outY.shape)

def train_loop(model,criterion,optimizer,inX,outY,epoch=50):
    for epoch in range(epoch):  #迭代次数
        predict_Y = model(inX) #根据输入获得当前参数下的输出值
        loss = criterion(predict_Y, outY) #计算误差
        
        loss.backward() #反向传播，计算梯度，
        optimizer.step() #更新模型参数
        optimizer.zero_grad() #清理模型里参数的梯度值
        
        print('epoch {}, loss {}'.format(epoch, loss.item()))

def run_Linear():
    learning_rate = 0.005
    model = LinearRegressionModel(2, 1) #模型初始化
    criterion = nn.MSELoss() #定义损失函数：均方误差
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #定义最优化算法

    train_loop(model,criterion,optimizer,inX,outY,1500)

    # z = 0.7605 * x + -0.1531 * y + 0.9337
    for name,p in model.named_parameters():
        print(p)
    
    delta = 1e-7
    pw = -model.state_dict()["linear.weight"][0][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    pb = -model.state_dict()["linear.bias"][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    print("weight:",pw,pb)
    plt.plot([0,1],[pb,pw+pb])
    plt.show()

def run_BinClass_with_MSE():
    learning_rate = 0.15
    model = BinClassModel(2) #模型初始化
    criterion = nn.MSELoss() #定义损失函数：均方误差
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #定义最优化算法

    train_loop(model,criterion,optimizer,inX,outY,1000)

    # z = 0.2264 * x - 0.2670 * y + 0.9261
    for name,p in model.named_parameters():
        print(p)
    
    delta = 1e-7
    pw = -model.state_dict()["linear.weight"][0][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    pb = -model.state_dict()["linear.bias"][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    print("weight:",pw,pb)
    plt.plot([0,1],[pb,pw+pb])
    plt.show()

def run_BinClass():
    learning_rate = 0.15
    model = BinClassModel(2) #模型初始化
    criterion = nn.BCELoss() #定义损失函数：均方误差
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #定义最优化算法

    train_loop(model,criterion,optimizer,inX,outY,1000)

    # z = 0.2264 * x - 0.2670 * y + 0.9261
    for name,p in model.named_parameters():
        print(p)
    
    delta = 1e-7
    pw = -model.state_dict()["linear.weight"][0][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    pb = -model.state_dict()["linear.bias"][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    print("weight:",pw,pb)
    plt.plot([0,1],[pb,pw+pb])
    plt.show()

def run_MultiClass():
    learning_rate = 0.08
    model = MultiClassModel(2,2) #模型初始化
    criterion = nn.NLLLoss() # 模型输出加了 logsoftmax
    # criterion = nn.CrossEntropyLoss() # 模型输出 未加 logsoftmax
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #定义最优化算法

    inX = torch.tensor(features, dtype=torch.float32) 
    outY = torch.tensor(labels, dtype=torch.long) #1D long型
    outY = outY.squeeze() 
    print(inX.shape,outY.shape)

    train_loop(model,criterion,optimizer,inX,outY,1000)
     
    for name,p in model.named_parameters():
        print(p) 
    
    # 绘出2条分割线    
    delta = 1e-7
    pw = -model.state_dict()["linear.weight"][0][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    pb = -model.state_dict()["linear.bias"][0].item()/(model.state_dict()["linear.weight"][0][1].item()+delta)
    print("weight:",pw,pb)
    plt.plot([0,1],[pb,pw+pb], color='blue')
    
    pw1 = -model.state_dict()["linear.weight"][1][0].item()/(model.state_dict()["linear.weight"][1][1].item()+delta)
    pb1 = -model.state_dict()["linear.bias"][1].item()/(model.state_dict()["linear.weight"][1][1].item()+delta)
    print("weight:",pw1,pb1)
    plt.plot([0,1],[pb1,pw1+pb1], color='green')
    
    plt.show()

# run_Linear()
# run_BinClass_with_MSE()
# run_BinClass()
run_MultiClass()
