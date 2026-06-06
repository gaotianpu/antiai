import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 参考pytorch的官方教程：https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class IrisDataSet(Dataset):
    def __init__(self,data_file):
        # iris.data 数据文件有5列，前4列分别是：萼片长度、萼片宽度、花瓣长度和花瓣宽度，最后一列为所属分类
        self.df = pd.read_csv(data_file, sep=",", header=None) #使用了pandas作为数据文件加载工具
        self.label_dict = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2} 
    
    def __len__(self):
        return len(self.df) # df.size 行数*列数； len(df),行数
    
    def __getitem__(self,idx):
        features = self.df.iloc[idx, 0:4].values.astype('float32') #由于read_csv时没有指定每列的文件类型，这里需要做单独的类型转换
        label = self.label_dict.get(self.df.iloc[idx,4])
        
        # 一般情况无需再手动转tensor，dataloadeer会处理
        # features = torch.tensor(features)
        # label = torch.tensor(label,dtype=torch.uint8) 
        
        return features, label

dataset = IrisDataSet('./Iris.data')
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
   
# 单条记录
# features, label = next(iter(dataset))
# print(features, label)
# print(features.shape, label.shape)

# batch
# features, label = next(iter(dataloader))
# print(features.shape, label.shape)


class MultiClassModel(nn.Module):
    def __init__(self, input_dim, class_count):
        super(MultiClassModel, self).__init__()
        hidden_dim = 2
        self.fc1 = nn.Linear(input_dim, hidden_dim,bias=True)  
        self.fc2 = nn.Linear(hidden_dim, class_count,bias=True)  
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = torch.sigmoid(self.fc1(x))  
        out = self.fc2(out) 
        # out = self.softmax(out)
        return out

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

learning_rate = 0.1
model = MultiClassModel(4,3)
# loss_fn = nn.NLLLoss()  #模型最后一层带 softmax
loss_fn = nn.CrossEntropyLoss() # 不带softmax
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader, model, loss_fn, optimizer) 
print("Done!")

# 0.353390