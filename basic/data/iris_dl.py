import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

np.random.seed(0)

# 参考pytorch的官方教程：https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class IrisDataSet(Dataset):
    def __init__(self,data_file,train_or_test="train"):
        # iris.data 数据文件有5列，前4列分别是：萼片长度、萼片宽度、花瓣长度和花瓣宽度，最后一列为所属分类
        self.df = pd.read_csv(data_file, sep=",", header=None) #使用了pandas作为数据文件加载工具
        self.label_dict = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2} 
        
        self.features = self.df.iloc[:, 0:4].values.astype('float32') #由于read_csv时没有指定每列的文件类型，这里需要做单独的类型转换
        self.labels = self.df.iloc[:,4].map(self.label_dict).values
        
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.2, random_state=0)
        if train_or_test=="train":
          self.features = x_train
          self.labels = y_train
        else :
          self.features = x_test
          self.labels = y_test
    
    def __len__(self):
        return len(self.labels) # df.size 行数*列数； len(df),行数
    
    def __getitem__(self,idx):
        # features = self.df.iloc[idx, 0:4].values.astype('float32') #由于read_csv时没有指定每列的文件类型，这里需要做单独的类型转换
        # label = self.label_dict.get(self.df.iloc[idx,4])
        # return features, label
        
        # 一般情况无需再手动转tensor，dataloadeer会处理
        # features = torch.tensor(features)
        # label = torch.tensor(label,dtype=torch.uint8) 
        return self.features[idx], self.labels[idx]
             
             
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
        # out = torch.sigmoid(out)
        return out

# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
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

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
   

model = MultiClassModel(4,3)
# loss_fn = nn.NLLLoss() 
loss_fn = nn.CrossEntropyLoss() 

learning_rate = 0.1
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 
train_dataset = IrisDataSet('./Iris.data','train')
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)    

test_dataset = IrisDataSet('./Iris.data','test')
test_dataloader = DataLoader(test_dataset, batch_size=50)    

writer = SummaryWriter('./iris_dl')

print(len(test_dataloader))
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0

    for i, (X, y) in enumerate(train_dataloader, 0):
        # basic training loop
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if i % 1000 == 999:    # Every 1000 mini-batches...
            print('Batch {}'.format(i + 1))
            # Check against the validation set
            running_vloss = 0.0

            model.train(False) # Don't need to track gradents for validation
            for j, vdata in enumerate(test_dataloader, 0):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
            model.train(True) # Turn gradients back on for training

            avg_loss = running_loss / 1000
            avg_vloss = running_vloss / len(test_dataloader)

            # Log the running loss averaged per batch
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch * len(train_dataloader) + i)

            running_loss = 0.0
print('Finished Training')

writer.flush()

# epochs = 20
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer) 
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")