import pandas as pd
import torch
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
    
# 单条记录
features, label = next(iter(dataset))
print(features, label)

#小批量加载
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
features, label = next(iter(dataloader))
print(features, label)