#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.static import InputSpec
from visualdl import LogWriter
from paddle.jit import to_static

# #paddle2.x使用静态图
# paddle.enable_static()

# 1. 定义模型
class SiameseModel(nn.Layer):
    def __init__(self, input_dim, output_dim=8):
        super(SiameseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 25),
            nn.ReLU(),
            nn.Linear(25, 30),
            nn.ReLU(),
            nn.Linear(30, output_dim)
        )

    @to_static
    def forward(self, input1, input2):
        v1 = self.model(input1)
        v2 = self.model(input2)
        return v1, v2

    def predict(self, input0):
        # 实际预测的时候，只需要输入一个，得到得分即可
        return self.model(input0)


# 模型初始化
input_dim = 4
output_dim = 2
model = SiameseModel(input_dim, output_dim)

# # 模型简单测试
batch_size = 2
input1 = paddle.rand(shape=[batch_size, input_dim])
input2 = paddle.rand(shape=[batch_size, input_dim])
ret = model(input1, input2)
# print(ret)
# input0 = paddle.rand(shape=[1,input_dim])
# ret0 = model.predict(input0)
# print(ret0)


# 2. 自定义损失函数
class ContrastiveLoss(nn.Layer):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        """两个向量直接减, 实际测试loss可收敛"""
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        target = paddle.cast(target, dtype='float64')
        losses = 0.5 * (target * distances +
                        (1 + -1 * target) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

#损失函数测试
criterion = ContrastiveLoss()
# label = paddle.rand(shape=[batch_size,1])
# ret = criterion(input1,input2,label)
# print(ret)


class IrisDataset(Dataset):
    def __init__(self, data_type="train"):
        assert data_type in ('train', 'test')
        self.labels = {'Iris-setosa': 0,
                       'Iris-versicolor': 1, 'Iris-virginica': 2}
        self.pd_frame = pd.read_csv(
            "../DeepLearning/dataset/iris/%s.csv" % (data_type), header=None)
        self.dataset = list(self.gen_pairs())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def gen_pairs(self):
        cnt = len(self.pd_frame)
        for i in range(cnt-1):
            i_label = self.pd_frame.iloc[i, 4]
            i_X = self.pd_frame.iloc[i, 0:4]
            for j in range(i+1, cnt):
                j_label = self.pd_frame.iloc[j, 4]
                j_X = self.pd_frame.iloc[j, 0:4]
                yield 1.0 if i_label == j_label else 0, i_X.to_numpy(np.float32), j_X.to_numpy(np.float32)


# 测试数据集
data = IrisDataset()
label, x1, x2 = next(iter(data))
# print(label,x1,x2)
train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
train_labels, train_features1, train_features2 = next(iter(train_dataloader))
# print(train_labels,train_features1,train_features2)
# print(train_labels.shape)
# print(train_features1.shape,train_features2.shape)

optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)


counter = []
loss_history = []
def train(epoch, dataloader, model, loss_fn, optimizer):
    model.train()  # 训练模式

    dataset_cnt = len(dataloader.dataset)
    batch_cnt = len(dataloader)
#     print(dataset_size,batch_size)

    for batch, (label, x1, x2) in enumerate(dataloader):
        output1, output2 = model(x1, x2)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward() #反向传播
        optimizer.step()
        optimizer.clear_grad()

        if batch % 2 == 0 or batch == (batch_cnt-1):
            print("Epoch:{} batch:{} loss:{} ".format(
                epoch, batch, loss_contrastive.item()))
            iteration_number = epoch*dataset_cnt + dataloader.batch_size * batch
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())


epochs = 30
for epoch in range(epochs):
    train(epoch, train_dataloader, model, criterion, optimizer)

# torch.save(model, 'siamese_module.pth')


# def show_plot(iteration, loss):
#     plt.plot(iteration, loss)
#     plt.show()
# show_plot(counter, loss_history)


class IrisPredictDataset(Dataset):
    def __init__(self,data_type="train"):
        assert data_type in ('train','test')
        self.labels = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
        self.pd_frame = pd.read_csv("../DeepLearning/dataset/iris/%s.csv" % (data_type),header=None)

    def __len__(self):
        return len(self.pd_frame)

    def __getitem__(self, idx):
        label = self.pd_frame.iloc[idx, 4]
        X = self.pd_frame.iloc[idx, 0:4]
        return X.to_numpy(np.float32),self.labels[label]

train1_dataloader = DataLoader(IrisPredictDataset("train"), batch_size=64)
test1_dataloader = DataLoader(IrisPredictDataset("test"), batch_size=64)
train_x,train_label = next(iter(train1_dataloader))
test_x,test_label = next(iter(test1_dataloader))
print(train_x.shape,train_label.shape)

# 输出向量结果
def compute_vector(model,dataloader):
    def compute():
        model.eval() 
        for x1,label in dataloader:
            ret = model.predict(x1)
            # for i,x in enumerate(x1):
            #     yield paddle.to_tensor(label[i]), paddle.to_tensor(x)
            for i,r in enumerate(ret):
                yield paddle.to_tensor(label[i]), r
    l = list(compute())
    labels = paddle.stack([label for label,vec in l],axis=0)
    vecs = paddle.stack([vec for label,vec in l],axis=0)
    return labels,vecs

train_label,train_vecotors = compute_vector(model,train1_dataloader)
test_label,test_vecotors = compute_vector(model,test1_dataloader)

def evaluate_1():
    # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/PairwiseDistance_cn.html#pairwisedistance
    dist = paddle.nn.PairwiseDistance()
    l_equal = []
    for i,test_vec in enumerate(test_vecotors):
        l = []
        for j,tain_vec in enumerate(train_vecotors): 
            distance = dist(paddle.unsqueeze(test_vec,axis=0), paddle.unsqueeze(tain_vec,axis=0))
            l.append(distance)
        min_idx = np.argmin(l)
        tr_label = train_label[min_idx].numpy()[0]
        te_label = test_label[i].numpy()[0]
        equal = 1 if tr_label==te_label else 0
        l_equal.append(equal) 
    print(sum(l_equal)/len(l_equal))
evaluate_1()

def evaluate():
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    matrix_dist = distance.cdist(test_vecotors, train_vecotors, 'euclidean') 
    min_idxs = paddle.to_tensor(matrix_dist.argmin(axis=1)) #距离最小的那个 
    most_close_sampe = paddle.index_select(train_label,min_idxs)  
    auc = (most_close_sampe==test_label).sum()/test_label.shape[0]
    print(auc.numpy()[0])  
evaluate()