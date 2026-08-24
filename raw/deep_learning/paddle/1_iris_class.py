#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.static import InputSpec
import pandas as pd
from visualdl import LogWriter


# 1. 定义模型架构
class IrisModel(paddle.nn.Layer):
    def __init__(self, input_dim, output_dim):
        super(IrisModel, self).__init__()
        self.fc_0 = nn.Linear(input_dim, 25)
        self.relu_0 = paddle.nn.ReLU()
        self.fc_1 = nn.Linear(25, 30)
        self.relu_1 = paddle.nn.ReLU()
        self.fc_2 = nn.Linear(30, output_dim)

    def forward(self, x):
        out = self.relu_0(self.fc_0(x))
        out = self.relu_1(self.fc_1(out))
        out = self.fc_2(out)
        return out


# 模型测试
model = IrisModel(4, 3)
# test_x = paddle.rand(shape=[4, 4])
# print(test_x)
# predict_y = model(test_x)
# print(predict_y)

# 2，定义数据集
class IrisDataset(Dataset):
    def __init__(self, data_type="train"):
        assert data_type in ('train', 'test')
        self.labels = {'Iris-setosa': 0,
                       'Iris-versicolor': 1, 'Iris-virginica': 2}

        data_file = "../DeepLearning/dataset/iris/%s.csv" % (data_type)
        self.items = []
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                fields = line.strip().split(",")
                d = fields[:4]
                d.append(self.labels[fields[-1]])
                self.items.append(d)

        # self.data_file = open("./dataset/iris/%s.csv" % (data_type),'r')
        # self.pd_frame = pd.read_csv("./dataset/iris/%s.csv" % (data_type))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # with open(self.data_file,'r') as f:
        #     for i,line in enumerate(f):
        #         if i == idx:
        #             fields = line.strip().split(",")
        #             return np.array(fields[:4]),self.labels[fields[-1]]
        # label = self.pd_frame.iloc[idx, 4]
        # X = self.pd_frame.iloc[idx, 0:4]
        # print("inner:",X.to_numpy(np.float32), self.labels[label])
        return np.array(self.items[idx][:4], dtype=np.float32), self.items[idx][-1]


# 数据集测试
custom_dataset = IrisDataset("train")
# print('=============custom dataset=============')
for data, label in custom_dataset:
    print(data.shape, label)
    print(data, label)
    break

BATCH_SIZE = 4
train_loader = DataLoader(
    IrisDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    IrisDataset("test"), batch_size=BATCH_SIZE, shuffle=True)
# # 如果要加载内置数据集，将 custom_dataset 换为 train_dataset 即可
# for batch_id, data in enumerate(train_loader()):
#     x_data = data[0]
#     y_data = data[1]

#     print(x_data.shape)
#     print(y_data.shape)
#     break

#3. 定义训练函数
def train(epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # 训练模式
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.clear_grad()  # 梯度清零

        # print(pred.shape, y.shape,paddle.unsqueeze(y, axis=1).shape)
        # print(pred)
        # 计算准确率 等价于 prepare 中metrics的设置
        acc = paddle.metric.accuracy(pred, paddle.unsqueeze(y, axis=1))

        # if writer:
        #     writer.add_scalar(tag="train_acc", step=epoch*size+batch, value=acc.numpy())

        if (batch+1) % 5 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(
                epoch, batch+1, loss.numpy(), acc.numpy()))

    #     if batch % 5 == 0:
    #         loss, current = loss.item(), batch * len(X)
    #         print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#5. 定义test函数
def test(epoch, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.eval()  # 预测模式
    test_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        acc = paddle.metric.accuracy(pred, paddle.unsqueeze(y, axis=1))
        # if writer:
        #     writer.add_scalar(tag="test_acc", step=epoch*size+batch, value=acc.numpy())

    print("test: loss is: {}, acc is: {}".format(loss.numpy(), acc.numpy()))


# 设置优化器
optim = paddle.optimizer.Adam(parameters=model.parameters())
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()

# 6. 启动训练
epochs = 10 #设置迭代次数
# with LogWriter(logdir="./log/scalar_test/train") as writer:
writer = None
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(t, train_loader, model, loss_fn, optim)
    test(t, test_loader, model, loss_fn)


# 1.对于训练调优场景，我们推荐使用paddle.save/load保存和载入模型；
# 2.对于推理部署场景，我们推荐使用paddle.jit.save/load（动态图）和 paddle.static.save/load_inference_model（静态图）保存载入模型。

# 一、训练调优场景的模型&参数保存载入
# 动态图的保存和加载
paddle.save(model.state_dict(), "output/model.pdparams")
paddle.save(optim.state_dict(), "output/optim.pdopt")
# model.set_state_dict(paddle.load("output/model.pdparams"))
# optim.set_state_dict(paddle.load("output/optim.pdopt"))

# 二、训练部署场景的模型&参数保存载入
# 动转静训练 + 模型&参数保存
# 动态图训练 + 模型&参数保存
path = "output/model.jit"
paddle.jit.save(
    layer=model,
    path=path,
    input_spec=[InputSpec(shape=[None, 4], dtype='float32')])
model_1 = paddle.jit.load(path)
test_x = paddle.rand(shape=[4, 4])
model_1.eval()
predict_y = model_1(test_x)
print(predict_y)

# save_path = 'output/model.onnx'
# x_spec = InputSpec([None, 4], 'float32', 'x')
# paddle.onnx.export(model, save_path, input_spec=[x_spec])
