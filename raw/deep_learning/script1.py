import numpy as np
import torchtext
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchtext.vocab import vocab
from collections import Counter, OrderedDict

# def test():
#     scalar = 1.2
#     vector = np.array([1.0,2.1,3.2],dtype=np.float)
#     print(vector.shape)
#     matrix = np.array([[1,2,3],[4,5,6]]) 
#     print(matrix.shape)
#     tensor = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
#     print(tensor.shape)

def softmax(x,axis=-1):
    exp_x = np.exp(x-np.max(x,axis=axis, keepdims=True))
    sum_exp_x = np.sum(exp_x,axis=axis, keepdims=True)
    y = exp_x / sum_exp_x
    return y 

# x = np.array([[1,2,3,4],[5,2,3,0]])
# ret = softmax(x)
# print(x.shape,ret.shape)
# print(ret) 
# print("~~~")

# from sklearn.metrics import log_loss 
# y_true = [[0, 0, 1], [0, 1, 0], [1, 0, 0]] 
# y_pred_1 = [[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]] 
# print("log_loss:",log_loss(y_true, y_pred_1))

# def cross_entropy(pred_y,real_y):
#     m = real_y.shape[0]
#     print("m:",m)
#     p = softmax(pred_y)
#     print("p:",p)
#     log_likehood = - np.log(p[range(m),real_y])
#     print("p[range(m),real_y]:",p[range(m),real_y])
#     print("log_likehood:",log_likehood)
#     loss = np.sum(log_likehood) / m 
#     print("loss",loss)
#     return loss 

# def test():
#     txt = "定风波·莫听穿林打叶声 三月七日，沙湖道中遇雨，雨具先去，同行皆狼狈，余独不觉。已而遂晴，故作此(词)。莫听穿林打叶声，何妨吟啸且徐行。竹杖芒鞋轻胜马，谁怕？一蓑烟雨任平生。料峭春风吹酒醒，微冷，山头斜照却相迎。回首向来萧瑟处，归去，也无风雨也无晴。"

#     unk_token = '<unk>'
#     default_index = -1
#     v2 = vocab(OrderedDict([(token, 1) for token in txt]), specials=[unk_token])
#     print(v2.__version__)
#     v2.set_default_index(default_index)
#     print(v2['<unk>']) #prints 0
#     print(v2['out of vocab']) #prints -1
#     #make default index same as index of unk_token
#     v2.set_default_index(v2[unk_token])
#     v2['out of vocab'] is v2[unk_token] #prints True

# # v2 = vocab(OrderedDict([(token, 1) for token in "hello,word"]))
# print(torchtext.__version__)
# test()
# print(cross_entropy(np.array([[1,2,3,4]]),np.array([[0,0,0,1]])))


print(softmax(np.array([[2.470296621322632,
        -3.4948902130126953,
        -3.187654972076416,
        3.8116281032562256]])))

print(softmax(np.array([[1.22227144241333,
        -3.4080684185028076,
        -3.168757915496826,
        4.585843086242676]])))


rnn = nn.RNN(10, 20, 2)