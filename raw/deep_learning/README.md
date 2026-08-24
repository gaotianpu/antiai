# 深度学习笔记

## 基础部分
### 一、现象和规律
物体垂直下落，用告诉摄影机拍摄，通过观察我们可以记录每个时间点上， 物体距离下落点的位置；向斜上方抛出篮球，同样用告诉摄影机拍摄，篮球的运动轨迹会是一个抛物线。
通过观察，物体在每个时间点的位置变化是现象，根据现象如何找到规律呢？如果知道了规律，会有什么作用呢，预测下一个时间点，物体的位置。如果要掌握现象背后的精确规律，就需要使用数学的工具。

现象->样本? 

现象，怎么用数学表示呢？

### 二、样本与空间  [数据的组织方式：标量scalar，向量vector，矩阵matrix，张量tensor](./1_scalar_vector_matrix_tensor.ipynb)
1. 标量 scalar
可以看做是一维直线上的点；身高、体重、年龄，学习成绩等等；一条直线，中间是0，讨论身高，张三1.70，在这个位置，李四1.82在这个位置；单一维度上的度量。 量化的过程，将一切形容词都可以用一个具体的数值表示，想一想，哪些还没办法量化？

2. 向量 vector
具备2+个维度空间上的点；平面2D，2维向量；空间，3D向量；更高的维度。对于一个物体的描述，通常是多维度的，例如一个人，身高、体重、年龄，学习成绩等多个维度，怎样表示这个人的这么多属性呢；引入了向量 先假设一个人有2个属性，身高、体重；在一个平面坐标系中，x表示身高、y表示体重，平面上的一个点就能代表某个人。如果有3个属性，则3d空间上的点可以表示， 当然，会有更高的维度，超过3d的，已经不能在我们的物理世界中找到直观的对应了，向量引入(a1,a2,a3,a4....an)。
人：
商品：
视频：
文章：
* 不同维度单位不一致，归一化(Normalization)解决.

3. 矩阵 matrix
同一个空间中向量的集合, 多个向量，空间中的多个点，每列代表向量的每个属性，每行代表不同的向量，叫做矩阵，二维的表格.

4. 张量tensor，多个矩阵组合在一起，叫张量；例如一张图片，rgb三个通道组成，每个通道又有长宽表格组成，每个单元格对应的像素值；

### 三、寻找样本的分布规律，[线性回归 Linear regression](./1_linear_regression.ipynb)
* 物体匀速运动，某个时刻距离触发点的位置，平面坐标系，x-t,y-d, 当然平面上的点，也可以看做二维向量(t,d)
* 平面空间中的一些点，按照某种规律分布，如何找到这个规律；
* 规律本身，y_predict = aX + bias 线性函数
* 随机确定a,b一个值，已知x，会生成一个预测的y点，预测的y点和实际y点的差值越小越好，均方误差越小越好 
y_truth - y_predict -> y_truth - (aX + b)  接近0，最好; 绝对值的形式； 
* 评价指标,损失函数: $(y_truth - aX - b )^2$ , y_truth, X为已知数，a,b为何值时，均方误差最小？
* 梯度求导，随机梯度下降，批量梯度下降等
* 梯度下降算法：拆开讲，没有b的情况，一元二次方程，y_truth - aX=0 最小， a = y_truth / X, 多个点取个均值；
有bias的情况，二元二次方程，偏导数，差值变大了还是变小了，决定增大或减少。


### 四、样本分类 [逻辑回归 Logistic regression](./1_softmax_cross_entropy.ipynb) 
* softmax，sigmod，交叉熵
* 和线性已知x，求y方式不同，空间上的点分布在2个不同的位置，需要找到一个分割线，将一个平面分割成2个部分；
* 空间中有两类不同的点，分布位置如图，画一条线，尽可能让2类点分布在这条线的两侧；
* 给定x点，分割线上y_split = aX + bias, y_truth > y_split 一类, 预测的 label_predic = 1， y_truth < y_split ,label_predict=0, 
label_precit = y_truth - (aX + bias) {>=0 则为1; <0则为0}
再根据实际的label值比对， label_truth - label_precit -> label_truth - y_truth + (aX + bias)  值越小越好.

y_split = aX + bias;

y_true - y_split，数值范围在正负无穷之间， label_true 只有 0,1 两个值，正负无穷映射到0,1之间？ sigmod $f(x)=1/(1+e^{-x})$

label_true - sigmod( y_true -  aX - bias)  值最小

label_predict = sigmod( a*x + b*y + bias ) ?  升维容易理解，x,y 是个平面， z轴 1, 0

$Loss_i = - (y_i * log^ {p_i} + (1 - y_i) * log^ {1-p_i} ) $

y_i, 真实标签，0 或 1
p_i, 预测值，取值范围 [0,1]


### 五、更复杂的样本分布规律 [深度网络: 激活函数--从线性到非线性](./1_deep_learing.ipynb) 
* 复杂的样本分布举例子
* 2个点一次方程，3个点二次方程，n个点n-1次方程；n元n次方程的拟合 (平面上3个、4个、5个、n个不在同一直线上的点，必定有一个n-1次方程可以完全覆盖) Yann Le Cun的《科学之路》少量样本没问题，但太复杂，与真实的规律不符合，过拟合。
* 引入非线性函数：将复合线性函数，还是线性函数，引入非线性函数。 MLP
* 多层的反向传播能力
* 实际应用：波士顿房价预测，[iris多分类问题](./1_iris.ipynb)
* 特征处理：归一化

### 六、[排序问题: point-wise,pair-wise,list-wise](./1_rankdnn.ipynb) 
[NDCG](https://www.cnblogs.com/by-dream/p/9403984.html)


## 自然语言处理 NLP
### 一、自然语言的向量表示
1. one-hot,词袋 Bag-of-words(BoW); [自然语言的向量表示: one-hot,词袋,词嵌入word embedding](./3_nlp.ipynb)
2. word-embedding: n-gram,Skip-gram,CBOW(Continuous Bag of Words Model)
3. 合适的token粒度,在存储空间和表示能力之间做平衡. unknown问题，字节，BPE

### 二、循环神经网络
1. RNN 文本分类 [循环神经网络：RNN,LSTM,GRU](./3_rnn_lstm_gru.ipynb)
2. LSTM
3. GRU
4. ELMo
5. 应用：语义相关性 [应用：语义相关性](./3_semantic_relevance.ipynb)
6. 应用：中文分词，词性标注，实体识别  [应用：中文分词，词性标注，实体识别](./3_crf.ipynb)
7. 应用：翻译、 [应用：翻译](/3_seq2seq.ipynb)
8. 摘要提取，问答

### 三、Transformer 
1. [注意力机制和Transformer](./3_transformer.ipynb)

### 四、掩码语言模型
1. [BERT: Bidirectional Encoder Representation from Transformers](./3_bert.ipynb) 预训练模型
2. BERT变种
3. 下游任务

### 五、自回归(生成式)语言模型
1. GPT-1,2,3
2. InstructGPT

### 六、稀疏混合专家模型


## 图像处理
### 一、图像的向量化表示、数据增广

### 二、[卷积网络](./2_cnn.ipynb) 
* [目标检测:YOLO v3,v4,v5](./2_yolo.ipynb)
* [图像语义分割](./2_image_semantic_segmentation.ipynb)
* [OCR](./2_ocr.ipynb)

### 三、ViT
1. ViT
2. MAE
3. ViTDet

## 多模
1. 多模融合：图像、文字、音频、视频 -> 知识图谱,人类的常识？


## 其他-待整理
### [工业化部署](./deploy.ipynb)
1. 量化、剪枝、蒸馏

### 数据处理
1. [主动学习: 筛选有价值的未标注样本送标](./4_active_learning.ipynb)


### 其他
1. 生成对抗网络
    * DCGan
1. 图神经网络，半结构化数据
1. 3D建模

### 一般应用
* [孪生网络: 少样本学习](./2_siamese.ipynb)
* 相似度

### 环境和工具
* conda
* python / c++
* numpy
* pandas
* scikit-learn
* matplotlib
* pytorch
* paddlepaddle