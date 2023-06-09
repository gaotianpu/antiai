# 事物的向量化表示

合抱之木，生于毫末；九层之台，起于累土；千里之行，始于足下 - 老子。 

智能的本质，是能够从观察到的现象中总结规律，再用规律去预测现象。而机器学习(人工智能)的前提是，能够用数学的方式去描述现象、规模。

## 1. 描述现象
数学中有一些特定的术语，用于定量描述单个事物的单个特征、具有多个特征的事物、事物的集合等，分别是：标量 scalar、向量 vector、矩阵 matrix、张量 tensor。接下来具体说明这些概念。

### 1.1. 标量 scalar
人类的语言中会有一些形容词用于描述一个人或事物单个特征，例如人的身高，有形容词高、矮；形容词还会与某些副词结合，例如：很高、非常高等。这种没有精确数字的描述方式，称之为定性描述。而定量描述则通过数字去表达，例如空间单位用米，这人身高175cm，重量单位用65公斤，学习好坏用考试成绩等等。 

描述事物的单个特征值，叫做标量。为了直观表述标量，通常会在平面上画出一条直线，中间的位置点上一个点表示0，左边直线上的点表示负数，右边直线上的点表示正数。小学中学到的任意一个自然数、整数、小数、分数、有理数、无理数等，都能在这条直线上找到他们对应的位置。

![标量](./images/scalar.png)<br/> 
图 1： 标量，单条直线上的点。

标量的计算规则有：加减、乘除、幂次等计算形式，小学初中基本都学过。

### 1.2. 向量(矢量) vector
如果要完整描述某个事物，通常它不会只有一个维度(特征)，例如描述一个静态概念的人，维度有：身高、体重、年龄、性别、一年级语文成绩等；当描述一个动态概念，例如一架飞机的运动轨迹，维度会有：时间，纬度、经度、高度。所有这些维度共同描述了一个事物，我们称之为向量。

假设一个向量只有2个维度(特征)，例如，假设我们只关注人的身高、体重，就可以通过平面直角坐标系，x轴表示身高，y轴表示体重，这样每个人都可以在平面坐标系上的某个点来表示。

![2维向量](./images/2d-vecor-1.png)<br/> 
图 2：2维向量。平面中的一个点。空间中的点为什么叫向量？顾名思义：有方向的量。举例：一个2D向量，x=3,y=4, 就是落在平面直角坐标系上一个多点， 从坐标原点(0,0)出发到该点结束，画一个线段。

如果向量有3个维度，就能用一个3维立体空间坐标系来表示。但是，当有个更多维度时，就没有直观的坐标系与之对应了，只能用数学式子表达了：v $(x_1,x_2,x_3...x_n)$.  

向量也有对应的加减乘除，比较大小等运算，这个后续会在需要时再详细展开。

笛卡尔在17世纪发明的坐标系，让抽象的代数和直观的几何建立起了关联关系。高维和低维的规律是一致的，通过在低维可视化的平面、3D坐标系中找到的规律，同样可以应用到高维不能直观可视的空间中。

### 1.3. 矩阵 matrix
同一个向量空间中，多个向量构成矩阵,可以直观对应空间中的点集。也可以理解成一张2维的表格；纵向为向量的某个维度；横向表示某个向量；可以将矩阵理解为对应维度空间的点集；

![矩阵](./images/matrix.png)<br/> 
图 4：矩阵的表示。左侧：空间中的点集。右侧：二维表格。

### 1.4. 张量 tensor
可以理解成矩阵里放的不是向量，而是另一个矩阵。例如：描述一张图片, 它本身就是一个平面、每个平面上的像素还区分RGB值。RGB这3个通道上的长、宽像素值构成的3个矩阵，合在一起叫做一个张量。


## 2. 普通向量数据的加载
在机器学习能力还不够强大的早期阶段，人们通过一些特征工程收集量化特征，例如鸢尾花数据，植物学家收集了鸢尾花的几种特征：萼片长度、萼片宽度、花瓣长度和花瓣宽度，以及对应所属那个分类。

另外一个例子，搜索引擎在一次查询下会召回大量的文档，每个文档都有自己不同的特征：发布时间、内容长度、作者站点、内容包含的图片数量，还有一些特征工程会为文档增加新的特征：文档和查询的相关度、文档作者的权威度、历史上被展现了多少次、用户点进去看了多少次等等。都可以作为文档这个向量的维度。

我们以鸢尾花数据举例，介绍下如何使用pytorch完成这类数据集的加载。

``` python
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 参考pytorch的官方教程：
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class IrisDataSet(Dataset):
    def __init__(self,data_file):
        # iris.data 数据文件有5列，
        # 前4列分别是：萼片长度、萼片宽度、花瓣长度和花瓣宽度
        # 最后一列为所属分类
        self.df = pd.read_csv(data_file, sep=",", header=None) #使用了pandas作为数据文件加载工具
        self.label_dict = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2} 
    
    def __len__(self):
        return len(self.df) # df.size 行数*列数； len(df),行数
    
    def __getitem__(self,idx):
        features = self.df.iloc[idx, 0:4].values.astype('float32') #由于read_csv时没有指定每列的文件类型，这里需要做单独的类型转换
        label = self.label_dict.get(self.df.iloc[idx,4])
        
        # 一般情况无需再手动转tensor，dataloader会处理
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

```

## 3. 文本的向量化表示
给定一段文本，该如何向量化呢？

![文本令牌化](./images/nlp-token.png)<br/>
图 5：首先，全部语料 -> 令牌化 -> 词典文件 {"token":idx}。每个文本输入 -> 令牌化 ->  +词典文件 ->  向量或矩阵。

首先，需要将文本分割成基本都单元，这个过程叫令牌化(Tokenizing)。通常会将整个文本语料按照某种令牌化方式(Tokenizer)分割成令牌，将全部的令牌去重后产生一个词典文件，在词典文件中，每个令牌都有编号，就可以根据令牌获取词典编号，或者根据词典编号获取令牌。 
第二步，将待处理的模型令牌化后得到令牌的集合，拿着这些令牌去词典文件中查找对应的位置编号，有了位置编号后再转成向量或矩阵。 

### 3.1 文本令牌化
令牌(token)的粒度的平衡，粒度太大，词典会变大；粒度太小，单个token的表意能力会变差。

词粒度：例如，英文按单词粒度(根据空格、标点符号等正则形式划分)，中文按词语粒度(根据分词软件)划分。

词素粒度：例如英文的dogs，分成 dog + s；中文则按汉字划分。

另外，随着GPT生成式的模型的流行，生成结果自由度要求提高，不能出现词典中没有的token。

BPE的思路：
1. 文本按照一定的规则(正则表达式)，分割成token
2. 将token转成二进制的字节
3. 相邻的2个组合，并统计数量
4. 将组合次数最多的一个pair合在一起，重复这样的合并操作，直到完成设定的合并次数，
5. 最终产出encoder.json， 合并后的token，idx
    vocab.bpe，bpe tree。
6. 具体的encode()会将一个输入文本按照1,2步骤产生byte pairs, 拿着这些byepairs到 vocab.bpe遍历找到最终的bpe，再到encoder.json中找到对应的idx。

Tokenizer: https://pytorch.org/text/stable/transforms.html

[MegaByte](../paper/MegaByte.md)

### 3.2 构造词典
举例来说，假设一个完整的语料由3个句子构成：
* 我爱中国！
* 我爱长城
* 我去过北京

按单个汉字令牌化，去重后得到的分词：

"我","爱","中","国","！","长","城","去","过","北","京"。由此得到的词典文件：

``` python
dict_data = {"我":0,"爱":1,"中":2,"国":3,"！":4,
"长":5,"城":6,"去":7,"过":8,"北":9,"京":10}.

# 根据token获得词典中的位置编码
# 实际代码中，可以批量输入tokens
def get_idx(token):
    return dict_data.get(token)

# 根据位置编码获得对应的token
def get_token(idx):
    return dict_data.get(idx)

```

vocab: https://pytorch.org/text/stable/vocab.html

### 3.3 文本的向量化表示
接3.2的例子，构造完词典后，如何将单个的文本转换成向量/矩阵？

#### 3.3.1. one-hot形式
将每个汉字视为一个向量，这个向量的维度等于词典中tokens的总数，该汉字词典中的位置编号对应的维度值是1，其他维度值都为0. 例如，"我" 在词典中的位置编号是第一个位置，它的向量就是[1,0,0,0,0,0,0,0,0,0]。这种表示形式叫做one-hot向量。而一个句子中的所有字(向量)的集合则构成一个矩阵：

``` python
# "我爱中国！", 由one-hot向量构成的稀疏矩阵表示
input = [[1,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0]
]
``` 

one-hot缺点：实际的词典大小会非常大，如此稀疏巨大的矩阵，计算起来会很费劲。

#### 3.3.2. BOW(bag of word) 词袋形式
一段文本，不用由one-hot向量构成的矩阵表示，而是有一个向量表示，该向量的长度仍然等于词典元素个数，词典元素编号和向量的列数对应，如果这个句子中出现某个字，该位置=1，否则=0. 例如：

* 我爱中国！[1,1,1,1,1,0,0,0,0,0]
* 我爱长城  [1,1,0,0,0,1,1,0,0,0]
* 我去过北京 [1,0,0,0,0,0,1,1,1,1]

除了简单的，出现=1，不出现==0外，还可以将1换成该汉字在句子中出现的次数，或者用计算tf/idf值填充。

$tf*idf = \frac{ 该词在文件中的出现次数 }{ 语料中所有字词的出现次数之和 } * log(\frac{语料库中的文件总数}{包含词语的文件数目 + 1 })  $

词袋模型的缺点：向量只记录了那些字出现过，出现频率等，但不像one-hot那样，该向量没有保留字在句子中的位置信息。

#### 3.3.3. 词嵌入 word embedding
在上述one-hot章节中，每个词向量维度=词典大小，且只有一个维度=1，其他维度=0。这种稀疏的大矩阵处理起来很费劲，能不能将这个稀疏矩阵所包含的信息压缩到一个相对小的稠密矩阵，例如，向量维度=256，每个维度的值不是简单1或0，而是有区分的浮点值？

例如，我们想象一下，可以从几个维度描述一个词：1.是否名词，2.是否形容词，3.是否经常出现在句子开头，4.是否经常出现在句尾等等... 。实际操作起来，这种手工搞词嵌入的方式会非常困难，需要哪些维度，每个维度的具体数值等，都不容易确定。

我们可以将词嵌入理解成一个带token位置编号的矩阵，根据token的位置编号获取词嵌入里token向量，每一个词向量的特征值可以先随机初始化，再通过特定的算法去确定合理的特征值。有以下几种思路：
* n-gram, 根据前面n个词，预测下一个词；
* skip-gram, 根据给定的词，预测它周围的词；
* CBOW(continuous BOW), 根据周围的词，预测中间的词；

在这里，我们先知道有这么个词嵌入用于对文本进行向量化表示。具体词嵌入训练算法实现，后面再讲，可以先看看pytorch中的词嵌入如何使用：

* Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

## 4. 图像的向量化表示
图像的向量化表示相对简单些，图像是由长*宽中的每个像素构成，灰度图像的像素只有一个通道(单个矩阵)，每个像素值0~255区间，而彩色图像为RGB三个通道的矩阵构成(3个矩阵)，分别表示RGB的值。

通常为了后续的处理，会先将每个像素值除以255，这样得到一个[0,1]区间的值，然后再根据需要，做标准化处理 (每个值-平均值)/标准差。

``` python

from torchvision.io import read_image,ImageReadMode
import torchvision.transforms as T

# 加载图像，返回uint8的张量 C*H*W, 每个像素值在 [0,255] 区间
# 灰度图像:ImageReadMode.GRAY，RGB图像: .RGB，带透明度的RGB：.RGB_ALPHA
# https://pytorch.org/vision/stable/generated/torchvision.io.read_image.html
image = read_image("./flower.png", ImageReadMode.RGB) 
print(image.shape) 

# 将tensor转换 PIL Image，否则下面的ToTensor()操作会出错误提示：
# pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>
image = T.ToPILImage()(image)

# 将每个像素的值都除以255, 做max-min归一化处理， 得到一个[0,1]之间的。
# 没有这步，会出错误：Input tensor should be a float tensor. Got torch.uint8
image = T.ToTensor()(image)
print(image.shape)

# 标准化： output = (input - mean) / std
# 把值区间从 [0,1] 转为 [-1,1]区间 。 和网络里的激活函数有关
image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
print(image.shape)

```

以上代码为图像加载的重点部分，dataset，dataloader方式见第二节。

## 5. 音频的向量化表示
待补充！

## 6. 延伸阅读：多模统一输入
新近发布的若干paper表明，未来的趋势是用字节编码统一表示文本、图像、音频的向量化。
* JPEG格式：[vit_jpeg](../paper/vit/vit_jpeg.md)
* 文本、图像、音频最底层都是字节编码，在这个级别上多模输入的统一。[megabyte](../paper/MegaByte.md)