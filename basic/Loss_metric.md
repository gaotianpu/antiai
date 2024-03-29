# 损失函数 和 评价指标 Loss and metric

损失函数获取梯度，指导模型参数更新；评价指标侧重人类对模型好坏的评估。

## 一、回归问题
均方误差
https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

## 二、分类问题
* 二分类: sigmod + 交叉熵损失函数 
https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

* 多分分类: softmax + 负对数似然
https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html 
https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

1. 分类结果的混淆矩阵 confusion matrix:

| |Positive|Negative
---|---|---
True | TP 真阳性| FN 假阴性
False| FP 假阳性 | TN 真阴性

True/False 实际的标签值Ture=1, False=0。
Positive/Negative 为模型预测的标签值P=1, N=0
* TP、True Positive   真阳性：预测为正，实际也为正
* FP、False Positive  假阳性：预测为正，实际为负
* FN、False Negative 假阴性：预测与负、实际为正
* TN、True Negative 真阴性：预测为负、实际也为负。

2. 准召率：
* $Precision = \frac{TP}{TP+FP} $ #查准率
* $Recall = \frac{TP}{TP+FN} $ #查全率(召回率)

3. P-R曲线
* x轴:是召全率, 
* y轴:为查准率，画出的曲线

4. ROC, Receiver Operating Characteristic 

* x轴: $FPr = \frac{FP}{TN+FP}$ # 分母都是实际负例
* y轴：$TPr = \frac{TP}{TP+FN}$ #查全率,分母都是实际正例

5. AUC, Area Under ROC curve, 
ROC下的曲线面积

$AUC = \frac{1}{2}\sum^{m-1}_{i=1}(x_{i+1}-x_i)*(y_i+y_{i+1}))$


F1-score

Linear Probing 线性探测？

## 三、排序问题
$LogSigLoss = -ln(\frac{1}{(1+e^{(-x)})})$, x=(chosen_reward - reject_reward)

$LogExpLoss =  ln(1 + e^{(-x)}) $, x=(chosen_reward - reject_reward)

从函数的形态上看，二者没有任何差别，LogExpLoss是LogSigLoss的简写, https://www.geogebra.org/graphing/ndfwrejz

```python
# 源： https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat/coati/models/loss.py
class LogSigLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2203.02155 
    Title:   InstructGPT: Training language models to follow instructions with human feedback . 式子-1 的实现
    chosen_reward，人类认为是好的生成；reject_reward，不好的内容
    y = -ln(1/(1+e**(-x))), x=(chosen_reward - reject_reward)
    """
    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(chosen_reward - reject_reward)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        return loss

class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    Title:   Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
    
    y = ln(1 + e**(-x) ),  x=(chosen_reward - reject_reward)
    """
    def forward(self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss
```

### 类别不平衡
除采样/下采样外，
* 平衡交叉熵函数(balanced cross entropy),样本分布角度对损失函数添加权重因子
* [Focal Loss](./Focal_Loss.md) 焦点损失, 解决正负样本或难易样本之间的类不平衡问题.
* [Quality Focal Loss (QFL):Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388) 质量焦点损失, 将分类分数和局部化质量联合表示，用于分类监督.
* VariFocal Loss(VFL) [VarifocalNet: An IoU-aware Dense Object Detector](https://arxiv.org/abs/2008.13367), 考虑不同重要性的正负样本，它平衡了来自两个样本的学习信号
* Distribution Focal Loss (DFL),分布焦点损失, 将边框子位置的基本连续分布简化为离散概率分布
* Poly Loss 多项式损失

## 四. 对比学习
* InfoNCE loss 温度系数？
* NCE(noise contrastive estimation) loss 噪声对比估计

$L_q = -log \frac{exp(q * k_+ / τ) }{ \sum^k_{i=0} exp(q * k_i / τ)  } $

https://zhuanlan.zhihu.com/p/506544456

对称交叉熵损失? symmetric 
multi-class N-pair loss


### 五、IoU


### Kullback-Leibler divergence 
* KL散度，衡量采用近似概率分布后，损失了多少信息。
* https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html


偏差-bias、方差-Variance



辅助损失

dice loss [73] 

非最大抑制(NMS), 最优传输问题 the optimal transport problem


困惑度 PPL

FID, 生成模型的评估指标

BLEU, 翻译质量评估
ROUGH， 摘要生成