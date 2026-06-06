# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
批归一化(BN): 通过减少内部协变量偏移加快深度网络训练 2015.2.11 原文: https://arxiv.org/abs/1502.03167

## Abstract
Training Deep Neural Networks is complicated by the fact that the distribution of each layer’s inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters. 

深层神经网络的训练因以下事实而变得复杂：在训练过程中，随着前一层的参数变化，每个层的输入分布也会发生变化。这要求较低的学习速率和仔细的参数初始化，从而减慢了训练速度，并使得训练具有饱和非线性的模型非常困难。我们将这种现象称为内部协变量偏移，并通过归一化层输入来解决这个问题。我们的方法的优势在于将归一化作为模型架构的一部分，并对每个训练小批次执行归一化。批处理归一化允许我们使用更高的学习速率，并且对初始化不太敏感。它还起到了正规化的作用，在某些情况下消除了Dropout的必要性。应用于最先进的图像分类模型，批归一化以14倍的训练步骤达到相同的精度，并大大优于原始模型。使用批归一化网络集成，我们改进了ImageNet分类的SOTA结果：前5个验证错误率达到4.9%的(和4.8%的测试错误)，超过了人工评估的准确度。

## 1 Introduction
Deep learning has dramatically advanced the state of the art in vision, speech, and many other areas. Stochastic gradient descent (SGD) has proved to be an effective way of training deep networks, and SGD variants such as momentum (Sutskever et al., 2013) and Adagrad (Duchi et al., 2011) have been used to achieve state of the art performance. SGD optimizes the parameters Θ of the network, so as to minimize the loss

深度学习极大地提高了视觉、语言和许多其他领域的SOTA。随机梯度下降(SGD)已被证明是训练深层网络的有效方法，动量(Sutskeveret al., 2013)和Adagrad(Duchiet al., 2011)等SGD变体已被用于实现最先进的性能。SGD优化了网络参数θ，以最小化损失

$Θ = arg min \frac{1}{N}\sum_{i=1}^Nℓ(x_i, Θ)$

where $x_1..._N$ is the training data set. With SGD, the training proceeds in steps, and at each step we consider a minibatch $x_1..._m$ of size m. The mini-batch is used to approximate the gradient of the loss function with respect to the parameters, by computing 

其中$x_1…_N$是训练数据集。对于SGD，训练分步骤进行，在每个步骤中，我们都考虑一个大小为m的小批次$x_1…_m$。通过计算，小批次用于近似损失函数相对于参数的梯度

$\frac{1}{m}\frac{∂ℓ(xi, Θ)}{∂Θ} $.

Using mini-batches of examples, as opposed to one example at a time, is helpful in several ways. First, the gradient of the loss over a mini-batch is an estimate of the gradient over the training set, whose quality improves as the batch size increases. Second, computation over a batch can be much more efficient than m computations for individual examples, due to the parallelism afforded by the modern computing platforms.

与一次使用一个样本相反，使用小批量样本在几个方面都很有用。首先，小批次损失的梯度是对训练集梯度的估计，训练集的质量随着批次大小的增加而提高。其次，由于现代计算平台所提供的并行性，批处理计算比单个样本的m计算效率更高。

While stochastic gradient is simple and effective, it requires careful tuning of the model hyper-parameters, specifically the learning rate used in optimization, as well as the initial values for the model parameters. The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers – so that small changes to the network parameters amplify as the network becomes deeper.

虽然随机梯度简单有效，但它需要仔细调整模型超参数，特别是优化中使用的学习速率，以及模型参数的初始值。由于每个层的输入都受到前面所有层的参数的影响，所以训练变得复杂，因此随着网络的加深，网络参数的微小变化会放大。

The change in the distributions of layers’ inputs presents a problem because the layers need to continuously adapt to the new distribution. When the input distribution to a learning system changes, it is said to experience covariate shift (Shimodaira, 2000). This is typically handled via domain adaptation (Jiang, 2008). However, the notion of covariate shift can be extended beyond the learning system as a whole, to apply to its parts, such as a sub-network or a layer. Consider a network computing 

层输入分布的变化带来了一个问题，因为层需要不断适应新的分布。当学习系统的输入分布发生变化时，据说会经历协变量移位(covariate shift)(Shimodaira，2000)。这通常通过域适配来处理(Jiang，2008)。然而，协变量移位的概念可以扩展到整个学习系统之外，适用于其各个部分，例如子网络或层。考虑网络计算

$ℓ = F_2(F_1(u, Θ_1), Θ_2)$ 

where $F_1$ and $F_2$ are arbitrary transformations, and the parameters $Θ_1$, $Θ_2$ are to be learned so as to minimize the loss ℓ. Learning Θ2 can be viewed as if the inputs $x = F_1(u, Θ_1)$ are fed into the sub-network 

其中，$F_1$和$F_2$是任意变换，并且要学习参数$О_1$、$О_2$，以最小化损失ℓ. 学习θ2可以被视为输入$x=F_1(u，θ_1)$被馈入子网络

$ℓ = F_2(x, Θ_2)$ .

For example, a gradient descent step

例如，梯度下降步骤

$Θ_2 ← Θ_2 − \frac{α}{m}\sum_{i=1}^m \frac{∂F_2(xi, Θ_2)}{∂Θ_2}$ 

(for batch size m and learning rate α) is exactly equivalent to that for a stand-alone network $F_2$ with input x. Therefore, the input distribution properties that make training more efficient – such as having the same distribution between the training and test data – apply to training the sub-network as well. As such it is advantageous for the distribution of x to remain fixed over time. Then, $Θ_2$ does not have to readjust to compensate for the change in the distribution of x.

(对于批量大小m和学习速率α)完全等同于具有输入x的独立网络$F_2$。因此，使训练更有效的输入分布属性(例如，在训练和测试数据之间具有相同的分布)也适用于子网络的训练。因此，随着时间的推移，x的分布保持不变是有利的。然后，$θ_2$不必重新调整以补偿x分布的变化。

Fixed distribution of inputs to a sub-network would have positive consequences for the layers outside the subnetwork, as well. Consider a layer with a sigmoid activation function $z = g(W_u + b)$ where u is the layer input, the weight matrix W and bias vector b are the layer parameters to be learned, and $g(x) = \frac{1}{1+exp(−x)}$ . As |x| increases, g′(x) tends to zero. This means that for all dimensions of $x = W_u+b$ except those with small absolute values, the gradient flowing down to u will vanish and the model will train slowly. However, since x is affected by W, b and the parameters of all the layers below, changes to those parameters during training will likely move many dimensions of x into the saturated regime of the nonlinearity and slow down the convergence. This effect is amplified as the network depth increases. In practice, the saturation problem and the resulting vanishing gradients are usually addressed by using Rectified Linear Units (Nair & Hinton, 2010) ReLU(x) = max(x, 0), careful initialization (Bengio & Glorot, 2010; Saxe et al., 2013), and small learning rates. If, however, we could ensure that the distribution of nonlinearity inputs remains more stable as the network trains, then the optimizer would be less likely to get stuck in the saturated regime, and the training would accelerate.

子网输入的固定分布也会对子网之外的层产生积极影响。考虑一个具有sigmoid激活函数$z=g(W_u+b)$的层，其中u是层输入，权重矩阵W和偏差向量b是要学习的层参数，$g(x)=\frac{1}{1+exp(−x) }$。随着|x|的增加，g′(x)趋于零。这意味着，对于$x=W_u+b$的所有维度(绝对值较小的维度除外)，流向u的梯度将消失，模型将缓慢训练。然而，由于x受W、b和下面所有层的参数的影响，在训练期间对这些参数的更改可能会将x的许多维度移动到非线性的饱和区域，并减慢收敛速度。随着网络深度的增加，这种影响会被放大。实际上，饱和问题和由此产生的梯度消失通常通过使用校正线性单位(Nair&Hinton，2010)ReLU(x)=max(x，0)、仔细初始化(Bengio&Glroot，2010; Saxe et al.，2013)和较小的学习率来解决。然而，如果我们可以确保非线性输入的分布在网络训练时保持更稳定，那么优化器就不太可能陷入饱和状态，训练也会加快。

We refer to the change in the distributions of internal nodes of a deep network, in the course of training, as Internal Covariate Shift. Eliminating it offers a promise of faster training. We propose a new mechanism, which we call Batch Normalization, that takes a step towards reducing internal covariate shift, and in doing so dramatically accelerates the training of deep neural nets. It accomplishes this via a normalization step that fixes the means and variances of layer inputs. Batch Normalization also has a beneficial effect on the gradient flow through the network, by reducing the dependence of gradients on the scale of the parameters or of their initial values. This allows us to use much higher learning rates without the risk of divergence. Furthermore, batch normalization regularizes the model and reduces the need for Dropout (Srivastava et al., 2014). Finally, Batch Normalization makes it possible to use saturating nonlinearities by preventing the network from getting stuck in the saturated modes.

在训练过程中，我们将深层网络内部节点分布的变化称为内部协变移位(Internal Covariate Shift)。消除它可以保证更快的训练。我们提出了一种新的机制，我们称之为批归一化(BN)，它朝着减少内部协变量偏移迈出了一步，并在这样做的过程中显著加快了深层神经网络的训练。它通过一个标准化步骤来实现这一点，该步骤修复了层输入的均值和方差。通过减少梯度对参数或其初始值规模的依赖性，批归一化(BN)也对通过网络的梯度流产生有利影响。这使得我们可以使用更高的学习率，而不会出现分歧。此外，批归一化使模型正则化，并减少了Dropout的需要(Srivastavaet al., 2014)。最后，通过防止网络陷入饱和模式，批归一化(BN)可以使用饱和非线性。

In Sec. 4.2, we apply Batch Normalization to the bestperforming ImageNet classification network, and show that we can match its performance using only 7% of the training steps, and can further exceed its accuracy by a substantial margin. Using an ensemble of such networks trained with Batch Normalization, we achieve the top-5 error rate that improves upon the best known results on ImageNet classification.

在第4.2节中，我们将批归一化应用于性能最佳的ImageNet分类网络，并表明我们仅使用7%的训练步骤就可以匹配其性能，并且可以进一步大大超过其准确性。使用经过批归一化训练的此类网络的集成，我们可以获得前5位的错误率，该错误率改进了ImageNet分类的SOTA结果2 。朝着减少内部协变量偏移

## 2 Towards Reducing Internal Covariate Shift 减少内部协变量偏移
We define Internal Covariate Shift as the change in the distribution of network activations due to the change in network parameters during training. To improve the training, we seek to reduce the internal covariate shift. By fixing the distribution of the layer inputs x as the training progresses, we expect to improve the training speed. It has been long known (LeCun et al., 1998b; Wiesler & Ney, 2011) that the network training converges faster if its inputs are whitened – i.e., linearly transformed to have zero means and unit variances, and decorrelated. As each layer observes the inputs produced by the layers below, it would be advantageous to achieve the same whitening of the inputs of each layer. By whitening the inputs to each layer, we would take a step towards achieving the fixed distributions of inputs that would remove the ill effects of the internal covariate shift.

我们将内部协变量偏移定义为训练期间网络参数的变化导致的网络激活分布的变化。为了改进训练过程，我们寻求减少内部协变量的偏移。随着训练的进行，通过固定层输入x的分布，我们希望提高训练速度。众所周知(LeCun et al.，1998b; Wiesler&Ney，2011)，如果输入被白化，网络训练收敛得更快，即线性变换为均值和单位方差为零，且不相关。由于每一层都观察到下面各层产生的输入，因此最好实现每一层输入的相同白化。通过白化每一层的输入，我们将朝着实现输入的固定分布迈出一步，从而消除内部协变量迁移的不良影响。

We could consider whitening activations at every training step or at some interval, either by modifying the network directly or by changing the parameters of the optimization algorithm to depend on the network activation values (Wiesler et al., 2014; Raiko et al., 2012; Povey et al., 2014; Desjardins & Kavukcuoglu). However, if these modifications are interspersed with the optimization steps, then the gradient descent step may attempt to update the parameters in a way that requires the normalization to be updated, which reduces the effect of the gradient step. For example, consider a layer with the input u that adds the learned bias b, and normalizes the result by subtracting the mean of the activation computed over the training data:  x = x − E[x] where x = u + b, $X = {x_1..._N }$ is the set of values of x over the training set, and $E[x] = \frac{1}{N}\sum_{i=1}^Nx_i$ . If a gradient descent step ignores the dependence of E[x] on b, then it will update b ← b + ∆b, where ∆b ∝ −∂ℓ/∂bx. Then u + (b + ∆b) − E[u + (b + ∆b)] = u + b − E[u + b].

我们可以考虑在每个训练步骤或某个时间间隔进行白化激活，要么直接修改网络，要么根据网络激活值改变优化算法的参数(Wiesleret al., 2014; Raikoet al., 2012; Poveyet al., 2014;Desjardins和Kavukcuoglu)。然而，如果这些修改穿插在优化步骤中，那么梯度下降步骤可能会尝试以需要更新归一化的方式更新参数，这会降低梯度步骤的效果。例如，考虑一个具有输入u的层，该层添加了学习偏差b，并通过减去通过训练数据计算的激活平均值来归一化结果：x=x− E[x]其中x=u+b，$x={x_1…_N}$是训练集中x的值集，$E[x]=\frac{1}{N}\sum_{i=1}^Nx_i$。如果梯度下降步骤忽略了E[x对b的依赖性，则它将更新b← b+∆b、 其中∆b∝ −∂ℓ/∂bx。然后是u+(b+∆b)− E[u+(b+∆b) ]=u+b− E[u+b]。

Thus, the combination of the update to b and subsequent change in normalization led to no change in the output of the layer nor, consequently, the loss. As the training continues, b will grow indefinitely while the loss remains fixed. This problem can get worse if the normalization not only centers but also scales the activations. We have observed this empirically in initial experiments, where the model blows up when the normalization parameters are computed outside the gradient descent step.

因此，对b的更新和随后的规一化更改相结合，不会导致层的输出发生变化，也不会导致损失。随着训练的继续，b将无限期增长，而损失保持不变。如果规一化不仅集中而且扩展激活，则此问题可能会变得更糟。我们在最初的实验中观察到了这一点，当在梯度下降步骤之外计算归一化参数时，模型会崩溃。

The issue with the above approach is that the gradient descent optimization does not take into account the fact that the normalization takes place. To address this issue, we would like to ensure that, for any parameter values, the network always produces activations with the desired distribution. Doing so would allow the gradient of the loss with respect to the model parameters to account for the normalization, and for its dependence on the model parameters Θ. Let again x be a layer input, treated as a 2 vector, and X be the set of these inputs over the training data set. The normalization can then be written as a transformation 

上述方法的问题是梯度下降优化没有考虑到发生归一化的事实。为了解决这个问题，我们希望确保，对于任何参数值，网络总是以所需的分布生成激活。这样做将允许损失相对于模型参数的梯度说明归一化及其对模型参数的依赖性。再次假设x是一个层输入，被视为一个2向量，x是这些输入在训练数据集上的集合。然后可以将归一化写为转换

x = Norm(x, X) 

which depends not only on the given training example x but on all examples X – each of which depends on Θ if x is generated by another layer. For backpropagation, we would need to compute the Jacobians 

这不仅取决于给定的训练样本x，还取决于所有样本x–如果x由另一层生成，则每个样本都取决于θ。对于反向传播，我们需要计算雅可比矩阵

$\frac{∂Norm(x, X)}{∂x} and \frac{∂Norm(x, X)}{∂X}  $; 

ignoring the latter term would lead to the explosion described above. Within this framework, whitening the layer inputs is expensive, as it requires computing the covariance matrix $Cov[x] = E_x∈X [xx^T ] − E[x]E[x]^T$ and its inverse square root, to produce the whitened activations

忽略后一项将导致上述爆炸。在此框架内，白化层输入是昂贵的，因为它需要计算协方差矩阵$Cov[x]=E_x∈X[xx^T]− E[x]E[x]^T$及其平方根的倒数，以产生增白活性

$Cov[x]^{−1/2}(x − E[x])$, as well as the derivatives of these transforms for backpropagation. This motivates us to seek an alternative that performs input normalization in a way that is differentiable and does not require the analysis of the entire training set after every parameter update.

$Cov[x]^{−1/2}(x− E[x])$，以及这些变换对反向传播的导数。这促使我们寻找一种替代方法，以可区分的方式执行输入归一化，并且不需要在每次参数更新后分析整个训练集。

Some of the previous approaches (e.g. (Lyu & Simoncelli, 2008)) use statistics computed over a single training example, or, in the case of image networks, over different feature maps at a given location. However, this changes the representation ability of a network by discarding the absolute scale of activations. We want to a preserve the information in the network, by normalizing the activations in a training example relative to the statistics of the entire training data. 

之前的一些方法(例如(Lyu&Simoncelli，2008))使用在单个训练样本上计算的统计数据，或者在图像网络的情况下，使用在给定位置的不同特征图上计算的数据。然而，这通过放弃激活的绝对规模改变了网络的表示能力。我们希望通过归一化与整个训练数据统计相关的训练样本中的激活，来保存网络中的信息。

## 3 Normalization via Mini-Batch Statistics
Since the full whitening of each layer’s inputs is costly and not everywhere differentiable, we make two necessary simplifications. The first is that instead of whitening the features in layer inputs and outputs jointly, we will normalize each scalar feature independently, by making it have the mean of zero and the variance of 1. For a layer with d-dimensional input $x = (x ^{(1)} . . . x^{(d)})$, we will normalize each dimension 

$b x(k) = x(k) − E[x(k)] p Var[x(k)] $ ???

where the expectation and variance are computed over the training data set. As shown in (LeCun et al., 1998b), such normalization speeds up convergence, even when the features are not decorrelated.

Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the nonlinearity. To address this, we make sure that the transformation inserted in the network can represent the identity transform. To accomplish this, we introduce, for each activation x(k) , a pair of parameters γ(k) , β(k) , which scale and shift the normalized value: y(k) = γ(k)bx(k) + β(k).

These parameters are learned along with the original model parameters, and restore the representation power of the network. Indeed, by setting γ(k) = p Var[x(k)] and β(k) = E[x(k)], we could recover the original activations, if that were the optimal thing to do.

In the batch setting where each training step is based on the entire training set, we would use the whole set to normalize activations. However, this is impractical when using stochastic optimization. Therefore, we make the second simplification: since we use mini-batches in stochastic gradient training, of the mean and variance each mini-batch produces estimates of each activation. This way, the statistics used for normalization can fully participate in the gradient backpropagation. Note that the use of minibatches is enabled by computation of per-dimension variances rather than joint covariances; in the joint case, regularization would be required since the mini-batch size is likely to be smaller than the number of activations being whitened, resulting in singular covariance matrices.

Consider a mini-batch B of size m. Since the normalization is applied to each activation independently, let us focus on a particular activation x(k) and omit k for clarity.

We have m values of this activation in the mini-batch,

B = {x1...m}.

Let the normalized values be b x1...m, and their linear transformations be y1...m. We refer to the transform

BNγ,β : x1...m → y1...m as the Batch Normalizing Transform. We present the BN

Transform in Algorithm 1. In the algorithm, ǫ is a constant added to the mini-batch variance for numerical stability.

Input: Values of x over a mini-batch: B = {x1...m};

Parameters to be learned: γ, β

Output: {yi = BNγ,β(xi)} µB ← 1m mXi=1 xi // mini-batch mean σ2B ← 1m mXi=1 (xi − µB)2 // mini-batch variance b xi ← xi − µB p σ2B + ǫ // normalize yi ← γbxi + β ≡ BNγ,β(xi) // scale and shift

Algorithm 1: Batch Normalizing Transform, applied to activation x over a mini-batch.

The BN transform can be added to a network to manipulate any activation. In the notation y = BNγ,β(x), we 3 indicate that the parameters γ and β are to be learned, but it should be noted that the BN transform does not independently process the activation in each training example. Rather, BNγ,β(x) depends both on the training example and the other examples in the mini-batch. The scaled and shifted values y are passed to other network layers. The normalized activations b x are internal to our transformation, but their presence is crucial. The distributions of values of any b x has the expected value of 0 and the variance of 1, as long as the elements of each mini-batch are sampled from the same distribution, and if we neglect ǫ. This can be seen by observing that

P mi=1 b xi = 0 and 1m P mi=1 b x2i = 1, and taking expectations. Each normalized activation b x(k) can be viewed as an input to a sub-network composed of the linear transform y(k) = γ(k)bx(k) + β(k) , followed by the other processing done by the original network. These sub-network inputs all have fixed means and variances, and although the joint distribution of these normalized b x(k) can change over the course of training, we expect that the introduction of normalized inputs accelerates the training of the sub-network and, consequently, the network as a whole.

During training we need to backpropagate the gradient of loss ℓ through this transformation, as well as compute the gradients with respect to the parameters of the

BN transform. We use chain rule, as follows (before simplification): ∂ℓ ∂bxi = ∂y ∂ℓ i · γ ∂ℓ ∂σ2B = P mi=1 ∂ℓ ∂bxi · (xi − µB) · −12 (σ2B + ǫ)−3/2 ∂ℓ ∂µB =  P mi=1 ∂ℓ ∂bxi · −1 √σ2B+ǫ + ∂ℓ ∂σ2B · P mi=1 −2(xi−µB) m ∂ℓ ∂xi = ∂ ∂ℓ b xi · √σ12B+ǫ + ∂ℓ ∂σ2B · 2(xi−µB) m + ∂µ ∂ℓ

B · 1m ∂ℓ ∂γ = P mi=1 ∂ℓ ∂yi · b xi ∂ℓ ∂β = P mi=1 ∂ℓ ∂yi

Thus, BN transform is a differentiable transformation that introduces normalized activations into the network. This ensures that as the model is training, layers can continue learning on input distributions that exhibit less internal covariate shift, thus accelerating the training. Furthermore, the learned affine transform applied to these normalized activations allows the BN transform to represent the identity transformation and preserves the network capacity.

### 3.1 Training and Inference with BatchNormalized Networks
To Batch-Normalize a network, we specify a subset of activations and insert the BN transform for each of them, according to Alg. 1. Any layer that previously received x as the input, now receives BN(x). A model employing

Batch Normalization can be trained using batch gradient descent, or Stochastic Gradient Descent with a mini-batch size m > 1, or with any of its variants such as Adagrad (Duchi et al., 2011). The normalization of activations that depends on the mini-batch allows efficient training, but is neither necessary nor desirable during inference; we want the output to depend only on the input, deterministically.

For this, once the network has been trained, we use the normalization b x = x − E[x] p

Var[x] + ǫ using the population, rather than mini-batch, statistics.

Neglecting ǫ, these normalized activations have the same mean 0 and variance 1 as during training. We use the unbiased variance estimate Var[x] = mm−1 · EB[σ2B], where the expectation is over training mini-batches of size m and σ2B are their sample variances. Using moving averages instead, we can track the accuracy of a model as it trains.

Since the means and variances are fixed during inference, the normalization is simply a linear transform applied to each activation. It may further be composed with the scaling by γ and shift by β, to yield a single linear transform that replaces BN(x). Algorithm 2 summarizes the procedure for training batch-normalized networks.

Input: Network N with trainable parameters Θ; subset of activations {x(k)}Kk=1

Output: Batch-normalized network for inference, N inf

BN 1: N tr

BN 2: for k ← = 1

N . . . K // Training BN network do 3: Add transformation y(k) = BNγ(k) ,β(k) (x(k)) to

N tr

BN (Alg. 1) 4: Modify each layer in N tr

BN with input x(k) to take y(k) instead 5: end for 6: Train N tr

BN to optimize the parameters Θ ∪ {γ(k) , β(k)}Kk=1 7: N inf

BN ← N tr

BN // Inference BN network with frozen // parameters 8: for k = 1 . . . K do 9: // For clarity, x ≡ x(k) , γ ≡ γ(k) , µB ≡ µ(k) B , etc. 10: Process multiple training mini-batches B, each of size m, and average over them:

E[x] ← EB[µB]

Var[x] ← mm−1EB[σ2B] 11: In N inf

BN, replace the transform y = BNγ,β(x) with y = √ γ

Var[x]+ǫ · x +  β − γ E[x] √

Var[x]+ǫ  12: end for

Algorithm 2: Training a Batch-Normalized Network

### 3.2 Batch-Normalized Convolutional Networks
Batch Normalization can be applied to any set of activations in the network. Here, we focus on transforms 4 that consist of an affine transformation followed by an element-wise nonlinearity: z = g(Wu + b) where W and b are learned parameters of the model, and g(·) is the nonlinearity such as sigmoid or ReLU. This formulation covers both fully-connected and convolutional layers. We add the BN transform immediately before the nonlinearity, by normalizing x = Wu+ b. We could have also normalized the layer inputs u, but since u is likely the output of another nonlinearity, the shape of its distribution is likely to change during training, and constraining its first and second moments would not eliminate the covariate shift. In contrast, Wu + b is more likely to have a symmetric, non-sparse distribution, that is “more Gaussian” (Hyv¨arinen & Oja, 2000); normalizing it is likely to produce activations with a stable distribution.

Note that, since we normalize Wu+b, the bias b can be ignored since its effect will be canceled by the subsequent mean subtraction (the role of the bias is subsumed by β in Alg. 1). Thus, z = g(Wu + b) is replaced with z = g(BN(Wu)) where the BN transform is applied independently to each dimension of x = Wu, with a separate pair of learned parameters γ(k), β(k) per dimension.

For convolutional layers, we additionally want the normalization to obey the convolutional property – so that different elements of the same feature map, at different locations, are normalized in the same way. To achieve this, we jointly normalize all the activations in a minibatch, over all locations. In Alg. 1, we let B be the set of all values in a feature map across both the elements of a mini-batch and spatial locations – so for a mini-batch of size m and feature maps of size p × q, we use the effective mini-batch of size m′ = |B| = m · p q. We learn a pair of parameters γ(k) and β(k) per feature map, rather than per activation. Alg. 2 is modified similarly, so that during inference the BN transform applies the same linear transformation to each activation in a given feature map.

### 3.3 Batch Normalization enables higher learning rates
In traditional deep networks, too-high learning rate may result in the gradients that explode or vanish, as well as getting stuck in poor local minima. Batch Normalization helps address these issues. By normalizing activations throughout the network, it prevents small changes to the parameters from amplifying into larger and suboptimal changes in activations in gradients; for instance, it prevents the training from getting stuck in the saturated regimes of nonlinearities.

Batch Normalization also makes training more resilient to the parameter scale. Normally, large learning rates may increase the scale of layer parameters, which then amplify the gradient during backpropagation and lead to the model explosion. However, with Batch Normalization, backpropagation through a layer is unaffected by the scale of its parameters. Indeed, for a scalar a,

BN(Wu) = BN((aW)u) and we can show that ∂BN((aW)u) ∂u = ∂BN ∂(uWu) ∂BN((aW)u) ∂(aW) = 1a · ∂BN(Wu) ∂W

The scale does not affect the layer Jacobian nor, consequently, the gradient propagation. Moreover, larger weights lead to smaller gradients, and Batch Normalization will stabilize the parameter growth.

We further conjecture that Batch Normalization may lead the layer Jacobians to have singular values close to 1, which is known to be beneficial for training (Saxe et al., 2013). Consider two consecutive layers with normalized inputs, and the transformation between these normalized vectors: b z = F(bx). If we assume that b x and b z are Gaussian and uncorrelated, and that F(bx) ≈ Jbx is a linear transformation for the given model parameters, then both b x and b z have unit covariances, and I = Cov[bz] = JCov[bx]JT =

JJT . Thus, JJT = I, and so all singular values of J are equal to 1, which preserves the gradient magnitudes during backpropagation. In reality, the transformation is not linear, and the normalized values are not guaranteed to be Gaussian nor independent, but we nevertheless expect

Batch Normalization to help make gradient propagation better behaved. The precise effect of Batch Normalization on gradient propagation remains an area of further study.

### 3.4 Batch Normalization regularizes the model
When training with Batch Normalization, a training example is seen in conjunction with other examples in the mini-batch, and the training network no longer producing deterministic values for a given training example. In our experiments, we found this effect to be advantageous to the generalization of the network. Whereas Dropout (Srivastava et al., 2014) is typically used to reduce over- fitting, in a batch-normalized network we found that it can be either removed or reduced in strength. 

## 4 Experiments
## 4.1 Activations over time
To verify the effects of internal covariate shift on training, and the ability of Batch Normalization to combat it, we considered the problem of predicting the digit class on the MNIST dataset (LeCun et al., 1998a). We used a very simple network, with a 28x28 binary image as input, and 5 10K 20K 30K 40K 50K


Figure 1: (a) The test accuracy of the MNIST network trained with and without Batch Normalization, vs. the number of training steps. Batch Normalization helps the network train faster and achieve higher accuracy. (b, c) The evolution of input distributions to a typical sigmoid, over the course of training, shown as {15, 50, 85}th percentiles. Batch Normalization makes the distribution more stable and reduces the internal covariate shift. 3 fully-connected hidden layers with 100 activations each.

Each hidden layer computes y = g(Wu+b) with sigmoid nonlinearity, and the weights W initialized to small random Gaussian values. The last hidden layer is followed by a fully-connected layer with 10 activations (one per class) and cross-entropy loss. We trained the network for 50000 steps, with 60 examples per mini-batch. We added

Batch Normalization to each hidden layer of the network, as in Sec. 3.1. We were interested in the comparison between the baseline and batch-normalized networks, rather than achieving the state of the art performance on MNIST (which the described architecture does not).

Figure 1(a) shows the fraction of correct predictions by the two networks on held-out test data, as training progresses. The batch-normalized network enjoys the higher test accuracy. To investigate why, we studied inputs to the sigmoid, in the original network N and batchnormalized network N tr

BN (Alg. 2) over the course of training. In Fig. 1(b,c) we show, for one typical activation from the last hidden layer of each network, how its distribution evolves. The distributions in the original network change significantly over time, both in their mean and the variance, which complicates the training of the subsequent layers. In contrast, the distributions in the batchnormalized network are much more stable as training progresses, which aids the training.

### 4.2 ImageNet classification
We applied Batch Normalization to a new variant of the Inception network (Szegedy et al., 2014), trained on the ImageNet classification task (Russakovsky et al., 2014).

The network has a large number of convolutional and pooling layers, with a softmax layer to predict the image class, out of 1000 possibilities. Convolutional layers use ReLU as the nonlinearity. The main difference to the network described in (Szegedy et al., 2014) is that the 5 × 5 convolutional layers are replaced by two consecutive layers of 3 × 3 convolutions with up to 128 filters. The network contains 13.6 · 106 parameters, and, other than the top softmax layer, has no fully-connected layers. More details are given in the Appendix. We refer to this model as Inception in the rest of the text. The model was trained using a version of Stochastic Gradient Descent with momentum (Sutskever et al., 2013), using the mini-batch size of 32. The training was performed using a large-scale, distributed architecture (similar to (Dean et al., 2012)). All networks are evaluated as training progresses by computing the validation accuracy @1, i.e. the probability of predicting the correct label out of 1000 possibilities, on a held-out set, using a single crop per image.

In our experiments, we evaluated several modifications of Inception with Batch Normalization. In all cases, Batch Normalization was applied to the input of each nonlinearity, in a convolutional way, as described in section 3.2, while keeping the rest of the architecture constant.

#### 4.2.1 Accelerating BN Networks
Simply adding Batch Normalization to a network does not take full advantage of our method. To do so, we further changed the network and its training parameters, as follows:

Increase learning rate. In a batch-normalized model, we have been able to achieve a training speedup from higher learning rates, with no ill side effects (Sec. 3.3).

Remove Dropout. As described in Sec. 3.4, Batch Normalization fulfills some of the same goals as Dropout. Removing Dropout from Modified BN-Inception speeds up training, without increasing overfitting.

Reduce the L2 weight regularization. While in Inception an L2 loss on the model parameters controls overfitting, in Modified BN-Inception the weight of this loss is reduced by a factor of 5. We find that this improves the accuracy on the held-out validation data.

Accelerate the learning rate decay. In training Inception, learning rate was decayed exponentially. Because our network trains faster than Inception, we lower the learning rate 6 times faster.

Remove Local Response Normalization While Inception and other networks (Srivastava et al., 2014) benefit from it, we found that with Batch Normalization it is not necessary.

Shuffle training examples more thoroughly. We enabled within-shard shuffling of the training data, which prevents the same examples from always appearing in a mini-batch together. This led to about 1% improvements in the validation accuracy, which is consistent with the view of Batch Normalization as a regularizer (Sec. 3.4): the randomization inherent in our method should be most bene- ficial when it affects an example differently each time it is seen.

Reduce the photometric distortions. Because batchnormalized networks train faster and observe each training example fewer times, we let the trainer focus on more “real” images by distorting them less. 6 5M 10M 15M 20M 25M 30M

Figure 2: Single crop validation accuracy of Inception and its batch-normalized variants, vs. the number of training steps.

Figure 3: For Inception and the batch-normalized variants, the number of training steps required to reach the maximum accuracy of Inception (72.2%), and the maximum accuracy achieved by the network.

#### 4.2.2 Single-Network Classification
We evaluated the following networks, all trained on the LSVRC2012 training data, and tested on the validation data:

Inception: the network described at the beginning of Section 4.2, trained with the initial learning rate of 0.0015.
BN-Baseline: Same as Inception with Batch Normalization before each nonlinearity.
BN-x5: Inception with Batch Normalization and the modifications in Sec. 4.2.1. The initial learning rate was increased by a factor of 5, to 0.0075. The same learning rate increase with original Inception caused the model parameters to reach machine infinity.
BN-x30: Like BN-x5, but with the initial learning rate 0.045 (30 times that of Inception).
BN-x5-Sigmoid: Like BN-x5, but with sigmoid nonlinearity g(t) = 1 1+exp(−x) instead of ReLU. We also attempted to train the original Inception with sigmoid, but the model remained at the accuracy equivalent to chance.

In Figure 2, we show the validation accuracy of the networks, as a function of the number of training steps.

Inception reached the accuracy of 72.2% after 31 · 106 training steps. The Figure 3 shows, for each network, the number of training steps required to reach the same 72.2% accuracy, as well as the maximum validation accuracy reached by the network and the number of steps to reach it.

By only using Batch Normalization (BN-Baseline), we match the accuracy of Inception in less than half the number of training steps. By applying the modifications in Sec. 4.2.1, we significantly increase the training speed of the network. BN-x5 needs 14 times fewer steps than Inception to reach the 72.2% accuracy. Interestingly, increasing the learning rate further (BN-x30) causes the model to train somewhat slower initially, but allows it to reach a higher final accuracy. It reaches 74.8% after 6·106 steps, i.e. 5 times fewer steps than required by Inception to reach 72.2%.

We also verified that the reduction in internal covariate shift allows deep networks with Batch Normalization to be trained when sigmoid is used as the nonlinearity, despite the well-known difficulty of training such networks. Indeed, BN-x5-Sigmoid achieves the accuracy of 69.8%. Without Batch Normalization, Inception with sigmoid never achieves better than 1/1000 accuracy.

#### 4.2.3 Ensemble Classification
The current reported best results on the ImageNet Large Scale Visual Recognition Competition are reached by the Deep Image ensemble of traditional models (Wu et al., 2015) and the ensemble model of (He et al., 2015). The latter reports the top-5 error of 4.94%, as evaluated by the ILSVRC server. Here we report a top-5 validation error of 4.9%, and test error of 4.82% (according to the ILSVRC server). This improves upon the previous best result, and exceeds the estimated accuracy of human raters according to (Russakovsky et al., 2014).

For our ensemble, we used 6 networks. Each was based on BN-x30, modified via some of the following: increased initial weights in the convolutional layers; using Dropout (with the Dropout probability of 5% or 10%, vs. 40% for the original Inception); and using non-convolutional, per-activation Batch Normalization with last hidden layers of the model. Each network achieved its maximum accuracy after about 6 · 106 training steps. The ensemble prediction was based on the arithmetic average of class probabilities predicted by the constituent networks. The details of ensemble and multicrop inference are similar to (Szegedy et al., 2014).

We demonstrate in Fig. 4 that batch normalization allows us to set new state-of-the-art by a healthy margin on the ImageNet classification challenge benchmarks. 

## 5 Conclusion
We have presented a novel mechanism for dramatically accelerating the training of deep networks. It is based on the premise that covariate shift, which is known to complicate the training of machine learning systems, also ap- 7

Figure 4: Batch-Normalized Inception comparison with previous state of the art on the provided validation set comprising 50000 images. *BN-Inception ensemble has reached 4.82% top-5 error on the 100000 images of the test set of the ImageNet as reported by the test server. plies to sub-networks and layers, and removing it from internal activations of the network may aid in training.

Our proposed method draws its power from normalizing activations, and from incorporating this normalization in the network architecture itself. This ensures that the normalization is appropriately handled by any optimization method that is being used to train the network. To enable stochastic optimization methods commonly used in deep network training, we perform the normalization for each mini-batch, and backpropagate the gradients through the normalization parameters. Batch Normalization adds only two extra parameters per activation, and in doing so preserves the representation ability of the network. We presented an algorithm for constructing, training, and performing inference with batch-normalized networks. The resulting networks can be trained with saturating nonlinearities, are more tolerant to increased training rates, and often do not require Dropout for regularization.

Merely adding Batch Normalization to a state-of-theart image classification model yields a substantial speedup in training. By further increasing the learning rates, removing Dropout, and applying other modifications afforded by Batch Normalization, we reach the previous state of the art with only a small fraction of training steps – and then beat the state of the art in single-network image classification. Furthermore, by combining multiple models trained with Batch Normalization, we perform better than the best known system on ImageNet, by a significant margin.

Interestingly, our method bears similarity to the standardization layer of (G¨ulc¸ehre & Bengio, 2013), though the two methods stem from very different goals, and perform different tasks. The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, and in our experiments we apply it before the nonlinearity since that is where matching the first and second moments is more likely to result in a stable distribution. On the contrary, (G¨ulc¸ehre & Bengio, 2013) apply the standardization layer to the output of the nonlinearity, which results in sparser activations. In our large-scale image classification experiments, we have not observed the nonlinearity inputs to be sparse, neither with nor without Batch Normalization. Other notable differentiating characteristics of Batch Normalization include the learned scale and shift that allow the BN transform to represent identity (the standardization layer did not require this since it was followed by the learned linear transform that, conceptually, absorbs the necessary scale and shift), handling of convolutional layers, deterministic inference that does not depend on the mini-batch, and batchnormalizing each convolutional layer in the network.

In this work, we have not explored the full range of possibilities that Batch Normalization potentially enables.

Our future work includes applications of our method to Recurrent Neural Networks (Pascanu et al., 2013), where the internal covariate shift and the vanishing or exploding gradients may be especially severe, and which would allow us to more thoroughly test the hypothesis that normalization improves gradient propagation (Sec. 3.3). We plan to investigate whether Batch Normalization can help with domain adaptation, in its traditional sense – i.e. whether the normalization performed by the network would allow it to more easily generalize to new data distributions, perhaps with just a recomputation of the population means and variances (Alg. 2). Finally, we believe that further theoretical analysis of the algorithm would allow still more improvements and applications.

## References