# Layer Normalization
2016.7.21 https://arxiv.org/abs/1607.06450

## Abstract
Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feedforward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.

训练最先进的深度神经网络在计算上是昂贵的。减少训练时间的一种方法是使神经元的活动归一化。最近引入的一种称为批量归一化的技术使用神经元的总输入在小批量训练情况上的分布来计算均值和方差，然后使用均值和方差来归一化每个训练情况下该神经元的总输出。这显著减少了前馈神经网络中的训练时间。然而，批量归一化的效果取决于小批量大小，如何将其应用于循环神经网络尚不清楚。在本文中，我们通过计算用于归一化的均值和方差，将批量归一化转换为层归一化，该均值和方差是在单个训练情况下从层中神经元的所有求和输入归一化的。像批量归一化一样，我们也给每个神经元自己的自适应偏差和增益，这些偏差和增益在归一化之后但在非线性之前应用。与批处理归一化不同，层归一化在训练和测试时执行完全相同的计算。通过在每个时间步长分别计算归一化统计量，也可以直接应用于循环神经网络。层归一化在稳定循环网络中的隐藏状态动力学方面非常有效。根据经验，我们表明，与先前发表的技术相比，层归一化可以显著减少训练时间。

## 1 Introduction
Deep neural networks trained with some version of Stochastic Gradient Descent have been shown to substantially outperform previous approaches on various supervised learning tasks in computer vision [Krizhevsky et al., 2012] and speech processing [Hinton et al., 2012]. But state-of-the-art deep neural networks often require many days of training. It is possible to speed-up the learning by computing gradients for different subsets of the training cases on different machines or splitting the neural network itself over many machines [Dean et al., 2012], but this can require a lot of communication and complex software. It also tends to lead to rapidly diminishing returns as the degree of parallelization increases. An orthogonal approach is to modify the computations performed in the forward pass of the neural net to make learning easier. Recently, batch normalization [Ioffe and Szegedy, 2015] has been proposed to reduce training time by including additional normalization stages in deep neural networks. The normalization standardizes each summed input using its mean and its standard deviation across the training data. Feedforward neural networks trained using batch normalization converge faster even with simple SGD. In addition to training time improvement, the stochasticity from the batch statistics serves as a regularizer during training.

在计算机视觉[Krizhevsky et al.，2012]和语音处理[Hinton et al.，2012]中的各种监督学习任务上，用某种版本的随机梯度下降训练的深度神经网络已被证明显著优于以前的方法。但最先进的深度神经网络通常需要很多天的训练。通过计算不同机器上训练案例的不同子集的梯度或将神经网络本身拆分到许多机器上，可以加快学习[Dien et al.，2012]，但这可能需要大量的通信和复杂的软件。随着并行化程度的增加，它也往往导致回报迅速减少。正交方法是修改在神经网络的前向传递中执行的计算，以使学习更容易。最近，有人提出了批量归一化[Iofe and Szegedy, 2015]，通过在深度神经网络中包括额外的归一化层来减少训练时间。归一化使用其在训练数据上的平均值和标准差来归一化每个求和的输入。使用批量归一化训练的前馈神经网络即使使用简单的SGD也能更快地收敛。除了训练时间的改进，来自批量统计的随机性在训练过程中起到了正则化的作用。

Despite its simplicity, batch normalization requires running averages of the summed input statistics. In feed-forward networks with fixed depth, it is straightforward to store the statistics separately for each hidden layer. However, the summed inputs to the recurrent neurons in a recurrent neural network (RNN) often vary with the length of the sequence so applying batch normalization to RNNs appears to require different statistics for different time-steps. Furthermore, batch normalization cannot be applied to online learning tasks or to extremely large distributed models where the minibatches have to be small.

尽管批处理归一化很简单，但它需要求和输入统计数据的平均值。在具有固定深度的前馈网络中，可以直接为每个隐藏层单独存储统计信息。然而，循环神经网络(RNN)中循环神经元的总输入通常随着序列的长度而变化，因此对RNN应用批量归一化似乎需要不同时间步长的不同统计。此外，批量归一化不能应用于在线学习任务或极小批量的超大分布式模型。

This paper introduces layer normalization, a simple normalization method to improve the training speed for various neural network models. Unlike batch normalization, the proposed method directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization does not introduce any new dependencies between training cases. We show that layer normalization works well for RNNs and improves both the training time and the generalization performance of several existing RNN models.

本文介绍了层归一化，这是一种简单的归一化方法，可以提高各种神经网络模型的训练速度。与批量归一化不同，所提出的方法直接从隐藏层内神经元的总输入估计归一化统计信息，因此归一化不会在训练案例之间引入任何新的相关性。我们证明了层归一化对RNN很有效，并提高了现有几种RNN模型的训练时间和泛化性能。

## 2 Background
A feed-forward neural network is a non-linear mapping from a input pattern x to an output vector y. Consider the $l^{th}$ hidden layer in a deep feed-forward, neural network, and let $a^l$ be the vector representation of the summed inputs to the neurons in that layer. The summed inputs are computed through a linear projection with the weight matrix $W^l$  and the bottom-up inputs $h^l$ given as follows:

前馈神经网络是从输入模型x到输出向量y的非线性映射。考虑深度前馈神经网络中的第$l^{th}$个隐藏层，并让A l是该层中神经元的总输入的向量表示。求和的输入是通过线性投影计算的，其中权重矩阵$W^l$ 和自下而上的输入$h^l$如下所示：

$a^l i = w^l_i^T h^l ,  h^{l+1}_i = f(a^l_i + b^l_i ) (1)

where f(·) is an element-wise non-linear function and w l i is the incoming weights to the i th hidden units and b l i is the scalar bias parameter. The parameters in the neural network are learnt using gradient-based optimization algorithms with the gradients being computed by back-propagation.

其中f(·)是逐元非线性函数，wli是第i个隐藏单元的输入权重，bli是标量偏差参数。使用基于梯度的优化算法来学习神经网络中的参数，其中梯度通过反向传播来计算。

One of the challenges of deep learning is that the gradients with respect to the weights in one layer are highly dependent on the outputs of the neurons in the previous layer especially if these outputs change in a highly correlated way. Batch normalization [Ioffe and Szegedy, 2015] was proposed to reduce such undesirable “covariate shift”. The method normalizes the summed inputs to each hidden unit over the training cases. Specifically, for the i th summed input in the $l^{th}$ layer, the batch normalization method rescales the summed inputs according to their variances under the distribution of the data

深度学习的挑战之一是，一层中相对于权重的梯度高度依赖于前一层中神经元的输出，特别是如果这些输出以高度相关的方式变化。批量归一化[Iofe and Szegedy,2015]被提出来减少这种不希望的“协变量偏移”。该方法对训练案例中每个隐藏单元的总输入进行归一化。具体来说，对于第l层中的第i个求和输入，批处理归一化方法根据数据分布下的方差重新缩放求和输入

(2) 

where ¯$a^l$ i is normalized summed inputs to the i th hidden unit in the $l^{th}$ layer and gi is a gain parameter scaling the normalized activation before the non-linear activation function. Note the expectation is under the whole training data distribution. It is typically impractical to compute the expectations in Eq. (2) exactly, since it would require forward passes through the whole training dataset with the current set of weights. Instead, µ and σ are estimated using the empirical samples from the current mini-batch. This puts constraints on the size of a mini-batch and it is hard to apply to recurrent neural networks.

其中，$a^l$ i是第l层中第i个隐藏单元的归一化求和输入，gi是在非线性激活函数之前缩放归一化激活的增益参数。注意，期望是在整个训练数据分布下。精确计算方程(2)中的期望值通常是不切实际的，因为它需要用当前的一组权重向前通过整个训练数据集。相反，使用当前小批量的经验样本来估计µ和σ。这限制了小批量的大小，并且很难应用于循环神经网络。

## 3 Layer normalization
We now consider the layer normalization method which is designed to overcome the drawbacks of batch normalization.

我们现在考虑层归一化方法，该方法旨在克服批量归一化的缺点。

Notice that changes in the output of one layer will tend to cause highly correlated changes in the summed inputs to the next layer, especially with ReLU units whose outputs can change by a lot. This suggests the “covariate shift” problem can be reduced by fixing the mean and the variance of the summed inputs within each layer. We, thus, compute the layer normalization statistics over al$l^{th}$e hidden units in the same layer as follows:

请注意，一层输出的变化往往会导致下一层的总输入发生高度相关的变化，尤其是ReLU单元的输出可能会发生很大变化。这表明，可以通过固定每层内求和输入的平均值和方差来减少“协变量偏移”问题。因此，我们计算同一层中所有隐藏单元的层归一化统计信息，如下所示：

(3)

where H denotes the number of hidden units in a layer. The difference between Eq. (2) and Eq. (3) is that under layer normalization, al$l^{th}$e hidden units in a layer share the same normalization terms µ and σ, but different training cases have different normalization terms. Unlike batch normalization, layer normaliztion does not impose any constraint on the size of a mini-batch and it can be used in the pure online regime with batch size 1.

其中H表示层中隐藏单元的数量。等式(2)和等式(3)之间的区别在于，在层归一化下，层中的所有隐藏单元共享相同的归一化项µ和σ。与批量归一化 在不同的训练情况具有不同的归一化项 不同，层归一化不对小批量的大小施加任何约束，并且它可以用于批量大小为1的纯在线状态。

### 3.1 Layer normalized recurrent neural networks
The recent sequence to sequence models [Sutskever et al., 2014] utilize compact recurrent neural networks to solve sequential prediction problems in natural language processing. It is common among the NLP tasks to have different sentence lengths for different training cases. This is easy to deal with in an RNN because the same weights are used at every time-step. But when we apply batch normalization to an RNN in the obvious way, we need to to compute and store separate statistics for each time step in a sequence. This is problematic if a test sequence is longer than any of the training sequences. Layer normalization does not have such problem because its normalization terms depend only on the summed inputs to a layer at the current time-step. It also has only one set of gain and bias parameters shared over all time-steps.

最近的序列到序列模型[Sutskever et al.，2014]利用紧凑循环神经网络来解决自然语言处理中的序列预测问题。在NLP任务中，针对不同的训练情况具有不同的句子长度是常见的。这在RNN中很容易处理，因为在每个时间步长使用相同的权重。但是，当我们以显而易见的方式将批处理归一化应用于RNN时，我们需要为序列中的每个时间步长计算和存储单独的统计信息。如果测试序列比任何训练序列都长，这是有问题的。层归一化不存在这样的问题，因为其归一化项仅取决于在当前时间步长对层的求和输入。它也只有一组在所有时间步长上共享的增益和偏差参数。

In a standard RNN, the summed inputs in the recurrent layer are computed from the current input x t and previous vector of hidden states h t−1 which are computed as a t = Whhh t−1 + Wxhx t . The layer normalized recurrent layer re-centers and re-scales its activations using the extra normalization terms similar to Eq. (3):

在标准RNN中，循环层中的求和输入是根据当前输入x t和隐藏状态的先前向量h t−1计算的，其计算为t=Whhh t−1+Wxxx t。使用类似于等式(3)的额外归一化项，层归一化的循环层重新定中心并重新缩放其激活。(3)：

(4)

where Whh is the recurrent hidden to hidden weights and Wxh are the bottom up input to hidden weights.  is the element-wise multiplication between two vectors. b and g are defined as the bias and gain parameters of the same dimension as h t .

其中Whh是循环的隐藏到隐藏权重，Wxh是自下而上的隐藏权重输入。是两个向量之间的逐元素相乘。b和g被定义为与ht具有相同维度的偏差和增益参数。

In a standard RNN, there is a tendency for the average magnitude of the summed inputs to the recurrent units to either grow or shrink at every time-step, leading to exploding or vanishing gradients. In a layer normalized RNN, the normalization terms make it invariant to re-scaling all of the summed inputs to a layer, which results in much more stable hidden-to-hidden dynamics.

在标准RNN中，循环单元的总输入的平均幅度在每个时间步长都有增长或收缩的趋势，导致梯度爆炸或消失。在层归一化的RNN中，归一化项使其对重新缩放层的所有求和输入保持不变，这导致更稳定的隐藏到隐藏的动力学。

## 4 Related work
Batch normalization has been previously extended to recurrent neural networks [Laurent et al., 2015, Amodei et al., 2015, Cooijmans et al., 2016]. The previous work [Cooijmans et al., 2016] suggests the best performance of recurrent batch normalization is obtained by keeping independent normalization statistics for each time-step. The authors show that initializing the gain parameter in the recurrent batch normalization layer to 0.1 makes significant difference in the final performance of the model. Our work is also related to weight normalization [Salimans and Kingma, 2016]. In weight normalization, instead of the variance, the L2 norm of the incoming weights is used to normalize the summed inputs to a neuron. Applying either weight normalization or batch normalization using expected statistics is equivalent to have a different parameterization of the original feed-forward neural network. Re-parameterization in the ReLU network was studied in the Pathnormalized SGD [Neyshabur et al., 2015]. Our proposed layer normalization method, however, is not a re-parameterization of the original neural network. The layer normalized model, thus, has different invariance properties than the other methods, that we will study in the following section.

批量归一化先前已扩展到循环神经网络[Launtet al.,2015，Amodeiet al.,2015;Cooijmanset al.,2016]。先前的工作[Cooijmans et al.，2016]表明，通过对每个时间步长保持独立的归一化统计，可以获得循环批量归一化的最佳性能。作者表明，将循环批量归一化层中的增益参数初始化为0.1会对模型的最终性能产生显著影响。我们的工作也与权重归一化有关[Salimans and Kingma, 2016]。在权重归一化中，引入权重的L2范数代替方差，用于归一化到神经元的求和输入。使用期望统计应用权重归一化或批量归一化相当于对原始前馈神经网络进行不同的参数化。在路径归一化SGD中研究了ReLU网络中的重新参数化[Neyshaburet al.,2015]。然而，我们提出的层归一化方法并不是对原始神经网络的重新参数化。因此，层归一化模型与我们将在下一节中研究的其他方法具有不同的不变性。

##　5 Analysis
In this section, we investigate the invariance properties of different normalization schemes.

在本节中，我们研究了不同归一化方案的不变性。

### 5.1 Invariance under weights and data transformations 5.1权重和数据变换下的不变性
The proposed layer normalization is related to batch normalization and weight normalization. Although, their normalization scalars are computed differently, these methods can be summarized as normalizing the summed inputs ai to a neuron through the two scalars µ and σ. They also learn an adaptive bias b and gain g for each neuron after the normalization.

所提出的层归一化与批量归一化和权重归一化有关。尽管它们的归一化标量的计算方式不同，但这些方法可以泛化为通过两个标量µ和σ对神经元的相加输入ai进行归一化。他们还学习了归一化后每个神经元的自适应偏差b和增益g。

hi = f( gi σi (ai − µi) + bi) (5)


Note that for layer normalization and batch normalization, µ and σ is computed according to Eq. 2 and 3. In weight normalization, µ is 0, and σ = k wk 2.

请注意，对于层归一化和批量归一化，µ和σ是根据等式2和3计算的。在权重归一化中，µ为0，σ=k wk 2。

Table 1: Invariance properties under the normalization methods.
表1：归一化方法下的不变性。

Table 1 highlights the following invariance results for three normalization methods.

表1强调了以下三种归一化方法的不变性结果。

Weight re-scaling and re-centering: First, observe that under batch normalization and weight normalization, any re-scaling to the incoming weights wi of a single neuron has no effect on the normalized summed inputs to a neuron. To be precise, under batch and weight normalization, if the weight vector is scaled by δ, the two scalar µ and σ will also be scaled by δ. The normalized summed inputs stays the same before and after scaling. So the batch and weight normalization are invariant to the re-scaling of the weights. Layer normalization, on the other hand, is not invariant to the individual scaling of the single weight vectors. Instead, layer normalization is invariant to scaling of the entire weight matrix and invariant to a shift to all of the incoming weights in the weight matrix. Let there be two sets of model parameters θ, θ 0 whose weight matrices W and W0 differ by a scaling factor δ and all of the incoming weights in W0 are also shifted by a constant vector γ, that is W0 = δW + 1γ > . Under layer normalization, the two models effectively compute the same output:

权重重新缩放和重新居中：首先，观察到在批量归一化和权重归一化下，对单个神经元的传入权重wi的任何重新缩放都不会对神经元的归一化求和输入产生影响。准确地说，在批量和权重归一化下，如果权重向量由δ缩放，那么两个标量µ和σ也将由δ缩放。归一化的求和输入在缩放前后保持不变。因此，批量和权重归一化对于权重的重新缩放是不变的。另一方面，层归一化对于单个权重向量的单独缩放不是不变的。相反，层归一化对于整个权重矩阵的缩放是不变的，并且对于权重矩阵中所有传入权重的移位是不变的。设有两组模型参数θ，θ0，其权重矩阵W和W0相差一个比例因子δ，并且W0中的所有输入权重也偏移一个常数向量γ，即W0=δW+1γ>。在层归一化下，这两个模型有效地计算出相同的输出：

(6)

Notice that if normalization is only applied to the input before the weights, the model will not be invariant to re-scaling and re-centering of the weights.

请注意，如果归一化仅应用于权重之前的输入，则模型将不会对权重的重新缩放和重新居中保持不变。

Data re-scaling and re-centering: We can show that al$l^{th}$e normalization methods are invariant to re-scaling the dataset by verifying that the summed inputs of neurons stays constant under the changes. Furthermore, layer normalization is invariant to re-scaling of individual training cases, because the normalization scalars µ and σ in Eq. (3) only depend on the current input data. Let x 0 be a new data point obtained by re-scaling x by δ. Then we have,

数据重新缩放和重新居中：我们可以通过验证神经元的总输入在变化下保持不变来证明所有归一化方法对重新缩放数据集都是不变的。此外，层归一化对个别训练情况的重新缩放是不变的，因为方程中的归一化标量µ和σ。(3)仅取决于当前输入数据。设x0是通过用δ重新缩放x而获得的新数据点。然后我们有，

. (7)


It is easy to see re-scaling individual data points does not change the model’s prediction under layer normalization. Similar to the re-centering of the weight matrix in layer normalization, we can also show that batch normalization is invariant to re-centering of the dataset.

很容易看出，在层归一化下，重新缩放单个数据点不会改变模型的预测。类似于层归一化中权重矩阵的重新定中心，我们还可以证明批量归一化对数据集的重新定中是不变的。

### 5.2 Geometry of parameter space during learning 学习过程中的参数空间几何
We have investigated the invariance of the model’s prediction under re-centering and re-scaling of the parameters. Learning, however, can behave very differently under different parameterizations, even though the models express the same underlying function. In this section, we analyze learning behavior through the geometry and the manifold of the parameter space. We show that the normalization scalar σ can implicitly reduce learning rate and makes learning more stable.

我们研究了在参数重新定中心和重新缩放的情况下模型预测的不变性。然而，学习在不同的参数化下可能表现得非常不同，即使模型表达了相同的基本函数。在本节中，我们通过参数空间的几何和流形来分析学习行为。我们证明了归一化标量σ可以隐含地降低学习率，使学习更加稳定。

#### 5.2.1 Riemannian metric 黎曼度量
The learnable parameters in a statistical model form a smooth manifold that consists of all possible input-output relations of the model. For models whose output is a probability distribution, a natural way to measure the separation of two points on this manifold is the Kullback-Leibler divergence between their model output distributions. Under the KL divergence metric, the parameter space is a Riemannian manifold.

统计模型中的可学习参数形成了一个光滑的流形，该流形由模型的所有可能的输入输出关系组成。对于输出是概率分布的模型，测量该流形上两点分离的一种自然方法是其模型输出分布之间的Kullback-Leibler散度。在KL散度度量下，参数空间是一个黎曼流形。

The curvature of a Riemannian manifold is entirely captured by its Riemannian metric, whose quadratic form is denoted as ds2 . That is the infinitesimal distance in the tangent space at a point in the parameter space. Intuitively, it measures the changes in the model output from the parameter space along a tangent direction. The Riemannian metric under KL was previously studied [Amari, 1998] and was shown to be well approximated under second order Taylor expansion using the Fisher 4 information matrix:

黎曼流形的曲率完全由其黎曼度量捕获，其二次形式表示为ds2。这是在参数空间中的一点上的切线空间中的无穷小距离。直观地，它测量从参数空间输出的模型沿切线方向的变化。先前研究了KL下的黎曼度量[Amari，1998]，并证明其在使用Fisher 4信息矩阵的二阶Taylor展开下很好地近似：

ds2 = DKL P(y | x; θ)k P(y | x; θ + δ) ≈ 1 2 δ > F(θ)δ, (8)



F(θ) = E x∼P (x),y∼P (y | x) " ∂ log P(y | x; θ) ∂θ ∂ log P(y | x; θ) ∂θ > # , (9)



where, δ is a small change to the parameters. The Riemannian metric above presents a geometric view of parameter spaces. The following analysis of the Riemannian metric provides some insight into how normalization methods could help in training neural networks.

其中，δ是参数的微小变化。上面的黎曼度量给出了参数空间的几何视图。下面对黎曼度量的分析为归一化方法如何帮助训练神经网络提供了一些见解。

#### 5.2.2 The geometry of normalized generalized linear models 归一化广义线性模型的几何
We focus our geometric analysis on the generalized linear model. The results from the following analysis can be easily applied to understand deep neural networks with block-diagonal approximation to the Fisher information matrix, where each block corresponds to the parameters for a single neuron.

我们把几何分析的重点放在广义线性模型上。以下分析的结果可以很容易地应用于理解具有Fisher信息矩阵的块对角近似的深度神经网络，其中每个块对应于单个神经元的参数。

A generalized linear model (GLM) can be regarded as parameterizing an output distribution from the exponential family using a weight vector w and bias scalar b. To be consistent with the previous sections, the log likelihood of the GLM can be written using the summed inputs a as the following:

广义线性模型(GLM)可以被视为使用权向量w和偏差标量b对指数族的输出分布进行参数化。为了与前面的章节一致，GLM的对数似然可以使用求和输入A写成如下：

log P(y | x; w, b) = (a + b)y − η(a + b) φ + c(y, φ), (10)



E[y | x] = f(a + b) = f(w > x + b), Var[y | x] = φf0 (a + b), (11)


where, f(·) is the transfer function that is the analog of the non-linearity in neural networks, f 0 (·) is the derivative of the transfer function, η(·) is a real valued function and c(·) is the log partition function. φ is a constant that scales the output variance. Assume a H-dimensional output vector y = [y1, y2, · · · , yH] is modeled using H independent GLMs and log P(y | x; W, b) = P H i=1 log P(yi | x; wi , bi). Let W be the weight matrix whose rows are the weight vectors of the individual GLMs, b denote the bias vector of length H and vec(·) denote the Kronecker vector operator. The Fisher information matrix for the multi-dimensional GLM with respect to its parameters θ = [w1 > , b1, · · · , w>H, bH] > = vec([W, b] > ) is simply the expected Kronecker product of the data features and the output covariance matrix:

其中，f(·)是神经网络中非线性的模拟传递函数，f0(·)为传递函数的导数，η(·)表示实值函数，c(·)则为对数配分函数。φ是缩放输出方差的常数。假设H维输出向量y=[y1，y2，··，yH]使用H无关的GLM和log P(y|x;W，b)=P H i=1 log P(yi|x;wi，bi)建模。设W是权重矩阵，其行是单个GLM的权重向量，b表示长度为H的偏差向量，vec(·)表示Kronecker向量算子。多维GLM关于其参数θ=[w1>，b1，···，w>H，bH]>=vec([w，b]>)的Fisher信息矩阵简单地是数据特征和输出协方差矩阵的预期Kronecker乘积：

(12)


We obtain normalized GLMs by applying the normalization methods to the summed inputs a in the original mode$l^{th}$rough µ and σ. Without loss of generality, we denote F¯ as the Fisher information matrix under the normalized multi-dimensional GLM with the additional gain parameters θ = vec([W, b, g] > ):

我们通过µ和σ将归一化方法应用于原始模型中的求和输入a，从而获得归一化GLM。在不失一般性的情况下，我们将F表示为具有附加增益参数θ=vec([W，b，g]>)的归一化多维GLM下的Fisher信息矩阵：

. (14)



Implicit learning rate reduction through the growth of the weight vector: Notice that, comparing to standard GLM, the block F¯ ij along the weight vector wi direction is scaled by the gain parameters and the normalization scalar σi . If the norm of the weight vector wi grows twice as large, even though the model’s output remains the same, the Fisher information matrix will be different. The curvature along the wi direction will change by a factor of 1 2 because the σi will also be twice as large. As a result, for the same parameter update in the normalized model, the norm of the weight vector effectively controls the learning rate for the weight vector. During learning, it is harder to change the orientation of the weight vector with large norm. The normalization methods, therefore, 

通过权重向量的增长来降低内隐学习率：注意，与标准GLM相比，沿着权重向量wi方向的块F’ij由增益参数和归一化标量σi缩放。如果权重向量wi的范数增长两倍大，即使模型的输出保持不变，Fisher信息矩阵也会不同。沿着wi方向的曲率将以12的因子变化，因为σi也将是其两倍大。结果，对于归一化模型中的相同参数更新，权重向量的范数有效地控制了权重向量的学习率。在学习过程中，很难改变具有大范数的权重向量的方向。因此，归一化方法是

Figure 1: Recall@K curves using order-embeddings with and without layer normalization.
图1：Recall@K使用具有和不具有层归一化的顺序嵌入的曲线。

Table 2: Average results across 5 test splits for caption and image retrieval. R@K is Recall@K (high is good). Mean r is the mean rank (low is good). Sym corresponds to the symmetric baseline while OE indicates order-embeddings. have an implicit “early stopping” effect on the weight vectors and help to stabilize learning towards convergence.
表2：标题和图像检索的5个测试拆分的平均结果。R@K是Recall@K(高即好)。平均r是平均等级(低即好)。Sym对应于对称基线，而OE表示有序嵌入。对权重向量具有隐含的“早期停止”效应，并有助于稳定向收敛的学习。

Learning the magnitude of incoming weights: In normalized models, the magnitude of the incoming weights is explicitly parameterized by the gain parameters. We compare how the model output changes between updating the gain parameters in the normalized GLM and updating the magnitude of the equivalent weights under original parameterization during learning. The direction along the gain parameters in F¯ captures the geometry for the magnitude of the incoming weights. We show that Riemannian metric along the magnitude of the incoming weights for the standard GLM is scaled by the norm of its input, whereas learning the gain parameters for the batch normalized and layer normalized models depends only on the magnitude of the prediction error. Learning the magnitude of incoming weights in the normalized model is therefore, more robust to the scaling of the input and its parameters than in the standard model. See Appendix for detailed derivations.

学习传入权重的大小：在归一化模型中，传入权重的幅度由增益参数明确参数化。我们比较了在学习过程中，在原始参数化下更新归一化GLM中的增益参数和更新等效权重的大小之间，模型输出如何变化。F’中沿增益参数的方向捕获了传入权重大小的几何结构。我们证明了标准GLM沿输入权重大小的黎曼度量是由其输入的范数缩放的，而学习批量归一化和层归一化模型的增益参数仅取决于预测误差的大小。因此，在归一化模型中学习传入权重的大小对输入及其参数的缩放比在标准模型中更稳健。详细推导见附录。

## 6 Experimental results 实验结果
We perform experiments with layer normalization on 6 tasks, with a focus on recurrent neural networks: image-sentence ranking, question-answering, contextual language modelling, generative modelling, handwriting sequence generation and MNIST classification. Unless otherwise noted, the default initialization of layer normalization is to set the adaptive gains to 1 and the biases to 0 in the experiments.

我们在6个任务上进行了层归一化实验，重点是循环神经网络：图像句子排序、问题回答、上下文语言建模、生成建模、手写序列生成和MNIST分类。除非另有说明，否则在实验中，层归一化的默认初始化是将自适应增益设置为1，将偏差设置为0。

### 6.1 Order embeddings of images and language 图像和语言的顺序嵌入
In this experiment, we apply layer normalization to the recently proposed order-embeddings model of Vendrov et al. [2016] for learning a joint embedding space of images and sentences. We follow the same experimental protocol as Vendrov et al. [2016] and modify their publicly available code to incorporate layer normalization 1 which utilizes Theano [Team et al., 2016]. Images and sentences from the Microsoft COCO dataset [Lin et al., 2014] are embedded into a common vector space, where a GRU [Cho et al., 2014] is used to encode sentences and the outputs of a pre-trained VGG ConvNet [Simonyan and Zisserman, 2015] (10-crop) are used to encode images. The orderembedding model represents images and sentences as a 2-level partial ordering and replaces the cosine similarity scoring function used in Kiros et al. [2014] with an asymmetric one.

在这个实验中，我们将层归一化应用于Vendrovet al.,最近提出的顺序嵌入模型。[2016]用于学习图像和句子的联合嵌入空间。我们遵循与Vendrovet al.,相同的实验协议。【2016】并修改其公开可用的代码，以纳入利用Theano的层归一化1【Teamet al.,2016】。来自Microsoft COCO数据集[Lin et al.，2014]的图像和句子被嵌入到公共向量空间中，其中GRU[Cho et al.，14]用于对句子进行编码，并且预训练的VGG ConvNet[Simonyan和Zisserman，2015](10作物)的输出用于对图像进行编码。有序嵌入模型将图像和句子表示为2级偏序，并将Kiroset al.,[2014]中使用的余弦相似性评分函数替换为非对称函数。

1 https://github.com/ivendrov/order-embedding

Figure 2: Validation curves for the attentive reader model. BN results are taken from [Cooijmans et al., 2016].
图2：专注读者模型的验证曲线。BN结果取自【Cooijmanset al.,2016】。

We trained two models: the baseline order-embedding model as well as the same model with layer normalization applied to the GRU. After every 300 iterations, we compute Recall@K (R@K) values on a held out validation set and save the model whenever R@K improves. The best performing models are then evaluated on 5 separate test sets, each containing 1000 images and 5000 captions, for which the mean results are reported. Both models use Adam [Kingma and Ba, 2014] with the same initial hyperparameters and both models are trained using the same architectural choices as used in Vendrov et al. [2016]. We refer the reader to the appendix for a description of how layer normalization is applied to GRU.

我们训练了两个模型：基线顺序嵌入模型以及应用于GRU的层归一化的同一模型。每300次迭代后，我们计算Recall@K(R@K)值，并随时保存模型R@K改进。然后，在5个单独的测试集上评估表现最好的模型，每个测试集包含1000张图像和5000个字幕，并报告其平均结果。两个模型都使用具有相同初始超参数的Adam[Kingma和Ba，2014]，并且两个模型使用与Vendrovet al.,相同的架构选择进行训练。[2016]。我们请读者参阅附录，了解层归一化如何应用于GRU。

Figure 1 illustrates the validation curves of the models, with and without layer normalization. We plot R@1, R@5 and R@10 for the image retrieval task. We observe that layer normalization offers a per-iteration speedup across all metrics and converges to its best validation model in 60% of the time it takes the baseline model to do so. In Table 2, the test set results are reported from which we observe that layer normalization also results in improved generalization over the original model. The results we report are state-of-the-art for RNN embedding models, with only the structure-preserving model of Wang et al. [2016] reporting better results on this task. However, they evaluate under different conditions (1 test set instead of the mean over 5) and are thus not directly comparable.

图1显示了模型的验证曲线，包括和不包括层归一化。我们策划R@1，R@5和R@10用于图像检索任务。我们观察到，层归一化在所有度量中提供了每次迭代的加速，并在基线模型所需时间的60%内收敛到其最佳验证模型。在表2中，报告了测试集结果，我们从中观察到，与原始模型相比，层归一化还提高了泛化能力。我们报告的结果对于RNN嵌入模型来说是最先进的，只有Wanget al.,的结构保持模型。[2016]报告了这项任务的更好结果。然而，它们在不同的条件下进行评估(1个测试集，而不是5个以上的平均值)，因此不能直接进行比较。

### 6.2 Teaching machines to read and comprehend 教机器阅读和理解
In order to compare layer normalization to the recently proposed recurrent batch normalization [Cooijmans et al., 2016], we train an unidirectional attentive reader model on the CNN corpus both introduced by Hermann et al. [2015]. This is a question-answering task where a query description about a passage must be answered by filling in a blank. The data is anonymized such that entities are given randomized tokens to prevent degenerate solutions, which are consistently permuted during training and evaluation. We follow the same experimental protocol as Cooijmans et al. [2016] and modify their public code to incorporate layer normalization 2 which uses Theano [Team et al., 2016]. We obtained the pre-processed dataset used by Cooijmans et al. [2016] which differs from the original experiments of Hermann et al. [2015] in that each passage is limited to 4 sentences. In Cooijmans et al. [2016], two variants of recurrent batch normalization are used: one where BN is only applied to the LSTM while the other applies BN everywhere throughout the model. In our experiment, we only apply layer normalization within the LSTM.

为了将层归一化与最近提出的循环批量归一化进行比较[Coijmans et al.，2016]，我们在Hermann et al.[2015]引入的CNN语料库上训练了一个单向注意读者模型。这是一个问答任务，必须通过填空来回答关于一段话的查询描述。数据是匿名的，因此实体被赋予随机令牌，以防止退化解，退化解在训练和评估过程中被一致地排列。我们遵循与Cooijmanset al.,相同的实验协议。【2016】并修改他们的公共代码，以纳入使用Theano的层归一化2【Teamet al.,2016】。我们获得了Cooijmanset al.,使用的预处理数据集。【2016】与Hermannet al.,【2015】的原始实验不同之处在于，每个段落限制为4个句子。在Cooijmanset al.,[2016]中，使用了两种循环批量归一化的变体：一种是BN仅应用于LSTM，而另一种是在整个模型中处处应用BN。在我们的实验中，我们只在LSTM中应用层归一化。

The results of this experiment are shown in Figure 2. We observe that layer normalization not only trains faster but converges to a better validation result over both the baseline and BN variants. In Cooijmans et al. [2016], it is argued that the scale parameter in BN must be carefully chosen and is set to 0.1 in their experiments. We experimented with layer normalization for both 1.0 and 0.1 scale initialization and found that the former model performed significantly better. This demonstrates that layer normalization is not sensitive to the initial scale in the same way that recurrent BN is. 3

这个实验的结果如图2所示。我们观察到，与基线和BN变体相比，层归一化不仅训练得更快，而且收敛到更好的验证结果。在Cooijmanset al.,[2016]中，有人认为BN中的尺度参数必须仔细选择，并且在他们的实验中设置为0.1。我们对1.0和0.1尺度初始化的层归一化进行了实验，发现前一个模型的性能明显更好。这表明，层归一化对初始尺度不敏感，就像反复出现的BN一样。3

### 6.3 Skip-thought vectors
Skip-thoughts [Kiros et al., 2015] is a generalization of the skip-gram model [Mikolov et al., 2013] for learning unsupervised distributed sentence representations. Given contiguous text, a sentence is encoded with a encoder RNN and decoder RNNs are used to predict the surrounding sentences. Kiros et al. [2015] showed that this model could produce generic sentence representations that perform well on several tasks without being fine-tuned. However, training this model is timeconsuming, requiring several days of training in order to produce meaningful results.

Skip things[Kiros et al.，2015]是Skip-gram模型[Mikolov et al.，2013]的推广，用于学习无监督的分布式句子表示。给定连续文本，使用编码器RNN对句子进行编码，解码器RNN用于预测周围的句子。Kiroset al.,[2015]表明，该模型可以产生在几个任务上表现良好的通用句子表示，而无需进行微调。然而，训练这个模型很耗时，需要几天的训练才能产生有意义的结果。

2 https://github.com/cooijmanstim/Attentive_reader/tree/bn


3 We only produce results on the validation set, as in the case of Cooijmans et al. [2016]
3我们只在验证集上产生结果，如Cooijmanset al.,的情况。【2016】

Figure 3: Performance of skip-thought vectors with and without layer normalization on downstream tasks as a function of training iterations. The original lines are the reported results in [Kiros et al., 2015]. Plots with error use 10-fold cross validation. Best seen in color.
图3：作为训练迭代的函数，具有和不具有层归一化的跳跃思想向量在下游任务上的性能。原始线是[Kiros et al.，2015]中报告的结果。有错误的绘图使用10倍交叉验证。最好的颜色。

Table 3: Skip-thoughts results. The first two evaluation columns indicate Pearson and Spearman correlation, the third is mean squared error and the remaining indicate classification accuracy. Higher is better for all evaluations except MSE. Our models were trained for 1M iterations with the exception of (†) which was trained for 1 month (approximately 1.7M iterations)
表3：跳过思考结果。前两个评估列表示Pearson和Spearman相关性，第三个是均方误差，其余表示分类准确性。对于除MSE之外的所有评估，越高越好。我们的模型训练了1M次迭代，但(†)除外，它训练了1个月(约1.7M次迭代)

In this experiment we determine to what effect layer normalization can speed up training. Using the publicly available code of Kiros et al. [2015] 4 , we train two models on the BookCorpus dataset [Zhu et al., 2015]: one with and one without layer normalization. These experiments are performed with Theano [Team et al., 2016]. We adhere to the experimental setup used in Kiros et al. [2015], training a 2400-dimensional sentence encoder with the same hyperparameters. Given the size of the states used, it is conceivable layer normalization would produce slower per-iteration updates than without. However, we found that provided CNMeM 5 is used, there was no significant difference between the two models. We checkpoint both models after every 50,000 iterations and evaluate their performance on five tasks: semantic-relatedness (SICK) [Marelli et al., 2014], movie review sentiment (MR) [Pang and Lee, 2005], customer product reviews (CR) [Hu and Liu, 2004], subjectivity/objectivity classification (SUBJ) [Pang and Lee, 2004] and opinion polarity (MPQA) [Wiebe et al., 2005]. We plot the performance of both models for each checkpoint on all tasks to determine whether the performance rate can be improved with LN.

在这个实验中，我们确定了层归一化可以在多大程度上加快训练。使用Kiroset al.,的公开代码。[2015]4，我们在BookCorpus数据集上训练了两个模型[Zhu et al.，2015]：一个有层归一化，一个没有层归一化。这些实验是用Theano进行的[Team et al.，2016]。我们坚持Kiroset al.,使用的实验设置。[2015]，用相同的超参数训练2400维句子编码器。考虑到所使用的状态的大小，可以想象，层归一化将产生比没有更慢的每次迭代更新。然而，我们发现，如果使用CNMeM 5，两种模型之间没有显著差异。我们在每50000次迭代后检查这两个模型，并评估它们在五项任务上的性能：语义相关性(SICK)[Marelli et al.，2014]、电影评论情感(MR)[Pang和Lee，2005]、客户产品评论(CR)[Hu和Liu，2004]、主观/客观分类(SUBJ)[Pong和Lee，2004]和意见极性(MPQA)[Wiebe et al。我们绘制了两个模型在所有任务中每个检查点的性能图，以确定LN是否可以提高性能。

The experimental results are illustrated in Figure 3. We observe that applying layer normalization results both in speedup over the baseline as well as better final results after 1M iterations are performed as shown in Table 3. We also let the model with layer normalization train for a total of a month, resulting in further performance gains across all but one task. We note that the performance differences between the original reported results and ours are likely due to the fact that the publicly available code does not condition at each timestep of the decoder, where the original model does.

实验结果如图3所示。我们观察到，如表3所示，在执行1M次迭代后，应用层归一化导致在基线上的加速以及更好的最终结果。我们还让具有层归一化的模型总共训练一个月，从而在除一个任务外的所有任务中进一步提高性能。我们注意到，原始报告的结果和我们的结果之间的性能差异可能是由于公开可用的代码在解码器的每个时间步长都不存在条件，而原始模型存在条件。

4 https://github.com/ryankiros/skip-thoughts

5 https://github.com/NVIDIA/cnmem

Figure 5: Handwriting sequence generation model negative log likelihood with and without layer normalization. The models are trained with mini-batch size of 8 and sequence length of 500.
图5：有和没有层归一化的手写序列生成模型负对数似然。模型以8的小批量大小和500的序列长度进行训练。

Figure 4: DRAW model test negative log likelihood with and without layer normalization.
图4:DRAW模型在有和没有层归一化的情况下测试负对数似然性。

We also experimented with the generative modeling on the MNIST dataset. Deep Recurrent Attention Writer (DRAW) [Gregor et al., 2015] has previously achieved the state-of-theart performance on modeling the distribution of MNIST digits. The model uses a differential attention mechanism and a recurrent neural network to sequentially generate pieces of an image. We evaluate the effect of layer normalization on a DRAW model using 64 glimpses and 256 LSTM hidden units. The model is trained with the default setting of Adam [Kingma and Ba, 2014] optimizer and the minibatch size of 128. Previous publications on binarized MNIST have used various training protocols to generate their datasets. In this experiment, we used the fixed binarization from Larochelle and Murray [2011]. The dataset has been split into 50,000 training, 10,000 validation and 10,000 test images.

我们还在MNIST数据集上进行了生成建模实验。深度循环注意力书写器(DRAW)[Gregor et al.，2015]此前在MNIST数字分布建模方面取得了最先进的性能。该模型使用差分注意力机制和循环神经网络来顺序生成图像片段。我们使用64个一瞥和256个LSTM隐藏单元来评估层归一化对DRAW模型的影响。该模型使用Adam[Kingma和Ba，2014]优化器的默认设置和128的小批量大小进行训练。先前关于二进制MNIST的出版物已经使用各种训练协议来生成其数据集。在这个实验中，我们使用了Larochelle和Murray[2011]的固定二值化。数据集分为50000个训练图像、10000个验证图像和10000个测试图像。

Figure 4 shows the test variational bound for the first 100 epoch. It highlights the speedup benefit of applying layer normalization that the layer normalized DRAW converges almost twice as fast than the baseline model. After 200 epoches, the baseline model converges to a variational log likelihood of 82.36 nats on the test data and the layer normalization model obtains 82.09 nats.

图4显示了第一个100历元的测试变分界限。它强调了应用层归一化的加速优势，即层归一化DRAW的收敛速度几乎是基线模型的两倍。在200个周期之后，基线模型在测试数据上收敛到82.36 nats的变分对数似然，并且层归一化模型获得82.09 nats。

### 6.5 Handwriting sequence generation 手写序列生成
The previous experiments mostly examine RNNs on NLP tasks whose lengths are in the range of 10 to 40. To show the effectiveness of layer normalization on longer sequences, we performed handwriting generation tasks using the IAM Online Handwriting Database [Liwicki and Bunke, 2005]. IAM-OnDB consists of handwritten lines collected from 221 different writers. When given the input character string, the goal is to predict a sequence of x and y pen co-ordinates of the corresponding handwriting line on the whiteboard. There are, in total, 12179 handwriting line sequences. The input string is typically more than 25 characters and the average handwriting line has a length around 700.

先前的实验主要研究长度在10到40之间的NLP任务上的RNN。为了显示层归一化对较长序列的有效性，我们使用IAM在线手写数据库[Liwicki和Bunke，2005]执行了手写生成任务。IAM OnDB由221位不同作者收集的手写行组成。当给定输入字符串时，目标是预测白板上相应手写线的x和y笔坐标序列。总共有12179个手写行序列。输入字符串通常超过25个字符，并且平均手写行的长度约为700。

We used the same model architecture as in Section (5.2) of Graves [2013]. The model architecture consists of three hidden layers of 400 LSTM cells, which produce 20 bivariate Gaussian mixture components at the output layer, and a size 3 input layer. The character sequence was encoded with one-hot vectors, and hence the window vectors were size 57. A mixture of 10 Gaussian functions was used for the window parameters, requiring a size 30 parameter vector. The total number of weights was increased to approximately 3.7M. The model is trained using mini-batches of size 8 and the Adam [Kingma and Ba, 2014] optimizer.

我们使用了与Graves[2013]第(5.2)节中相同的模型架构。该模型架构由400个LSTM单元的三个隐藏层组成，在输出层产生20个二元高斯混合分量，以及一个大小为3的输入层。字符序列是用一个热矢量编码的，因此窗口矢量的大小为57。10个高斯函数的混合用于窗口参数，需要大小为30的参数向量。权重总数增加到约370万。使用尺寸为8的小批量和Adam[Kingma和Ba，2014]优化器对模型进行训练。

The combination of small mini-batch size and very long sequences makes it important to have very stable hidden dynamics. Figure 5 shows that layer normalization converges to a comparable log likelihood as the baseline model but is much faster.

小批量和超长序列的结合使得具有非常稳定的隐藏动力学非常重要。图5显示，层归一化收敛到与基线模型相当的对数似然，但速度要快得多。

Figure 6: Permutation invariant MNIST 784-1000-1000-10 model negative log likelihood and test error with layer normalization and batch normalization. (Left) The models are trained with batchsize of 128. (Right) The models are trained with batch-size of 4.
图6：置换不变MNIST 784-1000-1000-10使用层归一化和批归一化对负对数似然和测试误差进行建模。(左)以128的批量大小训练模型。(右)使用批量大小为4的模型进行训练。

### 6.6 Permutation invariant MNIST 置换不变MNIST
In addition to RNNs, we investigated layer normalization in feed-forward networks. We show how layer normalization compares with batch normalization on the well-studied permutation invariant MNIST classification problem. From the previous analysis, layer normalization is invariant to input re-scaling which is desirable for the internal hidden layers. But this is unnecessary for the logit outputs where the prediction confidence is determined by the scale of the logits. We only apply layer normalization to the fully-connected hidden layers that excludes the last softmax layer.

除了RNN，我们还研究了前馈网络中的层归一化。我们演示了在充分研究的置换不变MNIST分类问题上，层归一化与批归一化的比较。根据先前的分析，层归一化对于输入重缩放是不变的，这对于内部隐藏层是期望的。但这对于logit输出来说是不必要的，其中预测置信度由logit的规模决定。我们只将层归一化应用于完全连接的隐藏层，不包括最后一个softmax层。

Al$l^{th}$e models were trained using 55000 training data points and the Adam [Kingma and Ba, 2014] optimizer. For the smaller batch-size, the variance term for batch normalization is computed using the unbiased estimator. The experimental results from Figure 6 highlight that layer normalization is robust to the batch-sizes and exhibits a faster training convergence comparing to batch normalization that is applied to all layers.

所有模型都使用55000个训练数据点和Adam[Kingma和Ba，2014]优化器进行了训练。对于较小的批量，使用无偏估计器计算批量归一化的方差项。图6的实验结果强调，与应用于所有层的批处理归一化相比，层归一化对批处理大小是稳健的，并且表现出更快的训练收敛。

### 6.7 Convolutional Networks
We have also experimented with convolutional neural networks. In our preliminary experiments, we observed that layer normalization offers a speedup over the baseline model without normalization, but batch normalization outperforms the other methods. With fully connected layers, al$l^{th}$e hidden units in a layer tend to make similar contributions to the final prediction and re-centering and rescaling the summed inputs to a layer works well. However, the assumption of similar contributions is no longer true for convolutional neural networks. The large number of the hidden units whose receptive fields lie near the boundary of the image are rarely turned on and thus have very different statistics from the rest of the hidden units within the same layer. We think further research is needed to make layer normalization work well in ConvNets.

我们还对卷积神经网络进行了实验。在我们的初步实验中，我们观察到，在没有归一化的情况下，层归一化比基线模型提供了加速，但批量归一化优于其他方法。对于完全连接的层，层中的所有隐藏单元往往会对最终预测做出类似的贡献，并且对层的总输入进行重新居中和重新缩放效果良好。然而，对于卷积神经网络来说，类似贡献的假设不再成立。感受野位于图像边界附近的大量隐藏单元很少被打开，因此与同一层内的其他隐藏单元具有非常不同的统计数据。我们认为需要进一步的研究来使层归一化在ConvNets中良好地工作。

## 7 Conclusion
In this paper, we introduced layer normalization to speed-up the training of neural networks. We provided a theoretical analysis that compared the invariance properties of layer normalization with batch normalization and weight normalization. We showed that layer normalization is invariant to per training-case feature shifting and scaling.

在本文中，我们引入了层归一化来加快神经网络的训练。我们提供了一个理论分析，比较了层归一化、批量归一化和权重归一化的不变性。我们证明了层归一化对于每个训练情况的特征移位和缩放是不变的。

Empirically, we showed that recurrent neural networks benefit the most from the proposed method especially for long sequences and small mini-batches.

经验表明，循环神经网络从所提出的方法中受益最大，尤其是对于长序列和小批量。

## Acknowledgments
This research was funded by grants from NSERC, CFI, and Google. 

## References
* Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural
networks. In NIPS, 2012.
* Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior,
Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neural networks for acoustic modeling in
speech recognition: The shared views of four research groups. IEEE, 2012.
* Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Andrew Senior, Paul Tucker,
Ke Yang, Quoc V Le, et al. Large scale distributed deep networks. In NIPS, 2012.
* Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing
internal covariate shift. ICML, 2015.
* Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In
Advances in neural information processing systems, pages 3104–3112, 2014.
* C´esar Laurent, Gabriel Pereyra, Phil´emon Brakel, Ying Zhang, and Yoshua Bengio. Batch normalized recurrent
neural networks. arXiv preprint arXiv:1510.01378, 2015.
* Dario Amodei, Rishita Anubhai, Eric Battenberg, Carl Case, Jared Casper, Bryan Catanzaro, Jingdong Chen,
Mike Chrzanowski, Adam Coates, Greg Diamos, et al. Deep speech 2: End-to-end speech recognition in
english and mandarin. arXiv preprint arXiv:1512.02595, 2015.
* Tim Cooijmans, Nicolas Ballas, C´esar Laurent, and Aaron Courville. Recurrent batch normalization. arXiv
preprint arXiv:1603.09025, 2016.
* Tim Salimans and Diederik P Kingma. Weight normalization: A simple reparameterization to accelerate train￾ing of deep neural networks. arXiv preprint arXiv:1602.07868, 2016.
* Behnam Neyshabur, Ruslan R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep
neural networks. In Advances in Neural Information Processing Systems, pages 2413–2421, 2015.
* Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation, 1998.
* Ivan Vendrov, Ryan Kiros, Sanja Fidler, and Raquel Urtasun. Order-embeddings of images and language. ICLR, 2016.
* The Theano Development Team, Rami Al-Rfou, Guillaume Alain, Amjad Almahairi, Christof Angermueller,
Dzmitry Bahdanau, Nicolas Ballas, Fr´ed´eric Bastien, Justin Bayer, Anatoly Belikov, et al. Theano: A python
framework for fast computation of mathematical expressions. arXiv preprint arXiv:1605.02688, 2016.
* Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and
C Lawrence Zitnick. Microsoft coco: Common objects in context. ECCV, 2014.
* Kyunghyun Cho, Bart Van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical
machine translation. EMNLP, 2014.
* Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. ICLR, 2015.
* Ryan Kiros, Ruslan Salakhutdinov, and Richard S Zemel. Unifying visual-semantic embeddings with multi￾modal neural language models. arXiv preprint arXiv:1411.2539, 2014.
* D. Kingma and J. L. Ba. Adam: a method for stochastic optimization. ICLR, 2014. arXiv:1412.6980.
* Liwei Wang, Yin Li, and Svetlana Lazebnik. Learning deep structure-preserving image-text embeddings. CVPR, 2016.
* Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman,
and Phil Blunsom. Teaching machines to read and comprehend. In NIPS, 2015.
* Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja
Fidler. Skip-thought vectors. In NIPS, 2015.
* Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in
vector space. arXiv preprint arXiv:1301.3781, 2013.
* Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja
Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading
books. In ICCV, 2015.
* Marco Marelli, Luisa Bentivogli, Marco Baroni, Raffaella Bernardi, Stefano Menini, and Roberto Zamparelli. Semeval-2014 task 1: Evaluation of compositional distributional semantic models on full sentences through
semantic relatedness and textual entailment. SemEval-2014, 2014.
* Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect
to rating scales. In ACL, pages 115–124, 2005.
* Minqing Hu and Bing Liu. Mining and summarizing customer reviews. In Proceedings of the tenth ACM
SIGKDD international conference on Knowledge discovery and data mining, 2004.
* Bo Pang and Lillian Lee. A sentimental education: Sentiment analysis using subjectivity summarization based
on minimum cuts. In ACL, 2004.
* Janyce Wiebe, Theresa Wilson, and Claire Cardie. Annotating expressions of opinions and emotions in lan￾guage. Language resources and evaluation, 2005.
* K. Gregor, I. Danihelka, A. Graves, and D. Wierstra. DRAW: a recurrent neural network for image generation. arXiv:1502.04623, 2015.
* Hugo Larochelle and Iain Murray. The neural autoregressive distribution estimator. In AISTATS, volume 6,
page 622, 2011.
* Marcus Liwicki and Horst Bunke. Iam-ondb-an on-line english sentence database acquired from handwritten
text on a whiteboard. In ICDAR, 2005.
* Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

## Supplementary Material
### Application of layer normalization to each experiment

This section describes how layer normalization is applied to each of the papers’ experiments. For notation convenience, we define layer normalization as a function mapping LN : R

D → R

D with two set of adaptive parameters, gains α and biases β:

LN(z; α, β) = (z − µ) σ  α + β, (15) µ = 1

D

D

X i=1 zi , σ = v u u t 1

D

D

X i=1 (zi − µ) 2, (16) where, zi is the i th element of the vector z.

### Teaching machines to read and comprehend and handwriting sequence generation
The basic LSTM equations used for these experiment are given by:   ft it ot gt   = Whht−1 + Wxxt + b (17) ct = σ(ft)  ct−1 + σ(it)  tanh(gt) (18) ht = σ(ot)  tanh(ct) (19)

The version that incorporates layer normalization is modified as follows:   ft it ot gt   = LN(Whht−1; α1, β1) + LN(Wxxt; α2, β2) + b (20) ct = σ(ft)  ct−1 + σ(it)  tanh(gt) (21) ht = σ(ot)  tanh(LN(ct; α3, β3)) (22) where αi , βi are the additive and multiplicative parameters, respectively. Each αi is initialized to a vector of zeros and each βi is initialized to a vector of ones.

### Order embeddings and skip-thought
These experiments utilize a variant of gated recurrent unit which is defined as follows:  zt rt  = Whht−1 + Wxxt (23) ˆht = tanh(Wxt + σ(rt)  (Uht−1)) (24) ht = (1 − σ(zt))ht−1 + σ(zt) ˆht (25)

Layer normalization is applied as follows:  zt rt  = LN(Whht−1; α1, β1) + LN(Wxxt; α2, β2) (26) ˆht = tanh(LN(Wxt; α3, β3) + σ(rt)  LN(Uht−1; α4, β4)) (27) ht = (1 − σ(zt))ht−1 + σ(zt) ˆht (28) just as before, αi is initialized to a vector of zeros and each βi is initialized to a vector of ones. 13

### Modeling binarized MNIST using DRAW

The layer norm is only applied to the output of the LSTM hidden states in this experiment:

The version that incorporates layer normalization is modified as follows:   ft it ot gt   = Whht−1 + Wxxt + b (29) ct = σ(ft)  ct−1 + σ(it)  tanh(gt) (30) ht = σ(ot)  tanh(LN(ct; α, β)) (31) where α, β are the additive and multiplicative parameters, respectively. α is initialized to a vector of zeros and β is initialized to a vector of ones.

### Learning the magnitude of incoming weights

We now compare how gradient descent updates changing magnitude of the equivalent weights between the normalized GLM and original parameterization. The magnitude of the weights are explicitly parameterized using the gain parameter in the normalized model. Assume there is a gradient update that changes norm of the weight vectors by δg. We can project the gradient updates to the weight vector for the normal GLM. The KL metric, ie how much the gradient update changes the model prediction, for the normalized model depends only on the magnitude of the prediction error.

Specifically, under batch normalization: ds2 = 1 2 vec([0, 0, δg] > ) > F¯(vec([W, b, g] > ) vec([0, 0, δg] > ) = 1 2 δg > E x∼P (x) 

Cov[y | x] φ2  δg. (32)

Under layer normalization: ds2 = 1 2 vec([0, 0, δg] > ) > F¯(vec([W, b, g] > ) vec([0, 0, δg] > ) = 1 2 δg > 1 φ2 E x∼P (x)    

Cov(y1, y1 | x) (a1−µ) 2 σ2 · · · Cov(y1, yH | x) (a1−µ)(aH−µ) σ2 . . . . . . . . .

Cov(yH, y1 | x) (aH−µ)(a1−µ) σ2 · · · Cov(yH, yH | x) (aH−µ) 2 σ2     δg (33)

Under weight normalization: ds2 = 1 2 vec([0, 0, δg] > ) > F¯(vec([W, b, g] > ) vec([0, 0, δg] > ) = 1 2 δg > 1 φ2 E x∼P (x)     

Cov(y1, y1 | x) a 2 1 k w1k 2 2 · · · Cov(y1, yH | x) k w a1aH 1k 2k wHk 2 . . . . . . . . .

Cov(yH, y1 | x) aHa1 k wHk 2k w1k 2 · · · Cov(yH, yH | x) a 2

H k wHk 2 2      δg. (34)

Whereas, the KL metric in the standard GLM is related to its activities ai = wi > x, that is depended on both its current weights and input data. We project the gradient updates to the gain parameter δgi of the i th neuron to its weight vector as δgi wi k wik 2 in the standard GLM model: 1 2 vec([δgi wi k wik 2 , 0, δgj wj k wik 2 , 0]> ) > F([wi > , bi , wj > , bj ] > ) vec([δgi wi k wik 2 , 0, δgj wj k wjk 2 , 0]> ) = δgiδgj 2φ2 E x∼P (x) 

Cov(yi , yj | x) aiaj k wik 2k wjk 2  (35)

The batch normalized and layer normalized models are therefore more robust to the scaling of the input and its parameters than the standard model. 14
