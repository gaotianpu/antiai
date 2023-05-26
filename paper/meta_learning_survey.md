# Meta-Learning in Neural Networks: A Survey
https://arxiv.org/abs/2004.05439

## Abstract
The field of meta-learning, or learning-to-learn, has seen a dramatic rise in interest in recent years. Contrary to conventional approaches to AI where tasks are solved from scratch using a fixed learning algorithm, meta-learning aims to improve the learning algorithm itself, given the experience of multiple learning episodes. This paradigm provides an opportunity to tackle many conventional challenges of deep learning, including data and computation bottlenecks, as well as generalization. This survey describes the contemporary meta-learning landscape. We first discuss definitions of meta-learning and position it with respect to related fields, such as transfer learning and hyperparameter optimization. We then propose a new taxonomy that provides a more comprehensive breakdown of the space of meta-learning methods today. We survey promising applications and successes of meta-learning such as few-shot learning and reinforcement learning. Finally, we discuss outstanding challenges and promising areas for future research.

近年来，元学习(或称学习如何学习)领域的兴趣急剧上升。与使用固定学习算法从头开始解决任务的传统人工智能方法相反，元学习旨在改进学习算法本身，考虑到多个学习阶段的经验。这种范式为解决深度学习的许多传统挑战提供了机会，包括数据和计算瓶颈以及泛化。这项调查描述了当代元学习环境。我们首先讨论元学习的定义，并将其与相关领域(如迁移学习和超参数优化)进行定位。然后，我们提出了一种新的分类法，它提供了当今元学习方法空间的更全面细分。我们调查了元学习的有前景的应用和成功，如少样本学习和强化学习。最后，我们讨论了突出的挑战和未来研究的前景。

Index Terms—Meta-Learning, Learning-to-Learn, Few-Shot Learning, Transfer Learning, Neural Architecture Search 

## 1 INTRODUCTION
Contemporary machine learning models are typically trained from scratch for a specific task using a fixed learning algorithm designed by hand. Deep learning-based approaches specifically have seen great successes in a variety of fields [1]–[3]. However there are clear limitations [4]. For example, successes have largely been in areas where vast quantities of data can be collected or simulated, and where huge compute resources are available. This excludes many applications where data is intrinsically rare or expensive [5], or compute resources are unavailable [6].

当代的机器学习模型通常使用手工设计的固定学习算法从零开始针对特定任务进行训练。基于深度学习的方法在各个领域都取得了巨大成功[1]–[3]。然而，存在明显的限制[4]。例如，成功的主要领域是可以收集或模拟大量数据的领域，以及可以使用大量计算资源的领域。这排除了许多应用程序，这些应用程序的数据本质上是稀有或昂贵的[5]，或者计算资源不可用[6]。

Meta-learning provides an alternative paradigm where a machine learning model gains experience over multiple learning episodes – often covering a distribution of related tasks – and uses this experience to improve its future learning performance. This ‘learning-to-learn’ [7] can lead to a variety of benefits such as data and compute efficiency, and it is better aligned with human and animal learning [8], where learning strategies improve both on a lifetime and evolutionary timescales [8]–[10].

元学习提供了一种替代范例，其中机器学习模型在多个学习阶段获得经验——通常涵盖相关任务的分布——并使用这种经验来提高其未来的学习性能。这种“学会学习”[7]可以带来各种好处，如数据和计算效率，并且它与人类和动物学习更为一致[8]，在人类和动物的学习中，学习策略在终身和进化时间尺度上都有所改进[8]-[10]。

Historically, the success of machine learning was driven by the choice of hand-engineered features [11], [12]. Deep learning realised the promise of joint feature and model learning [13], providing a huge improvement in performance for many tasks [1], [3]. Meta-learning in neural networks can be seen as aiming to provide the next step of integrating joint feature, model, and algorithm learning.

历史上，机器学习的成功是由手工设计特征的选择所驱动的[11]，[12]。深度学习实现了联合特征和模型学习的承诺[13]，为许多任务的性能提供了巨大的改进[1]，[3]。神经网络中的元学习可以被视为旨在提供下一步整合联合特征、模型和算法学习。

Neural network meta-learning has a long history [7], [14], [15]. However, its potential as a driver to advance the frontier of the contemporary deep learning industry has led to an explosion of recent research. In particular metalearning has the potential to alleviate many of the main criticisms of contemporary deep learning [4], for instance by improving data efficiency, knowledge transfer and unsupervised learning. Meta-learning has proven useful both in multi-task scenarios where task-agnostic knowledge is extracted from a family of tasks and used to improve learning of new tasks from that family [7], [16]; and single-task scenarios where a single problem is solved repeatedly and improved over multiple episodes [17]–[19]. Successful applications have been demonstrated in areas spanning few-shot image recognition [16], [20], unsupervised learning [21], data efficient [22], [23] and self-directed [24] reinforcement learning (RL), hyperparameter optimization [17], and neural architecture search (NAS) [18], [25], [26].

神经网络元学习有着悠久的历史[7]，[14]，[15]。然而，它作为推动当代深度学习行业前沿发展的潜在力量，导致了最近研究的爆炸式增长。特别是，元学习有可能缓解当代深度学习的许多主要批评[4]，例如通过提高数据效率、知识迁移和无监督学习。元学习已被证明在多任务场景中是有用的，其中任务未知知识是从一系列任务中提取的，并用于改进对该系列新任务的学习[7]，[16]; 以及单个任务场景，其中单个问题被重复解决并在多个事件中得到改进[17]-[19]。成功的应用已经在包括少样本图像识别[16]、[20]、无监督学习[21]、数据高效[22]、[23]和自主[24]强化学习(RL)、超参数优化[17]和神经架构搜索(NAS)[18]、[25]、[26]等领域得到了证明。

T. Hospedales is with Samsung AI Centre, Cambridge and University of Edinburgh. A. Antoniou, P. Micaelli and Storkey are with University of Edinburgh. Email: {t.hospedales,a.antoniou,paul.micaelli,a.storkey}@ed.ac.uk. 

Many perspectives on meta-learning can be found in the literature, in part because different communities use the term differently. Thrun [7] operationally defines learning-tolearn as occurring when a learner’s performance at solving tasks drawn from a given task family improves with respect to the number of tasks seen. (cf., conventional machine learning performance improves as more data from a single task is seen). This perspective [27]–[29] views meta-learning as a tool to manage the ‘no free lunch’ theorem [30] and improve generalization by searching for the algorithm (inductive bias) that is best suited to a given problem, or problem family. However, this definition can include transfer, multitask, feature-selection, and model-ensemble learning, which are not typically considered as meta-learning today. Another usage of meta-learning [31] deals with algorithm selection based on dataset features, and becomes hard to distinguish from automated machine learning (AutoML) [32], [33].

关于元学习的许多观点可以在文献中找到，部分原因是不同的社区对该术语的使用不同。Thrun[7]在操作上将学习定义为当学习者在解决从给定任务系列中提取的任务时的表现相对于所看到的任务数量有所改善时发生的学习。(参见，当看到来自单个任务的更多数据时，传统的机器学习性能会提高)。这种观点[27]–[29]将元学习视为一种工具，可以管理“没有免费午餐”定理[30]，并通过搜索最适合给定问题或问题族的算法(归纳偏差)来改进泛化。然而，这个定义可以包括迁移、多任务、特征选择和模型集成学习，这些在今天通常不被视为元学习。元学习的另一种用法[31]涉及基于数据集特征的算法选择，并且很难与自动机器学习(AutoML)[32]，[33]区分开来。

In this paper, we focus on contemporary neural-network meta-learning. We take this to mean algorithm learning as per [27], [28], but focus specifically on where this is achieved by end-to-end learning of an explicitly defined objective function (such as cross-entropy loss). Additionally we consider single-task meta-learning, and discuss a wider variety of (meta) objectives such as robustness and compute efficiency.

在本文中，我们关注当代神经网络元学习。根据[27]，[28]，我们认为这意味着算法学习，但特别关注通过明确定义的目标函数的端到端学习(如交叉熵损失)实现这一点的地方。此外，我们考虑了单任务元学习，并讨论了更广泛的(元)目标，如稳健性和计算效率。

This paper thus provides a unique, timely, and up-todate survey of the rapidly growing area of neural network meta-learning. In contrast, previous surveys are rather out of date and/or focus on algorithm selection for data mining [27], [31], [34], [35], AutoML [32], [33], or particular applications of meta-learning such as few-shot learning [36] or neural architecture search [37]. arXiv:2004.05439v2 [cs.LG] 7 Nov 2020 2

因此，本文对快速增长的神经网络元学习领域进行了独特、及时和最新的调查。相比之下，以前的调查相当过时，和/或侧重于数据挖掘的算法选择[27]、[31]、[34]、[35]、AutoML[32]、[33]，或元学习的特定应用，如少量学习[36]或神经架构搜索[37]。arXiv:2004.05439v2[cs.LG]2020年11月7日2

We address both meta-learning methods and applications. We first introduce meta-learning through a high-level problem formalization that can be used to understand and position work in this area. We then provide a new taxonomy in terms of meta-representation, meta-objective and metaoptimizer. This framework provides a design-space for developing new meta learning methods and customizing them for different applications. We survey several popular and emerging application areas including few-shot, reinforcement learning, and architecture search; and position metalearning with respect to related topics such as transfer and multi-task learning. We conclude by discussing outstanding challenges and areas for future research. 

我们同时研究元学习方法和应用。我们首先通过一个高层次的问题形式化来引入元学习，可以用来理解和定位这一领域的工作。然后，我们在元表示、元目标和元优化器方面提供了一种新的分类。该框架为开发新的元学习方法并为不同的应用程序定制它们提供了设计空间。我们调查了几个流行和新兴的应用领域，包括少样本、强化学习和架构搜索; 并将元学习定位于诸如迁移和多任务学习等相关主题。最后，我们讨论了悬而未决的挑战和未来研究的领域。

## 2 BACKGROUND
Meta-learning is difficult to define, having been used in various inconsistent ways, even within contemporary neuralnetwork literature. In this section, we introduce our definition and key terminology, and then position meta-learning with respect to related topics.

元学习很难定义，即使在当代神经网络文献中，元学习也被以各种不一致的方式使用。在本节中，我们将介绍我们的定义和关键术语，然后根据相关主题定位元学习。

Meta-learning is most commonly understood as learning to learn, which refers to the process of improving a learning algorithm over multiple learning episodes. In contrast, conventional ML improves model predictions over multiple data instances. During base learning, an inner (or lower/base) learning algorithm solves a task such as image classification [13], defined by a dataset and objective. During meta-learning, an outer (or upper/meta) algorithm updates the inner learning algorithm such that the model it learns improves an outer objective. For instance this objective could be generalization performance or learning speed of the inner algorithm. Learning episodes of the base task, namely (base algorithm, trained model, performance) tuples, can be seen as providing the instances needed by the outer algorithm to learn the base learning algorithm.

元学习通常被理解为学习，指的是在多个学习阶段中改进学习算法的过程。相比之下，传统的ML在多个数据实例上改进了模型预测。在基础学习期间，内部(或较低/基础)学习算法解决了由数据集和目标定义的任务，如图像分类[13]。在元学习期间，外部(或上层/元)算法更新内部学习算法，使得其学习的模型改进了外部目标。例如，这个目标可以是内部算法的泛化性能或学习速度。基本任务的学习片段，即(基本算法、训练模型、性能)元组，可以被视为提供外部算法学习基本学习算法所需的实例。

As defined above, many conventional algorithms such as random search of hyper-parameters by cross-validation could fall within the definition of meta-learning. The salient characteristic of contemporary neural-network metalearning is an explicitly defined meta-level objective, and endto-end optimization of the inner algorithm with respect to this objective. Often, meta-learning is conducted on learning episodes sampled from a task family, leading to a base learning algorithm that performs well on new tasks sampled from this family. However, in a limiting case all training episodes can be sampled from a single task. In the following section, we introduce these notions more formally.

如上所述，许多传统算法，例如通过交叉验证对超参数的随机搜索，可能属于元学习的定义。当代神经网络元学习的显著特点是明确定义的元级目标，并针对该目标对内部算法进行端到端优化。通常，元学习是在从任务族中采样的学习片段上进行的，从而产生了一种基本学习算法，该算法在从该族中采样到的新任务上表现良好。然而，在有限的情况下，可以从单个任务中对所有训练集进行采样。在下一节中，我们将更正式地介绍这些概念。

### 2.1 Formalizing Meta-Learning
Conventional Machine Learning. In conventional supervised machine learning, we are given a training dataset D = {(x1, y1), . . . ,(xN , yN )}, such as (input image, output label) pairs. We can train a predictive model yˆ = fθ(x) parameterized by θ, by solving: 

传统机器学习。在传统的监督机器学习中，我们得到一个训练数据集D={(x1，y1)，…，(xN，yN)}，例如(输入图像，输出标签)对。我们可以通过求解来训练由θ参数化的预测模型yˆ=fθ(x)：

θ∗ = arg min θ L(D; θ, ω) (1) 

where L is a loss function that measures the error between true labels and those predicted by fθ(·). The conditioning on ω denotes the dependence of this solution on assumptions about ‘how to learn’, such as the choice of optimizer for θ or function class for f. Generalization is then measured by evaluating a number of test points with known labels.

其中L是一个损失函数，用于测量真实标签与fθ(·)预测的标签之间的误差。ω的条件表示该解决方案对“如何学习”假设的依赖性，例如θ的优化器或f的函数类的选择。然后，通过评估具有已知标签的多个测试点来测量泛化。

The conventional assumption is that this optimization is performed from scratch for every problem D; and that ω is pre-specified. However, the specification of ω can drastically affect performance measures like accuracy or data efficiency. Meta-learning seeks to improve these measures by learning the learning algorithm itself, rather than assuming it is prespecified and fixed. This is often achieved by revisiting the first assumption above, and learning from a distribution of tasks rather than from scratch.

传统的假设是，对每个问题D从头开始进行优化; ω是预先指定的。然而，ω的规格会极大地影响性能指标，如准确性或数据效率。元学习试图通过学习学习算法本身来改进这些度量，而不是假设它是预先指定和固定的。这通常是通过重温上面的第一个假设，从任务分布中学习而不是从头开始来实现的。

Meta-Learning: Task-Distribution View A common view of meta-learning is to learn a general purpose learning algorithm that can generalize across tasks, and ideally enable each new task to be learned better than the last. We can evaluate the performance of ω over a distribution of tasks p(T ). Here we loosely define a task to be a dataset and loss function T = {D,L}. Learning how to learn thus becomes 

元学习：任务分布视图元学习的一个常见观点是学习一种通用学习算法，该算法可以在任务之间进行泛化，理想情况下可以使每个新任务比上一个更好地学习。我们可以评估ω在任务p(T)分布上的性能。这里，我们松散地将任务定义为数据集和损失函数T={D，L}。因此，学习如何学习变得

min ω E T ∼p(T )L(D; ω) (2) 

where L(D; ω) measures the performance of a model trained using ω on dataset D. ‘How to learn’, i.e. ω, is often referred to as across-task knowledge or meta-knowledge.

其中，L(D; ω)测量在数据集D上使用ω训练的模型的性能。“如何学习”，即ω，通常被称为跨任务知识或元知识。

To solve this problem in practice, we often assume access to a set of source tasks sampled from p(T ). Formally, we denote the set of M source tasks used in the meta-training stage as Dsource = {(Dtrain source, Dval source)(i)}Mi=1 where each task has both training and validation data. Often, the source train and validation datasets are respectively called support and query sets. The meta-training step of ‘learning how to learn’ can be written as: 

为了在实践中解决这个问题，我们通常假设访问从p(T)采样的一组源任务。形式上，我们将元训练阶段中使用的M个源任务集表示为Dsource={(Dtrain source，Dval source)(i)}Mi=1，其中每个任务都有训练和验证数据。通常，源数据集和验证数据集分别称为支持集和查询集。“学习如何学习”的元训练步骤可以写成：

ω∗ = arg max ω log p(ω|Dsource) (3)

Now we denote the set of Q target tasks used in the meta-testing stage as Dtarget = {(Dtrain target, Dtest target)(i)}Qi=1 where each task has both training and test data. In the metatesting stage we use the learned meta-knowledge ω∗ to train the base model on each previously unseen target task i: 

现在我们将元测试阶段使用的Q个目标任务集表示为Dtarget={(Dtrain target，Dtest target)(i)}Qi=1，其中每个任务都有训练和测试数据。在元测试阶段，我们使用学习到的元知识ω∗ 在每个先前未看到的目标任务i上训练基础模型：

θ∗ (i) = arg max θ log p(θ|ω∗, D train (i) target ) (4)

In contrast to conventional learning in Eq. 1, learning on the training set of a target task i now benefits from metaknowledge ω∗ about the algorithm to use. This could be an estimate of the initial parameters [16], or an entire learning model [38] or optimization strategy [39]. We can evaluate the accuracy of our meta-learner by the performance of θ∗ (i) on the test split of each target task D test (i) target .

与等式1中的传统学习相比，在目标任务i的训练集上学习现在受益于元知识ω∗ 关于要使用的算法。这可以是初始参数的估计[16]，也可以是整个学习模型[38]或优化策略[39]。我们可以通过θ∗ (i) 在每个目标任务D测试(i)目标的测试分割上。

This setup leads to analogies of conventional underfitting and overfitting: meta-underfitting and meta-overfitting. In particular, meta-overfitting is an issue whereby the metaknowledge learned on the source tasks does not generalize to the target tasks. It is relatively common, especially in the case where only a small number of source tasks are available. It can be seen as learning an inductive bias ω that constrains the hypothesis space of θ too tightly around solutions to the source tasks. 

这种设置导致类似于传统的欠拟合和过拟合：元欠拟合和元过拟合。特别是，元过拟合是一个问题，即在源任务上学习的元知识不能推广到目标任务。这是相对常见的，尤其是在只有少量源任务可用的情况下。它可以被看作是学习一种归纳偏差ω，它将θ的假设空间限制得太紧，围绕源任务的解。

Meta-Learning: Bilevel Optimization View. The previous discussion outlines the common flow of meta-learning in a multiple task scenario, but does not specify how to solve the meta-training step in Eq. 3. This is commonly done by casting the meta-training step as a bilevel optimization problem. While this picture is arguably only accurate for the optimizer-based methods (see section 3.1), it is helpful to visualize the mechanics of meta-learning more generally. Bilevel optimization [40] refers to a hierarchical optimization problem, where one optimization contains another optimization as a constraint [17], [41]. Using this notation, meta-training can be formalised as follows: 

元学习：双层优化视图。前面的讨论概述了多任务场景中的元学习的常见流程，但没有指定如何解决等式3中的元训练步骤。这通常通过将元训练步骤转化为两级优化问题来实现。虽然这张图可以说只适用于基于优化器的方法(参见第3.1节)，但它有助于更全面地可视化元学习的机制。双层优化[40]指的是分层优化问题，其中一个优化包含另一个优化作为约束[17]，[41]。使用此符号，元训练可以如下形式化：

ω∗ = arg min ω MXi=1 L meta(θ∗ (i)(ω), ω, D val (i) source) (5) 

s.t. θ∗(i)(ω) = arg min θ L task(θ, ω, D train (i) source ) (6) 

where L meta and L task refer to the outer and inner objectives respectively, such as cross entropy in the case of few-shot classification. Note the leader-follower asymmetry between the outer and inner levels: the inner level optimization Eq. 6 is conditional on the learning strategy ω defined by the outer level, but it cannot change ω during its training.

其中L元和L任务分别指外部和内部目标，例如在少样本分类的情况下的交叉熵。请注意外部和内部级别之间的领导者-追随者不对称：内部级别优化等式6以外部级别定义的学习策略ω为条件，但在训练期间不能改变ω。

Here ω could indicate an initial condition in non-convex optimization [16], a hyper-parameter such as regularization strength [17], or even a parameterization of the loss function to optimize L task [42]. Section 4.1 discusses the space of choices for ω in detail. The outer level optimization learns ω such that it produces models θ∗ (i)(ω) that perform well on their validation sets after training. Section 4.2 discusses how to optimize ω in detail. In Section 4.3 we consider what L meta can measure, such as validation performance, learning speed or model robustness.

这里，ω可以表示非凸优化中的初始条件[16]、超参数(如正则化强度[17])，甚至是损失函数的参数化，以优化L任务[42]。第4.1节详细讨论了ω的选择空间。外层优化学习ω，从而生成模型θ∗ (i) (ω)在训练后的验证集上表现良好。第4.2节详细讨论了如何优化ω。在第4.3节中，我们考虑了L meta可以测量的内容，如验证性能、学习速度或模型稳健性。

Finally, we note that the above formalization of metatraining uses the notion of a distribution over tasks. While common in the meta-learning literature, it is not a necessary condition for meta-learning. More formally, if we are given a single train and test dataset (M = Q = 1), we can split the training set to get validation data such that Dsource = (Dtrain source, Dval source) for meta-training, and for meta-testing we can use Dtarget = (Dtrain source ∪ Dval source, Dtest target). We still learn ω over several episodes, and different train-val splits are usually used during meta-training.

最后，我们注意到元训练的上述形式化使用了任务分布的概念。虽然在元学习文献中很常见，但它不是元学习的必要条件。更正式地说，如果给我们一个训练和测试数据集(M=Q=1)，我们可以拆分训练集以获得验证数据，这样Dsource=(Dtrain source，Dval source)用于元训练，而对于元测试，我们可以使用Dtarget=(Dstrain source∪ Dval源、Dtest目标)。我们仍然在几集中学习ω，在元训练中通常使用不同的训练值分割。

Meta-Learning: Feed-Forward Model View. As we will see, there are a number of meta-learning approaches that synthesize models in a feed-forward manner, rather than via an explicit iterative optimization as in Eqs. 5-6 above. While they vary in their degree of complexity, it can be instructive to understand this family of approaches by instantiating the abstract objective in Eq. 2 to define a toy example for metatraining linear regression [43]. 

元学习：前馈模型视图。正如我们将看到的，有许多元学习方法以前馈方式合成模型，而不是通过方程中的显式迭代优化。上面的5-6。尽管它们的复杂程度各不相同，但通过实例化等式2中的抽象目标来定义元训练线性回归的玩具样本来理解这一系列方法可能会很有帮助[43]。

min ω E T ∼p(T ) (Dtr ,Dval)∈T X (x,y)∈Dval h (xT gω(D tr) − y)2i (7)

Here we meta-train by optimizing over a distribution of tasks. For each task a train and validation set is drawn. The train set Dtr is embedded [44] into a vector gω which defines the linear regression weights to predict examples x from the validation set. Optimizing Eq. 7 ‘learns to learn’ by training the function gω to map a training set to a weight vector. Thus gω should provide a good solution for novel metatest tasks T te drawn from p(T ). Methods in this family vary in the complexity of the predictive model g used, and how the support set is embedded [44] (e.g., by pooling, CNN or RNN). These models are also known as amortized [45] because the cost of learning a new task is reduced to a feed-forward operation through gω(·), with iterative optimization already paid for during meta-training of ω.

在这里，我们通过优化任务分布来进行元训练。为每个任务绘制一个序列和验证集。将训练集Dtr嵌入[44]到向量gω中，该向量定义了线性回归权重，以从验证集预测样本x。通过训练函数gω将训练集映射到权重向量，优化等式7“学会学习”。因此，gω应该为从p(T)中提取的新元测试任务T te提供一个很好的解决方案。该家族中的方法在所使用的预测模型g的复杂性以及如何嵌入支持集方面各不相同[44](例如，通过池化、CNN或RNN)。这些模型也被称为摊销[45]，因为通过gω(·)将学习新任务的成本降低为前馈操作，在ω的元训练期间已经支付了迭代优化费用。

### 2.2 Historical Context of Meta-Learning
Meta-learning and learning-to-learn first appear in the literature in 1987 [14]. J. Schmidhuber introduced a family of methods that can learn how to learn, using self-referential learning. Self-referential learning involves training neural networks that can receive as inputs their own weights and predict updates for said weights. Schmidhuber proposed to learn the model itself using evolutionary algorithms.

元学习和学习于1987年首次出现在文献中[14]。J、 Schmidhuber介绍了一系列方法，可以通过自我参照学习来学习如何学习。自我参照学习涉及训练神经网络，神经网络可以接收自己的权重作为输入，并预测所述权重的更新。Schmidhuber建议使用进化算法来学习模型本身。

Meta-learning was subsequently extended to multiple areas. Bengio et al. [46], [47] proposed to meta-learn biologically plausible learning rules. Schmidhuber et al.continued to explore self-referential systems and meta-learning [48], [49]. S. Thrun et al. took care to more clearly define the term learning to learn in [7] and introduced initial theoretical justifications and practical implementations. Proposals for training meta-learning systems using gradient descent and backpropagation were first made in 1991 [50] followed by more extensions in 2001 [51], [52], with [27] giving an overview of the literature at that time. Meta-learning was used in the context of reinforcement learning in 1995 [53], followed by various extensions [54], [55].

元学习随后扩展到多个领域。Bengioet al [46]，[47]提出元学习生物学上合理的学习规则。Schmidhuberet al 继续探索自我参照系统和元学习[48]，[49]。S、 Thrunet al 在[7]中更明确地定义了学习这一术语，并介绍了最初的理论依据和实际实施。1991年首次提出了使用梯度下降和反向传播训练元学习系统的建议[50]，随后在2001年进行了更多扩展[51]，[52]，[27]对当时的文献进行了概述。1995年，元学习被用于强化学习[53]，随后进行了各种扩展[54]，[55]。

### 2.3 Related Fields
Here we position meta-learning against related areas whose relation to meta-learning is often a source of confusion.

在这里，我们将元学习与相关领域进行对比，这些领域与元学习的关系常常是一个混乱的根源。

Transfer Learning (TL).  TL [34], [56] uses past experience from a source task to improve learning (speed, data efficiency, accuracy) on a target task. TL refers both to this problem area and family of solutions, most commonly parameter transfer plus optional fine tuning [57] (although there are numerous other approaches [34]).

迁移学习(TL)。TL[34]，[56]使用源任务的过去经验来提高目标任务的学习(速度、数据效率、准确性)。TL指的是这个问题领域和解决方案系列，最常见的是参数迁移和可选微调[57](尽管还有许多其他方法[34])。


In contrast, meta-learning refers to a paradigm that can be used to improve TL as well as other problems. In TL the prior is extracted by vanilla learning on the source task without the use of a meta-objective. In meta-learning, the corresponding prior would be defined by an outer optimization that evaluates the benefit of the prior when learn a new task, as illustrated by MAML [16]. More generally, meta-learning deals with a much wider range of metarepresentations than solely model parameters (Section 4.1).

相比之下，元学习是指一种可以用来改善TL以及其他问题的范式。在TL中，通过对源任务的普通学习提取先验，而不使用元目标。在元学习中，对应的先验将由外部优化来定义，该外部优化在学习新任务时评估先验的益处，如MAML[16]所示。更一般地说，元学习处理的元表示范围比模型参数更广(第4.1节)。

Domain Adaptation (DA) and Domain Generalization (DG). Domain-shift refers to the situation where source and target problems share the same objective, but the input distribution of the target task is shifted with respect to the source task [34], [58], reducing model performance. DA is a variant of transfer learning that attempts to alleviate this issue by adapting the source-trained model using sparse or unlabeled data from the target. DG refers to methods to train 4 a source model to be robust to such domain-shift without further adaptation. Many knowledge transfer methods have been studied [34], [58] to boost target domain performance. However, as for TL, vanilla DA and DG don’t use a metaobjective to optimize ‘how to learn’ across domains. Meanwhile, meta-learning methods can be used to perform both DA [59] and DG [42] (see Sec. 5.8).

域自适应(DA)和域泛化(DG)。领域迁移是指源和目标问题共享相同目标，但目标任务的输入分布相对于源任务发生了迁移[34]，[58]，从而降低了模型性能的情况。DA是迁移学习的一种变体，它试图通过使用来自目标的稀疏或未标注数据来调整源训练模型来缓解这一问题。DG指的是将源模型训练为对这种域偏移稳健而无需进一步自适应的方法。已经研究了许多知识迁移方法[34]、[58]，以提高目标领域的性能。然而，对于TL，普通DA和DG没有使用元目标来优化跨领域的“如何学习”。同时，元学习方法可用于执行DA[59]和DG[42](参见第5.8节)。

Continual learning (CL)  Continual or lifelong learning [60]–[62] refers to the ability to learn on a sequence of tasks drawn from a potentially non-stationary distribution, and in particular seek to do so while accelerating learning new tasks and without forgetting old tasks. Similarly to metalearning, a task distribution is considered, and the goal is partly to accelerate learning of a target task. However most continual learning methodologies are not meta-learning methodologies since this meta objective is not solved for explicitly. Nevertheless, meta-learning provides a potential framework to advance continual learning, and a few recent studies have begun to do so by developing meta-objectives that encode continual learning performance [63]–[65].

持续学习(CL)持续或终身学习[60]-[62]是指从潜在的非平稳分布中提取一系列任务的学习能力，尤其是在加速学习新任务的同时不忘记旧任务。与元学习类似，考虑任务分配，目标部分是加速目标任务的学习。然而，大多数持续学习方法不是元学习方法，因为这个元目标并没有明确解决。然而，元学习为推进持续学习提供了一个潜在的框架，最近的一些研究已经开始通过制定编码持续学习绩效的元目标来实现这一目标[63]–[65]。

Multi-Task Learning (MTL) aims to jointly learn several related tasks, to benefit from regularization due to parameter sharing and the diversity of the resulting shared representation [66]–[68], as well as compute/memory savings. Like TL, DA, and CL, conventional MTL is a singlelevel optimization without a meta-objective. Furthermore, the goal of MTL is to solve a fixed number of known tasks, whereas the point of meta-learning is often to solve unseen future tasks. Nonetheless, meta-learning can be brought in to benefit MTL, e.g. by learning the relatedness between tasks [69], or how to prioritise among multiple tasks [70].

多任务学习(MTL)旨在联合学习多个相关任务，以受益于参数共享带来的正则化和结果共享表示的多样性[66]-[68]，以及计算/内存节省。与TL、DA和CL一样，传统的MTL是一个没有元目标的单级优化。此外，MTL的目标是解决固定数量的已知任务，而元学习的目的通常是解决看不见的未来任务。尽管如此，可以引入元学习来帮助MTL，例如，通过学习任务之间的相关性[69]，或者如何在多个任务之间进行优先排序[70]。

Hyperparameter Optimization (HO) is within the remit of meta-learning, in that hyperparameters like learning rate or regularization strength describe ‘how to learn’. Here we include HO tasks that define a meta objective that is trained end-to-end with neural networks, such as gradient-based hyperparameter learning [69], [71] and neural architecture search [18]. But we exclude other approaches like random search [72] and Bayesian Hyperparameter Optimization [73], which are rarely considered to be meta-learning.

超参数优化(HO)属于元学习的范畴，因为学习率或正则化强度等超参数描述了“如何学习”。这里我们包括定义元目标的HO任务，该元目标通过神经网络进行端到端训练，如基于梯度的超参数学习[69]、[71]和神经架构搜索[18]。但我们排除了其他方法，如随机搜索[72]和贝叶斯超参数优化[73]，它们很少被认为是元学习。

Hierarchical Bayesian Models (HBM) involve Bayesian learning of parameters θ under a prior p(θ|ω). The prior is written as a conditional density on some other variable ω which has its own prior p(ω). Hierarchical Bayesian models feature strongly as models for grouped data D = {Di|i = 1, 2, . . . , M}, where each group i has its own θi . The full model is h Q Mi=1 p(Di|θi)p(θi|ω)i p(ω). The levels of hierarchy can be increased further; in particular ω can itself be parameterized, and hence p(ω) can be learnt. Learning is usually full-pipeline, but using some form of Bayesian marginalisation to compute the posterior over ω: P(ω|D) ∼ p(ω) Q Mi=1 R dθip(Di|θi)p(θi|ω). The ease of doing the marginalisation depends on the model: in some (e.g. Latent Dirichlet Allocation [74]) the marginalisation is exact due to the choice of conjugate exponential models, in others (see e.g. [75]), a stochastic variational approach is used to calculate an approximate posterior, from which a lower bound to the marginal likelihood is computed.

分层贝叶斯模型(HBM)涉及先验p(θ|ω)下参数θ的贝叶斯学习。先验被写成另一个变量ω的条件密度，它有自己的先验p(ω)。分层贝叶斯模型作为分组数据D={Di|i=1，2，…，M}的模型具有很强的特征，其中每个组i都有自己的θi。完整的模型是hQMi=1p(Di|θi)p(θi|ω)i p(ω)。层级可以进一步提高; 特别是ω本身可以参数化，因此可以学习p(ω)。学习通常是完整的流水线，但使用某种形式的贝叶斯边缘化来计算ω：P(ω|D)的后验∼ p(ω)Q Mi=1 R dθip(Di|θi)p(θi|ω)。边缘化的容易程度取决于模型：在某些情况下(例如，潜在狄利克雷分配[74])，由于选择了共轭指数模型，边缘化是精确的，而在其他情况下(参见例如[75])，使用随机变分法计算近似后验，由此计算出边缘似然的下限。

Bayesian hierarchical models provide a valuable viewpoint for meta-learning, by providing a modeling rather than an algorithmic framework for understanding the metalearning process. In practice, prior work in HBMs has typically focused on learning simple tractable models θ while most meta-learning work considers complex inner-loop learning processes, involving many iterations. Nonetheless, some meta-learning methods like MAML [16] can be understood through the lens of HBMs [76].

贝叶斯层次模型为元学习提供了一个有价值的观点，它为理解元学习过程提供了一种建模而非算法框架。在实践中，HBM的先前工作通常侧重于学习简单的可处理模型θ，而大多数元学习工作考虑复杂的内环学习过程，涉及许多迭代。尽管如此，一些元学习方法，如MAML[16]，可以通过HBMs[76]的视角来理解。

AutoML: AutoML [31]–[33] is a rather broad umbrella for approaches aiming to automate parts of the machine learning process that are typically manual, such as data preparation, algorithm selection, hyper-parameter tuning, and architecture search. AutoML often makes use of numerous heuristics outside the scope of meta-learning as defined here, and focuses on tasks such as data cleaning that are less central to meta-learning. However, AutoML sometimes makes use of end-to-end optimization of a meta-objective, so meta-learning can be seen as a specialization of AutoML. 

AutoML:AutoML[31]–[33]是一种相当广泛的方法，旨在自动化机器学习过程中通常是手动的部分，如数据准备、算法选择、超参数调整和架构搜索。AutoML经常使用这里定义的元学习范围之外的许多启发式方法，并专注于数据清理等对元学习不太重要的任务。然而，AutoML有时会利用元目标的端到端优化，因此元学习可以被视为AutoML的一种专门化。

## 3 TAXONOMY
### 3.1 Previous Taxonomies
Previous [77], [78] categorizations of meta-learning methods have tended to produce a three-way taxonomy across optimization-based methods, model-based (or black box) methods, and metric-based (or non-parametric) methods.

以前的[77]、[78]元学习方法分类倾向于在基于优化的方法、基于模型(或黑盒)的方法和基于度量(或非参数)的方法之间产生三种分类。

Optimization. Optimization-based methods include those where the inner-level task (Eq. 6) is literally solved as an optimization problem, and focuses on extracting metaknowledge ω required to improve optimization performance. A famous example is MAML [16], which aims to learn the initialization ω = θ0, such that a small number of inner steps produces a classifier that performs well on validation data. This is also performed by gradient descent, differentiating through the updates of the base model. More elaborate alternatives also learn step sizes [79], [80] or train recurrent networks to predict steps from gradients [19], [39], [81]. Meta-optimization by gradient over long inner optimizations leads to several compute and memory challenges which are discussed in Section 6. A unified view of gradient-based meta learning expressing many existing methods as special cases of a generalized inner loop metalearning framework has been proposed [82].

优化。基于优化的方法包括将内部级任务(等式6)按字面意思解决为优化问题的方法，并侧重于提取提高优化性能所需的元知识ω。一个著名的例子是MAML[16]，它的目标是学习初始化ω=θ0，这样少量的内部步骤就可以生成一个在验证数据上表现良好的分类器。这也通过梯度下降来执行，通过基础模型的更新进行区分。更详细的替代方案还学习步长[79]、[80]或训练递归网络以根据梯度预测步长[19]、[39]、[81]。通过长内部优化的梯度进行的元优化导致了第6节中讨论的几个计算和内存挑战。已经提出了基于梯度的元学习的统一观点，将许多现有方法表示为广义内部循环元学习框架的特例[82]。

Black Box / Model-based. In model-based (or black-box) methods the inner learning step (Eq. 6, Eq. 4) is wrapped up in the feed-forward pass of a single model, as illustrated in Eq. 7. The model embeds the current dataset D into activation state, with predictions for test data being made based on this state. Typical architectures include recurrent networks [39], [51], convolutional networks [38] or hypernetworks [83], [84] that embed training instances and labels of a given task to define a predictor for test samples. In this case all the inner-level learning is contained in the activation states of the model and is entirely feed-forward. Outerlevel learning is performed with ω containing the CNN, RNN or hypernetwork parameters. The outer and innerlevel optimizations are tightly coupled as ω and D directly specify θ. Memory-augmented neural networks [85] use an explicit storage buffer and can be seen as a model-based 5 algorithm [86], [87]. Compared to optimization-based approaches, these enjoy simpler optimization without requiring second-order gradients. However, it has been observed that model-based approaches are usually less able to generalize to out-of-distribution tasks than optimization-based methods [88]. Furthermore, while they are often very good at data efficient few-shot learning, they have been criticised for being asymptotically weaker [88] as they struggle to embed a large training set into a rich base model.

基于黑匣子/模型。在基于模型(或黑盒)的方法中，内部学习步骤(等式6、等式4)被包裹在单个模型的前馈过程中，如等式7所示。该模型将当前数据集D嵌入激活状态，并基于该状态对测试数据进行预测。典型的架构包括递归网络[39]、[51]、卷积网络[38]或超网络[83]、[84]，它们嵌入给定任务的训练实例和标签，以定义测试样本的预测器。在这种情况下，所有的内部学习都包含在模型的激活状态中，并且完全是前馈的。使用包含CNN、RNN或超网络参数的ω进行外部学习。外部和内部优化紧密耦合，因为ω和D直接指定θ。记忆增强神经网络[85]使用显式存储缓冲器，可以看作是基于模型的5算法[86]，[87]。与基于优化的方法相比，这些方法具有更简单的优化，而不需要二阶梯度。然而，已经观察到，与基于优化的方法相比，基于模型的方法通常不太能够推广到分布外任务[88]。此外，尽管他们通常非常擅长数据高效的少样本学习，但由于他们难以将大量训练集嵌入到丰富的基础模型中，因此他们被批评为渐近较弱[88]。

Metric-Learning. Metric-learning or non-parametric algorithms are thus far largely restricted to the popular but specific few-shot application of meta-learning (Section 5.1.1). The idea is to perform non-parametric ‘learning’ at the inner (task) level by simply comparing validation points with training points and predicting the label of matching training points. In chronological order, this has been achieved with siamese [89], matching [90], prototypical [20], relation [91], and graph [92] neural networks. Here outer-level learning corresponds to metric learning (finding a feature extractor ω that represents the data suitably for comparison). As before ω is learned on source tasks, and used for target tasks.

指标学习。到目前为止，度量学习或非参数算法在很大程度上局限于流行但特定的元学习应用(第5.1.1节)。其想法是通过简单地比较验证点和训练点并预测匹配训练点的标签，在内部(任务)级别执行非参数“学习”。按时间顺序，这是通过siame[89]、匹配[90]、原型[20]、关系[91]和图[92]神经网络实现的。这里，外部级别学习对应于度量学习(找到一个特征提取器ω，它表示适合比较的数据)。如前所述，ω在源任务上学习，并用于目标任务。

Discussion. The common breakdown reviewed above does not expose all facets of interest and is insufficient to understand the connections between the wide variety of meta-learning frameworks available today. For this reason, we propose a new taxonomy in the following section.

讨论上面回顾的常见故障并没有暴露所有感兴趣的方面，也不足以理解当今各种元学习框架之间的联系。因此，我们在下一节中提出了一种新的分类法。

### 3.2 Proposed Taxonomy
We introduce a new breakdown along three independent axes. For each axis we provide a taxonomy that reflects the current meta-learning landscape.

我们沿着三个独立的轴引入了一个新的细分。对于每个轴，我们提供了反映当前元学习环境的分类法。

Meta-Representation (“What?”) The first axis is the choice of meta-knowledge ω to meta-learn. This could be anything from initial model parameters [16] to readable code in the case of program induction [93].

元表示(“什么？”)第一轴是元知识ω到元学习的选择。这可以是从初始模型参数[16]到程序归纳情况下的可读代码[93]的任何内容。

Meta-Optimizer (“How?”) The second axis is the choice of optimizer to use for the outer level during meta-training (see Eq. 5). The outer-level optimizer for ω can take a variety of forms from gradient-descent [16], to reinforcement learning [93] and evolutionary search [23].

元优化器(“如何？”)第二个轴是在元训练期间用于外部级别的优化器的选择(参见等式5)。ω的外层优化器可以采取多种形式，从梯度下降[16]到强化学习[93]和进化搜索[23]。

Meta-Objective (“Why?”) The third axis is the goal of meta-learning which is determined by choice of metaobjective L meta (Eq. 5), task distribution p(T ), and data- flow between the two levels. Together these can customize meta-learning for different purposes such as sample efficient few-shot learning [16], [38], fast many-shot optimization [93], [94], robustness to domain-shift [42], [95], label noise [96], and adversarial attack [97].

元目标(“为什么？”)第三轴是元学习的目标，它由元目标L元(等式5)、任务分布p(T)和两个级别之间的数据流的选择决定。总之，这些可以为不同的目的定制元学习，例如样本高效的少样本学习[16]、[38]、快速多样本优化[93]、[94]、对域偏移的稳健性[42]、[95]、标签噪声[96]和对抗性攻击[97]。

Together these axes provide a design-space for metalearning methods that can orient the development of new algorithms and customization for particular applications. Note that the base model representation θ isn’t included in this taxonomy, since it is determined and optimized in a way that is specific to the application at hand. 

这些轴一起为金属加工方法提供了设计空间，可以为特定应用的新算法和定制的开发提供方向。请注意，基本模型表示θ不包括在该分类中，因为它是以特定于当前应用程序的方式确定和优化的。

## 4 SURVEY: METHODOLOGIES
In this section we break down existing literature according to our proposed new methodological taxonomy.

在本节中，我们根据我们提出的新方法分类法对现有文献进行了分解。

### 4.1 Meta-Representation
Meta-learning methods make different choices about what meta-knowledge ω should be, i.e. which aspects of the learning strategy should be learned; and (by exclusion) which aspects should be considered fixed.

元学习方法对元知识ω应该是什么做出不同的选择，即应该学习学习策略的哪些方面; 以及(排除)哪些方面应该被认为是固定的。

Parameter Initialization. Here ω corresponds to the initial parameters of a neural network to be used in the inner optimization, with MAML being the most popular example [16], [98], [99]. A good initialization is just a few gradient steps away from a solution to any task T drawn from p(T ), and can help to learn without overfitting in few-shot learning. A key challenge with this approach is that the outer optimization needs to solve for as many parameters as the inner optimization (potentially hundreds of millions in large CNNs). This leads to a line of work on isolating a subset of parameters to meta-learn, for example by subspace [78], [100], by layer [83], [100], [101], or by separating out scale and shift [102]. Another concern is whether a single initial condition is sufficient to provide fast learning for a wide range of potential tasks, or if one is limited to narrow distributions p(T ). This has led to variants that model mixtures over multiple initial conditions [100], [103], [104].

参数初始化。这里ω对应于用于内部优化的神经网络的初始参数，MAML是最流行的例子[16]，[98]，[99]。一个好的初始化距离从p(T)得出的任何任务T的解仅几步之遥，并且可以帮助学习，而不会在少量学习中过度拟合。这种方法的一个关键挑战是，外部优化需要解决与内部优化一样多的参数(在大型CNN中可能有数亿个)。这导致了一系列工作，例如通过子空间[78]、[100]、层[83]、[1100]、[101]，或通过分离尺度和移位[102]，来隔离要进行元学习的参数子集。另一个问题是，单个初始条件是否足以为广泛的潜在任务提供快速学习，或者是否仅限于窄分布p(T)。这导致了在多个初始条件下对混合物进行建模的变体[100]、[103]、[104]。

Optimizer. The above parameter-centric methods usually rely on existing optimizers such as SGD with momentum or Adam [105] to refine the initialization when given some new task. Instead, optimizer-centric approaches [19], [39], [81], [94] focus on learning the inner optimizer by training a function that takes as input optimization states such as θ and ∇θL task and produces the optimization step for each base learning iteration. The trainable component ω can span simple hyper-parameters such as a fixed step size [79], [80] to more sophisticated pre-conditioning matrices [106], [107]. Ultimately ω can be used to define a full gradientbased optimizer through a complex non-linear transformation of the input gradient and other metadata [19], [39], [93], [94]. The parameters to learn here can be few if the optimizer is applied coordinate-wise across weights [19]. The initialization-centric and optimizer-centric methods can be merged by learning them jointly, namely having the former learn the initial condition for the latter [39], [79]. Optimizer learning methods have both been applied to for few-shot learning [39] and to accelerate and improve many-shot learning [19], [93], [94]. Finally, one can also meta-learn zeroth-order optimizers [108] that only require evaluations of L task rather than optimizer states such as gradients. These have been shown [108] to be competitive with conventional Bayesian Optimization [73] alternatives.

优化器。上述以参数为中心的方法通常依赖于现有的优化器，如具有动量的SGD或Adam[105]，以在给定一些新任务时优化初始化。相反，以优化器为中心的方法[19]、[39]、[81]、[94]侧重于通过训练一个函数来学习内部优化器，该函数将θ和∇θL任务，并为每个基础学习迭代生成优化步骤。可训练分量ω可以跨越简单的超参数，如固定步长[79]、[80]到更复杂的预处理矩阵[106]、[107]。最终，ω可以用于通过输入梯度和其他元数据的复杂非线性变换来定义一个基于全梯度的优化器[19]，[39]，[93]，[94]。如果优化器在权重上按坐标应用，这里要学习的参数可能很少[19]。以初始化为中心和以优化器为中心的方法可以通过联合学习来合并，即让前者学习后者的初始条件[39]，[79]。优化器学习方法已经应用于少样本学习[39]和加速和改进多样本学习[19]，[93]，[94]。最后，还可以元学习零阶优化器[108]，它只需要评估L任务，而不需要评估优化器状态，如梯度。这些已被证明[108]与传统的贝叶斯优化[73]备选方案具有竞争力。


Feed-Forward Models (FFMs. aka, Black-Box, Amortized). Another family of models trains learners ω that provide a feed-forward mapping directly from the support set to the parameters required to classify test instances, i.e., θ = gω(Dtrain) – rather than relying on a gradient-based iterative optimization of θ. These correspond to blackbox model-based learning in the conventional taxonomy (Sec. 3.1) and span from classic [109] to recent approaches such as CNAPs [110] that provide strong performance on challenging cross-domain few-shot benchmarks [111].

前馈模型(FFM.aka，黑匣子，摊销)。另一系列模型训练学习者ω，其提供直接从支持集到分类测试实例所需参数的前馈映射，即θ=gω(Dtrain)，而不是依赖于基于梯度的θ迭代优化。这些对应于传统分类法中基于黑匣子模型的学习(第3.1节)，从经典的[109]到最近的方法，如CNAP[110]，它们在挑战性的跨域少量测试基准上提供了强大的性能[111]。

These methods have connections to Hypernetworks [112], [113] which generate the weights of another neural network conditioned on some embedding – and are often used for compression or multi-task learning. Here ω is the hypernetwork and it synthesises θ given the source dataset in a feed-forward pass [100], [114]. Embedding the support set is often achieved by recurrent networks [51], [115], [116] convolution [38], or set embeddings [45], [110]. Research here often studies architectures for paramaterizing the classifier by the task-embedding network: (i) Which parameters should be globally shared across all tasks, vs synthesized per task by the hypernetwork (e.g., share the feature extractor and synthesize the classifier [83], [117]), and (ii) How to parameterize the hypernetwork so as to limit the number of parameters required in ω (e.g., via synthesizing only lightweight adapter layers in the feature extractor [110], or class-wise classifier weight synthesis [45]).
 
这些方法与超网络[112]、[113]有联系，超网络生成另一个基于某种嵌入的神经网络的权重，通常用于压缩或多任务学习。这里ω是超网络，它在前馈传递中合成给定源数据集的θ[100]，[114]。嵌入支持集通常通过递归网络[51]、[115]、[116]卷积[38]或集合嵌入[45]、[110]来实现。这里的研究经常研究通过任务嵌入网络对分类器进行参数化的架构：(i)哪些参数应该在所有任务中全局共享，而不是通过超网络对每个任务进行合成(例如，共享特征提取器并合成分类器[83]，[117])，以及(ii)如何参数化超网络以限制ω中所需的参数数量(例如，通过在特征提取器[110]中仅合成轻量级适配器层，或类分类器权重合成[45])。

Fig. 1. Overview of the meta-learning landscape including algorithm design (meta-optimizer, meta-representation, meta-objective), and applications. 

图1.元学习环境概述，包括算法设计(元优化器、元表示、元目标)和应用程序。

Some FFMs can also be understood elegantly in terms of amortized inference in probabilistic models [45], [109], making predictions for test data x as: 

一些FFM也可以用概率模型中的摊余推断来优雅地理解[45]，[109]，对测试数据x的预测如下：

qω(y|x, D tr) = Z p(y|x, θ)qω(θ|Dtr)dθ (8) 

where the meta-representation ω is a network qω(·) that approximates the intractable Bayesian inference for parameters θ that solve the task with training data Dtr, and the integral may be computed exactly [109], or approximated by sampling [45] or point estimate [110]. The model ω is then trained to minimise validation loss over a distribution of training tasks cf. Eq. 7.

其中，元表示ω是一个网络qω(·)，它近似于用训练数据Dtr解决任务的参数θ的难以处理的贝叶斯推断，积分可以精确计算[109]，或者通过采样[45]或点估计[110]来近似。然后对模型ω进行训练，以使训练任务分布中的验证损失最小化。

Finally, memory-augmented neural networks, with the ability to remember old data and assimilate new data quickly, typically fall in the FFM category as well [86], [87].

最后，记忆增强神经网络具有记忆旧数据和快速吸收新数据的能力，通常也属于FFM类别[86]，[87]。

Embedding Functions (Metric Learning). Here the metaoptimization process learns an embedding network ω that transforms raw inputs into a representation suitable for recognition by simple similarity comparison between query and support instances [20], [83], [90], [117] (e.g., with cosine similarity or euclidean distance). These methods are classified as metric learning in the conventional taxonomy (Section 3.1) but can also be seen as a special case of the feed-forward black-box models above. This can easily be seen for methods that produce logits based on the inner product of the embeddings of support and query images xs and xq, namely gTω (xq)gω(xs) [83], [117]. Here the support image generates ‘weights’ to interpret the query example, making it a special case of a FFM where the ‘hypernetwork’ generates a linear classifier for the query set. Vanilla methods in this family have been further enhanced by making the embedding task-conditional [101], [118], learning a more elaborate comparison metric [91], [92], or combining with gradient-based meta-learning to train other hyperparameters such as stochastic regularizers [119].

嵌入函数(度量学习)。这里，元优化过程学习嵌入网络ω，该网络通过查询和支持实例[20]、[83]、[90]、[117]之间的简单相似性比较(例如，具有余弦相似性或欧几里德距离)将原始输入转换为适合识别的表示。这些方法在传统分类法中被归类为度量学习(第3.1节)，但也可以被视为前馈黑匣子模型的特例。这对于基于支持和查询图像xs和xq嵌入的内积(即gTω(xq)gω(xs)[83]，[117])生成逻辑的方法很容易看出。这里，支持图像生成“权重”来解释查询样本，使其成为FFM的特例，其中“超网络”为查询集生成线性分类器。通过使嵌入任务具有条件[101]，[118]，学习更精细的比较度量[91]，[92]，或与基于梯度的元学习相结合来训练其他超参数，如随机正则化器[119]，进一步增强了该族中的普通方法。

Losses and Auxiliary Tasks. Analogously to the metalearning approach to optimizer design, these aim to learn the inner task-loss L task ω (·) for the base model. Loss-learning approaches typically define a small neural network that inputs quantities relevant to losses (e.g. predictions, features, or model parameters) and outputs a scalar to be treated as a loss by the inner (task) optimizer. This has potential benefits such as leading to a learned loss that is easier to optimize (e.g. less local minima) than commonly used ones [23], [120], [121], leads to faster learning with improved generalization [43], [122]–[124], or one whose minima correspond to a model more robust to domain shift [42]. Loss learning methods have also been used to learn to learn from unlabeled instances [101], [125], or to learn L task ω () as a differentiable approximation to a true non-differentiable task loss such as area under precision recall curve [126], [127].

损失和辅助任务。类似于优化器设计的元学习方法，这些方法旨在学习基本模型的内部任务损失L任务ω(·)。损失学习方法通常定义一个小神经网络，该网络输入与损失相关的量(例如预测、特征或模型参数)，并输出一个标量，由内部(任务)优化器作为损失处理。这具有潜在的好处，例如导致比常用损失更容易优化的学习损失(例如，较少的局部最小值)[23]、[120]、[121]，通过改进的泛化导致更快的学习[43]、[122]–[124]，或者其最小值对应于对域偏移更稳健的模型[42]。损失学习方法也被用于学习从未标记的实例中学习[101]，[125]，或学习L任务ω()，作为对真实不可微任务损失的可微近似，如精确召回曲线下的面积[126]，[127]。

Loss learning also arises in generalizations of selfsupervised [128] or auxiliary task [129] learning. In these problems unsupervised predictive tasks (such as colourising pixels in vision [128], or simply changing pixels in RL [129]) are defined and optimized with the aim of improving the representation for the main task. In this case the best auxiliary task (loss) to use can be hard to predict in advance, so meta-learning can be used to select among several auxiliary losses according to their impact on improving main task learning. I.e., ω is a per-auxiliary task weight [70]. More generally, one can meta-learn an auxiliary task generator that annotates examples with auxiliary labels [130].

损失学习也出现在自监督[128]或辅助任务[129]学习的推广中。在这些问题中，定义并优化了无监督预测任务(如给视觉中的像素着色[128]，或简单地改变RL[129]中的像素)，目的是改进主要任务的表示。在这种情况下，很难提前预测要使用的最佳辅助任务(损失)，因此可以使用元学习来根据其对改进主要任务学习的影响在几个辅助损失中进行选择。一、 例如，ω是每个辅助任务的权重[70]。更一般地，可以元学习辅助任务生成器，该生成器用辅助标签注释样本[130]。

Architectures.   Architecture discovery has always been an important area in neural networks [37], [131], and one that is not amenable to simple exhaustive search. Meta-Learning can be used to automate this very expensive process by learning architectures. Early attempts used evolutionary 7 algorithms to learn the topology of LSTM cells [132], while later approaches leveraged RL to generate descriptions for good CNN architectures [26]. Evolutionary Algorithms [25] can learn blocks within architectures modelled as graphs which could mutate by editing their graph. Gradient-based architecture representations have also been visited in the form of DARTS [18] where the forward pass during training consists in a softmax across the outputs of all possible layers in a given block, which are weighted by coefficients to be meta learned (i.e. ω). During meta-test, the architecture is discretized by only keeping the layers corresponding to the highest coefficients. Recent efforts to improve DARTS have focused on more efficient differentiable approximations [133], robustifying the discretization step [134], learning easy to adapt initializations [135], or architecture priors [136]. See Section 5.4 for more details.

建筑。架构发现一直是神经网络中的一个重要领域[37]，[131]，并且不适合简单的穷举搜索。元学习可以通过学习架构来自动化这个非常昂贵的过程。早期的尝试使用进化7算法来学习LSTM单元的拓扑[132]，而后来的方法利用RL来生成良好CNN架构的描述[26]。进化算法[25]可以学习被建模为图的架构中的块，这些块可以通过编辑它们的图而发生变化。基于梯度的架构表示也以DARTS[18]的形式进行了访问，其中训练期间的前向传递包括给定块中所有可能层的输出的软最大值，这些输出由待元学习的系数(即ω)加权。在元测试期间，仅通过保持与最高系数相对应的层来离散化架构。最近改进DARTS的努力集中在更有效的可微近似[133]、稳健化离散化步骤[134]、学习易于适应的初始化[135]或架构先验[136]。详见第5.4节。

Attention Modules. have been used as comparators in metric-based meta-learners [137], to prevent catastrophic forgetting in few-shot continual learning [138] and to summarize the distribution of text classification tasks [139].

注意模块。已被用作基于度量的元学习者[137]中的比较器，以防止少数样本连续学习中的灾难性遗忘[138]，并总结文本分类任务的分布[139]。

Modules. Modular meta-learning [140], [141] assumes that the task agnostic knowledge ω defines a set of modules, which are re-composed in a task specific manner defined by θ in order to solve each encountered task. These strategies can be seen as meta-learning generalizations of the typical structural approaches to knowledge sharing that are well studied in multi-task and transfer learning [67], [68], [142], and may ultimately underpin compositional learning [143].

模块。模块元学习[140]，[141]假设任务未知知识ω定义了一组模块，这些模块以θ定义的任务特定方式重新组合，以解决每个遇到的任务。这些策略可以被视为知识共享的典型结构方法的元学习概括，这些方法在多任务和迁移学习中得到了很好的研究[67]、[68]、[142]，并可能最终支持合成学习[143]。

Hyper-parameters. Here ω represents hyperparameters of the base learner such as regularization strength [17], [71], per-parameter regularization [95], task-relatedness in multitask learning [69], or sparsity strength in data cleansing [69].

超级参数。这里ω表示基础学习器的超参数，如正则化强度[17]、[71]、每参数正则化[95]、多任务学习中的任务相关性[69]或数据清理中的稀疏性强度[69]。

Hyperparameters. such as step size [71], [79], [80] can be seen as part of the optimizer, leading to an overlap between hyper-parameter and optimizer learning categories.

超参数。诸如步长[71]、[79]、[80]可以被视为优化器的一部分，导致超参数和优化器学习类别之间的重叠。

Data Augmentation. In supervised learning it is common to improve generalization by synthesizing more training data through label-preserving transformations on the existing data. The data augmentation operation is wrapped up in optimization steps of the inner problem (Eq. 6), and is conventionally hand-designed. However, when ω defines the data augmentation strategy, it can be learned by the outer optimization in Eq. 5 in order to maximize validation performance [144]. Since augmentation operations are typically non-differentiable, this requires reinforcement learning [144], discrete gradient-estimators [145], or evolutionary [146] methods. An open question is whether powerful GANbased data augmentation methods [147] can be used in inner-level learning and optimized in outer-level learning.

数据增广。在监督学习中，通常通过对现有数据进行标签保留变换来合成更多的训练数据，从而改进泛化。数据增广操作被包裹在内部问题的优化步骤中(等式6)，并且通常是手工设计的。然而，当ω定义数据增广策略时，可以通过等式5中的外部优化来学习，以最大化验证性能[144]。由于增强操作通常是不可微的，这需要强化学习[144]、离散梯度估计[145]或进化[146]方法。一个悬而未决的问题是，强大的基于GAN的数据增广方法[147]是否可以用于内部学习，并在外部学习中进行优化。

Minibatch Selection, Sample Weights, and Curriculum Learning When the base algorithm is minibatch-based stochastic gradient descent, a design parameter of the learning strategy is the batch selection process. Various handdesigned methods [148] exist to improve on randomlysampled minibatches. Meta-learning approaches can define ω as an instance selection probability [149] or neural network that picks instances [150] for inclusion in a minibatch. Related to mini-batch selection policies are methods that learn per-sample loss weights ω for the training set [151], [152]. This can be used to learn under label-noise by discounting noisy samples [151], [152], discount outliers [69], or correct for class imbalance [151]

小批量选择、样本权重和课程学习当基本算法是基于小批量的随机梯度下降时，学习策略的设计参数是批量选择过程。存在各种手工设计的方法[148]来改进随机抽样的小批次。元学习方法可以将ω定义为实例选择概率[149]或选择实例[150]以包含在小批量中的神经网络。与小批量选择策略相关的是学习训练集的每样本损失权重ω的方法[151]，[152]。这可以用于通过对噪声样本进行贴现来学习标签下噪声[151]、[152]、贴现异常值[69]或校正类不平衡[151]

More generally, the curriculum [153] refers to sequences of data or concepts to learn that produce better performance than learning items in a random order. For instance by focusing on instances of the right difficulty while rejecting too hard or too easy (already learned) instances. Instead of defining a curriculum by hand [154], meta-learning can automate the process and select examples of the right difficulty by defining a teaching policy as the meta-knowledge and training it to optimize the student’s progress [150], [155].

更一般地说，课程[153]指的是要学习的数据或概念序列，这些数据或概念以随机顺序比学习项目产生更好的表现。例如，通过关注正确难度的实例，同时拒绝太难或太容易(已经学会)的实例。代替手工定义课程[154]，元学习可以通过将教学策略定义为元知识并对其进行训练以优化学生的进步，从而自动化过程并选择正确难度的样本[150]，[155]。

Datasets, Labels and Environments. Another metarepresentation is the support dataset itself. This departs from our initial formalization of meta-learning which considers the source datasets to be fixed (Section 2.1, Eqs. 2-3).

数据集、标签和环境。另一个元表示是支持数据集本身。这与我们最初的元学习形式化不同，元学习认为源数据集是固定的(第2.1节，等式2-3)。

However, it can be easily understood in the bilevel view of Eqs. 5-6. If the validation set in the upper optimization is real and fixed, and a train set in the lower optimization is paramaterized by ω, the training dataset can be tuned by meta-learning to optimize validation performance.

然而，在方程的双层视图中很容易理解。5-6.如果上部优化中的验证集是真实的和固定的，并且下部优化中的训练集被ω参数化，则可以通过元学习调整训练数据集以优化验证性能。

In dataset distillation [156], [157], the support images themselves are learned such that a few steps on them allows for good generalization on real query images. This can be used to summarize large datasets into a handful of images, which is useful for replay in continual learning where streaming datasets cannot be stored.

在数据集提取[156]，[157]中，学习支持图像本身，从而对其进行几步操作，就可以很好地概括真实的查询图像。这可以用于将大型数据集归纳为少数图像，这对于无法存储流数据集的连续学习中的重放非常有用。

Rather than learning input images x for fixed labels y, one can also learn the input labels y for fixed images x. This can be used in distilling core sets [158] as in dataset distillation; or semi-supervised learning, for example to directly learn the unlabeled set’s labels to optimize validation set performance [159], [160].

除了学习固定标签y的输入图像x，还可以学习固定图像x的输入标签y。这可以用于提取核心集[158]，如数据集提取; 或半监督学习，例如，直接学习未标记集的标签以优化验证集性能[159]，[160]。

In the case of sim2real learning [161] in computer vision or reinforcement learning, one uses an environment simulator to generate data for training. In this case, as detailed in Section 5.3, one can also train the graphics engine [162] or simulator [163] so as to optimize the real-data (validation) performance of the downstream model after training on data generated by that environment simulator.

在计算机视觉或强化学习中的模拟真实学习[161]的情况下，使用环境模拟器生成训练数据。在这种情况下，如第5.3节所述，还可以训练图形引擎[162]或模拟器[163]，以便在对该环境模拟器生成的数据进行训练后，优化下游模型的真实数据(验证)性能。

Discussion: Transductive Representations and Methods. Most of the representations ω discussed above are parameter vectors of functions that process or generate data. However a few of the representations mentioned are transductive in the sense that the ω literally corresponds to data points [156], labels [159], or per-sample weights [152]. Therefore the number of parameters in ω to meta-learn scales as the size of the dataset. While the success of these methods is a testament to the capabilities of contemporary meta-learning [157], this property may ultimately limit their scalability.

讨论：传导性表达和方法。上面讨论的大多数表示ω是处理或生成数据的函数的参数向量。然而，在ω字面上对应于数据点[156]、标签[159]或每样本权重[152]的意义上，所提到的一些表示是传递的。因此，ω到元学习的参数数量随数据集的大小而变化。虽然这些方法的成功证明了当代元学习的能力[157]，但这一特性可能最终限制了它们的可扩展性。

Distinct from a transductive representation are methods that are transductive in the sense that they operate on the query instances as well as support instances [101], [130].

与转导表示不同的是转导的方法，因为它们在查询实例和支持实例上操作[101]，[130]。

Discussion: Interpretable Symbolic Representations A cross-cutting distinction that can be made across many of the meta-representations discussed above is between uninterpretable (sub-symbolic) and human interpretable (symbolic) representations. Sub-symbolic representations, such as when ω parameterizes a neural network [19], are more common and make up the majority of studies cited above. 

讨论：可解释的符号表示上文讨论的许多元表示之间的交叉区别是不可解释(亚符号)和人类可解释(符号)表示。子符号表示，如ω参数化神经网络[19]时，更为常见，并构成了上述大多数研究。

However, meta-learning with symbolic representations is also possible, where ω represents human readable symbolic functions such as optimization program code [93]. Rather than neural loss functions [42], one can train symbolic losses ω that are defined by an expression analogous to cross-entropy [123]. One can also meta-learn new symbolic activations [164] that outperform standards such as ReLU.

然而，具有符号表示的元学习也是可能的，其中ω表示人类可读的符号函数，如优化程序代码[93]。与神经损失函数[42]不同，我们可以训练符号损失ω，其由类似于交叉熵的表达式定义[123]。还可以元学习优于ReLU等标准的新符号激活[164]。

As these meta-representations are non-smooth, the metaobjective is non-differentiable and is harder to optimize (see Section 4.2). So the upper optimization for ω typically uses RL [93] or evolutionary algorithms [123]. However, symbolic representations may have an advantage [93], [123], [164] in their ability to generalize across task families. I.e., to span wider distributions p(T ) with a single ω during metatraining, or to have the learned ω generalize to an out of distribution task during meta-testing (see Section 6).

由于这些元表示是非光滑的，所以元目标是不可微的，并且更难优化(参见第4.2节)。因此，ω的上限优化通常使用RL[93]或进化算法[123]。然而，符号表示在跨任务族进行概括的能力方面可能具有优势[93]、[123]、[164]。一、 例如，在元训练期间使用单个ω跨越更宽的分布p(T)，或者在元测试期间将学习到的ω推广到分布外任务(参见第6节)。

Discussion: Amortization One way to relate some of the representations discussed is in terms of the degree of learning amortization entailed [45]. That is, how much task-specific optimization is performed during meta-testing vs how much learning is amortized during meta-training. Training from scratch, or conventional fine-tuning [57] perform full task-specific optimization at meta-testing, with no amortization. MAML [16] provides limited amortization by fitting an initial condition, to enable learning a new task by few-step fine-tuning. Pure FFMs [20], [90], [110] are fully amortized, with no task-specific optimization, and thus enable the fastest learning of new tasks. Meanwhile some hybrid approaches [100], [101], [111], [165] implement semiamortized learning by drawing on both feed-forward and optimization-based meta-learning in a single framework.

讨论：摊销将所讨论的一些表述联系起来的一种方法是根据所涉及的学习摊销程度[45]。即,在元测试期间执行了多少任务特定的优化，而在元训练期间摊销了多少学习。从头开始的训练或传统的微调[57]在元测试中执行完全特定于任务的优化，无需摊销。MAML[16]通过拟合初始条件提供了有限的摊销，从而能够通过几步微调来学习新任务。纯FFM[20]、[90]、[110]完全摊销，没有特定于任务的优化，因此能够最快地学习新任务。同时，一些混合方法[100]、[101]、[111]、[165]通过在单个框架中利用前馈和基于优化的元学习来实现半摊销学习。

### 4.2 Meta-Optimizer
Given a choice of which facet of the learning strategy to optimize, the next axis of meta-learner design is actual outer (meta) optimization strategy to use for training ω.

给定要优化学习策略的哪一方面的选择，元学习者设计的下一个轴是用于训练ω的实际外部(元)优化策略。

Gradient. A large family of methods use gradient descent on the meta parameters ω [16], [39], [42], [69]. This requires computing derivatives dL meta/dω of the outer objective, which are typically connected via the chain rule to the model parameter θ, dL meta/dω = (dL meta/dθ)(dθ/dω). These methods are potentially the most efficient as they exploit analytical gradients of ω. However key challenges include: (i) Efficiently differentiating through many steps of inner optimization, for example through careful design of differentiation algorithms [17], [71], [193] and implicit differentiation [157], [167], [194], and dealing tractably with the required second-order gradients [195]. (ii) Reducing the inevitable gradient degradation problems whose severity increases with the number of inner loop optimization steps. (iii) Calculating gradients when the base learner, ω, or L task include discrete or other non-differentiable operations.

坡度一大类方法对元参数ω[16]，[39]，[42]，[69]使用梯度下降。这需要计算外物镜的导数dL meta/dω，它们通常通过链式规则与模型参数θ相连接，dL meta/dω=(dL meta/dθ)(dθ/dω)。这些方法可能是最有效的，因为它们利用了ω的分析梯度。然而，关键的挑战包括：(i)通过内部优化的许多步骤有效地进行微分，例如通过仔细设计微分算法[17]、[71]、[193]和隐式微分[157]、[167]、[194]，以及处理所需的二阶梯度[195]。(ii)减少不可避免的梯度退化问题，其严重性随着内环优化步骤的数量而增加。(iii)当基本学习者、ω或L任务包括离散或其他不可微分操作时，计算梯度。

Reinforcement Learning. When the base learner includes non-differentiable steps [144], or the meta-objective L meta is itself non-differentiable [126], many methods [22] resort to RL to optimize the outer objective Eq. 5. This estimates the gradient ∇ωL meta, typically using the policy gradient theorem. However, alleviating the requirement for differentiability in this way is typically extremely costly. Highvariance policy-gradient estimates for ∇ωL meta mean that many outer-level optimization steps are required to converge, and each of these steps are themselves costly due to wrapping task-model optimization within them.

强化学习。当基础学习器包括不可微步骤[144]，或元目标L元本身不可微[126]时，许多方法[22]求助于RL来优化外部目标方程5∇ωL元，通常使用策略梯度定理。然而，以这种方式减轻对可差异性的要求通常是极其昂贵的。高方差策略梯度估计∇ωL meta意味着需要许多外部优化步骤才能收敛，并且由于将任务模型优化封装在这些步骤中，这些步骤本身都很昂贵。

Evolution. Another approach for optimizing the metaobjective are evolutionary algorithms (EA) [14], [131], [196]. Many evolutionary algorithms have strong connections to reinforcement learning algorithms [197]. However, their performance does not depend on the length and reward sparsity of the inner optimization as for RL.

进化另一种优化元目标的方法是进化算法(EA)[14]、[131]、[196]。许多进化算法与强化学习算法有很强的联系[197]。然而，对于RL，它们的性能并不取决于内部优化的长度和奖励稀疏性。

EAs are attractive for several reasons [196]: (i) They can optimize any base model and meta-objective with no differentiability constraint. (ii) Not relying on backpropagation avoids both gradient degradation issues and the cost of high-order gradient computation of conventional gradient-based methods. (iii) They are highly parallelizable for scalability. (iv) By maintaining a diverse population of solutions, they can avoid local minima that plague gradientbased methods [131]. However, they have a number of disadvantages: (i) The population size required increases rapidly with the number of parameters to learn. (ii) They can be sensitive to the mutation strategy and may require careful hyperparameter optimization. (iii) Their fitting ability is generally inferior to gradient-based methods, especially for large models such as CNNs.

EA之所以具有吸引力，有几个原因[196]：(i)它们可以在没有可微性约束的情况下优化任何基础模型和元目标。(ii)不依赖反向传播避免了梯度退化问题和传统基于梯度的方法的高阶梯度计算的成本。(iii)它们具有高度的可并行性和可扩展性。(iv)通过保持解决方案的多样性，他们可以避免困扰基于梯度的方法的局部最小值[131]。然而，它们有一些缺点：(i)所需的人口数量随着要学习的参数数量的增加而迅速增加。(ii)它们可能对突变策略敏感，可能需要仔细的超参数优化。(iii)它们的拟合能力通常不如基于梯度的方法，尤其是对于大型模型，如CNN。

EAs are relatively more commonly applied in RL applications [23], [172] (where models are typically smaller, and inner optimizations are long and non-differentiable). However they have also been applied to learn learning rules [198], optimizers [199], architectures [25], [131] and data augmentation strategies [146] in supervised learning. They are also particularly important in learning human interpretable symbolic meta-representations [123].

EA相对更常用于RL应用[23]、[172](其中模型通常较小，内部优化较长且不可微分)。然而，它们也被应用于监督学习中的学习规则[198]、优化器[199]、架构[25]、[131]和数据增广策略[146]。它们在学习人类可解释符号元表示方面也特别重要[123]。

### 4.3 Meta-Objective and Episode Design
The final component is to define the meta-learning goal through choice of meta-objective L meta, and associated data flow between inner loop episodes and outer optimizations. Most methods define a meta-objective using a performance metric computed on a validation set, after updating the task model with ω. This is in line with classic validation set approaches to hyperparameter and model selection. However, within this framework, there are several design options:

最后一个组成部分是通过选择元目标L元以及内部循环事件和外部优化之间的相关数据流来定义元学习目标。大多数方法在用ω更新任务模型后，使用在验证集上计算的性能度量来定义元目标。这与超参数和模型选择的经典验证集方法一致。然而，在此框架内，有几个设计选项：

Many vs Few-Shot Episode Design. According to whether the goal is improving few- or many-shot performance, inner loop learning episodes may be defined with many [69], [93], [94] or few- [16], [39] examples per-task.

多样本与少样本情节设计。根据目标是提高少样本还是多样本性能，每个任务可以用许多[69]、[93]、[94]或几个[16]、[39]样本来定义内环学习集。

Fast Adaptation vs Asymptotic Performance. When validation loss is computed at the end of the inner learning episode, meta-training encourages better final performance of the base task. When it is computed as the sum of the validation loss after each inner optimization step, then metatraining also encourages faster learning in the base task [80], [93], [94]. Most RL applications also use this latter setting.

快速适应与渐进性能。当在内部学习阶段结束时计算验证损失时，元训练鼓励更好地完成基本任务。当将其计算为每个内部优化步骤后的验证损失之和时，元训练也鼓励基础任务中更快的学习[80]、[93]、[94]。大多数RL应用程序也使用后一种设置。

Multi vs Single-Task. When the goal is to tune the learner to better solve any task drawn from a given family, then inner loop learning episodes correspond to a randomly drawn task from p(T ) [16], [20], [42]. When the goal is to tune the learner to simply solve one specific task better, then the inner loop learning episodes all draw data from the same underlying task [19], [69], [175], [183], [184], [200]. 

多任务与单任务。当目标是调整学习者以更好地解决来自给定家庭的任何任务时，内环学习集对应于来自p(T)[16]、[20]、[42]的随机抽取任务。当目标是调整学习者以更好地简单地解决一个特定任务时，内循环学习集都从相同的基础任务中提取数据[19]、[69]、[175]、[183]、[184]、[200]。



TABLE 1 Research papers according to our taxonomy. We use color to indicate salient meta-objective or application goal. We focus on the main goal of each paper for simplicity. The color code is: sample efficiency (red), learning speed (green), asymptotic performance (purple), cross-domain (blue).

表1根据我们的分类法研究论文。我们使用颜色表示突出的元目标或应用目标。为了简单起见，我们专注于每篇论文的主要目标。颜色代码是：样本效率(红色)、学习速度(绿色)、渐近性能(紫色)、跨域(蓝色)。

It is worth noting that these two meta-objectives tend to have different assumptions and value propositions. The multi-task objective obviously requires a task family p(T ) to work with, which single-task does not. Meanwhile for multi-task, the data and compute cost of meta-training can be amortized by potentially boosting the performance of multiple target tasks during meta-test; but single-task – without the new tasks for amortization – needs to improve the final solution or asymptotic performance of the current task, or meta-learn fast enough to be online.

值得注意的是，这两个元目标往往具有不同的假设和价值主张。多任务目标显然需要一个任务族p(T)来处理，而单个任务则不需要。同时，对于多任务，元训练的数据和计算成本可以通过在元测试期间潜在地提高多个目标任务的性能来分摊; 但是，单个任务(没有新的摊销任务)需要改进当前任务的最终解决方案或渐近性能，或者元学习速度足够快才能在线。

Online vs Offline. While the classic meta-learning pipeline defines the meta-optimization as an outer-loop of the inner base learner [16], [19], some studies have attempted to preform meta-optimization online within a single base learning episode [42], [183], [200], [201]. In this case the base model θ and learner ω co-evolve during a single episode. Since there is now no set of source tasks to amortize over, meta-learning needs to be fast compared to base model learning in order to benefit sample or compute efficiency.

在线vs离线。虽然经典的元学习管道将元优化定义为内部基础学习者的外循环[16]，[19]，但一些研究试图在单个基础学习事件[42]，[183]，[200]，[201]中在线执行元优化。在这种情况下，基本模型θ和学习者ω在单个事件中共同进化。由于现在没有一组源任务可以分摊，元学习需要比基础模型学习更快，以提高样本或计算效率。

Other Episode Design Factors. Other operators can be inserted into the episode generation pipeline to customize meta-learning for particular applications. For example one can simulate domain-shift between training and validation to meta-optimize for good performance under domainshift [42], [59], [95]; simulate network compression such as quantization [202] between training and validation to metaoptimize for network compressibility; provide noisy labels during meta-training to optimize for label-noise robustness [96], or generate an adversarial validation set to metaoptimize for adversarial defense [97]. These opportunities are explored in more detail in the following section. 

其他情节设计因素。可以将其他操作符插入到情节生成管道中，以便为特定应用程序定制元学习。例如，可以模拟训练和验证之间的域迁移，以在域迁移[42]、[59]、[95]下进行元优化以获得良好的性能; 模拟网络压缩，如训练和验证之间的量化[202]，以对网络压缩性进行元优化; 在元训练期间提供噪声标签以优化标签噪声稳健性[96]，或生成对抗性验证集以对对抗性防御进行元优化[97]。以下部分将更详细地探讨这些机会。

## 5 APPLICATIONS
In this section we briefly review the ways in which metalearning has been exploited in computer vision, reinforcement learning, architecture search, and so on.

在本节中，我们简要回顾了元学习在计算机视觉、强化学习、架构搜索等方面的应用方式。

### 5.1 Computer Vision and Graphics
Computer vision is a major consumer domain of metalearning techniques, notably due to its impact on few-shot learning, which holds promise to deal with the challenge posed by the long-tail of concepts to recognise in vision.

计算机视觉是元学习技术的一个主要消费领域，特别是由于它对少样本学习的影响，这有望应对视觉中识别概念的长尾带来的挑战。

#### 5.1.1 Few-Shot Learning Methods
Few-shot learning (FSL) is extremely challenging, especially for large neural networks [1], [13], where data volume is often the dominant factor in performance [203], and training large models with small datasets leads to overfitting or non-convergence. Meta-learning-based approaches are increasingly able to train powerful CNNs on small datasets in many vision problems. We provide a non-exhaustive representative summary as follows.

少样本学习(FSL)极具挑战性，尤其是对于大型神经网络[1]，[13]，其中数据量通常是性能的主导因素[203]，并且使用小数据集训练大型模型会导致过度拟合或不收敛。基于元学习的方法越来越能够在许多视觉问题的小数据集上训练强大的CNN。我们提供了以下非详尽的代表性摘要。

Classification. The most common application of metalearning is few-shot multi-class image recognition, where the inner and outer loss functions are typically the cross entropy over training and validation data respectively [20], [39], [77], [79], [80], [90], [92], [100], [101], [104], [107], [204]– [207]. Optimizer-centric [16], black-box [38], [83] and metric learning [90]–[92] models have all been considered.

分类元学习最常见的应用是少样本多类图像识别，其中内部和外部损失函数通常分别是训练和验证数据的交叉熵[20]、[39]、[77]、[79]、[80]、[90]、[92]、[100]、[101]、[104]、[107]、[204]–[207]。以优化器为中心的[16]、黑盒[38]、[83]和度量学习[90]-[92]模型都已被考虑。

This line of work has led to a steady improvement in performance compared to early methods [16], [89], [90]. However, performance is still far behind that of fully supervised methods, so there is more work to be done. Current research issues include improving cross-domain generalization [119], recognition within the joint label space defined by metatrain and meta-test classes [84], and incremental addition of new few-shot classes [138], [178].

与早期方法[16]、[89]、[90]相比，这一工作已导致性能稳步提高。然而，性能仍然远远落后于完全监督的方法，因此还有更多的工作要做。当前的研究问题包括改进跨域泛化[119]、在元训练和元测试类定义的联合标签空间内进行识别[84]，以及增加新的少数样本类[138]，[178]。

Object Detection Building on progress in few-shot classification, few-shot object detection [178], [208] has been demonstrated, often using feed-forward hypernetworkbased approaches to embed support set images and synthesize final layer classification weights in the base model.

对象检测基于少样本分类的进展，已经证明了少样本对象检测[178]，[208]，通常使用基于前馈超网络的方法来嵌入支持集图像并在基础模型中合成最终的层分类权重。

Landmark Prediction aims to locate a skeleton of key points within an image, such as such as joints of a human or robot. This is typically formulated as an image-conditional 10 regression. For example, a MAML-based model was shown to work for human pose estimation [209], modular-metalearning was successfully applied to robotics [140], while a hypernetwork-based model was applied to few-shot clothes fitting for novel fashion items [178].

地标预测旨在定位图像中关键点的骨架，例如人类或机器人的关节。这通常被公式化为图像条件10回归。例如，基于MAML的模型被证明可以用于人体姿态估计[209]，模块化金属学习成功地应用于机器人[140]，而基于超网络的模型被应用于适合新颖时尚项目的少量服装[178]。

Few-Shot Object Segmentation is important due to the cost of obtaining pixel-wise labeled images. Hypernetworkbased meta-learners have been applied in the one-shot regime [210], and performance was later improved by adapting prototypical networks [211]. Other models tackle cases where segmentation has low density [212].

由于获取逐像素标记图像的成本，很少样本对象分割很重要。基于超网络的元学习器已应用于一次性模式[210]，随后通过调整原型网络[211]提高了性能。其他模型处理分割密度低的情况[212]。

Image and Video Generation. In [45] an amortized probabilistic meta-learner is used to generate multiple views of an object from just a single image, generative query networks [213] render scenes from novel views, and talking faces are generated from little data by learning the initialization of an adversarial model for quick adaptation [214]. In video domain, [215] meta-learns a weight generator that synthesizes videos given few example images as cues.

图像和视频生成。在[45]中，摊销概率元学习器用于仅从单个图像生成对象的多个视图，生成性查询网络[213]从新颖的视图渲染场景，并且通过学习对手模型的初始化来从很少的数据生成说话的脸，以便快速适应[214]。在视频领域，[215]元学习了一个权重生成器，该权重生成器在给定少量样本图像作为线索的情况下合成视频。

Generative Models and Density Estimation Density estimators capable of generating images typically require many parameters, and as such overfit in the few-shot regime. Gradient-based meta-learning of PixelCNN generators was shown to enable their few-shot learning [216].

能够生成图像的生成模型和密度估计密度估计器通常需要许多参数，因此在少样本情况下会过度匹配。PixelCNN生成器的基于梯度的元学习被证明能够实现其少数样本学习[216]。

#### 5.1.2 Few-Shot Learning Benchmarks
Progress in AI and machine learning is often measured, and spurred, by well designed benchmarks [217]. Conventional ML benchmarks define a task and dataset for which a model should generalize from seen to unseen instances. In metalearning, benchmark design is more complex, since we are often dealing with a learner that should generalize from seen to unseen tasks. Benchmark design thus needs to define families of tasks from which meta-training and meta-testing tasks can be drawn. Established FSL benchmarks include miniImageNet [39], [90], Tiered-ImageNet [218], SlimageNet [219], Omniglot [90] and Meta-Dataset [111].

人工智能和机器学习的进展通常由精心设计的基准来衡量和推动[217]。传统的ML基准定义了一个任务和数据集，模型应该将其从可见实例概括为不可见实例。在元学习中，基准设计更为复杂，因为我们经常与学习者打交道，他们应该从可见任务概括到看不见的任务。因此，基准设计需要定义可从中提取元训练和元测试任务的任务系列。已建立的FSL基准包括miniImageNet[39]、[90]、分层ImageNet[218]、SlimageNet[219]、Omniglot[90]和元数据集[111]。

Dataset Diversity, Bias and Generalization. The standard benchmarks provide tasks for training and evaluation, but suffer from a lack of diversity (narrow p(T )) which makes performance on these benchmarks non-reflective of performance on real-world few shot task. For example, switching between different kinds of animal photos in miniImageNet is not a strong test of generalization. Ideally we would like to span more diverse categories and types of images (satellite, medical, agricultural, underwater, etc); and even be robust to domain-shifts between meta-train and metatest tasks.

数据集多样性、偏差和泛化。标准基准提供了用于训练和评估的任务，但缺乏多样性(窄p(T))，这使得这些基准的性能不能反映真实世界的少量任务的性能。例如，在miniImageNet中切换不同种类的动物照片并不是对泛化的有力测试。理想情况下，我们希望跨越更多种类和类型的图像(卫星、医疗、农业、水下等); 甚至对元训练和元测试任务之间的域转换具有稳健性。

There is work still to be done here as, even in the manyshot setting, fitting a deep model to a very wide distribution of data is itself non-trivial [220], as is generalizing to out-ofsample data [42], [95]. Similarly, the performance of metalearners often drops drastically when introducing a domain shift between the source and target task distributions [117]. This motivates the recent Meta-Dataset [111] and CVPR cross-domain few-shot challenge [221]. Meta-Dataset aggregates a number of individual recognition benchmarks to provide a wider distribution of tasks p(T ) to evaluate the ability to fit a wide task distribution and generalize across domain-shift. Meanwhile, [221] challenges methods to generalize from the everyday ImageNet images to medical, satellite and agricultural images. Recent work has begun to try and address these issues by meta-training for domainshift robustness as well as sample efficiency [119]. Generalization issues also arise in applying models to data from under-represented countries [222].

这里仍有工作要做，因为即使在许多截图设置中，将深度模型拟合到非常广泛的数据分布本身也是非常重要的[220]，这也是对样本外数据的概括[42]，[95]。类似地，当在源和目标任务分布之间引入域转换时，metalearners的性能通常会急剧下降[117]。这激发了最近的Meta Dataset[111]和CVPR跨域少样本挑战[222]。Meta Dataset集合了许多个体识别基准，以提供更广泛的任务分布p(T)，以评估适应广泛任务分布和跨域迁移的能力。同时，[221]对从日常ImageNet图像到医疗、卫星和农业图像的推广方法提出了挑战。最近的工作已经开始尝试通过域迁移稳健性和样本效率的元训练来解决这些问题[119]。在将模型应用于代表性不足国家的数据时也会出现泛化问题[222]。

### 5.2 Meta Reinforcement Learning and Robotics
Reinforcement learning is typically concerned with learning control policies that enable an agent to obtain high reward after performing a sequential action task within an environment. RL typically suffers from extreme sample inefficiency due to sparse rewards, the need for exploration, and the high-variance [223] of optimization algorithms. However, applications often naturally entail task families which meta-learning can exploit – for example locomotingto or reaching-to different positions [188], navigating within different environments [38], traversing different terrains [65], driving different cars [187], competing with different competitor agents [63], and dealing with different handicaps such as failures in individual robot limbs [65]. Thus RL provides a fertile application area in which meta-learning on task distributions has had significant successes in improving sample efficiency over standard RL algorithms. One can intuitively understand the efficacy of these methods. For instance meta-knowledge of a maze layout is transferable for all tasks that require navigating within the maze.

强化学习通常与学习控制策略有关，该策略使智能体能够在环境中执行顺序动作任务后获得高回报。RL通常由于奖励稀疏、需要探索以及优化算法的高方差[223]而遭受极端样本效率低下的困扰。然而，应用程序通常自然而然地需要元学习可以利用的任务系列——例如，移动到或到达不同位置[188]、在不同环境中导航[38]、穿越不同地形[65]、驾驶不同的汽车[187]、与不同的竞争对手智能体人竞争[63]、，以及处理不同的障碍，如单个机器人肢体的故障[65]。因此，RL提供了一个肥沃的应用领域，在该领域中，任务分布的元学习在提高样本效率方面比标准RL算法取得了显著的成功。人们可以直观地理解这些方法的功效。例如，迷宫布局的元知识可用于需要在迷宫中导航的所有任务。

#### 5.2.1 Methods
Several meta-representations that we have already seen have been explored in RL including learning the initial conditions [16], [173], hyperparameters [173], [177], step directions [79] and step sizes [176], which enables gradientbased learning to train a neural policy with fewer environmental interactions; and training fast convolutional [38] or recurrent [22], [116] black-box models to embed the experience of a given environment to synthesize a policy. Recent work has developed improved meta-optimization algorithms [169], [170], [172] for these tasks, and provided theoretical guarantees for meta-RL [224].

我们已经在RL中探索了几种元表示，包括学习初始条件[16]、[173]、超参数[173]和[177]、步长方向[79]和步长[176]，这使得基于梯度的学习能够训练具有较少环境交互的神经策略; 以及训练快速卷积[38]或递归[22]、[116]黑盒模型以嵌入给定环境的经验来合成策略。最近的工作为这些任务开发了改进的元优化算法[169]、[170]、[172]，并为元RL[224]提供了理论保证。

Exploration. A meta-representation rather unique to RL is the exploration policy. RL is complicated by the fact that the data distribution is not fixed, but varies according to the agent’s actions. Furthermore, sparse rewards may mean that an agent must take many actions before achieving a reward that can be used to guide learning. As such, how to explore and acquire data for learning is a crucial factor in any RL algorithm. Traditionally exploration is based on sampling random actions [225], or hand-crafted heuristics [226]. Several meta-RL studies have instead explicitly treated exploration strategy or curiosity function as metaknowledge ω; and modeled their acquisition as a metalearning problem [24], [186], [187], [227] – leading to sample efficiency improvements by ‘learning how to explore’.

勘探RL独有的元表示是探索策略。RL的复杂性在于数据分布不是固定的，而是根据智能体的行为而变化的。此外，稀疏奖励可能意味着智能体人必须采取许多行动才能获得可用于指导学习的奖励。因此，如何探索和获取用于学习的数据是任何RL算法中的关键因素。传统上，探索基于采样随机动作[225]，或手工制作的试探[226]。几项元RL研究明确地将探索策略或好奇功能视为元知识ω; 并将他们的获取建模为元学习问题[24]、[186]、[187]、[227]–通过“学习如何探索”来提高样本效率。

Optimization. RL is a difficult optimization problem where the learned policy is usually far from optimal, even on ‘training set’ episodes. This means that, in contrast to meta-SL, meta-RL methods are more commonly deployed to increase asymptotic performance [23], [177], [183] as 11 well as sample-efficiency, and can lead to significantly better solutions overall. The meta-objective of many meta-RL frameworks is the net return of the agent over a full episode, and thus both sample efficient and asymptotically performant learning are rewarded. Optimization difficulty also means that there has been relatively more work on learning losses (or rewards) [121], [124], [183], [228] which an RL agent should optimize instead of – or in addition to – the conventional sparse reward objective. Such learned losses may be easier to optimize (denser, smoother) compared to the true target [23], [228]. This also links to exploration as reward learning and can be considered to instantiate metalearning of learning intrinsic motivation [184].

优化。RL是一个困难的优化问题，其中学习到的策略通常远不是最优的，即使在“训练集”事件中也是如此。这意味着，与元SL相比，元RL方法更常用于提高渐近性能[23]、[177]、[183]和11以及样本效率，并可导致总体上显著更好的解决方案。许多元RL框架的元目标是智能体在整个事件中的净回报，因此样本效率和渐近性能学习都会得到回报。优化难度还意味着，在学习损失(或奖励)方面的工作相对较多[121]、[124]、[183]、[228]，RL智能体应该对其进行优化，而不是传统的稀疏奖励目标，或者除了传统的稀疏奖赏目标之外。与真实目标[23]、[228]相比，这种学习损失可能更容易优化(更密集、更平滑)。这也与探索作为奖励学习相联系，可以被认为是学习内在动机的元学习的实例化[184]。

Online meta-RL A significant fraction of meta-RL studies addressed the single-task setting, where the metaknowledge such as loss [121], [183], reward [177], [184], hyperparameters [175], [176], or exploration strategy [185] are trained online together with the base policy while learning a single task. These methods thus do not require task families and provide a direct improvement to their respective base learners’ performance.

在线元RL很大一部分元RL研究涉及单个任务设置，在学习单个任务时，元知识(如损失[121]、[183]、奖励[177]、[184]、超参数[175]、[176]或探索策略[185])与基本策略一起在线训练。因此，这些方法不需要任务族，并直接提高各自基础学习者的成绩。

On- vs Off-Policy meta-RL A major dichotomy in conventional RL is between on-policy and off-policy learning such as PPO [225] vs SAC [229]. Off-policy methods are usually significantly more sample efficient. However, offpolicy methods have been harder to extend to meta-RL, leading to more meta-RL methods being built on on-policy RL methods, thus limiting the absolute performance of meta-RL. Early work in off-policy meta-RL methods has led to strong results [114], [121], [171], [228]. Off-policy learning also improves the efficiency of the meta-train stage [114], which can be expensive in meta-RL. It also provides new opportunities to accelerate meta-testing by replay buffer sample from meta-training [171].

开策略与关策略元RL传统RL中的一个主要二分法是开策略和关策略学习，如PPO[225]与SAC[229]。非策略方法通常更具样本效率。然而，非策略方法更难扩展到元RL，导致更多的元RL方法建立在策略RL方法上，从而限制了元RL的绝对性能。非策略元RL方法的早期工作已经产生了强有力的结果[114]，[121]，[171]，[228]。非策略学习还提高了元训练阶段的效率[114]，这在元RL中可能是昂贵的。它还提供了通过重放元训练的缓冲样本来加速元测试的新机会[171]。

Other Trends and Challenges. [65] is noteworthy in demonstrating successful meta-RL on a real-world physical robot. Knowledge transfer in robotics is often best studied compositionally [230]. E.g., walking, navigating and object pick/place may be subroutines for a room cleaning robot. However, developing meta-learners with effective compositional knowledge transfer is an open question, with modular meta-learning [141] being an option. Unsupervised metaRL variants aim to perform meta-training without manually specified rewards [231], or adapt at meta-testing to a changed environment but without new rewards [232]. Continual adaptation provides an agent with the ability to adapt to a sequence of tasks within one meta-test episode [63]–[65], similar to continual learning. Finally, meta-learning has also been applied to imitation [115] and inverse RL [233].

其他趋势和挑战。[65]值得注意的是，在真实世界的物理机器人上演示了成功的元RL。机器人技术中的知识迁移通常在组合方面研究得最好[230]。E、 例如，行走、导航和对象拾取/放置可以是房间清洁机器人的子程序。然而，培养具有有效合成知识迁移的元学习者是一个开放的问题，模块化元学习[141]是一个选项。无监督的metaRL变体旨在在没有手动指定奖励的情况下执行元训练[231]，或者在元测试中适应变化的环境，但没有新的奖励[232]。持续适应为主体提供了适应一个元测试集内一系列任务的能力[63]-[65]，类似于持续学习。最后，元学习也应用于模仿[115]和逆RL[233]。

#### 5.2.2 Benchmarks
Meta-learning benchmarks for RL typically define a family to solve in order to train and evaluate an agent that learns how to learn. These can be tasks (reward functions) to achieve, or domains (distinct environments or MDPs).

RL的元学习基准通常定义要解决的族，以便训练和评估学习如何学习的智能体。这些可以是要实现的任务(奖励功能)，也可以是域(不同的环境或MDP)。

Discrete Control RL. An early meta-RL benchmark for vision-actuated control is the arcade learning environment (ALE) [234], which defines a set of classic Atari games split into meta-training and meta-testing. The protocol here is to evaluate return after a fixed number of timesteps in the meta-test environment. A challenge is the great diversity (wide p(T )) across games, which makes successful metatraining hard and leads to limited benefit from knowledge transfer [234]. Another benchmark [235] is based on splitting Sonic-hedgehog levels into meta-train/meta-test. The task distribution here is narrower and beneficial meta-learning is relatively easier to achieve. Cobbe et al. [236] proposed two purpose designed video games for benchmarking meta-RL. CoinRun game [236] provides 2 32 procedurally generated levels of varying difficulty and visual appearance. They show that some 10, 000 levels of meta-train experience are required to generalize reliably to new levels. CoinRun is primarily designed to test direct generalization rather than fast adaptation, and can be seen as providing a distribution over MDP environments to test generalization rather than over tasks to test adaptation. To better test fast learning in a wider task distribution, ProcGen [236] provides a set of 16 procedurally generated games including CoinRun.

离散控制RL。视觉驱动控制的早期元RL基准是街机学习环境(ALE)[234]，它定义了一组分为元训练和元测试的经典雅达利游戏。这里的协议是在元测试环境中的固定次数的时间步之后评估返回。一个挑战是游戏之间的巨大多样性(宽p(T))，这使得成功的元训练变得困难，并导致知识迁移带来的收益有限[234]。另一个基准[235]基于将Sonic hedgehog水平拆分为元训练/元测试。这里的任务分配更窄，有益的元学习相对更容易实现。Cobbeet al [236]提出了两种专门设计的视频游戏，用于对元RL进行基准测试。CoinRun游戏[236]提供了232个不同难度和视觉外观的程序生成级别。他们表明，要可靠地推广到新的水平，需要大约10000级的元训练经验。CoinRun主要用于测试直接泛化，而不是快速适应，它可以被视为在MDP环境上提供一个分布来测试泛化，但不是在任务上测试适应。为了在更广泛的任务分布中更好地测试快速学习，ProcGen[236]提供了一组16个程序生成的游戏，包括CoinRun。

Continuous Control RL. While common benchmarks such as gym [237] have greatly benefited RL research, there is less consensus on meta-RL benchmarks, making existing work hard to compare. Most continuous control meta-RL studies have proposed home-brewed benchmarks that are low dimensional parametric variants of particular tasks such as navigating to various locations or velocities [16], [114], or traversing different terrains [65]. Several multi-MDP benchmarks [238], [239] have recently been proposed but these primarily test generalization across different environmental perturbations rather than different tasks. The Meta-World benchmark [240] provides a suite of 50 continuous control tasks with state-based actuation, varying from simple parametric variants such as lever-pulling and door-opening. This benchmark should enable more comparable evaluation, and investigation of generalization within and across task distributions. The meta-world evaluation [240] suggests that existing meta-RL methods struggle to generalize over wide task distributions and meta-train/meta-test shifts. This may be due to our meta-RL models being too weak and/or benchmarks being too small, in terms of number and coverage tasks, for effective learning-to-learn. Another recent benchmark suitable for meta-RL is PHYRE [241] which provides a set of 50 vision-based physics task templates which can be solved with simple actions but are likely to require model-based reasoning to address efficiently. These also provide within and cross-template generalization tests.

连续控制RL。虽然健身房等常见基准测试[237]极大地促进了RL研究，但对元RL基准测试的共识较少，使得现有的研究难以比较。大多数连续控制元RL研究都提出了自制基准，这些基准是特定任务的低维参数变量，例如导航到不同位置或速度[16]、[114]或穿越不同地形[65]。最近提出了几个多MDP基准[238]、[239]，但这些基准主要测试不同环境扰动的泛化，而不是不同任务。Meta World基准[240]提供了一套50个连续控制任务，具有基于状态的驱动，不同于简单的参数变量，如杠杆拉动和开门。该基准应该能够进行更具可比性的评估，并调查任务分布内部和跨任务分布的泛化。元世界评估[240]表明，现有的元RL方法难以在广泛的任务分布和元训练/元测试迁移上进行推广。这可能是由于我们的元RL模型太弱和/或基准太小，在数量和覆盖任务方面，无法有效学习。另一个适用于元RL的最新基准是PHYRE[241]，它提供了一组50个基于视觉的物理任务模板，这些模板可以用简单的动作解决，但可能需要基于模型的推理才能有效解决。这些还提供了模板内和跨模板的泛化测试。

Discussion. One complication of vision-actuated metaRL is disentangling visual generalization (as in computer vision) with fast learning of control strategies more generally. For example CoinRun [236] evaluation showed large benefit from standard vision techniques such as batch norm suggesting that perception is a major bottleneck.

讨论视觉驱动的metaRL的一个复杂性是将视觉泛化(如计算机视觉)与控制策略的快速学习更为普遍地分离开来。例如，CoinRun[236]评估显示，标准视觉技术(如批量规范)带来了巨大的好处，这表明感知是一个主要瓶颈。

### 5.3 Environment Learning and Sim2Real
In Sim2Real we are interested in training a model in simulation that is able to generalize to the real-world. The classic domain randomization approach simulates a wide distribution over domains/MDPs, with the aim of training a sufficiently robust model to succeed in the real world – and has succeeded in both vision [242] and RL [161]. Nevertheless tuning the simulation distribution remains a challenge. This leads to a meta-learning setup where the inner-level optimization learns a model in simulation, the outer-level optimization L meta evaluates the model’s performance in the real-world, and the meta-representation ω corresponds to the parameters of the simulation environment. This paradigm has been used in RL [163] as well as vision [162], [243]. In this case the source tasks used for meta-train tasks are not a pre-provided data distribution, but paramaterized by omega, Dsource(ω). However, challenges remain in terms of costly back-propagation through a long graph of inner task learning steps; as well as minimising the number of real-world L meta evaluations in the case of Sim2Real.

在Sim2Real中，我们感兴趣的是在模拟中训练一个能够推广到现实世界的模型。经典的域随机化方法模拟了域/MDP上的广泛分布，目的是训练一个足够稳健的模型以在现实世界中取得成功——并且在视觉[242]和RL[161]中都取得了成功。然而，调整模拟分布仍然是一个挑战。这导致了元学习设置，其中内部级优化学习模拟中的模型，外部级优化L元评估模型在真实世界中的性能，元表示ω对应于模拟环境的参数。RL[163]以及视觉[162]、[243]中都使用了这种范式。在这种情况下，用于元训练任务的源任务不是预先提供的数据分布，而是由ω，Dsource(ω)参数化。然而，挑战仍然存在，即通过内部任务学习步骤的长图进行成本高昂的反向传播; 以及在Sim2Real的情况下最小化真实世界L元评估的数量。

### 5.4 Neural Architecture Search (NAS)
Architecture search [18], [25], [26], [37], [131] can be seen as a kind of hyperparameter optimization where ω specifies the architecture of a neural network. The inner optimization trains networks with the specified architecture, and the outer optimization searches for architectures with good validation performance. NAS methods have been analysed [37] according to ‘search space’, ‘search strategy’, and ‘performance estimation strategy’. These correspond to the hypothesis space for ω, the meta-optimization strategy, and the meta-objective. NAS is particularly challenging because: (i) Fully evaluating the inner loop is expensive since it requires training a many-shot neural network to completion. This leads to approximations such as sub-sampling the train set, early termination of the inner loop, and interleaved descent on both ω and θ [18] as in online meta-learning. (ii.) The search space is hard to define, and optimize. This is because most search spaces are broad, and the space of architectures is not trivially differentiable. This leads to reliance on celllevel search [18], [26] constraining the search space, RL [26], discrete gradient estimators [133] and evolution [25], [131].

架构搜索[18]、[25]、[26]、[37]、[131]可以看作是一种超参数优化，其中ω指定了神经网络的架构。内部优化训练具有指定架构的网络，外部优化搜索具有良好验证性能的架构。根据“搜索空间”、“搜索策略”和“性能评估策略”对NAS方法进行了分析[37]。这些对应于ω的假设空间、元优化策略和元目标。NAS特别具有挑战性，因为：(i)完全评估内部环路是昂贵的，因为它需要训练多目标神经网络才能完成。这导致了近似，如对训练集进行二次采样、内环的提前终止以及在线元学习中ω和θ的交错下降[18]。(ii)搜索空间很难定义和优化。这是因为大多数搜索空间都很宽，并且架构的空间不是微不足道的可区分的。这导致依赖于细胞级搜索[18]，[26]约束搜索空间，RL[26]，离散梯度估计器[133]和进化[25]，[131]。

Topical Issues. While NAS itself can be seen as an instance of hyper-parameter or hypothesis-class meta-learning, it can also interact with meta-learning in other forms. Since NAS is costly, a topical issue is whether discovered architectures can generalize to new problems [244]. Meta-training across multiple datasets may lead to improved cross-task generalization of architectures [136]. Finally, one can also define NAS meta-objectives to train an architecture suitable for few-shot learning [245], [246]. Similarly to fast-adapting initial condition meta-learning approaches such as MAML [16], one can train good initial architectures [135] or architecture priors [136] that are easy to adapt towards specific tasks.

热点问题。虽然NAS本身可以被视为超参数或假设类元学习的实例，但它也可以以其他形式与元学习相互作用。由于NAS成本高昂，一个热门问题是发现的架构是否可以概括为新问题[244]。跨多个数据集的元训练可以改进架构的跨任务泛化[136]。最后，还可以定义NAS元目标，以训练适合于少量学习的架构[245]，[246]。类似于快速适应初始条件元学习方法，如MAML[16]，可以训练好的初始架构[135]或易于适应特定任务的架构先验[136]。

Benchmarks. NAS is often evaluated on CIFAR-10, but it is costly to perform and results are hard to reproduce due to confounding factors such as tuning of hyperparameters [247]. To support reproducible and accessible research, the NASbenches [248] provide pre-computed performance measures for a large number of network architectures.

基准。NAS通常在CIFAR-10上进行评估，但执行成本很高，并且由于超参数调整等混杂因素，结果很难再现[247]。为了支持可复制和可访问的研究，NASbenches[248]为大量网络架构提供了预先计算的性能度量。

### 5.5 Bayesian Meta-learning
Bayesian meta-learning approaches formalize meta-learning via Bayesian hierarchical modelling, and use Bayesian inference for learning rather than direct optimization of parameters. In the meta-learning context, Bayesian learning is typically intractable, and so approximations such as stochastic variational inference or sampling are used.

贝叶斯元学习方法通过贝叶斯分层建模将元学习形式化，并使用贝叶斯推理进行学习，而不是直接优化参数。在元学习环境中，贝叶斯学习通常很难处理，因此使用了随机变分推理或抽样等近似方法。

Bayesian meta-learning importantly provides uncertainty measures for the ω parameters, and hence measures of prediction uncertainty which can be important for safety critical applications, exploration in RL, and active learning.

贝叶斯元学习重要地为ω参数提供了不确定性度量，因此，预测不确定性度量对于安全关键应用、RL探索和主动学习非常重要。

A number of authors have explored Bayesian approaches to meta-learning complex neural network models with competitive results. For example, extending variational autoencoders to model task variables explicitly [75]. Neural Processes [179] define a feed-forward Bayesian meta-learner inspired by Gaussian Processes but implemented with neural networks. Deep kernel learning is also an active research area that has been adapted to the meta-learning setting [249], and is often coupled with Gaussian Processes [250]. In [76] gradient based meta-learning is recast into a hierarchical empirical Bayes inference problem (i.e. prior learning), which models uncertainty in task-specific parameters θ. Bayesian MAML [251] improves on this model by using a Bayesian ensemble approach that allows non-Gaussian posteriors over θ, and later work removes the need for costly ensembles [45], [252]. In Probabilistic MAML [98], it is the uncertainty in the metaknowledge ω that is modelled, while a MAP estimate is used for θ. Increasingly, these Bayesian methods are shown to tackle ambiguous tasks, active learning and RL problems.

许多作者探索了贝叶斯方法来元学习复杂的神经网络模型，并获得了具有竞争力的结果。例如，扩展变分自动编码器以显式建模任务变量[75]。神经过程[179]定义了前馈贝叶斯元学习器，其灵感来自高斯过程，但使用神经网络实现。深度核学习也是一个活跃的研究领域，已经适应了元学习设置[249]，并且经常与高斯过程结合[250]。在[76]中，基于梯度的元学习被重构为分层经验贝叶斯推理问题(即先验学习)，该问题对任务特定参数θ中的不确定性进行建模。贝叶斯MAML[251]通过使用贝叶斯集成方法改进了该模型，该方法允许θ上的非高斯后验，随后的工作消除了对昂贵集成的需要[45]，[252]。在概率MAML[98]中，建模的是元知识ω中的不确定性，而对θ使用MAP估计。越来越多的贝叶斯方法被证明能够解决模糊任务、主动学习和RL问题。

Separate from the above, meta-learning has also been proposed to aid the Bayesian inference process itself, as in [253] where the authors adapt a Bayesian sampler to provide efficient adaptive sampling methods.

与上述内容不同，元学习也被提出来帮助贝叶斯推理过程本身，如[253]中所述，作者使用贝叶斯采样器来提供有效的自适应采样方法。

### 5.6 Unsupervised Meta-Learning
There are several distinct ways in which unsupervised learning can interact with meta-learning, depending on whether unsupervised learning in performed in the inner loop or outer loop, and during meta-train vs meta-test.

无监督学习可以通过几种不同的方式与元学习相互作用，这取决于无监督学习是在内部循环还是外部循环中进行的，以及元训练与元测试之间的关系。

Unsupervised Learning of a Supervised Learner. The aim here is to learn a supervised learning algorithm (e.g., via MAML [16] style initial condition for supervised finetuning), but do so without the requirement of a large set of source tasks for meta-training [254]–[256]. To this end, synthetic source tasks are constructed without supervision via clustering or class-preserving data augmentation, and used to define the meta-objective for meta-training.

受监督学习者的无监督学习。这里的目的是学习监督学习算法(例如，通过MAML[16]式的监督微调初始条件)，但这样做不需要大量的源任务用于元训练[254]–[256]。为此，合成源任务在没有监督的情况下通过聚类或类保留数据增广来构建，并用于定义元训练的元目标。

Supervised Learning of an Unsupervised Learner. This family of methods aims to meta-train an unsupervised learner. For example, by training the unsupervised algorithm such that it works well for downstream supervised learning tasks. One can train unsupervised learning rules [21] or losses [101], [125] such that downstream supervised learning performance is optimized – after re-using the unsupervised representation for a supervised task [21], or adapting based on unlabeled data [101], [125]. Alternatively, when unsupervised tasks such as clustering exist in a family, rather than in isolation, then learning-to-learn of ‘how-to-cluster’ on several source tasks can provide better performance on new clustering tasks in the family [180]– [182], [257], [258]. The methods in this group that make use of feed-forward models are often known as amortized clustering [181], [182], because they amortize the typically iterative computation of clustering algorithms into the cost of training a single inference model, which subsequently 13 performs clustering using a single feed-froward pass. Overall, these methods help to deal with the ill-definedness of the unsupervised learning problem by transforming it into a problem with a clear supervised (meta) objective.

无监督学习者的监督学习。这一系列方法旨在对无监督学习者进行元训练。例如，通过训练无监督算法，使其能够很好地用于下游监督学习任务。人们可以训练无监督学习规则[21]或损失[101]、[125]，以便在对监督任务[21]重新使用无监督表示或基于未标注数据进行调整[101]和[125]之后，优化下游监督学习性能。或者，当无监督任务(如集群)存在于一个家族中而非孤立时，学习如何在几个源任务上“集群”可以在家族中的新集群任务上提供更好的性能[180]–[182]，[257]，[258]。该组中使用前馈模型的方法通常被称为摊余聚类[181]，[182]，因为它们将聚类算法的典型迭代计算分摊到训练单个推理模型的成本中，该模型随后使用单个前馈传递执行聚类，这些方法通过将无监督学习问题转化为具有明确监督(元)目标的问题，有助于处理无监督学习的模糊性。

### 5.7 Continual, Online and Adaptive Learning
Continual Learning. refers to the human-like capability of learning tasks presented in sequence. Ideally this is done while exploiting forward transfer so new tasks are learned better given past experience, without forgetting previously learned tasks, and without needing to store past data [62]. Deep Neural Networks struggle to meet these criteria, especially as they tend to forget information seen in earlier tasks – a phenomenon known as catastrophic forgetting. Meta-learning can include the requirements of continual learning into a meta-objective, for example by defining a sequence of learning episodes in which the support set contains one new task, but the query set contains examples drawn from all tasks seen until now [107], [174]. Various meta-representations can be learned to improve continual learning performance, such as weight priors [138], gradient descent preconditioning matrices [107], or RNN learned optimizers [174], or feature representations [259]. A related idea is meta-training representations to support local editing updates [260] for improvement without interference.

持续学习。是指按顺序呈现的学习任务的类人能力。理想情况下，这是在利用前向传输的同时完成的，以便在过去的经验下更好地学习新任务，而不会忘记以前学习的任务，并且不需要存储过去的数据[62]。深度神经网络很难满足这些标准，尤其是当它们倾向于忘记在早期任务中看到的信息时——这种现象被称为灾难性遗忘。元学习可以将持续学习的要求包括在元目标中，例如，通过定义一系列学习事件，其中支持集包含一个新任务，但查询集包含从迄今为止看到的所有任务中提取的样本[107]，[174]。可以学习各种元表示来提高连续学习性能，例如权重先验[138]、梯度下降预处理矩阵[107]或RNN学习优化器[174]或特征表示[259]。一个相关的想法是元训练表示来支持本地编辑更新[260]，以便在没有干扰的情况下进行改进。

Online and Adaptive Learning also consider tasks arriving in a stream, but are concerned with the ability to effectively adapt to the current task in the stream, more than remembering the old tasks. To this end an online extension of MAML was proposed [99] to perform MAML-style meta-training online during a task sequence. Meanwhile others [63]–[65] consider the setting where meta-training is performed in advance on source tasks, before meta-testing adaptation capabilities on a sequence of target tasks.

在线学习和自适应学习也考虑到流中的任务，但更关注的是有效适应流中当前任务的能力，而不是记住旧任务。为此，提出了MAML的在线扩展[99]，以在任务序列期间在线执行MAML风格的元训练。同时，其他人[63]–[65]考虑在对一系列目标任务的适应能力进行元测试之前，对源任务预先进行元训练的设置。

Benchmarks. A number of benchmarks for continual learning work quite well with standard deep learning methods. However, most cannot readily work with meta-learning approaches as their their sample generation routines do not provide a large number of explicit learning sets and an explicit evaluation sets. Some early steps were made towards defining meta-learning ready continual benchmarks in [99], [174], [259], mainly composed of Omniglot and perturbed versions of MNIST. However, most of those were simply tasks built to demonstrate a method. More explicit benchmark work can be found in [219], which is built for meta and non meta-learning approaches alike.

基准。许多持续学习的基准与标准的深度学习方法非常匹配。然而，大多数人无法轻易使用元学习方法，因为他们的样本生成例程没有提供大量的显式学习集和显式评估集。在[99]、[174]和[259]中，在定义元学习就绪的连续基准方面采取了一些早期步骤，主要由Omniglot和MNIST的扰动版本组成。然而，其中大多数只是为了演示方法而构建的任务。更明确的基准工作可以在[219]中找到，它是为元学习和非元学习方法构建的。

### 5.8 Domain Adaptation and Domain Generalization
Domain-shift refers to the statistics of data encountered in deployment being different from those used in training. Numerous domain adaptation and generalization algorithms have been studied to address this issue in supervised, unsupervised, and semi-supervised settings [58].

域迁移是指部署中遇到的数据与培训中使用的数据不同的统计数据。已经研究了许多域自适应和泛化算法来解决监督、无监督和半监督设置中的这一问题[58]。

Domain Generalization. Domain generalization aims to train models with increased robustness to train-test domain shift [261], often by exploiting a distribution over training domains. Using a validation domain that is shifted with respect to the training domain [262], different kinds of metaknowledge such as regularizers [95], losses [42], and noise augmentation [119] can be (meta) learned to maximize the robustness of the learned model to train-test domain-shift.

域泛化。领域泛化旨在训练具有更强稳健性的模型，以训练测试领域迁移[261]，通常通过利用训练领域上的分布。使用相对于训练域偏移的验证域[262]，可以(元)学习不同类型的元知识，如正则化器[95]、损失[42]和噪声增强[119]，以最大化学习模型对训练测试域偏移的稳健性。

Domain Adaptation. To improve on conventional domain adaptation [58], meta-learning can be used to define a metaobjective that optimizes the performance of a base unsupervised DA algorithm [59].

域适应。为了改进传统的域自适应[58]，元学习可以用于定义优化基本无监督DA算法性能的元目标[59]。

Benchmarks. Popular benchmarks for DA and DG consider image recognition across multiple domains such as photo/sketch/cartoon. PACS [263] provides a good starter benchmark, with Visual Decathlon [42], [220] and MetaDataset [111] providing larger scale alternatives.

基准。DA和DG的流行基准考虑跨照片/草图/卡通等多个领域的图像识别。PACS[263]提供了一个良好的初学者基准，Visual Decathlon[42]、[220]和MetaDataset[111]提供了更大规模的替代方案。

### 5.9 Hyper-parameter Optimization
Meta-learning address hyperparameter optimization when considering ω to specify hyperparameters, such as regularization strength or learning rate. There are two main settings: we can learn hyperparameters that improve training over a distribution of tasks, just a single task. The former case is usually relevant in few-shot applications, especially in optimization based methods. For instance, MAML can be improved by learning a learning rate per layer per step [80]. The case where we wish to learn hyperparameters for a single task is usually more relevant for many-shot applications [71], [157], where some validation data can be extracted from the training dataset, as discussed in Section 2.1. End-to-end gradient-based meta-learning has already demonstrated promising scalability to millions of parameters (as demonstrated by MAML [16] and Dataset Distillation [156], [157], for example) in contrast to the classic approaches (such cross-validation by grid or random [72] search, or Bayesian Optimization [73]) which are typically only successful with dozens of hyper-parameters.

当考虑ω来指定超参数(如正则化强度或学习率)时，元学习解决了超参数优化问题。有两种主要设置：我们可以学习超参数，这些超参数可以改进任务分布的训练，而不是单个任务。前一种情况通常适用于少量应用，尤其是基于优化的方法。例如，MAML可以通过每个步骤学习每层的学习速率来改进[80]。我们希望学习单个任务的超参数的情况通常与多样本应用程序更相关[71]，[157]，其中可以从训练数据集中提取一些验证数据，如第2.1节所述。端到端基于梯度的元学习已经证明了对数百万个参数的良好可扩展性(例如，MAML[16]和数据集提取[156]，[157]所证明的)，而经典方法(例如，通过网格或随机[72]搜索的交叉验证，或贝叶斯优化[73])通常只在几十个超参数下成功。

### 5.10 Novel and Biologically Plausible Learners
Most meta-learning work that uses explicit (non feedforward/black-box) optimization for the base model is based on gradient descent by backpropagation. Metalearning can define the function class of ω so as to lead to the discovery of novel learning rules that are unsupervised [21] or biologically plausible [46], [264], [265], making use of ideas less commonly used in contemporary deep learning such as Hebbian updates [264] and neuromodulation [265].

大多数对基础模型使用显式(非前馈/黑箱)优化的元学习工作都是基于反向传播的梯度下降。元学习可以定义ω的函数类，从而发现新的学习规则，这些规则是无监督的[21]或生物学上合理的[46]、[264]、[265]，利用当代深度学习中不常用的思想，例如Hebbian更新[264]和神经调控[265]。

### 5.11 Language and Speech
Language Modelling Few-shot language modelling increasingly showcases the versatility of meta-learners. Early matching networks showed impressive performances on one-shot tasks such as filling in missing words [90]. Many more tasks have since been tackled, including text classifi- cation [139], neural program induction [266] and synthesis [267], English to SQL program synthesis [268], text-based relationship graph extractor [269], machine translation [270], and quickly adapting to new personas in dialogue [271]. Speech Recognition. Deep learning is now the dominant paradigm for state of the art automatic speech recognition (ASR). Meta-learning is beginning to be applied to address the many few-shot adaptation problems that arise within ASR including learning how to train for low-resource languages [272], cross-accent adaptation [273] and optimizing models for individual speakers [274]. 

语言建模很少有样本语言建模越来越多地展示了元学习者的多功能性。早期的匹配网络在一次性任务中表现出令人印象深刻的表现，例如填写缺失的单词[90]。此后，已经处理了更多的任务，包括文本分类[139]、神经程序归纳[266]和合成[267]、英语到SQL程序合成[268]、基于文本的关系图提取器[269]、机器翻译[270]，以及快速适应对话中的新角色[271]。语音识别深度学习现在是最先进的自动语音识别(ASR)的主要范例。元学习开始被应用于解决ASR中出现的许多少数样本适应问题，包括学习如何训练低资源语言[272]、交叉重音适应[273]和优化单个说话人的模型[274]。

### 5.12 Meta-learning for Social Good
Meta-learning lands itself to various challenging tasks that arise in applications of AI for social good such as medical image classification and drug discovery, where data is often scarce. Progress in the medical domain is especially relevant given the global shortage of pathologists [275]. In [5] an LSTM is combined with a graph neural network to predict the behaviour of a molecule (e.g. its toxicity) in the oneshot data regime. In [276] MAML is adapted to weaklysupervised breast cancer detection tasks, and the order of tasks are selected according to a curriculum. MAML is also combined with denoising autoencoders to do medical visual question answering [277], while learning to weigh support samples [218] is adapted to pixel wise weighting for skin lesion segmentation tasks that have noisy labels [278].

元学习将自身应用于各种具有挑战性的任务，这些任务出现在人工智能的社会公益应用中，如医学图像分类和药物发现，而这些领域的数据往往很稀缺。鉴于全局病理学家短缺，医学领域的进展尤其重要[275]。在[5]中，LSTM与图神经网络相结合，以预测单次数据状态下分子的行为(例如其毒性)。在[276]中，MAML适用于弱监督的乳腺癌检测任务，任务的顺序根据课程选择。MAML还与去噪自动编码器相结合，以进行医学视觉问题解答[277]，而学习加权支持样本[218]适用于具有噪声标签的皮肤损伤分割任务的像素加权[278]。

### 5.13 Abstract Reasoning
A long- term goal in deep learning is to go beyond simple perception tasks and tackle more abstract reasoning problems such as IQ tests in the form of Raven’s Progressive Matrices (RPMs) [279]. Solving RPMs can be seen as asking for few-shot generalization from the context panels to the answer panels. Recent meta-learning approaches to abstract reasoning with RPMs achieved significant improvement via meta-learning a teacher that defines the data generating distribution for the panels [280]. The teacher is trained jointly with the student, and rewarded by the student’s progress.

深度学习的一个长期目标是超越简单的感知任务，解决更抽象的推理问题，例如以乌鸦渐进矩阵(RPM)的形式进行的智商测试[279]。解决RPM可以被视为要求从上下文面板到答案面板的简单概括。最近使用RPM进行抽象推理的元学习方法通过元学习获得了显著改进，该教师定义了面板的数据生成分布[280]。教师与学生共同训练，并根据学生的进步给予奖励。

### 5.14 Systems
Network Compression. Contemporary CNNs require large amounts of memory that may be prohibitive on embedded devices. Thus network compression in various forms such as quantization and pruning are topical research areas [281]. Meta-learning is beginning to be applied to this objective as well, such as training gradient generator metanetworks that allow quantized networks to be trained [202], and weight generator meta-networks that allow quantized networks to be trained with gradient [282].

网络压缩。当代CNN需要大量的内存，这在嵌入式设备上可能是禁止的。因此，量化和剪枝等各种形式的网络压缩是热门研究领域[281]。元学习也开始应用于这一目标，例如训练允许训练量化网络的梯度生成器元网络[202]，以及允许使用梯度训练量化网络[282]的权重生成器元网络。

Communications. Deep learning is rapidly impacting communications systems. For example by learning coding systems that exceed the best hand designed codes for realistic channels [283]. Few-shot meta-learning can be used to provide rapid adaptation of codes to changing channel characteristics [284].

通信。深度学习正在迅速影响通信系统。例如，通过学习超出实际信道最佳手工设计代码的编码系统[283]。可以使用少量元学习来提供代码对不断变化的信道特性的快速适应[284]。

Active Learning (AL). methods wrap supervised learning, and define a policy for selective data annotation – typically in the setting where annotation can be obtained sequentially. The goal of AL is to find the optimal subset of data to annotate so as to maximize performance of downstream supervised learning with the fewest annotations. AL is a well studied problem with numerous hand designed algorithms [285]. Meta-learning can map active learning algorithm design into a learning task by: (i) defining the inner-level optimization as conventional supervised learning on the annotated dataset so far, (ii) defining ω to be a query policy that selects the best unlabeled datapoints to annotate, (iii), defining the meta-objective as validation performance after iterative learning and annotation according to the query policy, (iv) performing outer-level optimization to train the optimal annotation query policy [190]–[192]. However, if labels are used to train AL algorithms, they need to generalize across tasks to amortize their training cost [192].

主动学习(AL)。方法包装监督学习，并为选择性数据注释定义策略&通常在可以顺序获得注释的设置中。AL的目标是找到要注释的数据的最优子集，以便以最少的注释最大化下游监督学习的性能。AL是一个研究得很好的问题，有许多手工设计的算法[285]。元学习可以通过以下方式将主动学习算法设计映射到学习任务中：(i)将内部优化定义为迄今为止对带注释数据集的常规监督学习; (ii)将ω定义为选择最佳未标注数据点进行注释的查询策略，将元目标定义为根据查询策略进行迭代学习和注释后的验证性能，(iv)执行外部级优化以训练最优注释查询策略[190]–[192]。然而，如果标签用于训练AL算法，则需要在任务之间进行泛化，以分摊其训练成本[192]。

Learning with Label Noise. commonly arises when large datasets are collected by web scraping or crowd-sourcing. While there are many algorithms hand-designed for this situation, recent meta-learning methods have addressed label noise. For example by transductively learning sample-wise weighs to down-weight noisy samples [151], or learning an initial condition robust to noisy label training [96].

使用标签噪声学习。当通过web抓取或众包收集大型数据集时，通常会出现这种情况。尽管有许多算法是为这种情况手工设计的，但最近的元学习方法已经解决了标签噪声问题。例如，通过反导学习逐样本加权以降低噪声样本的权重[151]，或学习对噪声标签训练稳健的初始条件[96]。

Adversarial Attacks and Defenses. Deep Neural Networks can be fooled into misclassifying a data point that should be easily recognizable, by adding a carefully crafted human-invisible perturbation to the data [286]. Numerous attack and defense methods have been published in recent years, with defense strategies usually consisting in carefully hand-designed architectures or training algorithms. Analogous to the case in domain-shift, one can train the learning algorithm for robustness by defining a meta-loss in terms of performance under adversarial attack [97], [287].

对手进攻和防守。深度神经网络可以通过在数据中添加精心设计的人类不可见扰动来欺骗错误分类应该很容易识别的数据点[286]。近年来，已经发表了许多攻击和防御方法，防御策略通常由精心设计的架构或训练算法组成。与域迁移的情况类似，可以通过定义对抗性攻击下性能的元损失来训练学习算法的稳健性[97]，[287]。

Recommendation Systems are a mature consumer of machine learning in the commerce space. However, bootstrapping recommendations for new users with little historical interaction data, or new items for recommendation remains a challenge known as the cold-start problem. Meta-learning has applied black-box models to item cold-start [288] and gradient-based methods to user cold-start [289]. 

推荐系统是商业领域机器学习的成熟消费者。然而，为历史交互数据很少的新用户或新的推荐项目提供引导推荐仍然是一个被称为冷启动问题的挑战。元学习已将黑盒模型应用于项目冷启动[288]，并将基于梯度的方法应用于用户冷启动[289]。

## 6 CHALLENGES AND OPEN QUESTIONS
Diverse and multi-modal task distributions. The diffi- culty of fitting a meta-learner to a distribution of tasks p(T ) can depend on its width. Many big successes of meta-learning have been within narrow task families, while learning on diverse task distributions can challenge existing methods [111], [220], [240]. This may be partly due to conflicting gradients between tasks [290].

多样化和多模式任务分配。使元学习者适应任务p(T)分布的难度取决于其宽度。元学习的许多重大成功都是在狭窄的任务家族中，而在不同任务分布上的学习可能会挑战现有的方法[111]、[220]、[240]。这可能部分是由于任务之间的梯度冲突[290]。

Many meta-learning frameworks [16] implicitly assume that the distribution over tasks p(T ) is uni-modal, and a single learning strategy ω provides a good solution for them all. However task distributions are often multi-modal; such as medical vs satellite vs everyday images in computer vision, or putting pegs in holes vs opening doors [240] in robotics. Different tasks within the distribution may require different learning strategies, which is hard to achieve with today’s methods. In vanilla multi-task learning, this phenomenon is relatively well studied with, e.g., methods that group tasks into clusters [291] or subspaces [292]. However this is only just beginning to be explored in meta-learning [293].

许多元学习框架[16]隐含地假设任务p(T)的分布是单峰的，并且单个学习策略ω为它们提供了很好的解决方案。然而，任务分配通常是多模式的; 比如计算机视觉中的医疗与卫星对比日常图像，或者机器人技术中的钉入洞对比开门[240]。分布中的不同任务可能需要不同的学习策略，这在当今的方法中很难实现。在普通的多任务学习中，这一现象通过将任务分组为簇[291]或子空间[292]的方法得到了比较好的研究。然而，这在元学习中才刚刚开始探索[293]。

Meta-generalization. Meta-learning poses a new generalization challenge across tasks analogous to the challenge of generalizing across instances in conventional machine learning. There are two sub-challenges: (i) The first is generalizing from meta-train to novel meta-test tasks drawn from p(T ). This is exacerbated because the number of tasks available for meta-training is typically low (much less than the number of instances available in conventional supervised learning), making it difficult to generalize. One failure mode for generalization in few-shot learning has been well studied 15 under the guise of memorisation [204], which occurs when each meta-training task can be solved directly without performing any task-specific adaptation based on the support set. In this case models fail to generalize in meta-testing, and specific regularizers [204] have been proposed to prevent this kind of meta-overfitting. (ii) The second challenge is generalizing to meta-test tasks drawn from a different distribution than the training tasks. This is inevitable in many potential practical applications of meta-learning, for example generalizing few-shot visual learning from everyday training images of ImageNet to specialist domains such as medical images [221]. From the perspective of a learner, this is a meta-level generalization of the domain-shift problem, as observed in supervised learning. Addressing these issues through meta-generalizations of regularization, transfer learning, domain adaptation, and domain generalization are emerging directions [119]. Furthermore, we have yet to understand which kinds of meta-representations tend to generalize better under certain types of domain shifts.

元泛化。元学习提出了一个新的跨任务泛化挑战，类似于传统机器学习中跨实例泛化的挑战。有两个子挑战：(i)第一个是从元训练到从p(T)中提取的新元测试任务的概括。由于元训练可用的任务数量通常较低(比传统监督学习中可用的实例数量少得多)，这就加剧了这种情况，使其难以概括。少数样本学习中的一种泛化失败模式已经在记忆的伪装下得到了很好的研究15[204]，当每个元训练任务都可以直接解决而无需基于支持集执行任何特定任务的自适应时，就会出现这种情况。在这种情况下，模型无法在元测试中推广，并且已经提出了特定的正则化器[204]来防止这种元过度拟合。(ii)第二个挑战是从不同于训练任务的分布中提取元测试任务。这在元学习的许多潜在实际应用中是不可避免的，例如，将很少样本的视觉学习从ImageNet的日常训练图像推广到医学图像等专业领域[222]。从学习者的角度来看，这是在监督学习中观察到的领域迁移问题的元级概括。通过正则化、迁移学习、域自适应和域泛化的元泛化来解决这些问题是新兴的方向[119]。此外，我们还没有了解在某些类型的域迁移下，哪些类型的元表示倾向于更好地概括。

Task families. Many existing meta-learning frameworks, especially for few-shot learning, require task families for meta-training. While this indeed reflects lifelong human learning, in some applications data for such task families may not be available. Unsupervised meta-learning [254]– [256] and single-task meta-learning methods [42], [175], [183], [184], [200], could help to alleviate this requirement; as can improvements in meta-generalization discussed above.

任务族。许多现有的元学习框架，特别是针对少量学习的框架，需要任务族进行元训练。虽然这确实反映了人类的终身学习，但在某些应用程序中，可能无法获得此类任务族的数据。无监督元学习[254]–[256]和单任务元学习方法[42]、[175]、[183]、[184]、[200]可以帮助缓解这一需求; 正如上面讨论的元泛化的改进一样。

Computation Cost & Many-shot. A naive implementation of bilevel optimization as shown in Section 2.1 is expensive in both time (because each outer step requires several inner steps) and memory (because reverse-mode differentiation requires storing the intermediate inner states). For this reason, much of meta-learning has focused on the fewshot regime [16]. However, there is an increasing focus on methods which seek to extend optimization-based metalearning to the many-shot regime. Popular solutions include implicit differentiation of ω [157], [167], [294], forward-mode differentiation of ω [69], [71], [295], gradient preconditioning [107], solving for a greedy version of ω online by alternating inner and outer steps [18], [42], [201], truncation [296], shortcuts [297] or inversion [193] of the inner optimization. Many-step meta-learning can also be achieved by learning an initialization that minimizes the gradient descent trajectory length over task manifolds [298]. Finally, another family of approaches accelerate meta-training via closedform solvers in the inner loop [166], [168].

计算成本和多样本。第2.1节所示的两级优化的初始实现在时间(因为每个外部步骤需要几个内部步骤)和内存(因为反向模式微分需要存储中间内部状态)方面都很昂贵。出于这个原因，大部分元学习都集中在少数人的体制上[16]。然而，越来越多的人关注寻求将基于优化的金属学习扩展到多目标状态的方法。流行的解决方案包括ω的隐式微分[157]，[167]，[294]，ω的正向模式微分[69]，[71]，[295]，梯度预处理[107]，通过交替内部和外部步骤[18]，[42]，[201]，截断[296]，快捷方式[297]或内部优化的反转[193]在线求解贪婪版本的ω。多步元学习也可以通过学习最小化任务流形上梯度下降轨迹长度的初始化来实现[298]。最后，另一系列方法通过内部循环中的封闭式求解器加速元训练[166]，[168]。

Implicit gradients scale to large dimensions of ω but only provide approximate gradients for it, and require the inner task loss to be a function of ω. Forward-mode differentiation is exact and doesn’t have such constraints, but scales poorly with the dimension of ω. Online methods are cheap but suffer from a short-horizon bias [299]. Gradient degradation is also a challenge in the many-shot regime, and solutions include warp layers [107] or gradient averaging [71].

隐式梯度可扩展到ω的大维度，但仅为其提供近似梯度，并要求内部任务损失是ω的函数。前向模式微分是精确的，没有这样的约束，但在ω的维数下缩放很差。在线方法价格低廉，但存在短期偏差[299]。梯度退化在多样本模式中也是一个挑战，解决方案包括扭曲层[107]或梯度平均[71]。

In terms of the cost of solving new tasks at the meta-test stage, FFMs have a significant advantage over optimizationbased meta-learners, which makes them appealing for applications involving deployment of learning algorithms on mobile devices such as smartphones [6], for example to achieve personalisation. This is especially so because the embedded device versions of contemporary deep learning software frameworks typically lack support for backpropagationbased training, which FFMs do not require. 

就元测试阶段解决新任务的成本而言，FFM与基于优化的元学习者相比具有显著优势，这使其对涉及在智能手机等移动设备上部署学习算法的应用程序具有吸引力[6]，例如实现个性化。这尤其是因为当代深度学习软件框架的嵌入式设备版本通常缺乏对基于反向传播的训练的支持，而FFM不需要这种支持。

## 7 CONCLUSION
The field of meta-learning has seen a rapid growth in interest. This has come with some level of confusion, with regards to how it relates to neighbouring fields, what it can be applied to, and how it can be benchmarked. In this survey we have sought to clarify these issues by thoroughly surveying the area both from a methodological point of view – which we broke down into a taxonomy of meta-representation, meta-optimizer and meta-objective; and from an application point of view. We hope that this survey will help newcomers and practitioners to orient themselves to develop and exploit in this growing field, as well as highlight opportunities for future research.

元学习领域的兴趣迅速增长。这带来了一定程度的困惑，关于它与邻近油田的关系，它可以应用于什么，以及如何进行基准测试。在这项调查中，我们试图通过从方法论的角度彻底调查这一领域来澄清这些问题——我们将其分解为元表示、元优化器和元目标的分类; 并且从应用的角度来看。我们希望这项调查将帮助新来者和从业者在这个不断发展的领域中发展和利用，并突出未来研究的机会。

## ACKNOWLEDGMENTS
T. Hospedales was supported by the Engineering and Physical Sciences Research Council of the UK (EPSRC) Grant number EP/S000631/1 and the UK MOD University Defence Research Collaboration (UDRC) in Signal Processing, and EPSRC Grant EP/R026173/1.

## References
1. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning For Image Recognition,” in CVPR, 2016.
2. D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot et al., “Mastering The Game Of Go With Deep Neural Networks And Tree Search,” Nature, 2016.
3. J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining Of Deep Bidirectional Transformers For Language Understanding,” in ACL, 2019.
4. G. Marcus, “Deep Learning: A Critical Appraisal,” arXiv e-prints, 2018.
5. H. Altae-Tran, B. Ramsundar, A. S. Pappu, and V. S. Pande, “Low Data Drug Discovery With One-shot Learning,” CoRR, 2016.
6. A. Ignatov, R. Timofte, A. Kulik, S. Yang, K. Wang, F. Baum, M. Wu, L. Xu, and L. Van Gool, “AI Benchmark: All About Deep Learning On Smartphones In 2019,” arXiv e-prints, 2019.
7. S. Thrun and L. Pratt, “Learning To Learn: Introduction And Overview,” in Learning To Learn, 1998.
8. H. F. Harlow, “The Formation Of Learning Sets.” Psychological Review, 1949.
9. J. B. Biggs, “The Role of Meta-Learning in Study Processes,” British Journal of Educational Psychology, 1985.
10. A. M. Schrier, “Learning How To Learn: The Significance And Current Status Of Learning Set Formation,” Primates, 1984.
11. P. Domingos, “A Few Useful Things To Know About Machine Learning,” Commun. ACM, 2012.
12. D. G. Lowe, “Distinctive Image Features From Scale-Invariant,” International Journal of Computer Vision, 2004.
13. A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet Classifi- cation With Deep Convolutional Neural Networks,” in NeurIPS, 2012.
14. J. Schmidhuber, “Evolutionary Principles In Self-referential Learning,” On learning how to learn: The meta-meta-... hook, 1987.
15. J. Schmidhuber, J. Zhao, and M. Wiering, “Shifting Inductive Bias With Success-Story Algorithm, Adaptive Levin Search, And Incremental Self-Improvement,” Machine Learning, 1997.
16. C. Finn, P. Abbeel, and S. Levine, “Model-Agnostic Meta-learning For Fast Adaptation Of Deep Networks,” in ICML, 2017.
17. L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, “Bilevel Programming For Hyperparameter Optimization And Meta-learning,” in ICML, 2018. 16
18. H. Liu, K. Simonyan, and Y. Yang, “DARTS: Differentiable Architecture Search,” in ICLR, 2019.
19. M. Andrychowicz, M. Denil, S. G. Colmenarejo, M. W. Hoffman, D. Pfau, T. Schaul, and N. de Freitas, “Learning To Learn By Gradient Descent By Gradient Descent,” in NeurIPS, 2016.
20. J. Snell, K. Swersky, and R. S. Zemel, “Prototypical Networks For Few Shot Learning,” in NeurIPS, 2017.
21. L. Metz, N. Maheswaranathan, B. Cheung, and J. Sohl-Dickstein, “Meta-learning Update Rules For Unsupervised Representation Learning,” ICLR, 2019.
22. Y. Duan, J. Schulman, X. Chen, P. L. Bartlett, I. Sutskever, and P. Abbeel, “RL2 : Fast Reinforcement Learning Via Slow Reinforcement Learning,” in ArXiv E-prints, 2016.
23. R. Houthooft, R. Y. Chen, P. Isola, B. C. Stadie, F. Wolski, J. Ho, and P. Abbeel, “Evolved Policy Gradients,” NeurIPS, 2018.
24. F. Alet, M. F. Schneider, T. Lozano-Perez, and L. Pack Kaelbling, “Meta-Learning Curiosity Algorithms,” ICLR, 2020.
25. E. Real, A. Aggarwal, Y. Huang, and Q. V. Le, “Regularized Evolution For Image Classifier Architecture Search,” AAAI, 2019.
26. B. Zoph and Q. V. Le, “Neural Architecture Search With Reinforcement Learning,” ICLR, 2017.
27. R. Vilalta and Y. Drissi, “A Perspective View And Survey Of Meta-learning,” Artificial intelligence review, 2002.
28. S. Thrun, “Lifelong learning algorithms,” in Learning to learn. Springer, 1998, pp. 181–209.
29. J. Baxter, “Theoretical models of learning to learn,” in Learning to learn. Springer, 1998, pp. 71–94.
30. D. H. Wolpert, “The Lack Of A Priori Distinctions Between Learning Algorithms,” Neural Computation, 1996.
31. J. Vanschoren, “Meta-Learning: A Survey,” CoRR, 2018.
32. Q. Yao, M. Wang, H. J. Escalante, I. Guyon, Y. Hu, Y. Li, W. Tu, Q. Yang, and Y. Yu, “Taking Human Out Of Learning Applications: A Survey On Automated Machine Learning,” CoRR, 2018.
33. F. Hutter, L. Kotthoff, and J. Vanschoren, Eds., Automatic machine learning: methods, systems, challenges. Springer, 2019.
34. S. J. Pan and Q. Yang, “A Survey On Transfer Learning,” IEEE TKDE, 2010.
35. C. Lemke, M. Budka, and B. Gabrys, “Meta-Learning: A Survey Of Trends And Technologies,” Artificial intelligence review, 2015.
36. Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni, “Generalizing from a few examples: A survey on few-shot learning,” ACM Comput. Surv., vol. 53, no. 3, Jun. 2020.
37. T. Elsken, J. H. Metzen, and F. Hutter, “Neural Architecture Search: A Survey,” Journal of Machine Learning Research, 2019.
38. N. Mishra, M. Rohaninejad, X. Chen, and P. Abbeel, “A Simple Neural Attentive Meta-learner,” ICLR, 2018.
39. S. Ravi and H. Larochelle, “Optimization As A Model For FewShot Learning,” in ICLR, 2016.
40. H. Stackelberg, The Theory Of Market Economy. Oxford University Press, 1952.
41. A. Sinha, P. Malo, and K. Deb, “A Review On Bilevel Optimization: From Classical To Evolutionary Approaches And Applications,” IEEE Transactions on Evolutionary Computation, 2018.
42. Y. Li, Y. Yang, W. Zhou, and T. M. Hospedales, “Feature-Critic Networks For Heterogeneous Domain Generalization,” in ICML, 2019.
43. G. Denevi, C. Ciliberto, D. Stamos, and M. Pontil, “Learning To Learn Around A Common Mean,” in NeurIPS, 2018.
44. M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, and A. J. Smola, “Deep sets,” in NIPS, 2017.
45. J. Gordon, J. Bronskill, M. Bauer, S. Nowozin, and R. E. Turner, “Meta-Learning Probabilistic Inference For Prediction,” ICLR, 2019.
46. Y. Bengio, S. Bengio, and J. Cloutier, “Learning A Synaptic Learning Rule,” in IJCNN, 1990.
47. S. Bengio, Y. Bengio, and J. Cloutier, “On The Search For New Learning Rules For ANNs,” Neural Processing Letters, 1995.
48. J. Schmidhuber, J. Zhao, and M. Wiering, “Simple Principles Of Meta-Learning,” Technical report IDSIA, 1996.
49. J. Schmidhuber, “A Neural Network That Embeds Its Own Metalevels,” in IEEE International Conference On Neural Networks, 1993.
50. ——, “A possibility for implementing curiosity and boredom in model-building neural controllers,” in SAB, 1991.
51. S. Hochreiter, A. S. Younger, and P. R. Conwell, “Learning To Learn Using Gradient Descent,” in ICANN, 2001.
52. A. S. Younger, S. Hochreiter, and P. R. Conwell, “Meta-learning With Backpropagation,” in IJCNN, 2001.
53. J. Storck, S. Hochreiter, and J. Schmidhuber, “Reinforcement driven information acquisition in non-deterministic environments,” in ICANN, 1995.
54. M. Wiering and J. Schmidhuber, “Efficient model-based exploration,” in SAB, 1998.
55. N. Schweighofer and K. Doya, “Meta-learning In Reinforcement Learning,” Neural Networks, 2003.
56. L. Y. Pratt, J. Mostow, C. A. Kamm, and A. A. Kamm, “Direct transfer of learned information among neural networks.” in AAAI, vol. 91, 1991.
57. J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, “How Transferable Are Features In Deep Neural Networks?” in NeurIPS, 2014.
58. G. Csurka, Domain Adaptation In Computer Vision Applications. Springer, 2017.
59. D. Li and T. Hospedales, “Online Meta-Learning For MultiSource And Semi-Supervised Domain Adaptation,” in ECCV, 2020.
60. M. B. Ring, “Continual learning in reinforcement environments,” Ph.D. dissertation, USA, 1994.
61. G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual Lifelong Learning With Neural Networks: A Review,” Neural Networks, 2019.
62. Z. Chen and B. Liu, “Lifelong Machine Learning, Second Edition,” Synthesis Lectures on Artificial Intelligence and Machine Learning, 2018.
63. M. Al-Shedivat, T. Bansal, Y. Burda, I. Sutskever, I. Mordatch, and P. Abbeel, “Continuous Adaptation Via Meta-Learning In Nonstationary And Competitive Environments,” ICLR, 2018.
64. S. Ritter, J. X. Wang, Z. Kurth-Nelson, S. M. Jayakumar, C. Blundell, R. Pascanu, and M. Botvinick, “Been There, Done That: Meta-learning With Episodic Recall,” ICML, 2018.
65. I. Clavera, A. Nagabandi, S. Liu, R. S. Fearing, P. Abbeel, S. Levine, and C. Finn, “Learning To Adapt In Dynamic, RealWorld Environments Through Meta-Reinforcement Learning,” in ICLR, 2019.
66. R. Caruana, “Multitask Learning,” Machine Learning, 1997.
67. Y. Yang and T. M. Hospedales, “Deep Multi-Task Representation Learning: A Tensor Factorisation Approach,” in ICLR, 2017.
68. E. Meyerson and R. Miikkulainen, “Modular Universal Reparameterization: Deep Multi-task Learning Across Diverse Domains,” in NeurIPS, 2019.
69. L. Franceschi, M. Donini, P. Frasconi, and M. Pontil, “Forward And Reverse Gradient-Based Hyperparameter Optimization,” in ICML, 2017.
70. X. Lin, H. Baweja, G. Kantor, and D. Held, “Adaptive Auxiliary Task Weighting For Reinforcement Learning,” in NeurIPS, 2019.
71. P. Micaelli and A. Storkey, “Non-greedy gradient-based hyperparameter optimization over long horizons,” arXiv, 2020.
72. J. Bergstra and Y. Bengio, “Random Search For Hyper-Parameter Optimization,” in Journal Of Machine Learning Research, 2012.
73. B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas, “Taking The Human Out Of The Loop: A Review Of Bayesian Optimization,” Proceedings of the IEEE, 2016.
74. D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent Dirchlet allocation,” Journal of Machine Learning Research, vol. 3, pp. 993–1022, 2003.
75. H. Edwards and A. Storkey, “Towards A Neural Statistician,” in ICLR, 2017.
76. E. Grant, C. Finn, S. Levine, T. Darrell, and T. Griffiths, “Recasting Gradient-Based Meta-Learning As Hierarchical Bayes,” in ICLR, 2018.
77. H. Yao, X. Wu, Z. Tao, Y. Li, B. Ding, R. Li, and Z. Li, “Automated Relational Meta-learning,” in ICLR, 2020.
78. S. C. Yoonho Lee, “Gradient-Based Meta-Learning With Learned Layerwise Metric And Subspace,” in ICML, 2018.
79. Z. Li, F. Zhou, F. Chen, and H. Li, “Meta-SGD: Learning To Learn Quickly For Few Shot Learning,” arXiv e-prints, 2017.
80. A. Antoniou, H. Edwards, and A. J. Storkey, “How To Train Your MAML,” in ICLR, 2018.
81. K. Li and J. Malik, “Learning To Optimize,” in ICLR, 2017.
82. E. Grefenstette, B. Amos, D. Yarats, P. M. Htut, A. Molchanov, F. Meier, D. Kiela, K. Cho, and S. Chintala, “Generalized inner loop meta-learning,” arXiv preprint arXiv:1910.01727, 2019.
83. S. Qiao, C. Liu, W. Shen, and A. L. Yuille, “Few-Shot Image Recognition By Predicting Parameters From Activations,” CVPR, 2018. 17
84. S. Gidaris and N. Komodakis, “Dynamic Few-Shot Visual Learning Without Forgetting,” in CVPR, 2018.
85. A. Graves, G. Wayne, and I. Danihelka, “Neural Turing Machines,” in ArXiv E-prints, 2014.
86. A. Santoro, S. Bartunov, M. Botvinick, D. Wierstra, and T. Lillicrap, “Meta Learning With Memory-Augmented Neural Networks,” in ICML, 2016.
87. T. Munkhdalai and H. Yu, “Meta Networks,” in ICML, 2017.
88. C. Finn and S. Levine, “Meta-Learning And Universality: Deep Representations And Gradient Descent Can Approximate Any Learning Algorithm,” in ICLR, 2018.
89. G. Kosh, R. Zemel, and R. Salakhutdinov, “Siamese Neural Networks For One-shot Image Recognition,” in ICML, 2015.
90. O. Vinyals, C. Blundell, T. Lillicrap, D. Wierstra et al., “Matching Networks For One Shot Learning,” in NeurIPS, 2016.
91. F. Sung, Y. Yang, L. Zhang, T. Xiang, P. H. S. Torr, and T. M. Hospedales, “Learning To Compare: Relation Network For FewShot Learning,” in CVPR, 2018.
92. V. Garcia and J. Bruna, “Few-Shot Learning With Graph Neural Networks,” in ICLR, 2018.
93. I. Bello, B. Zoph, V. Vasudevan, and Q. V. Le, “Neural Optimizer Search With Reinforcement Learning,” in ICML, 2017.
94. O. Wichrowska, N. Maheswaranathan, M. W. Hoffman, S. G. Colmenarejo, M. Denil, N. de Freitas, and J. Sohl-Dickstein, “Learned Optimizers That Scale And Generalize,” in ICML, 2017.
95. Y. Balaji, S. Sankaranarayanan, and R. Chellappa, “MetaReg: Towards Domain Generalization Using Meta-Regularization,” in NeurIPS, 2018.
96. J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Learning To Learn From Noisy Labeled Data,” in CVPR, 2019.
97. M. Goldblum, L. Fowl, and T. Goldstein, “Adversarially Robust Few-shot Learning: A Meta-learning Approach,” arXiv e-prints, 2019.
98. C. Finn, K. Xu, and S. Levine, “Probabilistic Model-agnostic Meta-learning,” in NeurIPS, 2018.
99. C. Finn, A. Rajeswaran, S. Kakade, and S. Levine, “Online Metalearning,” ICML, 2019.
100. A. A. Rusu, D. Rao, J. Sygnowski, O. Vinyals, R. Pascanu, S. Osindero, and R. Hadsell, “Meta-Learning With Latent Embedding Optimization,” ICLR, 2019.
101. A. Antoniou and A. Storkey, “Learning To Learn By SelfCritique,” NeurIPS, 2019.
102. Q. Sun, Y. Liu, T.-S. Chua, and B. Schiele, “Meta-Transfer Learning For Few-Shot Learning,” in CVPR, 2018.
103. R. Vuorio, S.-H. Sun, H. Hu, and J. J. Lim, “Multimodal Model-Agnostic Meta-Learning Via Task-Aware Modulation,” in NeurIPS, 2019.
104. H. Yao, Y. Wei, J. Huang, and Z. Li, “Hierarchically Structured Meta-learning,” ICML, 2019.
105. D. Kingma and J. Ba, “Adam: A Method For Stochastic Optimization,” in ICLR, 2015.
106. E. Park and J. B. Oliva, “Meta-Curvature,” in NeurIPS, 2019.
107. S. Flennerhag, A. A. Rusu, R. Pascanu, F. Visin, H. Yin, and R. Hadsell, “Meta-learning with warped gradient descent,” in ICLR, 2020.
108. Y. Chen, M. W. Hoffman, S. G. Colmenarejo, M. Denil, T. P. Lillicrap, M. Botvinick, and N. de Freitas, “Learning To Learn Without Gradient Descent By Gradient Descent,” in ICML, 2017.
109. T. Heskes, “Empirical bayes for learning to learn,” in ICML, 2000.
110. J. Requeima, J. Gordon, J. Bronskill, S. Nowozin, and R. E. Turner, “Fast and flexible multi-task classification using conditional neural adaptive processes,” in NeurIPS, 2019.
111. E. Triantafillou, T. Zhu, V. Dumoulin, P. Lamblin, K. Xu, R. Goroshin, C. Gelada, K. Swersky, P. Manzagol, and H. Larochelle, “Meta-Dataset: A Dataset Of Datasets For Learning To Learn From Few Examples,” ICLR, 2020.
112. D. Ha, A. Dai, and Q. V. Le, “HyperNetworks,” ICLR, 2017.
113. A. Brock, T. Lim, J. M. Ritchie, and N. Weston, “SMASH: OneShot Model Architecture Search Through Hypernetworks,” ICLR, 2018.
114. K. Rakelly, A. Zhou, C. Finn, S. Levine, and D. Quillen, “Effi- cient Off-Policy Meta-Reinforcement Learning Via Probabilistic Context Variables,” in ICML, 2019.
115. Y. Duan, M. Andrychowicz, B. Stadie, O. J. Ho, J. Schneider, I. Sutskever, P. Abbeel, and W. Zaremba, “One-shot Imitation Learning,” in NeurIPS, 2017.
116. J. X. Wang, Z. Kurth-Nelson, D. Tirumala, H. Soyer, J. Z. Leibo, R. Munos, C. Blundell, D. Kumaran, and M. Botvinick, “Learning To Reinforcement Learn,” CoRR, 2016.
117. W.-Y. Chen, Y.-C. Liu, Z. Kira, Y.-C. Wang, and J.-B. Huang, “A Closer Look At Few-Shot Classification,” in ICLR, 2019.
118. B. Oreshkin, P. Rodr´ıguez L´opez, and A. Lacoste, “TADAM: Task Dependent Adaptive Metric For Improved Few-shot Learning,” in NeurIPS, 2018.
119. H.-Y. Tseng, H.-Y. Lee, J.-B. Huang, and M.-H. Yang, “”CrossDomain Few-Shot Classification Via Learned Feature-Wise Transformation”,” ICLR, Jan. 2020.
120. F. Sung, L. Zhang, T. Xiang, T. Hospedales, and Y. Yang, “Learning To Learn: Meta-critic Networks For Sample Efficient Learning,” arXiv e-prints, 2017.
121. W. Zhou, Y. Li, Y. Yang, H. Wang, and T. M. Hospedales, “Online Meta-Critic Learning For Off-Policy Actor-Critic Methods,” in NeurIPS, 2020.
122. G. Denevi, D. Stamos, C. Ciliberto, and M. Pontil, “OnlineWithin-Online Meta-Learning,” in NeurIPS, 2019.
123. S. Gonzalez and R. Miikkulainen, “Improved Training Speed, Accuracy, And Data Utilization Through Loss Function Optimization,” arXiv e-prints, 2019.
124. S. Bechtle, A. Molchanov, Y. Chebotar, E. Grefenstette, L. Righetti, G. Sukhatme, and F. Meier, “Meta-learning via learned loss,” arXiv preprint arXiv:1906.05374, 2019.
125. A. I. Rinu Boney, “Semi-Supervised Few-Shot Learning With MAML,” ICLR, 2018.
126. C. Huang, S. Zhai, W. Talbott, M. B. Martin, S.-Y. Sun, C. Guestrin, and J. Susskind, “Addressing The Loss-Metric Mismatch With Adaptive Loss Alignment,” in ICML, 2019.
127. J. Grabocka, R. Scholz, and L. Schmidt-Thieme, “Learning Surrogate Losses,” CoRR, 2019.
128. C. Doersch and A. Zisserman, “Multi-task Self-Supervised Visual Learning,” in ICCV, 2017.
129. M. Jaderberg, V. Mnih, W. M. Czarnecki, T. Schaul, J. Z. Leibo, D. Silver, and K. Kavukcuoglu, “Reinforcement Learning With Unsupervised Auxiliary Tasks,” in ICLR, 2017.
130. S. Liu, A. Davison, and E. Johns, “Self-supervised Generalisation With Meta Auxiliary Learning,” in NeurIPS, 2019.
131. K. O. Stanley, J. Clune, J. Lehman, and R. Miikkulainen, “Designing Neural Networks Through Neuroevolution,” Nature Machine Intelligence, 2019.
132. J. Bayer, D. Wierstra, J. Togelius, and J. Schmidhuber, “Evolving memory cell structures for sequence learning,” in ICANN, 2009.
133. S. Xie, H. Zheng, C. Liu, and L. Lin, “SNAS: Stochastic Neural Architecture Search,” in ICLR, 2019.
134. A. Zela, T. Elsken, T. Saikia, Y. Marrakchi, T. Brox, and F. Hutter, “Understanding and robustifying differentiable architecture search,” in ICLR, 2020. Online.. Available: https://openreview.net/forum?id=H1gDNyrKDS
135. D. Lian, Y. Zheng, Y. Xu, Y. Lu, L. Lin, P. Zhao, J. Huang, and S. Gao, “Towards Fast Adaptation Of Neural Architectures With Meta Learning,” in ICLR, 2020.
136. A. Shaw, W. Wei, W. Liu, L. Song, and B. Dai, “Meta Architecture Search,” in NeurIPS, 2019.
137. R. Hou, H. Chang, M. Bingpeng, S. Shan, and X. Chen, “Cross Attention Network For Few-shot Classification,” in NeurIPS, 2019.
138. M. Ren, R. Liao, E. Fetaya, and R. Zemel, “Incremental Few-shot Learning With Attention Attractor Networks,” in NeurIPS, 2019.
139. Y. Bao, M. Wu, S. Chang, and R. Barzilay, “Few-shot Text Classi- fication With Distributional Signatures,” in ICLR, 2020.
140. F. Alet, T. Lozano-P´erez, and L. P. Kaelbling, “Modular Metalearning,” in CORL, 2018.
141. F. Alet, E. Weng, T. Lozano-P´erez, and L. P. Kaelbling, “Neural Relational Inference With Fast Modular Meta-learning,” in NeurIPS, 2019.
142. C. Fernando, D. Banarse, C. Blundell, Y. Zwols, D. Ha, A. A. Rusu, A. Pritzel, and D. Wierstra, “PathNet: Evolution Channels Gradient Descent In Super Neural Networks,” in ArXiv E-prints, 2017.
143. B. M. Lake, “Compositional Generalization Through Meta Sequence-to-sequence Learning,” in NeurIPS, 2019.
144. E. D. Cubuk, B. Zoph, D. Man´e, V. Vasudevan, and Q. V. Le, “AutoAugment: Learning Augmentation Policies From Data,” CVPR, 2019. 18
145. Y. Li, G. Hu, Y. Wang, T. Hospedales, N. M. Robertson, and Y. Yang, “DADA: Differentiable Automatic Data Augmentation,” 2020.
146. R. Volpi and V. Murino, “Model Vulnerability To Distributional Shifts Over Image Transformation Sets,” in ICCV, 2019.
147. A. Antoniou, A. Storkey, and H. Edwards, “Data Augmentation Generative Adversarial Networks,” arXiv e-prints, 2017.
148. C. Zhang, C. ¨Oztireli, S. Mandt, and G. Salvi, “Active Mini-batch Sampling Using Repulsive Point Processes,” in AAAI, 2019.
149. I. Loshchilov and F. Hutter, “Online Batch Selection For Faster Training Of Neural Networks,” in ICLR, 2016.
150. Y. Fan, F. Tian, T. Qin, X. Li, and T. Liu, “Learning To Teach,” in ICLR, 2018.
151. J. Shu, Q. Xie, L. Yi, Q. Zhao, S. Zhou, Z. Xu, and D. Meng, “Meta-Weight-Net: Learning An Explicit Mapping For Sample Weighting,” in NeurIPS, 2019.
152. M. Ren, W. Zeng, B. Yang, and R. Urtasun, “Learning To Reweight Examples For Robust Deep Learning,” in ICML, 2018.
153. J. L. Elman, “Learning and development in neural networks: the importance of starting small,” Cognition, vol. 48, no. 1, pp. 71 – 99, 1993.
154. Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum Learning,” in ICML, 2009.
155. L. Jiang, Z. Zhou, T. Leung, L.-J. Li, and L. Fei-Fei, “Mentornet: Learning Data-driven Curriculum For Very Deep Neural Networks On Corrupted Labels,” in ICML, 2018.
156. T. Wang, J. Zhu, A. Torralba, and A. A. Efros, “Dataset Distillation,” CoRR, 2018.
157. J. Lorraine, P. Vicol, and D. Duvenaud, “Optimizing Millions Of Hyperparameters By Implicit Differentiation,” in AISTATS, 2020.
158. O. Bohdal, Y. Yang, and T. Hospedales, “Flexible dataset distillation: Learn labels instead of images,” arXiv, 2020.
159. W.-H. Li, C.-S. Foo, and H. Bilen, “Learning To Impute: A General Framework For Semi-supervised Learning,” arXiv e-prints, 2019.
160. Q. Sun, X. Li, Y. Liu, S. Zheng, T.-S. Chua, and B. Schiele, “Learning To Self-train For Semi-supervised Few-shot Classification,” in NeurIPS, 2019.
161. O. M. Andrychowicz, B. Baker, M. Chociej, R. J´ozefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba, “Learning dexterous in-hand manipulation,” The International Journal of Robotics Research, vol. 39, no. 1, pp. 3–20, 2020.
162. N. Ruiz, S. Schulter, and M. Chandraker, “Learning To Simulate,” ICLR, 2018.
163. Q. Vuong, S. Vikram, H. Su, S. Gao, and H. I. Christensen, “How To Pick The Domain Randomization Parameters For Sim-to-real Transfer Of Reinforcement Learning Policies?” CoRR, 2019.
164. Q. V. L. Prajit Ramachandran, Barret Zoph, “Searching For Activation Functions,” in ArXiv E-prints, 2017.
165. H. B. Lee, H. Lee, D. Na, S. Kim, M. Park, E. Yang, and S. J. Hwang, “Learning to balance: Bayesian meta-learning for imbalanced and out-of-distribution tasks,” ICLR, 2020.
166. K. Lee, S. Maji, A. Ravichandran, and S. Soatto, “Meta-Learning With Differentiable Convex Optimization,” in CVPR, 2019.
167. A. Rajeswaran, C. Finn, S. Kakade, and S. Levine, “Meta-Learning With Implicit Gradients,” in NeurIPS, 2019.
168. L. Bertinetto, J. F. Henriques, P. H. Torr, and A. Vedaldi, “Metalearning With Differentiable Closed-form Solvers,” in ICLR, 2019.
169. H. Liu, R. Socher, and C. Xiong, “Taming MAML: Efficient Unbiased Meta-reinforcement Learning,” in ICML, 2019.
170. J. Rothfuss, D. Lee, I. Clavera, T. Asfour, and P. Abbeel, “ProMP: Proximal Meta-Policy Search,” in ICLR, 2019.
171. R. Fakoor, P. Chaudhari, S. Soatto, and A. J. Smola, “Meta-QLearning,” in ICLR, 2020.
172. X. Song, W. Gao, Y. Yang, K. Choromanski, A. Pacchiano, and Y. Tang, “ES-MAML: Simple Hessian-Free Meta Learning,” in ICLR, 2020.
173. C. Fernando, J. Sygnowski, S. Osindero, J. Wang, T. Schaul, D. Teplyashin, P. Sprechmann, A. Pritzel, and A. Rusu, “MetaLearning By The Baldwin Effect,” in Proceedings Of The Genetic And Evolutionary Computation Conference Companion, 2018.
174. R. Vuorio, D.-Y. Cho, D. Kim, and J. Kim, “Meta Continual Learning,” arXiv e-prints, 2018.
175. Z. Xu, H. van Hasselt, and D. Silver, “Meta-Gradient Reinforcement Learning,” in NeurIPS, 2018.
176. K. Young, B. Wang, and M. E. Taylor, “Metatrace Actor-Critic: Online Step-Size Tuning By Meta-gradient Descent For Reinforcement Learning Control,” in IJCAI, 2019.
177. M. Jaderberg, W. M. Czarnecki, I. Dunning, L. Marris, G. Lever, A. G. Casta˜neda, C. Beattie, N. C. Rabinowitz, A. S. Morcos, A. Ruderman, N. Sonnerat, T. Green, L. Deason, J. Z. Leibo, D. Silver, D. Hassabis, K. Kavukcuoglu, and T. Graepel, “Humanlevel Performance In 3D Multiplayer Games With Populationbased Reinforcement Learning,” Science, 2019.
178. J.-M. Perez-Rua, X. Zhu, T. Hospedales, and T. Xiang, “Incremental Few-Shot Object Detection,” in CVPR, 2020.
179. M. Garnelo, D. Rosenbaum, C. J. Maddison, T. Ramalho, D. Saxton, M. Shanahan, Y. W. Teh, D. J. Rezende, and S. M. A. Eslami, “Conditional Neural Processes,” ICML, 2018.
180. A. Pakman, Y. Wang, C. Mitelut, J. Lee, and L. Paninski, “Neural clustering processes,” in ICML, 2019.
181. J. Lee, Y. Lee, J. Kim, A. Kosiorek, S. Choi, and Y. W. Teh, “Set transformer: A framework for attention-based permutationinvariant neural networks,” in ICML, 2019.
182. J. Lee, Y. Lee, and Y. W. Teh, “Deep amortized clustering,” 2019.
183. V. Veeriah, M. Hessel, Z. Xu, R. Lewis, J. Rajendran, J. Oh, H. van Hasselt, D. Silver, and S. Singh, “Discovery Of Useful Questions As Auxiliary Tasks,” in NeurIPS, 2019.
184. Z. Zheng, J. Oh, and S. Singh, “On Learning Intrinsic Rewards For Policy Gradient Methods,” in NeurIPS, 2018.
185. T. Xu, Q. Liu, L. Zhao, and J. Peng, “Learning To Explore With Meta-Policy Gradient,” ICML, 2018.
186. B. C. Stadie, G. Yang, R. Houthooft, X. Chen, Y. Duan, Y. Wu, P. Abbeel, and I. Sutskever, “Some Considerations On Learning To Explore Via Meta-Reinforcement Learning,” in NeurIPS, 2018.
187. F. Garcia and P. S. Thomas, “A Meta-MDP Approach To Exploration For Lifelong Reinforcement Learning,” in NeurIPS, 2019.
188. A. Gupta, R. Mendonca, Y. Liu, P. Abbeel, and S. Levine, “MetaReinforcement Learning Of Structured Exploration Strategies,” in NeurIPS, 2018.
189. H. B. Lee, T. Nam, E. Yang, and S. J. Hwang, “Meta Dropout: Learning To Perturb Latent Features For Generalization,” in ICLR, 2020.
190. P. Bachman, A. Sordoni, and A. Trischler, “Learning Algorithms For Active Learning,” in ICML, 2017.
191. K. Konyushkova, R. Sznitman, and P. Fua, “Learning Active Learning From Data,” in NeurIPS, 2017.
192. K. Pang, M. Dong, Y. Wu, and T. M. Hospedales, “Meta-Learning Transferable Active Learning Policies By Deep Reinforcement Learning,” CoRR, 2018.
193. D. Maclaurin, D. Duvenaud, and R. P. Adams, “Gradient-based Hyperparameter Optimization Through Reversible Learning,” in ICML, 2015.
194. C. Russell, M. Toso, and N. Campbell, “Fixing Implicit Derivatives: Trust-Region Based Learning Of Continuous Energy Functions,” in NeurIPS, 2019.
195. A. Nichol, J. Achiam, and J. Schulman, “On First-Order MetaLearning Algorithms,” in ArXiv E-prints, 2018.
196. T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever, “Evolution Strategies As A Scalable Alternative To Reinforcement Learning,” arXiv e-prints, 2017.
197. F. Stulp and O. Sigaud, “Robot Skill Learning: From Reinforcement Learning To Evolution Strategies,” Paladyn, Journal of Behavioral Robotics, 2013.
198. A. Soltoggio, K. O. Stanley, and S. Risi, “Born To Learn: The Inspiration, Progress, And Future Of Evolved Plastic Artificial Neural Networks,” Neural Networks, 2018.
199. Y. Cao, T. Chen, Z. Wang, and Y. Shen, “Learning To Optimize In Swarms,” in NeurIPS, 2019.
200. F. Meier, D. Kappler, and S. Schaal, “Online Learning Of A Memory For Learning Rates,” in ICRA, 2018.
201. A. G. Baydin, R. Cornish, D. Mart´ınez-Rubio, M. Schmidt, and F. D. Wood, “Online Learning Rate Adaptation With Hypergradient Descent,” in ICLR, 2018.
202. S. Chen, W. Wang, and S. J. Pan, “MetaQuant: Learning To Quantize By Learning To Penetrate Non-differentiable Quantization,” in NeurIPS, 2019.
203. C. Sun, A. Shrivastava, S. Singh, and A. Gupta, “Revisiting Unreasonable Effectiveness Of Data In Deep Learning Era,” in ICCV, 2017.
204. M. Yin, G. Tucker, M. Zhou, S. Levine, and C. Finn, “MetaLearning Without Memorization,” ICLR, 2020. 19
205. S. W. Yoon, J. Seo, and J. Moon, “Tapnet: Neural Network Augmented With Task-adaptive Projection For Few-shot Learning,” ICML, 2019.
206. J. W. Rae, S. Bartunov, and T. P. Lillicrap, “Meta-learning Neural Bloom Filters,” ICML, 2019.
207. A. Raghu, M. Raghu, S. Bengio, and O. Vinyals, “Rapid Learning Or Feature Reuse? Towards Understanding The Effectiveness Of Maml,” arXiv e-prints, 2019.
208. B. Kang, Z. Liu, X. Wang, F. Yu, J. Feng, and T. Darrell, “Few-shot Object Detection Via Feature Reweighting,” in ICCV, 2019.
209. L.-Y. Gui, Y.-X. Wang, D. Ramanan, and J. Moura, Few-Shot Human Motion Prediction Via Meta-learning. Springer, 2018.
210. A. Shaban, S. Bansal, Z. Liu, I. Essa, and B. Boots, “One-Shot Learning For Semantic Segmentation,” CoRR, 2017.
211. N. Dong and E. P. Xing, “Few-Shot Semantic Segmentation With Prototype Learning,” in BMVC, 2018.
212. K. Rakelly, E. Shelhamer, T. Darrell, A. A. Efros, and S. Levine, “Few-Shot Segmentation Propagation With Guided Networks,” ICML, 2019.
213. S. A. Eslami, D. J. Rezende, F. Besse, F. Viola, A. S. Morcos, M. Garnelo, A. Ruderman, A. A. Rusu, I. Danihelka, K. Gregor et al., “Neural scene representation and rendering,” Science, vol. 360, no. 6394, pp. 1204–1210, 2018.
214. E. Zakharov, A. Shysheya, E. Burkov, and V. S. Lempitsky, “FewShot Adversarial Learning Of Realistic Neural Talking Head Models,” CoRR, 2019.
215. T.-C. Wang, M.-Y. Liu, A. Tao, G. Liu, J. Kautz, and B. Catanzaro, “Few-shot Video-to-video Synthesis,” in NeurIPS, 2019.
216. S. E. Reed, Y. Chen, T. Paine, A. van den Oord, S. M. A. Eslami, D. J. Rezende, O. Vinyals, and N. de Freitas, “Few-shot Autoregressive Density Estimation: Towards Learning To Learn Distributions,” in ICLR, 2018.
217. O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein et al., “Imagenet Large Scale Visual Recognition Challenge,” International Journal of Computer Vision, 2015.
218. M. Ren, E. Triantafillou, S. Ravi, J. Snell, K. Swersky, J. B. Tenenbaum, H. Larochelle, and R. S. Zemel, “Meta-Learning For Semi-Supervised Few-Shot Classification,” ICLR, 2018.
219. A. Antoniou and M. O. S. A. Massimiliano, Patacchiola, “Defining Benchmarks For Continual Few-shot Learning,” arXiv eprints, 2020.
220. S.-A. Rebuffi, H. Bilen, and A. Vedaldi, “Learning Multiple Visual Domains With Residual Adapters,” in NeurIPS, 2017.
221. Y. Guo, N. C. F. Codella, L. Karlinsky, J. R. Smith, T. Rosing, and R. Feris, “A New Benchmark For Evaluation Of Cross-Domain Few-Shot Learning,” arXiv:1912.07200, 2019.
222. T. de Vries, I. Misra, C. Wang, and L. van der Maaten, “Does Object Recognition Work For Everyone?” in CVPR, 2019.
223. R. J. Williams, “Simple Statistical Gradient-Following Algorithms For Connectionist Reinforcement Learning,” Machine learning, 1992.
224. A. Fallah, A. Mokhtari, and A. Ozdaglar, “Provably Convergent Policy Gradient Methods For Model-Agnostic MetaReinforcement Learning,” arXiv e-prints, 2020.
225. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” arXiv e-prints, 2017.
226. O. Sigaud and F. Stulp, “Policy Search In Continuous Action Domains: An Overview,” Neural Networks, 2019.
227. J. Schmidhuber, “What’s interesting?” 1997.
228. L. Kirsch, S. van Steenkiste, and J. Schmidhuber, “Improving Generalization In Meta Reinforcement Learning Using Learned Objectives,” in ICLR, 2020.
229. T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft ActorCritic: Off-Policy Maximum Entropy Deep Reinforcement Learning With A Stochastic Actor,” in ICML, 2018.
230. O. Kroemer, S. Niekum, and G. D. Konidaris, “A Review Of Robot Learning For Manipulation: Challenges, Representations, And Algorithms,” CoRR, 2019.
231. A. Jabri, K. Hsu, A. Gupta, B. Eysenbach, S. Levine, and C. Finn, “Unsupervised Curricula For Visual Meta-Reinforcement Learning,” in NeurIPS, 2019.
232. Y. Yang, K. Caluwaerts, A. Iscen, J. Tan, and C. Finn, “Norml: No-reward Meta Learning,” in AAMAS, 2019.
233. S. K. Seyed Ghasemipour, S. S. Gu, and R. Zemel, “SMILe: Scalable Meta Inverse Reinforcement Learning Through ContextConditional Policies,” in NeurIPS, 2019.
234. M. C. Machado, M. G. Bellemare, E. Talvitie, J. Veness, M. Hausknecht, and M. Bowling, “Revisiting The Arcade Learning Environment: Evaluation Protocols And Open Problems For General Agents,” Journal of Artificial Intelligence Research, 2018.
235. A. Nichol, V. Pfau, C. Hesse, O. Klimov, and J. Schulman, “Gotta Learn Fast: A New Benchmark For Generalization In RL,” CoRR, 2018.
236. K. Cobbe, O. Klimov, C. Hesse, T. Kim, and J. Schulman, “Quantifying Generalization In Reinforcement Learning,” ICML, 2019.
237. G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “OpenAI Gym,” 2016.
238. C. Packer, K. Gao, J. Kos, P. Kr¨ahenb¨uhl, V. Koltun, and D. Song, “Assessing Generalization In Deep Reinforcement Learning,” arXiv e-prints, 2018.
239. C. Zhao, O. Siguad, F. Stulp, and T. M. Hospedales, “Investigating Generalisation In Continuous Deep Reinforcement Learning,” arXiv e-prints, 2019.
240. T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-world: A Benchmark And Evaluation For Multitask And Meta Reinforcement Learning,” CORL, 2019.
241. A. Bakhtin, L. van der Maaten, J. Johnson, L. Gustafson, and R. Girshick, “Phyre: A New Benchmark For Physical Reasoning,” in NeurIPS, 2019.
242. J. Tremblay, A. Prakash, D. Acuna, M. Brophy, V. Jampani, C. Anil, T. To, E. Cameracci, S. Boochoon, and S. Birchfield, “Training Deep Networks With Synthetic Data: Bridging The Reality Gap By Domain Randomization,” in CVPR, 2018.
243. A. Kar, A. Prakash, M. Liu, E. Cameracci, J. Yuan, M. Rusiniak, D. Acuna, A. Torralba, and S. Fidler, “Meta-Sim: Learning To Generate Synthetic Datasets,” CoRR, 2019.
244. B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, “Learning Transferable Architectures For Scalable Image Recognition,” in CVPR, 2018.
245. J. Kim, Y. Choi, M. Cha, J. K. Lee, S. Lee, S. Kim, Y. Choi, and J. Kim, “Auto-Meta: Automated Gradient Based Meta Learner Search,” CoRR, 2018.
246. T. Elsken, B. Staffler, J. H. Metzen, and F. Hutter, “Meta-Learning Of Neural Architectures For Few-Shot Learning,” in CVPR, 2019.
247. L. Li and A. Talwalkar, “Random Search And Reproducibility For Neural Architecture Search,” arXiv e-prints, 2019.
248. C. Ying, A. Klein, E. Christiansen, E. Real, K. Murphy, and F. Hutter, “NAS-Bench-101: Towards Reproducible Neural Architecture Search,” in ICML, 2019.
249. P. Tossou, B. Dura, F. Laviolette, M. Marchand, and A. Lacoste, “Adaptive Deep Kernel Learning,” CoRR, 2019.
250. M. Patacchiola, J. Turner, E. J. Crowley, M. O’Boyle, and A. Storkey, “Deep Kernel Transfer In Gaussian Processes For Few-shot Learning,” arXiv e-prints, 2019.
251. T. Kim, J. Yoon, O. Dia, S. Kim, Y. Bengio, and S. Ahn, “Bayesian Model-Agnostic Meta-Learning,” NeurIPS, 2018.
252. S. Ravi and A. Beatson, “Amortized Bayesian Meta-Learning,” in ICLR, 2019.
253. Z. Wang, Y. Zhao, P. Yu, R. Zhang, and C. Chen, “Bayesian meta sampling for fast uncertainty adaptation,” in ICLR, 2020.
254. K. Hsu, S. Levine, and C. Finn, “Unsupervised Learning Via Meta-learning,” ICLR, 2019.
255. S. Khodadadeh, L. Boloni, and M. Shah, “Unsupervised MetaLearning For Few-Shot Image Classification,” in NeurIPS, 2019.
256. A. Antoniou and A. Storkey, “Assume, Augment And Learn: Unsupervised Few-shot Meta-learning Via Random Labels And Data Augmentation,” arXiv e-prints, 2019.
257. Y. Jiang and N. Verma, “Meta-Learning To Cluster,” 2019.
258. V. Garg and A. T. Kalai, “Supervising Unsupervised Learning,” in NeurIPS, 2018.
259. K. Javed and M. White, “Meta-learning Representations For Continual Learning,” in NeurIPS, 2019.
260. A. Sinitsin, V. Plokhotnyuk, D. Pyrkin, S. Popov, and A. Babenko, “Editable Neural Networks,” in ICLR, 2020.
261. K. Muandet, D. Balduzzi, and B. Sch¨olkopf, “Domain Generalization Via Invariant Feature Representation,” in ICML, 2013.
262. D. Li, Y. Yang, Y. Song, and T. M. Hospedales, “Learning To Generalize: Meta-Learning For Domain Generalization,” in AAAI, 2018.
263. D. Li, Y. Yang, Y.-Z. Song, and T. Hospedales, “Deeper, Broader And Artier Domain Generalization,” in ICCV, 2017. 20
264. T. Miconi, J. Clune, and K. O. Stanley, “Differentiable Plasticity: Training Plastic Neural Networks With Backpropagation,” in ICML, 2018.
265. T. Miconi, A. Rawal, J. Clune, and K. O. Stanley, “Backpropamine: Training Self-modifying Neural Networks With Differentiable Neuromodulated Plasticity,” in ICLR, 2019.
266. J. Devlin, R. Bunel, R. Singh, M. J. Hausknecht, and P. Kohli, “Neural Program Meta-Induction,” in NIPS, 2017.
267. X. Si, Y. Yang, H. Dai, M. Naik, and L. Song, “Learning A Metasolver For Syntax-guided Program Synthesis,” ICLR, 2018.
268. P. Huang, C. Wang, R. Singh, W. Yih, and X. He, “Natural Language To Structured Query Generation Via Meta-Learning,” CoRR, 2018.
269. Y. Xie, H. Jiang, F. Liu, T. Zhao, and H. Zha, “Meta Learning With Relational Information For Short Sequences,” in NeurIPS, 2019.
270. J. Gu, Y. Wang, Y. Chen, V. O. K. Li, and K. Cho, “Meta-Learning For Low-Resource Neural Machine Translation,” in EMNLP, 2018.
271. Z. Lin, A. Madotto, C. Wu, and P. Fung, “Personalizing Dialogue Agents Via Meta-Learning,” CoRR, 2019.
272. J.-Y. Hsu, Y.-J. Chen, and H. yi Lee, “Meta Learning For End-toEnd Low-Resource Speech Recognition,” in ICASSP, 2019.
273. G. I. Winata, S. Cahyawijaya, Z. Liu, Z. Lin, A. Madotto, P. Xu, and P. Fung, “Learning Fast Adaptation On Cross-Accented Speech Recognition,” arXiv e-prints, 2020.
274. O. Klejch, J. Fainberg, and P. Bell, “Learning To Adapt: A Metalearning Approach For Speaker Adaptation,” Interspeech, 2018.
275. D. M. Metter, T. J. Colgan, S. T. Leung, C. F. Timmons, and J. Y. Park, “Trends In The US And Canadian Pathologist Workforces From 2007 To 2017,” JAMA Network Open, 2019.
276. G. Maicas, A. P. Bradley, J. C. Nascimento, I. D. Reid, and G. Carneiro, “Training Medical Image Analysis Systems Like Radiologists,” CoRR, 2018.
277. B. D. Nguyen, T.-T. Do, B. X. Nguyen, T. Do, E. Tjiputra, and Q. D. Tran, “Overcoming Data Limitation In Medical Visual Question Answering,” arXiv e-prints, 2019.
278. Z. Mirikharaji, Y. Yan, and G. Hamarneh, “Learning To Segment Skin Lesions From Noisy Annotations,” CoRR, 2019.
279. D. Barrett, F. Hill, A. Santoro, A. Morcos, and T. Lillicrap, “Measuring Abstract Reasoning In Neural Networks,” in ICML, 2018.
280. K. Zheng, Z.-J. Zha, and W. Wei, “Abstract Reasoning With Distracting Features,” in NeurIPS, 2019.
281. B. Dai, C. Zhu, and D. Wipf, “Compressing Neural Networks Using The Variational Information Bottleneck,” ICML, 2018.
282. Z. Liu, H. Mu, X. Zhang, Z. Guo, X. Yang, K.-T. Cheng, and J. Sun, “Metapruning: Meta Learning For Automatic Neural Network Channel Pruning,” in ICCV, 2019.
283. T. O’Shea and J. Hoydis, “An Introduction To Deep Learning For The Physical Layer,” IEEE Transactions on Cognitive Communications and Networking, 2017.
284. Y. Jiang, H. Kim, H. Asnani, and S. Kannan, “MIND: Model Independent Neural Decoder,” arXiv e-prints, 2019.
285. B. Settles, “Active Learning,” Synthesis Lectures on Artificial Intelligence and Machine Learning, 2012.
286. I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining And Harnessing Adversarial Examples,” in ICLR, 2015.
287. C. Yin, J. Tang, Z. Xu, and Y. Wang, “Adversarial Meta-Learning,” CoRR, 2018.
288. M. Vartak, A. Thiagarajan, C. Miranda, J. Bratman, and H. Larochelle, “A meta-learning perspective on cold-start recommendations for items,” in NIPS, 2017.
289. H. Bharadhwaj, “Meta-learning for user cold-start recommendation,” in IJCNN, 2019.
290. T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, “Gradient Surgery For Multi-Task Learning,” 2020.
291. Z. Kang, K. Grauman, and F. Sha, “Learning With Whom To Share In Multi-task Feature Learning,” in ICML, 2011.
292. Y. Yang and T. Hospedales, “A Unified Perspective On MultiDomain And Multi-Task Learning,” in ICLR, 2015.
293. K. Allen, E. Shelhamer, H. Shin, and J. Tenenbaum, “Infinite Mixture Prototypes For Few-shot Learning,” in ICML, 2019.
294. F. Pedregosa, “Hyperparameter optimization with approximate gradient,” in ICML, 2016.
295. R. J. Williams and D. Zipser, “A learning algorithm for continually running fully recurrent neural networks,” Neural Computation, vol. 1, no. 2, pp. 270–280, 1989.
296. A. Shaban, C.-A. Cheng, N. Hatch, and B. Boots, “Truncated backpropagation for bilevel optimization,” in AISTATS, 2019.
297. J. Fu, H. Luo, J. Feng, K. H. Low, and T.-S. Chua, “DrMAD: Distilling reverse-mode automatic differentiation for optimizing hyperparameters of deep neural networks,” in IJCAI, 2016.
298. S. Flennerhag, P. G. Moreno, N. Lawrence, and A. Damianou, “Transferring knowledge across learning processes,” in ICLR, 2019.
299. Y. Wu, M. Ren, R. Liao, and R. Grosse, “Understanding shorthorizon bias in stochastic meta-optimization,” in ICLR, 2018. 

## Authors
* Timothy Hospedales is a Professor at the University of Edinburgh, and Principal Researcher at Samsung AI Research. His research interest is in data efficient and robust learning-to-learn with diverse applications in vision, language, reinforcement learning, and beyond. 
* Antreas Antoniou is a PhD student at the University of Edinburgh, supervised by Amos Storkey. His research contributions in metalearning and few-shot learning are commonly seen as key benchmarks in the field. His main interests lie around meta-learning better learning priors such as losses, initializations and neural network layers, to improve few-shot and life-long learning. 
* Paul Micaelli is a PhD student at the University of Edinburgh, supervised by Amos Storkey and Timothy Hospedales. His research focuses on zero-shot knowledge distillation and on metalearning over long horizons for many-shot problems. Amos Storkey is Professor of Machine Learning and AI in the School of Informatics, University of Edinburgh. He leads a research team focused on deep neural networks, Bayesian and probabilistic models, efficient inference and meta-learning.
