# Training data-efficient image transformers & distillation through attention
通过注意力训练数据高效的图像转换器和蒸馏 2020.12.23 https://arxiv.org/abs/2012.12877


## Abstract
Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. These highperforming vision transformers are pre-trained with hundreds of millions of images using a large infrastructure, thereby limiting their adoption.

最近，纯基于注意力的神经网络被证明可以解决图像理解任务，如图像分类。这些高性能ViT使用大型基础设施预训练了数亿张图像，从而限制了它们的采用。

In this work, we produce competitive convolution-free transformers by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop) on ImageNet with no external data.

在这项工作中，我们仅通过在Imagenet上进行训练来生成具有竞争力的无卷积变换器。我们在不到3天的时间里用一台电脑训练他们。我们的参考ViT(86M参数)在没有外部数据的情况下，在ImageNet上实现了83.1%(单剪裁)的顶级精度。

More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models. 

更重要的是，我们引入了专门针对transformers的师生策略。它依赖于一个蒸馏令牌，确保学生通过注意力从老师那里学习。我们展示了这种基于令牌的蒸馏的兴趣，特别是当使用convnet作为教师时。这导致我们在Imagenet(我们获得了高达85.2%的准确率)和迁移到其他任务时报告的结果与convnets相比具有竞争力。我们共享代码和模型。

## 1 Introduction
Convolutional neural networks have been the main design paradigm for image understanding tasks, as initially demonstrated on image classification tasks. One of the ingredient to their success was the availability of a large training set, namely Imagenet [13, 42]. Motivated by the success of attention-based models in Natural Language Processing [14, 52], there has been increasing interest in architectures leveraging attention mechanisms within convnets [2, 34, 61]. More recently several researchers have proposed hybrid architecture transplanting transformer ingredients to convnets to solve vision tasks [6, 43].

卷积神经网络已经成为图像理解任务的主要设计范式，正如最初在图像分类任务中所证明的那样。他们成功的一个因素是提供了一个大型训练集，即Imagenet[13，42]。受自然语言处理中基于注意力模型的成功[14，52]的激励，人们对利用卷积网络内注意力机制的架构越来越感兴趣[2，34，61]。最近，一些研究人员提出了混合架构，将变换器组件移植到卷积网络以解决视觉任务[6，43]。

The vision transformer (ViT) introduced by Dosovitskiy et al. [15] is an architecture directly inherited from Natural Language Processing [52], but applied to image classification with raw image patches as input. Their paper presented excellent results with transformers trained with a large private labelled image dataset (JFT-300M [46], 300 millions images). The paper concluded that transformers “do not generalize well when trained on insufficient amounts of data”, and the training of these models involved extensive computing resources.

Dosovitskiyet al [15]引入的ViT(ViT)是一种直接继承自自然语言处理[52]的架构，但应用于以原始图像块作为输入的图像分类。他们的论文展示了使用大型私有标注图像数据集(JFT-300M[46]，3亿张图像)训练的变换器的出色结果。论文得出结论，变换器“在数据量不足的情况下进行训练时，不能很好地泛化”，并且这些模型的训练涉及大量的计算资源。

![fig1](../images/deiT/fig_1.png)<br/>
Figure 1: Throughput and accuracy on Imagenet of our methods compared to EfficientNets, trained on Imagenet1k only. The throughput is measured as the number of images processed per second on a V100 GPU. DeiT-B is identical to VIT-B, but the training is more adapted to a data-starving regime. It is learned in a few days on one machine. The symbol ⚗ refers to models trained with our transformer-specific distillation. See Table 5 for details and more models. 
图1：与仅在Imagenet1k上训练的EfficientNets相比，我们的方法在Imagenet上的吞吐量和准确性。吞吐量以V100 GPU上每秒处理的图像数量来衡量。DeiT-B与VIT-B相同，但训练更适合数据匮乏的情况。在一台机器上几天就能学会。符号⚗ 指的是经过我们变换器特定蒸馏训练的模型。有关详情和更多模型，请参见表5。

In this paper, we train a vision transformer on a single 8-GPU node in two to three days (53 hours of pre-training, and optionally 20 hours of fine-tuning) that is competitive with convnets having a similar number of parameters and efficiency. It uses Imagenet as the sole training set. We build upon the visual transformer architecture from Dosovitskiy et al. [15] and improvements included in the timm library [55]. With our Data-efficient image Transformers (DeiT), we report large improvements over previous results, see Figure 1. Our ablation study details the hyper-parameters and key ingredients for a successful training, such as repeated augmentation.

在本文中，我们在两到三天内(53小时的预训练，可选20小时的微调)在单个8-GPU节点上训练ViT，这与具有相似数量参数和效率的卷积网络相竞争。它使用Imagenet作为唯一的训练集。我们基于Dosovitskiyet al 的视觉转换器架构。[15]和timm库中的改进[55]。使用我们的数据高效图像转换器(DeiT)，我们报告了与先前结果相比的巨大改进，见图1。我们的消融研究详细介绍了成功训练的超参数和关键成分，如重复增强。

We address another question: how to distill these models? We introduce a token-based strategy, specific to transformers and denoted by DeiT ⚗ , and show that it advantageously replaces the usual distillation. 

我们解决了另一个问题：如何蒸馏这些模型？我们引入了一种基于令牌的策略，该策略特定于变换器，由DeiT表示⚗ , 并表明它有利地代替了通常的蒸馏。

In summary, our work makes the following contributions:
* We show that our neural networks that contains no convolutional layer can achieve competitive results against the state of the art on ImageNet with no external data. They are learned on a single node with 4 GPUs in three days(1We can accelerate the learning of the larger model DeiT-B by training it on 8 GPUs in two days. ) . Our two new models DeiT-S and DeiT-Ti have fewer parameters and can be seen as the counterpart of ResNet-50 and ResNet-18.
* We introduce a new distillation procedure based on a distillation token, which plays the same role as the class token, except that it aims at reproducing the label estimated by the teacher. Both tokens interact in the transformer through attention. This transformer-specific strategy outperforms vanilla distillation by a significant margin.
* Interestingly, with our distillation, image transformers learn more from a convnet than from another transformer with comparable performance.
* Our models pre-learned on Imagenet are competitive when transferred to different downstream tasks such as fine-grained classification, on several popular public benchmarks: CIFAR-10, CIFAR-100, Oxford-102 flowers, Stanford Cars and iNaturalist-18/19.

总之，我们的工作做出了以下贡献：
* 我们表明，我们的神经网络不包含卷积层，可以在没有外部数据的情况下，与ImageNet上的最新技术相比，获得具有竞争力的结果。它们在三天内通过4个GPU在单个节点上学习(1我们可以在两天内在8个GPU上训练更大模型的DeiT-B，从而加快学习速度)。我们的两个新模型DeiT-S和DeiT-Ti具有更少的参数，可以看作是ResNet-50和ResNet-18的对应。
* 我们引入了一种基于蒸馏令牌的新蒸馏程序，该程序与课堂令牌的作用相同，但其目的是复制老师估计的标签。这两个令牌通过注意力在转换器中交互。这种特定于变换器的策略显著优于普通蒸馏。
* 有趣的是，通过我们的蒸馏，图像转换器从convnet中学到的信息比从另一个性能相当的转换器中学到的更多。
* 我们在Imagenet上预先学习的模型在迁移到不同的下游任务(如细粒度分类)时具有竞争力，基于几个流行的公共基准：CIFAR-10、CIFAR-100、Oxford-102 flowers、Stanford Cars和iNaturalist-18/19。

This paper is organized as follows: we review related works in Section 2, and focus on transformers for image classification in Section 3. We introduce our distillation strategy for transformers in Section 4. The experimental section 5 provides analysis and comparisons against both convnets and recent transformers, as well as a comparative evaluation of our transformer-specific distillation. Section 6 details our training scheme. It includes an extensive ablation of our data-efficient training choices, which gives some insight on the key ingredients involved in DeiT. We conclude in Section 7. 

本文的组织结构如下：我们在第2节中回顾了相关工作，并在第3节中重点介绍了用于图像分类的变换器。我们在第4节中介绍了我们的变换器蒸馏策略。实验第5节提供了卷积网络和最新变换器的分析和比较，以及我们的变换器特定蒸馏的比较评估。第6节详细介绍了我们的训练计划。它包括对我们的数据高效训练选择的广泛删减，这对DeiT中涉及的关键要素提供了一些见解。我们在第7节中总结。

## 2 Related work
Image Classification is so core to computer vision that it is often used as a benchmark to measure progress in image understanding. Any progress usually translates to improvement in other related tasks such as detection or segmentation. Since 2012’s AlexNet [32], convnets have dominated this benchmark and have become the de facto standard. The evolution of the state of the art on the ImageNet dataset [42] reflects the progress with convolutional neural network architectures and learning [32, 44, 48, 50, 51, 57].

图像分类是计算机视觉的核心，经常被用作衡量图像理解进展的基准。任何进展通常会转化为其他相关任务的改进，如检测或分割。自2012年的AlexNet[32]以来，convnets一直主导着这一基准，并成为事实上的标准。ImageNet数据集[42]上最新技术的发展反映了卷积神经网络架构和学习的进展[32，44，48，50，51，57]。

Despite several attempts to use transformers for image classification [7], until now their performance has been inferior to that of convnets. Nevertheless hybrid architectures that combine convnets and transformers, including the self-attention mechanism, have recently exhibited competitive results in image classification [56], detection [6, 28], video processing [45, 53], unsupervised object discovery [35], and unified text-vision tasks [8, 33, 37]. 

尽管多次尝试使用变换器进行图像分类[7]，但到目前为止，它们的性能一直不如卷积网络。然而，结合了卷积网络和变换器的混合架构，包括自注意机制，最近在图像分类[56]、检测[6，28]、视频处理[45，53]、无监督对象发现[35]和统一文本视觉任务[8，33，37]中显示出了竞争性的结果。

Recently Vision transformers (ViT) [15] closed the gap with the state of the art on ImageNet, without using any convolution. This performance is remarkable since convnet methods for image classification have benefited from years of tuning and optimization [22, 55]. Nevertheless, according to this study [15], a pre-training phase on a large volume of curated data is required for the learned transformer to be effective. In our paper we achieve a strong performance without requiring a large training dataset, i.e., with Imagenet1k only.

最近，视觉变换器(ViT)[15]在不使用任何卷积的情况下，填补了与ImageNet上最先进技术的差距。这种性能是显著的，因为用于图像分类的convnet方法得益于多年的调整和优化[22，55]。然而，根据这项研究[15]，需要对大量精确数据进行预训练，以使学习的transformer有效。在我们的论文中，我们在不需要大型训练数据集的情况下实现了强大的性能，即仅使用Imagenet1k。

The Transformer architecture, introduced by Vaswani et al. [52] for machine translation are currently the reference model for all natural language processing (NLP) tasks. Many improvements of convnets for image classification are inspired by transformers. For example, Squeeze and Excitation [2], Selective Kernel [34] and Split-Attention Networks [61] exploit mechanism akin to transformers self-attention (SA) mechanism.

Vaswaniet al [52]为机器翻译引入的Transformer架构目前是所有自然语言处理(NLP)任务的参考模型。用于图像分类的卷积网络的许多改进都受到了transformers的启发。例如，挤压和激励[2]、选择性核[34]和分裂注意力网络[61]利用了类似于transformers的自注意力(SA)机制的机制。

Knowledge Distillation (KD), introduced by Hinton et al. [24], refers to the training paradigm in which a student model leverages “soft” labels coming from a strong teacher network. This is the output vector of the teacher’s softmax function rather than just the maximum of scores, wich gives a “hard” label. Such a training improves the performance of the student model (alternatively, it can be regarded as a form of compression of the teacher model into a smaller one – the student). On the one hand the teacher’s soft labels will have a similar effect to labels smoothing [58]. On the other hand as shown by Wei et al. [54] the teacher’s supervision takes into account the effects of the data augmentation, which sometimes causes a misalignment between the real label and the image. For example, let us consider image with a “cat” label that represents a large landscape and a small cat in a corner. If the cat is no longer on the crop of the data augmentation it implicitly changes the label of the image. KD can transfer inductive biases [1] in a soft way in a student model using a teacher model where they would be incorporated in a hard way. For example, it may be useful to induce biases due to convolutions in a transformer model by using a convolutional model as teacher. In our paper we study the distillation of a transformer student by either a convnet or a transformer teacher. We introduce a new distillation procedure specific to transformers and show its superiority. 

Hintonet al [24]引入的知识蒸馏(KD)是指学生模型利用来自强大教师网络的“软”标签的训练范式。这是老师的softmax函数的输出向量，而不仅仅是分数的最大值，它给出了一个“硬”标签。这样的训练提高了学生模型的性能(或者，它可以被视为将教师模型压缩成更小的模型——学生)。一方面，教师的软标签将具有类似于标签平滑的效果[58]。另一方面，如Weiet al 所示。[54]教师的监督考虑了数据增广的影响，这有时会导致真实标签和图像之间的错位。例如，让我们考虑一个带有“猫”标签的图像，它代表一个大的景观，一只小猫在角落里。如果猫不再处于数据增广的裁剪中，它会隐式地更改图像的标签。KD可以使用教师模型在学生模型中以软的方式传递归纳偏差[1]，并以硬的方式将其结合。例如，通过使用卷积模型作为教师，在transformer模型中引入由于卷积引起的偏差可能是有用的。在我们的论文中，我们研究了由一个convnet或一个transformers教师对transformer学生的蒸馏。我们介绍了一种适用于transformer的新蒸馏方法，并展示了其优越性。

## 3 Vision transformer: overview
In this section, we briefly recall preliminaries associated with the vision transformer [15, 52], and further discuss positional encoding and resolution.

在本节中，我们简要回顾了与视觉变换器相关的初步知识[15，52]，并进一步讨论了位置编码和分辨率。

### Multi-head Self Attention layers (MSA). 
The attention mechanism is based on a trainable associative memory with (key, value) vector pairs. A query vector ${q ∈ R^d}$ is matched against a set of k key vectors (packed together into a matrix $K ∈ R^{k×d} $) using inner products. These inner products are then scaled and 4 normalized with a softmax function to obtain k weights. The output of the attention is the weighted sum of a set of k value vectors (packed into $V ∈ R^{k×d} $). For a sequence of N query vectors (packed into $Q ∈ R^{N×d}$ ), it produces an output matrix (of size N × d):

注意力机制基于具有(键、值)向量对的可训练联想记忆。查询向量${q∈ R^d}$与一组k个关键向量匹配(一起打包成矩阵$k∈ R^{k×d}$)。然后对这些内积进行缩放，并用softmax函数对其进行4归一化，以获得k个权重。注意力的输出是一组k值向量的加权和(打包成$V∈ R^{k×d}$)。对于N个查询向量的序列(打包到$Q中∈ R^{N×d}$)，它产生一个输出矩阵(大小为N×d)：

$Attention(Q, K, V ) = Softmax(QK^T / \sqrt{d})V$, (1) 

where the Softmax function is applied over each row of the input matrix and the $\sqrt{d}$ term provides appropriate normalization.

其中Softmax函数应用于输入矩阵的每一行，$\sqrt{d}$项提供适当的归一化。

In [52], a Self-attention layer is proposed. Query, key and values matrices are themselves computed from a sequence of N input vectors (packed into $X ∈ R^{N×D}$): $Q = XW_Q$, $K = XW_K$, $V = XW_V$, using linear transformations $W_Q, W_K, W_V$ with the constraint k = N, meaning that the attention is in between all the input vectors.

在[52]中，提出了一种自注意层。查询、键和值矩阵本身是从N个输入向量序列(打包成$X∈ R^{N×D}$)：$Q=XW_Q$，$K=XW_K$，$V=XW_V$，使用线性变换$W_Q，W_K，W_V$和约束K=N，这意味着注意力在所有输入向量之间。

Finally, Multi-head self-attention layer (MSA) is defined by considering h attention “heads”, ie h self-attention functions applied to the input. Each head provides a sequence of size N × d. These h sequences are rearranged into a N × dh sequence that is reprojected by a linear layer into N × D.

最后，通过考虑h个注意“头部”(即应用于输入的h个自我注意函数)来定义多头部自我注意层(MSA)。每个头部提供大小为N×d的序列。这些h个序列被重新排列成N×dh序列，该序列被线性层重新投影成N×d。

### Transformer block for images. 
To get a full transformer block as in [52], we add a Feed-Forward Network (FFN) on top of the MSA layer. This FFN is composed of two linear layers separated by a GeLu activation [23]. The first linear layer expands the dimension from D to 4D, and the second layer reduces the dimension from 4D back to D. Both MSA and FFN are operating as residual operators thank to skip-connections, and with a layer normalization [3].

图像转换器块。为了获得[52]中的完整变换器块，我们在MSA层顶部添加了前馈网络(FFN)。该FFN由GeLu激活所分隔的两个线性层组成[23]。第一个线性层将维度从D扩展到4D，第二个层将维度由4D减小到D。由于跳过连接，MSA和FFN都作为残余算子运行，并且具有层规范化[3]。

In order to get a transformer to process images, our work builds upon the ViT model [15]. It is a simple and elegant architecture that processes input images as if they were a sequence of input tokens. The fixed-size input RGB image is decomposed into a batch of N patches of a fixed size of 16 × 16 pixels (N = 14 × 14). Each patch is projected with a linear layer that conserves its overall dimension 3 × 16 × 16 = 768.

为了获得处理图像的转换器，我们的工作建立在ViT模型的基础上[15]。它是一个简单而优雅的架构，可以处理输入图像，就像它们是一系列输入令牌一样。固定大小的输入RGB图像被分解为一批固定大小为16×16像素(N=14×14)的N个块。每个分块都投影有一个线性层，该层保持其整体尺寸3×16×16=768。

The transformer block described above is invariant to the order of the patch embeddings, and thus does not consider their relative position. The positional information is incorporated as fixed [52] or trainable [18] positional embeddings. They are added before the first transformer block to the patch tokens, which are then fed to the stack of transformer blocks.

上述变换器块对分块嵌入的顺序是不变的，因此不考虑它们的相对位置。位置信息被合并为固定的[52]或可训练的[18]位置嵌入。它们在第一个转换器块之前添加到分块令牌，然后将其馈送到转换器块堆栈。

### The class token 
is a trainable vector, appended to the patch tokens before the first layer, that goes through the transformer layers, and is then projected with a linear layer to predict the class. This class token is inherited from NLP [14], and departs from the typical pooling layers used in computer vision to predict the class. The transformer thus process batches of (N + 1) tokens of dimension D, of which only the class vector is used to predict the output. This architecture forces the self-attention to spread information between the patch tokens and the class token: at training time the supervision signal comes only from the class embedding, while the patch tokens are the model’s only variable input. 

类令牌。是一个可训练的向量，附加到第一层之前的分块标记上，经过变换器层，然后用线性层投影以预测类。该类标记继承自NLP[14]，与计算机视觉中用于预测类的典型池化层不同。因此，转换器处理一批(N+1)个维度D的令牌，其中只有类向量用于预测输出。这种架构迫使自注意在分块令牌和类令牌之间传播信息：在训练时，监督信号仅来自类嵌入，而分块令牌是模型的唯一变量输入。

### Fixing the positional encoding across resolutions. 
Touvron et al. [50] show that it is desirable to use a lower training resolution and fine-tune the network at the larger resolution. This speeds up the full training and improves the accuracy under prevailing data augmentation schemes. When increasing the resolution of an input image, we keep the patch size the same, therefore the number N of input patches does change. Due to the architecture of transformer blocks and the class token, the model and classifier do not need to be modified to process more tokens. In contrast, one needs to adapt the positional embeddings, because there are N of them, one for each patch. Dosovitskiy et al. [15] interpolate the positional encoding when changing the resolution and demonstrate that this method works with the subsequent fine-tuning stage. 

修复不同分辨率的位置编码。Touvronet al [50]表明，希望使用较低的训练分辨率，并以较高的分辨率微调网络。这加快了完整的训练，并提高了主流数据增广方案下的准确性。当增加输入图像的分辨率时，我们保持分块大小不变，因此输入分块的数量N会发生变化。由于转换器块和类令牌的架构，不需要修改模型和分类器来处理更多令牌。相比之下，需要调整位置嵌入，因为其中有N个，每个分块一个。Dosovitskiyet al [15]在改变分辨率时插入位置编码，并证明该方法适用于后续微调阶段。

## 4 Distillation through attention
In this section we assume we have access to a strong image classifier as a teacher model. It could be a convnet, or a mixture of classifiers. We address the question of how to learn a transformer by exploiting this teacher. As we will see in Section 5 by comparing the trade-off between accuracy and image throughput, it can be beneficial to replace a convolutional neural network by a transformer. This section covers two axes of distillation: hard distillation versus soft distillation, and classical distillation versus the distillation token.

Soft distillation [24, 54] minimizes the Kullback-Leibler divergence between the softmax of the teacher and the softmax of the student model.

Let Zt be the logits of the teacher model, Zs the logits of the student model.

We denote by τ the temperature for the distillation, λ the coefficient balancing the Kullback–Leibler divergence loss (KL) and the cross-entropy (LCE) on ground truth labels y, and ψ the softmax function. The distillation objective is

Lglobal = (1 − λ)LCE(ψ(Zs), y) + λτ 2KL(ψ(Zs/τ ), ψ(Zt/τ )). (2)

Hard-label distillation. We introduce a variant of distillation where we take the hard decision of the teacher as a true label. Let yt = argmaxcZt(c) be the hard decision of the teacher, the objective associated with this hard-label distillation is:

L hardDistill global = 1 2

LCE(ψ(Zs), y) + 1 2

LCE(ψ(Zs), yt). (3)

For a given image, the hard label associated with the teacher may change depending on the specific data augmentation. We will see that this choice is better than the traditional one, while being parameter-free and conceptually simpler: The teacher prediction yt plays the same role as the true label y.

Note also that the hard labels can also be converted into soft labels with label smoothing [47], where the true label is considered to have a probability of 1 − ε, and the remaining ε is shared across the remaining classes. We fix this parameter to ε = 0.1 in our all experiments that use true labels. 6 self-attention

FFN class token distillation token patch tokens

Figure 2: Our distillation procedure: we simply include a new distillation token.

It interacts with the class and patch tokens through the self-attention layers.

This distillation token is employed in a similar fashion as the class token, except that on output of the network its objective is to reproduce the (hard) label predicted by the teacher, instead of true label. Both the class and distillation tokens input to the transformers are learned by back-propagation.

Distillation token. We now focus on our proposal, which is illustrated in

Figure 2. We add a new token, the distillation token, to the initial embeddings (patches and class token). Our distillation token is used similarly as the class token: it interacts with other embeddings through self-attention, and is output by the network after the last layer. Its target objective is given by the distillation component of the loss. The distillation embedding allows our model to learn from the output of the teacher, as in a regular distillation, while remaining complementary to the class embedding.

Interestingly, we observe that the learned class and distillation tokens converge towards different vectors: the average cosine similarity between these tokens equal to 0.06. As the class and distillation embeddings are computed at each layer, they gradually become more similar through the network, all the way through the last layer at which their similarity is high (cos=0.93), but still lower than 1. This is expected since as they aim at producing targets that are similar but not identical. 7

We verified that our distillation token adds something to the model, compared to simply adding an additional class token associated with the same target label: instead of a teacher pseudo-label, we experimented with a transformer with two class tokens. Even if we initialize them randomly and independently, during training they converge towards the same vector (cos=0.999), and the output embedding are also quasi-identical. This additional class token does not bring anything to the classification performance. In contrast, our distillation strategy provides a significant improvement over a vanilla distillation baseline, as validated by our experiments in Section 5.2.

Fine-tuning with distillation. We use both the true label and teacher prediction during the fine-tuning stage at higher resolution. We use a teacher with the same target resolution, typically obtained from the lower-resolution teacher by the method of Touvron et al [50]. We have also tested with true labels only but this reduces the benefit of the teacher and leads to a lower performance.

Classification with our approach: joint classifiers. At test time, both the class or the distillation embeddings produced by the transformer are associated with linear classifiers and able to infer the image label. Yet our referent method is the late fusion of these two separate heads, for which we add the softmax output by the two classifiers to make the prediction. We evaluate these three options in Section 5. 

## 5 Experiments
This section presents a few analytical experiments and results. We first discuss our distillation strategy. Then we comparatively analyze the efficiency and accuracy of convnets and vision transformers.

## 5.1 Transformer models
As mentioned earlier, our architecture design is identical to the one proposed by Dosovitskiy et al. [15] with no convolutions. Our only differences are the training strategies, and the distillation token. Also we do not use a MLP head for the pre-training but only a linear classifier. To avoid any confusion, we refer to the results obtained in the prior work by ViT, and prefix ours by DeiT. If not specified, DeiT refers to our referent model DeiT-B, which has the same architecture as ViT-B. When we fine-tune DeiT at a larger resolution, we append the resulting operating resolution at the end, e.g, DeiT-B↑384. Last, when using our distillation procedure, we identify it with an alembic sign as DeiT ⚗.

The parameters of ViT-B (and therefore of DeiT-B) are fixed as D = 768, h = 12 and d = D/h = 64. We introduce two smaller models, namely DeiT-S and DeiT-Ti, for which we change the number of heads, keeping d fixed. Table 1 summarizes the models that we consider in our paper. 8

Table 1: Variants of our DeiT architecture. The larger model, DeiT-B, has the same architecture as the ViT-B [15]. The only parameters that vary across models are the embedding dimension and the number of heads, and we keep the dimension per head constant (equal to 64). Smaller models have a lower parameter count, and a faster throughput. The throughput is measured for images at resolution 224×224.


Table 2: We compare on ImageNet [42] the performance (top-1 acc., %) of the student as a function of the teacher model used for distillation.



## 5.2 Distillation
Our distillation method produces a vision transformer that becomes on par with the best convnets in terms of the trade-off between accuracy and throughput, see Table 5. Interestingly, the distilled model outperforms its teacher in terms of the trade-off between accuracy and throughput. Our best model on

ImageNet-1k is 85.2% top-1 accuracy outperforms the best Vit-B model pretrained on JFT-300M at resolution 384 (84.15%). For reference, the current state of the art of 88.55% achieved with extra training data was obtained by the ViTH model (600M parameters) trained on JFT-300M at resolution 512. Hereafter we provide several analysis and observations.

Convnets teachers. We have observed that using a convnet teacher gives better performance than using a transformer. Table 2 compares distillation results with different teacher architectures. The fact that the convnet is a better teacher is probably due to the inductive bias inherited by the transformers through distillation, as explained in Abnar et al. [1]. In all of our subsequent distillation experiments the default teacher is a RegNetY-16GF [40] (84M parameters) that we trained with the same data and same data-augmentation as DeiT. This teacher reaches 82.9% top-1 accuracy on ImageNet. 9

Table 3: Distillation experiments on Imagenet with DeiT, 300 epochs of pretraining. We report the results for our new distillation method in the last three rows. We separately report the performance when classifying with only one of the class or distillation embeddings, and then with a classifier taking both of them as input. In the last row (class+distillation), the result correspond to the late fusion of the class and distillation classifiers.



Comparison of distillation methods. We compare the performance of different distillation strategies in Table 3. Hard distillation significantly outperforms soft distillation for transformers, even when using only a class token: hard distillation reaches 83.0% at resolution 224×224, compared to the soft distillation accuracy of 81.8%. Our distillation strategy from Section 4 further improves the performance, showing that the two tokens provide complementary information useful for classification: the classifier on the two tokens is significantly better than the independent class and distillation classifiers, which by themselves already outperform the distillation baseline.

The distillation token gives slightly better results than the class token. It is also more correlated to the convnets prediction. This difference in performance is probably due to the fact that it benefits more from the inductive bias of convnets. We give more details and an analysis in the next paragraph. The distillation token has an undeniable advantage for the initial training.

Agreement with the teacher & inductive bias? As discussed above, the architecture of the teacher has an important impact. Does it inherit existing inductive bias that would facilitate the training? While we believe it difficult to formally answer this question, we analyze in Table 4 the decision agreement between the convnet teacher, our image transformer DeiT learned from labels only, and our transformer DeiT ⚗.

Our distilled model is more correlated to the convnet than with a transformer learned from scratch. As to be expected, the classifier associated with the distillation embedding is closer to the convnet that the one associated with the class embedding, and conversely the one associated with the class embedding is more similar to DeiT learned without distillation. Unsurprisingly, the joint class+distil classifier offers a middle ground. 10

Table 4: Disagreement analysis between convnet, image transformers and distillated transformers: We report the fraction of sample classified differently for all classifier pairs, i.e., the rate of different decisions. We include two models without distillation (a RegNetY and DeiT-B), so that we can compare how our distilled models and classification heads are correlated to these teachers. groundtruth no distillation DeiT ⚗ student (of the convnet) convnet DeiT class distillation DeiT ⚗ 



Number of epochs. Increasing the number of epochs significantly improves the performance of training with distillation, see Figure 3. With 300 epochs, our distilled network DeiT-B ⚗ is already better than DeiT-B. But while for the latter the performance saturates with longer schedules, our distilled network clearly benefits from a longer training time.

## 5.3 Efficiency vs accuracy: a comparative study with convnets
In the literature, the image classificaton methods are often compared as a compromise between accuracy and another criterion, such as FLOPs, number of parameters, size of the network, etc.

We focus in Figure 1 on the tradeoff between the throughput (images processed per second) and the top-1 classification accuracy on ImageNet. We focus on the popular state-of-the-art EfficientNet convnet, which has benefited from years of research on convnets and was optimized by architecture search on the

ImageNet validation set.

Our method DeiT is slightly below EfficientNet, which shows that we have almost closed the gap between vision transformers and convnets when training with Imagenet only. These results are a major improvement (+6.3% top-1 in a comparable setting) over previous ViT models trained on Imagenet1k only [15].

Furthermore, when DeiT benefits from the distillation from a relatively weaker

RegNetY to produce DeiT ⚗ , it outperforms EfficientNet. It also outperforms by 1% (top-1 acc.) the Vit-B model pre-trained on JFT300M at resolution 384 (85.2% vs 84.15%), while being significantly faster to train.

Table 5 reports the numerical results in more details and additional evaluations on ImageNet V2 and ImageNet Real, that have a test set distinct from the ImageNet validation, which reduces overfitting on the validation set. Our results show that DeiT-B ⚗ and DeiT-B ⚗ ↑384 outperform, by some margin, the state of the art on the trade-off between accuracy and inference time on GPU. 11 image throughput ImNet Real V2


Table 5: Throughput on and accuracy on Imagenet [42], Imagenet Real [5] and

Imagenet V2 matched frequency [41] of DeiT and of several state-of-the-art convnets, for models trained with no external data. The throughput is measured as the number of images that we can process per second on one 16GB

V100 GPU. For each model we take the largest possible batch size for the usual resolution of the model and calculate the average time over 30 runs to process that batch. With that we calculate the number of images processed per second.

Throughput can vary according to the implementation: for a direct comparison and in order to be as fair as possible, we use for each model the definition in the same GitHub [55] repository. ? : Regnet optimized with a similar optimization procedure as ours, which boosts the results. These networks serve as teachers when we use our distillation strategy. 12 ⚗↑ ⚗

Figure 3: Distillation on ImageNet [42] with DeiT-B: performance as a function of the number of training epochs. We provide the performance without distillation (horizontal dotted line) as it saturates after 400 epochs.

## 5.4 Transfer learning: Performance on downstream tasks
Although DeiT perform very well on ImageNet it is important to evaluate them on other datasets with transfer learning in order to measure the power of generalization of DeiT. We evaluated this on transfer learning tasks by fine-tuning on the datasets in Table 6. Table 7 compares DeiT transfer learning results to those of ViT [15] and state of the art convolutional architectures [48]. DeiT is on par with competitive convnet models, which is in line with our previous conclusion on ImageNet.

Comparison vs training from scratch. We investigate the performance when training from scratch on a small dataset, without Imagenet pre-training. We get the following results on the small CIFAR-10, which is small both w.r.t. the number of images and labels:



Table 6: Datasets used for our different tasks.



Table 7: We compare Transformers based models on different transfer learning task with ImageNet pre-training. We also report results with convolutional architectures for reference.
ules (up to 7200 epochs, which corresponds to 300 Imagenet epochs) so that the network has been fed a comparable number of images in total; (2) we rescale images to 224 × 224 to ensure that we have the same augmentation. The results are not as good as with Imagenet pre-training (98.5% vs 99.1%), which is expected since the network has seen a much lower diversity. However they show that it is possible to learn a reasonable transformer on CIFAR-10 only. 

## 6 Training details & ablation

In this section we discuss the DeiT training strategy to learn vision transformers in a data-efficient manner. We build upon PyTorch [39] and the timm library [55]2 . We provide hyper-parameters as well as an ablation study in which we analyze the impact of each choice.

Initialization and hyper-parameters. Transformers are relatively sensitive to initialization. After testing several options in preliminary experiments, some 2The timm implementation already included a training procedure that improved the accuracy of ViT-B from 77.91% to 79.35% top-1, and trained on Imagenet-1k with a 8xV100 GPU machine. 14 top-1 accuracy



Table 8: Ablation study on training methods on ImageNet [42]. The top row (”none”) corresponds to our default configuration employed for DeiT. The symbols ✓ and ✗ indicates that we use and do not use the corresponding method, respectively. We report the accuracy scores (%) after the initial training at resolution 224×224, and after fine-tuning at resolution 384×384. The hyper-parameters are fixed according to Table 9, and may be suboptimal. * indicates that the model did not train well, possibly because hyper-parameters are not adapted. of them not converging, we follow the recommendation of Hanin and Rolnick [20] to initialize the weights with a truncated normal distribution.

Table 9 indicates the hyper-parameters that we use by default at training time for all our experiments, unless stated otherwise. For distillation we follow the recommendations from Cho et al. [9] to select the parameters τ and λ. We take the typical values τ = 3.0 and λ = 0.1 for the usual (soft) distillation.

Data-Augmentation. Compared to models that integrate more priors (such as convolutions), transformers require a larger amount of data. Thus, in order to train with datasets of the same size, we rely on extensive data augmentation.

We evaluate different types of strong data augmentation, with the objective to reach a data-efficient training regime.

Auto-Augment [11], Rand-Augment [12], and random erasing [62] improve the results. For the two latter we use the timm [55] customizations, and after ablation we choose Rand-Augment instead of AutoAugment. Overall our experiments confirm that transformers require a strong data augmentation: almost all the data-augmentation methods that we evaluate prove to be useful.

One exception is dropout, which we exclude from our training procedure. 

Table 9: Ingredients and hyper-parameters for our method and Vit-B.

Regularization & Optimizers. We have considered different optimizers and cross-validated different learning rates and weight decays. Transformers are sensitive to the setting of optimization hyper-parameters. Therefore, during cross-validation, we tried 3 different learning rates (5.10−4 , 3.10−4 , 5.10−5 ) and 3 weight decay (0.03, 0.04, 0.05). We scale the learning rate according to the batch size with the formula: lrscaled = lr 512 al. [19] except that we use 512 instead of 256 as the base value. × batchsize, similarly to Goyal et

The best results use the AdamW optimizer with the same learning rates as

ViT [15] but with a much smaller weight decay, as the weight decay reported in the paper hurts the convergence in our setting.

We have employed stochastic depth [29], which facilitates the convergence of transformers, especially deep ones [16, 17]. For vision transformers, they were first adopted in the training procedure by Wightman [55]. Regularization like Mixup [60] and Cutmix [59] improve performance. We also use repeated augmentation [4, 25], which provides a significant boost in performance and is one of the key ingredients of our proposed training procedure.

Exponential Moving Average (EMA). We evaluate the EMA of our network obtained after training. There are small gains, which vanish after fine-tuning: the EMA model has an edge of is 0.1 accuracy points, but when fine-tuned the two models reach the same (improved) performance.

Fine-tuning at different resolution. We adopt the fine-tuning procedure from

Touvron et al. [51]: our schedule, regularization and optimization procedure are identical to that of FixEfficientNet but we keep the training-time data aug- 16 image throughput Imagenet [42] Real [5] V2 [41] size (image/s) acc. top-1 acc. top-1 acc. top-1 1602


Table 10: Performance of DeiT trained at size 2242 for varying finetuning sizes on ImageNet-1k, ImageNet-Real and ImageNet-v2 matched frequency. mentation (contrary to the dampened data augmentation of Touvron et al. [51]).

We also interpolate the positional embeddings: In principle any classical image scaling technique, like bilinear interpolation, could be used. However, a bilinear interpolation of a vector from its neighbors reduces its ` 2-norm compared to its neighbors. These low-norm vectors are not adapted to the pre-trained transformers and we observe a significant drop in accuracy if we employ use directly without any form of fine-tuning. Therefore we adopt a bicubic interpolation that approximately preserves the norm of the vectors, before fine-tuning the network with either AdamW [36] or SGD. These optimizers have a similar performance for the fine-tuning stage, see Table 8.

By default and similar to ViT [15] we train DeiT models with at resolution 224 and we fine-tune at resolution 384. We detail how to do this interpolation in Section 3. However, in order to measure the influence of the resolution we have finetuned DeiT at different resolutions. We report these results in Table 10.

Training time. A typical training of 300 epochs takes 37 hours with 2 nodes or 53 hours on a single node for the DeiT-B.As a comparison point, a similar training with a RegNetY-16GF [40] (84M parameters) is 20% slower. DeiT-S and

DeiT-Ti are trained in less than 3 days on 4 GPU. Then, optionally we fine-tune the model at a larger resolution. This takes 20 hours on a single node (8 GPU) to produce a FixDeiT-B model at resolution 384×384, which corresponds to 25 epochs. Not having to rely on batch-norm allows one to reduce the batch size without impacting performance, which makes it easier to train larger models.

Note that, since we use repeated augmentation [4, 25] with 3 repetitions, we only see one third of the images during a single epoch3 . 

## 7 Conclusion

In this paper, we have introduced DeiT, which are image transformers that do not require very large amount of data to be trained, thanks to improved 3Formally it means that we have 100 epochs, but each is 3x longer because of the repeated augmentations. We prefer to refer to this as 300 epochs in order to have a direct comparison on the effective training time with and without repeated augmentation. 17 training and in particular a novel distillation procedure. Convolutional neural networks have optimized, both in terms of architecture and optimization during almost a decade, including through extensive architecture search that is prone to overfiting, as it is the case for instance for EfficientNets [51]. For

DeiT we have started the existing data augmentation and regularization strategies pre-existing for convnets, not introducing any significant architectural beyond our novel distillation token. Therefore it is likely that research on dataaugmentation more adapted or learned for transformers will bring further gains.

Therefore, considering our results, where image transformers are on par with convnets already, we believe that they will rapidly become a method of choice considering their lower memory footprint for a given accuracy.

We provide an open-source implementation of our method. It is available at https://github.com/facebookresearch/deit.

## Acknowledgements
Many thanks to Ross Wightman for sharing his ViT code and bootstrapping training method with the community, as well as for valuable feedback that helped us to fix different aspects of this paper. Thanks to Vinicius Reis, Mannat Singh, Ari Morcos, Mark Tygert, Gabriel Synnaeve, and other colleagues at Facebook for brainstorming and some exploration on this axis. Thanks to Ross Girshick and Piotr Dollar for constructive comments.

## References
1. Samira Abnar, Mostafa Dehghani, and Willem Zuidema. Transferring inductive biases through knowledge distillation. arXiv preprint arXiv:2006.00555, 2020.
2. Jie Hu andLi Shen and Gang Sun. Squeeze-and-excitation networks. arXiv preprint arXiv:1709.01507, 2017.
3. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
4. Maxim Berman, Herv´e J´egou, Andrea Vedaldi, Iasonas Kokkinos, and Matthijs Douze. Multigrain: a unified image embedding for classes and instances. arXiv preprint arXiv:1902.05509, 2019.
5. Lucas Beyer, Olivier J. H´enaff, Alexander Kolesnikov, Xiaohua Zhai, and Aaron van den Oord. Are we done with imagenet? arXiv preprint arXiv:2006.07159, 2020.
6. Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European Conference on Computer Vision, 2020.
7. Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In International Conference on Machine Learning, 2020.
8. Yen-Chun Chen, Linjie Li, Licheng Yu, A. E. Kholy, Faisal Ahmed, Zhe Gan, Y. Cheng, and Jing jing Liu. Uniter: Universal image-text representation learning. In European Conference on Computer Vision, 2020.
9. J. H. Cho and B. Hariharan. On the efficacy of knowledge distillation. International Conference on Computer Vision, 2019.
10. P. Chu, Xiao Bian, Shaopeng Liu, and Haibin Ling. Feature space augmentation for long-tailed data. arXiv preprint arXiv:2008.03673, 2020.
11. Ekin Dogus Cubuk, Barret Zoph, Dandelion Man´e, Vijay Vasudevan, and Quoc V. Le. Autoaugment: Learning augmentation policies from data. arXiv preprint arXiv:1805.09501, 2018.
12. Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Randaugment: Practical automated data augmentation with a reduced search space. arXiv preprint arXiv:1909.13719, 2019.
13. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.
14. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pretraining of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
15. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.
16. Angela Fan, Edouard Grave, and Armand Joulin. Reducing transformer depth on demand with structured dropout. arXiv preprint arXiv:1909.11556, 2019. ICLR 2020.
17. Angela Fan, Pierre Stock, Benjamin Graham, Edouard Grave, R´emi Gribonval, Herv´e J´egou, and Armand Joulin. Training with quantization noise for extreme model compression. arXiv preprint arXiv:2004.07320, 2020. 19
18. Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122, 2017.
19. Priya Goyal, Piotr Doll´ar, Ross B. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.
20. Boris Hanin and David Rolnick. How to start training: The effect of initialization and architecture. NIPS, 31, 2018.
21. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Conference on Computer Vision and Pattern Recognition, June 2016.
22. Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of tricks for image classification with convolutional neural networks. In Conference on Computer Vision and Pattern Recognition, 2019.
23. Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.
24. Geoffrey E. Hinton, Oriol Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.
25. Elad Hoffer, Tal Ben-Nun, Itay Hubara, Niv Giladi, Torsten Hoefler, and Daniel Soudry. Augment your batch: Improving generalization through instance repetition. In Conference on Computer Vision and Pattern Recognition, 2020.
26. Grant Van Horn, Oisin Mac Aodha, Yang Song, Alexander Shepard, Hartwig Adam, Pietro Perona, and Serge J. Belongie. The inaturalist challenge 2018 dataset. arXiv preprint arXiv:1707.06642, 2018.
27. Grant Van Horn, Oisin Mac Aodha, Yang Song, Alexander Shepard, Hartwig Adam, Pietro Perona, and Serge J. Belongie. The inaturalist challenge 2019 dataset. arXiv preprint arXiv:1707.06642, 2019.
28. H. Hu, Jiayuan Gu, Zheng Zhang, Jifeng Dai, and Y. Wei. Relation networks for object detection. Conference on Computer Vision and Pattern Recognition, 2018.
29. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q. Weinberger. Deep networks with stochastic depth. In European Conference on Computer Vision, 2016.
30. Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In 4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13), 2013.
31. Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, CIFAR, 2009.
32. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.
33. Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. VisualBERT: a simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557, 2019.
34. Xiang Li, Wenhai Wang, Xiaolin Hu, and Jian Yang. Selective kernel networks. Conference on Computer Vision and Pattern Recognition, 2019.
35. Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, A. Dosovitskiy, and Thomas Kipf. Objectcentric learning with slot attention. arXiv preprint arXiv:2006.15055, 2020.
36. I. Loshchilov and F. Hutter. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101, 2017. 20
37. Jiasen Lu, Dhruv Batra, D. Parikh, and Stefan Lee. Vilbert: Pretraining taskagnostic visiolinguistic representations for vision-and-language tasks. In NIPS, 2019.
38. M-E. Nilsback and A. Zisserman. Automated flower classification over a large number of classes. In Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing, 2008.
39. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In Advances in neural information processing systems, pages 8026–8037, 2019.
40. Ilija Radosavovic, Raj Prateek Kosaraju, Ross B. Girshick, Kaiming He, and Piotr Doll´ar. Designing network design spaces. Conference on Computer Vision and Pattern Recognition, 2020.
41. B. Recht, Rebecca Roelofs, L. Schmidt, and V. Shankar. Do imagenet classifiers generalize to imagenet? arXiv preprint arXiv:1902.10811, 2019.
42. Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. International journal of Computer Vision, 2015.
43. Zhuoran Shen, Irwan Bello, Raviteja Vemulapalli, Xuhui Jia, and Ching-Hui Chen. Global self-attention networks for image recognition. arXiv preprint arXiv:2010.03019, 2020.
44. K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations, 2015.
45. C. Sun, A. Myers, Carl Vondrick, Kevin Murphy, and C. Schmid. Videobert: A joint model for video and language representation learning. Conference on Computer Vision and Pattern Recognition, 2019.
46. Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In Proceedings of the IEEE international conference on computer vision, pages 843–852, 2017.
47. Christian Szegedy, V. Vanhoucke, S. Ioffe, Jon Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. Conference on Computer Vision and Pattern Recognition, 2016.
48. Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks. arXiv preprint arXiv:1905.11946, 2019.
49. Hugo Touvron, Alexandre Sablayrolles, M. Douze, M. Cord, and H. J´egou. Grafit: Learning fine-grained image representations with coarse labels. arXiv preprint arXiv:2011.12982, 2020.
50. Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herve Jegou. Fixing the train-test resolution discrepancy. NIPS, 2019.
51. Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Herv´e J´egou. Fixing the train-test resolution discrepancy: Fixefficientnet. arXiv preprint arXiv:2003.08237, 2020.
52. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.
53. X. Wang, Ross B. Girshick, A. Gupta, and Kaiming He. Non-local neural networks. Conference on Computer Vision and Pattern Recognition, 2018. 21
54. Longhui Wei, An Xiao, Lingxi Xie, Xin Chen, Xiaopeng Zhang, and Qi Tian. Circumventing outliers of autoaugment with knowledge distillation. European Conference on Computer Vision, 2020.
55. Ross Wightman. Pytorch image models. https://github.com/rwightman/ pytorch-image-models, 2019.
56. Bichen Wu, Chenfeng Xu, Xiaoliang Dai, Alvin Wan, Peizhao Zhang, Masayoshi Tomizuka, Kurt Keutzer, and Peter Vajda. Visual transformers: Token-based image representation and processing for computer vision. arXiv preprint arXiv:2006.03677, 2020.
57. Qizhe Xie, Eduard H. Hovy, Minh-Thang Luong, and Quoc V. Le. Selftraining with noisy student improves imagenet classification. arXiv preprint arXiv:1911.04252, 2019.
58. L. Yuan, F. Tay, G. Li, T. Wang, and Jiashi Feng. Revisit knowledge distillation: a teacher-free framework. Conference on Computer Vision and Pattern Recognition, 2020.
59. Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. arXiv preprint arXiv:1905.04899, 2019.
60. Hongyi Zhang, Moustapha Ciss´e, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.
61. Hang Zhang, Chongruo Wu, Zhongyue Zhang, Yi Zhu, Zhi Zhang, Haibin Lin, Yue Sun, Tong He, Jonas Muller, R. Manmatha, Mu Li, and Alexander Smola. Resnest: Split-attention networks. arXiv preprint arXiv:2004.08955, 2020.
62. Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In AAAI, 2020. 22
