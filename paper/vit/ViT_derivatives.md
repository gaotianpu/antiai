# Vision Transformer: Vit and its Derivatives
视觉Transformer: Vit及其变种  2022-5-12 原文:https://arxiv.org/abs/2205.11239 
https://github.com/lucidrains/vit-pytorch

## Abstract
Transformer, an attention-based encoder-decoder architecture, has not only revolutionized the field of natural language processing (NLP), but has also done some pioneering work in the field of computer vision (CV). Compared to convolutional neural networks (CNNs), the Vision Transformer (ViT) relies on excellent modeling capabilities to achieve very good performance on several benchmarks such as ImageNet, COCO, and ADE20k. ViT is inspired by the self-attention mechanism in natural language processing, where word embeddings are replaced with patch embeddings.
This paper reviews the derivatives in the field of ViT and the cross-applications of ViT with other fields.

Transformer是一种基于注意力的编码器-解码器架构，它不仅革新了自然语言处理(NLP)领域，而且在计算机视觉(CV)领域也做了一些开创性工作。与卷积神经网络(CNN)相比，依赖于ViT的出色建模能力，在ImageNet、COCO和ADE20k等基准上实现非常好的性能。ViT的灵感来自自然语言处理中的自注意力机制，其中词嵌入被分块嵌入所取代。
本文综述了ViT领域的衍生物以及ViT在其他领域的交叉应用。

## 1 Pyramid Vision Transformer  金字塔ViT
To overcome the quadratic complexity of the attention mechanism, the Pyramid Vision Transformer (PVT) uses a variant of self-attention called Spatial-Reduced Attention (SRA). It is characterized by spatial reduction of keys and values, similar to Linformer attention in the NLP field.

为了克服注意力机制的二次复杂度，金字塔ViT(PVT)使用了一种称为空间减少注意(SRA)的自注意力变体。它的特点是键和值的空间缩减，类似于NLP领域中的Linformer注意力。

By applying SRA, the feature space dimension of the whole model is slowly reduced and the concept of order is enhanced by applying positional embedding in all transformer blocks.PVT has been used as a backbone network for object detection and semantic segmentation to process high resolution images.

通过应用SRA，在所有transformer块中应用位置嵌入，可以缓慢降低整个模型的特征空间维数，增强顺序概念。PVT已被用作对象检测和语义分割的主干网络，以处理高分辨率图像。

Figure 1: Overall architecture of the proposed Pyramid Vision Transformer (PVT).
图1：金字塔视觉transformer(PVT)的总体架构。

Later on, the research team further improved their PVT model named PVT-v2, with the following major improvements.
* overlapping patch embedding
* convolutional feedforward networks
* linear-complexity self-attention layers.

随后，研究团队进一步改进了名为PVT-v2的PVT模型，主要改进如下。
* 重叠分块嵌入
* 卷积前馈网络
* 线性复杂度自注意力层。

Figure 2: PVT-v2.

Overlapping patches is a simple and general idea to improve ViT, especially for dense tasks (e.g. semantic segmentation).By exploiting overlapping regions/patch, PVT-v2 can obtain more local continuity of image representations.

重叠分块是改进ViT的一个简单而通用的想法，特别是对于密集任务(例如语义分割)。通过利用重叠区域/分块，PVT-v2可以获得更多图像表示的局部连续性。

Convolution between fully connected layers (FC) eliminates the need for fixed size positional encoding in each layer. The 3x3 deep convolution with zero padding (p=1) is designed to compensate for the removal of positional encoding from the model (they are still present, but only in the input). This process allows more flexibility to handle multiple image resolutions.

全连接层(FC)之间的卷积消除了在每个层中固定大小位置编码的需要。带零填充(p=1)的3x3深度卷积旨在补偿从模型中删除位置编码(它们仍然存在，但仅存在于输入中)。此过程允许更灵活地处理多个图像分辨率。

Finally, using key and value pooling(p=7), the self-attentive layer is reduced to a complexity similar to that of a CNN.
最后，使用key和value池(p=7)，自注意力层的复杂度降低到与CNN类似的程度。

## 2 Swin Transformer: Hierarchical Vision Transformer using Shifted Windows  使用移位窗口的分层ViT
Swin Transformer aims to build the idea of locality from the standard NLP transformer, i.e. local or window attention:

Swin Transformer旨在从标准NLP Transformer构建局部化理念，即局部或窗口注意力：

In the Swin Transformer, local self-attention is used for non-overlapping windows. The next layer of window-to-window communication produces hierarchical representation by progressively merging windows.

在Swin Transformer中，局部自注意力用于非重叠窗口。窗口到窗口通信的下一层通过逐步合并窗口生成分层表示。

Figure 3: Swin-transformer

As shown in Figure 3, the left shows the regular window partitioning scheme in the first layer, where self-attention is computed within each window. The window partitioning in the second layer on the right is shifted by 2 image patches, resulting in crossing the boundary of the previous window.

如图3所示，左侧显示了第一层中的常规窗口分区方案，其中计算了每个窗口内的自注意力。右侧第二层中的窗口分区被两个分块移动，从而跨越前一个窗口的边界。

Figure 4: local attention 局部注意力

The local self-attention scales linearly with image size O(M ∗N) instead of O($N^2$) in the window size used for sequence length N and M.

局部自注意力与图像大小O(M∗N) 成线性关系, 而不是用于序列长度N和M的窗口大小中的O($N^2$)。

By merging and adding many local layers, there is a global representation. In addition, the spatial dimensions of the feature maps has been significantly reduced. The authors claim to have achieved promising results on both ImageNet-1K and ImageNet-21K.

通过合并和添加许多局部层，可以实现全局表示。此外，特征图的空间维度已显著降低。作者声称在ImageNet-1K和ImageNet-21K上都取得了可喜的结果。

## 3 Scaling Vision Transformer 缩放Vit
Deep learning and scale are related. In fact, scale is a key component in pushing the state-of-theart. In this study, the authors from Google Brain Research trained a slightly modified ViT model with 2 billion parameters and achieved a top-1 accuracy of 90.45% on ImageNet. This over-parameterized generalized model was tested on few-shot learning, with only 10 examples per class. A top-1 accuracy of 84.86% was achieved on ImageNet.

深度学习与规模相关。事实上，规模是推动最先进技术的关键因素。在这项研究中，谷歌大脑研究的作者用20亿个参数训练了一个稍加修改的ViT模型，并在ImageNet上获得了90.45%的top-1精度。这种过度参数化的广义模型在少样本学习中进行了测试，每个分类只有10个样本。ImageNet的准确率为84.86%，排名第一。

Few-shot learning refers to fine-tuning a model with an extremely limited number of samples. The goal of few-shot learning is to motivate generalization by slightly adapting the acquired pre-trained knowledge to a specific task. If large models are successfully pre-trained, it makes sense to perform well with a very limited understanding of the downstream task (provided by only a few examples).

少样本学习指的是用极其有限的样本数对模型进行微调。少样本学习的目标是通过将获得的预训练的知识稍稍适应特定任务来激发泛化。如果大型模型成功地进行了预先训练，那么在对下游任务的理解非常有限的情况下(仅通过几个样本提供)表现良好是有意义的。

The following are some of the core contributions and main results of this paper.
* Representation quality can be bottlenecked by model size, given that you have enough data to feed it;
* Large models benefit from additional supervised data, even over 1B images.
* Larger models are more sample efficient, achieving the same level of error rate with fewer visible images.
* To save memory, they remove class tokens (cls). Instead, they evaluated global average pooling and multi-head attention pooling to aggregate the representations of all patch tokens.
* They use different weight decay for the head and the rest of the layers called ’body’. The authors demonstrate this well in the Figure 6. The box values are few-shot accuracy, while the horizontal and vertical axes indicate the weight decay for the body and the head, respectively. Surprisingly, the stronger decay of the head produces the best results. The authors speculate that a strong weight decay of the head leads to representations with a larger margin between classes.

以下是本文的一些核心贡献和主要结果。
* 如果您有足够的数据提供给模型，那么表示质量可能会受到模型大小的限制; 
* 大型模型可以从额外的受监督数据中受益，甚至超过1B张图像。
* 模型越大，采样效率越高，以更少的可见图像达到相同的错误率水平。
* 为了节省内存，删除了类标记(cls)。相反，他们评估了全局平均池化和多头注意力池化，以汇总所有分块token的表示。
* 他们对头部和其他称为“主干”的层使用不同的重量衰减。作者在图6中很好地证明了这一点。框值几乎不精确，而水平轴和垂直轴分别表示身体和头部的重量衰减。令人惊讶的是，头部更强烈的衰退会产生最佳效果。作者推测，头部强烈的重量衰减会导致类与类之间的差异较大。

Figure 6: Weight decay decoupling effect
图6：重量衰减解耦效应

This is perhaps the most interesting finding that can be more widely applied to pre-training ViT.

这也许是最有趣的发现，可以更广泛地应用于预训练ViT。

They used a warm-up phase at the beginning of training and a cool-down phase at the end of training, where the learning rate linearly anneals to zero. In addition, they used the Adafactor optimizer, which has a memory overhead of 50% compared to traditional Adam.

他们在训练开始时使用了热身阶段，在训练结束时使用了冷却阶段，学习速度线性地降到零。此外，他们使用Adafactor优化器，与传统Adam相比，它的内存开销为50%。

Figure 5: scaling on jft data
图5:jft数据的缩放

Figure 5 depicts the effect of switching from a 300M image dataset (JFT-300M) to 3 billion images (JFT-3B) without any further scaling. Both the medium (B/32) and large (L/16) models benefit from adding data, roughly by a constant factor. Results are obtained by few-shot(linear) evaluation throughout the training process.

图5描述了从300M图像数据集(JFT-300M)切换到30亿图像(JFT-3B)而无需进一步缩放的效果。中型(B/32)和大型(L/16)模型都从添加数据中受益，大致是通过一个常数因子。在整个训练过程中，通过少样本(线性)评估获得结果。

## 4 Replacing self-attention: independent token + channel mixing methods 取代自注意力：独立令牌+通道数混合方法
It is well known that self-attention can be used as an information routing mechanism with fast weights. So far, 3 papers tell the same story: replacing self-attention with 2 information mixing layers; one for mixing token (projected patch vector) and one for mixing channel/feature information.

众所周知，自注意力可以作为一种具有快速权重的信息路由机制。到目前为止，有3篇论文讲述了相同的故事：用两个信息混合层取代自注意力; 一个用于混合token(投影分块向量)，另一个用于混用通道/特征信息。

### 4.1 MLP-Mixer
The MLP-Mixer contains two MLP layers: the first applied independently to the image patches (i.e., ” mixing” the features at each location) and the other across the patches (i.e., ” mixing” the spatial information). MLP Mixer architecture is shown in Figure 7.

MLP-Mixer包含两个MLP层：第一层独立应用于图像分块(即，“混合”每个位置的特征)，另一层跨分块(即“混合”空间信息)。MLP-Mixer架构如图7所示。

Figure 7: MLP-Mixer architecture

### 4.2 XCiT: Cross-Covariance Image Transformers
The other is the recent architecture XCiT, which aims to modify the core building block of ViT: self-attention applied to the token dimension.XCiT architecture is shown in Figure 8.

另一个是最近的架构XCiT，它旨在修改ViT的核心构建块：应用于token维度的自注意力。XCiT架构如图8所示。

Figure 8: XCiT architecture

XCA: For information mixing, the authors propose a cross-covariance attention (XCA) function that operates on the feature dimension of a token rather than on its own. Importantly, this method is only applicable to the L2-normalized set of queries, keys, and values. the L2 norm is denoted by the hat above the letters K and Q. The result of multiplication is also normalized to [-1,1] before softmax.

XCA：对于信息混合，作者提出了一个 交叉协方差注意力(XCA)函数，该函数在token的特征维度上操作，而不是单独操作。重要的是，此方法仅适用于二级规范化查询、键和值集。L2范数由字母K和Q上方的帽子表示。乘法结果也在softmax之前标准化为[-1,1]。

Local Patch Interaction: To achieve explicit communication between the patches, the researchers added two depth-wise 3×3 convolutional layers with Batch Normalization and GELU nonlinearity in between, as shown in Figure 9. Depth-wise convolution was applied to each channel (here the patch) independently.

局部分块交互：为了实现分块之间的显式通信，研究人员添加了两个深度方向的3×3卷积层，中间带有批归一化(BN)和GELU非线性，如图9所示。深度方向的卷积独立应用于每个通道(此处为分块)。

Figure 9: depthwise convolutions 深度卷积

### 4.3 ConvMixer
Figure 10: ConvMixer architecture

Self-attention and MLP are theoretically more general modeling mechanisms, as they allow for larger receptive fields and content-aware behaviour. Nevertheless, the inductive bias of convolution has undeniable results in computer vision tasks.

自注意力和MLP理论上是更通用的建模机制，因为它们允许更大的接受域和内容感知行为。然而，卷积的归纳偏差在计算机视觉任务中具有不可否认的结果。

Motivated by this, researchers have proposed another variant based on convolutional networks called ConvMixer, as shown in Figure 10. the main idea is that it operates directly on the patches as input, separating the mixing of spatial and channel dimensions and maintaining the same size and resolution throughout the network.

基于此，研究人员提出了另一种基于卷积网络的变体，称为ConvMixer，如图10所示。其主要思想是，它直接作用于作为输入的分块，分离空间和通道维度的混合，并在整个网络中保持相同的大小和分辨率。

More specifically, depthwise convolution is responsible for mixing spatial locations, while pointwise convolution (1 x 1 x channel kernel) for mixing channel locations, as shown in the Figure 11.

更具体地说，深度卷积负责混合空间位置，而点卷积(1x1x通道核数)负责混合通道位置，如图11所示。

Figure 11: depthwise convolution with pointwise convolution
图11：深度卷积与点卷积

Mixing of distant spatial locations can be achieved by selecting a larger kernel size to create a larger receptive field.

通过选择更大的内核大小来创建更大的感受野，可以实现远距离空间位置的混合。

## 5 Multiscale Vision Transformers 多尺度视觉Transformers
The CNN backbone architecture benefits from the gradual increase of channels while reducing the spatial dimension of the feature map. Similarly, the Multiscale Vision Transformer (MViT) exploits the idea of combining a multi-scale feature hierarchies with a Vision Transformer model. In practice, the authors start with an initial image size of 3 channels and gradually expand (hierarchically) the channel capacity while reducing the spatial resolution.

CNN主干架构受益于通道数的逐渐增加，同时减少了特征图的空间维度。类似地，多尺度ViT(MViT)利用了将多尺度特征层次与ViT模型相结合的思想。在实践中，作者从3个通道的初始图像大小开始，逐渐扩展(分层)通道容量，同时降低空间分辨率。

Thus, a multi-scale feature pyramid is created, as shown in Figure 12. Intuitively, the early layers will learn high-spatial with simple low-level visual information, while the deeper layers are responsible for complex high-dimensional features.

因此，创建了一个多尺度特征金字塔，如图12所示。直觉上，早期层将通过简单的低层视觉信息学习高空间，而深层则负责复杂的高维特征。

Figure 12: Multi-scale Vit 
图12：多尺度Vit

## 6 Video classification: Timesformer

Figure 13: Block-based and architecture-based / module-based space-time attention architectures for video recognition
图13：视频识别的基于块和基于架构/模块的时空注意架构

After a successful image task, the Vision Transformer is applied to video recognition. Two architectures are presented here,as shown in Figure 13.
* Right: Reducing the architecture level. The proposed method applies a spatial Transformer to the projection image patches and then has another network responsible for capturing time correlations. This is similar to the winning strategy of CNN+LSTM based on video processing.
* Left: Space-time attention that can be implemented at the self-attention level, with the best combination in the red box. Attention is applied sequentially in the time domain by first treating the image frames as tokens. Then, the combined space attention of the two spatial dimensions is applied before the MLP projection. Figure 14 is the t-SNE visualization of the method.

成功完成图像任务后，将ViT应用于视频识别。这里显示了两种架构，如图13所示。
* 右图：降低架构级别。该方法将空间变换器应用于投影图像块，然后有另一个网络负责捕获时间相关性。这类似于基于视频处理的CNN+LSTM的获胜策略。
* 左图：可以在自注意力水平上实现的时空注意力，红色框中有最佳组合。首先将图像帧视为标记，从而在时域中顺序应用注意力。然后，在MLP投影之前应用两个空间维度的组合空间注意。图14是该方法的t-SNE可视化。

In Figure 14, each video is visualized as a point. Videos belonging to the same action category have the same color. A TimeSformer with split space-time attention learns more separable features semantically than a TimeSformer with only space attention or ViT.

在图14中，每个视频都可视化为一个点。属于同一动作类别的视频具有相同的颜色。具有分裂时空注意力的TimeSformer比仅具有空间注意力或ViT的TimeSformer在语义上学习更多可分离的特征。

Figure 14: Feature visualization with t-SNE of Timesformer
图14：使用Timesformer的t-SNE进行特征可视化

## 7 ViT in semantic segmentation: SegFormer 语义切分中的ViT：SegFormer

Figure 15: Segformer architecture
图15:Segformer架构

NVIDIA has proposed a well-configured setup called SegFormer. SegFormer has an interesting design component. First, it consists of a hierarchical Transformer encoder that outputs multi-scale features. Second, it does not require positional encoding, which can deteriorate performance when the test resolution is different from the training.

NVIDIA提出了一种配置良好的设置，称为SegFormer。SegFormer有一个有趣的设计组件。首先，它由输出多尺度特征的分层Transformer编码器组成。其次，它不需要位置编码，当测试分辨率与训练分辨率不同时，位置编码会降低性能。

SegFormer ,as shown in Figure 15, uses a very simple MLP decoder to aggregate the multi-scale features of the encoder. Contrary to ViT, SegFormer uses small image patches, such as 4 x 4, which are known to favor intensive prediction tasks. The proposed Transformer encoder outputs 1/4, 1/8, 1/16, 1/32 multi-scale features at the original image resolution. These multi-level features are provided to the MLP decoder to predict the segmentation mask.

如图15所示，SegFormer使用一个非常简单的MLP解码器来聚合编码器的多尺度特征。与ViT相反，SegFormer使用较小的图像分块，如4 x 4，这是众所周知的，有利于密集预测任务。提出的Transformer编码器以原始图像分辨率输出1/4、1/8、1/16、1/32多尺度特征。这些多级特征被提供给MLP解码器以预测分割掩码。

Mix-FFN in Figure 15: In order to mitigate the impact of positional encoding, the researchers use zero-padding 3 × 3 convolutional layers to leak location information. Mix-FFN can be expressed as follows.

图15中的Mix-FFN：为了减轻位置编码的影响，研究人员使用零填充的3×3卷积层来泄漏位置信息。Mix-FFN可以表示如下。

$x_{out} = MLP(GELU(Conv(MLP(x_{in})))) + x_{in}$

Efficient self-attention is proposed in PVT, which uses a reduction ratio to reduce the length of the sequence. The results can be measured qualitatively by visualizing the effective receptive field (ERF) as shown in Figure 16.

PVT中提出了高效的自我注意，它使用一个约简比来减少序列的长度。结果可以通过可视化有效感受野(ERF)进行定性测量，如图16所示。

Figure 16: SegFormer’s encoder naturally produces local attention, similar to the convolution of lower stages, while being able to output highly non-local attention, effectively capturing the context of Stage4. As shown in the enlarged patch, the ERF in the MLP header (blue box) differs from Stage-4 (red box) in that local attention is significantly stronger in addition to non-local attention.

图16：SegFormer的编码器自然产生局部注意，类似于较低阶段的卷积，同时能够输出高度非局部注意，有效地捕获阶段-4的上下文。如放大的分块所示，MLP头(蓝色框)中的ERF与第4阶段(红色框)的不同之处在于，除了非局部注意力外，局部注意力明显更强。

## 8 Vision Transformers in Medical imaging: Unet + ViT = UNETR 医学成像中的视觉Transformers：Unet+ViT=UNETR
Figure 17: Unetr architecture

Although there are other attempts in medical imaging, UNETR provides the most convincing results. In this approach, ViT is applied to 3D medical image segmentation. It was shown that a simple adaptation is sufficient to improve the baselines for several 3D segmentation tasks.

尽管在医学成像方面还有其他尝试，但UNETR提供了最令人信服的结果。在这种方法中，ViT被应用于三维医学图像分割。结果表明，简单的调整就足以改善几个3D分割任务的基线。

Essentially, UNETR uses the Transformer as an encoder to learn the sequence representation of the input audio, as in Figure 17. Similar to the Unet model, it aims to efficiently capture global multi-scale information that can be passed to the decoder through long skip connections, forming skip connections at different resolutions to compute the final semantic segmentation output.

本质上，UNETR使用Transformer作为编码器来学习输入音频的序列表示，如图17所示。与Unet模型类似，它旨在有效捕获全局多尺度信息，这些信息可以通过长跳转连接传递给解码器，形成不同分辨率的跳转连接，以计算最终的语义分段输出。

## References
1. Wang, W., Xie, E., Li, X., Fan, D. P., Song, K., Liang, D., ... & Shao, L. (2021). Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. arXiv preprint arXiv:2102.12122.
2. Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020). Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768.
3. Wang, W., Xie, E., Li, X., Fan, D. P., Song, K., Liang, D., ... & Shao, L. (2021). Pvtv2: Improved baselines with pyramid vision transformer. arXiv preprint arXiv:2106.13797.
4. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. arXiv preprint arXiv:2103.14030.
5. Zhai, X., Kolesnikov, A., Houlsby, N., & Beyer, L. (2021). Scaling vision transformers. arXiv preprint arXiv:2106.04560.
6. Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X., Unterthiner, T., ... & Dosovitskiy, A. (2021). Mlp-mixer: An all-mlp architecture for vision. arXiv preprint arXiv:2105.01601.
7. El-Nouby, A., Touvron, H., Caron, M., Bojanowski, P., Douze, M., Joulin, A., ... & Jegou, H. (2021). XCiT: Cross-Covariance Image Transformers. arXiv preprint arXiv:2106.09681.
8. Patches Are All You Need? Anonymous ICLR 2021 submission
9. Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., & Feichtenhofer, C. (2021). Multiscale
vision transformers. arXiv preprint arXiv:2104.11227.
10. Bertasius, G., Wang, H., & Torresani, L. (2021). Is Space-Time Attention All You Need for Video
Understanding?. arXiv preprint arXiv:2102.05095.
11. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. arXiv preprint arXiv:2105.15203.
12. Hatamizadeh, A., Yang, D., Roth, H., & Xu, D. (2021). Unetr: Transformers for 3d medical image segmentation. arXiv preprint arXiv:2103.10504.