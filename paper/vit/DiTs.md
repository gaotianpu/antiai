# Scalable Diffusion Models with Transformers
基于Transformers的可扩展扩散模型 2022.12.19 https://arxiv.org/abs/2212.09748

## 阅读笔记
* FID, Fre ́chet Inception Distance
* Gflops

## Abstract
We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward pass complexity as measured by Gflops. We find that DiTs with higher Gflops—through increased transformer depth/width or increased number of input tokens—consistently have lower FID. In addition to possessing good scalability properties, our largest DiT-XL/2 models outperform all prior diffusion models on the classconditional ImageNet 512×512 and 256×256 benchmarks, achieving a state-of-the-art FID of 2.27 on the latter. 

我们探索了基于Transformer架构的一类新的扩散模型。我们训练图像的潜在扩散模型，将常用的U-Net主干替换为在潜在分块上运行的Transformer。我们通过Gflops测量的前向通过复杂度来分析我们的扩散Transformer(DiTs)的可扩展性。我们发现，通过增加transformer深度/宽度或增加输入令牌数量，具有较高Gflops的DiT始终具有较低的FID。除了具有良好的可扩展性外，我们的largest DiT-XL/2模型在类条件ImageNet 512×512和256×256基准上优于所有先前的扩散模型，在后者上实现了2.27的最先进FID。 此处提供代码和项目页面：https://www.wpeebles.com/DiT.html.

Figure 1. Diffusion models with transformer backbones achieve state-of-the-art image quality. We show selected samples from two of our class-conditional DiT-XL/2 models trained on ImageNet at 512×512 and 256×256 resolution, respectively.
图1。带有Transformer主干的扩散模型实现了最先进的图像质量。我们展示了在ImageNet上分别以512×512和256×256分辨率训练的两个类条件DiT XL/2模型中的选定样本。

## 1. Introduction
Machine learning is experiencing a renaissance powered by transformers. Over the past five years, neural architectures for natural language processing [8, 39], vision [10] and several other domains have largely been subsumed by transformers [57]. Many classes of image-level generative models remain holdouts to the trend, though—while transformers see widespread use in autoregressive models [3,6,40,44], they have seen less adoption in other generative modeling frameworks. For example, diffusion models have been at the forefront of recent advances in image-level generative models [9,43]; yet, they all adopt a convolutional U-Net architecture as the de-facto choice of backbone. 

机器学习正在经历由Transformer驱动的复兴。在过去的五年中，自然语言处理[8，39]、视觉[10]和其他几个领域的神经架构在很大程度上被Transformer[57]所涵盖。许多类别的图像级通用模型仍然坚持这一趋势，尽管Transformer在自回归模型中得到广泛应用[3，6，40，44]，但它们在其他通用建模框架中的应用较少。例如，扩散模型一直是图像级生成模型最新进展的前沿[9,43]; 然而，它们都采用卷积U-Net架构作为事实上的主干选择。

Figure 2. ImageNet generation with Diffusion Transformers (DiTs). Bubble area indicates the flops of the diffusion model. Left: FID-50K (lower is better) of our DiT models at 400K training iterations. Performance steadily improves in FID as model flops increase. Right: Our best model, DiT-XL/2, is compute-efficient and outperforms all prior U-Net-based diffusion models, like ADM and LDM.
图2:使用扩散Transformer(DiTs)生成ImageNet。气泡区表示扩散模型的flops。左图：FID-50K(越低越好)我们的DiT模型在400K训练迭代中。随着模型flops的增加，FID的性能稳步提高。右：我们的最佳模型DiT XL/2计算效率高，优于所有现有的基于U-Net的扩散模型，如ADM和LDM。

The seminal work of Ho et al. [19] first introduced the U-Net backbone for diffusion models. The design choice was inherited from PixelCNN++ [49, 55], an autoregressive generative model, with a few architectural changes. The model is convolutional, comprised primarily of ResNet [15] blocks. In contrast to the standard U-Net [46], additional spatial self-attention blocks, which are essential components in transformers, are interspersed at lower resolutions. Dhariwal and Nichol [9] ablated several architecture choices for the U-Net, such as the use of adaptive normalization layers [37] to inject conditional information and channel counts for convolutional layers. However, the highlevel design of the U-Net from Ho et al. has largely remained intact.

Hoet al 的开创性工作[19]首先介绍了用于扩散模型的U-Net主干。设计选择继承自PixelCNN++[49，55]，这是一种自回归生成模型，并进行了一些架构更改。该模型是卷积的，主要由ResNet[15]块组成。与标准U-Net[46]不同的是，Transformer中的重要组成部分 —— 额外的空间自注意块以较低的分辨率分布。Dhariwal和Nichol[9]提出了U-Net的几种架构选择，例如使用自适应非矩阵化层[37]为卷积层注入条件信息和信道计数。然而，Hoet al 的U-Net的高级设计在很大程度上保持了完整。

With this work, we aim to demystify the significance of architectural choices in diffusion models and offer empirical baselines for future generative modeling research. We show that the U-Net inductive bias is not crucial to the performance of diffusion models, and they can be readily replaced with standard designs such as transformers. As a result, diffusion models are well-poised to benefit from the recent trend of architecture unification—e.g., by inheriting best practices and training recipes from other domains, as well as retaining favorable properties like scalability, robustness and efficiency. A standardized architecture would also open up new possibilities for cross-domain research.

通过这项工作，我们的目标是揭开扩散模型中架构选择的意义，并为未来生成建模研究提供经验基础。我们表明，U-Net归纳偏差对扩散模型的性能不是至关重要的，它们可以很容易地用标准设计(如Transformer)进行替换。因此，扩散模型已经做好了充分的准备，可以从架构统一的最新趋势中受益 —— 例如，通过继承其他领域的最佳实践和训练配置，以及保持良好的性能，如可扩展性、健壮性和效率。标准化架构也将为跨领域研究开辟新的可能性。

In this paper, we focus on a new class of diffusion models based on transformers. We call them Diffusion Transformers, or DiTs for short. DiTs adhere to the best practices of Vision Transformers (ViTs) [10], which have been shown to scale more effectively for visual recognition than traditional convolutional networks (e.g., ResNet [15]).

在本文中，我们关注一类新的基于Transformer的扩散模型。我们称之为扩散transformer，简称DiTs。DiTs遵循视觉transformer(ViTs)[10]的最佳实践，已证明其比传统卷积网络(例如，ResNet[15])更有效地扩展视觉识别。

More specifically, we study the scaling behavior of transformers with respect to network complexity vs. sample quality. We show that by constructing and benchmarking the DiT design space under the Latent Diffusion Models (LDMs) [45] framework, where diffusion models are trained within a VAE’s latent space, we can successfully replace the U-Net backbone with a transformer. We further show that DiTs are scalable architectures for diffusion models: there is a strong correlation between the network complexity (measured by Gflops) vs. sample quality (measured by FID). By simply scaling-up DiT and training an LDM with a high-capacity backbone (118.6 Gflops), we are able to achieve a state-of-the-art result of 2.27 FID on the classconditional 256 × 256 ImageNet generation benchmark.

更具体地说，我们研究了transformer在网络复杂度与样本质量之间的缩放行为。我们表明，通过在潜在扩散模型(LDM)[45]框架下构建和基准化DiT设计空间，其中扩散模型在VAE的潜在空间内进行训练，我们可以成功地用Transformer替换U-Net主干。我们进一步表明，DiT是用于扩散模型的可扩展架构：网络复杂性(由Gflops测量)与样本质量(由FID测量)之间存在很强的相关性。通过简单地扩展DiT并训练具有高容量主干(118.6 Gflops)的LDM，我们能够在类条件256×256 ImageNet生成基准上获得2.27 FID的最新结果。

## 2. Related Work
### Transformers. 
Transformers [57] have replaced domainspecific architectures across language, vision [10], reinforcement learning [5, 23] and meta-learning [36]. They have shown remarkable scaling properties under increasing model size, training compute and data in the language domain [24], as generic autoregressive models [17] and as ViTs [60]. Beyond language, transformers have been trained to autoregressively predict pixels [6, 7, 35]. They have also been trained on discrete codebooks [56] as both autoregressive models [11, 44] and masked generative models [4, 14]; the former has shown excellent scaling behavior up to 20B parameters [59]. Finally, transformers have been explored in DDPMs to synthesize non-spatial data; e.g., to generate CLIP image embeddings in DALL·E 2 [38, 43]. In this paper, we study the scaling properties of transformers when used as the backbone of diffusion models of images. 

Transformers。Transformers[57]已经取代了语言、视觉[10]、强化学习[5,23]和元学习[36]等领域特定架构。它们在语言领域[24]、通用自回归模型[17]和ViT[60]中，在模型大小、训练计算和数据不断增加的情况下显示出显著的缩放特性。除了语言之外，transformer还被训练为自回归预测像素[6，7，35]。他们还被训练在离散码本[56]上，作为自回归模型[11，44]和掩码生成模型[4，14]; 前者已显示出高达20B参数的优良缩放行为[59]。最后，在DDPM中探索了Transformer，以合成非空间数据; 例如，在DALL·E-2中生成CLIP图像嵌入[38，43]。在本文中，我们研究了Transformer作为图像扩散模型的主干时的缩放特性。

Figure 3. The Diffusion Transformer (DiT) architecture. Left: We train conditional latent DiT models. The input latent is decomposed into patches and processed by several DiT blocks. Right: Details of our DiT blocks. We experiment with variants of standard transformer blocks that incorporate conditioning via adaptive layer norm, cross-attention and extra input tokens. Adaptive layer norm works best.
图3。扩散Transformer(DiT)架构。左：我们训练条件潜DiT模型。输入潜像被分解成块并由几个DiT块处理。右：DiT区块的详情。我们实验了通过自适应层规范、交叉注意力和额外的输入标记结合调节的标准转换器块的变体。自适应层规范效果最佳。

### Denoising diffusion probabilistic models (DDPMs).
Diffusion [19, 51] and score-based generative models [22, 53] have been particularly successful as generative models of images [32, 43, 45, 47], in many cases outperforming generative adversarial networks (GANs) [12] which had previously been state-of-the-art. Improvements in DDPMs over the past two years have largely been driven by improved sampling techniques [19, 25, 52], most notably classifierfree guidance [21], reformulating diffusion models to predict noise instead of pixels [19] and using cascaded DDPM pipelines where low-resolution base diffusion models are trained in parallel with upsamplers [9, 20]. For all the diffusion models listed above, convolutional U-Nets [46] are the de-facto choice of backbone architecture.

去噪扩散概率模型(DDPM)。 扩散[19，51]和基于分数的生成模型[22，53]作为图像的生成模型[32，43，45，47]特别成功，在许多情况下，其性能优于先前最先进的生成对抗网络(GAN)[12]。过去两年，DDPM的改进主要是由改进的采样技术[19，25，52]推动的，最显著的是无分类器指导[21]，将扩散模型重新表述为预测噪声而不是像素[19]，并使用级联DDPM管道，其中低分辨率基础扩散模型与上采样器并行训练[9，20]。对于上面列出的所有差异模型，卷积U-Nets[46]是骨干架构的实际选择。

### Architecture complexity. 
When evaluating architecture complexity in the image generation literature, it is fairly common practice to use parameter counts. In general, parameter counts can be poor proxies for the complexity of image models since they do not account for, e.g., image resolution which significantly impacts performance [41, 42]. Instead, much of the model complexity analysis in this paper is through the lens of theoretical Gflops. This brings us in-line with the architecture design literature where Gflops are widely-used to gauge complexity. In practice, the golden complexity metric is still up for debate as it frequently depends on particular application scenarios. Nichol and Dhariwal’s seminal work improving diffusion models [9, 33] is most related to us—there, they analyzed the scalability and Gflops properties of the U-Net architecture class. In this paper, we focus on the transformer class.

架构复杂度。 在图像生成文献中评估架构复杂度时，使用参数计数是相当普遍的做法。一般来说，参数计数可能是图像模型复杂性的较差代表，因为它们不考虑(例如)显著影响性能的图像分辨率[41，42]。相反，本文中的许多模型复杂性分析都是通过理论Gflops的视角进行的。这使我们与架构设计文献保持一致，其中Gflops被广泛用于衡量复杂性。在实践中，黄金复杂度度量仍然存在争议，因为它经常取决于特定的应用场景。Nichol和Dhariwal改进扩散模型[9，33]的开创性工作与我们最相关，他们分析了U-Net架构类的可扩展性和Gflops特性。在本文中，我们关注Transformer类。

## 3. Diffusion Transformers
### 3.1. Preliminaries 准备工作
#### Diffusion formulation. 
Before introducing our architecture, we briefly review some basic concepts needed to understand diffusion models (DDPMs) [19, 51]. Gaussian diffusion models assume a forward noising process which gradually applies noise to real data x0: q(xt|x0) = N (xt; √α ̄tx0, (1 − α ̄t)I), where constants α ̄t are hyperparameters. By applying the reparameterization trick, we can sample xt = √α ̄tx0 + √1 − α ̄tεt, where εt ∼ N (0, I).

扩散公式。 在介绍我们的架构之前，我们简要回顾了理解扩散模型(DDPM)所需的一些基本概念[19，51]。高斯扩散模型假设一个正向噪声过程，该过程将噪声逐渐应用于实际数据x0:q(xt | x0)=N(xt; √ᾱtx0，(1−α\772; t)I)，其中常数α\772; t是超参数。通过应用重新参数化技巧，我们可以采样xt=√ᾱtx0+√1−α\772 tεt，其中εt～N(0，I)。

Diffusion models are trained to learn the reverse process that inverts forward process corruptions: pθ(xt−1|xt) = N (μθ (xt ), Σθ (xt )), where neural networks are used to predict the statistics of pθ. The reverse process model is trained with the variational lower bound [27] of the loglikelihood of x0, which reduces to L(θ) = −p(x0|x1) + 􏰀t DKL(q∗(xt−1|xt, x0)||pθ(xt−1|xt)), excluding an additional term irrelevant for training. Since both q∗ and pθ are Gaussian, DKL can be evaluated with the mean and covariance of the two distributions. By reparameterizing μθ as a noise prediction network εθ, the model can be trained using simple mean-squared error between the predicted noise εθ (xt ) and the ground truth sampled Gaussian noise εt : Lsimple(θ) = ||εθ(xt) − εt||2. But, in order to train diffusion models with a learned reverse process covariance Σθ, the full DKL term needs to be optimized. We follow Nichol and Dhariwal’s approach [33]: train εθ with Lsimple, and train Σθ with the full L. Once pθ is trained, new images can be sampled by initializing xtmax ∼ N (0, I) and sampling xt−1 ∼ pθ (xt−1 |xt ) via the reparameterization trick. 

扩散模型被训练来学习反转正向过程腐蚀的反向过程：pθ(xt−1|xt)=N(μθ(xt)，∑θ(xt))，其中神经网络用于预测pθ的统计。反向过程模型使用x0的对数似然的变分下界[27]进行训练，其降为L(θ)=−p(x0|x1)+􏰀t DKL(q*(xt−1|xt，x0)||pθ(xt−1 |xt))，不包括与训练无关的附加术语。由于q*和pθ都是高斯分布，因此可以用两个分布的平均值和协方差来评估DKL。通过将μθ重新参数化为噪声预测网络εθ，可以使用预测噪声εθ(xt)和地面真实采样高斯噪声εt:Lsimple(θ)=||εθ(xt)−εt||2之间的简单均方误差来训练模型。但是，为了训练具有学习的逆过程协方差∑θ的扩散模型，需要优化整个DKL项。我们遵循Nichol和Dhariwal的方法[33]：用Lsimple训练εθ，用完整的L训练∑θ。一旦训练了pθ，就可以通过初始化xtmax～N(0，I)和通过重新参数化技巧对xt−1～pθ(xt−1|xt)进行采样来对新图像进行采样。

#### Classifier-free guidance. 
Conditional diffusion models take extra information as input, such as a class label c. In this case, the reverse process becomes pθ(xt−1|xt,c), where εθ and Σθ are conditioned on c. In this setting, classifier-free guidance can be used to encourage the sampling procedure to find x such that log p(c|x) is high [21]. By Bayes Rule, log p(c|x) ∝ log p(x|c) − log p(x), and hence ∇x log p(c|x) ∝ ∇x log p(x|c) − ∇x log p(x). By interpreting the output of diffusion models as the score function, the DDPM sampling procedure can be guided to sample x with high p(x|c) by: εˆθ(xt, c) = εθ(xt, ∅) + s · ∇x log p(x|c) ∝ εθ(xt, ∅)+s·(εθ(xt, c)−εθ(xt, ∅)), where s > 1 indicates the scale of the guidance (note that s = 1 recovers standard sampling). Evaluating the diffusion model with c = ∅ is done by randomly dropping out c during training and replacing it with a learned “null” embedding ∅. Classifier-free guidance is widely-known to yield significantly improved samples over generic sampling techniques [21, 32, 43], and the trend holds for our DiT models.

无分类器指导。条件扩散模型将额外的信息作为输入，例如类标签c。在这种情况下，反向过程变为pθ(xt−1|xt，c)，其中εθ和∑θ以c为条件。在这种设置下，可以使用无分类器指导来鼓励采样过程找到x，以使log p(c|x)较高[21]。根据贝叶斯规则，log p(c|x)⑪log p(x|c)−log p(x)，因此，可得到♥x log p(c |x)⇔⇔x log p(x|x)−⑪x log(x)。通过将扩散模型的输出解释为得分函数，DDPM采样程序可以通过以下方式引导到具有高p(x|c)的样本x：。使用c=∅评估扩散模型是通过在训练过程中随机丢弃c，并用学习到的“空”嵌入∅替换。众所周知，无分类器指导比一般采样技术产生了显著改善的样本[21，32，43]，我们的DiT模型的趋势也是如此。

#### Latent diffusion models. 
Training diffusion models directly in high-resolution pixel space can be computationally prohibitive. Latent diffusion models (LDMs) [45] tackle this issue with a two-stage approach: (1) learn an autoencoder that compresses images into smaller spatial representations with a learned encoder E; (2) train a diffusion model of representations z = E(x) instead of a diffusion model of images x (E is frozen). New images can then be generated by sampling a representation z from the diffusion model and subsequently decoding it to an image with the learned decoder x = D(z).

潜在扩散模型。 直接在高分辨率像素空间中训练扩散模型在计算上可能是禁止的。潜在扩散模型(LDM)[45]通过两阶段方法解决了这一问题：(1)学习自动编码器，该编码器使用学习的编码器E将图像压缩成更小的空间表示; (2) 训练表示z＝E(x)的扩散模型，而不是图像x的扩散模型(E被冻结)。然后可以通过从扩散模型中采样表示z并随后利用学习解码器x＝D(z)将其解码为图像来生成新图像。

As shown in Figure 2, LDMs achieve good performance while using a fraction of the Gflops of pixel space diffusion models like ADM. Since we are concerned with compute efficiency, this makes them an appealing starting point for architecture exploration. In this paper, we apply DiTs to latent space, although they could be applied to pixel space without modification as well. This makes our image generation pipeline a hybrid-based approach; we use off-the-shelf convolutional VAEs and transformer-based DDPMs.

如图2所示，LDM在使用像ADM这样的像素空间扩散模型的Gflops的一小部分时获得了良好的性能。由于我们关注计算效率，这使它们成为架构探索的一个吸引人的起点。在本文中，我们将DiTs应用于潜在空间，尽管它们也可以应用于像素空间而无需修改。这使得我们的图像生成管道成为一种基于混合的方法; 我们使用现成的卷积VAE和基于Transformer的DDPM。

### 3.2. Diffusion Transformer Design Space 扩散Transformer设计空间
We introduce Diffusion Transformers (DiTs), a new architecture for diffusion models. We aim to be as faithful to the standard transformer architecture as possible to retain its scaling properties. Since our focus is training DDPMs of images (specifically, spatial representations of images), DiT is based on the Vision Transformer (ViT) architecture which operates on sequences of patches [10]. DiT retains many of the best practices of ViTs. Figure 3 shows an overview of the complete DiT architecture. In this section, we describe the forward pass of DiT, as well as the components of the design space of the DiT class.

我们介绍了扩散Transformer(DiTs)，一种用于扩散模型的新架构。我们的目标是尽可能忠实于标准Transformer架构，以保持其缩放特性。由于我们的重点是训练图像的DDPM(特别是图像的空间表示)，因此DiT基于视觉transformer(ViT)架构，该架构对分块序列进行操作[10]。DiT保留了许多ViT的最佳实践。图3显示了完整的DiT架构的概述。在本节中，我们将描述DiT的前向传递，以及DiT类设计空间的组成部分。

Figure 4. Input specifications for DiT. Given patch size p × p, a spatial representation (the noised latent from the VAE) of shape I × I × C is “patchified” into a sequence of length T = (I/p)2 with hidden dimension d. A smaller patch size p results in a longer sequence length and thus more Gflops.

图4。DiT的输入规格。给定分块大小p×p，形状I×I×C的空间表示(来自VAE的噪声潜像)被“分块化”为长度T＝(I/p)2的序列，具有隐藏维度d。较小的分块大小p导致更长的序列长度，从而导致更多的Gflops。

#### Patchify. 
The input to DiT is a spatial representation z (for 256 × 256 × 3 images, z has shape 32 × 32 × 4). The first layer of DiT is “patchify,” which converts the spatial input into a sequence of T tokens, each of dimension d, by linearly embedding each patch in the input. Following patchify, we apply standard ViT frequency-based positional embeddings (the sine-cosine version) to all input tokens. The number of tokens T created by patchify is determined by the patch size hyperparameter p. As shown in Figure 4, halving p will quadruple T , and thus at least quadruple total transformer Gflops. Although it has a significant impact on Gflops, note that changing p has no meaningful impact on downstream parameter counts.

修补。 DiT的输入是空间表示z(对于256×256×3的图像，z的形状为32×32×4)。DiT的第一层是“分块”(patchify)，通过将每个分块线性嵌入到输入中，将空间输入转换为一系列T标记，每个标记的维度为d。在patchify之后，我们将标准的基于ViT频率的位置嵌入(正弦余弦版本)应用于所有输入令牌。patchify创建的令牌T的数量由分块大小超参数p决定。如图4所示，将p减半将使T增加四倍，从而至少增加四倍的总transformerGflops。尽管它对Gflops有重大影响，但请注意，更改p对下游参数计数没有任何意义。

We add p = 2, 4, 8 to the DiT design space.

我们将p=2、4、8添加到DiT设计空间。

#### DiT block design. 
Following patchify, the input tokens are processed by a sequence of transformer blocks. In addition to noised image inputs, diffusion models sometimes process additional conditional information such as noise timesteps t, class labels c, natural language, etc. We explore four variants of transformer blocks that process conditional inputs differently. The designs introduce small, but important, modifications to the standard ViT block design. The designs of all blocks are shown in Figure 3. 

* In-context conditioning. We simply append the vector embeddings of t and c as two additional tokens in the input sequence, treating them no differently from the image tokens. This is similar to cls tokens in ViTs, and it allows us to use standard ViT blocks without modification. After the final block, we remove the conditioning tokens from the sequence. This approach introduces negligible new Gflops to the model. 
* Cross-attentionblock.Weconcatenatetheembeddings of t and c into a length-two sequence, separate from the image token sequence. The transformer block is modified to include an additional multi-head crossattention layer following the multi-head self-attention block, similar to original design from Vaswani et al. [57], and also similar to the one used by LDM for conditioning on class labels. Cross-attention adds the most Gflops to the model, roughly a 15% overhead. 
* Adaptive layer norm (adaLN) block. Following the widespread usage of adaptive normalization layers [37] in GANs [2, 26] and diffusion models with UNet backbones [9], we explore replacing standard layer norm layers in transformer blocks with adaptive layer norm (adaLN). Rather than directly learn dimensionwise scale and shift parameters γ and β, we regress them from the sum of the embedding vectors of t and c. Of the three block designs we explore, adaLN adds the least Gflops and is thus the most compute-efficient. It is also the only conditioning mechanism that is restricted to apply the same function to all tokens. 
* adaLN-Zero block. Prior work on ResNets has found that initializing each residual block as the identity function is beneficial. For example, Goyal et al. found that zero-initializing the final batch norm scale factor γ in each block accelerates large-scale training in the supervised learning setting [13]. Diffusion U-Net models use a similar initialization strategy, zero-initializing the final convolutional layer in each block prior to any residual connections. We explore a modification of the adaLN DiT block which does the same. In addition to regressing γ and β, we also regress dimensionwise scaling parameters α that are applied immediately prior to any residual connections within the DiT block.

DiT区块设计。在patchify之后，输入令牌由一系列转换器块处理。除了有噪声的图像输入，扩散模型有时还处理附加的条件信息，如噪声时间步长t、类标签c、自然语言等。这些设计对标准ViT模块设计进行了小的但重要的修改。所有模块的设计如图3所示。
* 上下文条件。我们简单地将t和c的向量嵌入作为两个额外的标记附加到输入序列中，对它们的处理与图像标记没有区别。这类似于ViT中的cls令牌，它允许我们使用标准的ViT块而无需修改。在最后一个块之后，我们从序列中移除条件标记。这种方法在模型中引入了可以忽略不计的新Gflops。
* 交叉关注块。将t和c的嵌入连接成长度为2的序列，与图像标记序列分开。Transformer块经过修改，在多头自关注块之后增加了一个多头交叉关注层，类似于Vaswaniet al [57]的原始设计，也类似于LDM用于调整类标签的设计。交叉关注为模型增加了最多的Gflops，大约有15%的开销。
* 自适应层规范(adaLN)块。在GAN[2，26]中广泛使用自适应归一化层[37]和具有UNet骨干的扩散模型[9]之后，我们探索用自适应层范数(adaLN)替换Transformer块中的标准层范数层。我们不是直接学习维度尺度和移位参数γ和β，而是从t和c的嵌入向量的和回归它们。在我们探索的三个块设计中，adaLN添加的Gflops最少，因此计算效率最高。它也是唯一被限制将相同功能应用于所有令牌的条件机制。
* adaLN零块。先前对ResNets的研究发现，将每个残差块初始化为恒等函数是有益的。例如，Goyalet al 发现，零初始化每个块中的最终批次规范比例因子γ加速了监督学习环境中的大规模训练[13]。扩散U-Net模型使用类似的初始化策略，在任何残差连接之前，对每个块中的最终卷积层进行零初始化。我们探索了adaLN DiT区块的一种修改，其效果相同。除了回归γ和β之外，我们还回归了在DiT块内任何残差连接之前立即应用的维度缩放参数α。

Figure 5. Comparing different conditioning strategies. adaLNZero outperforms cross-attention and in-context conditioning at all stages of training. 
图5。比较不同的条件调节策略。adaLNZero在训练的所有阶段都优于交叉注意和情境调节。

Table 1. Details of DiT models. We follow ViT [10] model configurations for the Small (S), Base (B) and Large (L) variants; we also introduce an XLarge (XL) config as our largest model.
表1。DiT模型的详情。我们遵循小型(S)、基础(B)和大型(L)变体的ViT[10]模型配置; 我们还引入了XLarge(XL)配置作为我们的最大模型。

We initialize the MLP to output the zero-vector for all α; this initializes the full DiT block as the identity function. As with the vanilla adaLN block, adaLNZero adds negligible Gflops to the model.

我们初始化MLP以输出所有α的零向量; 这将整个DiT块初始化为身份函数。与普通的adaLN块一样，adaLNZero在模型中添加了可忽略的Gflops。

We include the in-context, cross-attention, adaptive layer norm and adaLN-Zero blocks in the DiT design space.
我们在DiT设计空间中包括上下文、交叉关注、自适应层规范和adaLN零块。

#### Model size. 
We apply a sequence of N DiT blocks, each operating at the hidden dimension size d. Following ViT, we use standard transformer configs that jointly scale N, d and attention heads [10, 60]. Specifically, we use four configs: DiT-S, DiT-B, DiT-L and DiT-XL. They cover a wide range of model sizes and flop allocations, from 0.3 to 118.6 Gflops, allowing us to gauge scaling performance. Table 1 gives details of the configs.

模型尺寸。我们应用一系列N个DiT块，每个块以隐藏维度大小d运行。在ViT之后，我们使用标准Transformer配置，共同缩放N、d和注意力头部[10，60]。具体来说，我们使用四种配置：DiT-S、DiT-B、DiT-L和DiT-XL。它们涵盖了范围广泛的模型大小和触发器分配，从0.3到118.6 G触发器，允许我们衡量缩放性能。表1给出了配置的详情。

We add B, S, L and XL configs to the DiT design space.

我们将B、S、L和XL配置添加到DiT设计空间。

#### Transformer decoder. 
After the final DiT block, we need to decode our sequence of image tokens into an output noise prediction and an output diagonal covariance prediction. Both of these outputs have shape equal to the original spatial input. We use a standard linear decoder to do this; we apply the final layer norm (adaptive if using adaLN) and linearly decode each token into a p×p×2C tensor, where C is the number of channels in the spatial input to DiT. Finally, we rearrange the decoded tokens into their original spatial layout to get the predicted noise and covariance.

Transformer解码器。在最后的DiT块之后，我们需要将图像标记序列解码为输出噪声预测和输出对角协方差预测。这两个输出都具有与原始空间输入相等的形状。我们使用标准的线性解码器来实现这一点; 我们应用最终层范数(如果使用adaLN，则为自适应)，并将每个令牌线性解码为p×p×2C张量，其中C是DiT空间输入中的信道数。最后，我们将解码的令牌重新排列到它们的原始空间布局中，以获得预测的噪声和协方差。

The complete DiT design space we explore is patch size, transformer block architecture and model size.

我们探索的完整DiT设计空间是分块大小、Transformer块架构和模型大小。

## 4. Experimental Setup
We explore the DiT design space and study the scaling properties of our model class. Our models are named according to their configs and latent patch sizes p; for example, DiT-XL/2 refers to the XLarge config and p = 2.

我们探索DiT设计空间并研究模型类的缩放特性。我们的模型是根据它们的配置和潜在分块大小p来命名的; 例如，DiT XL/2指的是XLarge配置，p=2。

### Training. 
We train class-conditional latent DiT models at 256 × 256 and 512 × 512 image resolution on the ImageNet dataset [28], a highly-competitive generative modeling benchmark. We initialize the final linear layer with zeros and otherwise use standard weight initialization techniques from ViT. We train all models with AdamW [27,30]. 

训练。我们在ImageNet数据集[28]上以256×256和512×512图像分辨率训练类条件潜DiT模型，这是一个极具竞争力的生成建模基准。我们用零初始化最后的线性层，否则使用ViT的标准权重初始化技术。我们使用AdamW训练所有模型[27，30]。

Figure 6. Scaling the DiT model improves FID at all stages of training. We show FID-50K over training iterations for 12 of our DiT models. Top row: We compare FID holding patch size constant. Bottom row: We compare FID holding model size constant. Scaling the transformer backbone yields better generative models across all model sizes and patch sizes.

图6。缩放DiT模型可以提高训练所有阶段的FID。我们展示了12个DiT模型的FID-50K训练迭代。顶行：我们比较FID保持分块大小不变。最后一行：我们比较FID保持模型大小不变。缩放Transformer主干可以在所有模型大小和分块大小上生成更好的生成模型。

We use a constant learning rate of 1 × 10−4, no weight decay and a batch size of 256. The only data augmentation we use is horizontal flips. Unlike much prior work with ViTs [54, 58], we did not find learning rate warmup nor regularization necessary to train DiTs to high performance. Even without these techniques, training was highly stable across all model configs and we did not observe any loss spikes commonly seen when training transformers. Following common practice in the generative modeling literature, we maintain an exponential moving average (EMA) of DiT weights over training with a decay of 0.9999. All results reported use the EMA model. We use identical training hyperparameters across all DiT model sizes and patch sizes. Our training hyperparameters are almost entirely retained from ADM. We did not tune learning rates, decay/warm-up schedules, Adam β1/β2 or weight decays.

我们使用1×10−4的恒定学习率，无重量衰减，批量大小为256。我们使用的唯一数据增广是水平翻转。与ViTs的许多先前工作不同[54，58]，我们没有发现将DiTs训练为高性能所需的学习率预热或正则化。即使没有这些技术，训练在所有模型配置中都是高度稳定的，我们没有观察到训练Transformer时常见的任何损耗峰值。根据生成建模文献中的常见实践，我们在训练期间保持DiT权重的指数移动平均值(EMA)，衰减为0.9999。所有报告的结果均使用EMA模型。我们在所有DiT模型大小和分块大小上使用相同的训练超参数。我们的训练超参数几乎完全保留在ADM.我们没有调整学习率，衰减/热身计划，亚当β1/β2或体重衰减。

### Diffusion. 
We use an off-the-shelf pre-trained variational autoencoder (VAE) model [27] from Stable Diffusion [45]. The VAE encoder has a downsample factor of 8—given an RGBimagexwithshape256×256×3,z = E(x)has shape 32 × 32 × 4. Across all experiments in this section, our diffusion models operate in this Z-space. After sampling a new latent from our diffusion model, we decode it to pixels using the VAE decoder x = D(z). We retain diffusion hyperparameters from ADM [9]; specifically, we use a tmax =1000linearvarianceschedulerangingfrom1×10−4 to 2 × 10−2, ADM’s parameterization of the covariance Σθ and their method for embedding input timesteps and labels.

扩散。我们使用来自稳定扩散[45]的现成的预训练变分自动编码器(VAE)模型[27]。VAE编码器的下采样因子为8，RGBimagex的形状为256×256×3，z=E(x)的形状为32×32×4。在本节的所有实验中，我们的扩散模型在这个Z空间中运行。在从我们的扩散模型中采样新的潜像之后，我们使用VAE解码器x=D(z)将其解码为像素。我们保留了ADM的扩散超参数[9]; 具体来说，我们使用了从1×10−4到2×10−2的tmax=1000线性变量调度、ADM的协方差∑θ参数化及其嵌入输入时间步长和标签的方法。

### Evaluation metrics. 
We measure scaling performance with Fre ́chet Inception Distance (FID) [18], the standard metric for evaluating generative models of images.

评估指标。 我们使用Fre ́chet Inception Distance(FID)[18]测量缩放性能，这是评估图像生成模型的标准度量。

We follow convention when comparing against prior works and report FID-50K using 250 DDPM sampling steps. FID is known to be sensitive to small implementation details [34]; to ensure accurate comparisons, all values reported in this paper are obtained by exporting samples and using ADM’s TensorFlow evaluation suite [9]. FID numbers reported in this section do not use classifier-free guidance except where otherwise stated. We additionally report Inception Score [48], sFID [31] and Precision/Recall [29] as secondary metrics.

我们在与之前的工作进行比较时遵循惯例，并使用250 DDPM采样步骤报告FID-50K。已知FID对小的实施细节敏感[34]; 为了确保准确的比较，本文中报告的所有值都是通过导出样本并使用ADM的TensorFlow评估套件获得的[9]。除非另有说明，否则本节中报告的FID编号不使用无分类器指南。我们还报告了初始得分[48]、sFID[31]和精度/召回[29]作为次要指标。

### Compute. 
We implement all models in JAX [1] and train them using TPU-v3 pods. DiT-XL/2, our most computeintensive model, trains at roughly 5.7 iterations/second on a TPU v3-256 pod with a global batch size of 256.

计算。我们在JAX[1]中实现了所有模型，并使用TPU-v3 pods训练它们。DiT XL/2是我们最复杂的模型，在TPU v3-256 pod上以大约5.7次/秒的速度训练，全局批量大小为256。

## 5. Experiments 
### DiT block design.
We train four of our highest Gflops DiT-XL/2 models, each using a different block design— in-context (119.4 Gflops), cross-attention (137.6 Gflops), adaptive layer norm (adaLN, 118.6 Gflops) or adaLN-zero (118.6 Gflops). We measure FID over the course of training. Figure 5 shows the results. The adaLN-Zero block yields lower FID than both cross-attention and in-context conditioning while being the most compute-efficient. At 400K training iterations, the FID achieved with the adaLN-Zero model is nearly half that of the in-context model, demonstrating that the conditioning mechanism critically affects model quality. Initialization is also important—adaLNZero, which initializes each DiT block as the identity function, significantly outperforms vanilla adaLN. For the rest of the paper, all models will use adaLN-Zero DiT blocks. 

DiT区块设计。 我们训练了四个最高Gflops DiT XL/2模型，每个模型都使用不同的块设计——上下文(119.4 Gflops)、交叉关注(137.6 Gflops)、自适应层规范(adaLN，118.6 Gflops)或adaLN零(118.6 Glops)。我们在训练过程中测量FID。图5显示了结果。adaLN Zero块产生的FID低于交叉注意和上下文条件，同时是最有效的计算。在400K次训练迭代中，adaLN Zero模型实现的FID几乎是上下文模型的一半，表明调节机制严重影响模型质量。初始化也是重要的adaLNZero，它将每个DiT块初始化为身份函数，显著优于普通adaLN。在本文的其余部分，所有模型都将使用adaLN Zero DiT块。

Figure 7. Increasing transformer forward pass Gflops increases sample quality. Best viewed zoomed-in. We sample from all 12 of our DiT models after 400K training steps using the same input latent noise and class label. Increasing the Gflops in the model—either by increasing transformer depth/width or increasing the number of input tokens—yields significant improvements in visual fidelity. 
图7。增加Transformer正向通Gflops可提高采样质量。最佳视图放大。在400K训练步骤之后，我们使用相同的输入潜在噪声和类别标签从所有12个DiT模型中进行采样。通过增加transformer深度/宽度或增加输入令牌的数量来增加模型中的Gflops，可以显著提高视觉逼真度。

Figure 8. Transformer Gflops are strongly correlated with FID. We plot the Gflops of each of our DiT models and each model’s FID-50K after 400K training steps.
图8。TransformerGflops与FID密切相关。在400K训练步骤后，我们绘制了每个DiT模型的Gflops和每个模型的FID-50K。

### Scaling model size and patch size. 
We train 12 DiT models, sweeping over model configs (S, B, L, XL) and patch sizes (8, 4, 2). Note that DiT-L and DiT-XL are significantly closer to each other in terms of relative Gflops than other configs. Figure 2 (left) gives an overview of the Gflops of each model and their FID at 400K training iterations. In all cases, we find that increasing model size and decreasing patch size yields considerably improved diffusion models.

缩放模型大小和分块大小。我们训练了12个DiT模型，扫描了模型配置(S、B、L、XL)和分块大小(8、4、2)。注意，DiT-L和DiT-XL在相对Gflops方面比其他配置更接近彼此。图2(左)概述了每个模型的Gflops及其在400K训练迭代时的FID。在所有情况下，我们发现增加模型大小和减少分块大小会显著改善扩散模型。

Figure 6 (top) demonstrates how FID changes as model size is increased and patch size is held constant. Across all four configs, significant improvements in FID are obtained over all stages of training by making the transformer deeper and wider. Similarly, Figure 6 (bottom) shows FID as patch size is decreased and model size is held constant. We again observe considerable FID improvements throughout training by simply scaling the number of tokens processed by DiT, holding parameters approximately fixed.
图6(顶部)显示了FID如何随着模型大小的增加而变化，分块大小保持不变。在所有四种配置中，通过使Transformer更深和更宽，在训练的所有阶段，FID都得到了显著的改进。类似地，图6(底部)显示了FID，因为分块大小减小，模型大小保持不变。我们再次观察到，通过简单地缩放DiT处理的令牌的数量，保持参数近似固定，在整个训练过程中，FID显著改善。

### DiT Gflops are critical to improving performance. 
The results of Figure 6 suggest that parameter counts are ultimately not important in determining the quality of a DiT model. As model size is held constant and patch size is decreased, the transformer’s total parameters are effectively unchanged, and only Gflops are increased. These results indicate that scaling model Gflops is actually the key to improved performance. To investigate this further, we plot the FID-50K at 400K training steps against model Gflops in Figure 8. The results demonstrate that DiT models that have different sizes and tokens ultimately obtain similar FID values when their total Gflops are similar (e.g., DiT-S/2 and DiT-B/4). Indeed, we find a strong negative correlation between model Gflops and FID-50K, suggesting that additional model compute is the critical ingredient for improved DiT models. In Figure 12 (appendix), we find that this trend holds for other metrics such as Inception Score.

DiT触发器对提高性能至关重要。 图6的结果表明，在确定DiT模型的质量时，参数计数最终并不重要。当模型大小保持不变且分块大小减小时，Transformer的总参数有效地保持不变，仅Gflops增加。这些结果表明，缩放模型Gflops实际上是提高性能的关键。为了进一步研究这一点，我们在图8中绘制了400K训练步骤下的FID-50K与Gflops模型的对比图。结果表明，具有不同大小和令牌的DiT模型在其总Gflops相似(例如，DiT-S/2和DiT-B/4)时最终获得相似的FID值。事实上，我们发现Gflos模型和FID-50K之间存在强烈的负相关，这表明额外的模型计算是改进DiT模型的关键因素。在图12(附录)中，我们发现这一趋势适用于其他指标，如初始得分。

Figure 9. Larger DiT models use large compute more efficiently. We plot FID as a function of total training compute.
图9。更大的DiT模型更有效地使用大型计算。我们将FID绘制为总训练计算的函数。

### Larger DiT models are more compute-efficient. 
In Figure 9, we plot FID as a function of total training compute for all DiT models. We estimate training compute as model Gflops · batch size · training steps · 3, where the factor of 3 roughly approximates the backwards pass as being twice as compute-heavy as the forward pass. We find that small DiT models, even when trained longer, eventually become compute-inefficient relative to larger DiT models trained for fewer steps. Similarly, we find that models that are identical except for patch size have different performance profiles even when controlling for training Gflops. For example, XL/4 is outperformed by XL/2 after roughly 1010 Gflops.

较大的DiT模型计算效率更高。在图9中，我们将FID绘制为所有DiT模型的总训练计算的函数。我们将训练计算估计为模型Gflops·批量大小·训练步骤·3，其中因子3大致近似于向后传递的计算量是向前传递的两倍。类似地，我们发现，除了分块大小之外，相同的模型即使在控制训练Gflops时也具有不同的性能曲线。例如，在大约1010 Gflops之后，XL/4的表现优于XL/2。

### Visualizing scaling. 
We visualize the effect of scaling on sample quality in Figure 7. At 400K training steps, we sample an image from each of our 12 DiT models using identical starting noise xtmax , sampling noise and class labels. This lets us visually interpret how scaling affects DiT sample quality. Indeed, scaling both model size and the number of tokens yields notable improvements in visual quality.

可视化缩放。我们在图7中可视化了缩放对样本质量的影响。在400K训练步骤中，我们使用相同的起始噪声xtmax、采样噪声和类标签从12个DiT模型中的每一个模型中采样图像。这使我们可以直观地解释缩放如何影响DiT采样质量。事实上，缩放模型大小和令牌数量可以显著提高视觉质量。

### 5.1. State-of-the-Art Diffusion Models
256×256 ImageNet. Following our scaling analysis, we continue training our highest Gflops model, DiT-XL/2, for 7M steps. We show samples from the model in Figures 1, and we compare against state-of-the-art class-conditional generative models. We report results in Table 2. When using classifier-free guidance, DiT-XL/2 outperforms all prior diffusion models, decreasing the previous best FID-50K of 3.60 achieved by LDM to 2.27. Figure 2 (right) shows that DiT-XL/2 (118.6 Gflops) is compute-efficient relative to latent space U-Net models like LDM-4 (103.6 Gflops) and substantially more efficient than pixel space U-Net models such as ADM (1120 Gflops) or ADM-U (742 Gflops). 

256×256 ImageNet。在我们的缩放分析之后，我们继续训练我们的最高Gflops模型DiT XL/2，达到7M步。我们在图1中展示了模型的样本，并与最先进的类条件生成模型进行了比较。我们在表2中报告了结果。当使用无分类器引导时，DiT XL/2优于所有先前的扩散模型，将LDM实现的先前最佳FID-50K(3.60)降至2.27。图2(右)显示，DiT XL/2(118.6 Gflops)相对于潜在空间U-Net模型(如LDM-4(103.6 Gflops))具有计算效率，并且比像素空间U-Net(如ADM(1120 Gflos)或ADM-U(742 Gflops))更高效。

Table 2. Benchmarking class-conditional image generation on ImageNet 256×256. DiT-XL/2 achieves state-of-the-art FID.
表2。ImageNet 256×256上的基准类条件图像生成。DiT XL/2实现了最先进的FID。

Table 3. Benchmarking class-conditional image generation on ImageNet 512×512. Note that prior work [9] measures Precision and Recall using 1000 real samples for 512 × 512 resolution; for consistency, we do the same.
表3。ImageNet 512×512上的基准类条件图像生成。注意，先前的工作[9]使用1000个512×512分辨率的真实样本来测量精度和召回; 为了保持一致，我们也这样做。

Our method achieves the lowest FID of all prior generative models, including the previous state-of-the-art StyleGANXL [50]. Finally, we also observe that DiT-XL/2 achieves higher recall values at all tested classifier-free guidance scales compared to LDM-4 and LDM-8. When trained for only 2.35M steps (similar to ADM), XL/2 still outperforms all prior diffusion models with an FID of 2.55. 

我们的方法实现了所有现有生成模型中最低的FID，包括之前最先进的StyleGANXL[50]。最后，我们还观察到，与LDM-4和LDM-8相比，DiT XL/2在所有测试的无分类器指导量表上都实现了更高的召回值。当仅训练2.35M步(类似于ADM)时，XL/2仍优于所有先前的扩散模型，FID为2.55。

512×512 ImageNet. We train a new DiT-XL/2 model on ImageNet at 512 × 512 resolution for 3M iterations with identical hyperparameters as the 256 × 256 model. With a patch size of 2, this XL/2 model processes a total of 1024 tokens after patchifying the 64 × 64 × 4 input latent (524.6 Gflops). Table 3 shows comparisons against state-of-the-art methods. XL/2 again outperforms all prior diffusion models at this resolution, improving the previous best FID of 3.85 achieved by ADM to 3.04. Even with the increased number of tokens, XL/2 remains compute-efficient. For example, ADM uses 1983 Gflops and ADM-U uses 2813 Gflops; XL/2 uses 524.6 Gflops. We show samples from the highresolution XL/2 model in Figure 1 and the appendix.

512×512 ImageNet。我们在ImageNet上以512×512分辨率训练了一个新的DiT XL/2模型，用于3M次迭代，其超参数与256×256模型相同。在分块大小为2的情况下，该XL/2模型在修补64×64×4输入潜伏(524.6 Gflops)后总共处理1024个令牌。表3显示了与最先进方法的比较。XL/2在该分辨率下再次优于所有先前的扩散模型，将ADM实现的3.85的先前最佳FID提高到3.04。即使令牌数量增加，XL/2仍保持计算效率。例如，ADM使用1983 Gflops，ADM-U使用2813 Gflops; XL/2使用524.6 G触发器。我们在图1和附录中展示了高分辨率XL/2模型的样本。

Figure 10. More sampling compute does not compensate for less model compute. For each of our DiT models trained for 400K iterations, we compute FID-10K using [16, 32, 64, 128, 256, 1000] sampling steps. For each number of steps, we plot the FID as well as the total Gflops used to sample each image. Small models cannot close the performance gap with our large models, even if they sample with more test-time Gflops than the large models.

图10。更多的采样计算不能补偿更少的模型计算。对于针对400K迭代训练的每个DiT模型，我们使用[16，32，64，128，256，1000]采样步骤计算FID-10K。对于每个步骤数，我们绘制FID以及用于对每个图像进行采样的总Gflops。小型模型无法弥补与大型模型之间的性能差距，即使它们比大型模型使用更多的测试时间Gflops进行采样。

### 5.2. Model Compute vs. Sampling Compute 模型计算与采样计算
Unlike most generative models, diffusion models are unique in that they can use additional compute after training by increasing the number of sampling steps when generating an image. Given the importance of model Gflops in sample quality, in this section we study if smaller-model compute DiTs can outperform larger ones by using more sampling compute. We compute FID for all 12 of our DiT models after 400K training steps, using [16, 32, 64, 128, 256, 1000] sampling steps per-image. The main results are in Figure 10. Consider DiT-L/2 using 1000 sampling steps versus DiT-XL/2 using 128 steps. In this case, L/2 uses 80.7 Tflops to sample each image; XL/2 uses 5× less compute— 15.2 Tflops—to sample each image. Nonetheless, XL/2 has the better FID-10K (23.7 vs 25.9). In general, sampling compute cannot compensate for a lack of model compute.

与大多数生成模型不同，扩散模型的独特之处在于，它们可以通过在生成图像时增加采样步骤的数量，在训练后使用额外的计算。考虑到模型Gflops在样本质量中的重要性，在本节中，我们研究较小的模型计算DiTs是否可以通过使用更多的采样计算而优于较大的DiTs。在400K训练步骤之后，我们使用每个图像的[16，32，64，128，256，1000]采样步骤来计算所有12个DiT模型的FID。主要结果如图10所示。考虑使用1000个采样步骤的DiT-L/2与使用128个步骤的DiT XL/2。在这种情况下，L/2使用80.7Tflops对每个图像进行采样; XL/2使用5×更少的计算-15.2Tflops对每个图像进行采样。尽管如此，XL/2具有更好的FID-10K(23.7比25.9)。通常，采样计算不能弥补模型计算的不足。

## 6. Conclusion
We introduce Diffusion Transformers (DiTs), a simple transformer-based backbone for diffusion models that outperforms prior U-Net models and inherits the excellent scaling properties of the transformer model class. Given the promising scaling results in this paper, future work should continue to scale DiTs to larger models and token counts. DiT could also be explored as a drop-in backbone for textto-image models like DALL·E 2 and Stable Diffusion.

我们介绍了扩散transformer(DiTs)，这是一种简单的基于transformer的扩散模型主干，其性能优于先前的U-Net模型，并继承了transformer模型类的优秀缩放特性。鉴于本文中有希望的缩放结果，未来的工作应该继续将DiT缩放到更大的模型和令牌计数。DiT也可以作为DALL·E 2和稳定扩散等文本到图像模型的主干线。

## Acknowledgements. 
We thank Kaiming He, Ronghang Hu, Alexander Berg, Shoubhik Debnath, Tim Brooks, Ilija Radosavovic and Tete Xiao for helpful discussions. William Peebles is supported by the NSF GRFP.

## References
1. James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. 6
2. Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In ICLR, 2019. 5, 9
3. Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020. 1
4. HuiwenChang,HanZhang,LuJiang,CeLiu,andWilliamT Freeman. Maskgit: Masked generative image transformer. In CVPR, pages 11315–11325, 2022. 2
5. Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. In NeurIPS, 2021. 2
6. Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, 2020. 1, 2
7. Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019. 2
8. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HCT, 2019. 1
9. Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 1, 2, 3, 5, 6, 9, 12
10. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2020. 1, 2, 4, 5
11. Patrick Esser, Robin Rombach, and Bjo ̈rn Ommer. Taming transformers for high-resolution image synthesis, 2020. 2
12. Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS, 2014. 3
13. Priya Goyal, Piotr Dolla ́r, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv:1706.02677, 2017. 5
14. Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, and Baining Guo. Vector quantized diffusion model for text-to-image synthesis. In CVPR, pages 10696–10706, 2022. 2
15. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 2
16. Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. 12
17. Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020. 2
18. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. 2017. 6
19. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2, 3
20. Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. arXiv:2106.15282, 2021. 3, 9
21. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 3, 4
22. Aapo Hyva ̈rinen and Peter Dayan. Estimation of nonnormalized statistical models by score matching. Journal of Machine Learning Research, 6(4), 2005. 3
23. Michael Janner, Qiyang Li, and Sergey Levine. Offline reinforcement learning as one big sequence modeling problem. In NeurIPS, 2021. 2
24. Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv:2001.08361, 2020. 2
25. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Proc. NeurIPS, 2022. 3
26. Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019. 5
27. Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015. 3, 5, 6
28. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NeurIPS, 2012. 5
29. Tuomas Kynka ̈a ̈nniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In NeurIPS, 2019. 6
30. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv:1711.05101, 2017. 5
31. CharlieNash,JacobMenick,SanderDieleman,andPeterW Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021. 6
32. Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv:2112.10741, 2021. 3, 4
33. Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021. 3 10 
34. Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022. 6
35. Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, and Dustin Tran. Image transformer. In International conference on machine learning, pages 4055–4064. PMLR, 2018. 2
36. William Peebles, Ilija Radosavovic, Tim Brooks, Alexei Efros, and Jitendra Malik. Learning to learn with generative models of neural network checkpoints. arXiv preprint arXiv:2209.12892, 2022. 2
37. Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In AAAI, 2018. 2, 5
38. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2
39. Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018. 1
40. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. 2019. 1
41. IlijaRadosavovic,JustinJohnson,SainingXie,Wan-YenLo, and Piotr Dolla ́r. On network design spaces for visual recognition. In ICCV, 2019. 3
42. Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dolla ́r. Designing network design spaces. In CVPR, 2020. 3
43. AdityaRamesh,PrafullaDhariwal,AlexNichol,CaseyChu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv:2204.06125, 2022. 1, 2, 3, 4
44. Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 1, 2
45. Robin Rombach, Andreas Blattmann, Dominik Lorenz, PatrickEsser,andBjo ̈rnOmmer.High-resolutionimagesynthesis with latent diffusion models. In CVPR, 2022. 2, 3, 4, 6, 9
46. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 2, 3
47. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. Photorealistic text-toimage diffusion models with deep language understanding. arXiv:2205.11487, 2022. 3
48. Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen, and Xi Chen. Improved techniques for training GANs. In NeurIPS, 2016. 6
49. Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. PixelCNN++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. arXiv preprint arXiv:1701.05517, 2017. 2
50. Axel Sauer, Katja Schwarz, and Andreas Geiger. Styleganxl: Scaling stylegan to large diverse datasets. In SIGGRAPH, 2022. 9
51. Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 3
52. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv:2010.02502, 2020. 3
53. Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. 3
54. AndreasSteiner,AlexanderKolesnikov,XiaohuaZhai,Ross Wightman, Jakob Uszkoreit, and Lucas Beyer. How to train your ViT? data, augmentation, and regularization in vision transformers. TMLR, 2022. 6
55. Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 2
56. Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 2
57. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 1, 2, 5
58. Tete Xiao, Piotr Dollar, Mannat Singh, Eric Mintun, Trevor Darrell, and Ross Girshick. Early convolutions help transformers see better. In NeurIPS, 2021. 6
59. Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv:2206.10789, 2022. 2
60. XiaohuaZhai,AlexanderKolesnikov,NeilHoulsby,andLucas Beyer. Scaling vision transformers. In CVPR, 2022. 2, 5 11  


Figure 11. Additional selected samples from our 512×512 and 256×256 resolution DiT-XL/2 models. We use a classifier-free guidance scale of 6.0 for the 512 × 512 model and 4.0 for the 256 × 256 model. Both models use the ft-EMA VAE decoder. 
图11。从我们的512×512和256×256分辨率DiT XL/2模型中选择的其他样本。对于512×512模型，我们使用6.0的无分类器制导尺度，对于256×256模型，我们采用4.0的无分类器引导尺度。两种模型都使用ft-EMA VAE解码器。

## Appendix
### A. Additional Implementation Details
We include information about all of our DiT models in Table 4, including both 256 × 256 and 512 × 512 models. We include Gflops counts, parameters, training details, FIDs and more. We also include Gflops counts for DDPM U-Net models from ADM and LDM in Table 6.

我们在表4中包含了所有DiT模型的信息，包括256×256和512×512模型。我们包括Gflops计数、参数、训练细节、FID等。我们还将ADM和LDM的DDPM U-Net模型的Gflops计数包括在表6中。

DiT model details. To embed input timesteps, we use a 256-dimensional frequency embedding [9] followed by a two-layer MLP with dimensionality equal to the transformer’s hidden size and SiLU activations. Each adaLN layer feeds the sum of the timestep and class embeddings into a SiLU nonlinearity and a linear layer with output neurons equal to either 4× (adaLN) or 6× (adaLN-Zero) the transformer’s hidden size. We use GELU nonlinearities (approximated with tanh) in the core transformer [16].

DiT模型详情。为了嵌入输入时间步长，我们使用256维频率嵌入[9]，然后是两层MLP，其维度等于Transformer的隐藏大小和SiLU激活。每个adaLN层将时间步长和类嵌入的总和馈送到SiLU非线性和线性层，输出神经元等于Transformer的隐藏大小的4×(adaLN)或6×(adaLNZero)。我们在核心Transformer中使用了GELU非线性(近似于tanh)[16]。

Table 4. Details of all DiT models. We computed without classifier-free guidance. encoder and decoder. For both the 256 × 256 and 512 × 512 DiT-XL/2 models, we never observed FID saturate and continued training them as long as possible. Numbers reported in this table use the ft-MSE VAE decoder. 
表4。所有DiT模型的详情。我们在没有分类器的指导下进行计算。编码器和解码器。对于256×256和512×512 DiT XL/2模型，我们从未观察到FID饱和，并尽可能长时间地继续训练它们。此表中报告的数字使用ft-MSE VAE解码器。

Table 5. Decoder ablation. We tested different pre-trained stabilityai/sd-vae-ft-mse. Different pre-trained decoder weights yield comparable results on ImageNet 256 × 256.
表5。解码器消融。我们测试了不同的预训练稳定性。不同的预训练解码器权重在ImageNet 256×256上产生可比较的结果。

Table 6. Gflops counts for baseline diffusion models that use U-Net backbones.
表6。Gflops用于使用U-Net主干的基线扩散模型。

### B. VAE Decoder Ablations VAE解码器消融
We used off-the-shelf, pre-trained VAEs across our experiments. The VAE models (ft-MSE and ft-EMA) are finetuned versions of the original LDM “f8” model (only the decoder weights are fine-tuned). We monitored metrics for our scaling analysis in Section 5 using the ft-MSE decoder, and we used the ft-EMA decoder for our final metrics reported in Tables 2 and 3. In this section, we ablate three different choices of the VAE decoder; the original one used by LDM and the two fine-tuned decoders used by Stable Diffusion. Because the encoders are identical across models, the decoders can be swapped-in without retraining the diffusion model. Table 5 shows results; XL/2 outperforms all prior diffusion models when using the LDM decoder.

我们在实验中使用了现成的、经过预先训练的VAE。VAE模型(ft-MSE和ft-EMA)是原始LDM“f8”模型的微调版本(只有解码器权重被微调)。我们在第5节中使用ft-MSE解码器监测了缩放分析的指标，并使用ft-EMA解码器监测了表2和3中报告的最终指标。在本节中，我们讨论了VAE解码器的三种不同选择; LDM使用的原始解码器和Stable Diffusion使用的两个微调解码器。由于编码器在不同模型之间是相同的，因此可以在不重新训练扩散模型的情况下交换解码器。表5显示了结果; 当使用LDM解码器时，XL/2优于所有先前的扩散模型。


### C. Model Samples
We show samples from our two DiT-XL/2 models at 512 × 512 and 256 × 256 resolution trained for 3M and 7M steps, respectively. Figures 1 and 11 show selected samples from both models. Figures 13 through 32 show uncurated samples from the two models across a range of classifierfree guidance scales and input class labels (generated with 250 DDPM sampling steps and the ft-EMA VAE decoder). As with prior work using guidance, we observe that larger scales increase visual fidelity and decrease sample diversity.

我们展示了分别针对3M和7M步长训练的512×512和256×256分辨率的两个DiT XL/2模型的样本。图1和图11显示了从两个模型中选择的样本。图13至图32显示了两个模型在一系列无分类指导量表和输入类别标签(使用250 DDPM采样步骤和ft EMA VAE解码器生成)上的未分级样本。与使用引导的先前工作一样，我们观察到更大的尺度增加了视觉保真度并减少了样本多样性。


                                                                                                                                Figure 12. DiT scaling behavior on several generative modeling metrics. Left: We plot model performance as a function of total training compute for FID, sFID, Inception Score, Precision and Recall. Right: We plot model performance at 400K training steps for all 12 DiT variants against transformer Gflops, finding strong correlations across metrics. All values were computed using the ft-MSE VAE decoder.
图12。基于若干生成建模度量的DiT缩放行为。左图：我们将模型性能绘制为FID、sFID、初始得分、精度和召回的总训练计算的函数。右：我们绘制了所有12个DiT变体在400K训练步骤下的模型性能，并与transformer Gflops进行比较，发现了度量之间的强相关性。使用ft-MSE VAE解码器计算所有值。

Figure 13. Uncurated 512 × 512 DiT-XL/2 samples. 
Classifier-free guidance scale = 6.0 
Class label = “arctic wolf” (270)


Figure 14. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 6.0
Class label = “volcano” (980)


Figure 15. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “husky” (250)<br/>
Figure 16. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “sulphur-crested cockatoo” (89)


Figure 17. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “cliff drop-off” (972)


Figure 18. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “balloon” (417)


Figure 19. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “lion” (291)


Figure 20. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “otter” (360)


Figure 21. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 2.0
Class label = “red panda” (387)


Figure 22. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 2.0
Class label = “panda” (388)<br/>
Figure 23. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 1.5
Class label = “coral reef” (973)<br/>
Figure 24. Uncurated 512 × 512 DiT-XL/2 samples. Classifier-free guidance scale = 1.5
Class label = “macaw” (88)<br/>
Figure 25. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “macaw” (88)<br/>
Figure 26. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “dog sled” (537)<br/>
Figure 27. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “arctic fox” (279)<br/>
Figure 28. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 4.0
Class label = “loggerhead sea turtle” (33)<br/>
Figure 29. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 2.0
Class label = “golden retriever” (207)<br/>
Figure 30. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 2.0
Class label = “lake shore” (975)<br/>
Figure 31. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 1.5
Class label = “space shuttle” (812)<br/>
Figure 32. Uncurated 256 × 256 DiT-XL/2 samples. Classifier-free guidance scale = 1.5
Class label = “ice cream” (928)

