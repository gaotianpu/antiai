# High-Resolution Image Synthesis with Latent Diffusion Models
基于潜在扩散模型的高分辨率图像合成 2021.12.20  https://arxiv.org/abs/2112.10752 https://github.com/CompVis/latent-diffusion

## 阅读笔记
* denoising autoencoders ， 去噪自动编码器
* pixel space 像素空间，庞大计算量 / latent space， 潜在空间  
* 引导机制，文本、边框
* 交叉注意力, cross-attention
* GAN主要局限于可变性相对有限的数据，因为对抗性学习过程不容易扩展到建模复杂的多模态分布，模式崩溃和训练不稳定性
* class-conditional image synthesis
* 去噪自动编码器的层次结构构建的扩散模型
* autoregressive (AR) 自回归


## Abstract
By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve new state-of-the-art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including text-to-image synthesis, unconditional image generation and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs. 

通过将图像形成过程分解为去噪自动编码器的顺序应用，扩散模型(DM)在图像数据合成方面实现和超越了最先进的结果。此外，它们的公式允许引导机制来控制图像生成过程而无需再训练。然而，由于这些模型通常直接在像素空间中运行，因此对强大的DM的优化通常需要数百GPU天，并且由于连续评估，推理成本高昂。为了在有限的计算资源上实现DM训练，同时保持其质量和灵活性，我们将其应用于强大的预训练自动编码器的潜在空间。与之前的工作相比，基于这种表示的训练扩散模型首次允许在复杂度降低和细节保留之间达到接近最佳的点，大大提高了视觉逼真度。通过在模型架构中引入交叉注意力层，我们将扩散模型转化为强大而灵活的生成器，用于文本或边界框等通用条件输入，并且以卷积方式实现高分辨率合成。我们的潜在扩散模型(LDM)在图像修复和类条件图像合成方面获得了新的最先进分数，在各种任务(包括文本到图像合成、无条件图像生成和超分辨率)上具有高度竞争力的性能，同时与基于像素的DM相比，显著降低了计算要求。

## 1. Introduction
Image synthesis is one of the computer vision fields with the most spectacular recent development, but also among those with the greatest computational demands. Especially high-resolution synthesis of complex, natural scenes is presently dominated by scaling up likelihood-based models, potentially containing billions of parameters in autoregressive (AR) transformers [66,67]. In contrast, the promising results of GANs [3, 27, 40] have been revealed to be mostly confined to data with comparably limited variability as their adversarial learning procedure does not easily scale to modeling complex, multi-modal distributions. Recently, diffusion models [82], which are built from a hierarchy of denoising autoencoders, have shown to achieve impressive results in image synthesis [30,85] and beyond [7,45,48,57], and define the state-of-the-art in class-conditional image synthesis [15,31] and super-resolution [72]. Moreover, even unconditional DMs can readily be applied to tasks such as inpainting and colorization [85] or stroke-based synthesis [53], in contrast to other types of generative models [19,46,69]. Being likelihood-based models, they do not exhibit mode-collapse and training instabilities as GANs and, by heavily exploiting parameter sharing, they can model highly complex distributions of natural images without involving billions of parameters as in AR models [67].

图像合成是近年来发展最为壮观的计算机视觉领域之一，也是计算需求最大的领域之一。特别是复杂自然场景的高分辨率合成目前主要由基于似然的模型进行放大，可能在自回归(AR)变换器中包含数十亿个参数[66，67]。相比之下，GAN[3，27，40]的有希望的结果已被证明主要局限于可变性相对有限的数据，因为其对抗性学习过程不容易扩展到建模复杂的多模态分布。最近，从去噪自动编码器的层次结构构建的扩散模型[82]显示，在图像合成[30，85]和其他[7，45，48，57]中取得了令人印象深刻的结果，并定义了最先进的类内条件图像合成[15，31]和超分辨率[72]。此外，与其他类型的生成模型[19，46，69]相比，即使是无条件的DM也可以很容易地应用于修复和着色[85]或基于笔划的合成[53]等任务。作为基于似然的模型，它们不像GAN那样表现出模式崩溃和训练不稳定性，并且通过大量利用参数共享，它们可以模拟自然图像的高度复杂分布，而不需要像AR模型中那样涉及数十亿个参数[67]。

Figure 1. Boosting the upper bound on achievable quality with less agressive downsampling. Since diffusion models offer excellent inductive biases for spatial data, we do not need the heavy spatial downsampling of related generative models in latent space, but can still greatly reduce the dimensionality of the data via suitable autoencoding models, see Sec. 3. Images are from the DIV2K [1] validation set, evaluated at 5122 px. We denote the spatial downsampling factor by f. Reconstruction FIDs [29] and PSNR are calculated on ImageNet-val. [12]; see also Tab. 8. 
图1：通过较少的下采样提高可实现质量的上限。由于扩散模型为空间数据提供了极好的归纳偏差，我们不需要对潜在空间中的相关生成模型进行大量的空间下采样，但仍然可以通过适当的自动编码模型大大降低数据的维数，见第3节。图像来自DIV2K[1]验证集，评估值为5122像素。我们用f表示空间下采样因子。重建FID[29]和PSNR是在ImageNet-val上计算的。[12]; 另见表8。

Democratizing High-Resolution Image Synthesis. DMs belong to the class of likelihood-based models, whose mode-covering behavior makes them prone to spend excessive amounts of capacity (and thus compute resources) on modeling imperceptible details of the data [16, 73]. Although the reweighted variational objective [30] aims to address this by undersampling the initial denoising steps, DMs are still computationally demanding, since training and evaluating such a model requires repeated function evaluations (and gradient computations) in the high-dimensional space of RGB images. As an example, training the most powerful DMs often takes hundreds of GPU days (e.g. 150 - 1000 V100 days in [15]) and repeated evaluations on a noisy version of the input space render also inference expensive, so that producing 50k samples takes approximately 5 days [15] on a single A100 GPU. This has two consequences for the research community and users in general: Firstly, training such a model requires massive computational resources only available to a small fraction of the field, and leaves a huge carbon footprint [65, 86]. Secondly, evaluating an already trained model is also expensive in time and memory, since the same model architecture must run sequentially for a large number of steps (e.g. 25 - 1000 steps in [15]).

民主化高分辨率图像合成。 DM属于基于似然性的模型，其模式覆盖行为使其易于在建模数据不可察觉的细节上花费过多的容量(从而计算资源)[16，73]。尽管重新加权的变分目标[30]旨在通过对初始去噪步骤进行欠采样来解决这一问题，但DM仍然需要计算，因为训练和评估这样的模型需要在RGB图像的高维空间中重复进行函数评估(和梯度计算)。例如，训练最强大的DM通常需要数百GPU天(例如，[15]中的150-1000 V100天)，对输入空间的噪声版本进行重复评估也会导致成本高昂，因此在单个A100 GPU上生成50k个样本大约需要5天[15]。这对研究界和一般用户有两个后果：首先，训练这样的模型需要大量的计算资源，只有一小部分领域可用，并留下巨大的碳足迹[65，86]。其次，评估已经训练的模型在时间和内存上也很昂贵，因为相同的模型架构必须连续运行大量步骤(例如[15]中的25-1000个步骤)。

To increase the accessibility of this powerful model class and at the same time reduce its significant resource consumption, a method is needed that reduces the computational complexity for both training and sampling. Reducing the computational demands of DMs without impairing their performance is, therefore, key to enhance their accessibility.

为了增加这个强大模型类的可用性，同时减少其大量资源消耗，需要一种方法来降低训练和采样的计算复杂度。因此，在不影响DM性能的情况下减少DM的计算需求是增强其可用性的关键。

Departure to Latent Space. Our approach starts with the analysis of already trained diffusion models in pixel space: Fig. 2 shows the rate-distortion trade-off of a trained model. As with any likelihood-based model, learning can be roughly divided into two stages: First is a perceptual compression stage which removes high-frequency details but still learns little semantic variation. In the second stage, the actual generative model learns the semantic and conceptual composition of the data (semantic compression). We thus aim to first find a perceptually equivalent, but computationally more suitable space, in which we will train diffusion models for high-resolution image synthesis.

离开潜在空间。 我们的方法从分析像素空间中已经训练的扩散模型开始：图2显示了训练模型的失真率权衡。与任何基于似然的模型一样，学习可以大致分为两个阶段：第一个是感知压缩阶段，它去除了高频细节，但仍然学习到一些语义变化。在第二阶段，实际生成模型学习数据的语义和概念组成(语义压缩)。因此，我们的目标是首先找到一个在感知上等效但在计算上更合适的空间，在其中我们将训练用于高分辨率图像合成的扩散模型。

Figure 2. Illustrating perceptual and semantic compression: Most bits of a digital image correspond to imperceptible details. While DMs allow to suppress this semantically meaningless information by minimizing the responsible loss term, gradients (during training) and the neural network backbone (training and inference) still need to be evaluated on all pixels, leading to superfluous computations and unnecessarily expensive optimization and inference. We propose latent diffusion models (LDMs) as an effective generative model and a separate mild compression stage that only eliminates imperceptible details. Data and images from [30]. 
图2：说明感知和语义压缩：数字图像的大部分对应于难以察觉的细节。尽管DM允许通过最小化负责任损失项来抑制这种语义上无意义的信息，但仍需要在所有像素上评估梯度(在训练期间)和神经网络主干(训练和推理)，这导致了多余的计算和不必要的昂贵优化和推理。我们提出潜在扩散模型(LDM)作为一种有效的生成模型和一个单独的轻度压缩阶段，仅消除不可察觉的细节。数据和图像来自[30]。

Following common practice [11, 23, 66, 67, 96], we separate training into two distinct phases: First, we train an autoencoder which provides a lower-dimensional (and thereby efficient) representational space which is perceptually equivalent to the data space. Importantly, and in contrast to previous work [23,66], we do not need to rely on excessive spatial compression, as we train DMs in the learned latent space, which exhibits better scaling properties with respect to the spatial dimensionality. The reduced complexity also provides efficient image generation from the latent space with a single network pass. We dub the resulting model class Latent Diffusion Models (LDMs).

根据常见实践[11，23，66，67，96]，我们将训练分为两个不同的阶段：首先，我们训练一个自动编码器，该编码器提供了一个较低维度(从而有效)的表示空间，该空间在感知上等同于数据空间。重要的是，与之前的工作[23，66]相比，我们不需要依赖过度的空间压缩，因为我们在学习的潜在空间中训练DM，这在空间维度方面表现出更好的缩放特性。降低的复杂度还提供了通过单个网络通道从潜在空间高效生成图像。我们将生成的模型称为潜在扩散模型(LDM)。

A notable advantage of this approach is that we need to train the universal autoencoding stage only once and can therefore reuse it for multiple DM trainings or to explore possibly completely different tasks [81]. This enables effi- cient exploration of a large number of diffusion models for various image-to-image and text-to-image tasks. For the latter, we design an architecture that connects transformers to the DM’s UNet backbone [71] and enables arbitrary types of token-based conditioning mechanisms, see Sec. 3.3.

这种方法的一个显著优点是，我们只需要训练通用自动编码阶段一次，因此可以将其用于多个DM训练或探索可能完全不同的任务[81]。这使得能够有效地探索各种图像到图像和文本到图像任务的大量扩散模型。对于后者，我们设计了一种架构，将transformer连接到DM的UNet主干[71]，并启用任意类型的基于令牌的调节机制，见第3.3节。

In sum, our work makes the following contributions: 
1. In contrast to purely transformer-based approaches [23, 66], our method scales more graceful to higher dimensional data and can thus (a) work on a compression level which provides more faithful and detailed reconstructions than previous work (see Fig. 1) and (b) can be efficiently applied to high-resolution synthesis of megapixel images. 
2. We achieve competitive performance on multiple tasks (unconditional image synthesis, inpainting, stochastic super-resolution) and datasets while significantly lowering computational costs. Compared to pixel-based diffusion approaches, we also significantly decrease inference costs. 
3. We show that, in contrast to previous work [93] which learns both an encoder/decoder architecture and a score-based prior simultaneously, our approach does not require a delicate weighting of reconstruction and generative abilities. This ensures extremely faithful reconstructions and requires very little regularization of the latent space. 
4. We find that for densely conditioned tasks such as super-resolution, inpainting and semantic synthesis, our model can be applied in a convolutional fashion and render large, consistent images of ∼ 10242 px. 
5. Moreover, we design a general-purpose conditioning mechanism based on cross-attention, enabling multi-modal training. We use it to train class-conditional, text-to-image and layout-to-image models. 
6. Finally, we release pretrained latent diffusion and autoencoding models at https://github.com/CompVis/latent-diffusion which might be reusable for a various tasks besides training of DMs [81].

总之，我们的工作做出了以下贡献：
1. 与纯粹基于transformer的方法相比[23，66]，我们的方法对更高维度的数据进行了更优雅的缩放，因此可以(a)在压缩级别上工作，该压缩级别提供比先前工作更可靠和详细的重建(见图1)，并且(b)可以有效地应用于百万像素图像的高分辨率合成。
2. 我们在多个任务(无条件图像合成、修复、随机超分辨率)和数据集上实现了具有竞争力的性能，同时显著降低了计算成本。与基于像素的扩散方法相比，我们还显著降低了推理成本。
3. 我们表明，与同时学习编码器/解码器架构和基于分数的先验的先前工作[93]相比，我们的方法不需要对重构和生成能力进行精确加权。这确保了极其可靠的重建，并且几乎不需要潜在空间的正则化。
4. 我们发现，对于条件密集的任务，如超分辨率、修复和语义合成，我们的模型可以以卷积的方式应用，并呈现出大的、一致的图像∼ 10242像素。
5. 此外，我们设计了基于交叉注意力的通用条件调节机制，实现了多模式训练。我们使用它来训练类条件、文本到图像以及布局到图像模型。
6. 最后，我们将预训练的latent diffusion and autoencoding模型发布在 https://github.com/CompVis/latent-diffusion , 除了DM[81]的训练之外，它还可用于各种任务。

## 2. Related Work
Generative Models for Image Synthesis. The high dimensional nature of images presents distinct challenges to generative modeling. Generative Adversarial Networks (GAN) [27] allow for efficient sampling of high resolution images with good perceptual quality [3, 42], but are diffi- 2 cult to optimize [2, 28, 54] and struggle to capture the full data distribution [55]. In contrast, likelihood-based methods emphasize good density estimation which renders optimization more well-behaved. Variational autoencoders (VAE) [46] and flow-based models [18, 19] enable efficient synthesis of high resolution images [9, 44, 92], but sample quality is not on par with GANs. While autoregressive models (ARM) [6, 10, 94, 95] achieve strong performance in density estimation, computationally demanding architectures [97] and a sequential sampling process limit them to low resolution images. Because pixel based representations of images contain barely perceptible, high-frequency details [16,73], maximum-likelihood training spends a disproportionate amount of capacity on modeling them, resulting in long training times. To scale to higher resolutions, several two-stage approaches [23,67,101,103] use ARMs to model a compressed latent image space instead of raw pixels.

图像合成的生成模型。图像的高维特性对生成建模提出了明显的挑战。生成对抗网络(GAN)[27]允许对具有良好感知质量的高分辨率图像进行有效采样[3，42]，但难以优化[2，28，54]，难以捕获完整的数据分布[55]。相比之下，基于似然性的方法强调良好的密度估计，这使得优化更加良好。可变自动编码器(VAE)[46]和基于流的模型[18，19]能够高效合成高分辨率图像[9，44，92]，但样本质量与GAN不符。虽然自回归模型(ARM)[6，10，94，95]在密度估计中实现了强大的性能，但计算要求较高的架构[97]和顺序采样过程将其限制在低分辨率图像上。由于基于像素的图像表示包含几乎不可感知的高频细节[16，73]，最大似然训练在建模上花费了不成比例的容量，导致训练时间过长。为了缩放到更高的分辨率，几种两阶段方法[23，67101103]使用ARMs来建模压缩的潜像空间，而不是原始像素。

Recently, Diffusion Probabilistic Models (DM) [82], have achieved state-of-the-art results in density estimation [45] as well as in sample quality [15]. The generative power of these models stems from a natural fit to the inductive biases of image-like data when their underlying neural backbone is implemented as a UNet [15, 30, 71, 85]. The best synthesis quality is usually achieved when a reweighted objective [30] is used for training. In this case, the DM corresponds to a lossy compressor and allow to trade image quality for compression capabilities. Evaluating and optimizing these models in pixel space, however, has the downside of low inference speed and very high training costs. While the former can be partially adressed by advanced sampling strategies [47, 75, 84] and hierarchical approaches [31, 93], training on high-resolution image data always requires to calculate expensive gradients. We adress both drawbacks with our proposed LDMs, which work on a compressed latent space of lower dimensionality. This renders training computationally cheaper and speeds up inference with almost no reduction in synthesis quality (see Fig. 1).

最近，扩散概率模型(DM)[82]在密度估计[45]和样本质量[15]方面取得了最先进的结果。这些模型的生成能力源于当其基础神经主干被实现为UNet时，对图像类数据的归纳偏差的自然拟合[15，30，71，85]。当使用重新加权的目标[30]进行训练时，通常可以获得最佳的合成质量。在这种情况下，DM对应于有损压缩器，并允许以图像质量换取压缩能力。然而，在像素空间中评估和优化这些模型具有推理速度低和训练成本高的缺点。虽然前者可以通过高级采样策略[47，75，84]和分层方法[31，93]部分解决，但对高分辨率图像数据的训练总是需要计算昂贵的梯度。我们用我们提出的LDM解决了这两个缺点，LDM在较低维度的压缩潜在空间上工作。这使得训练在计算上更便宜，并且在几乎不降低合成质量的情况下加快了推理(见图1)。

Two-Stage Image Synthesis To mitigate the shortcomings of individual generative approaches, a lot of research [11, 23, 67, 70, 101, 103] has gone into combining the strengths of different methods into more efficient and performant models via a two stage approach. VQ-VAEs [67, 101] use autoregressive models to learn an expressive prior over a discretized latent space. [66] extend this approach to text-to-image generation by learning a joint distributation over discretized image and text representations. More generally, [70] uses conditionally invertible networks to provide a generic transfer between latent spaces of diverse domains. Different from VQ-VAEs, VQGANs [23, 103] employ a first stage with an adversarial and perceptual objective to scale autoregressive transformers to larger images. However, the high compression rates required for feasible ARM training, which introduces billions of trainable parameters [23, 66], limit the overall performance of such approaches and less compression comes at the price of high computational cost [23, 66]. Our work prevents such tradeoffs, as our proposed LDMs scale more gently to higher dimensional latent spaces due to their convolutional backbone. Thus, we are free to choose the level of compression which optimally mediates between learning a powerful first stage, without leaving too much perceptual compression up to the generative diffusion model while guaranteeing high- fidelity reconstructions (see Fig. 1).

两阶段图像合成为了减轻单个生成方法的缺点，许多研究[11，23，67，70，101，103]已经通过两阶段方法将不同方法的优点结合到更有效和更高性能的模型中。VQ VAEs[67101]使用自回归模型来学习离散化潜在空间上的表达先验。[66]通过学习离散化图像和文本表示上的联合分布，将这种方法扩展到文本到图像生成。更一般地，[70]使用条件可逆网络来提供不同域的潜在空间之间的一般传递。与VQ VAE不同，VQGAN[23103]采用具有对抗性和感知目标的第一阶段，以将自回归变换器缩放为更大的图像。然而，可行的ARM训练所需的高压缩率(引入了数十亿个可训练参数[23，66])限制了此类方法的总体性能，而较少的压缩是以高计算成本为代价的[23，66]。我们的工作防止了这种权衡，因为我们提出的LDM由于其卷积骨架而更平缓地扩展到更高维度的潜在空间。因此，我们可以自由选择在学习强大的第一阶段之间进行最佳调解的压缩级别，而不会在保证高保真重建的同时，为生成扩散模型留下太多的感知压缩(见图1)。

While approaches to jointly [93] or separately [80] learn an encoding/decoding model together with a score-based prior exist, the former still require a difficult weighting between reconstruction and generative capabilities [11] and are outperformed by our approach (Sec. 4), and the latter focus on highly structured images such as human faces.

虽然存在联合[93]或单独[80]学习编码/解码模型以及基于分数的先验的方法，但前者仍然需要在重建和生成能力之间进行困难的加权[11]，并且我们的方法(第4节)的表现优于前者，后者侧重于高度结构化的图像，如人脸。

## 3. Method
To lower the computational demands of training diffusion models towards high-resolution image synthesis, we observe that although diffusion models allow to ignore perceptually irrelevant details by undersampling the corresponding loss terms [30], they still require costly function evaluations in pixel space, which causes huge demands in computation time and energy resources.

为了降低训练扩散模型对高分辨率图像合成的计算需求，我们观察到，尽管扩散模型允许通过对相应的损失项进行欠采样来忽略感知上不相关的细节[30]，但它们仍然需要像素空间中昂贵的函数评估，这导致了计算时间和能量资源的巨大需求。

We propose to circumvent this drawback by introducing an explicit separation of the compressive from the generative learning phase (see Fig. 2). To achieve this, we utilize an autoencoding model which learns a space that is perceptually equivalent to the image space, but offers significantly reduced computational complexity.

我们建议通过引入压缩学习阶段与生成学习阶段的显式分离来规避这一缺陷(见图2)。为了实现这一点，我们使用了一个自动编码模型，该模型学习一个在感知上与图像空间等效的空间，但提供了显著降低的计算复杂度。

Such an approach offers several advantages: (i) By leaving the high-dimensional image space, we obtain DMs which are computationally much more efficient because sampling is performed on a low-dimensional space. (ii) We exploit the inductive bias of DMs inherited from their UNet architecture [71], which makes them particularly effective for data with spatial structure and therefore alleviates the need for aggressive, quality-reducing compression levels as required by previous approaches [23, 66]. (iii) Finally, we obtain general-purpose compression models whose latent space can be used to train multiple generative models and which can also be utilized for other downstream applications such as single-image CLIP-guided synthesis [25].

这种方法提供了几个优点：
1. 通过离开高维图像空间，我们获得了计算效率更高的DM，因为采样是在低维空间上执行的。
2. 我们利用了从UNet架构中继承的DM的归纳偏差[71]，这使其对具有空间结构的数据特别有效，因此减轻了对先前方法所要求的激进、降低质量的压缩级别的需求[23，66]。
3. 最后，我们获得了通用压缩模型，其潜在空间可用于训练多个生成模型，也可用于其他下游应用，如单图像CLIP引导合成[25]。

### 3.1. Perceptual Image Compression 感知图像压缩
Our perceptual compression model is based on previous work [23] and consists of an autoencoder trained by combination of a perceptual loss [106] and a patch-based [33] adversarial objective [20, 23, 103]. This ensures that the reconstructions are confined to the image manifold by enforcing local realism and avoids bluriness introduced by relying solely on pixel-space losses such as $L_2$ or $L_1$ objectives.

我们的感知压缩模型基于先前的工作[23]，由通过感知损失[106]和基于分块的[33]对抗目标[20，23，103]的组合训练的自动编码器组成。这确保了通过强制局部真实性将重建限制在图像流形上，并避免了仅依赖于像素空间损失(例如$L_2$或$L_1$目标)而引入的模糊性。

More precisely, given an image x ∈ $R^{H×W×3}$ in RGB space, the encoder E encodes x into a latent representation z = E(x), and the decoder D reconstructs the image from the latent, giving ˜x = D(z) = D(E(x)), where z ∈ $R^{h×w×c}$ . Importantly, the encoder downsamples the image by a factor f = H/h = W/w, and we investigate different downsampling factors f = $2^m$, with m ∈ N.

更准确地说，给定图像x∈ $在RGB空间中的R^{H×W×3}$，编码器E将x编码为潜像表示z＝E(x)，解码器D从潜像重建图像，给出~x＝D(z)＝D(E(x))，其中z∈ $R ^｛h×w×c｝$。重要的是，编码器通过因子f=H/H=W/W对图像进行下采样，我们研究了不同的下采样因子f=$2^m$，其中m∈ N

In order to avoid arbitrarily high-variance latent spaces, we experiment with two different kinds of regularizations. The first variant, KL-reg., imposes a slight KL-penalty towards a standard normal on the learned latent, similar to a VAE [46, 69], whereas VQ-reg. uses a vector quantization layer [96] within the decoder. This model can be interpreted as a VQGAN [23] but with the quantization layer absorbed by the decoder. Because our subsequent DM is designed to work with the two-dimensional structure of our learned latent space z = E(x), we can use relatively mild compression rates and achieve very good reconstructions. This is in contrast to previous works [23, 66], which relied on an arbitrary 1D ordering of the learned space z to model its distribution autoregressively and thereby ignored much of the inherent structure of z. Hence, our compression model preserves details of x better (see Tab. 8). The full objective and training details can be found in the supplement.

为了避免任意高方差的潜在空间，我们尝试了两种不同的正则化。第一种变体KL-reg.对学习到的潜像施加了轻微的KL惩罚，类似于VAE[46，69]，而VQ-reg。使用解码器内的矢量量化层[96]。该模型可以解释为VQGAN[23]，但量化层被解码器吸收。因为我们的后续DM被设计为与我们学习的潜在空间z＝E(x)的二维结构一起工作，所以我们可以使用相对温和的压缩率并实现非常好的重建。这与之前的工作[23，66]形成对比，之前的工作依赖于学习空间z的任意1D排序来对其分布进行自回归建模，从而忽略了z的大部分固有结构。因此，我们的压缩模型更好地保留了x的细节(见表8)。完整的目标和培训细节可以在补充中找到。

### 3.2. Latent Diffusion Models 潜在扩散模型
Diffusion Models [82] are probabilistic models designed to learn a data distribution p(x) by gradually denoising a normally distributed variable, which corresponds to learning the reverse process of a fixed Markov Chain of length T. For image synthesis, the most successful models [15,30,72] rely on a reweighted variant of the variational lower bound on p(x), which mirrors denoising score-matching [85]. These models can be interpreted as an equally weighted sequence of denoising autoencoders $e_θ(xt, t)$; t = 1 . . . T, which are trained to predict a denoised variant of their input $x_t$, where $x_t$ is a noisy version of the input x. The corresponding objective can be simplified to (Sec. B) 

扩散模型[82]是设计用于通过对正态分布变量逐渐去噪来学习数据分布p(x)的概率模型，这对应于学习长度为T的固定马尔可夫链的反向过程，这反映了去噪分数匹配[85]。这些模型可以解释为去噪自动编码器$e_θ(xt，t)$的等权重序列; t=1。T、 其被训练以预测其输入$x_T$的去噪变量，其中$x_T$是输入x的噪声版本。相应的目标可以简化为(第B节)

// $L_{DM} = E_{x,e}∼N(0,1),th k  −  θ(xt, t)k 22i $, (1) 

with t uniformly sampled from {1, . . . , T}.
其中t从{1，…，t}均匀采样。

Generative Modeling of Latent Representations. With our trained perceptual compression models consisting of E and D, we now have access to an efficient, low-dimensional latent space in which high-frequency, imperceptible details are abstracted away. Compared to the high-dimensional pixel space, this space is more suitable for likelihood-based generative models, as they can now (i) focus on the important, semantic bits of the data and (ii) train in a lower dimensional, computationally much more efficient space.

潜在表征的生成建模。通过我们训练的由E和D组成的感知压缩模型，我们现在可以访问高效、低维的潜在空间，在该空间中，高频、不可察觉的细节被抽象掉。与高维像素空间相比，该空间更适合基于似然的生成模型，因为它们现在可以(i)关注数据的重要语义位，并且(ii)在低维、计算效率更高的空间中训练。

![Figure 3](./images/LatentDiffusion/fig_3.png)<br/>
Figure 3. We condition LDMs either via concatenation or by a more general cross-attention mechanism. See Sec. 3.3 
图3.我们通过串联或更一般的交叉关注机制来调节LDM。见第3.3节

Unlike previous work that relied on autoregressive, attention-based transformer models in a highly compressed, discrete latent space [23,66,103], we can take advantage of image-specific inductive biases that our model offers. This includes the ability to build the underlying UNet primarily from 2D convolutional layers, and further focusing the objective on the perceptually most relevant bits using the reweighted bound, which now reads

与以前的工作不同，以前的工作依赖于高度压缩、离散的潜在空间中的自回归、基于注意力的变压器模型[23，66103]，我们可以利用我们的模型提供的图像特定的归纳偏差。这包括主要从2D卷积层构建基础UNet的能力，并进一步使用重新加权的界限将目标聚焦于感知上最相关的比特，现在该界限为

LLDM := EE(x),∼N(0,1),th k  −  θ(zt, t)k 22i . (2)

The neural backbone  θ(◦, t) of our model is realized as a time-conditional UNet [71]. Since the forward process is fixed, zt can be efficiently obtained from E during training, and samples from p(z) can be decoded to image space with a single pass through D.

神经主干θ(◦, t) 我们模型的一部分被实现为时间条件UNet[71]。由于前向过程是固定的，因此可以在训练期间从E中有效地获得zt，并且可以通过D将来自p(z)的样本解码到图像空间。

### 3.3. Conditioning Mechanisms 调节机制
Similar to other types of generative models [56, 83], diffusion models are in principle capable of modeling conditional distributions of the form p(z|y). This can be implemented with a conditional denoising autoencoder  θ(zt, t, y) and paves the way to controlling the synthesis process through inputs y such as text [68], semantic maps [33, 61] or other image-to-image translation tasks [34].

与其他类型的生成模型[56，83]相似，扩散模型原则上能够对p(z|y)形式的条件分布进行建模。这可以通过条件去噪自动编码器θ(zt，t，y)实现，并为通过输入y(如文本[68]、语义图[33，61]或其他图像到图像的翻译任务[34])控制合成过程铺平了道路。

In the context of image synthesis, however, combining the generative power of DMs with other types of conditionings beyond class-labels [15] or blurred variants of the input image [72] is so far an under-explored area of research.

然而，在图像合成的背景下，将DM的生成能力与除了类标签[15]或输入图像的模糊变体[72]之外的其他类型的条件相结合是迄今为止探索不足的研究领域。

We turn DMs into more flexible conditional image generators by augmenting their underlying UNet backbone with the cross-attention mechanism [97], which is effective for learning attention-based models of various input modalities [35,36]. To pre-process y from various modalities (such as language prompts) we introduce a domain specific encoder τθ that projects y to an intermediate representation τθ(y) ∈ RM×dτ , which is then mapped to the intermediate layers of the UNet via a cross-attention layer implementing Attention(Q, K, V ) = softmax  QKT √d  · V , with

我们通过使用交叉注意力机制增强DMs的基础UNet主干，将其转化为更灵活的条件图像生成器[97]，这对于学习各种输入模式的基于注意力的模型是有效的[35，36]。为了从各种模态(如语言提示)预处理y，我们引入了一个特定于域的编码器τθ，该编码器将y投影到中间表示τθ(y)∈ RM×dτ，然后通过实现注意力(Q，K，V)=softmax QKT的交叉注意力层映射到UNet的中间层√d·V，其中

Q = W(i) Q · ϕi(zt), K = W(i) K · τθ(y), V = W(i) V · τθ(y).

Here, ϕi(zt) ∈ RN×di denotes a (flattened) intermediate representation of the UNet implementing  θ and W(i) V ∈ Rd×di , W(i) Q ∈ Rd×dτ & W(i) K ∈ Rd×dτ are learnable projection matrices [36, 97]. See Fig. 3 for a visual depiction.

这里，⑪i(zt)∈ RN×di表示实现θ和W(i)V的UNet的(平坦)中间表示∈ Rd×di，W(i)Q∈ Rd×dτ和W(i)K∈ Rd×dτ是可学习的投影矩阵[36，97]。如图3所示。

Figure 4. Samples from LDMs trained on CelebAHQ [39], FFHQ [41], LSUN-Churches [102], LSUN-Bedrooms [102] and classconditional ImageNet [12], each with a resolution of 256 × 256. Best viewed when zoomed in. For more samples cf . the supplement.
图4.在CelebAHQ[39]、FFHQ[41]、LSUN教堂[102]、LSUN卧室[102]和分类条件ImageNet[12]上训练的LDM的样本，每个样本的分辨率为256×256。放大时最佳。有关更多样本，请参阅附录。


Based on image-conditioning pairs, we then learn the conditional LDM via

基于图像条件对，我们然后通过

LLDM := EE(x),y,∼N(0,1),th k  − θ(zt, t, τθ(y))k 22i , (3) 

where both τθ and  θ are jointly optimized via Eq. 3. This conditioning mechanism is flexible as τθ can be parameterized with domain-specific experts, e.g. (unmasked) transformers [97] when y are text prompts (see Sec. 4.3.1)

其中τθ和θ都通过等式3进行了联合优化。这种调节机制是灵活的，因为τθ可以通过特定领域的专家进行参数化，例如，当y是文本提示时，(未掩码)变压器[97](见第4.3.1节)

## 4. Experiments
LDMs provide means to flexible and computationally tractable diffusion based image synthesis of various image modalities, which we empirically show in the following. Firstly, however, we analyze the gains of our models compared to pixel-based diffusion models in both training and inference. Interestingly, we find that LDMs trained in VQregularized latent spaces sometimes achieve better sample quality, even though the reconstruction capabilities of VQregularized first stage models slightly fall behind those of their continuous counterparts, cf . Tab. 8. A visual comparison between the effects of first stage regularization schemes on LDM training and their generalization abilities to resolutions > 2562 can be found in Appendix D.1. In E.2 we list details on architecture, implementation, training and evaluation for all results presented in this section.

LDM为各种图像模态的基于扩散的图像合成提供了灵活且可计算的方法，我们在下面的实验中对此进行了说明。然而，首先，我们分析了我们的模型在训练和推理方面与基于像素的扩散模型相比的增益。有趣的是，我们发现，在VQ正则化的潜在空间中训练的LDM有时会获得更好的样本质量，尽管VQ正则的第一阶段模型的重建能力稍微落后于连续模型，cf。表8。附录D.1中对第一阶段正则化方案对LDM训练的影响及其对分辨率>2562的泛化能力进行了直观比较。在E.2中，我们列出了本节中所有结果的架构、实施、训练和评估的详情。

### 4.1. On Perceptual Compression Tradeoffs 感知压缩权衡
This section analyzes the behavior of our LDMs with different downsampling factors f ∈ {1, 2, 4, 8, 16, 32} (abbreviated as LDM-f, where LDM-1 corresponds to pixel-based DMs). To obtain a comparable test-field, we fix the computational resources to a single NVIDIA A100 for all experiments in this section and train all models for the same number of steps and with the same number of parameters.

本节分析了LDM在不同下采样因子下的行为∈ {1，2，4，8，16，32}(缩写为LDM-f，其中LDM-1对应于基于像素的DM)。为了获得可比的测试场，我们将本节中所有实验的计算资源固定为单个NVIDIA A100，并使用相同数量的步骤和相同数量的参数训练所有模型。

Tab. 8 shows hyperparameters and reconstruction performance of the first stage models used for the LDMs compared in this section. Fig. 6 shows sample quality as a function of training progress for 2M steps of class-conditional models on the ImageNet [12] dataset. We see that, i) small downsampling factors for LDM-{1,2} result in slow training progress, whereas ii) overly large values of f cause stagnating fidelity after comparably few training steps. Revisiting the analysis above (Fig. 1 and 2) we attribute this to i) leaving most of perceptual compression to the diffusion model and ii) too strong first stage compression resulting in information loss and thus limiting the achievable quality. LDM-{4-16} strike a good balance between efficiency and perceptually faithful results, which manifests in a significant FID [29] gap of 38 between pixel-based diffusion (LDM-1) and LDM-8 after 2M training steps.

表8显示了用于本节比较的LDM的第一阶段模型的超参数和重建性能。图6显示了ImageNet[12]数据集上2M步类条件模型的样本质量与训练进度的函数关系。我们发现，i)LDM-{1,2}的小降采样因子导致训练进度缓慢，而ii)过大的f值在相对较少的训练步骤后导致保真度停滞。回顾上述分析(图1和2)，我们将其归因于i)将大部分感知压缩留给扩散模型，以及ii)第一阶段压缩过强，导致信息丢失，从而限制了可实现的质量。LDM-{4-16}在效率和感知上可靠的结果之间取得了良好的平衡，这表现为在2M个训练步骤之后，基于像素的扩散(LDM-1)和LDM-8之间的显著FID[29]差距为38。

In Fig. 7, we compare models trained on CelebAHQ [39] and ImageNet in terms sampling speed for different numbers of denoising steps with the DDIM sampler [84] and plot it against FID-scores [29]. LDM-{4-8} outperform models with unsuitable ratios of perceptual and conceptual compression. Especially compared to pixel-based LDM-1, they achieve much lower FID scores while simultaneously significantly increasing sample throughput. Complex datasets such as ImageNet require reduced compression rates to avoid reducing quality. In summary, LDM-4 and -8 offer the best conditions for achieving high-quality synthesis results.

在图7中，我们使用DDIM采样器[84]比较了在CelebAHQ[39]和ImageNet上训练的模型，并将其与FID分数进行比较[29]。LDM-{4-8}在感知和概念压缩比率不合适的情况下优于模型。特别是与基于像素的LDM-1相比，它们实现了更低的FID分数，同时显著提高了样本吞吐量。像ImageNet这样的复杂数据集需要降低压缩率以避免降低质量。总之，LDM-4和LDM-8为获得高质量的合成结果提供了最佳条件。

### 4.2. Image Generation with Latent Diffusion 基于潜在扩散的图像生成
We train unconditional models of 2562 images on CelebA-HQ [39], FFHQ [41], LSUN-Churches and -Bedrooms [102] and evaluate the i) sample quality and ii) their coverage of the data manifold using ii) FID [29] and ii) Precision-and-Recall [50]. Tab. 1 summarizes our results. On CelebA-HQ, we report a new state-of-the-art FID of 5.11, outperforming previous likelihood-based models as well as GANs. We also outperform LSGM [93] where a latent diffusion model is trained jointly together with the first stage. In contrast, we train diffusion models in a fixed space and avoid the difficulty of weighing reconstruction quality against learning the prior over the latent space, see Fig. 1-2.

我们在CelebA HQ[39]、FFHQ[41]、LSUN教堂和卧室[102]上训练2562张图像的无条件模型，并使用ii)FID[29]和ii)精度和召回[50]评估i)样本质量和ii)它们对数据歧管的覆盖率。表1总结了我们的结果。在CelebA HQ上，我们报告了5.11的新的最先进FID，超过了以前基于概率的模型以及GAN。我们还优于LSGM[93]，其中潜在扩散模型与第一阶段一起训练。相反，我们在固定空间中训练扩散模型，避免了在潜在空间上权衡重建质量与学习先验的困难，见图1-2。 

Figure 5. Samples for user-defined text prompts from our model for text-to-image synthesis, LDM-8 (KL), which was trained on the LAION [78] database. Samples generated with 200 DDIM steps and η = 1.0. We use unconditional guidance [32] with s = 10.0.

图5.来自我们的文本到图像合成模型LDM-8(KL)的用户定义文本提示样本，该模型在LAION[78]数据库上训练。使用200 DDIM步骤生成的样本，η=1.0。我们使用s=10.0的无条件指导[32]。

Figure 6. Analyzing the training of class-conditional LDMs with different downsampling factors f over 2M train steps on the ImageNet dataset. Pixel-based LDM-1 requires substantially larger train times compared to models with larger downsampling factors (LDM-{4-16}). Too much perceptual compression as in LDM-32 limits the overall sample quality. All models are trained on a single NVIDIA A100 with the same computational budget. Results obtained with 100 DDIM steps [84] and κ = 0.
图6.分析ImageNet数据集上超过2M个训练步骤的具有不同下采样因子的类条件LDM的训练。与具有更大下采样因子(LDM-{4-16})的模型相比，基于像素的LDM-1需要更大的训练时间。LDM-32中过多的感知压缩限制了总体样本质量。所有模型都在一个NVIDIA A100上训练，计算预算相同。使用100个DDIM步骤[84]和κ=0获得的结果。

Figure 7. Comparing LDMs with varying compression on the CelebA-HQ (left) and ImageNet (right) datasets. Different markers indicate {10, 20, 50, 100, 200} sampling steps using DDIM, from right to left along each line. The dashed line shows the FID scores for 200 steps, indicating the strong performance of LDM- {4-8}. FID scores assessed on 5000 samples. All models were trained for 500k (CelebA) / 2M (ImageNet) steps on an A100. 
图7.比较CelebA HQ(左)和ImageNet(右)数据集上不同压缩的LDM。不同的标记指示使用DDIM的{10，20，50，100，200}采样步骤，沿着每条线从右到左。虚线显示了200个步骤的FID分数，表明LDM-{4-8}的强大性能。对5000份样本进行FID评分。所有模型都在A100上训练了500k(CelebA)/2M(ImageNet)步数。

We outperform prior diffusion based approaches on all but the LSUN-Bedrooms dataset, where our score is close to ADM [15], despite utilizing half its parameters and requiring 4-times less train resources (see Appendix E.3.5).

我们在除LSUN卧室数据集之外的所有数据集上都优于先前的基于扩散的方法，尽管使用了一半的参数，所需的训练资源减少了4倍，但我们的得分接近ADM[15](见附录E.3.5)。

Table 1. Evaluation metrics for unconditional image synthesis. CelebA-HQ results reproduced from [43, 63, 100], FFHQ from [42, 43]. †: N-s refers to N sampling steps with the DDIM [84] sampler. ∗ : trained in KL-regularized latent space. Additional results can be found in the supplementary.
表1.无条件图像合成的评估指标。CelebA HQ结果转载自[43，63，100]，FFHQ转载自[42，43]。†：N-s是指DDIM[84]采样器的N个采样步骤。∗ : 在KL规则化的潜在空间中训练。其他结果可在补充资料中找到。

Table 2. Evaluation of text-conditional image synthesis on the 256 × 256-sized MS-COCO [51] dataset: with 250 DDIM [84] steps our model is on par with the most recent diffusion [59] and autoregressive [26] methods despite using significantly less parameters. †/∗ :Numbers from [109]/ [26]
表2.256×256大小MS-COCO[51]数据集上的文本条件图像合成评估：使用250 DDIM[84]步，我们的模型与最新的扩散[59]和自回归[26]方法相当，尽管使用的参数明显较少。†/∗ :[109]/[26]中的数字

Moreover, LDMs consistently improve upon GAN-based methods in Precision and Recall, thus confirming the advantages of their mode-covering likelihood-based training objective over adversarial approaches. In Fig. 4 we also show qualitative results on each dataset. 

此外，LDM在精度和召回方面不断改进基于GAN的方法，从而证实了其模式覆盖基于可能性的训练目标优于对抗性方法的优势。在图4中，我们还显示了每个数据集的定性结果。

Figure 8. Layout-to-image synthesis with an LDM on COCO [4], see Sec. 4.3.1. Quantitative evaluation in the supplement D.3.
图8.在COCO[4]上使用LDM进行图像合成的布局，参见附录D.3中的第4.3.1节定量评估。

### 4.3. Conditional Latent Diffusion 条件潜在扩散
### 4.3.1 Transformer Encoders for LDMs LDM的变压器编码器
By introducing cross-attention based conditioning into LDMs we open them up for various conditioning modalities previously unexplored for diffusion models. For textto-image image modeling, we train a 1.45B parameter KL-regularized LDM conditioned on language prompts on LAION-400M [78]. We employ the BERT-tokenizer [14] and implement τθ as a transformer [97] to infer a latent code which is mapped into the UNet via (multi-head) crossattention (Sec. 3.3). This combination of domain specific experts for learning a language representation and visual synthesis results in a powerful model, which generalizes well to complex, user-defined text prompts, cf . Fig. 8 and 5. For quantitative analysis, we follow prior work and evaluate text-to-image generation on the MS-COCO [51] validation set, where our model improves upon powerful AR [17, 66] and GAN-based [109] methods, cf . Tab. 2. We note that applying classifier-free diffusion guidance [32] greatly boosts sample quality, such that the guided LDM-KL-8-G is on par with the recent state-of-the-art AR [26] and diffusion models [59] for text-to-image synthesis, while substantially reducing parameter count. To further analyze the flexibility of the cross-attention based conditioning mechanism we also train models to synthesize images based on semantic layouts on OpenImages [49], and finetune on COCO [4], see Fig. 8. See Sec. D.3 for the quantitative evaluation and implementation details.

通过在LDM中引入基于交叉注意力的调节，我们为之前未用于扩散模型的各种调节模式打开了大门。对于文本到图像的图像建模，我们训练了一个1.45B参数KL正则化LDM，条件是LAION-400M上的语言提示[78]。我们使用BERT标记器[14]，并将τθ作为变换器[97]来推理通过(多头)交叉关注映射到UNet的潜在代码(第3.3节)。用于学习语言表示和视觉合成的领域特定专家的这种组合产生了一个强大的模型，它很好地概括为复杂的、用户定义的文本提示，参见图8和图5。对于定量分析，我们遵循先前的工作，并评估MS-COCO[51]验证集上的文本到图像生成，其中，我们的模型改进了强大的AR[17，66]和基于GAN的[109]方法，参见表2。我们注意到，应用无分类器扩散指导[32]大大提高了样本质量，使得指导的LDM-KL-8-G与最近最先进的AR[26]和用于文本到图像合成的扩散模型[59]不相上下，同时大大减少了参数计数。为了进一步分析基于交叉注意力的条件调节机制的灵活性，我们还训练模型以基于OpenImages上的语义布局合成图像[49]，并对COCO[4]进行微调，见图8。有关定量评估和实施细节，请参见第D.3节。

Lastly, following prior work [3, 15, 21, 23], we evaluate our best-performing class-conditional ImageNet models with f ∈ {4, 8} from Sec. 4.1 in Tab. 3, Fig. 4 and Sec. D.4. Here we outperform the state of the art diffusion model ADM [15] while significantly reducing computational requirements and parameter count, cf . Tab 18.

最后，根据先前的工作[3，15，21，23]，我们使用f∈ ｛4，8｝摘自表3第4.1节，图4和第D.4节。在这里，我们优于现有技术的扩散模型ADM[15]，同时显著降低了计算要求和参数计数，参见表18。

### 4.3.2 Convolutional Sampling Beyond 2562 2562以上的卷积采样
By concatenating spatially aligned conditioning information to the input of  θ, LDMs can serve as efficient general purpose image-to-image translation models. We use this to train models for semantic synthesis, super-resolution (Sec. 4.4) and inpainting (Sec. 4.5). For semantic synthesis, we use images of landscapes paired with semantic maps [23, 61] and concatenate downsampled versions of the semantic maps with the latent image representation of a f = 4 model (VQ-reg., see Tab. 8). We train on an input resolution of 2562 (crops from 3842 ) but find that our model generalizes to larger resolutions and can generate images up to the megapixel regime when evaluated in a convolutional manner (see Fig. 9). We exploit this behavior to also apply the super-resolution models in Sec. 4.4 and the inpainting models in Sec. 4.5 to generate large images between 5122 and 10242 . For this application, the signal-to-noise ratio (induced by the scale of the latent space) significantly affects the results. In Sec. D.1 we illustrate this when learning an LDM on (i) the latent space as provided by a f = 4 model (KL-reg., see Tab. 8), and (ii) a rescaled version, scaled by the component-wise standard deviation.

通过将空间对齐的调节信息连接到θ的输入，LDM可以作为有效的通用图像到图像转换模型。我们使用它来训练语义合成、超分辨率(第4.4节)和修复(第4.5节)的模型。对于语义合成，我们使用与语义图配对的风景图像[23，61]，并将语义图的下采样版本与f=4模型的潜在图像表示相连接(VQ reg，见表8)。我们训练2562的输入分辨率(从3842裁剪而来)，但发现我们的模型推广到更大的分辨率，当以卷积方式评估时，可以生成高达百万像素的图像(见图9)。我们利用这种行为还应用了第4.4节中的超分辨率模型和第4.5节中的修复模型，以生成5122和10242之间的大图像。对于这种应用，信噪比(由潜在空间的尺度引起)显著影响结果。在第D.1节中，我们在学习LDM时说明了这一点：(i)f=4模型(KL reg，见表8)提供的潜在空间，以及(ii)按组件标准偏差缩放的重新缩放版本。

Table 3. Comparison of a class-conditional ImageNet LDM with recent state-of-the-art methods for class-conditional image generation on ImageNet [12]. A more detailed comparison with additional baselines can be found in D.4, Tab. 10 and F. c.f.g. denotes classifier-free guidance with a scale s as proposed in [32]. 
表3.类条件ImageNet LDM与ImageNet上最新的类条件图像生成方法的比较[12]。与其他基线的更详细比较见D.4，表10，F.c.F.g.表示具有[32]中提出的尺度s的无分类器制导。

The latter, in combination with classifier-free guidance [32], also enables the direct synthesis of > 2562 images for the text-conditional LDM-KL-8-G as in Fig. 13.

后者结合无分类器指导[32]，还可以直接合成文本条件LDM-KL-8-G的>2562张图像，如图13所示。

Figure 9. A LDM trained on 2562 resolution can generalize to larger resolution (here: 512×1024) for spatially conditioned tasks such as semantic synthesis of landscape images. See Sec. 4.3.2.
图9.在2562分辨率上训练的LDM可以推广到更大的分辨率(此处：512×1024)，用于空间条件任务，如景观图像的语义合成。见第4.3.2节。

### 4.4. Super-Resolution with Latent Diffusion 具有潜在扩散的超分辨率
LDMs can be efficiently trained for super-resolution by diretly conditioning on low-resolution images via concatenation (cf . Sec. 3.3). In a first experiment, we follow SR3 7 bicubic LDM-SR SR3

通过级联直接调节低分辨率图像，可以有效地训练LDM的超分辨率(参见第3.3节)。在第一个实验中，我们遵循SR3 7双三次LDM-SR SR3

Figure 10. ImageNet 64→256 super-resolution on ImageNet-Val. LDM-SR has advantages at rendering realistic textures but SR3 can synthesize more coherent fine structures. See appendix for additional samples and cropouts. SR3 results from [72]. 
图10.ImageNet 64→ImageNet-Val256超分辨率。LDM-SR在渲染真实纹理方面具有优势，但SR3可以合成更连贯的精细结构。更多样品和卷嘴参见附录。SR3由[72]得出。

[72] and fix the image degradation to a bicubic interpolation with 4×-downsampling and train on ImageNet following SR3’s data processing pipeline. We use the f = 4 autoencoding model pretrained on OpenImages (VQ-reg., cf . Tab. 8) and concatenate the low-resolution conditioning y and the inputs to the UNet, i.e. τθ is the identity. Our qualitative and quantitative results (see Fig. 10 and Tab. 5) show competitive performance and LDM-SR outperforms SR3 in FID while SR3 has a better IS. A simple image regression model achieves the highest PSNR and SSIM scores; however these metrics do not align well with human perception [106] and favor blurriness over imperfectly aligned high frequency details [72]. Further, we conduct a user study comparing the pixel-baseline with LDM-SR. We follow SR3 [72] where human subjects were shown a low-res image in between two high-res images and asked for preference. The results in Tab. 4 affirm the good performance of LDM-SR. PSNR and SSIM can be pushed by using a post-hoc guiding mechanism [15] and we implement this image-based guider via a perceptual loss, see Sec. D.6.

[72]并将图像退化修复为具有4×。我们使用在OpenImages上预处理的f=4自动编码模型(VQ reg.，cf.Tab.8)，并连接低分辨率条件y和UNet的输入，即τθ是恒等式。我们的定性和定量结果(见图10和表5)显示了竞争性能，LDM-SR在FID中优于SR3，而SR3具有更好的IS。简单的图像回归模型获得最高的PSNR和SSIM分数; 然而，这些度量与人类感知不太一致[106]，并且与不完全对齐的高频细节相比，更倾向于模糊[72]。此外，我们还进行了一项用户研究，将像素基线与LDM-SR进行了比较。我们遵循SR3[72]，在两张高分辨率图像之间向人类对象显示低分辨率图像，并询问偏好。表4中的结果证实了LDM-SR的良好性能。PSNR和SSIM可以通过使用事后引导机制来推动[15]，我们通过感知损失来实现这种基于图像的引导器，见第D.6节。

Table 4. Task 1: Subjects were shown ground truth and generated image and asked for preference. Task 2: Subjects had to decide between two generated images. More details in E.3.6
表4.任务1：向受试者展示真实情况和生成的图像，并询问其偏好。任务2：受试者必须在两个生成的图像之间做出决定。更多详情请参见E.3.6

Since the bicubic degradation process does not generalize well to images which do not follow this pre-processing, we also train a generic model, LDM-BSR, by using more diverse degradation. The results are shown in Sec. D.6.1.

由于双三次退化过程不能很好地推广到不遵循此预处理的图像，我们还通过使用更多样的退化来训练通用模型LDM-BSR。结果见第D.6.1节。

Table 5. ×4 upscaling results on ImageNet-Val. (2562 ); † : FID features computed on validation split, ‡ : FID features computed on train split; ∗ : Assessed on a NVIDIA A100 
表5.×4 ImageNet-Val.的放大结果。(2562 ); † : 验证拆分时计算的FID特征，‡：列车拆分时计算出的FID特征; ∗ : 通过NVIDIA A100评估

Table 6. Assessing inpainting efficiency. † : Deviations from Fig. 7 due to varying GPU settings/batch sizes cf . the supplement.
表6.评估修补效率†：由于GPU设置/批次大小的变化，与图7的偏差见附录。

### 4.5. Inpainting with Latent Diffusion 使用潜在扩散修复
Inpainting is the task of filling masked regions of an image with new content either because parts of the image are are corrupted or to replace existing but undesired content within the image. We evaluate how our general approach for conditional image generation compares to more specialized, state-of-the-art approaches for this task. Our evaluation follows the protocol of LaMa [88], a recent inpainting model that introduces a specialized architecture relying on Fast Fourier Convolutions [8]. The exact training & evaluation protocol on Places [108] is described in Sec. E.2.2.

修复是用新内容填充图像的蒙版区域的任务，因为图像的部分已损坏，或者替换图像中现有但不需要的内容。我们评估了我们用于条件图像生成的一般方法与用于此任务的更专业、最先进的方法的比较。我们的评估遵循LaMa[88]的协议，这是一种最近的修复模型，它引入了一种基于快速傅里叶卷积的专门架构[8]。第E.2.2节描述了场所[108]的准确培训和评估方案。

We first analyze the effect of different design choices for the first stage. In particular, we compare the inpainting ef- ficiency of LDM-1 (i.e. a pixel-based conditional DM) with LDM-4, for both KL and VQ regularizations, as well as VQLDM-4 without any attention in the first stage (see Tab. 8), where the latter reduces GPU memory for decoding at high resolutions. For comparability, we fix the number of parameters for all models. Tab. 6 reports the training and sampling throughput at resolution 2562 and 5122 , the total training time in hours per epoch and the FID score on the validation split after six epochs. Overall, we observe a speed-up of at least 2.7× between pixel- and latent-based diffusion models while improving FID scores by a factor of at least 1.6×.

我们首先分析第一阶段不同设计选择的影响。特别是，我们比较了LDM-1(即基于像素的条件DM)和LDM-4在KL和VQ正则化以及VQLDM-4方面的修复效率，而在第一阶段(见表8)中没有任何注意，后者减少了用于高分辨率解码的GPU内存。为了便于比较，我们确定了所有模型的参数数量。表6报告了第2562和5122号决议的训练和采样吞吐量、每个时期的总训练时间(以小时为单位)以及六个时期后验证分割的FID分数。总体而言，我们观察到基于像素的扩散模型和基于潜在的扩散模型之间的速度至少提高了2.7倍，同时将FID分数提高了1.6倍。

The comparison with other inpainting approaches in Tab. 7 shows that our model with attention improves the overall image quality as measured by FID over that of [88]. LPIPS between the unmasked images and our samples is slightly higher than that of [88]. We attribute this to [88] only producing a single result which tends to recover more of an average image compared to the diverse results produced by our LDM cf . Fig. 21. Additionally in a user study (Tab. 4) human subjects favor our results over those of [88].

表7中与其他修复方法的比较表明，我们的模型与关注相比，提高了FID测量的整体图像质量[88]。无掩模图像和我们的样本之间的LPIPS略高于[88]。我们将此归因于[88]仅产生一个结果，与LDM cf产生的不同结果相比，该结果倾向于恢复更多的平均图像。图21。此外，在一项用户研究(表4)中，人类受试者更喜欢我们的结果，而不是[88]。

Based on these initial results, we also trained a larger diffusion model (big in Tab. 7) in the latent space of the VQregularized first stage without attention. Following [15], the UNet of this diffusion model uses attention layers on three levels of its feature hierarchy, the BigGAN [3] residual block for up- and downsampling and has 387M parameters instead of 215M. After training, we noticed a discrepancy in the quality of samples produced at resolutions 2562 and 5122 , which we hypothesize to be caused by the additional attention modules. However, fine-tuning the model for half an epoch at resolution 5122 allows the model to adjust to the new feature statistics and sets a new state of the art FID on image inpainting (big, w/o attn, w/ ft in Tab. 7, Fig. 11.).

基于这些初始结果，我们还在VQ正则化的第一阶段的潜在空间中训练了一个更大的扩散模型(表7中的大)，而不需要注意。在[15]之后，该扩散模型的UNet在其特征层次的三个层次上使用关注层，BigGAN[3]残差块用于上采样和下采样，并且具有387M个参数，而不是215M个参数。训练后，我们注意到第2562和5122号决议中产生的样本质量存在差异，我们假设这是由额外的注意力模块造成的。然而，以分辨率5122对模型进行半个周期的微调允许模型调整到新的特征统计，并在图像修复上设置新的最先进FID(大，w/o attn，w/ft，表7，图11)。

Figure 11. Qualitative results on object removal with our big, w/ ft inpainting model. For more results, see Fig. 22. 
图11：使用我们的大型w/ft修复模型去除物体的定性结果。有关更多结果，请参见图22。

## 5. Limitations & Societal Impact 限制和社会影响
Limitations While LDMs significantly reduce computational requirements compared to pixel-based approaches, their sequential sampling process is still slower than that of GANs. Moreover, the use of LDMs can be questionable when high precision is required: although the loss of image quality is very small in our f = 4 autoencoding models (see Fig. 1), their reconstruction capability can become a bottleneck for tasks that require fine-grained accuracy in pixel space. We assume that our superresolution models (Sec. 4.4) are already somewhat limited in this respect.

局限性与基于像素的方法相比，LDM显著降低了计算需求，但其顺序采样过程仍比GAN慢。此外，当需要高精度时，LDM的使用可能会受到质疑：尽管在我们的f=4自动编码模型中，图像质量损失非常小(见图1)，但它们的重建能力可能成为需要像素空间中细粒度精度的任务的瓶颈。我们假设我们的超分辨率模型(第4.4节)在这方面已经有些局限。

Societal Impact Generative models for media like imagery are a double-edged sword: On the one hand, they enable various creative applications, and in particular approaches like ours that reduce the cost of training and inference have the potential to facilitate access to this technology and democratize its exploration. On the other hand, it also means that it becomes easier to create and disseminate manipulated data or spread misinformation and spam. In particular, the deliberate manipulation of images (“deep fakes”) is a common problem in this context, and women in particular are disproportionately affected by it [13, 24].

社会影响-媒体图像的生成模型是一把双刃剑：一方面，它们支持各种创造性应用，特别是像我们这样降低培训和推理成本的方法，有可能促进获取这项技术并使其探索民主化。另一方面，这也意味着创建和传播被操纵的数据或传播错误信息和垃圾邮件变得更加容易。特别是，故意操纵图像(“深度假货”)是这方面的一个常见问题，尤其是女性受其影响更大[13，24]。

Table 7. Comparison of inpainting performance on 30k crops of size 512 × 512 from test images of Places [108]. The column 40- 50% reports metrics computed over hard examples where 40-50% of the image region have to be inpainted. † recomputed on our test set, since the original test set used in [88] was not available. 
表7.从地点[108]的测试图像中对比了30k株大小为512×512的作物的修补性能。40-50%列报告了在硬样本中计算的度量，其中40-50%的图像区域必须修复。†由于[88]中使用的原始测试集不可用，因此在我们的测试集上重新计算。

Generative models can also reveal their training data [5, 90], which is of great concern when the data contain sensitive or personal information and were collected without explicit consent. However, the extent to which this also applies to DMs of images is not yet fully understood.

生成模型还可以显示其训练数据[5，90]，当数据包含敏感或个人信息且未经明确同意而收集时，这一点非常令人担忧。然而，这在多大程度上也适用于图像的DM还没有完全理解。

Finally, deep learning modules tend to reproduce or exacerbate biases that are already present in the data [22, 38, 91]. While diffusion models achieve better coverage of the data distribution than e.g. GAN-based approaches, the extent to which our two-stage approach that combines adversarial training and a likelihood-based objective misrepresents the data remains an important research question.

最后，深度学习模块倾向于再现或加剧数据中已经存在的偏差[22，38，91]。尽管扩散模型比基于GAN的方法更好地覆盖了数据分布，但我们结合对抗性训练和基于可能性的目标的两阶段方法在多大程度上歪曲了数据仍然是一个重要的研究问题。

For a more general, detailed discussion of the ethical considerations of deep generative models, see e.g. [13].

关于深度生成模型的伦理考虑的更一般、更详细的讨论，请参见例如[13]。

## 6. Conclusion
We have presented latent diffusion models, a simple and efficient way to significantly improve both the training and sampling efficiency of denoising diffusion models without degrading their quality. Based on this and our crossattention conditioning mechanism, our experiments could demonstrate favorable results compared to state-of-the-art methods across a wide range of conditional image synthesis tasks without task-specific architectures.

我们提出了潜在扩散模型，这是一种简单而有效的方法，可以显著提高去噪扩散模型的训练和采样效率，而不会降低其质量。基于这一点和我们的交叉注意力调节机制，我们的实验可以在没有任务特定架构的情况下，在广泛的条件图像合成任务中显示出与最先进的方法相比的良好结果。

This work has been supported by the German Federal Ministry for Economic Affairs and Energy within the project ’KI-Absicherung - Safe AI for automated driving’ and by the German Research Foundation (DFG) project 421703927. 

这项工作得到了德国联邦经济事务和能源部在“KI Absichrung-自动驾驶安全AI”项目和德国研究基金会(DFG)421703927项目的支持。

## References
1. Eirikur Agustsson and Radu Timofte. NTIRE 2017 challenge on single image super-resolution: Dataset and study. In 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2017, Honolulu, HI, USA, July 21-26, 2017, pages 1122–1131. IEEE Computer Society, 2017. 1
2. Martin Arjovsky, Soumith Chintala, and L´eon Bottou. Wasserstein gan, 2017. 3
3. Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In Int. Conf. Learn. Represent., 2019. 1, 2, 7, 8, 22, 28
4. Holger Caesar, Jasper R. R. Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18- 22, 2018, pages 1209–1218. Computer Vision Foundation / IEEE Computer Society, 2018. 7, 20, 22
5. Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650, 2021. 9
6. Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In ICML, volume 119 of Proceedings of Machine Learning Research, pages 1691–1703. PMLR, 2020. 3
7. Nanxin Chen, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. Wavegrad: Estimating gradients for waveform generation. In ICLR. OpenReview.net, 2021. 1
8. Lu Chi, Borui Jiang, and Yadong Mu. Fast fourier convolution. In NeurIPS, 2020. 8
9. Rewon Child. Very deep vaes generalize autoregressive models and can outperform them on images. CoRR, abs/2011.10650, 2020. 3
10. Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. CoRR, abs/1904.10509, 2019. 3
11. Bin Dai and David P. Wipf. Diagnosing and enhancing VAE models. In ICLR (Poster). OpenReview.net, 2019. 2, 3
12. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. IEEE Computer Society, 2009. 1, 5, 7, 22
13. Emily Denton. Ethical considerations of generative ai. AI for Content Creation Workshop, CVPR, 2021. 9
14. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018. 7
15. Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis. CoRR, abs/2105.05233, 2021. 1, 2, 3, 4, 6, 7, 8, 18, 22, 25, 26, 28
16. Sander Dieleman. Musings on typicality, 2020. 1, 3
17. Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-toimage generation via transformers. CoRR, abs/2105.13290,2021. 6, 7
18. Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation, 2015. 3
19. Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 1, 3
20. Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity metrics based on deep networks. In Daniel D. Lee, Masashi Sugiyama, Ulrike von Luxburg, Isabelle Guyon, and Roman Garnett, editors, Adv. Neural Inform. Process. Syst., pages 658–666, 2016. 3
21. Patrick Esser, Robin Rombach, Andreas Blattmann, and Bj¨orn Ommer. Imagebart: Bidirectional context with multinomial diffusion for autoregressive image synthesis. CoRR, abs/2108.08827, 2021. 6, 7, 22
22. Patrick Esser, Robin Rombach, and Bj¨orn Ommer. A note on data biases in generative models. arXiv preprint arXiv:2012.02516, 2020. 9
23. Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. CoRR, abs/2012.09841, 2020. 2, 3, 4, 6, 7, 21, 22, 29, 34, 36
24. Mary Anne Franks and Ari Ezra Waldman. Sex, lies, and videotape: Deep fakes and free speech delusions. Md. L. Rev., 78:892, 2018. 9
25. Kevin Frans, Lisa B. Soros, and Olaf Witkowski. Clipdraw: Exploring text-to-drawing synthesis through languageimage encoders. ArXiv, abs/2106.14843, 2021. 3
26. Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Make-a-scene: Scenebased text-to-image generation with human priors. CoRR, abs/2203.13131, 2022. 6, 7, 16
27. Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial networks. CoRR, 2014. 1, 2
28. Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans, 2017. 3
29. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Adv. Neural Inform. Process. Syst., pages 6626– 6637, 2017. 1, 5, 26
30. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 1, 2, 3, 4, 6, 17
31. Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. CoRR, abs/2106.15282, 2021. 1, 3, 22 10
32. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 6, 7, 16, 22, 28, 37, 38
33. Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. In CVPR, pages 5967–5976. IEEE Computer Society, 2017. 3, 4
34. Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5967–5976,2017. 4
35. Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier J. H´enaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, and Jo˜ao Carreira. Perceiver IO: A general architecture for structured inputs &outputs. CoRR, abs/2107.14795, 2021. 4
36. Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Jo˜ao Carreira. Perceiver: General perception with iterative attention. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 4651–4664. PMLR, 2021. 4, 5
37. Manuel Jahn, Robin Rombach, and Bj¨orn Ommer. Highresolution complex scene synthesis with transformers. CoRR, abs/2105.06458, 2021. 20, 22, 27
38. Niharika Jain, Alberto Olmo, Sailik Sengupta, Lydia Manikonda, and Subbarao Kambhampati. Imperfect imaganation: Implications of gans exacerbating biases on facial data augmentation and snapchat selfie lenses. arXiv preprint arXiv:2001.09528, 2020. 9
39. Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. CoRR, abs/1710.10196, 2017. 5, 6
40. Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4401– 4410, 2019. 1
41. T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. 5, 6
42. Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. CoRR, abs/1912.04958,2019. 2, 6, 28
43. Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon. Score matching model for unbounded data score. CoRR, abs/2106.05527, 2021. 6
44. Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems, 2018. 3
45. Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. CoRR, abs/2107.00630, 2021. 1, 3, 16
46. Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations, ICLR, 2014. 1, 3, 4, 29
47. Zhifeng Kong and Wei Ping. On fast sampling of diffusion probabilistic models. CoRR, abs/2106.00132, 2021. 3
48. Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. In ICLR. OpenReview.net, 2021. 1
49. Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper R. R. Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Tom Duerig, and Vittorio Ferrari. The open images dataset V4: unified image classi- fication, object detection, and visual relationship detection at scale. CoRR, abs/1811.00982, 2018. 7, 20, 22
50. Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. CoRR, abs/1904.06991, 2019. 5, 26
51. Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. CoRR, abs/1405.0312, 2014. 6, 7, 27
52. Yuqing Ma, Xianglong Liu, Shihao Bai, Le-Yi Wang, Aishan Liu, Dacheng Tao, and Edwin Hancock. Region-wise generative adversarial imageinpainting for large missing areas. ArXiv, abs/1909.12507, 2019. 9
53. Chenlin Meng, Yang Song, Jiaming Song, Jiajun Wu, JunYan Zhu, and Stefano Ermon. Sdedit: Image synthesis and editing with stochastic differential equations. CoRR, abs/2108.01073, 2021. 1
54. Lars M. Mescheder. On the convergence properties of GAN training. CoRR, abs/1801.04406, 2018. 3
55. Luke Metz, Ben Poole, David Pfau, and Jascha SohlDickstein. Unrolled generative adversarial networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. 3
56. Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. CoRR, abs/1411.1784, 2014. 4
57. Gautam Mittal, Jesse H. Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. CoRR, abs/2103.16091, 2021. 1
58. Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Z. Qureshi, and Mehran Ebrahimi. Edgeconnect: Generative image inpainting with adversarial edge learning. ArXiv, abs/1901.00212, 2019. 9
59. Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. CoRR, abs/2112.10741, 2021. 6, 7, 16
60. Anton Obukhov, Maximilian Seitzer, Po-Wei Wu, Semen Zhydenko, Jonathan Kyl, and Elvis Yu-Jing Lin. 11 High-fidelity performance metrics for generative models in pytorch, 2020. Version: 0.3.0, DOI: 10.5281/zenodo.4957738. 26, 27
61. Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and JunYan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019. 4, 7
62. Taesung Park, Ming-Yu Liu, Ting-Chun Wang, and JunYan Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019. 22
63. Gaurav Parmar, Dacheng Li, Kwonjoon Lee, and Zhuowen Tu. Dual contradistinctive generative autoencoder. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 823–832. Computer Vision Foundation / IEEE, 2021. 6
64. Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On buggy resizing libraries and surprising subtleties in fid calculation. arXiv preprint arXiv:2104.11222, 2021. 26
65. David A. Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild, David R. So, Maud Texier, and Jeff Dean. Carbon emissions and large neural network training. CoRR, abs/2104.10350,2021. 2
66. Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. CoRR, abs/2102.12092, 2021. 1, 2, 3, 4, 7, 21, 27
67. Ali Razavi, A¨aron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In NeurIPS, pages 14837–14847, 2019. 1, 2, 3, 22
68. Scott E. Reed, Zeynep Akata, Xinchen Yan, Lajanugen Logeswaran, Bernt Schiele, and Honglak Lee. Generative adversarial text to image synthesis. In ICML, 2016. 4
69. Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of the 31st International Conference on International Conference on Machine Learning, ICML, 2014. 1, 4, 29
70. Robin Rombach, Patrick Esser, and Bj¨orn Ommer. Network-to-network translation with conditional invertible neural networks. In NeurIPS, 2020. 3
71. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In MICCAI (3), volume 9351 of Lecture Notes in Computer Science, pages 234–241. Springer, 2015. 2, 3, 4
72. Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. CoRR, abs/2104.07636, 2021. 1, 4, 8, 16, 22, 23, 27
73. Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma. Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications. CoRR, abs/1701.05517, 2017. 1, 3
74. Dave Salvator. NVIDIA Developer Blog. https : / / developer . nvidia . com / blog / getting - immediate- speedups- with- a100- tf32, 2020. 28
75. Robin San-Roman, Eliya Nachmani, and Lior Wolf. Noise estimation for generative diffusion models. CoRR, abs/2104.02600, 2021. 3
76. Axel Sauer, Kashyap Chitta, Jens M¨uller, and Andreas Geiger. Projected gans converge faster. CoRR, abs/2111.01007, 2021. 6
77. Edgar Sch¨onfeld, Bernt Schiele, and Anna Khoreva. A unet based discriminator for generative adversarial networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 8204–8213. Computer Vision Foundation / IEEE, 2020. 6
78. Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion- 400m: Open dataset of clip-filtered 400 million image-text pairs, 2021. 6, 7
79. Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In Yoshua Bengio and Yann LeCun, editors, Int. Conf. Learn. Represent., 2015. 29, 43, 44, 45
80. Abhishek Sinha, Jiaming Song, Chenlin Meng, and Stefano Ermon. D2C: diffusion-denoising models for few-shot conditional generation. CoRR, abs/2106.06819, 2021. 3
81. Charlie Snell. Alien Dreams: An Emerging Art Scene. https : / / ml . berkeley . edu / blog / posts / clip-art/, 2021. Online; accessed November-2021.. 2
82. Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. CoRR, abs/1503.03585, 2015. 1, 3, 4, 18
83. Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. 4
84. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR. OpenReview.net,2021. 3, 5, 6, 22
85. Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. CoRR, abs/2011.13456, 2020. 1, 3, 4, 18
86. Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for modern deep learning research. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 13693–13696. AAAI Press, 2020. 2 
87. Wei Sun and Tianfu Wu. Learning layout and style reconfigurable gans for controllable image synthesis. CoRR, abs/2003.11571, 2020. 22, 27
88. Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor S. Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. ArXiv, abs/2109.07161, 2021. 8, 9, 26, 32
89. Tristan Sylvain, Pengchuan Zhang, Yoshua Bengio, R. Devon Hjelm, and Shikhar Sharma. Object-centric image generation from layouts. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 2647–2655. AAAI Press, 2021. 20, 22, 27
90. Patrick Tinsley, Adam Czajka, and Patrick Flynn. This face does not exist... but it might be yours! identity leakage in generative models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1320–1328, 2021. 9
91. Antonio Torralba and Alexei A Efros. Unbiased look at dataset bias. In CVPR 2011, pages 1521–1528. IEEE, 2011. 9
92. Arash Vahdat and Jan Kautz. NVAE: A deep hierarchical variational autoencoder. In NeurIPS, 2020. 3
93. Arash Vahdat, Karsten Kreis, and Jan Kautz. Scorebased generative modeling in latent space. CoRR, abs/2106.05931, 2021. 2, 3, 5, 6
94. Aaron van den Oord, Nal Kalchbrenner, Lasse Espeholt, koray kavukcuoglu, Oriol Vinyals, and Alex Graves. Conditional image generation with pixelcnn decoders. In Advances in Neural Information Processing Systems, 2016. 3
95. A¨aron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. CoRR, abs/1601.06759, 2016. 3
96. A¨aron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NIPS, pages 6306–6315, 2017. 2, 4, 29
97. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pages 5998–6008, 2017. 3, 4, 5, 7
98. Rivers Have Wings. Tweet on Classifier-free guidance for autoregressive models. https : / / twitter . com / RiversHaveWings / status / 1478093658716966912, 2022. 6
99. Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´emi Louf, Morgan Funtowicz, and Jamie Brew. Huggingface’s transformers: State-of-the-art natural language processing. CoRR, abs/1910.03771, 2019. 26
100. Zhisheng Xiao, Karsten Kreis, Jan Kautz, and Arash Vahdat. VAEBM: A symbiosis between variational autoencoders and energy-based models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. 6
101. Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using VQ-VAE and transformers. CoRR, abs/2104.10157, 2021. 3
102. Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. LSUN: construction of a large-scale image dataset using deep learning with humans in the loop. CoRR, abs/1506.03365, 2015. 5
103. Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan, 2021. 3, 4
104. Jiahui Yu, Zhe L. Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang. Free-form image inpainting with gated convolution. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4470–4479, 2019. 9
105. K. Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. ArXiv, abs/2103.14006, 2021. 23
106. Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018. 3, 8, 19
107. Shengyu Zhao, Jianwei Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I-Chao Chang, and Yan Xu. Large scale image completion via co-modulated generative adversarial networks. ArXiv, abs/2103.10428, 2021. 9
108. Bolei Zhou, `Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40:1452–1464, 2018. 8, 9, 26
109. Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, and Tong Sun. LAFITE: towards language-free training for text-to-image generation. CoRR, abs/2111.13792, 2021. 6, 7, 16 13 

 
## Appendix
Figure 12. Convolutional samples from the semantic landscapes model as in Sec. 4.3.2, finetuned on 5122 images.
图12.第4.3.2节中语义景观模型的卷积样本，对5122幅图像进行了微调。

Figure 13. Combining classifier free diffusion guidance with the convolutional sampling strategy from Sec. 4.3.2, our 1.45B parameter text-to-image model can be used for rendering images larger than the native 2562 resolution the model was trained on. 
图13。将无分类器扩散引导与第4.3.2节中的卷积采样策略相结合，我们的1.45B参数文本到图像模型可用于渲染比模型所训练的原始2562分辨率更大的图像。

### A. Changelog
Here we list changes between this version (https://arxiv.org/abs/2112.10752v2) of the paper and the previous version, i.e. https://arxiv.org/abs/2112.10752v1. 
* We updated the results on text-to-image synthesis in Sec. 4.3 which were obtained by training a new, larger model (1.45B parameters). This also includes a new comparison to very recent competing methods on this task that were published on arXiv at the same time as ( [59, 109]) or after ( [26]) the publication of our work.
* We updated results on class-conditional synthesis on ImageNet in Sec. 4.1, Tab. 3 (see also Sec. D.4) obtained by retraining the model with a larger batch size. The corresponding qualitative results in Fig. 26 and Fig. 27 were also updated. Both the updated text-to-image and the class-conditional model now use classifier-free guidance [32] as a measure to increase visual fidelity.
* We conducted a user study (following the scheme suggested by Saharia et al [72]) which provides additional evaluation for our inpainting (Sec. 4.5) and superresolution models (Sec. 4.4).
* Added Fig. 5 to the main paper, moved Fig. 18 to the appendix, added Fig. 13 to the appendix.

这里我们列出了此版本之间的更改(https://arxiv.org/abs/2112.10752v2)论文和上一版本，即。https://arxiv.org/abs/2112.10752v1.
* 我们更新了第4.3节中文本到图像合成的结果，这些结果是通过训练一个新的更大的模型(1.45B参数)获得的。这还包括与最近在arXiv上发表的关于这项任务的竞争方法的新比较，这些方法与我们的工作发表的同时([59109])或之后([26])发表在arXiv上。
* 我们更新了ImageNet第4.1节表3(另见第D.4节)中的类条件合成结果，该结果是通过用更大的批量重新训练模型获得的。图26和图27中相应的定性结果也进行了更新。更新的文本到图像和类条件模型现在都使用无分类器指导[32]作为提高视觉逼真度的措施。
* 我们进行了一项用户研究(遵循Sahariaet al [72]提出的方案)，为我们的修复(第4.5节)和超分辨率模型(第4.4节)提供了额外的评估。
* 将图5添加到主文件中，将图18移动到附录中，并将图13添加到附录中。

### B. Detailed Information on Denoising Diffusion Models 去噪扩散模型的详情
Diffusion models can be specified in terms of a signal-to-noise ratio SNR(t) = α2t σt2 consisting of sequences (αt)Tt=1 and (σt)Tt=1 which, starting from a data sample x0, define a forward diffusion process q as q(xt|x0) = N (xt|αtx0, σt2I) (4) with the Markov structure for s < t: 

扩散模型可以根据信噪比SNR(t)=α2tσt2来规定，该信噪比由序列(αt)Tt=1和(σt)Tt=1组成，从数据样本x0开始，将前向扩散过程q定义为q(xt | x0)=N(xt |αtx0，σt2I)(4)，具有s＜t:

q(xt|xs) = N (xt|αt|sxs, σt2|sI) (5) αt|s = αt αs (6) σt2|s = σt2 − αt2|sσs2 (7)

Denoising diffusion models are generative models p(x0) which revert this process with a similar Markov structure running backward in time, i.e. they are specified as 

p(x0) = Z z p(xT ) TtY=1 p(xt−1|xt) (8)

The evidence lower bound (ELBO) associated with this model then decomposes over the discrete time steps as 

− log p(x0) ≤ KL(q(xT |x0)|p(xT )) + TXt=1 Eq(xt|x0)KL(q(xt−1|xt, x0)|p(xt−1|xt)) (9)

The prior p(xT ) is typically choosen as a standard normal distribution and the first term of the ELBO then depends only on the final signal-to-noise ratio SNR(T). To minimize the remaining terms, a common choice to parameterize p(xt−1|xt) is to specify it in terms of the true posterior q(xt−1|xt, x0) but with the unknown x0 replaced by an estimate xθ(xt, t) based on the current step xt. This gives [45] p(xt−1|xt) := q(xt−1|xt, xθ(xt, t)) (10) = N (xt−1|µθ(xt, t), σt2|t−1 σ2t−1 σ2t I), (11) where the mean can be expressed as µθ(xt, t) = αt|t−1σt2−1 σ2t xt + αt−1σ2t|t−1 σ2t xθ(xt, t). (12) 16

In this case, the sum of the ELBO simplify to

TXt=1

Eq(xt|x0)KL(q(xt−1|xt, x0)|p(xt−1) =

TXt=1

EN( |0,I) 12(SNR(t − 1) − SNR(t))k x0 − xθ(αtx0 + σt, t)k 2 (13)

Following [30], we use the reparameterization  θ(xt, t) = (xt − αtxθ(xt, t))/σt (14) to express the reconstruction term as a denoising objective, k x0 − xθ(αtx0 + σt, t)k 2= σt2 α2t k  −  θ(αtx0 + σt, t)k 2 (15) and the reweighting, which assigns each of the terms the same weight and results in Eq. (1). 17

### C. Image Guiding Mechanisms 图像引导机制
Figure 14. On landscapes, convolutional sampling with unconditional models can lead to homogeneous and incoherent global structures (see column 2). $L_2$-guiding with a low resolution image can help to reestablish coherent global structures.

图14.在景观上，无条件模型的卷积采样可以导致均匀和不一致的全局结构(见第2列)$利用低分辨率图像进行L_2$引导可以帮助重建相干全局结构。

An intriguing feature of diffusion models is that unconditional models can be conditioned at test-time [15, 82, 85]. In particular, [15] presented an algorithm to guide both unconditional and conditional models trained on the ImageNet dataset with a classifier log pΦ(y|xt), trained on each xt of the diffusion process. We directly build on this formulation and introduce post-hoc image-guiding:

扩散模型的一个有趣的特点是，无条件模型可以在测试时进行调节[15，82，85]。特别是，[15]提出了一种算法，以指导在ImageNet数据集上训练的无条件和有条件模型，该算法使用在扩散过程的每个xt上训练的分类器log pΦ(y|xt)。我们直接建立在这一公式的基础上，并引入事后形象指导：

For an epsilon-parameterized model with fixed variance, the guiding algorithm as introduced in [15] reads: 
对于具有固定方差的ε参数化模型，[15]中介绍的指导算法如下：

 ˆ ←  θ(zt, t) + q 1 − αt2 ∇zt log pΦ(y|zt) . (16)

This can be interpreted as an update correcting the “score”  θ with a conditional distribution log pΦ(y|zt).

这可以解释为用条件分布对数pΦ(y|zt)修正“分数”θ的更新。

So far, this scenario has only been applied to single-class classification models. We re-interpret the guiding distribution pΦ(y|T(D(z0(zt)))) as a general purpose image-to-image translation task given a target image y, where T can be any differentiable transformation adopted to the image-to-image translation task at hand, such as the identity, a downsampling operation or similar. 

到目前为止，这个场景只应用于单类分类模型。我们将引导分布pΦ(y|T(D(z0(zt)))重新解释为给定目标图像y的通用图像到图像翻译任务，其中T可以是对手头的图像到图像转换任务采用的任何可微变换，例如恒等式、下采样操作或类似操作。

As an example, we can assume a Gaussian guider with fixed variance σ2 = 1, such that 

log pΦ(y|zt) = −12k y − T(D(z0(zt)))k 22 (17) 

becomes a $L_2$ regression objective.

Fig. 14 demonstrates how this formulation can serve as an upsampling mechanism of an unconditional model trained on 2562 images, where unconditional samples of size 2562 guide the convolutional synthesis of 5122 images and T is a 2× bicubic downsampling. Following this motivation, we also experiment with a perceptual similarity guiding and replace the $L_2$ objective with the LPIPS [106] metric, see Sec. 4.4. 
图14展示了该公式如何作为在2562张图像上训练的无条件模型的上采样机制，其中2562大小的无条件样本指导5122张图像的卷积合成，T是2×双三次下采样。根据这一动机，我们还试验了感知相似性指导，并用LPIPS[106]度量代替$L_2$目标，见第4.4.19节

### D. Additional Results 其他结果
#### D.1. Choosing the Signal-to-Noise Ratio for High-Resolution Synthesis 选择高分辨率合成的信噪比
Figure 15. Illustrating the effect of latent space rescaling on convolutional sampling, here for semantic image synthesis on landscapes. See Sec. 4.3.2 and Sec. D.1.
图15。说明了潜在空间重新缩放对卷积采样的影响，此处用于景观上的语义图像合成。见第4.3.2节和第D.1节。

As discussed in Sec. 4.3.2, the signal-to-noise ratio induced by the variance of the latent space (i.e. Var(z)/σt2 ) significantly affects the results for convolutional sampling. For example, when training a LDM directly in the latent space of a KLregularized model (see Tab. 8), this ratio is very high, such that the model allocates a lot of semantic detail early on in the reverse denoising process. In contrast, when rescaling the latent space by the component-wise standard deviation of the latents as described in Sec. G, the SNR is descreased. We illustrate the effect on convolutional sampling for semantic image synthesis in Fig. 15. Note that the VQ-regularized space has a variance close to 1, such that it does not have to be rescaled.

如第4.3.2节所述，潜在空间方差(即Var(z)/σt2)引起的信噪比显著影响卷积采样的结果。例如，当直接在KL正则化模型的潜在空间中训练LDM时(见表8)，这个比率非常高，因此模型在反向去噪过程的早期分配了大量语义细节。相反，当按第G节所述的延迟的分量标准差重新缩放潜在空间时，SNR会降低。我们在图15中说明了对语义图像合成卷积采样的影响。注意，VQ正则化空间的方差接近1，因此不必重新缩放。

#### D.2. Full List of all First Stage Models 所有第一阶段模型的完整列表
We provide a complete list of various autoenconding models trained on the OpenImages dataset in Tab. 8.

我们在表8中提供了在OpenImages数据集上训练的各种自动编码模型的完整列表。

#### D.3. Layout-to-Image Synthesis 图像合成布局
Here we provide the quantitative evaluation and additional samples for our layout-to-image models from Sec. 4.3.1. We train a model on the COCO [4] and one on the OpenImages [49] dataset, which we subsequently additionally finetune on COCO. Tab 9 shows the result. Our COCO model reaches the performance of recent state-of-the art models in layout-toimage synthesis, when following their training and evaluation protocol [89]. When finetuning from the OpenImages model, we surpass these works. Our OpenImages model surpasses the results of Jahn et al [37] by a margin of nearly 11 in terms of FID. In Fig. 16 we show additional samples of the model finetuned on COCO.

在这里，我们为第4.3.1节中的图像模型布局提供了定量评估和额外样本。我们在COCO[4]和OpenImages[49]数据集上训练了一个模型，随后我们对COCO进行了进一步微调。表9显示了结果。当遵循其训练和评估协议时，我们的COCO模型在布局到图像合成方面达到了最先进模型的性能[89]。当从OpenImages模型进行微调时，我们超越了这些工作。我们的OpenImages模型在FID方面超过Jahnet al [37]的结果近11。在图16中，我们显示了在COCO上微调的模型的其他样本。

#### D.4. Class-Conditional Image Synthesis on ImageNet ImageNet上的类条件图像合成
Tab. 10 contains the results for our class-conditional LDM measured in FID and Inception score (IS). LDM-8 requires significantly fewer parameters and compute requirements (see Tab. 18) to achieve very competitive performance. Similar to previous work, we can further boost the performance by training a classifier on each noise scale and guiding with it, see Sec. C. Unlike the pixel-based methods, this classifier is trained very cheaply in latent space. For additional qualitative results, see Fig. 26 and Fig. 27. 

表10包含在FID和初始得分(IS)中测量的类条件LDM的结果。LDM-8需要更少的参数和计算需求(见表18)，以实现极具竞争力的性能。与之前的工作类似，我们可以通过在每个噪声尺度上训练分类器并进行引导来进一步提高性能，请参见第C节。与基于像素的方法不同，这种分类器在潜在空间中的训练成本非常低。其他定性结果见图26和图27。

Table 8. Complete autoencoder zoo trained on OpenImages, evaluated on ImageNet-Val. † denotes an attention-free autoencoder. 
表8.在OpenImages上训练的完整自动编码器动物园，在ImageNet-Val上评估†表示无需注意的自动编码器。

Figure 16. More samples from our best model for layout-to-image synthesis, LDM-4, which was trained on the OpenImages dataset and finetuned on the COCO dataset. Samples generated with 100 DDIM steps and η = 0. Layouts are from the COCO validation set. 
图16.从布局到图像合成的最佳模型LDM-4的更多样本，该模型在OpenImages数据集上进行了训练，并在COCO数据集上得到了微调。使用100个DDIM步骤和η=0生成的样本。布局来自COCO验证集。

Table 9. Quantitative comparison of our layout-to-image models on the COCO [4] and OpenImages [49] datasets. † : Training from scratch on COCO; ∗ : Finetuning from OpenImages.
表9.我们的布局与COCO[4]和OpenImages[49]数据集上的图像模型的定量比较†：从头开始COCO培训; ∗ : 从OpenImages进行微调。

Table 10. Comparison of a class-conditional ImageNet LDM with recent state-of-the-art methods for class-conditional image generation on the ImageNet [12] dataset.∗ : Classifier rejection sampling with the given rejection rate as proposed in [67].
表10.类条件ImageNet LDM与ImageNet[12]数据集上类条件图像生成的最新技术方法的比较。∗ : 分类器拒绝采样，具有[67]中提出的给定拒绝率。

#### D.5. Sample Quality vs. V100 Days (Continued from Sec. 4.1) 样品质量与V100天(续第4.1节)<br/>
Figure 17. For completeness we also report the training progress of class-conditional LDMs on the ImageNet dataset for a fixed number of 35 V100 days. Results obtained with 100 DDIM steps [84] and κ = 0. FIDs computed on 5000 samples for efficiency reasons.
图17.为了完整性，我们还报告了ImageNet数据集上固定数量35 V100天的类条件LDM的训练进度。使用100个DDIM步骤[84]和κ=0获得的结果。出于效率原因，在5000个样本上计算了FID。

For the assessment of sample quality over the training progress in Sec. 4.1, we reported FID and IS scores as a function of train steps. Another possibility is to report these metrics over the used resources in V100 days. Such an analysis is additionally provided in Fig. 17, showing qualitatively similar results. 

为了评估第4.1节中培训过程中的样本质量，我们报告了FID和IS分数作为培训步骤的函数。另一种可能是在V100天内报告这些指标。图17中还提供了这种分析，显示了定性相似的结果。

Table 11. ×4 upscaling results on ImageNet-Val. (2562 ); † : FID features computed on validation split, ‡ : FID features computed on train split. We also include a pixel-space baseline that receives the same amount of compute as LDM-4. The last two rows received 15 epochs of additional training compared to the former results.
表11.×4 ImageNet-Val.的放大结果。(2562 ); † : 验证拆分时计算的FID特征，‡：列车拆分时计算出的FID特征。我们还包括一个像素空间基线，它接收的计算量与LDM-4相同。与前一个结果相比，最后两行接受了15个时期的额外训练。

#### D.6. Super-Resolution
For better comparability between LDMs and diffusion models in pixel space, we extend our analysis from Tab. 5 by comparing a diffusion model trained for the same number of steps and with a comparable number 1 of parameters to our LDM. The results of this comparison are shown in the last two rows of Tab. 11 and demonstrate that LDM achieves better performance while allowing for significantly faster sampling. A qualitative comparison is given in Fig. 20 which shows random samples from both LDM and the diffusion model in pixel space.

为了提高LDM和扩散模型在像素空间中的可比性，我们从表5中扩展了我们的分析，通过将针对相同数量的步骤训练的扩散模型与我们的LDM进行比较，并使用可比数量的1个参数。这种比较的结果显示在表11的最后两行中，并表明LDM实现了更好的性能，同时允许更快的采样。图20给出了定性比较，其中显示了像素空间中LDM和扩散模型的随机样本。

##### D.6.1 LDM-BSR: General Purpose SR Model via Diverse Image Degradation  LDM-BSR：通过多种图像退化的通用SR模型
Figure 18. LDM-BSR generalizes to arbitrary inputs and can be used as a general-purpose upsampler, upscaling samples from a classconditional LDM (image cf . Fig. 4) to 10242 resolution. In contrast, using a fixed degradation process (see Sec. 4.4) hinders generalization.

图18.LDM-BSR概括为任意输入，可以用作通用上采样器，将样本从类条件LDM(图4)上缩放到10242分辨率。相反，使用固定的退化过程(见第4.4节)阻碍了通用化。

To evaluate generalization of our LDM-SR, we apply it both on synthetic LDM samples from a class-conditional ImageNet model (Sec. 4.1) and images crawled from the internet. Interestingly, we observe that LDM-SR, trained only with a bicubicly downsampled conditioning as in [72], does not generalize well to images which do not follow this pre-processing. Hence, to obtain a superresolution model for a wide range of real world images, which can contain complex superpositions of camera noise, compression artifacts, blurr and interpolations, we replace the bicubic downsampling operation in LDM-SR with the degration pipeline from [105]. The BSR-degradation process is a degradation pipline which applies JPEG compressions noise, camera sensor noise, different image interpolations for downsampling, Gaussian blur kernels and Gaussian noise in a random order to an image. We found that using the bsr-degredation process with the original parameters as in [105] leads to a very strong degradation process. Since a more moderate degradation process seemed apppropiate for our application, we adapted the parameters of the bsr-degradation (our adapted degradation process can be found in our code base at https://github.com/CompVis/latent-diffusion ). Fig. 18 illustrates the effectiveness of this approach by directly comparing LDM-SR with LDM-BSR. The latter produces images much sharper than the models confined to a fixed preprocessing, making it suitable for real-world applications. Further results of LDM-BSR are shown on LSUN-cows in Fig. 19. 

为了评估LDM-SR的通用性，我们将其应用于来自类条件ImageNet模型(第4.1节)的合成LDM样本和从互联网抓取的图像。有趣的是，我们观察到LDM-SR(如[72]所示，仅使用双三次下采样条件训练)不能很好地推广到不遵循此预处理的图像。因此，为了获得广泛的真实世界图像的超分辨率模型，该模型可能包含相机噪声、压缩伪影、模糊和插值的复杂叠加，我们用[105]中的降阶流水线替换了LDM-SR中的双三次下采样操作。BSR退化过程是一个退化管道，它以随机顺序将JPEG压缩噪声、相机传感器噪声、用于下采样的不同图像插值、高斯模糊核和高斯噪声应用于图像。我们发现，使用如[105]中的原始参数的bsr退化过程会导致非常强的退化过程。由于更温和的降级过程似乎适合我们的应用程序，我们调整了bsr降级的参数(我们调整的降级过程可以在我们的代码库中找到https://github.com/CompVis/latent-diffusion ). 图18通过直接比较LDM-SR和LDM-BSR说明了该方法的有效性。后者生成的图像比仅限于固定预处理的模型更清晰，使其适用于真实世界的应用。LDM-BSR在LSUN奶牛上的进一步结果如图19所示。

1It is not possible to exactly match both architectures since the diffusion model operates in the pixel space 
1由于扩散模型在像素空间中运行，因此不可能完全匹配这两种架构

### E. Implementation Details and Hyperparameters 实施细节和超参数
#### E.1. Hyperparameters 超参数
We provide an overview of the hyperparameters of all trained LDM models in Tab. 12, Tab. 13, Tab. 14 and Tab. 15.

我们在表12、表13、表14和表15中概述了所有训练LDM模型的超参数。

Table 12. Hyperparameters for the unconditional LDMs producing the numbers shown in Tab. 1. All models trained on a single NVIDIA A100.
表12.产生表1所示数字的无条件LDM的超参数。在单个NVIDIA A100上训练的所有模型。

Table 13. Hyperparameters for the conditional LDMs trained on the ImageNet dataset for the analysis in Sec. 4.1. All models trained on a single NVIDIA A100.
表13.在ImageNet数据集上训练的条件LDM的超参数，用于第4.1节中的分析。在单个NVIDIA A100上训练的所有模型。

#### E.2. Implementation Details
##### E.2.1 Implementations of τθ for conditional LDMs
For the experiments on text-to-image and layout-to-image (Sec. 4.3.1) synthesis, we implement the conditioner τθ as an unmasked transformer which processes a tokenized version of the input y and produces an output ζ := τθ(y), where ζ ∈ RM×dτ . More specifically, the transformer is implemented from N transformer blocks consisting of global self-attention layers, layer-normalization and position-wise MLPs as follows2: 2 adapted from https://github.com/lucidrains/x-transformers 

对于文本到图像和布局到图像(第4.3.1节)合成的实验，我们将调节器τθ实现为无掩模转换器，它处理输入y的标记化版本，并产生输出ζ：=τθ(y)，其中ζ∈ RM×dτ。更具体地说，变换器由N个变换器块实现，该变换器块由全局自关注层、层规范化和按位置MLP组成，如下所示：2https://github.com/lucidrains/x-transformers

Table 14. Hyperparameters for the unconditional LDMs trained on the CelebA dataset for the analysis in Fig. 7. All models trained on a single NVIDIA A100. ∗ : All models are trained for 500k iterations. If converging earlier, we used the best checkpoint for assessing the provided FID scores.
表14.在CelebA数据集上训练的用于图7中分析的无条件LDM的超参数。在单个NVIDIA A100上训练的所有模型。∗ : 所有模型都经过500k次迭代的训练。如果提前收敛，我们使用最佳检查点来评估提供的FID分数。

Table 15. Hyperparameters for the conditional LDMs from Sec. 4. All models trained on a single NVIDIA A100 except for the inpainting model which was trained on eight V100. 
表15.第4节中条件LDM的超参数。除在八个V100上训练的修补模型外，所有模型均在单个NVIDIA A100上训练。

ζ ← TokEmb(y) + PosEmb(y) (18) 
for i = 1, . . . , N : ζ1 ← LayerNorm(ζ) (19) 
ζ2 ← MultiHeadSelfAttention(ζ1) + ζ (20) 
ζ3 ← LayerNorm(ζ2) (21) 
ζ ← MLP(ζ3) + ζ2 (22) 
ζ ← LayerNorm(ζ) (23) (24)

With ζ available, the conditioning is mapped into the UNet via the cross-attention mechanism as depicted in Fig. 3. We modify the “ablated UNet” [15] architecture and replace the self-attention layer with a shallow (unmasked) transformer consisting of T blocks with alternating layers of (i) self-attention, (ii) a position-wise MLP and (iii) a cross-attention layer; see Tab. 16. Note that without (ii) and (iii), this architecture is equivalent to the “ablated UNet”.

ζ可用时，调节通过交叉注意机制映射到UNet中，如图3所示。我们修改了“消融的UNet”[15]架构，并用浅(无掩模)变压器替换自注意层，该变压器由T块组成，交替层为(i)自注意，(ii)位置方向MLP和(iii)交叉注意层; 见表16。注意，如果没有(ii)和(iii)，该架构等同于“消融UNet”。

While it would be possible to increase the representational power of τθ by additionally conditioning on the time step t, we do not pursue this choice as it reduces the speed of inference. We leave a more detailed analysis of this modification to future work.

虽然可以通过额外调节时间步长t来增加τθ的表示力，但我们不追求这种选择，因为它降低了推理的速度。我们将对这一修改进行更详细的分析，留待以后的工作。

For the text-to-image model, we rely on a publicly available3 tokenizer [99]. The layout-to-image model discretizes the spatial locations of the bounding boxes and encodes each box as a (l, b, c)-tuple, where l denotes the (discrete) top-left and b the bottom-right position. Class information is contained in c. See Tab. 17 for the hyperparameters of τθ and Tab. 13 for those of the UNet for both of the above tasks.

对于文本到图像模型，我们依赖于公开可用的3标记器[99]。布局到图像模型将边界框的空间位置离散化，并将每个框编码为(l，b，c)元组，其中l表示(离散的)左上角位置，b表示右下角位置。关于τθ的超参数，见表17; 关于上述两项任务的UNet超参数，参见表13。

Note that the class-conditional model as described in Sec. 4.1 is also implemented via cross-attention, where τθ is a single learnable embedding layer with a dimensionality of 512, mapping classes y to ζ ∈ R1×512 . 

注意，第4.1节中描述的类条件模型也是通过交叉关注实现的，其中τθ是维度为512的单个可学习嵌入层，将类y映射到ζ∈ ×512。

Table 16. Architecture of a transformer block as described in Sec. E.2.1, replacing the self-attention layer of the standard “ablated UNet” architecture [15]. Here, nh denotes the number of attention heads and d the dimensionality per head.
表16.第E.2.1节所述变压器块的结构，取代了标准“消融UNet”结构的自注意层[15]。这里，nh表示注意力头部的数量，d表示每个头部的维度。

Table 17. Hyperparameters for the experiments with transformer encoders in Sec. 4.3.
表17.第4.3节中变压器编码器实验的超参数。

##### E.2.2 Inpainting 修补
For our experiments on image-inpainting in Sec. 4.5, we used the code of [88] to generate synthetic masks. We use a fixed set of 2k validation and 30k testing samples from Places [108]. During training, we use random crops of size 256 × 256 and evaluate on crops of size 512 × 512. This follows the training and testing protocol in [88] and reproduces their reported metrics (see † in Tab. 7). We include additional qualitative results of LDM-4, w/ attn in Fig. 21 and of LDM-4, w/o attn, big, w/ ft in Fig. 22.

对于第4.5节中的图像修复实验，我们使用代码[88]生成合成掩模。我们使用了一组固定的来自Places的2k个验证和30k个测试样本[108]。在训练期间，我们使用大小为256×256的随机作物，并对大小为512×512的作物进行评估。这遵循[88]中的训练和测试协议，并复制其报告的指标(见表7中的†)。我们在图21中包括了LDM-4 w/attn的其他定性结果，在图22中包括LDM-4 w/o attn，big，w/ft的定性结果。

#### E.3. Evaluation Details
This section provides additional details on evaluation for the experiments shown in Sec. 4.

本节提供了第4节所示实验评估的其他详情。

##### E.3.1 Quantitative Results in Unconditional and Class-Conditional Image Synthesis 无条件和类条件图像合成的定量结果
We follow common practice and estimate the statistics for calculating the FID-, Precision- and Recall-scores [29,50] shown in Tab. 1 and 10 based on 50k samples from our models and the entire training set of each of the shown datasets. For calculating FID scores we use the torch-fidelity package [60]. However, since different data processing pipelines might lead to different results [64], we also evaluate our models with the script provided by Dhariwal and Nichol [15]. We find that results 3https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast 26 mainly coincide, except for the ImageNet and LSUN-Bedrooms datasets, where we notice slightly varying scores of 7.76 (torch-fidelity) vs. 7.77 (Nichol and Dhariwal) and 2.95 vs 3.0. For the future we emphasize the importance of a unified procedure for sample quality assessment. Precision and Recall are also computed by using the script provided by Nichol and Dhariwal.

我们遵循常规做法，并根据来自我们模型的50k个样本和每个显示数据集的整个训练集，估计用于计算表1和10中所示FID、精度和召回分数[29，50]的统计数据。为了计算FID分数，我们使用火炬保真度包[60]。然而，由于不同的数据处理管道可能导致不同的结果[64]，我们还使用Dhariwal和Nichol提供的脚本评估了我们的模型[15]。我们发现结果3https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast除了ImageNet和LSUN Bedrooms数据集，我们注意到分数略有不同，分别为7.76(火炬保真度)和7.77(Nichol和Dhariwal)以及2.95和3.0。对于未来，我们强调统一的样品质量评估程序的重要性。精度和召回率也通过使用Nichol和Dhariwal提供的脚本进行计算。

##### E.3.2 Text-to-Image Synthesis 文本到图像合成
Following the evaluation protocol of [66] we compute FID and Inception Score for the Text-to-Image models from Tab. 2 by comparing generated samples with 30000 samples from the validation set of the MS-COCO dataset [51]. FID and Inception Scores are computed with torch-fidelity.

根据[66]的评估协议，我们通过将生成的样本与MS-COCO数据集验证集的30000个样本进行比较，计算表2中文本到图像模型的FID和初始得分[51]。FID和初始得分采用火炬保真度计算。

##### E.3.3 Layout-to-Image Synthesis 图像合成布局
For assessing the sample quality of our Layout-to-Image models from Tab. 9 on the COCO dataset, we follow common practice [37, 87, 89] and compute FID scores the 2048 unaugmented examples of the COCO Segmentation Challenge split. To obtain better comparability, we use the exact same samples as in [37]. For the OpenImages dataset we similarly follow their protocol and use 2048 center-cropped test images from the validation set.

为了评估COCO数据集表9中我们的布局到图像模型的样本质量，我们遵循常见做法[37，87，89]，并计算COCO分割挑战分割的2048个未分段样本的FID分数。为了获得更好的可比性，我们使用了与[37]中完全相同的样本。对于OpenImages数据集，我们同样遵循他们的协议，并使用验证集中2048个中心裁剪的测试图像。

##### E.3.4 Super Resolution
We evaluate the super-resolution models on ImageNet following the pipeline suggested in [72], i.e. images with a shorter size less than 256 px are removed (both for training and evaluation). On ImageNet, the low-resolution images are produced using bicubic interpolation with anti-aliasing. FIDs are evaluated using torch-fidelity [60], and we produce samples on the validation split. For FID scores, we additionally compare to reference features computed on the train split, see Tab. 5 and Tab. 11.

我们按照[72]中建议的管道在ImageNet上评估超分辨率模型，即去除尺寸小于256像素的图像(用于训练和评估)。在ImageNet上，使用具有抗锯齿的双三次插值生成低分辨率图像。FID使用火炬保真度进行评估[60]，我们在验证分割上制作样本。对于FID分数，我们还将其与列车分割计算的参考特征进行了比较，见表5和表11。

##### E.3.5 Efficiency Analysis
For efficiency reasons we compute the sample quality metrics plotted in Fig. 6, 17 and 7 based on 5k samples. Therefore, the results might vary from those shown in Tab. 1 and 10. All models have a comparable number of parameters as provided in Tab. 13 and 14. We maximize the learning rates of the individual models such that they still train stably. Therefore, the learning rates slightly vary between different runs cf . Tab. 13 and 14.

出于效率原因，我们基于5k个样本计算了图6、17和7中绘制的样本质量度量。因此，结果可能与表1和表10中所示的结果不同。所有模型都具有表13和表14中提供的可比数量的参数。我们最大化了各个模型的学习率，使它们仍然稳定地训练。因此，不同跑步之间的学习率略有不同cf。表13和14。

##### E.3.6 User Study
For the results of the user study presented in Tab. 4 we followed the protocoll of [72] and and use the 2-alternative force-choice paradigm to assess human preference scores for two distinct tasks. In Task-1 subjects were shown a low resolution/masked image between the corresponding ground truth high resolution/unmasked version and a synthesized image, which was generated by using the middle image as conditioning. For SuperResolution subjects were asked: ’Which of the two images is a better high quality version of the low resolution image in the middle?’. For Inpainting we asked ’Which of the two images contains more realistic inpainted regions of the image in the middle?’. In Task-2, humans were similarly shown the lowres/masked version and asked for preference between two corresponding images generated by the two competing methods. As in [72] humans viewed the images for 3 seconds before responding.

对于表4中所示的用户研究结果，我们遵循[72]的协议，并使用2种替代力选择范式来评估两种不同任务的人类偏好得分。在任务1中，受试者被显示在相应的地面真相高分辨率/无掩模版本和合成图像之间的低分辨率/掩模图像，合成图像通过使用中间图像作为条件生成。对于超分辨率，受试者被问及：“两幅图像中哪一幅是中间低分辨率图像的高质量版本？”。对于Inpainting，我们问“这两幅图像中哪一幅包含了图像中间更逼真的修复区域？”。在任务2中，人类同样被展示了低分辨率/蒙面版本，并被要求在两种竞争方法生成的两个对应图像之间进行偏好选择。与[72]中一样，人类在回应之前观看了3秒钟的图像。

### F. Computational Requirements 计算要求
Table 18. Comparing compute requirements during training and inference throughput with state-of-the-art generative models. Compute during training in V100-days, numbers of competing methods taken from [15] unless stated differently;∗ : Throughput measured in samples/sec on a single NVIDIA A100;† : Numbers taken from [15] ;‡ : Assumed to be trained on 25M train examples; ††: R-FID vs. ImageNet validation set

表18.将训练期间的计算需求和推理吞吐量与最先进的生成模型进行比较。在V100天的培训期间，计算[15]中的竞争方法数量，除非另有说明; ∗ : 单个NVIDIA A100上以样本/秒为单位测量的吞吐量; †：数字取自[15]; ‡：假设接受25M列车样本培训; ††：R-FID与ImageNet验证集

In Tab 18 we provide a more detailed analysis on our used compute ressources and compare our best performing models on the CelebA-HQ, FFHQ, LSUN and ImageNet datasets with the recent state of the art models by using their provided numbers, cf . [15]. As they report their used compute in V100 days and we train all our models on a single NVIDIA A100 GPU, we convert the A100 days to V100 days by assuming a ×2.2 speedup of A100 vs V100 [74]4 . To assess sample quality, we additionally report FID scores on the reported datasets. We closely reach the performance of state of the art methods as StyleGAN2 [42] and ADM [15] while significantly reducing the required compute resources. 4This factor corresponds to the speedup of the A100 over the V100 for a U-Net, as defined in Fig. 1 in [74] 28

在表18中，我们对我们使用的计算资源进行了更详细的分析，并使用提供的数字将CelebA HQ、FFHQ、LSUN和ImageNet数据集上的最佳性能模型与最新的最先进模型进行比较，cf。[15]。由于他们在V100天内报告了他们使用的计算，并且我们在单个NVIDIA A100 GPU上训练了所有模型，因此我们假设A100与V100的速度比为2.2倍，从而将A100天转换为V100天[74]4。为了评估样本质量，我们还报告了报告数据集上的FID分数。我们接近StyleGAN2[42]和ADM[15]等现有技术方法的性能，同时显著减少了所需的计算资源。4该系数对应于U-Net的A100比V100的加速，如[74]28中的图1所示

### G. Details on Autoencoder Models 自动编码器模型详情
We train all our autoencoder models in an adversarial manner following [23], such that a patch-based discriminator Dψ is optimized to differentiate original images from reconstructions D(E(x)). To avoid arbitrarily scaled latent spaces, we regularize the latent z to be zero centered and obtain small variance by introducing an regularizing loss term Lreg. We investigate two different regularization methods: (i) a low-weighted Kullback-Leibler-term between qE (z|x) = N (z; Eµ, Eσ2 ) and a standard normal distribution N (z; 0, 1) as in a standard variational autoencoder [46, 69], and, (ii) regularizing the latent space with a vector quantization layer by learning a codebook of |Z| different exemplars [96]. To obtain high-fidelity reconstructions we only use a very small regularization for both scenarios, i.e. we either weight the KL term by a factor ∼ 10−6 or choose a high codebook dimensionality |Z|.

我们在[23]之后以对抗的方式训练我们的所有自动编码器模型，使得基于分块的鉴别器Dψ被优化以区分原始图像和重建D(E(x))。为了避免任意缩放的潜在空间，我们将潜在z正则化为零中心，并通过引入正则化损失项Lreg来获得小方差。我们研究了两种不同的正则化方法：(i)在标准变分自动编码器[46，69]中，qE(z|x)=N(z; Eµ，Eσ2)和标准正态分布N(z，0，1)之间的低权重Kullback-Leibler项，以及(ii)通过学习|z|不同样本的码本，用矢量量化层正则化潜在空间[96]。为了获得高保真度重建，我们只对这两种场景使用非常小的正则化，即我们通过因子对KL项进行加权∼ 10−6或选择高码本维度|Z|。

The full objective to train the autoencoding model (E, D) reads:

训练自动编码模型(E，D)的完整目标如下：

LAutoencoder = min E,D max ψ  Lrec(x, D(E(x))) − Ladv(D(E(x))) + log Dψ(x) + Lreg(x; E, D) (25)

DM Training in Latent Space. Note that for training diffusion models on the learned latent space, we again distinguish two cases when learning p(z) or p(z|y) (Sec. 4.3): (i) For a KL-regularized latent space, we sample z = Eµ(x)+Eσ(x)·ε =: E(x), where ε ∼ N (0, 1). When rescaling the latent, we estimate the component-wise variance 

潜在空间中的DM培训。注意，对于学习的潜在空间上的训练扩散模型，我们再次区分了学习p(z)或p(z|y)时的两种情况(第4.3节)：(i)对于KL正则化的潜在空间，我们采样z=Eµ(x)+Eσ(x)·ε=：E(x)，其中ε∼ N(0，1)。当重新缩放潜在值时，我们估计分量方差

σˆ2 = 1 bchw X b,c,h,w (z b,c,h,w − µˆ)2 

from the first batch in the data, where µˆ = 1 bchw P b,c,h,w z b,c,h,w. The output of E is scaled such that the rescaled latent has unit standard deviation, i.e. z ← zˆσ = E(x) ˆσ . (ii) For a VQ-regularized latent space, we extract z before the quantization layer and absorb the quantization operation into the decoder, i.e. it can be interpreted as the first layer of D.

从数据中的第一批数据中，其中µ=1 bchw P b，c，h，w z b，c、h，w。对E的输出进行缩放，使得重新缩放的潜像具有单位标准偏差，即z← (ii)对于VQ正则化潜空间，我们在量化层之前提取z，并将量化操作吸收到解码器中，即它可以被解释为D的第一层。

### H. Additional Qualitative Results 其他定性结果
Finally, we provide additional qualitative results for our landscapes model (Fig. 12, 23, 24 and 25), our class-conditional ImageNet model (Fig. 26 - 27) and our unconditional models for the CelebA-HQ, FFHQ and LSUN datasets (Fig. 28 - 31). Similar as for the inpainting model in Sec. 4.5 we also fine-tuned the semantic landscapes model from Sec. 4.3.2 directly on 5122 images and depict qualitative results in Fig. 12 and Fig. 23. For our those models trained on comparably small datasets, we additionally show nearest neighbors in VGG [79] feature space for samples from our models in Fig. 32 - 34. 

最后，我们为景观模型(图12、23、24和25)、类条件ImageNet模型(图26-27)和CelebA HQ、FFHQ和LSUN数据集的无条件模型(图28-31)提供了额外的定性结果。与第4.5节中的修复模型类似，我们还直接在5122张图像上微调了第4.3.2节中的语义景观模型，并在图12和图23中描述了定性结果。对于在相对较小的数据集上训练的那些模型，我们还显示了VGG[79]特征空间中的最近邻居，用于图32-34中我们模型的样本。

Figure 19. LDM-BSR generalizes to arbitrary inputs and can be used as a general-purpose upsampler, upscaling samples from the LSUNCows dataset to 10242 resolution. 
图19.LDM-BSR概括为任意输入，可以用作通用上采样器，将LSUNCows数据集的样本放大到10242分辨率。

Figure 20. Qualitative superresolution comparison of two random samples between LDM-SR and baseline-diffusionmodel in Pixelspace.Evaluated on imagenet validation-set after same amount of training steps. 
图20.Pixelspace中LDM-SR和基线扩散模型之间两个随机样本的定性超分辨率比较。在相同数量的训练步骤后，在imagenet验证集上进行评估。

Figure 21. Qualitative results on image inpainting. In contrast to [88], our generative approach enables generation of multiple diverse samples for a given input. 
图21.图像修复的定性结果。与[88]相反，我们的生成方法能够为给定输入生成多个不同的样本。

Figure 22. More qualitative results on object removal as in Fig. 11. 
图22.如图11所示，物体移除的更多定性结果。

Figure 23. Convolutional samples from the semantic landscapes model as in Sec. 4.3.2, finetuned on 5122 images. 
图23.第4.3.2节中语义景观模型的卷积样本，对5122幅图像进行了微调。

Figure 24. A LDM trained on 2562 resolution can generalize to larger resolution for spatially conditioned tasks such as semantic synthesis of landscape images. See Sec. 4.3.2.  
图24.基于2562分辨率训练的LDM可以推广到更大的分辨率，用于空间条件任务，如景观图像的语义合成。见第4.3.2节。

Figure 25. When provided a semantic map as conditioning, our LDMs generalize to substantially larger resolutions than those seen during training. Although this model was trained on inputs of size 2562 it can be used to create high-resolution samples as the ones shown here, which are of resolution 1024 × 384. 
图25.当提供语义图作为条件时，我们的LDM概括为比训练期间看到的分辨率大得多的分辨率。尽管该模型是在2562大小的输入上训练的，但它可以用于创建如这里所示的高分辨率样本，分辨率为1024×384。

Figure 26. Random samples from LDM-4 trained on the ImageNet dataset. Sampled with classifier-free guidance [32] scale s = 5.0 and 200 DDIM steps with η = 1.0. 
图26.在ImageNet数据集上训练的LDM-4随机样本。采用无分类器制导[32]尺度s=5.0和200 DDIM步长η=1.0进行采样。

Figure 27. Random samples from LDM-4 trained on the ImageNet dataset. Sampled with classifier-free guidance [32] scale s = 3.0 and 200 DDIM steps with η = 1.0. 38
图27.在ImageNet数据集上训练的LDM-4随机样本。采用无分类器制导[32]尺度s=3.0和200 DDIM步长进行采样，η=1.0.38

Figure 28. Random samples of our best performing model LDM-4 on the CelebA-HQ dataset. Sampled with 500 DDIM steps and η = 0 (FID = 5.15). 
图28.CelebA HQ数据集上性能最佳的LDM-4模型的随机样本。以500 DDIM步进采样，η=0(FID=5.15)。

Figure 29. Random samples of our best performing model LDM-4 on the FFHQ dataset. Sampled with 200 DDIM steps and η = 1 (FID = 4.98). 
图29.FFHQ数据集上性能最佳的LDM-4模型的随机样本。采用200 DDIM步骤采样，η=1(FID=4.98)。

Figure 30. Random samples of our best performing model LDM-8 on the LSUN-Churches dataset. Sampled with 200 DDIM steps and η = 0 (FID = 4.48).
图30.LSUN Churches数据集上性能最佳的LDM-8模型的随机样本。以200 DDIM步进采样，η=0(FID=4.48)。

Figure 31. Random samples of our best performing model LDM-4 on the LSUN-Bedrooms dataset. Sampled with 200 DDIM steps and η = 1 (FID = 2.95). 
图31.LSUN卧室数据集上性能最佳的LDM-4模型的随机样本。采用200 DDIM步骤采样，η=1(FID=2.95)。

Figure 32. Nearest neighbors of our best CelebA-HQ model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors. 
图32.在VGG-16的特征空间中计算的最佳CelebA HQ模型的最近邻居[79]。最左边的样本来自我们的模型。每行中的剩余样本是其10个最近的邻居。

Figure 33. Nearest neighbors of our best FFHQ model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors. 
图33.在VGG-16的特征空间中计算的最佳FFHQ模型的最近邻居[79]。最左边的样本来自我们的模型。每行中的剩余样本是其10个最近的邻居。

Figure 34. Nearest neighbors of our best LSUN-Churches model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors. 
图34.在VGG-16的特征空间中计算的最佳LSUN Churches模型的最近邻居[79]。最左边的样本来自我们的模型。每行中的剩余样本是其10个最近的邻居。
