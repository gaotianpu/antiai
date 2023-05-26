# Consistency Models 
一致性模型 2023.3.2  https://arxiv.org/abs/2303.01469

## Abstract 摘要
Diffusion models have made significant breakthroughs in image, audio, and video generation, but they depend on an iterative generation process that causes slow sampling speed and caps their potential for real-time applications. To overcome this limitation, we propose consistency models, a new family of generative models that achieve high sample quality without adversarial training. They support fast one-step generation by design, while still allowing for few-step sampling to trade compute for sample quality. They also support zero-shot data editing, like image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either as a way to distill pre-trained diffusion models, or as standalone generative models. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in oneand few-step generation. For example, we achieve the new state-of-the-art FID of 3.55 on CIFAR- 10 and 6.20 on ImageNet 64 ˆ 64 for one-step generation. When trained as standalone generative models, consistency models also outperform single-step, non-adversarial generative models on standard benchmarks like CIFAR-10, ImageNet 64*64 and LSUN 256*256.

扩散模型在图像、音频和视频生成方面取得了重大突破，但它们依赖于迭代生成过程，这会导致采样速度慢，并限制其实时应用的潜力。为了克服这一限制，我们提出了一致性模型，这是一种新的生成模型家族，可以在没有对抗性训练的情况下实现高样本质量。它们支持通过设计快速一步生成，同时仍然允许进行几步采样，以换取计算和样本质量。它们还支持零样本数据编辑，如图像修复、彩色化和超分辨率，而无需对这些任务进行明确训练。一致性模型可以作为提取预先训练的扩散模型的一种方式进行训练，也可以作为独立的生成模型进行训练。通过广泛的实验，我们证明它们在一步和几步生成中优于现有的扩散模型蒸馏技术。例如，我们在CIFAR-10上实现了新的最先进的FID 3.55，在ImageNet 64上实现了6.20，用于一步生成。当作为独立的生成模型进行训练时，一致性模型在CIFAR-10、ImageNet 64*64和LSUN 256*256等标准基准上也优于单步非对抗性生成模型。

## 1. Introduction
Diffusion models (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; 2020; Ho et al., 2020; Song et al., 2021), also known as score-based generative models, have achieved unprecedented success across multiple fields, including image generation (Dhariwal & Nichol, 2021; Nichol et al., 2021; Ramesh et al., 2022; Saharia et al., 2022; Rombach et al., 2022), audio synthesis (Kong et al., 2020; Chen et al., 2021; Popov et al., 2021), and video generation (Ho et al., 2022b;a). Unlike Generative Adversarial Networks (GANs, Goodfellow et al. (2014)), these models do not rely on adversarial training and are thus less prone to issues such as unstable training and mode collapse. Additionally, diffusion models do not impose the same strict constraints on model architectures as in autoregressive models (Bengio & Bengio, 1999; Uria et al., 2013; 2016; Van Den Oord et al., 2016), variational autoencoders (VAEs, Kingma & Welling (2014); Rezende et al. (2014)), or normalizing flows (Dinh et al., 2015; 2017; Kingma & Dhariwal, 2018).

扩散模型(Sohl-Dickstein et al., 2015; Song&Ermon，2019;2020; Ho et al., 2020;Song et al., 2021)，也称为基于分数的生成模型，在多个领域取得了前所未有的成功，包括图像生成(Dhariwal&Nichol，2021;Nichol et al., 2021;Ramesh et al., 2022;Saharia et al., 2022.Rombach et al., 2022-)，音频合成(Kong et al., 2020;Chen et al., 2021;Popov et al., 2021)和视频生成(Ho et al., 2022b;a)。与生成对抗性网络(GANs，Goodfellow et al.(2014))不同，这些模型不依赖于对抗性训练，因此不太容易出现不稳定训练和模式崩溃等问题。此外，扩散模型并不像自回归模型(Bengio&Bengio，1999;Uria et al., 2013;2016;Van Den Oord et al., 2016)、变分自动编码器(VAEs，Kingma&Welling(2014);Rezende等人(2014))，或归一化流(Dinh et al., 2015;2017;Kingma和Dhariwal，2018) 那样对模型架构施加严格的约束。

Figure 1: Given a Probability Flow (PF) ODE that smoothly converts data to noise, we learn to map any point (e.g., xt, xt 1 , and xT ) on the ODE trajectory to its origin (e.g., x0) for generative modeling. Models of these mappings are called consistency models, as their outputs are trained to be consistent for points on the same trajectory.
图1：给定一个将数据平滑转换为噪声的概率流(PF)ODE，我们学习将ODE轨迹上的任何点(例如，xt、xt 1和xt)映射到其原点(例如，x0)，以进行生成建模。这些映射的模型被称为一致性模型，因为它们的输出被训练为对于同一轨迹上的点是一致的。

Key to the success of diffusion models is their iterative sampling process which progressively removes noise from a random noise vector. This iterative refinement procedure repetitively evaluates the diffusion model, allowing for the trade-off of compute for sample quality: by using extra compute for more iterations, a small-sized model can unroll into a larger computational graph and generate higher quality samples. Iterative generation is also crucial for the zeroshot data editing capabilities of diffusion models, enabling them to solve challenging inverse problems ranging from image inpainting, colorization, stroke-guided image editing, to Computed Tomography and Magnetic Resonance Imaging (Song & Ermon, 2019; Song et al., 2021; 2022; 2023; Kawar et al., 2021; 2022; Chung et al., 2023; Meng et al., 2021). However, compared to single-step generative models like GANs, VAEs, and normalizing flows, the iterative generation procedure of diffusion models typically requires 10–2000 times more compute (Song & Ermon, 2020; Ho et al., 2020; Song et al., 2021; Zhang & Chen, 2022; Lu et al., 2022), causing slow inference and limiting their real time applications.

扩散模型成功的关键是其迭代采样过程，该过程从随机噪声向量中逐步去除噪声。这种迭代精化过程反复评估扩散模型，允许计算与样本质量的权衡：通过使用额外的计算进行更多的迭代，小型模型可以展开到更大的计算图中，并生成更高质量的样本。迭代生成对于扩散模型的零样本数据编辑功能也至关重要，使其能够解决从图像修复、彩色化、笔划引导图像编辑到计算层析成像和核磁共振成像等具有挑战性的反问题(Song&Ermon，2019;Song et al.，2021;2022;2023;Kawar et al.、2021;2022年;Chung et al.和Meng et al..，2021)。然而，与单步生成模型(如GAN、VAE和归一化流)相比，扩散模型的迭代生成过程通常需要10–2000倍的计算量(Song&Ermon，2020;Ho et al.，2020;Song et al., 2021;Zhang&Chen，2022;Lu et al., 2022)，这导致推理速度慢，限制了它们的实时应用。

Our objective is to create generative models that facilitate efficient, single-step generation without sacrificing important advantages of iterative refinement. These advantages include the ability to trade-off compute for sample quality when necessary, as well as the capability to perform zeroshot data editing tasks. As illustrated in Fig. 1, we build on top of the probability flow (PF) ordinary differential equation (ODE) in continuous-time diffusion models (Song et al., 2021), whose trajectories smoothly transition the data distribution into a tractable noise distribution. We propose to learn a model that maps any point at any time step to the trajectory’s starting point. A notable property of our model is self-consistency: points on the same trajectory map to the same initial point. We therefore refer to such models as consistency models. Consistency models allow us to generate data samples (initial points of ODE trajectories, e.g., x0 in Fig. 1) by converting random noise vectors (endpoints of ODE trajectories, e.g., xT in Fig. 1) with only one network evaluation. Importantly, by chaining the outputs of consistency models at multiple time steps, we can improve sample quality and perform zero-shot data editing at the cost of more compute, similar to what iterative refinement enables for diffusion models.

我们的目标是创建生成模型，在不牺牲迭代精化的重要优势的情况下，促进高效的单步生成。这些优势包括在必要时权衡计算和样本质量的能力，以及执行零样本数据编辑任务的能力。如图1所示，我们在连续时间扩散模型(Song et al.，2021)中的概率流(PF)常微分方程(ODE)的基础上构建，其轨迹将数据分布平稳过渡到可控制的噪声分布。我们建议学习一个模型，该模型将任何时间步长的任何点映射到轨迹的起点。我们模型的一个显著特性是自一致性：同一轨迹上的点映射到同一初始点。因此，我们将此类模型称为一致性模型。一致性模型允许我们生成数据样本(ODE轨迹的初始点，例如，图1中的x0)，通过转换随机噪声向量(ODE轨道的端点，例如，在图1中，xT)，只需一次网络评估。重要的是，通过在多个时间步长链接一致性模型的输出，我们可以提高样本质量并以更多计算为代价执行零样本数据编辑，这与迭代求精对扩散模型的支持类似。

To train a consistency model, we offer two methods based on enforcing the self-consistency property. The first method relies on using numerical ODE solvers and a pre-trained diffusion model to generate pairs of adjacent points on a PF ODE trajectory. By minimizing the difference between model outputs for these pairs, we can effectively distill a diffusion model into a consistency model, which allows generating high-quality samples with one network evaluation. By contrast, our second method eliminates the need for a pre-trained diffusion model altogether, allowing us to train a consistency model in isolation. This approach situates consistency models as an independent family of generative models. Crucially, neither approach requires adversarial training, and both training methods permit flexible neural network architectures for consistency models.

为了训练一致性模型，我们提供了两种基于增广自一致性属性的方法。第一种方法依赖于使用数值ODE解算器和预先训练的扩散模型来生成PF ODE轨迹上的成对相邻点。通过最小化这些对的模型输出之间的差异，我们可以有效地将扩散模型提取为一致性模型，这允许通过一个网络评估生成高质量的样本。相比之下，我们的第二种方法完全消除了对预先训练的扩散模型的需要，使我们能够孤立地训练一致性模型。这种方法将一致性模型定位为一个独立的生成模型家族。至关重要的是，这两种方法都不需要对抗性训练，而且这两种训练方法都允许灵活的神经网络架构用于一致性模型。

We demonstrate the efficacy of consistency models on several challenging image benchmarks, including CIFAR-10 (Krizhevsky et al., 2009), ImageNet 64 ˆ 64 (Deng et al., 2009), and LSUN 256 ˆ 256 (Yu et al., 2015). Empirically, we observe that as a distillation approach, consistency models outperform progressive distillation (Salimans & Ho, 2022) across a variety of datasets and number of sampling steps. On CIFAR-10, consistency models reach new state-of-the-art FIDs of 3.55 and 2.93 for one-step and two-step generation. On ImageNet 64 ˆ 64, it achieves record-breaking FIDs of 6.20 and 4.70 with one and two network evaluations respectively. When trained as standalone generative models, consistency models achieve comparable performance to progressive distillation for single step generation, despite having no access to pre-trained diffusion models. They are able to outperform many GANs, and all other non-adversarial, single-step generative models across multiple datasets. We also show that consistency models can be used to perform zero-shot data editing tasks, including image denoising, interpolation, inpainting, colorization, super-resolution, and stroke-guided image editing (SDEdit, Meng et al. (2021)).

我们证明了一致性模型在几个具有挑战性的图像基准上的有效性，包括CIFAR-10(Krizhevsky et al.，2009)、ImageNet 64(Deng et al.，09)和LSUN 256(Yu et al.，2015)。根据经验，我们观察到，作为一种蒸馏方法，一致性模型在各种数据集和采样步骤数量上都优于渐进蒸馏(Salimans&Ho，2022)。在CIFAR-10上，一致性模型达到了最先进的一步和两步生成的3.55和2.93的FID。在ImageNet 64上，它分别通过一次和两次网络评估实现了6.20和4.70的破纪录FID。当作为独立的生成模型进行训练时，一致性模型在单步生成中实现了与渐进蒸馏相当的性能，尽管无法访问预先训练的扩散模型。它们能够在多个数据集上胜过许多Gan和所有其他非对抗性的单步生成模型。我们还表明，一致性模型可以用于执行零样本数据编辑任务，包括图像去噪、插值、修复、彩色化、超分辨率和笔划引导图像编辑(SDEdit，Meng et al.(2021))。

## 2. Diffusion Models  扩散模型
Consistency models are heavily inspired by the theory of (continuous-time) diffusion models (Song et al., 2021). Diffusion models generate data by progressively perturbing data to noise via Gaussian perturbations, then creating samples from noise via sequential denoising steps. Let pdatapxq denote the data distribution. Diffusion models start by diffusing pdatapxq with a stochastic differential equation (SDE) (Song et al., 2021; Karras et al., 2022)

一致性模型深受(连续时间)扩散模型理论的启发(Song et al., 2021)。扩散模型通过高斯扰动将数据逐渐扰动为噪声，然后通过顺序去噪步骤从噪声中创建样本，从而生成数据。让updateapxq表示数据分布。扩散模型从使用随机微分方程(SDE)扩散pdatapxq开始(Song et al., 2021;Karras et al., 2022年)

dxt ( µpxt, tq dt + σptq dwt, (1)

where t P r0, Ts, T ą 0 is a fixed constant, µp¨, ¨q and gp¨q are the drift and diffusion coefficients respectively, and twtutPr0,Ts denotes the standard Brownian motion. We denote the distribution of xt as ptpxq and as a result p0pxq ” pdatapxq. A remarkable property of this SDE is the existence of an ordinary differential equation (ODE), dubbed the Probability Flow (PF) ODE by Song et al. (2021), whose solution trajectories sampled at t are distributed according to ptpxq:

其中t P r0，Ts，tã0是一个固定常数，µP¨，¨q和gp¨q分别是漂移系数和扩散系数，twtutPr0，Ts表示标准布朗运动。我们将xt的分布表示为ptpxq，结果是p0pxq“updateapxq”。此SDE的一个显著特性是存在一个常微分方程(ODE)，Song等人(2021)将其称为概率流(PF)ODE，其在t处采样的解轨迹根据ptpxq分布：

dxt ( „ µpxt, tq - 1 2 σptq 2∇ log ptpxtq  dt. (2)

Here ∇ log ptpxq is the score function of ptpxq; hence diffusion models are also known as score-based generative models (Song & Ermon, 2019; 2020; Song et al., 2021).

这里，Şlog-ptpxq是ptpxq的得分函数;因此，扩散模型也称为基于分数的生成模型(Song&Ermon，2019;2020;Song et al., 2021)。

Typically, the SDE in Eq. (1) is designed such that pT pxq is close to a tractable Gaussian distribution πpxq. We hereafter adopt the configurations in Karras et al. (2022), who set µpx, tq ( 0 and σptq ( ? 2t. In this case, we have ptpxq ( pdatapxq b N p0, t2Iq, where b denotes the convolution operation, and πpxq ( N p0, T2Iq. For sampling, we first train a score model sφpx, tq « ∇ log ptpxq via score matching (Hyv¨arinen & Dayan, 2005; Vincent, 2011; Song et al., 2019; Song & Ermon, 2019; Ho et al., 2020), then plug it into Eq. (2) to obtain an empirical estimate of the PF ODE, which takes the form of

通常，方程(1)中的SDE被设计为使得pT pxq接近于可处理的高斯分布πpxq。此后，我们采用了Karras等人(2022)中的配置，他设置了µpx，tq(0和σptq(？2t。在这种情况下，我们有ptpxq(updateapxq b N p0，t2Iq，其中b表示卷积运算)和πpxq，然后将其代入等式。(2)以获得PF ODE的经验估计，其形式为

dxt dt ( -tsφpxt, tq. (3)

We call Eq. (3) the empirical PF ODE. Next, we sample ˆxT „ π ( N p0, T2Iq to initialize the empirical PF ODE and solve it backwards in time with any numerical ODE solver, such as Euler (Song et al., 2020; 2021) and Heun solvers (Karras et al., 2022), to obtain the solution trajectory tˆxtutPr0,Ts . The resulting ˆx0 can then be viewed as an approximate sample from the data distribution pdatapxq. To avoid numerical instability, one typically stops the solver at t ( $\epsilon$ , where $\epsilon$ is a fixed small positive number, and instead accepts ˆx$\epsilon$ as the approximate sample. Following Karras et al. (2022), we rescale pixel values in images to r-1, 1s, then set T ( 80, and $\epsilon$ ( 0.002.

我们称方程(3)为经验PF ODE。接下来，我们采样ˆxT·π(N p0，T2Iq)来初始化经验PF ODE，并使用任何数值ODE解算器(如Euler(Song et al.，2020;2021)和Heun解算器，如Karras et al.(2022))及时反向求解，以获得解轨迹t \710]xttPr0，Ts。然后，可以将得到的Plot x0视为来自数据分布pdateapxq的近似样本。为了避免数值不稳定，通常会将解算器停止在t($\epsilon$，其中$\epsiron$是一个固定的小正数，而接受ξx$\epsi隆$作为近似样本。根据Karras等人(2022)，我们将图像中的像素值重新缩放为r-1，1s，然后设置t(80，和$\epsilon$(0.002)。

Figure 2: Consistency models are trained to map points on any trajectory of the PF ODE to the trajectory’s origin.
图2：一致性模型被训练成将PF ODE的任何轨迹上的点映射到轨迹的原点。

Diffusion models are bottlenecked by their slow sampling speed. Clearly, using ODE solvers for sampling requires many evaluations of the score model sφpx, tq, which is computationally costly. Existing methods for fast sampling include faster numerical ODE solvers (Song et al., 2020; Zhang & Chen, 2022; Lu et al., 2022; Dockhorn et al., 2022), and distillation techniques (Luhman & Luhman, 2021; Salimans & Ho, 2022; Meng et al., 2022; Zheng et al., 2022). However, ODE solvers still need more than 10 evaluation steps to generate competitive samples. Most distillation methods like Luhman & Luhman (2021) and Zheng et al. (2022) rely on collecting a large dataset of samples from the diffusion model prior to distillation, which itself is computationally expensive. To our best knowledge, the only distillation approach that does not suffer from this drawback is progressive distillation (PD, Salimans & Ho (2022)), with which we compare consistency models extensively in our experiments.

扩散模型由于采样速度慢而受到瓶颈。显然，使用ODE解算器进行采样需要对分数模型sφpx，tq进行多次评估，这在计算上是昂贵的。现有的快速采样方法包括更快的数值常微分方程解算器(Song et al., 2020年;Zhang&Chen，2022年;Lu et al., 2022;Dockhorn et al., 2022中)和蒸馏技术(Luhman&Luhman，2021;Salimans&Ho，2022，Meng et al., 2022.Zheng et al., 2022-)。然而，ODE求解器仍然需要10个以上的评估步骤才能生成有竞争力的样本。大多数蒸馏方法，如Luhman&Luhman(2021)和Zheng等人(2022年)，都依赖于在蒸馏之前从扩散模型中收集大量样本数据集，这本身计算成本很高。据我们所知，唯一没有这种缺点的蒸馏方法是渐进蒸馏(PD，Salimans&Ho(2022))，我们在实验中广泛比较了一致性模型。

## 3. Consistency Models  一致性模型
We propose consistency models, a new type of generative models that support single-step generation at the core of its design, while still allowing iterative generation for zeroshot data editing and trade-offs between sample quality and compute. Consistency models can be trained in either the distillation mode or the isolation mode. In the former case, consistency models distill the knowledge of pre-trained diffusion models into a single-step sampler, significantly improving other distillation approaches in sample quality, while allowing zero-shot image editing applications. In the latter case, consistency models are trained in isolation, with no dependence on pre-trained diffusion models. This makes them an independent new class of generative models.

我们提出了一致性模型，这是一种新型的生成模型，在其设计的核心支持单步生成，同时仍然允许零样本数据编辑的迭代生成以及样本质量和计算之间的权衡。稠度模型可以在蒸馏模式或分离模式中进行训练。在前一种情况下，一致性模型将预处理扩散模型的知识提取到一个单步采样器中，显著改进了样品质量的其他提取方法，同时允许使用零样本图像编辑应用程序。在后一种情况下，一致性模型是孤立训练的，不依赖于预先训练的扩散模型。这使它们成为一类独立的新生成模型。

Below we introduce the definition, parameterization, and sampling of consistency models, plus a brief discussion on their applications to zero-shot data editing.

下面我们介绍一致性模型的定义、参数化和采样，并简要讨论它们在零样本数据编辑中的应用。

Definition Given a solution trajectory txtutPr$\epsilon$,Ts of the PF ODE in Eq. (2), we define the consistency function as f : pxt, tq ÞÑ x$\epsilon$ . A consistency function has the property of self-consistency: its outputs are consistent for arbitrary pairs of pxt, tq that belong to the same PF ODE trajectory, i.e., fpxt, tq ( fpxt 1 , t1 q for all t, t1 P r$\epsilon$, Ts. With fixed time argument, fp¨, tq is always an invertible function. As illustrated in Fig. 2, the goal of a consistency model, symbolized as fθ, is to estimate this consistency function f from data by learning to enforce the self-consistency property (details in Sections 4 and 5).

定义给定方程中PF ODE的解轨迹txtutPr$\epsilon$，Ts。(2)，我们将一致性函数定义为f:pxt，tqÞñx$\epsiron$。一致性函数具有自一致性的性质：它的输出对于属于同一PF ODE轨迹的任意对pxt，tq是一致的，即fpxt，tq(fpxt 1，所有t的t1 q，t1 P r$\epsilon$，Ts。对于固定时间自变量fp¨，tq始终是一个可逆函数。如图6所示，2，一致性模型(符号为fθ)的目标是通过学习增广自一致性属性，从数据中估计该一致性函数f(详见第4节和第5节)。

Parameterization For any consistency function fp¨, ¨q, we have fpx$\epsilon$ , $\epsilon$q ( x$\epsilon$ , i.e., fp¨, $\epsilon$q is an identity function. We call this constraint the boundary condition. A valid consistency model has to respect this boundary condition. For consistency models based on deep neural networks, we discuss two ways to implement this boundary condition almost for free. Suppose we have a free-form deep neural network Fθpx, tq whose output has the same dimensionality as x. The first way is to simply parameterize the consistency model as fθpx, tq ( # x t ( $\epsilon$

参数化对于任何一致性函数fp¨，¨q，我们有fpx$\epsilon$，$\ε$q(x$\epsilon$，即fp¨，$\epsilon$q是一个恒等函数。我们将此约束称为边界条件。有效的一致性模型必须尊重此边界条件。对于基于深度神经网络的一致性建模，我们讨论了两种几乎免费实现此边界条件的方法。假设我们有一个自由形式的深度神经网络Fθpx，tq，其输出具有相同的维数nality为x。第一种方法是简单地将一致性模型参数化为fθpx，tq(#x t($\epsilon$

Fθpx, tq t P p$\epsilon$, Ts . (4)

The second method is to parameterize the consistency model using skip connections, that is,

第二种方法是使用跳过连接来参数化一致性模型，

fθpx, tq ( cskipptqx + coutptqFθpx, tq, (5)

where cskipptq and coutptq are differentiable functions such that cskipp$\epsilon$ q ( 1, and coutp$\epsilon$ q ( 0. This way, the consistency model is differentiable at t ( $\epsilon$ if Fθpx, tq and scaling coefficients are differentiable, which is critical for training continuous-time consistency models (Appendices B.1 and B.2). The parameterization in Eq. (5) bears strong resemblance to many successful diffusion models (Karras et al., 2022; Balaji et al., 2022), making it easier to borrow powerful diffusion model architectures for constructing consistency models. We therefore follow the second parameterization in all experiments.

其中cskiptq和couptq是可微函数，使得cskipp$\epsilon$q(1)和coutp$\epsilon$q(0。这样，一致性模型在t($\epsilon$，如果Fθpx、tq和缩放系数是可微的，这对于训练连续时间一致性模型至关重要(附录B.1和B.2)。(5)中的参数化与许多成功的扩散模型非常相似(Karras et al., 2022;Balaji et al., 2022)，使得更容易借用强大的扩散模型架构来构建一致性模型。因此，我们在所有实验中都遵循第二个参数化。

Sampling With a well-trained consistency model fθp¨, ¨q, we can generate samples by sampling from the initial distribution ˆxT „ N p0, T2Iq and then evaluating the consistency model for ˆx$\epsilon$ ( fθpˆxT , Tq. This involves only one forward pass through the consistency model and therefore

采样有了训练有素的一致性模型fθp¨，¨q，我们可以通过从初始分布中采样来生成样本，即对初始分布进行采样，然后评估对一致性模型的一致性建模(fθp对一致性建模仅进行一次正向传递，因此

Algorithm 1 Multistep Consistency Sampling

算法1多步一致性采样

generates samples in a single step. Importantly, one can also evaluate the consistency model multiple times by alternating denoising and noise injection steps for improved sample quality. Summarized in Algorithm 1, this multistep sampling procedure provides the flexibility to trade compute for sample quality. It also has important applications in zero-shot data editing. In practice, we find time points in Algorithm 1 with a greedy algorithm, where the time points are pinpointed one at a time using ternary search to optimize the FID of samples obtained from Algorithm 1.

在单个步骤中生成样本。重要的是，还可以通过交替的去噪和噪声注入步骤来多次评估一致性模型，以提高样本质量。在算法1中总结，这种多步骤采样过程提供了以计算换采样本质量的灵活性。它在零样本数据编辑中也有重要应用。在实践中，我们使用贪婪算法在算法1中找到时间点，其中使用三元搜索来优化从算法1获得的样本的FID，一次一个地精确定位时间点。

Zero-Shot Data Editing Consistency models enable various data editing and manipulation applications in zero shot; they do not require explicit training to perform these tasks. For example, consistency models define a one-to-one mapping from a Gaussian noise vector to a data sample. Similar to latent variable models like GANs, VAEs, and normalizing flows, consistency models can easily interpolate between samples by traversing the latent space (Fig. 11). As consistency models are trained to recover x$\epsilon$ from any noisy input xt where t P r$\epsilon$, Ts, they can perform denoising for various noise levels (Fig. 12). Moreover, the multistep generation procedure in Algorithm 1 is useful for solving certain inverse problems in zero shot by using an iterative replacement procedure similar to that of diffusion models (Song & Ermon, 2019; Song et al., 2021; Ho et al., 2022b). This enables many applications in the context of image editing, including inpainting (Fig. 10), colorization (Fig. 8), super-resolution (Fig. 6b) and stroke-guided image editing (Fig. 13) as in SDEdit (Meng et al., 2021). In Section 6.3, we empirically demonstrate the power of consistency models on many zero-shot image editing tasks.

零样本数据编辑一致性模型支持各种零样本数据编辑和操作应用程序;他们不需要明确的训练来执行这些任务。例如，一致性模型定义了从高斯噪声向量到数据样本的一对一映射。与GAN、VAE和归一化流等潜在变量模型类似，一致性模型可以通过遍历潜在空间来轻松地在样本之间进行插值(图11)。由于一致性模型被训练为从任何有噪声的输入xt中恢复x$\epsilon$，其中t P r$\epsilion$，Ts，它们可以对各种噪声水平执行去噪(图12)。此外，算法1中的多步生成过程有助于通过使用类似于扩散模型的迭代替换过程来解决零样本中的某些逆问题(Song&Ermon，2019;Song et al.，2021;Ho et al，2022b)。这使许多应用程序能够在图像编辑环境中进行，包括修补(图10)、彩色化(图8)、超分辨率(图6b)和笔划引导图像编辑(图13)，如SDEdit(Meng et al., 2021)。在第6.3节中，我们实证证明了一致性模型在许多零样本图像编辑任务中的威力。

## 4. Training Consistency Models via Distillatio  通过Distractio训练一致性模型
We present our first method for training consistency models based on distilling a pre-trained score model sφpx, tq. Our discussion revolves around the empirical PF ODE in Eq. (3), obtained by plugging the score model sφpx, tq into the PF ODE. Consider discretizing the time horizon r$\epsilon$, Ts into N - 1 sub-intervals, with boundaries t1 ( $\epsilon$ ă t2 ă ¨ ¨ ¨ ă tN = T. In practice, we follow Karras et al. (2022) to determine the boundaries with the formula ti ( p$\epsilon$ 1/ρ + i-1/N-1pT 1/ρ - $\epsilon$ 1/ρ qqρ , where ρ ( 7. When N is sufficiently large, we can obtain an accurate estimate of xtn from xtn+1 by running one discretization step of a numerical ODE solver. This estimate, which we denote as ˆx φ tn , is defined by ˆx φ tn :

我们提出了第一种基于提取预先训练的分数模型sφpx，tq来训练一致性模型的方法。我们的讨论围绕着方程中的经验PF ODE展开。(3)，通过将分数模型sφpx，tq插入PF ODE中获得。考虑将时间范围r$\epsilon$，Ts离散为N-1个子区间，边界为t1($\epsilon$ăt2ă¨¨ătN=T)。在实践中，我们遵循Karras等人(2022)，用公式ti(p$\epsi隆$1/ρ+i-1/N-1pT 1/ρ-$\epsiron$1/ρqqρ)确定边界，其中ρ(7)。当N足够大时，我们可以通过运行数值ODE求解器的一个离散化步骤，从xtn+1获得xtn的精确估计。这个估计，我们将其表示为：

( xtn+1 + ptn - tn+1qΦpxt

where Φp¨ ¨ ¨ ; φq represents the update function of a onestep ODE solver applied to the empirical PF ODE. For example, when using the Euler solver, we have Φpx, t; φq ( -tsφpx, tq which corresponds to the following update rule

其中Φp¨¨;φq表示应用于经验PF ODE的一步ODE求解器的更新函数。例如，当使用欧拉解算器时，我们有Φpx，t;φq(-tsφpx，tq，对应于以下更新规则

ˆx φ tn ( xtn+1 - ptn - tn+1qtn+1sφpxtn+1 , tn+1q.

For simplicity, we only consider one-step ODE solvers in this work. It is straightforward to generalize our framework to multistep ODE solvers and we leave it as future work.

为了简单起见，我们在这项工作中只考虑一步ODE解算器。将我们的框架推广到多步骤ODE求解器是很简单的，我们将其作为未来的工作。

Due to the connection between the PF ODE in Eq. (2) and the SDE in Eq. (1) (see Section 2), one can sample along the distribution of ODE trajectories by first sampling x „ pdata, then adding Gaussian noise to x. Specifically, given a data point x, we can generate a pair of adjacent data points pˆx φ tn , xtn+1 q on the PF ODE trajectory efficiently by sampling x from the dataset, followed by sampling xtn+1 from the transition density of the SDE N px, t2 n+1Iq, and then computing ˆx φ tn using one discretization step of the numerical ODE solver according to Eq. (6). Afterwards, we train the consistency model by minimizing its output differences on the pair pˆx φ tn , xtn+1 q. This motivates our following consistency distillation loss for training consistency models.

由于方程(2)中的PF ODE和方程(1)中的SDE之间的联系。(1)(见第2节)，可以通过首先采样x“pdatea，然后向x添加高斯噪声来沿着ODE轨迹的分布进行采样。具体而言，给定数据点x，我们可以通过从数据集中采样x来有效地在PF ODE轨迹上生成一对相邻的数据点põxφtn，xtn+1 q，然后从SDE N px的跃迁密度中采样xtn+1，t2 N+1Iq，然后根据等式使用数值ODE解算器的一个离散化步骤来计算ξxφtn。(6)。然后，我们通过最小化一致性模型在对põxφtn，xtn+1 q上的输出差异来训练一致性模型。这激发了我们训练一致性建模的以下一致性蒸馏损失。

Definition 1. The consistency distillation loss is defined as

定义1。稠度蒸馏损失定义为

L N CDpθ, θ -; φq :( Erλptnqdpfθpxtn+1 , tn+1q, fθ- pˆx φ tn , tnqqs, (7)

where the expectation is taken with respect to x „ pdata, n „ UJ 1, N -1K , and xtn+1 „ N px;t 2 n+1Iq. Here UJ 1, N -1K denotes the uniform distribution over t1, 2, ¨ ¨ ¨ , N - 1u, λp¨q P R + is a positive weighting function, ˆx φ tn is given by Eq. (6), θ - denotes a running average of the past values of θ during the course of optimization, and dp¨, ¨q is a metric function that satisfies @x, y : dpx, yq ě 0 and d(x, y)= 0 if and only if x = y.

其中，期望值是关于x“updatea”、n“UJ 1”、n-1K和xtn+1“n px;t 2 n+1数量。其中，UJ1，N-1K表示t1，2，¨¨¨，N-1u，λp¨q p R+是一个正加权函数，由等式给出。(6)，θ-表示优化过程中θ的过去值的运行平均值，dp¨，¨q是一个满足@x，y:dpx，yqŞ0和d(x，y)=0的度量函数，当且仅当x=y。

Unless otherwise stated, we adopt the notations in Defi- nition 1 throughout this paper, and use Er¨s to denote the expectation over all relevant random variables. In our experiments, we consider the squared + 2 distance dpx, yq ( }x - y} 2 2 , + 1 distance dpx, yq ( }x - y}1, and the Learned Perceptual Image Patch Similarity (LPIPS, Zhang et al. (2018)). We find λptnq ” 1 performs well across all

除非另有说明，否则我们在本文中采用定义1中的符号，并使用Er表示对所有相关随机变量的期望。在我们的实验中，我们考虑了平方+2距离dpx，yq(}x-y}2，+1距离dpx、yq({x-y}1)和习得感知图像分块相似性(LPIPS，Zhang et al.(2018))。我们发现λptnq“1在所有情况下都表现良好

Algorithm 2 Consistency Distillation (CD)

算法2稠度蒸馏(CD)

tasks and datasets. In practice, we minimize the objective by stochastic gradient descent on the model parameters θ, while updating θ - with exponential moving average (EMA). That is, given a decay rate 0 ď µ ă 1, we perform the following update after each optimization step:

任务和数据集。在实践中，我们通过模型参数θ的随机梯度下降来最小化目标，同时用指数移动平均(EMA)更新θ-。即,给定衰减率0ďµă1，我们在每个优化步骤后执行以下更新：

θ - Ð stopgradpµθ - + p1 - µqθq. (8)

The overall training procedure is summarized in Algorithm 2. In alignment with the convention in deep reinforcement learning (Mnih et al., 2013; 2015; Lillicrap et al., 2015) and momentum based contrastive learning (Grill et al., 2020; He et al., 2020), we refer to fθ- as the (target network”, and fθ as the (online network”. We find that compared to simply setting θ - ( θ, the EMA update and (stopgrad” operator in Eq. (8) can greatly stabilize the training process and improve the final performance of the consistency model.

算法2中总结了整个训练过程。与深度强化学习的惯例一致(Mnih et al., 2013;2015;Lillicrap et al., 2015)和基于动量的对比学习(Grill et al., 2020;He et al., 2020)，我们将fθ-称为“目标网络”，将fθ称为“在线网络”。我们发现，与简单地设置θ-(θ)相比，方程中的EMA更新和(stopgrad)算子。(8)可以极大地稳定训练过程，提高一致性模型的最终性能。

Below we provide a theoretical justification for consistency distillation based on asymptotic analysis.

下面我们提供了基于渐近分析的一致性蒸馏的理论依据。

Theorem 1. Let ∆t :( maxnPJ 1,N-1K t|tn+1 - tn|u, and fp¨, ¨; φq be the consistency function of the empirical PF ODE in Eq. (3). Assume fθ satisfies the Lipschitz condition: there exists L ą 0 such that for all t P r$\epsilon$, Ts, x, and y, we have k fθpx, tq - fθpy, tqk 2 ď Lk x - yk 2 . Assume further that for all n P J 1, N - 1K , the ODE solver called at tn+1 has local error uniformly bounded by Opptn+1 - tnq p+1 q with p ě 1. Then, if L N CDpθ, θ; φq ( 0, we have

定理1。设∆t:(maxnPJ 1，N-1K t|tn+1-tn|u，以及fp¨，¨;φq为公式(3)中经验PF ODE的一致性函数。假设fθ满足Lipschitz条件：存在Lã0，使得对于所有的t P r$\epsilon$，Ts，x和y，我们有k fθpx，tq-fθpy，tqk2ďLk x-yk2。进一步假设，对于所有n个P J 1，n-1K，在tn+1处调用的ODE解算器具有由Opptn+1-tnq P+1 q一致地定界的局部误差，其中PŞ1。那么，如果L N CDpθ，θ;φq(0，我们有

sup n,x }fθpx, tnq - fpx, tn; φq}2 ( Opp∆tq p q.

Proof. The proof is based on induction and parallels the classic proof of global error bounds for numerical ODE solvers (S¨uli & Mayers, 2003). We provide the full proof in Appendix A.2.

证据该证明基于归纳，与数值ODE解算器的全局误差界的经典证明相似(S¨uli&Mayers，2003)。我们在附录A.2中提供了完整的证据。

Since θ - is a running average of the history of θ, we have θ - ( θ when the optimization of Algorithm 2 converges.

由于θ-是θ历史的运行平均值，因此当算法2的优化收敛时，我们有θ-(θ。

Algorithm 3 Consistency Training (CT)

算法3一致性训练(CT)

That is, the target and online consistency models will eventually match each other. If the consistency model additionally achieves zero consistency distillation loss, then Theorem 1 implies that, under some regularity conditions, the estimated consistency model can become arbitrarily accurate, as long as the step size of the ODE solver is sufficiently small.

即,目标和在线一致性模型最终将相互匹配。如果一致性模型额外实现了零一致性蒸馏损失，那么定理1意味着，在一些正则性条件下，只要ODE求解器的步长足够小，估计的一致性模型就可以变得任意精确。

The consistency distillation loss L N CDpθ, θ -; φq can be extended to hold for infinitely many time steps (N Ñ 8) if θ - ( θ or θ - ( stopgradpθq. The resulting continuoustime loss functions do not require specifying N nor the time steps tt1, t2, ¨ ¨ ¨ , tN u. Nonetheless, they involve Jacobianvector products and require forward-mode automatic differentiation for efficient implementation, which may not be well-supported in some deep learning frameworks. We provide these continuous-time distillation loss functions in Theorems 3 to 5, and relegate details to Appendix B.1.

稠度蒸馏损失L N CDpθ，θ-;如果θ-(θ或θ-(stopgradpθq)，φq可以扩展为保持无限多个时间步长(Nñ8)。由此产生的连续时间损失函数不需要指定N，也不需要指定时间步长tt1、t2、¨¨¨¨、tN u。尽管如此，它们涉及雅可比向量产品，并且需要前向模式的自动区分才能有效实现，这在一些深度学习框架中可能得不到很好的支持。我们在定理3到5中提供了这些连续时间蒸馏损失函数，并将详情放在附录B.1中。

## 5. Training Consistency Models in Isolation  孤立地训练一致性模型
Consistency models can be trained without relying on any pre-trained diffusion models. This differs from diffusion distillation techniques, making consistency models a new independent family of generative models.

可以在不依赖于任何预先训练的扩散模型的情况下训练一致性模型。这与扩散蒸馏技术不同，使一致性模型成为一个新的独立生成模型家族。

In consistency distillation, we use a pre-trained score model sφpx, tq to approximate the ground truth score function ∇ log ptpxq. To get rid of this dependency, we need to seek other ways to estimate the score function. In fact, there exists an unbiased estimator of ∇ log ptpxtq due to the following identity (Lemma 1 in Appendix A):

在一致性提取中，我们使用预先训练的分数模型sφpx，tq来近似基本事实分数函数Şlog ptpxq。为了摆脱这种依赖，我们需要寻找其他方法来估计分数函数。事实上，由于以下恒等式(附录A中的引理1)，存在一个对log ptpxtq的无偏估计量：

∇ log ptpxtq ( -E „ xt - x t 2 ˇ ˇ ˇ ˇ xt  ,

对数ptpxtq(-E·xt-x t2) ,

where x „ pdata and xt „ N px;t 2Iq. That is, given x and xt, we can form a Monte Carlo estimate of ∇ log ptpxtq with -pxt - xq/t 2 . We now show that this estimate actually suffices to replace the pre-trained diffusion model in Consistency Models consistency distillation, when using the Euler method (or any higher order method) as the ODE solver in the limit of N Ñ 8.

其中x“updatea”和xt“N px;t 2质量。即,在给定x和xt的情况下，我们可以用-pxt-xq/t2形成一个Γlog ptpxtq的蒙特卡罗估计。我们现在表明，当使用欧拉方法(或任何更高阶方法)作为Nñ8极限下的ODE解算器时，该估计实际上足以取代一致性模型一致性蒸馏中预先训练的扩散模型。

More precisely, we have the following theorem.

更确切地说，我们有以下定理。

Theorem 2. Let ∆t :( maxnPJ 1,N-1K t|tn+1 - tn|u. Assume d and fθ- are both twice continuously differentiable with bounded second derivatives, the weighting function λp¨q is bounded, and Erk∇ log ptn pxtn qk 2 2 s ă 8. Assume further that we use the Euler ODE solver, and the pre-trained score model matches the ground truth, i.e., @t P r$\epsilon$, Ts : sφpx, tq ” ∇ log ptpxq. Then,

定理2。设∆t:(maxnPJ 1，N-1K t|tn+1-tn|u)。假设d和fθ-都是具有有界二阶导数的两次连续可微，加权函数λp¨q是有界的，ErkŞlog ptn pxtn qk 2 2 să8。进一步假设我们使用Euler ODE解算器，并且预训练的分数模型与基本事实相匹配，即@t P r$\epsilon$，Ts:sφpx，tq“Şlog ptpxq。然后

L N CDpθ, θ -; φq ( L N CTpθ, θ -q + op∆tq, (9)

where the expectation is taken with respect to x „ pdata, n „ UJ 1, N - 1K , and xtn+1 „ N px;t 2 n+1Iq. The consistency training objective, denoted by L N CTpθ, θ -q, is defined as

其中，期望值是关于x“updatea”、n“UJ 1”、n-1K和xtn+1“n px;t 2 n+1数量。一致性训练目标，用L N CTpθ，θ-q表示，定义为

Erλptnqdpfθpx + tn+1z, tn+1q, fθ- px + tnz, tnqqs, (10)

where z „ N p0, Iq. Moreover, L N CTpθ, θ -q ě Op∆tq if infN L N CDpθ, θ -; φq ą 0.

其中z“N p0，Iq。此外，L N CTpθ，θ-qŞOp∆tq，如果infN L N CDpθ，thet-;φqń0。

Proof. The proof is based on Taylor series expansion and properties of score functions (Lemma 1). A complete proof is provided in Appendix A.3.

证据该证明基于泰勒级数展开和分数函数的属性(引理1)。附录A.3中提供了完整的证明。

We refer to Eq. (10) as the consistency training (CT) loss. Crucially, Lpθ, θ -q only depends on the online network fθ, and the target network fθ- , while being completely agnostic to diffusion model parameters φ. The loss function Lpθ, θ -q ě Op∆tq decreases at a slower rate than the remainder op∆tq and thus will dominate the loss in Eq. (9) as N Ñ 8 and ∆t Ñ 0.

我们将等式(10)称为一致性训练(CT)损失。至关重要的是，Lpθ，θ-q仅取决于在线网络fθ和目标网络fθ-，而对扩散模型参数φ完全未知。损失函数Lpθ，θ-qŞOp∆tq的下降速度慢于余数Op∆tq的下降速度，因此将主导方程中的损失。(9)作为Nñ8和∆tñ0。

For improved practical performance, we propose to progressively increase N during training according to a schedule function Np¨q. The intuition (cf ., Fig. 3d) is that the consistency training loss has less (variance” but more (bias” with respect to the underlying consistency distillation loss (i.e., the left-hand side of Eq. (9)) when N is small (i.e., ∆t is large), which facilitates faster convergence at the beginning of training. On the contrary, it has more (variance” but less (bias” when N is large (i.e., ∆t is small), which is desirable when closer to the end of training. For best performance, we also find that µ should change along with N, according to a schedule function µp¨q. The full algorithm of consistency training is provided in Algorithm 3, and the schedule functions used in our experiments are given in Appendix C.

为了提高实际表现，我们建议在训练期间根据时间表函数Np–q逐步增加N。直觉(cf.，图3d)是，当N小(即∆t大)时，一致性训练损失相对于潜在的一致性蒸馏损失(即方程(9)的左侧)具有较小的(方差)但较多的(偏差)，这有助于在训练开始时更快地收敛。相反，当N较大(即∆t较小)时，它具有更多的(方差)但较少的(偏差)，这在接近训练结束时是可取的。为了获得最佳性能，我们还发现，根据时间表函数µp¨q，µ应该随N而变化。算法3中提供了一致性训练的完整算法，附录C中给出了我们实验中使用的调度函数。

Similar to consistency distillation, the consistency training loss L N CT (i.e., N pθ, θ -q can be extended to hold in continuous time Ñ 8) if θ - ( stopgradpθq, as shown in Theorem 6. This continuous-time loss function does not require schedule functions for N or µ, but requires forward-mode automatic differentiation for efficient implementation. Unlike the discrete-time CT loss, there is no undesirable (bias” associated with the continuous-time objective, as we effectively take ∆ Ñ 0 in Theorem 2. We relegate more details to Appendix B.2.

与一致性蒸馏类似，如果θ-(stopgradpθq，如定理6所示，则一致性训练损失L N CT(即，N pθ，θ-q可以扩展为在连续时间内保持)。这种连续时间损失函数不需要N或µ的调度函数，但需要前向模式自动微分才能有效执行。与离散时间CT损失不同，不存在与连续时间目标相关的不良(偏差)，因为我们在定理2中有效地取∆ñ0。我们将更多细节放在附录B.2中。

## 6. Experiments  实验
We employ consistency distillation and consistency training to learn consistency models on real image datasets, including CIFAR-10 (Krizhevsky et al., 2009), ImageNet 64 ˆ 64 (Deng et al., 2009), LSUN Bedroom 256 ˆ 256, and LSUN Cat 256 ˆ 256 (Yu et al., 2015). Results are compared according to Fr-echet Inception Distance (FID, Heusel et al. (2017), lower is better), Inception Score (IS, Salimans et al. (2016), higher is better), Precision (Prec., Kynk¨a¨anniemi et al. (2019), higher is better), and Recall (Rec., Kynk¨a¨anniemi et al. (2019), higher is better). Additional experimental details are provided in Appendix C.

我们使用一致性蒸馏和一致性训练来学习真实图像数据集上的一致性模型，包括CIFAR-10(Krizhevsky et al.，2009)、ImageNet 64 v2。根据Fr-echet-Inception Distance(FID，Heusel et al.(2017)，越低越好)、Inception Score(is，Salimans et al.(2016)，越高越好)、Precision(Prec.，Kynk¨a¨anniemi et al.(2019)，越高越好)和Recall(Rec.，Kynk´a¨annemi et al.(2019.，越高越好)对结果进行比较。附录C中提供了额外的实验细节。

### 6.1. Training Consistency Models  训练一致性模型
We perform a series of experiments on CIFAR-10 to understand the effect of various hyperparameters on the performance of consistency models trained by consistency distillation (CD) and consistency training (CT). We first focus on the effect of the metric function dp¨, ¨q, the ODE solver, and the number of discretization steps N in CD, then investigate the effect of the schedule functions Np¨q and µp¨q in CT.

我们在CIFAR-10上进行了一系列实验，以了解各种超参数对通过一致性蒸馏(CD)和一致性训练(CT)训练的一致性模型性能的影响。我们首先关注度量函数dp¨，¨q、ODE求解器和CD中离散化步骤数N的影响，然后研究调度函数Np¨q和µp¨q在CT中的影响。

To set up our experiments for CD, we consider the squared + 2 distance dpx, yq ( }x - y} 2 2 , + 1 distance dpx, yq ( }x - y}1, and the Learned Perceptual Image Patch Similarity (LPIPS, Zhang et al. (2018)) as the metric function. For the ODE solver, we compare Euler’s forward method and Heun’s second order method as detailed in Karras et al. (2022). For the number of discretization steps N, we compare N P t9, 12, 18, 36, 50, 60, 80, 120u. All consistency models trained by CD in our experiments are initialized with the corresponding pre-trained diffusion models, whereas models trained by CT are randomly initialized.

为了建立我们的CD实验，我们考虑平方+2距离dpx、yq(}x-y}2，+1距离dpx，yq({x-y}1)和习得感知图像分块相似性(LPIPS，Zhang et al.(2018))作为度量函数。对于ODE解算器，我们比较了欧拉正演方法和Heun二阶方法，如Karras等人所述。(2022)。对于离散化步骤的数量N，我们比较N P t9、12、18、36、50、60、80、120u。在我们的实验中，CD训练的所有一致性模型都用相应的预先训练的扩散模型初始化，而CT训练的模型是随机初始化的。

As visualized in Fig. 3a, the optimal metric for CD is LPIPS, which outperforms both + 1 and + 2 by a large margin over all training iterations. This is expected as the outputs of consistency models are images on CIFAR-10, and LPIPS is specifically designed for measuring the similarity between natural images. Next, we investigate which ODE solver and which discretization step N work the best for CD. As shown in Figs. 3b and 3c, Heun ODE solver and N = 18 are the best choices. Both are in line with the recommendation of Karras et al. (2022) despite the fact that we are training consistency models, not diffusion models. Moreover, Fig. 3b shows that with the same N, Heun’s second order solver uniformly outperforms Euler’s first order solver. This corroborates with Theorem 1, which states that the optimal consistency models trained by higher order ODE solvers have smaller estimation errors with the same N. The results of Fig. 3c also indicate that once N is sufficiently large, the performance of CD becomes insensitive to N. Given these insights, we hereafter use LPIPS and Heun ODE solver for CD unless otherwise stated. For N in CD, we follow the suggestions in Karras et al. (2022) on CIFAR-10 and ImageNet 64 ˆ 64. We tune N separately on other datasets (details in Appendix C).

如图6所示，如图3a所示，CD的最佳度量是LPIPS，它在所有训练迭代中都以很大的优势优于+1和+2。这是意料之中的，因为一致性模型的输出是CIFAR-10上的图像，而LPIPS是专门为测量自然图像之间的相似性而设计的。接下来，我们研究哪种ODE解算器和哪种离散化步骤N最适合CD。如图3b和3c所示，Heun ODE解器和N=18是最佳选择。两者都符合Karras等人的建议。(2022)尽管我们正在训练一致性模型，而不是扩散模型。此外，图3b显示，在相同的N下，Heun的二阶解算器均匀地优于Euler的一阶解算。这与定理1相证实，定理1指出，由高阶ODE解算器训练的最优一致性模型在相同的N下具有较小的估计误差。图3c的结果还表明，一旦N足够大，CD的性能就对N不敏感。鉴于这些见解，除非另有说明，否则我们在下文中对CD使用LPIPS和Heun ODE解器。对于CD中的N，我们遵循Karras等人(2022)关于CIFAR-10和ImageNet 64的建议。我们在其他数据集上分别调整N(详情见附录C)。

Figure 3: Various factors that affect consistency distillation (CD) and consistency training (CT) on CIFAR-10. The best configuration for CD is LPIPS, Heun ODE solver, and N = 18. Our adaptive schedule functions for N and µ make CT converge significantly faster than fixing them to be constants during the course of optimization.
图3：影响CIFAR-10一致性蒸馏(CD)和一致性训练(CT)的各种因素。CD的最佳配置是LPIPS、Heun ODE解算器和N=18。我们针对N和µ的自适应调度函数使CT的收敛速度明显快于在优化过程中将它们固定为常数。

Figure 4: Multistep image generation with consistency distillation (CD). CD outperforms progressive distillation (PD) across all datasets and sampling steps. The only exception is single-step generation on Bedroom 256 ˆ 256.
图4：具有稠度蒸馏(CD)的多步骤图像生成。CD在所有数据集和采样步骤中都优于渐进蒸馏(PD)。唯一的例外是在Bedroom 256上的单步生成。

Due to the strong connection between CD and CT, we adopt LPIPS for our CT experiments throughout this paper. Unlike CD, there is no need for using Heun’s second order solver in CT as the loss function does not rely on any particular numerical ODE solver. As demonstrated in Fig. 3d, the convergence of CT is highly sensitive to N—smaller N leads to faster convergence but worse samples, whereas larger N leads to slower convergence but better samples upon convergence. This matches our analysis in Section 5, and motivates our practical choice of progressively growing N and µ for CT to balance the trade-off between convergence speed and sample quality. As shown in Fig. 3d, adaptive schedules of N and µ significantly improve the convergence speed and sample quality of CT. In our experiments, we tune the schedules Np¨q and µp¨q separately for images of different resolutions, with more details in Appendix C.

由于CD和CT之间的紧密联系，我们在本文的CT实验中采用了LPIPS。与CD不同，在CT中不需要使用Heun的二阶解算器，因为损失函数不依赖于任何特定的数值ODE解算器。如图6所示，如图3d所示，CT的收敛对N高度敏感。较小的N导致更快的收敛但较差的样本，而较大的N导致较慢的收敛但在收敛时更好的样本。这与我们在第5节中的分析相匹配，并促使我们为CT选择逐渐增长的N和µ，以平衡收敛速度和样本质量之间的权衡。如图3d所示，N和µ的自适应调度显著提高了CT的收敛速度和样本质量。在我们的实验中，我们针对不同分辨率的图像分别调整了调度Np¨q和µp¨q，更多细节见附录C。

### 6.2. Few-Step Image Generation  少步图像生成
Distillation In current literature, the most directly comparable approach to our consistency distillation (CD) is progressive distillation (PD, Salimans & Ho (2022)); both are thus far the only distillation approaches that do not construct synthetic data before distillation. In stark contrast, other distillation techniques, such as knowledge distillation (Luhman & Luhman, 2021) and DFNO (Zheng et al., 2022), have to prepare a large synthetic dataset by generating numerous samples from the diffusion model with expensive numerical ODE solvers. We perform comprehensive comparison between PD and CD on CIFAR-10, ImageNet 64 ˆ 64, and LSUN 256 ˆ 256, with all results reported in Fig. 4. All methods distill from an EDM (Karras et al., 2022) model that we pre-trained in-house. We note that across all sampling iterations, using the LPIPS metric uniformly improves PD compared to the squared + 2 distance in the original paper of Salimans & Ho (2022). Both PD and CD improve as we take more sampling steps. We find that CD uniformly outperforms PD across all datasets, sampling steps, and metric functions considered, except for single-step generation on Bedroom 256 ˆ 256, where CD with + 2 slightly underperforms PD with + 2. As shown in Table 1, CD even outperforms distillation approaches that require synthetic dataset construction, such as Knowledge Distillation (Luhman & Luhman, 2021) and DFNO (Zheng et al., 2022).

蒸馏在目前的文献中，与我们的稠度蒸馏(CD)最直接可比的方法是渐进蒸馏(PD，Salimans&Ho(2022));到目前为止，这两种方法都是唯一不在蒸馏前构建合成数据的蒸馏方法。与此形成鲜明对比的是，其他蒸馏技术，如知识蒸馏(Luhman&Luhman，2021)和DFNO(Zheng et al., 2022年)，必须通过使用昂贵的数值常微分方程求解器从扩散模型生成大量样本来制备大型合成数据集。我们在CIFAR-10、ImageNet 64和LSUN 256上对PD和CD进行了全面比较，所有结果如图4所示。所有方法都是从我们内部预先训练的EDM(Karras et al.，2022)模型中提取的。我们注意到，在所有采样迭代中，与Salimans&Ho(2022)的原始论文中的平方+2距离相比，使用LPIPS度量均匀地提高了PD。随着我们采取更多的采样步骤，PD和CD都有所改善。我们发现，CD在所有数据集、采样步骤和所考虑的度量函数中都一致优于PD，除了Bedroom 256的单步生成，其中+2的CD略低于+2的PD。如表1所示，CD甚至优于需要构建合成数据集的蒸馏方法，例如知识蒸馏(Luhman&Luhman，2021)和DFNO(Zheng et al., 2022年)。

Table 1: Sample quality on CIFAR-10. ˚Methods that require synthetic data construction for distillation.
表1:CIFAR-10的样品质量。˚蒸馏需要合成数据构建的方法。

Table 2: Sample quality on ImageNet 64 ˆ 64, and LSUN Bedroom & Cat 256 ˆ 256. :Distillation techniques.
表2：ImageNet 64和LSUN Bedroom&Cat 256的样本质量：蒸馏技术。

Figure 5: Samples generated by EDM (top), CT + single-step generation (middle), and CT + 2-step generation (Bottom). All corresponding images are generated from the same initial noise.
图5：EDM生成的样本(顶部)、CT+单步生成(中间)和CT+两步生成(底部)。所有对应的图像都是从相同的初始噪声中生成的。

Consistency Models (a) Left: The gray-scale image. Middle: Colorized images. Right: The ground-truth image. (b) Left: The downsampled image (32 ˆ 32). Middle: Full resolution images (256 ˆ 256). Right: The ground-truth image (256 ˆ 256). (c) Left: A stroke input provided by users. Right: Stroke-guided image generation.

一致性模型(a)左图：灰度图像。中间：彩色图像。右图：基准实况图像。(b) 左图：下采样图像(32？32)。中间：全分辨率图像(256色256)。右图：基准实况图像(256色256)。(c) 左图：用户提供的笔画输入。右图：笔划引导的图像生成。

Figure 6: Zero-shot image editing with a consistency model trained by consistency distillation on LSUN Bedroom 256ˆ256.
图6：在LSUN Bedroom 256ˆ256上使用一致性蒸馏训练的一致性模型进行零样本图像编辑。

Direct Generation In Tables 1 and 2, we compare the sample quality of consistency training (CT) with other generative models using one-step and two-step generation. We also include PD and CD results for reference. Both tables report PD results obtained from the + 2 metric function, as this is the default setting used in the original paper of Salimans & Ho (2022). For fair comparison, we train both PD and CD to distill the same EDM models. In Tables 1 and 2, we observe that CT outperforms all single-step, non-adversarial generative models, i.e., VAEs and normalizing flows, by a significant margin on CIFAR-10. Moreover, CT obtains comparable quality to PD for single-step generation without relying on distillation. In Fig. 5, we provide EDM samples (top), single-step CT samples (middle), and two-step CT samples (bottom). In Appendix E, we show additional samples for both CD and CT in Figs. 14 to 21. Importantly, all samples obtained from the same initial noise vector share significant structural similarity, even though CT and EDM models are trained independently from one another. This indicates that CT is unlikely to suffer from mode collapse, as EDMs do not.

直接生成在表1和表2中，我们将一致性训练(CT)的样本质量与其他使用一步和两步生成的生成模型进行了比较。我们还包括PD和CD结果以供参考。两个表都报告了从+2度量函数获得的PD结果，因为这是Salimans&Ho(2022)的原始论文中使用的默认设置。为了进行公平的比较，我们训练PD和CD来提取相同的EDM模型。在表1和表2中，我们观察到CT在CIFAR-10上显著优于所有单步非对抗性生成模型，即VAE和归一化流。此外，在不依赖蒸馏的情况下，对于单步生成，CT可以获得与PD相当的质量。在图5中，我们提供了EDM样本(顶部)、单步CT样本(中间)和两步CT样本(底部)。在附录E中，我们在图14至21中显示了CD和CT的额外样本。重要的是，从相同的初始噪声向量获得的所有样本都具有显著的结构相似性，即使CT和EDM模型是相互独立训练的。这表明CT不太可能遭受模式崩溃，因为EDM不会。

### 6.3. Zero-Shot Image Editing  零样本图像编辑
Similar to diffusion models, consistency models allow zeroshot image editing by modifying the multistep sampling process in Algorithm 1. We demonstrate this capability with a consistency model trained on the LSUN bedroom dataset using consistency distillation. In Fig. 6a, we show such a consistency model can colorize gray-scale bedroom images at test time, even though it has never been trained on colorization tasks. In Fig. 6b, we show the same consistency model can generate high-resolution images from low-resolution inputs. In Fig. 6c, we additionally demonstrate that it can generate images based on stroke inputs created by humans, as in SDEdit for diffusion models (Meng et al., 2021). Again, this editing capability is zero-shot, as the model has not been trained on stroke inputs. In Appendix D, we additionally demonstrate the zero-shot capability of consistency models on inpainting (Fig. 10), interpolation (Fig. 11) and denoising (Fig. 12), with more examples on colorization (Fig. 8), super-resolution (Fig. 9) and stroke-guided image generation (Fig. 13).

与扩散模型类似，一致性模型允许通过修改算法1中的多步采样过程进行零拍摄图像编辑。我们通过使用一致性蒸馏在LSUN卧室数据集上训练的一致性模型来证明这种能力。在图6中，6a，我们展示了这样一个一致性模型可以在测试时对灰度卧室图像进行着色，尽管它从未在着色任务中进行过训练。在图6b中，我们展示了相同的一致性模型可以从低分辨率输入生成高分辨率图像。在图6c中，我们还证明了它可以根据人类创建的笔划输入生成图像，如扩散模型的SDEdit(Meng et al., 2021)。同样，这种编辑功能是零样本，因为模型没有经过笔划输入训练。在附录D中，我们还演示了一致性模型在修复(图10)、插值(图11)和去噪(图12)方面的零样本能力，并提供了更多关于彩色化(图8)、超分辨率(图9)和笔划引导图像生成(图13)的样本。

## 7. Conclusion  结论
We have introduced consistency models, a type of generative models that are specifically designed to support one-step and few-step generation. We have empirically demonstrated that our consistency distillation method outshines the existing distillation techniques for diffusion models on multiple image benchmarks and various sampling iterations. Furthermore, as a standalone generative model, consistency models outdo other available models that permit single-step generation, barring GANs. Similar to diffusion models, they also allow zero-shot image editing applications such as inpainting, colorization, super-resolution, denoising, interpolation, and stroke-guided image generation.

我们引入了一致性模型，这是一种专门为支持一步和少步生成而设计的生成模型。我们已经实证证明，我们的一致性蒸馏方法在多个图像基准和各种采样迭代上的扩散模型的现有蒸馏技术中相形见绌。此外，作为一个独立的生成模型，一致性模型优于其他允许单步生成的可用模型，排除了GANs。与扩散模型类似，它们还允许使用零样本图像编辑应用程序，如修复、着色、超分辨率、去噪、插值和笔划引导图像生成。

In addition, consistency models share striking similarities with techniques employed in other fields, including deep Q-learning (Mnih et al., 2015) and momentum-based contrastive learning (Grill et al., 2020; He et al., 2020). This offers exciting prospects for cross-pollination of ideas and methods among these diverse fields.

此外，一致性模型与其他领域使用的技术有着惊人的相似之处，包括深度Q学习(Mnih et al., 2015)和基于动量的对比学习(Grill et al., 2020;He et al., 2020)。这为这些不同领域的思想和方法的交叉授粉提供了令人兴奋的前景。

## Acknowledgements 鸣谢
We thank Alex Nichol for reviewing the manuscript and providing valuable feedback, Chenlin Meng for providing stroke inputs needed in our stroke-guided image generation experiments, and all members of the OpenAI Algorithms team for helpful suggestions during the project.

我们感谢Alex Nichol审阅了手稿并提供了宝贵的反馈，感谢Chenlin Meng在我们的笔划引导图像生成实验中提供了所需的笔划输入，感谢OpenAI算法团队的所有成员在项目期间提供了有用的建议。

## References
* Balaji, Y., Nah, S., Huang, X., Vahdat, A., Song, J., Kreis,K., Aittala, M., Aila, T., Laine, S., Catanzaro, B., Karras, T., and Liu, M.-Y. ediff-i: Text-to-image diffusionmodels with ensemble of expert denoisers. arXiv preprintarXiv:2211.01324, 2022.
* Bengio, Y. and Bengio, S. Modeling high-dimensionaldiscrete data with multi-layer neural networks. Advancesin Neural Information Processing Systems, 12, 1999.
* Brock, A., Donahue, J., and Simonyan, K. Large scaleGAN training for high fidelity natural image synthesis. InInternational Conference on Learning Representations,2019. URL https://openreview.net/forum?id=B1xsqj09Fm.
* Chen, N., Zhang, Y., Zen, H., Weiss, R. J., Norouzi, M., andChan, W. Wavegrad: Estimating gradients for waveformgeneration. In International Conference on LearningRepresentations (ICLR), 2021.
* Chen, R. T., Behrmann, J., Duvenaud, D. K., and Jacobsen,J.-H. Residual flows for invertible generative modeling. In Advances in Neural Information Processing Systems,pp. 9916–9926, 2019a.
* Chen, T., Zhai, X., Ritter, M., Lucic, M., and Houlsby,N. Self-supervised gans via auxiliary rotation loss. InProceedings of the IEEE/CVF conference on computervision and pattern recognition, pp. 12154–12163, 2019b.
* Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., and Ye,J. C. Diffusion posterior sampling for general noisy inverse problems. In International Conference on LearningRepresentations, 2023. URL https://openreview.net/forum?id=OnD9zGAGT0k.
* Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei,L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and patternrecognition, pp. 248–255. Ieee, 2009.
* Dhariwal, P. and Nichol, A. Diffusion models beat ganson image synthesis. Advances in Neural InformationProcessing Systems (NeurIPS), 2021.
* Dinh, L., Krueger, D., and Bengio, Y. NICE: Non-linearindependent components estimation. International Conference in Learning Representations Workshop Track,2015.
* Dinh, L., Sohl-Dickstein, J., and Bengio, S. Density estimation using real NVP. In 5th International Conference on Learning Representations, ICLR 2017, Toulon,France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https://openreview.net/forum?id=HkpbnH9lx.
* Dockhorn, T., Vahdat, A., and Kreis, K. Genie: Higherorder denoising diffusion solvers. arXiv preprintarXiv:2210.05475, 2022.
* Gong, X., Chang, S., Jiang, Y., and Wang, Z. Autogan:Neural architecture search for generative adversarial networks. In Proceedings of the IEEE/CVF InternationalConference on Computer Vision, pp. 3224–3234, 2019.
* Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B.,Warde-Farley, D., Ozair, S., Courville, A., and Bengio,Y. Generative adversarial nets. In Advances in neuralinformation processing systems, pp. 2672–2680, 2014.
* Grci´c, M., Grubiˇsi´c, I., and ˇSegvi´c, S. Densely connectednormalizing flows. Advances in Neural Information Processing Systems, 34:23968–23982, 2021.
* Grill, J.-B., Strub, F., Altch´e, F., Tallec, C., Richemond, P.,Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z.,Gheshlaghi Azar, M., et al. Bootstrap your own latent-anew approach to self-supervised learning. Advances inneural information processing systems, 33:21271–21284,2020.
* He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representationlearning. In Proceedings of the IEEE/CVF conference oncomputer vision and pattern recognition, pp. 9729–9738,2020.
* Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., andHochreiter, S. GANs trained by a two time-scale updaterule converge to a local Nash equilibrium. In Advances inNeural Information Processing Systems, pp. 6626–6637,2017.
* Ho, J., Jain, A., and Abbeel, P. Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems, 33, 2020.
* Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko,A., Kingma, D. P., Poole, B., Norouzi, M., Fleet, D. J.,et al. Imagen video: High definition video generationwith diffusion models. arXiv preprint arXiv:2210.02303,2022a.
* Ho, J., Salimans, T., Gritsenko, A. A., Chan, W., Norouzi,M., and Fleet, D. J. Video diffusion models. In ICLRWorkshop on Deep Generative Models for Highly Structured Data, 2022b. URL https://openreview.net/forum?id=BBelR2NdDZ5.
* Hyv¨arinen, A. and Dayan, P. Estimation of non-normalizedstatistical models by score matching. Journal of MachineLearning Research (JMLR), 6(4), 2005.
* Jiang, Y., Chang, S., and Wang, Z. Transgan: Two puretransformers can make one strong gan, and that can scaleup. Advances in Neural Information Processing Systems,34:14745–14758, 2021.
* Karras, T., Aila, T., Laine, S., and Lehtinen, J. Progressive growing of GANs for improved quality, stability,and variation. In International Conference on LearningRepresentations, 2018. URL https://openreview.net/forum?id=Hk99zCeAb.
* Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J.,and Aila, T. Analyzing and improving the image qualityof stylegan. 2020.
* Karras, T., Aittala, M., Aila, T., and Laine, S. Elucidatingthe design space of diffusion-based generative models. InProc. NeurIPS, 2022.
* Kawar, B., Vaksman, G., and Elad, M. Snips: Solvingnoisy inverse problems stochastically. arXiv preprintarXiv:2105.14951, 2021.
* Kawar, B., Elad, M., Ermon, S., and Song, J. Denoisingdiffusion restoration models. In Advances in Neural Information Processing Systems, 2022.
* Kingma, D. P. and Dhariwal, P. Glow: Generative flowwith invertible 1x1 convolutions. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K., Cesa-Bianchi, N.,and Garnett, R. (eds.), Advances in Neural InformationProcessing Systems 31, pp. 10215–10224. 2018.
* Kingma, D. P. and Welling, M. Auto-encoding variationalbayes. In International Conference on Learning Representations, 2014.
* Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro,B. DiffWave: A Versatile Diffusion Model for AudioSynthesis. arXiv preprint arXiv:2009.09761, 2020.
* Krizhevsky, A., Hinton, G., et al. Learning multiple layersof features from tiny images. 2009.
* Kynk¨a¨anniemi, T., Karras, T., Laine, S., Lehtinen, J., andAila, T. Improved precision and recall metric for assessing generative models. Advances in Neural InformationProcessing Systems, 32, 2019.
* Lee, K., Chang, H., Jiang, L., Zhang, H., Tu, Z., and Liu,C. Vitgan: Training gans with vision transformers. arXivpreprint arXiv:2107.04589, 2021.
* Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez,T., Tassa, Y., Silver, D., and Wierstra, D. Continuouscontrol with deep reinforcement learning. arXiv preprintarXiv:1509.02971, 2015.
* Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., andHan, J. On the variance of the adaptive learning rate andbeyond. arXiv preprint arXiv:1908.03265, 2019.
* Liu, X., Gong, C., and Liu, Q. Flow straight and fast:Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
* Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J.Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. arXiv preprintarXiv:2206.00927, 2022.
* Luhman, E. and Luhman, T. Knowledge distillation initerative generative models for improved sampling speed. arXiv preprint arXiv:2101.02388, 2021.
* Meng, C., Song, Y., Song, J., Wu, J., Zhu, J.-Y., and Ermon,S. Sdedit: Image synthesis and editing with stochasticdifferential equations. arXiv preprint arXiv:2108.01073,2021.
* Meng, C., Gao, R., Kingma, D. P., Ermon, S., Ho, J., andSalimans, T. On distillation of guided diffusion models. arXiv preprint arXiv:2210.03142, 2022.
* Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A.,Antonoglou, I., Wierstra, D., and Riedmiller, M. Playingatari with deep reinforcement learning. arXiv preprintarXiv:1312.5602, 2013.
* Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness,J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level controlthrough deep reinforcement learning. nature, 518(7540):529–533, 2015.
* Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin,P., McGrew, B., Sutskever, I., and Chen, M. Glide:Towards photorealistic image generation and editingwith text-guided diffusion models. arXiv preprintarXiv:2112.10741, 2021.
* Parmar, G., Li, D., Lee, K., and Tu, Z. Dual contradistinctivegenerative autoencoder. In Proceedings of the IEEE/CVFConference on Computer Vision and Pattern Recognition,pp. 823–832, 2021.
* Popov, V., Vovk, I., Gogoryan, V., Sadekova, T., and Kudinov, M. Grad-TTS: A diffusion probabilistic model fortext-to-speech. arXiv preprint arXiv:2105.06337, 2021.
* Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen,M. Hierarchical text-conditional image generation withclip latents. arXiv preprint arXiv:2204.06125, 2022.
* Rezende, D. J., Mohamed, S., and Wierstra, D. Stochasticbackpropagation and approximate inference in deep generative models. In Proceedings of the 31st InternationalConference on Machine Learning, pp. 1278–1286, 2014.
* Rombach, R., Blattmann, A., Lorenz, D., Esser, P., andOmmer, B. High-resolution image synthesis with latentdiffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.10684–10695, 2022.
* Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton,E., Ghasemipour, S. K. S., Ayan, B. K., Mahdavi, S. S.,Lopes, R. G., et al. Photorealistic text-to-image diffusionmodels with deep language understanding. arXiv preprintarXiv:2205.11487, 2022.
* Salimans, T. and Ho, J. Progressive distillation for fastsampling of diffusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=TIdIXIpzhoI.
* Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V.,Radford, A., and Chen, X. Improved techniques for training gans. In Advances in neural information processingsystems, pp. 2234–2242, 2016.
* Sauer, A., Schwarz, K., and Geiger, A. Stylegan-xl: Scalingstylegan to large diverse datasets. In ACM SIGGRAPH2022 conference proceedings, pp. 1–10, 2022.
* Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., andGanguli, S. Deep Unsupervised Learning Using Nonequilibrium Thermodynamics. In International Conferenceon Machine Learning, pp. 2256–2265, 2015.
* Song, J., Meng, C., and Ermon, S. Denoising diffusionimplicit models. arXiv preprint arXiv:2010.02502, 2020.
* Song, J., Vahdat, A., Mardani, M., and Kautz, J.
* Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=9_gsMA8MRKQ.
* Song, Y. and Ermon, S. Generative Modeling by EstimatingGradients of the Data Distribution. In Advances in NeuralInformation Processing Systems, pp. 11918–11930, 2019.
* Song, Y. and Ermon, S. Improved Techniques for TrainingScore-Based Generative Models. Advances in NeuralInformation Processing Systems, 33, 2020.
* Song, Y., Garg, S., Shi, J., and Ermon, S. Sliced scorematching: A scalable approach to density and score estimation. In Proceedings of the Thirty-Fifth Conference onUncertainty in Artificial Intelligence, UAI 2019, Tel Aviv,Israel, July 22-25, 2019, pp. 204, 2019.
* Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A.,Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations,2021. URL https://openreview.net/forum?id=PxTIG12RRHS.
* Song, Y., Shen, L., Xing, L., and Ermon, S. Solving inverseproblems in medical imaging with score-based generative models. In International Conference on LearningRepresentations, 2022. URL https://openreview.net/forum?id=vaRCHVj0uGI.
* S¨uli, E. and Mayers, D. F. An introduction to numericalanalysis. Cambridge university press, 2003.
* Tian, Y., Wang, Q., Huang, Z., Li, W., Dai, D., Yang, M.,Wang, J., and Fink, O. Off-policy reinforcement learning for efficient and effective gan architecture search. InComputer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings,Part VII 16, pp. 175–192. Springer, 2020.
* Uria, B., Murray, I., and Larochelle, H. Rnade: the realvalued neural autoregressive density-estimator. In Proceedings of the 26th International Conference on Neural Information Processing Systems-Volume 2, pp. 2175–2183, 2013.
* Uria, B., Cˆot´e, M.-A., Gregor, K., Murray, I., andLarochelle, H. Neural autoregressive distribution estimation. The Journal of Machine Learning Research, 17(1):7184–7220, 2016.
* Vahdat, A., Kreis, K., and Kautz, J. Score-based generativemodeling in latent space. Advances in Neural InformationProcessing Systems, 34:11287–11302, 2021.
* Van Den Oord, A., Kalchbrenner, N., and Kavukcuoglu, K.Pixel recurrent neural networks. In Proceedings of the33rd International Conference on International Conference on Machine Learning - Volume 48, ICML’16, pp.1747–1756. JMLR.org, 2016. URL http://dl.acm.org/citation.cfm?id=3045390.3045575.
* Vincent, P. A Connection Between Score Matching andDenoising Autoencoders. Neural Computation, 23(7):1661–1674, 2011.
* Wu, J., Huang, Z., Acharya, D., Li, W., Thoma, J., Paudel,D. P., and Gool, L. V. Sliced wasserstein generativemodels. In Proceedings of the IEEE/CVF Conferenceon Computer Vision and Pattern Recognition, pp. 3713–3722, 2019.
* Xiao, Z., Yan, Q., and Amit, Y. Generative latent flow. arXivpreprint arXiv:1905.10485, 2019.
* Xu, Y., Liu, Z., Tegmark, M., and Jaakkola, T. S. Poisson flow generative models. In Oh, A. H., Agarwal, A.,Belgrave, D., and Cho, K. (eds.), Advances in NeuralInformation Processing Systems, 2022. URL https://openreview.net/forum?id=voV_TRqcWh.
* Yu, F., Seff, A., Zhang, Y., Song, S., Funkhouser, T., andXiao, J. Lsun: Construction of a large-scale image datasetusing deep learning with humans in the loop. arXivpreprint arXiv:1506.03365, 2015.
* Zhang, H., Zhang, Z., Odena, A., and Lee, H. Consistencyregularization for generative adversarial networks. arXivpreprint arXiv:1910.12027, 2019.
* Zhang, Q. and Chen, Y. Fast sampling of diffusionmodels with exponential integrator. arXiv preprintarXiv:2204.13902, 2022.
* Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang,O. The unreasonable effectiveness of deep features as aperceptual metric. In CVPR, 2018.
* Zheng, H., Nie, W., Vahdat, A., Azizzadenesheli, K., andAnandkumar, A. Fast sampling of diffusion modelsvia operator learning. arXiv preprint arXiv:2211.13449,2022.

## Appendices

### A. Proofs
#### A.1. Notations
We use fθpx, tq to denote a consistency model parameterized by θ, and fpx, t; φq the consistency function of the empirical PF ODE in Eq. (3). Here φ symbolizes its dependency on the pre-trained score model sφpx, tq. For the consistency function of the PF ODE in Eq. (2), we denote it as fpx, tq. Given a multi-variate function hpx, yq, we let B1hpx, yq denote the

Jacobian of h over x, and analogously B2hpx, yq denote the Jacobian of h over y. Unless otherwise stated, x is supposed to be a random variable sampled from the data distribution pdatapxq, n is sampled uniformly at random from J 1, N ´ 1K , and xtn is sampled from N px;t 2 nIq. Here J 1, N ´ 1K represents the set of integers t1, 2, ¨ ¨ ¨ , N ´ 1u. Furthermore, recall that we define ˆx φ tn :“ xtn`1 ` ptn ´ tn`1qΦpxtn`1 , tn`1; φq, where Φp¨ ¨ ¨ ; φq denotes the update function of a one-step ODE solver for the empirical PF ODE defined by the score model sφpx, tq. By default, Er¨s denotes the expectation over all relevant random variables in the expression.

#### A.2. Consistency Distillation
Theorem 1. Let ∆t :“ maxnPJ 1,N´1K t|tn`1 ´ tn|u, and fp¨, ¨; φq be the consistency function of the empirical PF ODE in Eq. (3). Assume fθ satisfies the Lipschitz condition: there exists L ą 0 such that for all t P r, Ts, x, and y, we have k fθpx, tq ´ fθpy, tqk 2 ď Lk x ´ yk 2 . Assume further that for all n P J 1, N ´ 1K , the ODE solver called at tn`1 has local error uniformly bounded by Opptn`1 ´ tnq p`1 q with p ě 1. Then, if L

N

CDpθ, θ; φq “ 0, we have sup n,x }fθpx, tnq ´ fpx, tn; φq}2 “ Opp∆tq p q.

Proof. From L

N

CDpθ, θ; φq “ 0, we have

L

N

CDpθ, θ; φq “ Erλptnqdpfθpxtn`1 , tn`1q, fθpˆx φ tn , tnqqs “ 0. (11)

According to the definition, we have ptn pxtn q “ pdatapxq b N p0, t2 nIq where tn ě  ą 0. It follows that ptn pxtn q ą 0 for every xtn and 1 ď n ď N. Therefore, Eq. (11) entails λptnqdpfθpxtn`1 , tn`1q, fθpˆx φ tn , tnqq ” 0. (12)

Because λp¨q ą 0 and dpx, yq “ 0 ô x “ y, this further implies that fθpxtn`1 , tn`1q ” fθpˆx φ tn , tnq. (13)

Now let en represent the error vector at tn, which is defined as en :“ fθpxtn , tnq ´ fpxtn , tn; φq.

We can easily derive the following recursion relation en`1 “ fθpxtn`1 , tn`1q ´ fpxtn`1 , tn`1; φq

Consistency Models piq “ fθpˆx φ tn , tnq ´ fpxtn , tn; φq “ fθpˆx φ tn , tnq ´ fθpxtn , tnq ` fθpxtn , tnq ´ fpxtn , tn; φq “ fθpˆx φ tn , tnq ´ fθpxtn , tnq ` en, (14) where (i) is due to Eq. (13) and fpxtn`1 , tn`1; φq “ fpxtn , tn; φq. Because fθp¨, tnq has Lipschitz constant L, we have k en`1k 2 ď k enk 2 ` L      ˆx φ tn ´ xtn      2 piq “ k enk 2 ` L ¨ Opptn`1 ´ tnq p`1 q “ k enk 2 ` Opptn`1 ´ tnq p`1 q, where (i) holds because the ODE solver has local error bounded by Opptn`1 ´ tnq p`1 q. In addition, we observe that e1 “ 0, because e1 “ fθpxt1 , t1q ´ fpxt1 , t1; φq piq “ xt1 ´ fpxt1 , t1; φq piiq “ xt1 ´ xt1 “ 0.

Here (i) is true because the consistency model is parameterized such that fpxt1 , t1; φq “ xt1 and (ii) is entailed by the definition of fp¨, ¨; φq. This allows us to perform induction on the recursion formula Eq. (14) to obtain k enk 2 ď k e1k 2 ` n´1 k ÿ “1

Opptk`1 ´ tkq p`1 q “ n´1 k ÿ “1

Opptk`1 ´ tkq p`1 q “ n´1 k ÿ “1 ptk`1 ´ tkqOpptk`1 ´ tkq p q ď n´1 k ÿ “1 ptk`1 ´ tkqOpp∆tq p q “ Opp∆tq p q n´1 k ÿ “1 ptk`1 ´ tkq “ Opp∆tq p qptn ´ t1q ď Opp∆tq p qpT ´  q “ Opp∆tq p q, which completes the proof.

#### A.3. Consistency Training
The following lemma provides an unbiased estimator for the score function, which is crucial to our proof for Theorem 2.

Lemma 1. Let x „ pdatapxq, xt „ N px;t 2Iq, and ptpxtq “ pdatapxqbN p0, t2Iq. We have ∇ log ptpxq “ ´Er xt´x t 2 | xts.

Proof. According to the definition of ptpxtq, we have ∇ log ptpxtq “ ∇xt log ş pdatapxqppxt | xq dx, where ppxt | xq “

N pxt; x, t2Iq. This expression can be further simplified to yield ∇ log ptpxtq “ ş pdatapxq∇xt ppxt | xq dx ş pdatapxqppxt | xq dx

Consistency Models “ ş pdatapxqppxt | xq∇xt log ppxt | xq dx ş pdatapxqppxt | xq dx “ ş pdatapxqppxt | xq∇xt log ppxt | xq dx ptpxtq “ ż pdatapxqppxt | xq ptpxtq ∇xt log ppxt | xq dx piq “ ż ppx | xtq∇xt log ppxt | xq dx “ Er∇xt log ppxt | xq | xts “ ´E „ xt ´ x t 2 | xt  , where (i) is due to Bayes’ rule.

Theorem 2. Let ∆t :“ maxnPJ 1,N´1K t|tn`1 ´ tn|u. Assume d and fθ´ are both twice continuously differentiable with bounded second derivatives, the weighting function λp¨q is bounded, and Erk∇ log ptn pxtn qk 2 2 s ă 8. Assume further that we use the Euler ODE solver, and the pre-trained score model matches the ground truth, i.e., @t P r, Ts : sφpx, tq ” ∇ log ptpxq. Then,

L

N

CDpθ, θ ´; φq “ L

N

CTpθ, θ ´q ` op∆tq, where the expectation is taken with respect to x „ pdata, n „ UJ 1, N ´ 1K , and xtn`1 „ N px;t 2 n`1Iq. The consistency training objective, denoted by L

N

CTpθ, θ ´q, is defined as

Erλptnqdpfθpx ` tn`1z, tn`1q, fθ´ px ` tnz, tnqqs, where z „ N p0, Iq. Moreover, L

N

CTpθ, θ ´q ě Op∆tq if infN L

N

CDpθ, θ ´; φq ą 0.

Proof. With Taylor expansion, we have

L

N

CDpθ, θ ´; φq “ Erλptnqdpfθpxtn`1 , tn`1q, fθ´ pˆx φ tn , tnqs “Erλptnqdpfθpxtn`1 , tn`1q, fθ´ pxtn`1 ` ptn`1 ´ tnqtn`1∇ log ptn`1 pxtn`1 q, tnqqs “Erλptnqdpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1q ` B1fθ´ pxtn`1 , tn`1qptn`1 ´ tnqtn`1∇ log ptn`1 pxtn`1 q ` B2fθ´ pxtn`1 , tn`1qptn ´ tn`1q ` op|tn`1 ´ tn|qqs “Etλptnqdpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qq ` λptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqr

B1fθ´ pxtn`1 , tn`1qptn`1 ´ tnqtn`1∇ log ptn`1 pxtn`1 q ` B2fθ´ pxtn`1 , tn`1qptn ´ tn`1q ` op|tn`1 ´ tn|qsu “Erλptnqdpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqs ` EtλptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqrB1fθ´ pxtn`1 , tn`1qptn`1 ´ tnqtn`1∇ log ptn`1 pxtn`1 qsu ` EtλptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqrB2fθ´ pxtn`1 , tn`1qptn ´ tn`1qsu ` Erop|tn`1 ´ tn|qs. (15)

Then, we apply Lemma 1 to Eq. (15) and use Taylor expansion in the reverse direction to obtain

L

N

CDpθ, θ ´; φq “Erλptnqdpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqs ` E " λptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qq „

B1fθ´ pxtn`1 , tn`1qptn ´ tn`1qtn`1E „ xtn`1 ´ x t 2 ˇ ˇ ˇxtn`1 * ` EtλptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqrB2fθ´ pxtn`1 , tn`1qptn ´ tn`1qsu ` Erop|tn`1 ´ tn|qs piq “Erλptnqdpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqs ` E " λptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qq „

B1fθ´ pxtn`1 , tn`1qptn ´ tn`1qtn`1 ˆ xtn`1 ´ x t 2 ˙*

Consistency Models ` EtλptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqrB2fθ´ pxtn`1 , tn`1qptn ´ tn`1qsu ` Erop|tn`1 ´ tn|qs “E „ λptnqdpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qq ` λptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qq „

B1fθ´ pxtn`1 , tn`1qptn ´ tn`1qtn`1 ˆ xtn`1 ´ x t 2 ˙ ` λptnqB2dpfθpxtn`1 , tn`1q, fθ´ pxtn`1 , tn`1qqrB2fθ´ pxtn`1 , tn`1qptn ´ tn`1qs ` op|tn`1 ´ tn|q ` Erop|tn`1 ´ tn|qs “E „ λptnqd ˆ fθpxtn`1 , tn`1q, fθ´ ˆ xtn`1 ` ptn ´ tn`1qtn`1 xtn`1 ´ x t 2 n`1 , tn ˙˙ ` Erop|tn`1 ´ tn|qs “E „ λptnqd ˆ fθpxtn`1 , tn`1q, fθ´ ˆ xtn`1 ` ptn ´ tn`1q xtn`1 ´ x tn`1 , tn ˙˙ ` Erop|tn`1 ´ tn|qs “E rλptnqd pfθpx ` tn`1z, tn`1q, fθ´ px ` tn`1z ` ptn ´ tn`1qz, tnqqs ` Erop|tn`1 ´ tn|qs “E rλptnqd pfθpx ` tn`1z, tn`1q, fθ´ px ` tnz, tnqqs ` Erop|tn`1 ´ tn|qs “E rλptnqd pfθpx ` tn`1z, tn`1q, fθ´ px ` tnz, tnqqs ` Erop∆tqs “E rλptnqd pfθpx ` tn`1z, tn`1q, fθ´ px ` tnz, tnqqs ` op∆tq “L

N

CTpθ, θ ´q ` op∆tq, (16) where (i) is due to the law of total expectation, and z :“ xtn`1´x tn`1 is distributed according to the standard Gaussian distribution. This implies L

N

CDpθ, θ ´; φq “ L

N

CTpθ, θ ´q ` op∆tq and thus completes the proof for Eq. (9). Moreover, we have L

N

CTpθ, θ ´q ě Op∆tq whenever infN L

N

CDpθ, θ ´; φq ą 0. Otherwise, L

N

CTpθ, θ ´q ă Op∆tq and thus lim∆tÑ0 L

N

CDpθ, θ ´; φq “ 0, a clear contradiction to infN L

N

CDpθ, θ ´; φq ą 0.

### B. Continuous-Time Extensions
The consistency distillation and consistency training objectives can be generalized to hold for infinite time steps (N Ñ 8) under suitable conditions.

#### B.1. Consistency Distillation in Continuous Time
Depending on whether θ ´ “ θ or θ ´ “ stopgradpθq (same as setting µ “ 0), there are two possible continuous-time extensions for the consistency distillation objective L

N

CDpθ, θ ´; φq. Given a twice continuously differentiable metric function dpx, yq, we define Gpxq as a matrix, whose pi, jq-th entry is given by rGpxqsij :“

B 2dpx, yq

ByiByj ˇ ˇ ˇ ˇ y“x .

Similarly, we define Hpxq as rHpxqsij :“

B 2dpy, xq

ByiByj ˇ ˇ ˇ ˇ y“x .

The matrices G and H play a crucial role in forming continuous-time objectives for consistency distillation. Additionally, we denote the Jacobian of fθpx, tq with respect to x as

Bfθpx,tq

Bx .

When θ ´ “ θ (with no stopgrad operator), we have the following theoretical result.

Theorem 3. Let tn “ pn´1q

N´1 pT ´  q `  and ∆t “

T ´

N´1 , where n P J 1, NK . Assume d is three times continuously differentiable with bounded third derivatives, and fθ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function λp¨q is bounded, and supx,tPr,Ts k sφpx, tqk 2 ă 8. Suppose we use the Euler ODE solver. Then, lim ∆tÑ0

L

N

CDpθ, θ; φq p∆tq 2 “ L 8

CDpθ, θ; φq, (17)

Consistency Models where

L 8

CDpθ, θ; φq :“ 1 2

E « λptq ˆ

Bfθpxt, tq

Bt ´ t

Bfθpxt, tq

Bxt sφpxt, tq ˙T

Gpfθpxt, tqq ˆ

Bfθpxt, tq

Bt ´ t

Bfθpxt, tq

Bxt sφpxt, tq ˙ff . (18)

Here the expectation above is taken over x „ pdata, t „ Ur, Ts, and xt „ N px, t2Iq.

Proof. First, we can derive the following equation with Taylor expansion: fθpˆx φ tn , tnq ´ fθpxtn`1 , tn`1q “ fθpxtn`1 ` tn`1sφpxtn`1 , tn`1q∆t, tnq ´ fθpxtn`1 , tn`1q “tn`1

Bfθpxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q∆t ´

Bfθpxtn`1 , tn`1q

Btn`1 ∆t ` Opp∆tq 2 q. (19)

Applying Taylor expansion again, we get 1 p∆tq 2

L

N

CDpθ, θ; φq “ 1 p∆tq 2

Erλptnqdpfθpxtn`1 , tn`1q, fθpˆx φ tn , tnqs piq “ 1 2p∆tq 2 ˆ

Etλptnqrfθpˆx φ tn , tnq ´ fθpxtn`1 , tn`1qsTGpfθpxtn`1 , tn`1qq ¨ rfθpˆx φ tn , tnq ´ fθpxtn`1 , tn`1qsu ` ErOp|∆t| 3 qs˙ piiq “ 1 2

E „ λptnq ˆ

Bfθpxtn`1 , tn`1q

Btn`1 ´ tn`1

Bfθpxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q ˙T

Gpfθpxtn`1 , tn`1qq ¨ ˆ

Bfθpxtn`1 , tn`1q

Btn`1 ´ tn`1

Bfθpxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q ˙ ` ErOp|∆t|qs (20) where we obtain (i) by first expanding dpfθpxtn`1 , tn`1q, ¨q to second order, and then noting that dpx, xq ” 0 and ∇ydpx, yq|y“x ” 0. We obtain (ii) using Eq. (19). By taking the limit for both sides of Eq. (20) as ∆t Ñ 0, we arrive at

Eq. (17), which completes the proof.

Remark 1. Although Theorem 3 assumes uniform timesteps and the Euler ODE solver for technical simplicity, we believe an analogous result can be derived without those assumptions, as all time boundaries and ODE solvers should perform similarly as N Ñ 8. We leave a more general version of Theorem 3 as future work.

Remark 2. Theorem 3 implies that consistency models can be trained by minimizing L 8

CDpθ, θ; φq. In particular, when dpx, yq “ k x ´ yk 2 2 , we have

L 8

CDpθ, θ; φq “ E « λptq       

Bfθpxt, tq

Bt ´ t

Bfθpxt, tq

Bxt sφpxt, tq        2 2 ff . (21)

However, this continuous-time objective requires computing Jacobian-vector products as a subroutine to evaluate the loss function, which can be slow and laborious to implement in deep learning frameworks that do not support forward-mode automatic differentiation.

Remark 3. If fθpx, tq matches the ground truth consistency function for the empirical PF ODE of sφpx, tq, then

Bfθpx, tq

Bt ´ t

Bfθpx, tq

Bx sφpx, tq ” 0 and therefore L 8

CDpθ, θ; φq “ 0. This can be proved by noting that fθpxt, tq ” x for all t P r, Ts, and then taking the time-derivative of this identity: fθpxt, tq ” x ðñ

Bfθpxt, tq

Bxt dxt dt `

Bfθpxt, tq

Bt ” 0

Consistency Models ðñ

Bfθpxt, tq

Bxt r´tsφpxt, tqs ` Bfθpxt, tq

Bt ” 0 ðñ

Bfθpxt, tq

Bt ´ t

Bfθpxt, tq

Bxt sφpxt, tq ” 0.

The above observation provides another motivation for L 8

CDpθ, θ; φq, as it is minimized if and only if the consistency model matches the ground truth consistency function.

For some metric functions, such as the ` 1 norm, the Hessian Gpxq is zero so Theorem 3 is vacuous. Below we show that a non-vacuous statement holds for the ` 1 norm with just a small modification of the proof for Theorem 3.

Theorem 4. Let tn “ pn´1q

N´1 pT ´  q `  and ∆t “

T ´

N´1 , where n P J 1, NK . Assume fθ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function λp¨q is bounded, and supx,tPr,Ts k sφpx, tqk 2 ă 8. Suppose we use the Euler ODE solver, and set dpx, yq “ k x ´ yk 1 . Then, lim ∆tÑ0

L

N

CDpθ, θ; φq ∆t “ L 8

CD, ` 1 pθ, θ; φq, (22) where

L 8

CD, ` 1 pθ, θ; φq :“ E „ λptq       t

Bfθpxt, tq

Bxt sφpxt, tq ´ Bfθpxt, tq

Bt         1  where the expectation above is taken over x „ pdata, t „ Ur, Ts, and xt „ N px, t2Iq.

Proof. We have 1 ∆t

L

N

CDpθ, θ; φq “ 1 ∆t

Erλptnq}fθpxtn`1 , tn`1q ´ fθpˆx φ tn , tnq}1s piq “ 1 ∆t

E „ λptnq       tn`1

Bfθpxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q∆t ´

Bfθpxtn`1 , tn`1q

Btn`1 ∆t ` Opp∆tq 2 q        1  “E „ λptnq       tn`1

Bfθpxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q ´ Bfθpxtn`1 , tn`1q

Btn`1 ` Op∆tq        1  (23) where (i) is obtained by plugging Eq. (19) into the previous equation. Taking the limit for both sides of Eq. (23) as ∆t Ñ 0 leads to Eq. (22), which completes the proof.

Remark 4. According to Theorem 4, consistency models can be trained by minimizing L 8

CD, ` 1 pθ, θ; φq. Moreover, the same reasoning in Remark 3 can be applied to show that L 8

CD, ` 1 pθ, θ; φq “ 0 if and only if fθpxt, tq “ x for all xt P R d and t P r, Ts.

In the second case where θ ´ “ stopgradpθq, we can derive a pseudo-objective whose gradient matches the gradient of

L

N

CDpθ, θ ´; φq in the limit of N Ñ 8. Minimizing this pseudo-objective with gradient descent gives another way to train consistency models via distillation. This pseudo-objective is provided by the theorem below.

Theorem 5. Let tn “ pn´1q

N´1 pT ´  q `  and ∆t “

T ´

N´ ously differentiable with bounded third derivatives, and 1 , where n P J 1, NK . Assume d is three times continufθ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function λp¨q is bounded, supx,tPr,Ts k sφpx, tqk 2 ă 8, and supx,tPr,Ts k∇θfθpx, tqk 2 ă 8. Suppose we use the Euler ODE solver, and θ ´ “ stopgradpθq. Then, lim ∆tÑ0 ∇θL

N

CDpθ, θ ´; φq ∆t “ ∇θL 8

CDpθ, θ ´; φq, (24) where

L 8

CDpθ, θ ´; φq :“ E „ λptqfθpxt, tq

THpfθ´ pxt, tqq ˆ

Bfθ´ pxt, tq

Bt ´ t

Bfθ´ pxt, tq

Bxt sφpxt, tq ˙ . (25)

Here the expectation above is taken over x „ pdata, t „ Ur, Ts, and xt „ N px, t2Iq.



Proof. First, we leverage Taylor series expansion to obtain 1 ∆t

L

N

CDpθ, θ ´; φq “ ∆ 1 t

Erλptnqdpfθpxtn`1 , tn`1q, fθ´ pˆx φ tn , tnqs piq “ 1 2∆t ˆ

Etλptnqrfθpxtn`1 , tn`1q ´ fθ´ pˆx φ tn , tnqsTHpfθ´ pˆx φ tn , tnqq ¨ rfθpxtn`1 , tn`1q ´ fθ´ pˆx φ tn , tnqsu ` ErOp|∆t| 3 qs˙ “ 1 2∆t

Etλptnqrfθpxtn`1 , tn`1q ´ fθ´ pˆx φ tn , tnqsTHpfθ´ pˆx φ tn , tnqqrfθpxtn`1 , tn`1q ´ fθ´ pˆx φ tn , tnqsu ` ErOp|∆t| 2 qs (26) where (i) is derived by first expanding dp¨, fθ´ pˆx φ tn , tnqq to second order, and then noting that dpx, xq ” 0 and ∇ydpy, xq|y“x ” 0. Next, we compute the gradient of Eq. (26) with respect to θ and simplify the result to obtain 1 ∆t ∇θL

N

CDpθ, θ ´; φq piq “ 1 2∆t ∇θEtλptnqrfθpxtn`1 , tn`1q ´ fθ´ pˆx φ tn , tnqsTHpfθ´ pˆx φ tn , tnqqrfθpxtn`1 , tn`1q ´ fθ´ pˆx φ tn , tnqsu ` ErOp|∆t| 2 qs “ 1 ∆t

Etλptnqr∇θfθpxtn`1 , tn`1qsTHpfθ´ pˆx φ tn , tnqqrfθpxtn`1 , tn`1q ´ fθ´ pˆx φ tn , tnqsu ` ErOp|∆t| 2 qs piiq “ 1 ∆t

E " λptnqr∇θfθpxtn`1 , tn`1qsTHpfθ´ pˆx φ tn , tnqq„ tn`1

Bfθ´ pxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q∆t ´

Bfθ´ pxtn`1 , tn`1q

Btn`1 ∆t * ` ErOp|∆t|qs “E " λptnqr∇θfθpxtn`1 , tn`1qsTHpfθ´ pˆx φ tn , tnqq„ tn`1

Bfθ´ pxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q ´

Bfθ´ pxtn`1 , tn`1q

Btn`1 * ` ErOp|∆t|qs “∇θE " λptnqrfθpxtn`1 , tn`1qsTHpfθ´ pˆx φ tn , tnqq„ tn`1

Bfθ´ pxtn`1 , tn`1q

Bxtn`1 sφpxtn`1 , tn`1q ´

Bfθ´ pxtn`1 , tn`1q

Btn`1 * ` ErOp|∆t|qs (27)

Here (i) results from the chain rule, and (ii) follows from Eq. (19) and fθpx, tq ” fθ´ px, tq, since θ ´ “ stopgradpθq.

Taking the limit for both sides of Eq. (27) as ∆t Ñ 0 yields Eq. (24), which completes the proof.

Remark 5. When dpx, yq “ k x ´ yk 2 2 , the pseudo-objective L 8

CDpθ, θ ´; φq can be simplified to

L 8

CDpθ, θ ´; φq “ 2E „ λptqfθpxt, tq

T ˆ

Bfθ´ pxt, tq

Bt ´ t

Bfθ´ pxt, tq

Bxt sφpxt, tq ˙ . (28)

Remark 6. The objective L 8

CDpθ, θ ´; φq defined in Theorem 5 is only meaningful in terms of its gradient—one cannot measure the progress of training by tracking the value of L 8

CDpθ, θ ´; φq, but can still apply gradient descent to this objective to distill consistency models from pre-trained diffusion models. Because this objective is not a typical loss function, we refer to it as the “pseudo-objective” for consistency distillation.

Remark 7. Following the same reasoning in Remark 3, we can easily derive that L 8

CDpθ, θ ´; φq “ 0 and ∇θL 8

CDpθ, θ ´; φq “ 0 if fθpx, tq matches the ground truth consistency function for the empirical PF ODE that involves sφpx, tq. However, the converse does not hold true in general. This distinguishes L 8

CDpθ, θ ´; φq from L 8

CDpθ, θ; φq, the latter of which is a true loss function.

#### B.2. Consistency Training in Continuous Time
A remarkable observation is that the pseudo-objective in Theorem 5 can be estimated without any pre-trained diffusion models, which enables direct consistency training of consistency models. More precisely, we have the following result.



Theorem 6. Let tn “ pn´1q

N´1 pT ´  q `  and ∆t “

T ´

N´1 , where n P J 1, NK . Assume d is three times continuously differentiable with bounded third derivatives, and fθ is twice continuously differentiable with bounded first and second derivatives. Assume further that the weighting function λp¨q is bounded, Erk∇ log ptn pxtn qk 2 2 s ă 8, and supx,tPr,Ts k∇θfθpx, tqk 2 ă 8. Suppose we use the Euler ODE solver and θ ´ “ stopgradpθq. Then, lim ∆tÑ0 ∇θL

N

CTpθ, θ ´q ∆t “ ∇θL 8

CTpθ, θ ´q, (29) where

L 8

CTpθ, θ ´q :“ E „ λptqfθpxt, tq

THpfθ´ pxt, tqq ˆ

Bfθ´ pxt, tq

Bt `

Bfθ´ pxt, tq

Bxt ¨ xt ´ x t ˙ . (30)

Here the expectation above is taken over x „ pdata, t „ Ur, Ts, and xt „ N px, t2Iq.

Proof. The proof mostly follows that of Theorem 5. First, we leverage Taylor series expansion to obtain 1 ∆t

L

N

CTpθ, θ ´q “ ∆ 1 t

Erλptnqdpfθpx ` tn`1z, tn`1q, fθ´ px ` tnz, tnqqs piq “ 1 2∆t ˆ

Etλptnqrfθpx ` tn`1z, tn`1q ´ fθ´ px ` tnz, tnqsTHpfθ´ px ` tnz, tnqq ¨ rfθpx ` tn`1z, tn`1q ´ fθ´ px ` tnz, tnqsu ` ErOp|∆t| 3 qs˙ “ 1 2∆t

Etλptnqrfθpx ` tn`1z, tn`1q ´ fθ´ px ` tnz, tnqsTHpfθ´ px ` tnz, tnqq ¨ rfθpx ` tn`1z, tn`1q ´ fθ´ px ` tnz, tnqsu ` ErOp|∆t| 2 qs (31) where (i) is derived by first expanding dp¨, fθ´ px ` tnz, tnqq to second order, and then noting that dpx, xq ” 0 and ∇ydpy, xq|y“x ” 0. Next, we compute the gradient of Eq. (31) with respect to θ and simplify the result to obtain 1 ∆t ∇θL

N

CTpθ, θ ´q “ 1 2∆t ∇θEtλptnqrfθpx ` tn`1z, tn`1q ´ fθ´ px ` tnz, tnqsTHpfθ´ px ` tnz, tnqq ¨ rfθpx ` tn`1z, tn`1q ´ fθ´ px ` tnz, tnqsu ` ErOp|∆t| 2 qs piq “ 1 ∆t

Etλptnqr∇θfθpx ` tn`1z, tn`1qsTHpfθ´ px ` tnz, tnqq ¨ rfθpx ` tn`1z, tn`1q ´ fθ´ px ` tnz, tnqsu ` ErOp|∆t| 2 qs piiq “ 1 ∆t

E " λptnqr∇θfθpx ` tn`1z, tn`1qsTHpfθ´ px ` tnz, tnqq„ ∆tB1fθ´ px ` tnz, tnqz ` B2fθ´ px ` tnz, tnq∆t * ` ErOp|∆t|qs “E " λptnqr∇θfθpx ` tn`1z, tn`1qsTHpfθ´ px ` tnz, tnqq„

B1fθ´ px ` tnz, tnqz ` B2fθ´ px ` tnz, tnq * ` ErOp|∆t|qs “∇θE " λptnqrfθpx ` tn`1z, tn`1qsTHpfθ´ px ` tnz, tnqq„

B1fθ´ px ` tnz, tnqz ` B2fθ´ px ` tnz, tnq * ` ErOp|∆t|qs “∇θE " λptnqrfθpxtn`1 , tn`1qsTHpfθ´ pxtn , tnqq„

B1fθ´ pxtn , tnq xtn ´ x tn ` B2fθ´ pxtn , tnq * ` ErOp|∆t|qs (32)

Here (i) results from the chain rule, and (ii) follows from Taylor expansion. Taking the limit for both sides of Eq. (32) as ∆t Ñ 0 yields Eq. (29), which completes the proof.

Consistency Models (a) Consistency Distillation (b) Consistency Training

Figure 7: Comparing discrete consistency distillation/training algorithms with continuous counterparts.

Remark 8. Note that L 8

CTpθ, θ ´q does not depend on the diffusion model parameter φ and hence can be optimized without any pre-trained diffusion models.

Remark 9. When dpx, yq “ k x ´ yk 2 2 , the continuous-time consistency training objective becomes

L 8

CTpθ, θ ´q “ 2E „ λptqfθpxt, tq

T ˆ

Bfθ´ pxt, tq

Bt `

Bfθ´ pxt, tq

Bxt ¨ xt ´ x t ˙ . (33)

Remark 10. Under suitable regularity conditions, we can combine the insights of Theorems 2 and 6 to derive lim ∆tÑ0 ∇θL

N

CDpθ, θ ´; φq ∆t “ ∇θL 8

CTpθ, θ ´q.

As in Theorems 2 and 6, here φ represents diffusion model parameters that satisfy sφpx, tq ” ∇ log ptpxq, and θ ´ “ stopgradpθq.

Remark 11. Similar to L 8

CDpθ, θ ´; φq in Theorem 5, L 8

CTpθ, θ ´q is a pseudo-objective; one cannot track training by monitoring the value of L 8

CTpθ, θ ´q, but can still apply gradient descent on this loss function to train a consistency model fθpx, tq directly from data. Moreover, the same observation in Remark 7 holds true: L 8

CTpθ, θ ´q “ 0 and ∇θL 8

CTpθ, θ ´q “ 0 if fθpx, tq matches the ground truth consistency function for the PF ODE.

#### B.3. Experimental Verifications
To experimentally verify the efficacy of our continuous-time CD and CT objectives, we train consistency models with a variety of loss functions on CIFAR-10. All results are provided in Fig. 7. Unless otherwise noted, models are trained with hyperparameters given in Table 3. We occasionally modify some hyperparameters for improved performance. For distillation, we compare the following objectives: 
* CD p` 2q: Consistency distillation L

N

CD with N “ 18 and the ` 2 metric. 
* CD p` 1q: Consistency distillation L

N

CD with N “ 18 and the ` 1 metric. We set the learning rate to 2e-4. 
* CD (LPIPS): Consistency distillation L

N

CD with N “ 18 and the LPIPS metric. 
* CD8 p` 2q: Consistency distillation L 8

CD in Theorem 3 with the ` 2 metric. We set the learning rate to 1e-3 and dropout to 0.13. 
* CD8 p` 1q: Consistency distillation L 8

CD in Theorem 4 with the ` 1 metric. We set the learning rate to 1e-3 and dropout to 0.3. 
* CD8 (stopgrad, ` 2): Consistency distillation L 8

CD in Theorem 5 with the ` 2 metric. We set the learning rate to 5e-6.



Table 3: Hyperparameters used for training CD and CT models

Hyperparameter CIFAR-10 ImageNet 64 ˆ 64 LSUN 256 ˆ 256

CD CT CD CT CD CT

Learning rate 4e-4 4e-4 8e-6 8e-6 1e-5 1e-5

Batch size 512 512 2048 2048 2048 2048 µ 0 0.95 0.95 µ0 0.9 0.95 0.95 s0 2 2 2 s1 150 200 150

EMA decay rate 0.9999 0.9999 0.999943 0.999943 0.999943 0.999943

Training iterations 800k 800k 600k 800k 600k 1000k

Mixed-Precision (FP16) No No Yes Yes Yes Yes

Dropout probability 0.0 0.0 0.0 0.0 0.0 0.0

Number of GPUs 8 8 64 64 64 64 
* CD8 (stopgrad, LPIPS): Consistency distillation L 8

CD in Theorem 5 with the LPIPS metric. We set the learning rate to 5e-6.

We did not investigate using the LPIPS metric in Theorem 3 because minimizing the resulting objective would require back-propagating through second order derivatives of the VGG network used in LPIPS, which is computationally expensive and prone to numerical instability. As revealed by Fig. 7a, the stopgrad version of continuous-time distillation (Theorem 5) works better than the non-stopgrad version (Theorem 3) for both the LPIPS and ` 2 metrics, and the LPIPS metric works the best for all distillation approaches. Additionally, discrete-time consistency distillation outperforms continuous-time consistency distillation, possibly due to the larger variance in continuous-time objectives, and the fact that one can use effective higher-order ODE solvers in discrete-time objectives.

For consistency training (CT), we find it important to initialize consistency models from a pre-trained EDM model in order to stabilize training when using continuous-time objectives. We hypothesize that this is caused by the large variance in our continuous-time loss functions. For fair comparison, we thus initialize all consistency models from the same pre-trained

EDM model on CIFAR-10 for both discrete-time and continuous-time CT, even though the former works well with random initialization. We leave variance reduction techniques for continuous-time CT to future research.

We empirically compare the following objectives: 
* CT (LPIPS): Consistency training L

N

CT with N “ 120 and the LPIPS metric. We set the learning rate to 4e-4, and the

EMA decay rate for the target network to 0.99. We do not use the schedule functions for N and µ here because they cause slower learning when the consistency model is initialized from a pre-trained EDM model. 
* CT8 p` 2q: Consistency training L 8

CT with the ` 2 metric. We set the learning rate to 5e-6. 
* CT8 (LPIPS): Consistency training L 8

CT with the LPIPS metric. We set the learning rate to 5e-6.

As shown in Fig. 7b, the LPIPS metric leads to improved performance for continuous-time CT. We also find that continuoustime CT outperforms discrete-time CT with the same LPIPS metric. This is likely due to the bias in discrete-time CT, as ∆t ą 0 in Theorem 2 for discrete-time objectives, whereas continuous-time CT has no bias since it implicitly drives ∆t to 0.

### C. Additional Experimental Details
Model Architectures We follow Song et al. (2021); Dhariwal & Nichol (2021) for model architectures. Specifically, we use the NCSN++ architecture in Song et al. (2021) for all CIFAR-10 experiments, and take the corresponding network architectures from Dhariwal & Nichol (2021) when performing experiments on ImageNet 64 ˆ 64, LSUN Bedroom 256 ˆ 256 and LSUN Cat 256 ˆ 256.

Parameterization for Consistency Models We use the same architectures for consistency models as those used for

EDMs. The only difference is we slightly modify the skip connections in EDM to ensure the boundary condition holds for

Consistency Models consistency models. Recall that in Section 3 we propose to parameterize a consistency model in the following form: fθpx, tq “ cskipptqx ` coutptqFθpx, tq.

In EDM (Karras et al., 2022), authors choose cskipptq “ σ 2 data t 2 ` σ 2 data , coutptq “ σdatat a σ 2 data ` t 2 , where σdata “ 0.5. However, this choice of cskip and cout does not satisfy the boundary condition when the smallest time instant  ‰ 0. To remedy this issue, we modify them to cskipptq “ σ 2 data pt ´  q 2 ` σ 2 data , coutptq “ σdatapt ´  q a σ 2 data ` t 2 , which clearly satisfies cskipp q “ 1 and coutp q “ 0.

Schedule Functions for Consistency Training As discussed in Section 5, consistency generation requires specifying schedule functions Np¨q and µp¨q for best performance. Throughout our experiments, we use schedule functions that take the form below:

Npkq “ Sc k

K pps1 ` 1q 2 ´ s 2 0 q ` s 2 0 ´ 1

W ` 1 µpkq “ exp ˆ s0 log µ0

Npkq ˙ , where K denotes the total number of training iterations, s0 denotes the initial discretization steps, s1 ą s0 denotes the target discretization steps at the end of training, and µ0 ą 0 denotes the EMA decay rate at the beginning of model training.

Training Details In both consistency distillation and progressive distillation, we distill EDMs (Karras et al., 2022). We trained these EDMs ourselves according to the specifications given in Karras et al. (2022). The original EDM paper did not provide hyperparameters for the LSUN Bedroom 256 ˆ 256 and Cat 256 ˆ 256 datasets, so we mostly used the same hyperparameters as those for the ImageNet 64 ˆ 64 dataset. The difference is that we trained for 600k and 300k iterations for the LSUN Bedroom and Cat datasets respectively, and reduced the batch size from 4096 to 2048.

We used the same EMA decay rate for LSUN 256 ˆ 256 datasets as for the ImageNet 64 ˆ 64 dataset. For progressive distillation, we used the same training settings as those described in Salimans & Ho (2022) for CIFAR-10 and ImageNet 64 ˆ 64. Although the original paper did not test on LSUN 256 ˆ 256 datasets, we used the same settings for ImageNet 64 ˆ 64 and found them to work well.

In all distillation experiments, we initialized the consistency model with pre-trained EDM weights. For consistency training, we initialized the model randomly, just as we did for training the EDMs. We trained all consistency models with the

Rectified Adam optimizer (Liu et al., 2019), with no learning rate decay or warm-up, and no weight decay. We also applied

EMA to the weights of the online consistency models in both consistency distillation and consistency training, as well as to the weights of the training online consistency models according to Karras et al. (2022). For LSUN 256 ˆ 256 datasets, we chose the EMA decay rate to be the same as that for ImageNet 64 ˆ 64, except for consistency distillation on LSUN

Bedroom 256 ˆ 256, where we found that using zero EMA worked better.

When using the LPIPS metric on CIFAR-10 and ImageNet 64 ˆ 64, we rescale images to resolution 224 ˆ 224 with bilinear upsampling before feeding them to the LPIPS network. For LSUN 256 ˆ 256, we evaluated LPIPS without rescaling inputs.

In addition, we performed horizontal flips for data augmentation for all models and on all datasets. We trained all models on a cluster of Nvidia A100 GPUs. Additional hyperparameters for consistency training and distillation are listed in Table 3.

### D. Additional Results on Zero-Shot Image Editing
With consistency models, we can perform a variety of zero-shot image editing tasks. As an example, we present additional results on colorization (Fig. 8), super-resolution (Fig. 9), inpainting (Fig. 10), interpolation (Fig. 11), denoising (Fig. 12),



Algorithm 4 Zero-Shot Image Editing 1: Input: Consistency model fθp¨, ¨q, sequence of time points t1 ą t2 ą ¨ ¨ ¨ ą tN , reference image y, invertible linear transformation A, and binary image mask Ω 2: y Ð A´1 rpAyq d p1 ´ Ωq ` 0 d Ωs 3: Sample x „ N py, t2 1Iq 4: x Ð fθpx, t1q 5: x Ð A´1 rpAyq d p1 ´ Ωq ` pAxq d Ωs 6: for n “ 2 to N do 7: Sample x „ N px,pt 2 n ´  2 qIq 8: x Ð fθpx, tnq 9: x Ð A´1 rpAyq d p1 ´ Ωq ` pAxq d Ωs 10: end for 11: Output: x and stroke-guided image generation (SDEdit, Meng et al. (2021), Fig. 13). The consistency model used here is trained via consistency distillation on the LSUN Bedroom 256 ˆ 256.

All these image editing tasks, except for image interpolation and denoising, can be performed via a small modification to the multistep sampling algorithm in Algorithm 1. The resulting pseudocode is provided in Algorithm 4. Here y is a reference image that guides sample generation, Ω is a binary mask, d computes element-wise products, and A is an invertible linear transformation that maps images into a latent space where the conditional information in y is infused into the iterative generation procedure by masking with Ω. Unless otherwise stated, we choose ti “ ˆ

T 1{ρ ` i ´ 1

N ´ 1 p 1{ρ ´ T 1{ρ q ˙ρ in our experiments, where N “ 40 for LSUN Bedroom 256 ˆ 256.

Below we describe how to perform each task using Algorithm 4.

Inpainting When using Algorithm 4 for inpainting, we let y be an image where missing pixels are masked out, Ω be a binary mask where 1 indicates the missing pixels, and A be the identity transformation.

Colorization The algorithm for image colorization is similar, as colorization becomes a special case of inpainting once we transform data into a decoupled space. Specifically, let y P R hˆwˆ3 be a gray-scale image that we aim to colorize, where all channels of y are assumed to be the same, i.e., yr:, :, 0s “ yr:, :, 1 each channel of this gray scale image is obtained from a colorful image by averaging the RGB channels with s “ yr:, :, 2s in NumPy notation. In our experiments,

##### 0.2989R ` 0.5870G ` 0.1140B.
We define Ω P t0, 1u hˆwˆ3 to be a binary mask such that

Ωri, j, ks “ # 1, k “ 1 or 2 0, k “ 0 .

Let Q P R 3ˆ3 be an orthogonal matrix whose first column is proportional to the vector p0.2989, 0.5870, 0.1140q. This orthogonal matrix can be obtained easily via QR decomposition, and we use the following in our experiments

Q “ ¨ ˝

#### 0.4471 ´0.8204 0.3563
### 0.8780 0.4785 0
#### 0.1705 ´0.3129 ´0.9343
 ˛ ‚.

We then define the linear transformation A : x P R hˆwˆ3

ÞÑ y P R hˆwˆ3 , where yri, j, ks “ 2 l ÿ “0 xri, j, lsQrl, ks.



Because Q is orthogonal, the inversion A´1 : y P R hˆw ÞÑ x P R hˆwˆ3 is easy to compute, where xri, j, ks “ 2 l ÿ “0 yri, j, lsQrk, ls.

With A and Ω defined as above, we can now use Algorithm 4 for image colorization.

Super-resolution With a similar strategy, we employ Algorithm 4 for image super-resolution. For simplicity, we assume that the down-sampled image is obtained by averaging non-overlapping patches of size p ˆ p. Suppose the shape of full resolution images is h ˆ w ˆ 3. Let y P R hˆwˆ3 denote a low-resolution image naively up-sampled to full resolution, where pixels in each non-overlapping patch share the same value. Additionally, let Ω P t0, 1u h{pˆw{pˆp 2ˆ3 be a binary mask such that

Ωri, j, k, ls “ # 1, k ě 1 0, k “ 0 .

Similar to image colorization, super-resolution requires an orthogonal matrix Q P R p 2ˆp 2 whose first column is p 1{p, 1{p, ¨ ¨ ¨ , 1{pq. This orthogonal matrix can be obtained with QR decomposition. To perform super-resolution, we define the linear transformation A : x P R hˆwˆ3

ÞÑ y P R h{pˆw{pˆp 2ˆ3 , where yri, j, k, ls “ p 2´1 m ÿ “0 xri ˆ p ` pm ´ m mod pq{p, j ˆ p ` m mod p, lsQrm, ks.

The inverse transformation A´1 : y P R h{pˆw{pˆp 2ˆ3

ÞÑ x P R hˆwˆ3 is easy to derive, with xri, j, k, ls “ p 2´1 m ÿ “0 yri ˆ p ` pm ´ m mod pq{p, j ˆ p ` m mod p, lsQrk, ms.

Above definitions of A and Ω allow us to use Algorithm 4 for image super-resolution.

Stroke-guided image generation We can also use Algorithm 4 for stroke-guided image generation as introduced in

SDEdit (Meng et al., 2021). Specifically, we let y P R hˆwˆ3 be a stroke painting. We set A “ I, and define Ω P R hˆwˆ3 as a matrix of ones. In our experiments, we set t1 “ 5.38 and t2 “ 2.24, with N “ 2.

Denoising It is possible to denoise images perturbed with various scales of Gaussian noise using a single consistency model. Suppose the input image x is perturbed with N p0; σ 2Iq. As long as σ P r, Ts, we can evaluate fθpx, σq to produce the denoised image.

Interpolation We can interpolate between two images generated by consistency models. Suppose the first sample x1 is produced by noise vector z1, and the second sample x2 is produced by noise vector z2. In other words, x1 “ fθpz1, Tq and x2 “ fθpz2, Tq. To interpolate between x1 and x2, we first use spherical linear interpolation to get z “ sinrp1 ´ αqψs sinpψq z1 ` sinpαψq sinpψq z2, where α P r0, 1s and ψ “ arccosp z

T 1z2 k z1k 2 k z2k 2 q, then evaluate fθpz, Tq to produce the interpolated image.

### E. Additional Samples from Consistency Models
We provide additional samples from consistency distillation (CD) and consistency training (CT) on CIFAR-10 (Figs. 14 and 18), ImageNet 64 ˆ 64 (Figs. 15 and 19), LSUN Bedroom 256 ˆ 256 (Figs. 16 and 20) and LSUN Cat 256 ˆ 256 (Figs. 17 and 21).

Figure 8: Gray-scale images (left), colorized images by a consistency model (middle), and ground truth (right).

Figure 9: Downsampled images of resolution 32 ˆ 32 (left), full resolution (256 ˆ 256) images generated by a consistency model (middle), and ground truth images of resolution 256 ˆ 256 (right).

Figure 10: Masked images (left), imputed images by a consistency model (middle), and ground truth (right).

Figure 11: Interpolating between leftmost and rightmost images with spherical linear interpolation. All samples are generated by a consistency model trained on LSUN Bedroom 256 ˆ 256.

Figure 12: Single-step denoising with a consistency model. The leftmost images are ground truth. For every two rows, the top row shows noisy images with different noise levels, while the bottom row gives denoised images.

Figure 13: SDEdit with a consistency model. The leftmost images are stroke painting inputs. Images on the right side are the results of stroke-guided image generation (SDEdit).

Figure 14: Uncurated samples from CIFAR-10 32 ˆ 32. All corresponding samples use the same initial noise.

Figure 15: Uncurated samples from ImageNet 64 ˆ 64. All corresponding samples use the same initial noise.

Figure 16: Uncurated samples from LSUN Bedroom 256 ˆ 256. All corresponding samples use the same initial noise.

Figure 17: Uncurated samples from LSUN Cat 256 ˆ 256. All corresponding samples use the same initial noise.

Figure 18: Uncurated samples from CIFAR-10 32 ˆ 32. All corresponding samples use the same initial noise.

Figure 19: Uncurated samples from ImageNet 64 ˆ 64. All corresponding samples use the same initial noise.


Figure 20: Uncurated samples from LSUN Bedroom 256 ˆ 256. All corresponding samples use the same initial noise.


Figure 21: Uncurated samples from LSUN Cat 256 ˆ 256. All corresponding samples use the same initial noise.

