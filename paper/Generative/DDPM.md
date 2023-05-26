# Denoising Diffusion Probabilistic Models
去噪扩散概率模型 2020.6.19 https://arxiv.org/abs/2006.11239

## Abstract
We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion. 

我们使用扩散概率模型呈现高质量的图像合成结果，扩散概率模型是一类受非平衡热力学考虑启发的潜在变量模型。我们的最佳结果是通过根据扩散概率模型与Langevin动力学的去噪分数匹配之间的新联系设计的加权变分边界进行训练而获得的，并且我们的模型自然地承认渐进式有损解压缩方案，可以解释为自回归解码的推广。在无条件的CIFAR10数据集上，我们获得了9.46的初始分数和3.17的FID分数。在256x256 LSUN上，我们获得了类似于ProgressiveGAN的样品质量。我们的实施可在 https://github.com/hojonathanho/diffusion 获得。

## 1 Introduction
Deep generative models of all kinds have recently exhibited high quality samples in a wide variety of data modalities. Generative adversarial networks (GANs), autoregressive models, flows, and variational autoencoders (VAEs) have synthesized striking image and audio samples [14, 27, 3, 58, 38, 25, 10, 32, 44, 57, 26, 33, 45], and there have been remarkable advances in energy-based modeling and score matching that have produced images comparable to those of GANs [11, 55].

各种深度生成模型最近展示了各种数据模式的高质量样本。生成对抗网络(GAN)、自回归模型、流和变分自编码器(VAE)已经合成了引人注目的图像和音频样本[14，27，3，58，38，25，10，32，44，57，26，33，45]，并且在基于能量的建模和分数匹配方面取得了显著进展，产生了与GAN相当的图像[11，55]。

Figure 1: Generated samples on CelebA-HQ 256 × 256 (left) and unconditional CIFAR10 (right) 
图 1：在 CelebA-HQ 256 × 256(左)和无条件 CIFAR10(右)上生成的样本

![Figure 2](./images/ddpm/fig_2.png)<br/>
Figure 2: The directed graphical model considered in this work.
图 2：本文中考虑的有向图模型。

This paper presents progress in diffusion probabilistic models [53]. A diffusion probabilistic model (which we will call a “diffusion model” for brevity) is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. Transitions of this chain are learned to reverse a diffusion process, which is a Markov chain that gradually adds noise to the data in the opposite direction of sampling until signal is destroyed. When the diffusion consists of small amounts of Gaussian noise, it is sufficient to set the sampling chain transitions to conditional Gaussians too, allowing for a particularly simple neural network parameterization.

本文介绍了扩散概率模型的研究进展[53]。扩散概率模型(为简洁起见，我们称之为“扩散模型”)是使用变分推理训练的参数化马尔可夫链，可在有限时间后生成与数据匹配的样本。该链的跃迁被学习以逆转扩散过程，这是一个马尔可夫链，它逐渐向采样相反方向的数据添加噪声，直到信号被破坏。当扩散由少量高斯噪声组成时，将采样链跃迁设置为条件高斯也就足够了，从而实现特别简单的神经网络参数化。

Diffusion models are straightforward to define and efficient to train, but to the best of our knowledge, there has been no demonstration that they are capable of generating high quality samples. We show that diffusion models actually are capable of generating high quality samples, sometimes better than the published results on other types of generative models (Section 4). In addition, we show that a certain parameterization of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and with annealed Langevin dynamics during sampling (Section 3.2) [55, 61]. We obtained our best sample quality results using this parameterization (Section 4.2), so we consider this equivalence to be one of our primary contributions.

扩散模型易于定义且训练效率高，但据我们所知，尚未证明它们能够生成高质量的样本。我们表明，扩散模型实际上能够生成高质量的样本，有时比其他类型的生成模型的已发表结果更好(第4节)。此外，我们表明，扩散模型的某种参数化揭示了训练期间在多个噪声水平上的去噪分数匹配以及采样期间退火的Langevin动力学的等价性(第3.2节)[55，61]。我们使用此参数化(第 4.2 节)获得了最佳的样本质量结果，因此我们认为这种等价性是我们的主要贡献之一。

Despite their sample quality, our models do not have competitive log likelihoods compared to other likelihood-based models (our models do, however, have log likelihoods better than the large estimates annealed importance sampling has been reported to produce for energy based models and score matching [11, 55]). We find that the majority of our models’ lossless codelengths are consumed to describe imperceptible image details (Section 4.3). We present a more refined analysis of this phenomenon in the language of lossy compression, and we show that the sampling procedure of diffusion models is a type of progressive decoding that resembles autoregressive decoding along a bit ordering that vastly generalizes what is normally possible with autoregressive models. 

尽管样本质量高，但与其他基于可能性的模型相比，我们的模型没有竞争性的对数似然(然而，我们的模型确实具有比基于能量的模型和分数匹配产生的大型估计退火重要性抽样更好的对数似然[11，55])。我们发现，我们模型的大部分无损代码长度都用于描述难以察觉的图像细节(第 4.3 节)。我们用有损压缩的语言对这种现象进行了更精细的分析，并且我们表明扩散模型的采样过程是一种渐进式解码，类似于沿位顺序的自回归解码，极大地概括了自回归模型通常可能发生的事情。

## 2 Background
Diffusion models [53] are latent variable models of the form pθ(x0) :=

R pθ(x0:T ) dx1:T , where x1, . . . , xT are latents of the same dimensionality as the data x0 ∼ q(x0). The joint distribution pθ(x0:T ) is called the reverse process, and it is defined as a Markov chain with learned Gaussian transitions starting at p(xT ) = N (xT ; 0, I): pθ(x0:T ) := p(xT )

T

Y t=1 pθ(xt−1|xt), pθ(xt−1|xt) := N (xt−1; µθ (xt, t), Σθ(xt, t)) (1)

What distinguishes diffusion models from other types of latent variable models is that the approximate posterior q(x1:T |x0), called the forward process or diffusion process, is fixed to a Markov chain that gradually adds Gaussian noise to the data according to a variance schedule β1, . . . , βT : q(x1:T |x0) :=

T

Y t=1 q(xt|xt−1), q(xt|xt−1) := N (xt; p 1 − βtxt−1, βtI) (2)

Training is performed by optimizing the usual variational bound on negative log likelihood:

E [− log pθ(x0)] ≤ Eq  − log pθ(x0:T ) q(x1:T |x0)  = Eq  − log p(xT ) −

X t≥1 log pθ(xt−1|xt) q(xt|xt−1)  =: L (3)

The forward process variances βt can be learned by reparameterization [33] or held constant as hyperparameters, and expressiveness of the reverse process is ensured in part by the choice of

Gaussian conditionals in pθ(xt−1|xt), because both processes have the same functional form when βt are small [53]. A notable property of the forward process is that it admits sampling xt at an arbitrary timestep t in closed form: using the notation αt := 1 − βt and ¯αt :=

Q t s=1 αs, we have q(xt|x0) = N (xt; √ ¯αtx0,(1 − ¯αt)I) (4) 2 −!

Efficient training is therefore possible by optimizing random terms of L with stochastic gradient descent. Further improvements come from variance reduction by rewriting L (3) as:

Eq  DKL(q(xT |x0) k p(xT )) | {z }

LT +

X t>1

DKL(q(xt−1|xt, x0) k pθ(xt−1|xt)) | {z }

Lt−1 − log pθ(x0|x1) | {z

L0 }  (5) (See Appendix A for details. The labels on the terms are used in Section 3.) Equation (5) uses KL divergence to directly compare pθ(xt−1|xt) against forward process posteriors, which are tractable when conditioned on x0: q(xt−1|xt, x0) = N (xt−1; ˜µt (xt, x0), β˜ tI), (6) where ˜µt (xt, x0) := √ ¯αt−1βt 1 − ¯αt x0 + √ αt(1 − ¯αt−1) 1 − ¯αt xt and β˜ t := 1 − ¯αt−1 1 − ¯αt βt (7)

Consequently, all KL divergences in Eq. (5) are comparisons between Gaussians, so they can be calculated in a Rao-Blackwellized fashion with closed form expressions instead of high variance

Monte Carlo estimates. 

## 3 Diffusion models and denoising autoencoders

Diffusion models might appear to be a restricted class of latent variable models, but they allow a large number of degrees of freedom in implementation. One must choose the variances βt of the forward process and the model architecture and Gaussian distribution parameterization of the reverse process. To guide our choices, we establish a new explicit connection between diffusion models and denoising score matching (Section 3.2) that leads to a simplified, weighted variational bound objective for diffusion models (Section 3.4). Ultimately, our model design is justified by simplicity and empirical results (Section 4). Our discussion is categorized by the terms of Eq. (5).

## 3.1 Forward process and LT
We ignore the fact that the forward process variances βt are learnable by reparameterization and instead fix them to constants (see Section 4 for details). Thus, in our implementation, the approximate posterior q has no learnable parameters, so LT is a constant during training and can be ignored.

## 3.2 Reverse process and L1:T −1
Now we discuss our choices in pθ(xt−1|xt) = N (xt−1; µθ (xt, t), Σθ(xt, t)) for 1 < t ≤ T. First, we set Σθ(xt, t) = σt 2

I to untrained time dependent constants. Experimentally, both σt 2 = βt and σt 2 = β˜ t = 1−¯αt−1 1−¯αt βt had similar results. The first choice is optimal for x0 ∼ N (0, I), and the second is optimal for x0 deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance [53].

Second, to represent the mean µθ (xt, t), we propose a specific parameterization motivated by the following analysis of Lt. With pθ(xt−1|xt) = N (xt−1; µθ (xt, t), σt 2

I), we can write:

Lt−1 = Eq  1 2σ 2 t k ˜µt (xt, x0) − µθ (xt, t)k 2  + C (8) where C is a constant that does not depend on θ. So, we see that the most straightforward parameterization of µθ is a model that predicts ˜µt , the forward process posterior mean. However, we can expand

Eq. (8) further by reparameterizing Eq. (4) as xt(x0,  ) = √ ¯αtx0 + √ 1 − ¯αt for  ∼ N (0, I) and applying the forward process posterior formula (7):

Lt−1 − C = Ex0, " 1 2σ 2 t         ˜µt  xt(x0,  ), √ 1 ¯αt (xt(x0,  ) − √ 1 − ¯αt )  − µθ (xt(x0,  ), t)        2

#
 (9) = Ex0, " 1 2σ 2 t         1 √ αt  xt(x0,  ) − βt √ 1 − ¯αt   − µθ (xt(x0,  ), t)        2

#
 (10) 3

Algorithm 1 Training 1: repeat 2: x0 ∼ q(x0) 3: t ∼ Uniform({1, . . . , T}) 4:  ∼ N (0, I) 5: Take gradient descent step on ∇θ     −  θ( √ ¯αtx0 + √ 1 − ¯αt , t)    2 6: until converged

Algorithm 2 Sampling 1: xT ∼ N (0, I) 2: for t = T, . . . , 1 do 3: z ∼ N (0, I) if t > 1, else z = 0 4: xt−1 = √1 αt  xt − 1−αt √1−¯αt  θ(xt, t)  + σtz 5: end for 6: return x0

Equation (10) reveals that µθ must predict √ 1 αt  xt − βt √ 1−¯αt input to the model, we may choose the parameterization   given xt. Since xt is available as µθ (xt, t) = ˜µt  xt, √ 1 ¯αt (xt − √ 1 − ¯αt θ(xt)) = √ 1 αt  xt − βt √ 1 − ¯αt  θ(xt, t)  (11) where  θ is a function approximator intended to predict  from xt. To sample xt−1 ∼ pθ(xt−1|xt) is to compute xt−1 = √ 1 αt  xt − βt √ 1−¯αt  θ(xt, t)  +σtz, where z ∼ N (0, I). The complete sampling procedure, Algorithm 2, resembles Langevin dynamics with  θ as a learned gradient of the data density. Furthermore, with the parameterization (11), Eq. (10) simplifies to:

Ex0,  β 2 t 2σt 2αt(1 − ¯αt)      −  θ( √ ¯αtx0 + √ 1 − ¯αt , t)    2  (12) which resembles denoising score matching over multiple noise scales indexed by t [55]. As Eq. (12) is equal to (one term of) the variational bound for the Langevin-like reverse process (11), we see that optimizing an objective resembling denoising score matching is equivalent to using variational inference to fit the finite-time marginal of a sampling chain resembling Langevin dynamics.

To summarize, we can train the reverse process mean function approximator µθ to predict ˜µt , or by modifying its parameterization, we can train it to predict  . (There is also the possibility of predicting x0, but we found this to lead to worse sample quality early in our experiments.) We have shown that the  -prediction parameterization both resembles Langevin dynamics and simplifies the diffusion model’s variational bound to an objective that resembles denoising score matching. Nonetheless, it is just another parameterization of pθ(xt−1|xt), so we verify its effectiveness in Section 4 in an ablation where we compare predicting  against predicting ˜µt .

## 3.3 Data scaling, reverse process decoder, and L0
We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]. This ensures that the neural network reverse process operates on consistently scaled inputs starting from the standard normal prior p(xT ). To obtain discrete log likelihoods, we set the last term of the reverse process to an independent discrete decoder derived from the Gaussian N (x0; µθ (x1, 1), σ2 1

I): pθ(x0|x1) =

D i

Y=1

Z δ+(x i 0 ) δ−(x i 0 )

N (x; µ i θ (x1, 1), σ2 1 ) dx δ+(x) =  ∞ if x = 1 x + 1 255 if x < 1 δ−(x) =  −∞ if x = −1 x − 1 255 if x > −1 (13) where D is the data dimensionality and the i superscript indicates extraction of one coordinate. (It would be straightforward to instead incorporate a more powerful decoder like a conditional autoregressive model, but we leave that to future work.) Similar to the discretized continuous distributions used in VAE decoders and autoregressive models [34, 52], our choice here ensures that the variational bound is a lossless codelength of discrete data, without need of adding noise to the data or incorporating the Jacobian of the scaling operation into the log likelihood. At the end of sampling, we display µθ (x1, 1) noiselessly.

## 3.4 Simplified training objective
With the reverse process and decoder defined above, the variational bound, consisting of terms derived from Eqs. (12) and (13), is clearly differentiable with respect to θ and is ready to be employed for 4

Table 1: CIFAR10 results. NLL measured in bits/dim.

Model IS FID NLL Test (Train)

Conditional

EBM [11] 8.30 37.9

JEM [17] 8.76 38.4

BigGAN [3] 9.22 14.73

StyleGAN2 + ADA (v1) [29] 10.06 2.67

Unconditional

Diffusion (original) [53] ≤ 5.40

Gated PixelCNN [59] 4.60 65.93 3.03 (2.90)

Sparse Transformer [7] 2.80

PixelIQN [43] 5.29 49.46

EBM [11] 6.78 38.2

NCSNv2 [56] 31.75

NCSN [55] 8.87±0.12 25.32

SNGAN [39] 8.22±0.05 21.7

SNGAN-DDLS [4] 9.09±0.10 15.42

StyleGAN2 + ADA (v1) [29] 9.74 ± 0.05 3.26

Ours (L, fixed isotropic Σ) 7.67±0.13 13.51 ≤ 3.70 (3.69)

Ours (Lsimple) 9.46±0.11 3.17 ≤ 3.75 (3.72)<br/>
Table 2: Unconditional CIFAR10 reverse process parameterization and training objective ablation. Blank entries were unstable to train and generated poor samples with out-ofrange scores.

Objective IS FID ˜µ prediction (baseline)

L, learned diagonal Σ 7.28±0.10 23.69

L, fixed isotropic Σ 8.06±0.09 13.22 k ˜µ − ˜µθk 2 – –  prediction (ours)

L, learned diagonal Σ – –

L, fixed isotropic Σ 7.67±0.13 13.51 k ˜ −  θk 2 (Lsimple) 9.46±0.11 3.17 training. However, we found it beneficial to sample quality (and simpler to implement) to train on the following variant of the variational bound:

Lsimple(θ) := Et,x0, h     −  θ( √ ¯αtx0 + √ 1 − ¯αt , t)    2 i (14) where t is uniform between 1 and T. The t = 1 case corresponds to L0 with the integral in the discrete decoder definition (13) approximated by the Gaussian probability density function times the bin width, ignoring σ 2 1 and edge effects. The t > 1 cases correspond to an unweighted version of

Eq. (12), analogous to the loss weighting used by the NCSN denoising score matching model [55]. (LT does not appear because the forward process variances βt are fixed.) Algorithm 1 displays the complete training procedure with this simplified objective.

Since our simplified objective (14) discards the weighting in Eq. (12), it is a weighted variational bound that emphasizes different aspects of reconstruction compared to the standard variational bound [18, 22]. In particular, our diffusion process setup in Section 4 causes the simplified objective to down-weight loss terms corresponding to small t. These terms train the network to denoise data with very small amounts of noise, so it is beneficial to down-weight them so that the network can focus on more difficult denoising tasks at larger t terms. We will see in our experiments that this reweighting leads to better sample quality. 

## 4 Experiments

We set T = 1000 for all experiments so that the number of neural network evaluations needed during sampling matches previous work [53, 55]. We set the forward process variances to constants increasing linearly from β1 = 10−4 to βT = 0.02. These constants were chosen to be small relative to data scaled to [−1, 1], ensuring that reverse and forward processes have approximately the same functional form while keeping the signal-to-noise ratio at xT as small as possible (LT =

DKL(q(xT |x0) k N (0, I)) ≈ 10−5 bits per dimension in our experiments).

To represent the reverse process, we use a U-Net backbone similar to an unmasked PixelCNN++ [52, 48] with group normalization throughout [66]. Parameters are shared across time, which is specified to the network using the Transformer sinusoidal position embedding [60]. We use self-attention at the 16 × 16 feature map resolution [63, 60]. Details are in Appendix B.

## 4.1 Sample quality
Table 1 shows Inception scores, FID scores, and negative log likelihoods (lossless codelengths) on

CIFAR10. With our FID score of 3.17, our unconditional model achieves better sample quality than most models in the literature, including class conditional models. Our FID score is computed with respect to the training set, as is standard practice; when we compute it with respect to the test set, the score is 5.24, which is still better than many of the training set FID scores in the literature. 5

Figure 3: LSUN Church samples. FID=7.89 Figure 4: LSUN Bedroom samples. FID=4.90

Algorithm 3 Sending x0 1: Send xT ∼ q(xT |x0) using p(xT ) 2: for t = T − 1, . . . , 2, 1 do 3: Send xt ∼ q(xt|xt+1, x0) using pθ(xt|xt+1) 4: end for 5: Send x0 using pθ(x0|x1)

Algorithm 4 Receiving 1: Receive xT using p(xT ) 2: for t = T − 1, . . . , 1, 0 do 3: Receive xt using pθ(xt|xt+1) 4: end for 5: return x0

We find that training our models on the true variational bound yields better codelengths than training on the simplified objective, as expected, but the latter yields the best sample quality. See Fig. 1 for

CIFAR10 and CelebA-HQ 256 × 256 samples, Fig. 3 and Fig. 4 for LSUN 256 × 256 samples [71], and Appendix D for more.

## 4.2 Reverse process parameterization and training objective ablation
In Table 2, we show the sample quality effects of reverse process parameterizations and training objectives (Section 3.2). We find that the baseline option of predicting ˜µ works well only when trained on the true variational bound instead of unweighted mean squared error, a simplified objective akin to Eq. (14). We also see that learning reverse process variances (by incorporating a parameterized diagonal Σθ(xt) into the variational bound) leads to unstable training and poorer sample quality compared to fixed variances. Predicting  , as we proposed, performs approximately as well as predicting ˜µ when trained on the variational bound with fixed variances, but much better when trained with our simplified objective.

## 4.3 Progressive coding
Table 1 also shows the codelengths of our CIFAR10 models. The gap between train and test is at most 0.03 bits per dimension, which is comparable to the gaps reported with other likelihood-based models and indicates that our diffusion model is not overfitting (see Appendix D for nearest neighbor visualizations). Still, while our lossless codelengths are better than the large estimates reported for energy based models and score matching using annealed importance sampling [11], they are not competitive with other types of likelihood-based generative models [7].

Since our samples are nonetheless of high quality, we conclude that diffusion models have an inductive bias that makes them excellent lossy compressors. Treating the variational bound terms L1 +· · ·+LT as rate and L0 as distortion, our CIFAR10 model with the highest quality samples has a rate of 1.78 bits/dim and a distortion of 1.97 bits/dim, which amounts to a root mean squared error of 0.95 on a scale from 0 to 255. More than half of the lossless codelength describes imperceptible distortions.

Progressive lossy compression We can probe further into the rate-distortion behavior of our model by introducing a progressive lossy code that mirrors the form of Eq. (5): see Algorithms 3 and 4, which assume access to a procedure, such as minimal random coding [19, 20], that can transmit a sample x ∼ q(x) using approximately DKL(q(x) k p(x)) bits on average for any distributions p and q, for which only p is available to the receiver beforehand. When applied to x0 ∼ q(x0), Algorithms 3 and 4 transmit xT , . . . , x0 in sequence using a total expected codelength equal to Eq. (5). The receiver, 6 at any time t, has the partial information xt fully available and can progressively estimate: x0 ≈ ˆx0 =  xt − √ 1 − ¯αt θ(xt)  / √ ¯αt (15) due to Eq. (4). (A stochastic reconstruction x0 ∼ pθ(x0|xt) is also valid, but we do not consider it here because it makes distortion more difficult to evaluate.) Figure 5 shows the resulting ratedistortion plot on the CIFAR10 test set. At each time t, the distortion is calculated as the root mean squared error p k x0 − ˆx0k 2/D, and the rate is calculated as the cumulative number of bits received so far at time t. The distortion decreases steeply in the low-rate region of the rate-distortion plot, indicating that the majority of the bits are indeed allocated to imperceptible distortions. 0 200 400 600 800 1,000 0 20 40 60 80

Reverse process steps (T − t) 0 200 400 600 800 1,000 0

## 0.5
 1

## 1.5
Reverse process steps (T − t) 0 0.5 1 1.5 0 20 40 60 80

Rate (bits/dim)<br/>
Figure 5: Unconditional CIFAR10 test set rate-distortion vs. time. Distortion is measured in root mean squared error on a [0, 255] scale. See Table 4 for details.

Progressive generation We also run a progressive unconditional generation process given by progressive decompression from random bits. In other words, we predict the result of the reverse process, ˆx0, while sampling from the reverse process using Algorithm 2. Figures 6 and 10 show the resulting sample quality of ˆx0 over the course of the reverse process. Large scale image features appear first and details appear last. Figure 7 shows stochastic predictions x0 ∼ pθ(x0|xt) with xt frozen for various t. When t is small, all but fine details are preserved, and when t is large, only large scale features are preserved. Perhaps these are hints of conceptual compression [18].

Figure 6: Unconditional CIFAR10 progressive generation (ˆx0 over time, from left to right). Extended samples and sample quality metrics over time in the appendix (Figs. 10 and 14).

Figure 7: When conditioned on the same latent, CelebA-HQ 256 × 256 samples share high-level attributes.

Bottom-right quadrants are xt, and other quadrants are samples from pθ(x0|xt).

Connection to autoregressive decoding Note that the variational bound (5) can be rewritten as:

L = DKL(q(xT ) k p(xT )) + Eq "

X t≥1

DKL(q(xt−1|xt) k pθ(xt−1|xt)) # + H(x0) (16) (See Appendix A for a derivation.) Now consider setting the diffusion process length T to the dimensionality of the data, defining the forward process so that q(xt|x0) places all probability mass on x0 with the first t coordinates masked out (i.e. q(xt|xt−1) masks out the t th coordinate), setting p(xT ) to place all mass on a blank image, and, for the sake of argument, taking pθ(xt−1|xt) to 7

Distortion (RMSE)

Rate (bits/dim)

Distortion (RMSE)<br/>
Figure 8: Interpolations of CelebA-HQ 256x256 images with 500 timesteps of diffusion. be a fully expressive conditional distribution. With these choices, DKL(q(xT ) k p(xT )) = 0, and minimizing DKL(q(xt−1|xt) k pθ(xt−1|xt)) trains pθ to copy coordinates t + 1, . . . , T unchanged and to predict the t th coordinate given t + 1, . . . , T. Thus, training pθ with this particular diffusion is training an autoregressive model.

We can therefore interpret the Gaussian diffusion model (2) as a kind of autoregressive model with a generalized bit ordering that cannot be expressed by reordering data coordinates. Prior work has shown that such reorderings introduce inductive biases that have an impact on sample quality [38], so we speculate that the Gaussian diffusion serves a similar purpose, perhaps to greater effect since

Gaussian noise might be more natural to add to images compared to masking noise. Moreover, the

Gaussian diffusion length is not restricted to equal the data dimension; for instance, we use T = 1000, which is less than the dimension of the 32 × 32 × 3 or 256 × 256 × 3 images in our experiments.

Gaussian diffusions can be made shorter for fast sampling or longer for model expressiveness.

## 4.4 Interpolation
We can interpolate source images x0, x 00 ∼ q(x0) in latent space using q as a stochastic encoder, xt, x 0t ∼ q(xt|x0), then decoding the linearly interpolated latent ¯xt = (1 − λ)x0 + λx 00 into image space by the reverse process, ¯x0 ∼ p(x0|¯xt). In effect, we use the reverse process to remove artifacts from linearly interpolating corrupted versions of the source images, as depicted in Fig. 8 (left). We fixed the noise for different values of λ so xt and x 0t remain the same. Fig. 8 (right) shows interpolations and reconstructions of original CelebA-HQ 256 × 256 images (t = 500). The reverse process produces high-quality reconstructions, and plausible interpolations that smoothly vary attributes such as pose, skin tone, hairstyle, expression and background, but not eyewear. Larger t results in coarser and more varied interpolations, with novel samples at t = 1000 (Appendix Fig. 9). 

## 5 Related Work

While diffusion models might resemble flows [9, 46, 10, 32, 5, 16, 23] and VAEs [33, 47, 37], diffusion models are designed so that q has no parameters and the top-level latent xT has nearly zero mutual information with the data x0. Our  -prediction reverse process parameterization establishes a connection between diffusion models and denoising score matching over multiple noise levels with annealed Langevin dynamics for sampling [55, 56]. Diffusion models, however, admit straightforward log likelihood evaluation, and the training procedure explicitly trains the Langevin dynamics sampler using variational inference (see Appendix C for details). The connection also has the reverse implication that a certain weighted form of denoising score matching is the same as variational inference to train a Langevin-like sampler. Other methods for learning transition operators of Markov chains include infusion training [2], variational walkback [15], generative stochastic networks [1], and others [50, 54, 36, 42, 35, 65].

By the known connection between score matching and energy-based modeling, our work could have implications for other recent work on energy-based models [67–69, 12, 70, 13, 11, 41, 17, 8]. Our rate-distortion curves are computed over time in one evaluation of the variational bound, reminiscent of how rate-distortion curves can be computed over distortion penalties in one run of annealed importance sampling [24]. Our progressive decoding argument can be seen in convolutional DRAW and related models [18, 40] and may also lead to more general designs for subscale orderings or sampling strategies for autoregressive models [38, 64]. 8 

## 6 Conclusion

We have presented high quality image samples using diffusion models, and we have found connections among diffusion models and variational inference for training Markov chains, denoising score matching and annealed Langevin dynamics (and energy-based models by extension), autoregressive models, and progressive lossy compression. Since diffusion models seem to have excellent inductive biases for image data, we look forward to investigating their utility in other data modalities and as components in other types of generative models and machine learning systems.

## Broader Impact

Our work on diffusion models takes on a similar scope as existing work on other types of deep generative models, such as efforts to improve the sample quality of GANs, flows, autoregressive models, and so forth. Our paper represents progress in making diffusion models a generally useful tool in this family of techniques, so it may serve to amplify any impacts that generative models have had (and will have) on the broader world.

Unfortunately, there are numerous well-known malicious uses of generative models. Sample generation techniques can be employed to produce fake images and videos of high profile figures for political purposes. While fake images were manually created long before software tools were available, generative models such as ours make the process easier. Fortunately, CNN-generated images currently have subtle flaws that allow detection [62], but improvements in generative models may make this more difficult. Generative models also reflect the biases in the datasets on which they are trained. As many large datasets are collected from the internet by automated systems, it can be difficult to remove these biases, especially when the images are unlabeled. If samples from generative models trained on these datasets proliferate throughout the internet, then these biases will only be reinforced further.

On the other hand, diffusion models may be useful for data compression, which, as data becomes higher resolution and as global internet traffic increases, might be crucial to ensure accessibility of the internet to wide audiences. Our work might contribute to representation learning on unlabeled raw data for a large range of downstream tasks, from image classification to reinforcement learning, and diffusion models might also become viable for creative uses in art, photography, and music.

## Acknowledgments and Disclosure of Funding

This work was supported by ONR PECASE and the NSF Graduate Research Fellowship under grant number DGE-1752814. Google’s TensorFlow Research Cloud (TFRC) provided Cloud TPUs.

## References
1. Guillaume Alain, Yoshua Bengio, Li Yao, Jason Yosinski, Eric Thibodeau-Laufer, Saizheng Zhang, and Pascal Vincent. GSNs: generative stochastic networks. Information and Inference: A Journal of the IMA, 5(2):210–249, 2016.
2. Florian Bordes, Sina Honari, and Pascal Vincent. Learning to generate samples from noise through infusion training. In International Conference on Learning Representations, 2017.
3. Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.
4. Tong Che, Ruixiang Zhang, Jascha Sohl-Dickstein, Hugo Larochelle, Liam Paull, Yuan Cao, and Yoshua Bengio. Your GAN is secretly an energy-based model and you should use discriminator driven latent sampling. arXiv preprint arXiv:2003.06060, 2020.
5. Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, pages 6571–6583, 2018.
6. Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. PixelSNAIL: An improved autoregressive generative model. In International Conference on Machine Learning, pages 863–871, 2018.
7. Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019. 9
8. Yuntian Deng, Anton Bakhtin, Myle Ott, Arthur Szlam, and Marc’Aurelio Ranzato. Residual energy-based models for text generation. arXiv preprint arXiv:2004.11714, 2020.
9. Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.
10. Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. arXiv preprint arXiv:1605.08803, 2016.
11. Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. In Advances in Neural Information Processing Systems, pages 3603–3613, 2019.
12. Ruiqi Gao, Yang Lu, Junpei Zhou, Song-Chun Zhu, and Ying Nian Wu. Learning generative ConvNets via multi-grid modeling and sampling. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 9155–9164, 2018.
13. Ruiqi Gao, Erik Nijkamp, Diederik P Kingma, Zhen Xu, Andrew M Dai, and Ying Nian Wu. Flow contrastive estimation of energy-based models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7518–7528, 2020.
14. Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672–2680, 2014.
15. Anirudh Goyal, Nan Rosemary Ke, Surya Ganguli, and Yoshua Bengio. Variational walkback: Learning a transition operator as a stochastic recurrent net. In Advances in Neural Information Processing Systems, pages 4392–4402, 2017.
16. Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, and David Duvenaud. FFJORD: Free-form continuous dynamics for scalable reversible generative models. In International Conference on Learning Representations, 2019.
17. Will Grathwohl, Kuan-Chieh Wang, Joern-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. Your classifier is secretly an energy based model and you should treat it like one. In International Conference on Learning Representations, 2020.
18. Karol Gregor, Frederic Besse, Danilo Jimenez Rezende, Ivo Danihelka, and Daan Wierstra. Towards conceptual compression. In Advances In Neural Information Processing Systems, pages 3549–3557, 2016.
19. Prahladh Harsha, Rahul Jain, David McAllester, and Jaikumar Radhakrishnan. The communication complexity of correlation. In Twenty-Second Annual IEEE Conference on Computational Complexity (CCC’07), pages 10–23. IEEE, 2007.
20. Marton Havasi, Robert Peharz, and José Miguel Hernández-Lobato. Minimal random code learning: Getting bits back from compressed model parameters. In International Conference on Learning Representations, 2019.
21. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, pages 6626–6637, 2017.
22. Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-VAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, 2017.
23. Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, and Pieter Abbeel. Flow++: Improving flow-based generative models with variational dequantization and architecture design. In International Conference on Machine Learning, 2019.
24. Sicong Huang, Alireza Makhzani, Yanshuai Cao, and Roger Grosse. Evaluating lossy compression rates of deep generative models. In International Conference on Machine Learning, 2020.
25. Nal Kalchbrenner, Aaron van den Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Video pixel networks. In International Conference on Machine Learning, pages 1771–1779, 2017.
26. Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron van den Oord, Sander Dieleman, and Koray Kavukcuoglu. Efficient neural audio synthesis. In International Conference on Machine Learning, pages 2410–2419, 2018.
27. Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.
28. Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 10 4401–4410, 2019.
29. Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. arXiv preprint arXiv:2006.06676v1, 2020.
30. Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8110–8119, 2020.
31. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.
32. Diederik P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In Advances in Neural Information Processing Systems, pages 10215–10224, 2018.
33. Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.
34. Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In Advances in Neural Information Processing Systems, pages 4743–4751, 2016.
35. John Lawson, George Tucker, Bo Dai, and Rajesh Ranganath. Energy-inspired models: Learning with sampler-induced distributions. In Advances in Neural Information Processing Systems, pages 8501–8513, 2019.
36. Daniel Levy, Matt D. Hoffman, and Jascha Sohl-Dickstein. Generalizing Hamiltonian Monte Carlo with neural networks. In International Conference on Learning Representations, 2018.
37. Lars Maaløe, Marco Fraccaro, Valentin Liévin, and Ole Winther. BIVA: A very deep hierarchy of latent variables for generative modeling. In Advances in Neural Information Processing Systems, pages 6548–6558, 2019.
38. Jacob Menick and Nal Kalchbrenner. Generating high fidelity images with subscale pixel networks and multidimensional upscaling. In International Conference on Learning Representations, 2019.
39. Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.
40. Alex Nichol. VQ-DRAW: A sequential discrete VAE. arXiv preprint arXiv:2003.01599, 2020.
41. Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, and Ying Nian Wu. On the anatomy of MCMC-based maximum likelihood learning of energy-based models. arXiv preprint arXiv:1903.12370, 2019.
42. Erik Nijkamp, Mitch Hill, Song-Chun Zhu, and Ying Nian Wu. Learning non-convergent non-persistent short-run MCMC toward energy-based model. In Advances in Neural Information Processing Systems, pages 5233–5243, 2019.
43. Georg Ostrovski, Will Dabney, and Remi Munos. Autoregressive quantile networks for generative modeling. In International Conference on Machine Learning, pages 3936–3945, 2018.
44. Ryan Prenger, Rafael Valle, and Bryan Catanzaro. WaveGlow: A flow-based generative network for speech synthesis. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3617–3621. IEEE, 2019.
45. Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQVAE-2. In Advances in Neural Information Processing Systems, pages 14837–14847, 2019.
46. Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, pages 1530–1538, 2015.
47. Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International Conference on Machine Learning, pages 1278–1286, 2014.
48. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015.
49. Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pages 901–909, 2016.
50. Tim Salimans, Diederik Kingma, and Max Welling. Markov Chain Monte Carlo and variational inference: Bridging the gap. In International Conference on Machine Learning, pages 1218–1226, 2015. 11
51. Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pages 2234–2242, 2016.
52. Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. PixelCNN++: Improving the PixelCNN with discretized logistic mixture likelihood and other modifications. In International Conference on Learning Representations, 2017.
53. Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265, 2015.
54. Jiaming Song, Shengjia Zhao, and Stefano Ermon. A-NICE-MC: Adversarial training for MCMC. In Advances in Neural Information Processing Systems, pages 5140–5150, 2017.
55. Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, pages 11895–11907, 2019.
56. Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. arXiv preprint arXiv:2006.09011, 2020.
57. Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. WaveNet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.
58. Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. International Conference on Machine Learning, 2016.
59. Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with PixelCNN decoders. In Advances in Neural Information Processing Systems, pages 4790–4798, 2016.
60. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.
61. Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011.
62. Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot...for now. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020.
63. Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7794–7803, 2018.
64. Auke J Wiggers and Emiel Hoogeboom. Predictive sampling with forecasting autoregressive models. arXiv preprint arXiv:2002.09928, 2020.
65. Hao Wu, Jonas Köhler, and Frank Noé. Stochastic normalizing flows. arXiv preprint arXiv:2002.06707, 2020.
66. Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European Conference on Computer Vision (ECCV), pages 3–19, 2018.
67. Jianwen Xie, Yang Lu, Song-Chun Zhu, and Yingnian Wu. A theory of generative convnet. In International Conference on Machine Learning, pages 2635–2644, 2016.
68. Jianwen Xie, Song-Chun Zhu, and Ying Nian Wu. Synthesizing dynamic patterns by spatial-temporal generative convnet. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7093–7101, 2017.
69. Jianwen Xie, Zilong Zheng, Ruiqi Gao, Wenguan Wang, Song-Chun Zhu, and Ying Nian Wu. Learning descriptor networks for 3d shape synthesis and analysis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8629–8638, 2018.
70. Jianwen Xie, Song-Chun Zhu, and Ying Nian Wu. Learning energy-based spatial-temporal generative convnets for dynamic patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.
71. Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. LSUN: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.
72. Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016. 



## Extra information

LSUN FID scores for LSUN datasets are included in Table 3. Scores marked with ∗ are reported by StyleGAN2 as baselines, and other scores are reported by their respective authors.

Table 3: FID scores for LSUN 256 × 256 datasets

Model LSUN Bedroom LSUN Church LSUN Cat

ProgressiveGAN [27] 8.34 6.42 37.52

StyleGAN [28] 2.65 4.21∗ 8.53∗

StyleGAN2 [30] - 3.86 6.93

Ours (Lsimple) 6.36 7.89 19.75

Ours (Lsimple, large) 4.90 - -

Progressive compression Our lossy compression argument in Section 4.3 is only a proof of concept, because Algorithms 3 and 4 depend on a procedure such as minimal random coding [20], which is not tractable for high dimensional data. These algorithms serve as a compression interpretation of the variational bound (5) of Sohl-Dickstein et al. [53], not yet as a practical compression system.

Table 4: Unconditional CIFAR10 test set rate-distortion values (accompanies Fig. 5)

Reverse process time (T − t + 1) Rate (bits/dim) Distortion (RMSE [0, 255]) 1000 1.77581 0.95136 900 0.11994 12.02277 800 0.05415 18.47482 700 0.02866 24.43656 600 0.01507 30.80948 500 0.00716 38.03236 400 0.00282 46.12765 300 0.00081 54.18826 200 0.00013 60.97170 100 0.00000 67.60125

## A Extended derivations

Below is a derivation of Eq. (5), the reduced variance variational bound for diffusion models. This material is from Sohl-Dickstein et al. [53]; we include it here only for completeness.

L = Eq  − log pθ(x0:T ) q(x1:T |x0)  (17) = Eq  − log p(xT ) −

X t≥1 log pθ(xt−1|xt) q(xt|xt−1)   (18) = Eq " − log p(xT ) −

X t>1 log pθ(xt−1|xt) q(xt|xt−1) − log pθ(x0|x1) q(x1|x0)


 (19) = Eq " − log p(xT ) −

X t>1 log pθ(xt−1|xt) q(xt−1|xt, x0) · q(xt−1|x0) q(xt|x0) − log pθ(x0|x1) q(x1|x0)


 (20) = Eq " − log p(xT ) q(xT |x0) −

X t>1 log pθ(xt−1|xt) q(xt−1|xt, x0) − log pθ(x0|x1)

 (21)
 13 = Eq " DKL(q(xT |x0) k p(xT )) +X t>1

DKL(q(xt−1|xt, x0) k pθ(xt−1|xt)) − log pθ(x0|x1)


 (22)

The following is an alternate version of L. It is not tractable to estimate, but it is useful for our discussion in Section 4.3.

L = Eq  − log p(xT ) −

X t≥1 log pθ(xt−1|xt) q(xt|xt−1)   (23) = Eq  − log p(xT ) −

X t≥1 log pθ(xt−1|xt) q(xt−1|xt) · q(xt−1) q(xt)   (24) = Eq  − log p(xT ) q(xT ) −

X t≥1 log pθ(xt−1|xt) q(xt−1|xt) − log q(x0)   (25) = DKL(q(xT ) k p(xT )) + Eq  

X t≥1

DKL(q(xt−1|xt) k pθ(xt−1|xt))   + H(x0) (26)

##　B Experimental details

Our neural network architecture follows the backbone of PixelCNN++ [52], which is a U-Net [48] based on a Wide ResNet [72]. We replaced weight normalization [49] with group normalization [66] to make the implementation simpler. Our 32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4), and our 256 × 256 models use six. All models have two convolutional residual blocks per resolution level and self-attention blocks at the 16 × 16 resolution between the convolutional blocks [6]. Diffusion time t is specified by adding the Transformer sinusoidal position embedding [60] into each residual block. Our CIFAR10 model has 35.7 million parameters, and our LSUN and

CelebA-HQ models have 114 million parameters. We also trained a larger variant of the LSUN

Bedroom model with approximately 256 million parameters by increasing filter count.

We used TPU v3-8 (similar to 8 V100 GPUs) for all experiments. Our CIFAR model trains at 21 steps per second at batch size 128 (10.6 hours to train to completion at 800k steps), and sampling a batch of 256 images takes 17 seconds. Our CelebA-HQ/LSUN (2562 ) models train at 2.2 steps per second at batch size 64, and sampling a batch of 128 images takes 300 seconds. We trained on

CelebA-HQ for 0.5M steps, LSUN Bedroom for 2.4M steps, LSUN Cat for 1.8M steps, and LSUN

Church for 1.2M steps. The larger LSUN Bedroom model was trained for 1.15M steps.

Apart from an initial choice of hyperparameters early on to make network size fit within memory constraints, we performed the majority of our hyperparameter search to optimize for CIFAR10 sample quality, then transferred the resulting settings over to the other datasets:
• We chose the βt schedule from a set of constant, linear, and quadratic schedules, all constrained so that LT ≈ 0. We set T = 1000 without a sweep, and we chose a linear schedule from β1 = 10−4 to βT = 0.02.
• We set the dropout rate on CIFAR10 to 0.1 by sweeping over the values {0.1, 0.2, 0.3, 0.4}.

Without dropout on CIFAR10, we obtained poorer samples reminiscent of the overfitting artifacts in an unregularized PixelCNN++ [52]. We set dropout rate on the other datasets to zero without sweeping.
• We used random horizontal flips during training for CIFAR10; we tried training both with and without flips, and found flips to improve sample quality slightly. We also used random horizontal flips for all other datasets except LSUN Bedroom.
• We tried Adam [31] and RMSProp early on in our experimentation process and chose the former. We left the hyperparameters to their standard values. We set the learning rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5 for the 256 × 256 images, which seemed unstable to train with the larger learning rate. 14
• We set the batch size to 128 for CIFAR10 and 64 for larger images. We did not sweep over these values.
• We used EMA on model parameters with a decay factor of 0.9999. We did not sweep over this value.

Final experiments were trained once and evaluated throughout training for sample quality. Sample quality scores and log likelihood are reported on the minimum FID value over the course of training.

On CIFAR10, we calculated Inception and FID scores on 50000 samples using the original code from the OpenAI [51] and TTUR [21] repositories, respectively. On LSUN, we calculated FID scores on 50000 samples using code from the StyleGAN2 [30] repository. CIFAR10 and CelebA-HQ were loaded as provided by TensorFlow Datasets (https://www.tensorflow.org/datasets), and LSUN was prepared using code from StyleGAN. Dataset splits (or lack thereof) are standard from the papers that introduced their usage in a generative modeling context. All details can be found in the source code release.

## C Discussion on related work

Our model architecture, forward process definition, and prior differ from NCSN [55, 56] in subtle but important ways that improve sample quality, and, notably, we directly train our sampler as a latent variable model rather than adding it after training post-hoc. In greater detail:
1. We use a U-Net with self-attention; NCSN uses a RefineNet with dilated convolutions. We condition all layers on t by adding in the Transformer sinusoidal position embedding, rather than only in normalization layers (NCSNv1) or only at the output (v2).

2. Diffusion models scale down the data with each forward process step (by a √ 1 − βt factor) so that variance does not grow when adding noise, thus providing consistently scaled inputs to the neural net reverse process. NCSN omits this scaling factor.
3. Unlike NCSN, our forward process destroys signal (DKL(q(xT |x0) k N (0, I)) ≈ 0), ensuring a close match between the prior and aggregate posterior of xT . Also unlike NCSN, our
 βt are very small, which ensures that the forward process is reversible by a Markov chain with conditional Gaussians. Both of these factors prevent distribution shift when sampling.

 4. Our Langevin-like sampler has coefficients (learning rate, noise scale, etc.) derived rigorously from βt in the forward process. Thus, our training procedure directly trains our
 sampler to match the data distribution after T steps: it trains the sampler as a latent variable model using variational inference. In contrast, NCSN’s sampler coefficients are set by hand post-hoc, and their training procedure is not guaranteed to directly optimize a quality metric of their sampler.

## D Samples

Additional samples Figure 11, 13, 16, 17, 18, and 19 show uncurated samples from the diffusion models trained on CelebA-HQ, CIFAR10 and LSUN datasets.

Latent structure and reverse process stochasticity During sampling, both the prior xT ∼

N (0, I) and Langevin dynamics are stochastic. To understand the significance of the second source of noise, we sampled multiple images conditioned on the same intermediate latent for the CelebA 256 × 256 dataset. Figure 7 shows multiple draws from the reverse process x0 ∼ pθ(x0|xt) that share the latent xt for t ∈ {1000, 750, 500, 250}. To accomplish this, we run a single reverse chain from an initial draw from the prior. At the intermediate timesteps, the chain is split to sample multiple images. When the chain is split after the prior draw at xT =1000, the samples differ significantly.

However, when the chain is split after more steps, samples share high-level attributes like gender, hair color, eyewear, saturation, pose and facial expression. This indicates that intermediate latents like x750 encode these attributes, despite their imperceptibility.

Coarse-to-fine interpolation Figure 9 shows interpolations between a pair of source CelebA 256 × 256 images as we vary the number of diffusion steps prior to latent space interpolation.

Increasing the number of diffusion steps destroys more structure in the source images, which the 15 model completes during the reverse process. This allows us to interpolate at both fine granularities and coarse granularities. In the limiting case of 0 diffusion steps, the interpolation mixes source images in pixel space. On the other hand, after 1000 diffusion steps, source information is lost and interpolations are novel samples.

Source Rec. λ=0.1 λ=0.2 λ=0.3 λ=0.4 λ=0.5 λ=0.6 λ=0.7 λ=0.8 λ=0.9 Rec. Source 1000 steps 875 steps 750 steps 625 steps 500 steps 375 steps 250 steps 125 steps 0 steps

Figure 9: Coarse-to-fine interpolations that vary the number of diffusion steps prior to latent mixing. 0 200 400 600 800 1,000 2 4 6 8 10

Reverse process steps (T − t) 0 200 400 600 800 1,000 0 100 200 300

Reverse process steps (T − t)<br/>
Figure 10: Unconditional CIFAR10 progressive sampling quality over time 16

Inception Score

FID

Figure 11: CelebA-HQ 256 × 256 generated samples 17 (a) Pixel space nearest neighbors (b) Inception feature space nearest neighbors

Figure 12: CelebA-HQ 256 × 256 nearest neighbors, computed on a 100 × 100 crop surrounding the faces. Generated samples are in the leftmost column, and training set nearest neighbors are in the remaining columns. 18

Figure 13: Unconditional CIFAR10 generated samples 19

Figure 14: Unconditional CIFAR10 progressive generation 20 (a) Pixel space nearest neighbors (b) Inception feature space nearest neighbors

Figure 15: Unconditional CIFAR10 nearest neighbors. Generated samples are in the leftmost column, and training set nearest neighbors are in the remaining columns. 21

Figure 16: LSUN Church generated samples. FID=7.89 22

Figure 17: LSUN Bedroom generated samples, large model. FID=4.90 23

Figure 18: LSUN Bedroom generated samples, small model. FID=6.36 24

Figure 19: LSUN Cat generated samples. FID=19.75 25
