# On the Representation Collapse of Sparse Mixture of Experts
关于专家稀疏混合的表示坍塌 2022.4.20 https://arxiv.org/abs/2204.09179

## Abstract
Sparse mixture of experts provides larger model capacity while requiring a constant computational overhead. It employs the routing mechanism to distribute input tokens to the best-matched experts according to their hidden representations. However, learning such a routing mechanism encourages token clustering around expert centroids, implying a trend toward representation collapse. In this work, we propose to estimate the routing scores between tokens and experts on a lowdimensional hypersphere. We conduct extensive experiments on cross-lingual language model pre-training and fine-tuning on downstream tasks. Experimental results across seven multilingual benchmarks show that our method achieves consistent gains. We also present a comprehensive analysis on the representation and routing behaviors of our models. Our method alleviates the representation collapse issue and achieves more consistent routing than the baseline mixture-of-experts methods.

稀疏专家混合提供了更大的模型容量，同时需要恒定的计算开销。 它采用路由机制根据隐藏表示将输入令牌分配给最匹配的专家。 然而，学习这样的路由机制会鼓励令牌围绕专家质心聚类，这意味着表示崩溃的趋势。 在这项工作中，我们建议估计低维超球体上令牌和专家之间的路由分数。 我们对跨语言语言模型预训练和下游任务微调进行了广泛的实验。 七个多语言基准测试的实验结果表明，我们的方法获得了一致的收益。 我们还对我们模型的表示和路由行为进行了全面分析。 我们的方法减轻了表示崩溃问题，并实现了比基线专家混合方法更一致的路由。

## 1 Introduction
Scaling up model capacities has shown to be a promising way to achieve better performance on a wide range of problems such as language model pre-training (Radford et al., 2019; Raffel et al., 2020), and visual representation learning (Dosovitskiy et al., 2021; Bao et al., 2022). Despite the effectiveness, increasing the number of parameters leads to larger computational cost, which motivates recent studies to explore Sparse Mixture-of-Experts (SMoE) models (Shazeer et al., 2017; Fedus et al., 2021; Lepikhin et al., 2021). SMoE increases the model capacity by building several sparselyactivated neural networks. With nearly constant computational overhead, SMoE models achieve better performance than dense models on various tasks, including machine translation (Lepikhin et al., 2021), image classification (Riquelme et al., 2021), and speech recognition (Kumatani et al., 2021).

扩大模型容量已被证明是在语言模型预训练(Radford et al., 2019; Raffel et al., 2020)和视觉表示学习(Dosovitskiy et al., 2021; Bao et al., 2022)广泛问题上实现更好性能的有前途的方法。 尽管有效，但增加参数数量会导致更大的计算成本，这促使最近的研究探索稀疏专家混合 (SMoE) 模型(Shazeer et al., 2017; Fedus et al., 2021; Lepikhin et al., 2021; Lepikhin et al., 2021). SMoE 通过构建多个稀疏激活的神经网络来增加模型容量。 由于几乎恒定的计算开销，SMoE 模型在各种任务上取得了比密集模型更好的性能，包括机器翻译(Lepikhin et al., 2021)、图像分类(Riquelme et al., 2021)和语音识别(Kumatani et al.,  2021)。

The routing mechanism plays an important role in SMoE models. Given an input token, the router measures the similarity scores between each token and experts. Then we distribute tokens to the bestmatched experts according to the routing scores. Recent studies explored various token assignment algorithms to improve SMoE training. For instance, Lewis et al. (2021) formulate SMoE routing as a linear assignment problem that globally maximizes token-expert similarities. Zhou et al. (2022) have experts selecting top tokens rather than assigning tokens to top experts. Roller et al. (2021) and Dai et al. (2022) propose to keep routing choices consistent. Many studies in recent years focus on how to design the token-expert assignment algorithm. In this paper, we present that current routing mechanisms tend to push hidden representations clustering around expert centroids, implying a trend toward representation collapse, which in turn harms model performance. 

路由机制在 SMoE 模型中起着重要作用。 给定输入令牌，路由器会测量每个令牌与专家之间的相似性分数。 然后我们根据路由分数将令牌分配给最匹配的专家。 最近的研究探索了各种令牌分配算法以改进 SMoE 训练。 例如，Lewis et al. (2021)将 SMoE 路由制定为一个线性分配问题，可以全局最大化令牌专家相似性。 Zhou et al. (2022) 让专家选择顶级令牌，而不是将令牌分配给顶级专家。 Roller et al. (2021) and Dai et al. (2022) 建议保持路由选择的一致性。 近年来的许多研究都集中在如何设计令牌专家分配算法上。 在本文中，我们提出当前的路由机制倾向于推动围绕专家质心的隐藏表示聚类，这意味着表示崩溃的趋势，这反过来会损害模型性能。

In order to alleviate the representation collapse issue, we introduce a simple yet effective routing algorithm for sparse mixture-of-experts models. More specifically, rather than directly using the hidden vectors for routing, we project the hidden vectors into a lower-dimensional space. Then, we apply L2 normalization to both token representations and expert embeddings, i.e., measuring routing scores on a low-dimensional hypersphere. Besides, we propose a soft expert gate with learnable temperature, which learns to control the activation of experts.

为了缓解表示崩溃问题，我们为稀疏混合专家模型引入了一种简单而有效的路由算法。 更具体地说，我们不是直接使用隐藏向量进行路由，而是将隐藏向量投影到低维空间中。 然后，我们将 L2 归一化应用于令牌表示和专家嵌入，即测量低维超球体上的路由分数。 此外，我们提出了一种具有可学习温度的软专家门，它可以学习控制专家的激活。

We evaluate the proposed method on cross-lingual language model pre-training and fine-tuning on downstream tasks. Experimental results show that our model consistently outperforms the baseline SMoE models in terms of both language modeling and fine-tuning performance. Moreover, analysis indicates that our method alleviates the representation collapse issue compared with the SMoE baseline. Our method also achieves more consistent routing behaviors during both pre-training and fine-tuning, which confirms the effectiveness of the proposed routing algorithm.

我们评估了所提出的跨语言语言模型预训练和下游任务微调的方法。 实验结果表明，我们的模型在语言建模和微调性能方面始终优于基线 SMoE 模型。 此外，分析表明，与 SMoE 基线相比，我们的方法减轻了表示崩溃问题。 我们的方法还在预训练和微调期间实现了更一致的路由行为，这证实了所提出的路由算法的有效性。

Our contributions are summarized as follows: 
* We point out the representation collapse issue in sparse mixture-of-experts models, which is under-explored in previous work. 
* We propose to estimate routing scores between tokens and experts on a low-dimensional hypersphere in order to alleviate representation collapse. 
* We conduct extensive experiments on cross-lingual language model pre-training and finetuning on downstream tasks. 
* We present a detailed analysis of routing behaviors and representation properties, which shows that our method improves performance and achieves more consistent routing. 

我们的贡献总结如下：
* 指出稀疏混合专家模型中的表示崩溃问题，这在以前的工作中未得到充分探索。
* 建议在低维超球体上估计令牌和专家之间的路由分数，以减轻表示崩溃。
* 对下游任务的跨语言语言模型预训练和微调进行了广泛的实验。
* 对路由行为和表示属性进行了详细分析，这表明我们的方法提高了性能并实现了更一致的路由。

## 2 Background
### 2.1 Sparse Mixture of Experts
Sparse Mixture-of-Experts (SMoE) models take advantage of conditional computation, and have shown to be a promising way to scale up the number of parameters. In this work, we consider SMoE for Transformers, where SMoE layers are inserted into neighboring Transformer blocks. Each SMoE layer consists of a router and several expert networks. Following most previous work (Fedus et al., 2021), we use feed-forward networks as experts, instead of self-attention modules.

For the input token x with its hidden representation h ∈ Rd , the router computes the routing score between h and the i-th expert by a dot-product similarity metric si = h · ei , where ei ∈ Rd is a learnable expert embedding, and d is the hidden size of the model. Then, the router utilizes a sparse gating function g(r) to make the expert network conditionally activated.

In this paper, we mainly focus on top-1 routing, i.e., only the expert with the largest routing score is activated. Formally, considering a SMoE layer with N experts, the forward function of SMoE can be written as: 


k = arg max i si = arg max i h · ei (1) f

SMoE(h) = h + g(sk)fk

FFN(h) (2) 

where fk FFN(·) stands for the k-th expert network that is implemented as stacked feed-forward networks. Moreover, we explore both softmax gating (Lepikhin et al., 2021; Fedus et al., 2021) and sigmoid gating (Lewis et al., 2021; Dai et al., 2022) for the function g(sk): g(sk) = ( exp(sk)/P Nj=1 exp(sj ), softmax gating σ(sk), sigmoid gating , (3) where σ(·) is the sigmoid function.

### 2.2 Representation Collapse of Sparse Mixture-of-Experts
We present how representation collapse happens in sparse mixture-of-experts models. For convenience, we use h0 = f

SMoE(h) to denote the output of the SMoE layer as in Equation (2), Sk = g(sk) 2 to denote the k-th output of the softmax function, and h

FFN = fk

FFN(h) to denote the output of the k-th expert network. The Jacobian matrix with respect to h is given by:

J = J1 + J2 = (I + SkJ

FFN) +

NXj=1

Sk(δkj − Sj )h

FFNe>j , (4) where δkj is a Kronecker delta. The equation means that the Jacobian matrix can be decomposed into two terms. The first term J1 represents producing a better token representation given the current activation Sk. The second term J2 means to learn better gating function for appropriate activation score Sk. After back-propagation, the gradient is received from the above two paths, written as ∇hL = J1> ∇h0 L + J2> ∇h0 L. The second term can be expanded as:

J2> ∇h0 L = NXj=1

Sk(δkj − Sj )(h

FFN> ∇h0 L)ej = NXj=1 cjej , (5) where cj = Sk(δkj − Sj )(h

FFN> ∇h0 L). The above equation indicates that the token representation h tends to be updated toward a linear combination of the expert embeddings.

The finding also holds for top-K routing (Lepikhin et al., 2021) where the top K experts (K ≤ N) are activated for each token. The forward function of top-K routing is k1, k2, ..., kK = topK(sk) and h0 = f

SMoE(h) = h + P i=1...K g(ski )fk

FFN i (h). The gating function is defined as 

g(ski ) = exp(ski )/P j=1...K exp(skj ). Similar to Equation (5), 

we have

J2> ∇h0 L = KXi=1 KXj=1

Ski (δkikj − Skj )(h

FFNki> ∇h0 L)ekj = KXj=1 cjekj . (6)

Therefore, the above finding holds for top-K routing.

We consider that such behavior potentially harms the representation capacity of Transformers. Firstly, consider that the N expert vectors can span a N-dimensional space at most via linear combinations.

As N is much smaller than the hidden size d in practice, the spanning subspace does not fully utilize the entire available capacity. Thus, the mechanism renders the Transformer hidden vector h collapsed to an N-dimensional subspace, implying a trend toward representation collapse from Rd to RN where N  d in practice. Secondly, Equation (5) indicates that the hidden vector h tends to be similar to the expert embedding that it is routed to. If the hidden states were routed to the same expert, they are going to be pushed closer. However, we would like to encourage the representations more diverse, so that they can be more expressive and discriminative. The phenomenon possibly restricts the expressibility of hidden states, especially when an expert is inclined to dominate routing. 

## 3 Methods
We introduce the routing algorithm for sparse mixture of experts, which measures the routing scores between tokens and experts on a low-dimensional hypersphere. As shown in Figure 1b, we address the representation collapse issue of SMoE by applying dimensionality reduction and L2 normalization for the token representations and expert embeddings. Then, we describe how to incorporate the routing algorithm into an SMoE model under the pre-training-then-fine-tuning paradigm.

### 3.1 Routing Algorithm
Dimension Reduction In order to alleviate the representation collapse issue mentioned in Section 2.2, we represent the expert embedding ei and the token vector h in a low-dimensional space instead of the original high-dimensional hidden space. Specifically, we first parameterize the experts with lower-dimensional embeddings ei ∈ Rde such that de is much smaller than the Transformer hidden size d. Next, we conduct a projection over the hidden states f proj(h), which projects h to the expert embedding space. We use a linear projection f proj(h) = W h such that W ∈ Rde×d.

Thus, the routing scoring function between the tokens and experts can be written as si = (W h) · ei.

Typically we set de = N/2 (i.e., half of the number of experts) in our implementation.

Inspired by Jing et al. (2022), dimension reduction mitigates the issues described in Section 2.2 from two perspectives. First, linear projection W h isolates the direct interaction between hidden vector h 3

![Figure 1](../images/X-MoE/fig_1.png)<br/>
Figure 1: Illustration of a typical SMoE layer and the proposed X-MOE layer. (a) An SMoE layer consists of a router and expert networks, where the experts are sparsely activated according to dot-product token-expert routing scores. (b) X-MOE improves the routing algorithm via dimension reduction, L2 normalization, and gating temperature. and expert embedding ei , which tends to relieve cascaded collapse for representations. Second, it is natural to apply a low-rank projector for hidden vectors, as the number of experts is usually much smaller than the hidden size of Transformers. Hence the reduced dimension better fits in with the low-rank nature of routing.

L2 Normalization After dimension reduction, we apply L2 normalization to both token representations and expert embeddings. Our routing score is defined as: 

si = (W h) · ei k W hkk eik , (7) 

where k · k is L2 normalization. Thus, the resulting representations are transformed into a certain scale with stabilized routing scoring.

As described in Section 2.2, if an expert dominated a set of hidden states, the representations were pushed toward the expert embedding. In order to fully utilize the space, we favor larger uniformity of representations while avoiding dominated experts. Given a hidden vector h, the dot-product routing si = (W h) · ei is affected by both k eik and cos(W h, ei). So some experts are allocated with more tokens because of larger values of k eik . In contrast, L2 normalization projects vectors on the unit hypersphere, which suppresses the undesired effect of k eik . The visualization in Figure 2b also confirms that our method improves the uniformity of learned representations.

Implementation Tips. When scaling the model to more experts, empirically, we observe that the resulting token assignments can be in a fluctuation if the expert embedding norm k eik is small.

Therefore, we initialize the expert embeddings with L2 norm of 0.1 and keep the norm unchanged during training. Since the expert embeddings are parameterized in the space of Rde , the change rate of the angle of ei is in inverse proportion to k eik . As a result, if the norm is small, the angle of ei is updated fast, finally leading to the fluctuation of token assignments, especially when scaling up the model with more experts.

Gating with Learnable Temperature In addition, we add a learnable temperature scalar τ in the SMoE gating function g(sk). Because L2 normalization rescales the routing scores sk to the range [−1, 1], directly using the scores for SMoE gating tends to make expert activation too conservative.

The introduced temperature enables the router to adjust the gating g(sk) accordingly. To be more specific, our gating function is: g(sk) = ( exp(sk/τ) P Nj=1 exp(sj /τ) , softmax gating σ(sk/τ ), sigmoid gating , (8) where σ(·) is the sigmoid function, and the temperature scalar τ is learnable. 

### 3.2 Training Objective
The training objective is jointly minimizing the loss of the target task and an auxiliary load balancing loss (Fedus et al., 2021). The load balancing loss is separately computed for each router. For each router, given the frequency ti of how many tokens are routed to the i-th expert and the routing score si , the load balancing loss is computed via:

L balance = N |B| NXi=1 token X∈B ti exp(si/τ0) P Nj=1 exp(sj/τ0), (9) 

where N is the number of the experts, B is a batch of training examples, |B| is the number of tokens, and τ0 stands for a constant temperature. Different from the learnable τ in Equation (8), τ0 is kept fixed during training. The overall training objective is to minimize:

L = Ltask + αL balance , (10) 

where α is a coefficient for load balancing . The term Ltask is determined by the specific task that Transformer learns. For example, we employ the masked language modeling loss (Devlin et al., 2019) for pre-training, and the sequence-to-sequence learning objective for neural machine translation.

### 3.3 Frozen Routing During Fine-tuning
We evaluate SMoE under the pre-training-then-fine-tuning paradigm in our work. During fine-tuning, we freeze all the parameters of experts, including both the router and expert networks. Because the fine-tuning datasets are usually small compared with pre-training corpora. We find that SMoE models tend to overfit downstream tasks, which often leads to inconsistent routing. Freezing SMoE parameters helps to relieve the above issues. Notice that we still use load balancing loss although the routers are kept fixed, which empirically improves fine-tuning performance in our experiments. 

## 4 Experiments
We conduct experiments on cross-lingual language model pre-training (Devlin et al., 2019). We evaluate the performance by fine-tuning the pretrained models on various downstream benchmarks.

We also compare validation losses of the masked language modeling task. Our method is named as X-MOE in the following sections.

### 4.1 Experimental Setup
Pre-training Data Following (Chi et al., 2021), we use the combination of CCNet (Wenzek et al., 2019) and Wikipedia dump as pre-training corpora. We sample sentences in 94 languages from the corpora, and employ a re-balanced distribution introduced by Conneau and Lample (2019), which increases the probability of low-resource languages.

Model Architecture and Hyperparameters We construct our X-MOE models using the Transformer (Vaswani et al., 2017) encoder (L = 12, H = 768, A = 12) with the vocabulary provided by Conneau et al. (2020) as the backbone architecture. Following Lewis et al. (2021), we build a 32-expert sparse layer with 3 FFN sub-layers, and insert it after the 6-th Transformer layer. The routing dimension de is set as 16. The gating temperature τ0 is set as 0.3 and 0.07 for the softmax gate and sigmoid gate, respectively. The detailed hyperparameters of X-MOE models can be found in Appendix A. X-MOE models are pretrained with the Adam optimizer (β1 = 0.9, β2 = 0.98) using a batch size of 2, 048 for 125K steps. The pre-training procedure takes 2 days on 2 Nvidia DGX-2

Stations. Appendix B and Appendix C provide the detailed hyperparameters for X-MOE pre-training and fine-tuning.

Baselines We consider two baselines in our experiments. (1) Dense is a dense Transformer encoder without sparsely-activated modules. (2) SMoE is our implementation of Switch Transformers (Fedus et al., 2021). The SMoE baseline is built with the same setting with X-MOE. In addition to its original softmax-gating implementation, we also implement a sigmoid-gating (Lewis et al., 2021; Dai et al., 2022) variant of Switch Transformers as a baseline approach. Notice that the baseline models are pretrained with the same training data as X-MOE for a fair comparison. 5

Table 1: Evaluation results on the cross-lingual XTREME benchmark. The models are fine-tuned on the English training data and directly evaluated in all target languages. SMoE models are grouped according to the choice of gating function. The results are averaged over five runs.


Table 2: Results of upstream evaluation.


Table 3: Ablation studies of X-MOE components. The models employ various combinations of dimension reduction, L2 normalization, and frozen routing. Average fine-tuning results of five random seeds are reported.



### 4.2 Downstream Evaluation
We conduct a downstream evaluation on seven widely-used cross-lingual understanding benchmarks from XTREME (Hu et al., 2020). Specifically, we conduct experiments on Universal Dependencies v2.5 part-of-speech tagging (Zeman et al., 2019), WikiAnn named entity recognition (Pan et al., 2017; Rahimi et al., 2019), natural language inference (XNLI; Conneau et al. 2018), paraphrase adversaries from word scrambling (PAWS-X; Yang et al. 2019), and question answering on MLQA (Lewis et al., 2020), XQuAD (Artetxe et al., 2020), and TyDiQA-GoldP (Clark et al., 2020). Among the benchmarks, we adopt the cross-lingual transfer setting, where the models are fine-tuned with the training data in English and evaluated in all target languages.

Table 1 presents the evaluation results on the seven downstream tasks from the XTREME benchmark.

For each task, the results are first averaged among the test languages and then averaged over five random seeds. Overall, the softmax-gating X-MOE model obtains the best performance, achieving an average score of 65.3. Comparing SMoE models with the dense model, SMoE models show notable improvement, indicating that SMoE models benefit from the large model capacity. Comparing

X-MOE with the two SMoE baselines, it shows that X-MOE models provide consistent gains on downstream tasks, demonstrating the effectiveness of our proposed routing algorithm. We also validate X-MOE under the top-2 routing setting. Table 7 presents the evaluation results on XNLI, showing consistent improvements over the baseline for both top-1 and top-2 routing settings.

### 4.3 Upstream Evaluation
We compare the pretrained models for the upstream performance by the validation perplexity on masked language modeling (MLM). We sample multilingual sentences from mC4 (Xue et al., 2020), and construct an MLM validation dataset that contains 65, 536 sequences with lengths around 512.

The results are shown in Table 2. Similar to the downstream results, we observe that SMoE models perform better than the dense model. In terms of the SMoE models, X-MOE models with both softmax 

Table 4: BLEU scores on multilingual machine translation on WMT-10. The models are evaluated in the directions of ‘x → en’.



Table 5: Comparison of routing dimensions for dimensionality reduction. ‘N’ stands for the number of experts.


Table 6: Effects of load balancing during fine-tuning. The models are fine-tuned with various weights for the auxiliary load balancing loss.


Table 7: Evaluation results on XNLI under the top-1 and top-2 routing settings. The models use the softmax gating functions.


and sigmoid gating achieve lower masked language modeling perplexities than their counterparts.

Among all the pretrained models, the softmax-gating X-MOE the achieves the lowest validation perplexity. The results show that our method not only works well for learning transferable text representations for downstream tasks, but also brings improvements to the upstream masked language modeling task. Comparing the upstream results with the downstream results, it shows that achieving a lower upstream perplexity does not promise better downstream performance. For instance, the sigmoid-gating X-MOE model has larger perplexity than the softmax-gating SMoE baseline has, but outperforms the fine-tuning performance of the baseline on the downstream tasks.

We also conduct experiments on the multilingual machine translation task. As shown in Table 4, we present the BLEU scores on the WMT-10 (Wang et al., 2020) dataset where the models are evaluated in the directions of ‘x → en’. X-MOE consistently outperforms both the dense model and the SMoE baseline in eight translation directions.

### 4.4 Ablation Studies
Routing Algorithm To better understand our routing algorithm, we pretrain several variants of sigmoid-gating X-MOE models with various combinations of dimension reduction (Dim. Red.), L2 normalization (L2 Norm), and routing frozen (Frozen). For a fair comparison, all the models are pretrained and fine-tuned under the same setup, i.e., training data, steps, and the random seeds. We evaluate the models on XNLI and MLQA, and report the results in Table 3. Jointly using the three routing methods achieves the best performance. When ablating one of the three routing methods, the model performs less well, demonstrating that X-MOE benefits from all the three components.

Dimension of Expert Embedding We conduct experiments by adjusting the routing dimension for dimensionality reduction. Specifically, we compare sigmoid-gating X-MOE models with routing dimensions of N/4, N/2, N, 2N, and 4N, where N is the number of the experts. Table 5 shows the downstream performance. It shows that using the routing dimension of N/2 provides the best performance for XNLI and N/4 is the best for MLQA. The results also confirm that dimension reduction better fits in with the low-rank nature of SMoE routing.

Load Balancing During Fine-tuning We explore whether load balancing is beneficial for finetuning SMoE models. To this end, we add load balancing loss to the total loss with various weights when fine-tuning X-MOE models on XNLI and MLQA. Table 6 shows the average validation scores where we search the load balancing coefficient α ranging from 0 to 10−1 . We observe that using balance loss during fine-tuning is slightly beneficial for X-MOE. When removing the balance loss, X-MOE still remains comparable results on both XNLI and MLQA.  

![Figure 2](../images/X-MoE/fig_2.png)<br/>
Figure 2: Analysis on the representation collapse of the Transformer hidden states. Figure (a) and (b) visualize the spatial structure of the experts. Each data point represents a token to be routed, and its color stands for the expert that it is assigned to. Figure (c) presents the curves of representation collapse (RC), which measures the within-class variability of hidden states. Larger RC values indicate less collapse.

### 4.5 Analysis
Representation Collapse We qualitatively analyze the representation collapse issue by visualizing the experts. Figure 2a and 2b illustrate the spatial structure of the experts of SMoE baseline and X-MOE in hyperbolic space, which is produced by Uniform Manifold Approximation and Projection (UMAP; McInnes et al. 2018) with n-neighbor of 100 and min-dist of 1. Each data point represents a token to be routed, where we use the hidden states for SMoE baseline and the projected token representations for X-MOE. Each color stands for an expert that the tokens are assigned to.

Figure 2a shows that most of the points are mixed together with a large amount of available room unused, which suggests a representation collapse in the expert embedding space. In contrast, X-MOE in Figure 2b shows a well-organized feature space with clear distinctions between clusters. It indicates that our routing methods successfully project the tokens to the expert embedding space with routing features preserved.

Additionally, we conduct quantitative analysis on the degree of representation collapse for the learned Transformer hidden states that are fed into SMoE routing. We use the representation collapse metric proposed in (Zhu et al., 2021). Given the representations to be measured, we use ΣW and ΣB to denote the within-class and between-class covariance matrices, respectively. The representations collapse (RC) metric is calculated via:

RC = Tr(ΣW Σ†B), (11) 

where Σ†B is the pseudo inverse of ΣB. Smaller RC values indicate representation collapse to a greater extent. Figure 2c illustrates the metrics during pre-training, where the data is sampled from the validation set mentioned in Section 4.3. SMoE baseline is unlike unconstrained feature models that can empirically collapse to almost zero RC, but still shows a consistent descending trend through pre-training, implying a trend toward representation collapse. Differently, X-MOE obtains larger RC scores than SMoE baseline with uptrend through pre-training.

Routing Consistency Through Pre-training We examine whether our proposed routing algorithm achieves more consistent routing through training. We measure the routing consistency via the routing fluctuation (RF) ratio metric. Routing fluctuation is defined as the change of the target expert of an input token. Correspondingly, the RF ratio measures the ratio of RF between the current and the last checkpoints for the same input. A lower RF ratio indicates better routing consistency. As shown in Figure 3a, we present the RF ratio on the MLM validation set mentioned in Section 4.3. After the 15K step, X-MOE shows a much lower RF ratio than the SMoE baseline, indicating that our model produces more consistent routing behaviors.

Inter-run Consistency Through Fine-tuning In the experiments of the downstream evaluation, we find that the routing behaviors of SMoE baseline models can be sensitive to random seeds. As the learned token assignments are various for different training data orders, the final downstream performance can be diverse among runs. Therefore, we study the routing behaviors of the SMoE  


![Figure 3](../images/X-MoE/fig_3.png)<br/>
Figure 3: The routing behaviors of SMoE baseline and X-MOE. (a) Routing fluctuation (RF) ratio measures the ratio of the tokens that change their target experts between two checkpoints. Smaller

RF values indicate more stable routing. (b) Inter-run consistency measures the correlation among the token assignments of various fine-tuning runs. Larger values indicate more consistent routing. baseline and X-MOE models through fine-tuning. To achieve this, we develop a metric, named inter-run consistency, which measures how closely the token assignments converge among the runs with different seeds. Considering a model with N experts, let l = [n1, ..., nN ] denote the total load of the experts, where ni stands for the number of the tokens that are assigned to the i-th expert. Given two loads l1 and l2 from two runs with different seeds, the similarity between l1 and l2 is defined as the Pearson correlation coefficient (PCC) between them, which is denoted as ρl1,l2 . Here PCC only serves as a similarity metric rather than measuring linear correlation between variables. By extending it to m runs with different seeds for each run, we define the inter-run consistency as the average of correlation matrix IC = P i,j∈{1...m} ρli,lj /m2.

We fine-tune X-MOE and SMoE baseline models on XNLI for 12 runs separately. Then we compute the inter-run consistency for every 100 mini-batches, i.e., the expert loads are accumulated for 100 steps. Figure 3b illustrates the inter-run consistency. The SMoE baseline converges toward different routing solutions across multiple runs of fine-tuning, even though the only difference between runs is the random seed. In comparison, X-MOE obtains substantially better inter-run consistency than the SMoE baseline. The curve of X-MOE indicates that the models have various routing behaviors at the beginning of the fine-tuning, but finally converge to almost the same routing behaviors. 

## 5 Related Work
### SMoE for Large-Scale Models  用于大型模型的SMoE
Sparse Mixture-of-Experts (SMoE) models are introduced by Shazeer et al. (2017), which extends mixture of experts (Jacobs et al., 1991; Jordan and Jacobs, 1994) with conditional computation (Bengio et al., 2013; 2015) techniques. Taking advantage of computational computation, SMoE enables a massive increase in model capacity while maintaining computational efficiency. To explore the potential of SMoE, recent studies apply SMoE in a wide range of machine learning problems such as machine translation (Lepikhin et al., 2021), image classification (Riquelme et al., 2021), speech recognition (Kumatani et al., 2021). In addition to the supervised learning scenario, there has been work on exploring SMoE under the pre-training- fine-tuning paradigm, and observing discrepancies between strong pre-training quality and poor fine-tuning performance (Fedus et al., 2021; Artetxe et al., 2021; Zoph et al., 2022). Besides, the scaling behaviors of SMoE are also studied (Clark et al., 2022; Du et al., 2021).

Shazeer et al. (2017)介绍了稀疏专家混合 (SMoE) 模型，它扩展了专家混合 (Jacobs et al., 1991; Jordan and Jacobs, 1994) 与条件计算 (Bengio et al., 2013; 2015) 技术。 利用计算计算，SMoE 可以在保持计算效率的同时大幅增加模型容量。 为了探索 SMoE 的潜力，最近的研究将 SMoE 应用于广泛的机器学习问题，例如机器翻译（Lepikhin et al., 2021）、图像分类（Riquelme et al., 2021）、语音识别（Kumatani et al., 2021）。 , 2021). 除了监督学习场景之外，还有一些工作是在预训练-微调范式下探索 SMoE，并观察预训练质量高和微调性能差之间的差异（Fedus et al., 2021; Artetxe et al., 2021; Zoph et al., 2022）。 此外，还研究了 SMoE 的缩放行为 (Clark et al., 2022; Du et al., 2021)。

### SMoE Routing Algorithms SMoE 路由算法. 
Many recent studies explore the token assignment algorithms for SMoE routing. BASE layers (Lewis et al., 2021) formulate the token routing problem as a linear assignment problem. Hash Layers (Roller et al., 2021) employ a parameter-free assignment algorithm that routes tokens by hashing. Zhou et al. (2022) let each expert select top-k tokens rather than distribute tokens to experts. Dai et al. (2022) propose to freeze the routing function in order to relieve routing fluctuation. These methods focus on the assignment algorithm in routing, but our routing algorithm focuses on improving the underlying routing scoring metric, which is still under-explored. 

许多最近的研究探索了 SMoE 路由的令牌分配算法。 BASE 层 (Lewis et al., 2021) 将令牌路由问题表述为线性分配问题。 哈希层 (Roller et al., 2021) 采用无参数分配算法，通过哈希路由令牌。 Zhou et al. (2022) 让每个专家选择 top-k 令牌，而不是将令牌分发给专家。 Dai et al. （2022）建议冻结路由功能以缓解路由波动。 这些方法侧重于路由中的分配算法，但我们的路由算法侧重于改进底层路由评分指标，这仍未得到充分探索。

### Representation Collapse 表示崩溃
Representation collapse, also termed neural collapse, is the degeneration of the representations during the training of neural networks. Several studies observe that the withinclass variation of the representations in classification networks becomes negligible at the terminal phase of training (Papyan et al., 2020; Zhu et al., 2021; Tirer and Bruna, 2022). Besides, this phenomenon has also been observed in language model fine-tuning (Aghajanyan et al., 2021), and visual representation learning (Chen and He, 2021; Ermolov et al., 2021; Jing et al., 2022). These studies focus on densely-activated neural networks. In this work, we point out the representation collapse issue in SMoE models. 

表示崩溃，也称为神经崩溃，是神经网络训练过程中表示的退化。 几项研究观察到，分类网络中表示的类内变化在训练的末期变得可以忽略不计（Papyan et al., 2020; Zhu et al., 2021; Tirer 和 Bruna，2022）。 此外，在语言模型微调 (Aghajanyan et al., 2021) 和视觉表示学习 (Chen and He, 2021; Ermolov et al., 2021; Jing et al., 2022) 中也观察到了这种现象。 这些研究侧重于密集激活的神经网络。 在这项工作中，我们指出了 SMoE 模型中的表示崩溃问题。

## 6 Conclusion
In this work, we point out the representation collapse issue in sparse mixture-of-experts (SMoE) models, and propose a routing algorithm that estimates the routing scores on a low-dimensional hypersphere. We conduct extensive experiments on cross-lingual language model pre-training. Experimental results across various benchmarks demonstrate that our method brings consistent improvements over SMoE baselines in terms of both language modeling and fine-tuning performance. Besides, our method alleviates the trend toward representation collapse and achieves more consistent routing. We are going to improve the work from the following perspectives. First, most current XMOE experiments are conducted on language tasks, such as multilingual language model pre-training, and machine translation. We will also evaluate the proposed method on vision pretraining (Bao et al., 2022; Peng et al., 2022) and multimodal pretraining (Wang et al., 2022). Second, we would like to report the results of scaling up model size. The performance gain tends to be greater with a larger number of experts.

在这项工作中，我们指出了稀疏混合专家 (SMoE) 模型中的表示崩溃问题，并提出了一种估计低维超球面上路由分数的路由算法。 我们对跨语言语言模型预训练进行了广泛的实验。 各种基准测试的实验结果表明，我们的方法在语言建模和微调性能方面比 SMoE 基线带来了一致的改进。 此外，我们的方法减轻了表示崩溃的趋势并实现了更一致的路由。 我们将从以下几个方面改进工作。 首先，目前大多数XMOE实验都是针对语言任务进行的，例如多语言语言模型预训练、机器翻译等。 我们还将在视觉预训练 (Bao et al., 2022; Peng et al., 2022) 和多模态预训练 (Wang et al., 2022) 上评估所提出的方法。 其次，我们想报告扩大模型规模的结果。 专家数量越多，性能增益往往越大。

Ethical Considerations One of the negative societal impacts of training large-scale models is the high computational and environmental cost. Our paper focuses on improving SMoE, which is usually more efficient than dense model training with the same number of parameters. So better SMoE algorithms potentially save required computation and lessen CO2 emissions from computing. Moreover, X-MOE improves multilingual pre-training and fine-tuning, so that we can better transfer cross-lingual knowledge from high- to low-resource languages. The bless of larger model size brought by SMoE reduces the parameter conflicts of multilinguality, while keeping the computation cost manageable.

伦理考虑 训练大型模型的负面社会影响之一是高计算和环境成本。 我们的论文侧重于改进 SMoE，这通常比具有相同数量参数的密集模型训练更有效。 因此，更好的 SMoE 算法可能会节省所需的计算并减少计算产生的二氧化碳排放量。 此外，X-MOE 改进了多语言预训练和微调，使我们能够更好地将跨语言知识从高资源语言迁移到低资源语言。 SMoE 带来的更大模型尺寸的加持减少了多语言的参数冲突，同时保持计算成本可控。

## Acknowledgement
We would like to acknowledge Bo Zheng and Zhiliang Peng for the helpful discussions.

## References
* Armen Aghajanyan, Akshat Shrivastava, Anchit Gupta, Naman Goyal, Luke Zettlemoyer, and Sonal Gupta. Better fine-tuning by reducing representational collapse. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=OQ08SN70M1V. 
* Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. On the cross-lingual transferability of monolingual representations. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4623–4637, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.421. URL https://www.aclweb.org/anthology/2020.acl-main.421. 
* Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts. arXiv preprint arXiv:2112.10684, 2021. 
* Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEiT: BERT pre-training of image transformers. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=p-BhZSz59o4. 
* Emmanuel Bengio, Pierre-Luc Bacon, Joelle Pineau, and Doina Precup. Conditional computation in neural networks for faster models. arXiv preprint arXiv:1511.06297, 2015. 
* Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. 
* Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15750–15758, 2021. 
* Zewen Chi, Li Dong, Bo Zheng, Shaohan Huang, Xian-Ling Mao, Heyan Huang, and Furu Wei. 
* Improving pretrained cross-lingual language models via self-labeled word alignment. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3418–3430, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.265. URL https://aclanthology.org/2021.acl-long.265. 
* Aidan Clark, Diego de las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. arXiv preprint arXiv:2202.01169, 2022. 
* Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. TyDi QA: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics, 8:454–470, 2020. doi: 10.1162/tacl_a_00317. URL https://www.aclweb.org/anthology/2020.tacl-1.30. 
* Alexis Conneau and Guillaume Lample. Cross-lingual language model pretraining. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, pages 7057–7067, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/c04c19c2c2474dbf5f7ac4372c5b9af1-Abstract.html. 
* Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov. XNLI: Evaluating cross-lingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2475–2485, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1269. URL https://www.aclweb.org/anthology/D18-1269. 
* Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, pages 8440–8451. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.747. URL https://doi.org/10.18653/v1/2020.acl-main.747. 
* Damai Dai, Li Dong, Shuming Ma, Bo Zheng, Zhifang Sui, Baobao Chang, and Furu Wei. StableMoE: Stable routing strategy for mixture of experts. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7085–7095, Dublin, Ireland, May 2022. Association for Computational Linguistics. URL https://aclanthology.org/2022.acl-long.489. 
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Volume 1 (Long and Short Papers), pages 4171– 4186. Association for Computational Linguistics, 2019. doi: 10.18653/v1/n19-1423. URL https://doi.org/10.18653/v1/n19-1423. 
* Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=YicbFdNTTy. 
* Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. arXiv preprint arXiv:2112.06905, 2021. 
* Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for selfsupervised representation learning. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 3015–3024. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ermolov21a.html. 
* William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. arXiv preprint arXiv:2101.03961, 2021. 
* Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. XTREME: A massively multilingual multi-task benchmark for evaluating cross-lingual generalization. arXiv preprint arXiv:2003.11080, 2020. 
* Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1):79–87, 1991. 
* Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=YevsQ05DEN7. 
* Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. Neural computation, 6(2):181–214, 1994. 
* Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/ D18-2012. URL https://www.aclweb.org/anthology/D18-2012. 
* Kenichi Kumatani, Robert Gmyr, Felipe Cruz Salinas, Linquan Liu, Wei Zuo, Devang Patel, Eric Sun, and Yu Shi. Building a great multi-lingual teacher with sparsely-gated mixture of experts for speech recognition. arXiv preprint arXiv:2112.05820, 2021. 
* Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb. 
* Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In International Conference on Machine Learning, pages 6265–6274. PMLR, 2021. 
* Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. MLQA: Evaluating cross-lingual extractive question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7315–7330, Online, July 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.acl-main.653. 
* Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426, 2018. 
* Xiaoman Pan, Boliang Zhang, Jonathan May, Joel Nothman, Kevin Knight, and Heng Ji. Cross-lingual name tagging and linking for 282 languages. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1946–1958, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1178. URL https://www.aclweb.org/anthology/P17-1178. 
* Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40): 24652–24663, 2020. 
* Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. BEiT v2: Masked image modeling with vector-quantized visual tokenizers. ArXiv, abs/2208.06366, 2022. 
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 2019. URL http://www.persagen.com/files/misc/radford2019language.pdf. 
* Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html. 
* Afshin Rahimi, Yuan Li, and Trevor Cohn. Massively multilingual transfer for NER. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 151–164, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1015. URL https://www.aclweb.org/anthology/P19-1015. 
* Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems, 34, 2021. 
* Stephen Roller, Sainbayar Sukhbaatar, Jason Weston, et al. Hash layers for large sparse models. Advances in Neural Information Processing Systems, 34, 2021. 
* Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=B1ckMDqlg. 
* Tom Tirer and Joan Bruna. Extended unconstrained features model for exploring deep neural collapse. arXiv preprint arXiv:2202.08087, 2022. 
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, pages 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/ hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html. 
* Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. Image as a foreign language: BEiT pretraining for all vision and vision-language tasks. ArXiv, abs/2208.10442, 2022. 
* Yiren Wang, ChengXiang Zhai, and Hany Hassan. Multi-task learning for multilingual neural machine translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1022–1034, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.75. URL https://aclanthology.org/2020.emnlp-main.75. 
* Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzman, Armand Joulin, and Edouard Grave. CCNet: Extracting high quality monolingual datasets from web crawl data. arXiv preprint arXiv:1911.00359, 2019. 
* Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. arXiv preprint arXiv:2010.11934, 2020. 
* Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3687–3692, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1382. URL https://www.aclweb.org/ anthology/D19-1382. 
* Daniel Zeman, Joakim Nivre, Mitchell Abrams, and et al. Universal dependencies 2.5, 2019. URL http://hdl.handle.net/11234/1-3105. LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University. 
* Yanqi Zhou, Tao Lei, Han-Chu Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M. Dai, Zhifeng Chen, Quoc Le, and James Laudon. Mixture-of-experts with expert choice routing. arXiv preprint arXiv:2202.09368, 2022. 
* Zhihui Zhu, Tianyu DING, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, and Qing Qu. A geometric analysis of neural collapse with unconstrained features. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=KRODJAa6pzE. 
* Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. Designing effective sparse expert models. arXiv preprint arXiv:2202.08906, 2022. 

 

## A Model Hyperparameters
Table 8 presents the model hyperparameters of X-MOE. The gating temperature τ0 is initialized as 0.3 and 0.07 for the softmax gating and sigmoid gating, respectively. We use the same vocabulary as XLM-R (Conneau et al., 2020) with 250K subwords tokenized by SentencePiece (Kudo and Richardson, 2018).

Table 8: Model hyperparameters of X-MOE.


## B Hyperparameters for Pre-training

Table 9 presents the hyperparameters for pre-training.

Table 9: Hyperparameters for pre-training.


## C Hyperparameters for Fine-tuning

Table 10 presents the hyperparameters for fine-tuning.

Table 10: Hyperparameters for fine-tuning on the XTREME downstream tasks.
