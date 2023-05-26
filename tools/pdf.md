## 2. MEGABYTE Transformer
### 2.1. Overview
MEGABYTE is an autoregressive model for efficiently modeling long input sequences. MEGABYTE is comprised of 3 components: (1) a patch embedder that inputs a discrete sequence, embeds each element, and chunks it into patches of length P (2) a large global Transformer that contextualizes patch representations by performing self-attention over previous patches, and (3) a smaller local Transformer that inputs a contextualized patch representation from the global model, and autoregressively predict the next patch.

### 2.2. Components
Patch Embedder with patch size of P maps a byte sequence x0..T to a sequence of patch embeddings of length $K = \frac{T}{P}$ and dimension $P · D_G$.

First, each byte is embedded with a lookup table $E^{global-embed} ∈ R^{V ×D_G}$ to an embedding of size $D_G$ and positional embeddings are added. 

$h^{embed}_t = E_{x_t}^{global-embed} + E_t^{pos}$   t ∈ [0..T] (1)

Then, byte embeddings are reshaped into a sequence of K patch embeddings with dimension $P · D_G$. To allow autoregressive modelling, the patch sequence is padded to start with a trainable patch-sized padding embedding ($E^{global-pad} ∈ R^{P ×D_G}$ ), and the last patch is removed from the input. This sequence is the input to the global model, and is denoted $h^{global-in} ∈ R^{K×(P ·D_G})$ . 

$ h^{global-in}_k =  \begin{cases}  E^{global-pad} , if \ k = 0, \\ h^{embed}_{((k−1)·P ):(k·P )} , \  k ∈ [1, .., K)  \end{cases}$ (2)

Global Model is a decoder-only Transformer with dimension $P · D_G$ that operates on a sequence of K patches. It incorporates a self-attention mechanism and causal masking to capture dependencies between patches. It inputs a sequence of K patch representations $h^{global-in}_{0:K}$ , and outputs an updated representation $h^{global-out}_{0:K}$ by performing self-attention over previous patches. 

$h^{global-out}_{0:K} = transformer^{global}(h^{global-in}_{0:K} )$ (3)

The output of the final global layer $h^{global}_{0:K}$ contains K patch representations of dimension $P · D_G$. For each of these, we reshape them into sequences of length P and dimension $D_G$, where position p uses dimensions $p · D_G$ to $(p + 1) · D_G$. Each position is then projected to the dimension of the local model with a matrix $w^{GL} ∈ R^{D_G×D_L}$ where $D_L$ is the local model dimension. We then combine these with byte embeddings of size $D_L$ for the tokens in the next patch $E_{x(k·P +p−1)}^{local-embed} $ . The local byte embeddings is offset by one with a trainable local padding embedding ($E^{local-pad} ∈ R^{D_L}$ ) to allow autoregressive modelling within a patch. This results in a tensor $h^{local-in} ∈ R^{K×P ×D_L}$ . 

$h^{local-in}_{k,p} = w^{GL}h^{global-out}_{k,(p·D_G):((p+1)·D_G)} + E_{x(k·P +p−1)}^{local-embed} $ (4)

Figure 2. Summary of MEGABYTE with vocabulary V , sequence length T, global and local dimensions $D_G$ and $D_L$, and K patches of size P. Transformer layers use masked self attention to not observe information from future timesteps. 

Local Model is a smaller decoder-only Transformer of dimension $D_L$ that operates on a single patch k containing P elements, each of which is the sum of an output from the global model and an embedding of the previous byte in the sequence. K copies of the local models are run on each patch independently (and in parallel during training), computing a representation $h ^{local-out} ∈ R^{K×P ·D_L}$ . 

$h^{local-out}_{k,0:P} = transformer^{local}(h^{local-in}_{k,0:P} )$  (5)

Finally, we can compute the probability distribution over the vocabulary at each position. The pth element of the kth patch corresponds to element t of the complete sequence, where t = k · P + p: 

$p(x_t|x_{0:t}) = softmax(E^{local-embed}h^{local-out}_{k,p} )_{x_t}$ (6)

### 2.3. Variations and Extensions
We experiment with several extensions of MEGABYTE.

#### 2.3.1. CONVOLUTIONAL PATCH ENCODER
One limitation of chunking sequences into patches is that it is not translation invariant, and byte sequences may receive a different representation depending on their position in the patch. This may mean, for example, that a model has to relearn the meaning of a word at different offsets. To mitigate this issue, we experimented with augmenting the Patch Embedder with causal convolutional layers, which allow translation-invariant contextual representations of the bytes before they are chunked into patches. We use a stack of convolutional layers, with filter sizes of 3, 5 and 7.

#### 2.3.2. CROSS-PATCH ATTENTION
The Local model uses short sequences for efficiency, and relies on the Global model for long-range information. However, we can increase the context of the Local model with little overhead by allowing it to condition on r elements from the previous patch. This approach allows the Global model to focus on a longer-range context. Specifically, when computing self-attention in each layer, we concatenate the keys and values with the last r keys and queries from the previous patch. We use rotary embeddings (Su et al., 2021) to model relative positions between elements in the sequence. This approach is reminiscent of TransformerXL (Dai et al., 2019) but differs by being fully differentiable.

#### 2.3.3. STRIDED INFERENCE
We observed empirically that the per-token loss within each patch would increase towards the end of the patch, as the prediction relies more on the weaker Local model. To alleviate this issue, we propose strided inference, in which we predict the sequence with two forward passes of the full model, whose inputs are offset by p/2 positions from each other. We then combine the first p/2 positions in each patch for our predictions to predict the complete sequence. Similarly to sliding window techniques (Press et al., 2020), this approach doubles the cost of inference but improves results.

### 2.4. Motivation
Having described the model, we briefly discuss the motivation behind some of the architectural choices.

Why is the local model needed? Many of the efficiency advantages of the MEGABYTE design could be realized with the Global model alone, which would resemble a decoder version of ViT (Dosovitskiy et al., 2020). However, the joint distribution over the patch $p(x_{t+1}, .., x_{t+P} |x_{0..t})$ has an output space of size 256P so direct modeling is only tractable for very small patches. We could instead factor the joint distribution into conditionally independent distributions $p(x_{t+1}|x_{0..t})..p(x_{t+P} |x_{0..t})$, but this would greatly limit the model’s expressive power. For example, it would be unable to express a patch distribution such as 50% cat and 50% dog, and would instead have to assign probability mass to strings such as cag and dot. Instead, our autoregressive Local model conditions on previous characters within the patch, allowing it to only assign probability to the desired strings.

Increasing Parameters for Fixed Compute Transformer models have shown consistent improvements with parameter counts (Kaplan et al., 2020). However, the size of models is limited by their increasing computational cost. MEGABYTE allows larger models for the same cost, both by making self attention sub-quadratic, and by using large feedforward layers across patches rather than individual tokens.

Re-use of Established Components MEGABYTE consists of two transformer models interleaved with shifting, reshaping and a linear projection. This re-use increases the likelihood that the architecture will inherit the desirable scaling properties of transformers.

## 3. Efficiency Analysis
### 3.1. Training Efficiency
We analyze the cost of different architectures when scaling both the sequence length and size of the models.

Attention The cost of the self attention in a transformer architecture for a sequence of length T has $O(T^2)$ complexity. Much work has been explored reducing this; for example, Sparse Transformers (Child et al., 2019) and Routing Transformers (Roy et al., 2020) show strong results with a complexity $O(T^{\frac{3}{2}})$. Numerous linear attention mechanisms have also been proposed (Katharopoulos et al., 2020; Choromanski et al., 2020), although we are not aware of competitive results on large scale language modeling tasks. As a function of sequence length T and patch size P, the Global model has a sequence of length $\frac{P}{T}$ so uses $O(\frac{T^2}{P^2})$ operations, and the Local model uses PT sequences of length P so uses $O(\frac{T P^2}{P} ) = O(P^T)$ operations. The overall cost of MEGABYTE is therefore in $O(\frac{T^2}{P^2} +TP)$. P is a hyperparameter that is chosen to create an architecture for sequences of size T. By setting $P = T^\frac{1}{3}$ the complexity is in $O(T^\frac{4}{3} ).Using much shorter patches of $P = T^\frac{1}{5}$ would give a complexity of $O(T^\frac{8}{5} )$. The cost is less than the transformer for all non-trivial values of P such that 1 < P < T.

Figure 3. Computational cost (FLOPS/token) for different model architectures at different scales. MEGABYTE architectures (here with P = 8) use less FLOPS than equivalently sized Transformers and Linear Transformers (Katharopoulos et al., 2020) across a wide range of model sizes and sequence lengths, allowing larger models to be used for the same computational cost.

Feedforward Layers However, attention is not the main cost in large transformers. Instead of increasing the sequence length, transformers are more commonly scaled by increasing the dimension of their latent state d, and the feedforward network cost dominates the model’s overall cost (Kaplan et al., 2020). For example, in the GPT3 architecture, the quadratic self-attention computation accounts for only 1.4% of FLOPS. Following the approximation of (Kaplan et al., 2020), a forward pass with a large transformer with m non-embedding parameters on a sequence of length T uses roughly 2mT FLOPS. MEGABYTE contains two transformers: the Global model uses mg parameters on a sequence of length $\frac{T}{P}$ , and a Local model with ml parameters that sees $\frac{T}{P}$ sequences of length P, giving an estimate of $2T(\frac{m_g}{P} + m_l)$ FLOPS. When $m_g >> m_l$ , the FLOPS used by MEGABYTE is approximately $\frac{2Tm_g}{P}$ , allowing a model P times larger than a transformer with equivalent FLOPS. This analysis holds irrespective of any efficient attention mechanisms used in the transformer.

Combined Analysis To understand efficiency at different sequence lengths and model sizes, we calculate the total FLOPS used by transformers, Linear Transformers and MEGABYTE. For each operation, we use FLOP estimates from (Kaplan et al., 2020), except for attention in Linear Transformers, which we estimate as 9D FLOPS/- token1 , where D is the model embedding dimension. Figure 3 shows that for models of size 660M to 173B and sequence lengths of up to 1M tokens, MEGABYTE with P = 8 uses less FLOPS than either transformers or Linear Transformers. Baseline model architectures are based on GPT3, and Megabyte global/local model sizes are 452M/151M, 5.8B/604M, 170B/3.2B respectively. 

1This may underestimate the time taken by Linear Transformer decoders, which use a recurrence mechanism that is harder to parallelize on current hardware.

### 3.2. Generation Efficiency
Generating long sequences with transformers is slow, because the input to each timestep is the output from the previous timestep, meaning each layer must be computed for each token serially. As running a layer on a single token typically does not saturate the amount of parallelism available within a GPU, for analysis, we model each layer as a constant cost independently of size. Consider a MEGABYTE model with $L_{global}$ layers in the Global model and $L_{local}$ layers in the Local model and patch size P, compared with a Transformer architecture with $L_{local} + L_{global}$ layers. Generating each patch with MEGABYTE requires a sequence of $O(L_{global} + P · L_{local})$ serial operations, whereas the Transformer requires $O(P · L_{global} + P · L_{local})$ serial operations. When $L_{global} >> L_{local}$ (i.e. the Global model has many more layers than the Local model), MEGABYTE can reduce inference costs by a factor close to P.

## 4. Experimental setup
### 4.1. Controlling for Compute and Data
Models show consistent improvements when increasing both data and compute (Kaplan et al., 2020; Hoffmann et al., 2022), meaning that one model can outperform another because of an increased training budget instead of an improved architecture. However, in practice, both compute and data are typically limited. We conduct experiments using a fixed compute and data budget across all models to focus comparisons solely on the model architecture rather than training resources. To achieve this, we adjust model hyperparameters (mainly, number of layers) within each architecture so that the forward pass time taken per byte is matched, and then train all models for the same number of bytes.

### 4.2. Comparison Systems
We compare MEGABYTE with both a standard decoderonly Transformer and PerceiverAR (Hawthorne et al., 2022).PerceiverAR extends the original transformer with a single cross-attention layer over a much longer context sequence, and is the best performing general purpose autoregressive model we are aware of and achieves state-of-the-art results across several modalities. We implemented both models in the same codebase, and all models share a similar data loader, preprocessing step, and trainer to avoid any artifacts in our compute-controlled experiments.

### 4.3. Training Procedure
All models were trained using the Metaseq2 code base (Zhang et al., 2022b). The training used the PyTorch framework (Paszke et al., 2019), with fairscale to improve memory efficiency through fully sharded model and optimizer states (Baines et al., 2021). Mixed precision training was used to improve training efficiency at scale (Micikevicius et al., 2017). More training details and various model parameters can be found in Section A.1 in the Appendix.

To validate our implementation of PerceiverAR, we reproduced their experiments on downsized ImageNet at 64 pixels. By carefully matching hyperparameters, we achieved a bits per byte (bpb) score of 3.53, compared to the reporte 3.54 in the original paper.

### 4.4. Inference Methods
Several techniques have been proposed for trading off speed for performance during inference with language models, including sliding windows (Press et al., 2020) and our strided inference (Section 2.3.3). We only use these methods when comparing with prior published work (Tables 8 and 4).

## 5. Language Modeling
We evaluated the performance of MEGABYTE on language modeling on a set of 5 diverse datasets emphasizing longrange dependencies: Project Gutenberg (PG-19), Books,Stories, arXiv, and Code.

Datasets We experiment on a range of long form text datasets. The PG-19 dataset (Rae et al., 2019b) consists of English-language books written before 1919 and is extracted from the Project Gutenberg online library. The Stories dataset (Trinh & Le, 2018) is a subset of CommonCrawl data meant to emulate Winograd schemas. Books (Gao et al., 2020) is another collection of English-language books. The arXiv dataset is a collection of technical publications written in LATEX from the arXiv online archive. Finally, the Code dataset is a large publicly available dataset of open source code, under Apache, BSD or MIT licenses. More details on dataset sizes and document lengths are shared in Table 6.

Controlled Experiments Table 7, lists bpb on each dataset.Each model is trained for 80 billion bytes, and models are scaled to use the same compute budget. We carefully tune hyperparameters for all architectures to best utilize the available compute budget. MEGABYTE consistently outperforms both baseline transformers and PerceiverAR across all datasets. We use the same sets of parameters on all datasest. In all experiments presented in Table 7, transformer has size of 320M with context length of 1024, PerceiverAR has size of 248M with context size of 8192 and latent size of 1024, and MEGABYTE global/local model sizes are 758M/262M with context length of 8192 and patch size of 8.

2 https://github.com/facebookresearch/metaseq

Table 1. Text dataset sizes and mean document lengths.

Table 2. Performance (bits-per-byte) of compute and data controlled MEGABYTE, PerceiverAR, and Transformer models on various text modalities. 

Scaling Experiment We scale up our training data on PG- 19 (Table 8), and compare MEGABYTE with byte baselines, as well as converting all results to word-level perplexities to benchmark with state-of-art token based models.

We train a byte-level Transformer, PerceiverAR and MEGABYTE models for 400B bytes and the same compute budget using same model parameters as in the controlled experiments. We find that MEGABYTE outperforms other byte-level models by a wide margin at this scale.3

We also compare with the best previously reported numbers for sub-word models. These results may be confounded by differing amounts of compute and tuning used, but show that MEGABYTE gives results competitive with state-of-theart models trained on subwords. These results suggest that MEGABYTE may allow future large language models to be tokenization-free.

## 6. Image Modeling
### 6.1. Sequence Modeling on ImageNet
We test MEGABYTE on variants of the autoregressive image generation task on ImageNet (Oord et al., 2016), to measure its ability to efficiently use long context. We test on three different resolutions of images, ranging from 64×64 to 640×640 pixels – the latter requiring the effective modeling of sequences with over 1.2M tokens. This generation task becomes increasingly challenging as the image’s resolution grows: doing well on this task requires the modeling of local patterns (textures, lines, etc.) and long-range context that provides information about the high level structure of the image. Inspired by recent works in Vision Transformers (Dosovitskiy et al., 2020), we model image data patch by patch (more details can be found in Appendix D.1).

3The only prior byte-level experiments we are aware of are at a smaller scale in Hutchins et al. (2022), who report results equivalent to test perplexities of 46.5 with a version of the BlockRecurrent transformer, and 49.5 with Memorizing Transformers (Wu et al., 2022), compared to 36.4 with our model. 

### 6.2. Comparison with State of the Art
We train a large MEGABYTE model on ImageNet 64x64 with Global and Local models sized 2.7B and 350M parameters, respectively, for 1.4T tokens. We estimate that training this model consumed less than half the GPU hours we would have needed to reproduce the best PerceiverAR model described by (Hawthorne et al., 2022). As shown in Table 8,MEGABYTE matches the state-of-the-art performance of PerceiverAR whilst using only half the compute.

### 6.3. Scaling to higher resolutions
We compare three transformer variants (vanilla, PerceiverAR, MEGABYTE) to test scalability to long sequences on increasingly large image resolutions. We use our own implementations of these in the same framework and budget the same amount of GPU hours and data to train each of these model variants.

MEGABYTE is able to handle all sequence lengths with a single forward pass of up to 1.2M tokens. We found neither the standard Transformer nor PerceiverAR could model such long sequences at a reasonable model size, so instead we split images into segments of size 1024 and 12000 respectively. For Megabyte, we set patch size as 12 for Image64 and patch size as 192 for Image256 and Image640 datasets. Model sizes are adjusted to match overall training speeds across models and we do not use any form of sliding window evaluation in this experiment. As seen in Table 5, MEGABYTE outperforms baselines across all resolutions in this compute-controlled setting. The precise settings used for each of the baseline models such as context length and number of latents are summarized in Table 14.

Results show that MEGABYTE outperforms the other systems at all resolutions, demonstrating an effective model of sequences of over 1M bytes.

## 7. Language Modeling
We evaluated the performance of MEGABYTE on language modeling on a set of 5 diverse datasets emphasizing longrange dependencies: Project Gutenberg (PG-19), Books,Stories, arXiv, and Code.


Table 3. Larger scale experiments on PG19, converting bits-per-byte to word-level perplexities for comparison with prior work. Results below the line are compute-matched. MEGABYTE outperforms other byte models by a wide margin, and gives results competitive with state-of-the-art models trained on subwords.

Table 4. Bits per byte (bpb) on ImageNet 64×64. MEGABYTE matches the current state-of-the-art while only using half the amount of GPU hours to train.

Table 5. Bits per byte (bpb) on ImageNet with different resolutions. All models use the same compute and data. MEGABYTE scales well to sequences of over 1M tokens.

Datasets We experiment on a range of long form text datasets. The PG-19 dataset (Rae et al., 2019b) consists of English-language books written before 1919 and is extracted from the Project Gutenberg online library. The Stories dataset (Trinh & Le, 2018) is a subset of CommonCrawl data meant to emulate Winograd schemas. Books (Gao et al., 2020) is another collection of English-language books. The arXiv dataset is a collection of technical publications written in LATEX from the arXiv online archive. Finally, the Code dataset is a large publicly available dataset of open source code, under Apache, BSD or MIT licenses. More details on dataset sizes and document lengths are shared in Table 6.

Controlled Experiments Table 7, lists bpb on each dataset.Each model is trained for 80 billion bytes, and models are scaled to use the same compute budget. We carefully tune hyperparameters for all architectures to best utilize the available compute budget. MEGABYTE consistently outperforms both baseline transformers and PerceiverAR across all datasets. We use the same sets of parameters on all datasest.

Table 6. Text dataset sizes and mean document lengths.

In all experiments presented in Table 7, transformer has size of 320M with context length of 1024, PerceiverAR has size of 248M with context size of 8192 and latent size of 1024, and MEGABYTE global/local model sizes are 758M/262M with context length of 8192 and patch size of 8.

Scaling Experiment We scale up our training data on PG- 19 (Table 8), and compare MEGABYTE with byte baselines, as well as converting all results to word-level perplexities to benchmark with state-of-art token based models.

We train a byte-level Transformer, PerceiverAR and MEGABYTE models for 400B bytes and the same compute budget using same model parameters as in the controlled experiments. We find that MEGABYTE outperforms other byte-level models by a wide margin at this scale.4

We also compare with the best previously reported numbers for sub-word models. These results may be confounded by differing amounts of compute and tuning used, but show that MEGABYTE gives results competitive with state-of-theart models trained on subwords. These results suggest that MEGABYTE may allow future large language models to be tokenization-free. 

4The only prior byte-level experiments we are aware of are at a smaller scale in Hutchins et al. (2022), who report results equivalent to test perplexities of 46.5 with a version of the BlockRecurrent transformer, and 49.5 with Memorizing Transformers (Wu et al., 2022), compared to 36.4 with our model.

Table 7. Performance (bits-per-byte) of compute and data controlled MEGABYTE, PerceiverAR, and Transformer models on various text modalities.

## 8. Audio Modeling
Audio has aspects of both the sequential structure of text and the continuous nature of images, so is an interesting application for MEGABYTE.

Raw audio is typically stored as a sequence of 16-bit integer values (one per timestep); a softmax layer would need to output 65,536 probabilities per timestep to model all possible values. To address this issue, various techniques have been developed to reduce the memory and computational requirements of the softmax layer. For instance, van den Oord et al. (2016) apply µ-law companding transformation and quantizes the input into 256 possible values. Alternatively, van den Oord et al. (2017) model the samples using the discretized mixture of logistics distribution introduced by Salimans et al. (2017). Finally, Kalchbrenner et al. (2018) use a dual softmax technique to produce 8 coarse and 8 fine bits. In our approach, we simplify the audio modeling process by directly reading the bytes (256 possible values) from the audio file and conducting an autoregressive language model on top of that. This greatly streamlines the modeling process, making it easier and more efficient.

Our audio modeling approach focuses on 16 kHz, 16-bit audio, which equates to 32k bytes per one-second clip. We use an extensive audio dataset consisting of 2 terabytes (roughly 18,000 hours) of audio. We use a sequence length of 524,288, a patch size of 32, and a batch size of 32 to facilitate model training. By utilizing these settings, we can effectively train our model on large volumes of audio data, helping to improve its accuracy and efficacy.

Our model obtains bpb of 3.477, much lower than the results with perceiverAR (3.543) and vanilla transformer model (3.567). More ablation results are presented in Table 10.

## 9. Analysis
### 9.1. Generation speed
We also compare the text generation speed between MEGABYTE and a transformer. We compare a 350M parameter baseline transfomer and a MEGABYTE model with a 1.3B parameter Global model and a 218M parameter local model, trained on PG19 with equal compute. As shown in Table 9, the MEGABYTE model achieves much lower perplexity as expected. However, MEGABYTE also generates a sequence of 8192 tokens 40% faster than transformer, despite having over 4 times the parameters. This speed up is due to the bulk of the parameters being in the Global model, which only needs to be computed once for every 8 tokens, whereas all the parameters in the baseline model are used on every token.

Figure 4. Average log probability assigned to the token at different positions within the context length by MEGABYTE model with 8192 context size and by a vanilla transformer model trained using the same compute (PG19 test set). MEGABYTE likelihoods rise throughout its context window, demonstrating that it can use tokens from 8k bytes previously to improve its predictions. 

### 9.2. Model Components
In Table 10, we analyze the significance of different components in the MEGABYTE architecture by studying arXiv,Librilight-L and ImageNet256 datasets. Removing Local (w/o local model) or global (w/o global model) model, we observe a substantial increase in bpb on all datasets, showing that both parts are crucial. The performance of the model without the cross-patch local model (w/o cross-patch local model) is competitive, indicating that the architecture is robust to this modification. We observe slight improvement on the Librilight-L and ImageNet256 datasets by augmenting the MEGABYTE model with a CNN encoder (w/ CNN encoder). This suggests that the MEGABYTE architecture can benefit from integrating alternative encoding mechanisms.

### 9.3. Effective Use of Context
Long-context models often struggle to benefit from the full context (Sun et al., 2021). Figure 4 shows that later tokens within each context window consistently have a higher likelihood, indicating that MEGABYTE can effectively use at least 8k bytes of context on the PG19 dataset.

### 9.4. Strided Inference
We find that within a single patch, on average, the MEGABYTE performs worse on later tokens within a patch (see Figure 5). Section 2.3.3 proposes strided inference as a solution, where two forward passes are performed offset by P 2 tokens, and results from the first half of each patch are combined. Table 11 shows performance improvements from strided inference, which are additive with the standard sliding window.


Table 8. Larger scale experiments on PG19, converting bits-per-byte to word-level perplexities for comparison with prior work. Results below the line are compute-matched. MEGABYTE outperforms other byte models by a wide margin, and gives results competitive with state-of-the-art models trained on subwords.

Table 9. Comparison of bits per byte (bpb) and generation speed of 8192 bytes of transformer model (with context length 1024) and MEGABYTE with context length 8192 and patch size 8.

Table 10. Ablation of MEGABYTE model components, showing that both Local and Global models are critical to strong performance, but the architecture is robust to other modifications. We report bits-per-byte on text, audio, and image prediction tasks. All models within a column are trained using the same compute and data. The hyperparameters are listed in Table 14. 



### 9.5. Hyperparameters
MEGABYTE introduces several additional hyperparameters. We tuned these parameters independently for different modalities and reported performance based on the best setting we found. All experiments in the same group use the same compute.

Patch Size. We experimented with various patch sizes on Image256 dataset and found that there is a wide range of values where MEGABYTE performs similarly. We found similar robustness against the choice of this hyperparameter across all modalities, although the optimal patch size itself

Figure 5. An illustration of strided inference with patch size 8. Lines below the text represent the patches used in the two rounds of inference, the plot above it represents the average probability assigned to the token at a given position within a patch. By considering only the first half of each patch from the two rounds of inference and combining them (bold lines on top), we achieve a better overall bpb.

Table 11. Performance of various inference techniques on the PG19 test set using our best MEGABYTE model. can be different across modalities.

Table 12. Effects of patch size on performance on the Image256 dataset. All versions use the same amount of GPU hours and data.

Local to Global model Size Ratio. We experimented with different Local/Global model size ratios on PG19 dataset. By grouping bytes into patches, MEGABYTE effectively uses P times less tokens for the Global model as on the Local model—enabling us to increase the size of the Global model without reduced cost. We find that a given compute budget is spent optimally when the Global model has more parameters than the Local model. This trend was consistent across all modalities and various patch sizes.

Table 13. Effects of Local / Global model size on performance on the PG19 dataset. Increasing the capacity of global model improves performance. Models are compute and data matched.

## 10. Related Work
Prior research has explored the possibility of improving the efficiency of Transformers on long sequences, primarily motivated by mitigating the quadratic cost of self-attention.

Efficient Encoder Models Several related techniques to ours have been developed for transformer encoder architectures but cannot be straightforwardly applied to decoders. In particular, patchifying operations have previously been used in image encoder models such as ViT (Dosovitskiy et al., 2020), and down- and up-sampling operations have been used for text encoders (Clark et al., 2022), but such methods cannot be naively applied to decoder-only models without leaking information to future bytes in the same patch. MEGABYTE generalizes these approaches to an effi- cient decoder model by using a intra-patch transformer to predict each sequence element’s likelihood, and offseting the inputs to the two models to avoid leaking information. Jaegle et al. (2021) use self-attention on a shorter latent sequence also resembles patchification, but this technique cannot easily be applied to decoder architectures without leaking information to future timesteps.

Efficient Decoder models Improving the efficiency of decoder models is more challenging because of the need to make one prediction per timestep, and not leak information to future timesteps. The most popular approaches can be categorized as (1) chunking sequences into smaller blocks, and propagating information from previous blocks with either recurrence (Dai et al., 2019; Hutchins et al., 2022) or crossattention (Hawthorne et al., 2022), (2) linear alternatives to attention, which typically involve forms of token-level recurrence (Katharopoulos et al., 2020) or state space models (Gu et al., 2021; Smith et al., 2022; Ma et al., 2022), or (3) sparse approximations of attention (Kitaev et al., 2020; Beltagy et al., 2020; Child et al., 2019; Wu et al., 2022).

However, the performance of dense attention means it is typically still chosen for large scale decoders (Touvron et al., 2023; Chowdhery et al., 2022). MEGABYTE takes the alternative approach of decomposing the complete sequence into two shorter sequences, giving sub-quadratic attention. We also note that feedforward networks are the dominant cost in large decoders, not self-attention. Our approach to compressing sequences allows much larger models than would be possible when using large feedforward networks at every timestep.

Tokenization The most common approach to shortening sequence lengths in Transformer decoders is to pre-process the input with a form of tokenization, in which multiple bytes are mapped to a single discrete token from a fixed vocabulary. For text, this can be done losslessly using methods such as BPE (Sennrich et al., 2015) and SentencePiece (Kudo & Richardson, 2018), but these approaches can require language-specific heuristics (Radford et al., 2019), limit out-of-domain performance (Sharami et al., 2023), and can affect prompting and truncated sampling in unpredictable ways.5 The amount of high-frequency information in images and audio means that tokenization cannot be performed losslessly, and instead clustering (Hsu et al., 2021) or discrete auto-encoders (Ramesh et al., 2021) are used to compress the inputs, which lose information and likely limit generative model performance. Our patches are analogous to traditional lossless tokens, and the Local model performs the role of mapping a hidden state to a distribution over possible patches.

## 11. Conclusion
We introduced MEGABYTE, a scaleable architecture for modeling long sequences. MEGABYTE outperforms existing byte-level models across a range of tasks and modalities, allowing large models of sequences of over 1 million tokens. It also gives competitive language modeling results with subword models, which may allow byte-level models to replace tokenization. However, the scale of experiments here is far below those of state-of-the-art language models (Brown et al., 2020), and future work should explore scaling MEGABYTE to much larger models and datasets.

## References

## A. Appendices
### A.1. Training Details
To ensure stable training, we applied gradient clipping with a maximum norm of 1.0 and used the Adam optimizer with β1 = 0.9, β2 = 0.98 (Kingma & Ba, 2015). We used the built-in polynomial decay learning rate scheduler in MetaSeq with 500 warmup updates and the end learning rate set to 0. All models are trained with pre-norm and using ReLU activation. We apply a dropout of 0.1 throughout, but we do not apply any dropout to embeddings. We also use weight decay of 0.1. To initialize the weights, we use a variant based on Megatron-LM codebase, which involves using a normal distribution with a mean of zero and a standard deviation of 0.006. We truncate this normal distribution within two standard deviations and observed substantial gain in both training stability and performance.

### A.2. Model Details
As discussed in Section 4.1, we conduct experiments using a fixed compute and data budget across all models to focus our comparisons solely on the model architecture rather than training resources. To achieve this, we adjust model hyperparameters within each architecture so that the time taken for a single update is matched and then train all models for the same number of updates. We list all of model details in Table 14 and Table 15.

Table 14. Common Model architecture details by size. For each model size, we show the number of layers (#L), the embedding size (dmodel), the number of attention heads (#H), the dimension of each attention head (dhead).

Table 15. Model architecture details. We report the model size, the embedding size (D), number of layaers(L), total batch size (BS), learning rate(LR), and context length. When we vary the number of model layers from the standard amount for the given size (Table 14), we note this accordingly. For PerceiverAR models, we note the number of latents used, and for MEGABYTE models we note the patch sizes.

### B. Pseudocode 

### C. PerceiverAR Implementation
To reproduce PerceiverAR in a compute-controlled setting we extended the standard transformer implementation in metaseq with an additonal cross attention layer to compute the latents and match the architecture of PerceiverAR. We trained the model by sampling random spans from each text, matching the procedure used in the PerceiverAR codebase. To be consistent with the original work, we use sliding window evaluation with a stride of num latents/2 unless otherwise noted. In several cases we used the standard metaseq implementation as opposed to specific techniques reported in the original paper: 1) we used standard attention dropout instead of cross-attention dropout 2) We did not implement chunked attention. We verified our implementation by reproducing the ”Standard Ordering” experiments in Table 5 of the Perceiver AR paper. After carefully matching context size, number of latents, the amount of data and training steps used and learning rate, we achieved 3.53 bpb vs 3.54 reported in the original paper.

### D. More results
#### D.1. Patch scan Implementation
Images have a natural structure, containing a grid of n×n pixels each composed of 3 bytes (corresponding to color channels). We explore two ways of converting images to sequences for modeling (see Figure 6). Firstly, raster scan where the pixels are linearized into 3 bytes and concatenated row-by-row. Secondly, patch scan where we create patches of shape p × p × 3 bytes where $p = \sqrt{\frac{P}{3}}$ , and then use a raster scan both within and between patches. Unless otherwise specified, MEGABYTE models use patch scan for image data.

Figure 6. Two ways to model 2D data sequentially. Left, raster scan, by taking bytes row by row and left to right; right, patch scan, where we first split an image into patches, and do raster scan across patches and within a patch. (T=36, K=9, P=4).

#### D.2. Patch scan vs Raster scan
The patch scan method is inspired by recent works in Vision Transformers (Dosovitskiy et al., 2020), and it is more effective than raster scan for modeling image sequencing. We found it improves both MEGABYTE and Perceiver AR.  

Table 16. ImageNet256 performance with patch scan vs raster scan for MEGABYTE and Perceiver AR.

#### D.3. Longer sequence modeling
For our pg19 scaling experiment, we also use longer context length for MEGABYTE. The results are shown in Table 17. With longer sequence, we didn’t observer further improvement, consistent with findings in Hawthorne et al. (2022). We think we will benefit more from longer sequence when we futher scale up the model size and data. 
