# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
FlashAttention：具有IO意识的快速高效记忆精确注意力 2022.5.27 https://arxiv.org/abs/2205.14135

## Abstract
Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IOaware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4× speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

由于自我注意的时间和记忆复杂性在序列长度上是二次型的，因此变换器在长序列上是缓慢的，并且需要记忆。近似注意力方法试图通过权衡模型质量来降低计算复杂度来解决这个问题，但通常无法实现挂钟加速。我们认为，一个缺失的原理是使注意力算法IO-aware考虑GPU内存级别之间的读写。我们提出了FlashAttention，这是一种IO感知的精确注意力算法，它使用平铺来减少GPU高带宽存储器（HBM）和GPU片上SRAM之间的存储器读/写次数。我们分析了FlashAttention的IO复杂性，表明它比标准注意力需要更少的HBM访问，并且对于一系列SRAM大小来说是最优的。我们还将FlashAttention扩展到块稀疏注意力，产生了一种比任何现有近似注意力方法都快的近似注意力算法。FlashAttention训练变形金刚的速度比现有基线快：与MLPerf 1.1训练速度记录相比，BERT大型（序列长度512）上的端到端挂钟加速率为15%，GPT-2上的加速率为3倍（序列长度1K），远程竞技场上的加速度为2.4倍（序列长1K-4K）。FlashAttention和块稀疏FlashAttendance在Transformers中实现更长的上下文，产生了更高质量的模型（GPT-2的困惑度提高了0.7，长文档分类的提升点提高了6.4）和全新的功能：第一个在Path-X挑战（序列长度16K，61.4%精度）和Path-256（序列长度64K，63.1%精度）上实现了优于机会性能的变形金刚。

## 1 Introduction
Transformer models [82] have emerged as the most widely used architecture in applications such as natural language processing and image classification. Transformers have grown larger [5] and deeper [83], but equipping them with longer context remains difficult [80], since the self-attention module at their heart has time and memory complexity quadratic in sequence length. An important question is whether making attention faster and more memory-efficient can help Transformer models address their runtime and memory challenges for long sequences.

Transformer模型[82]已成为自然语言处理和图像分类等应用中使用最广泛的架构。变形金刚越来越大[5]，越来越深[83]，但为它们配备更长的上下文仍然很困难[80]，因为它们核心的自我注意模块的时间和记忆复杂度是序列长度的二次方。一个重要的问题是，提高注意力的速度和内存效率是否有助于Transformer模型解决长序列的运行时和内存挑战。

Many approximate attention methods have aimed to reduce the compute and memory requirements of attention. These methods range from sparse-approximation [51, 74] to low-rank approximation [12, 50, 84], and their combinations [3, 9, 92]. Although these methods reduce the compute requirements to linear or near-linear in sequence length, many of them do not display wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate with wall-clock speed) and tend to ignore overheads from memory access (IO).

许多近似注意力方法旨在减少注意力的计算和记忆需求。这些方法的范围从稀疏近似[51，74]到低秩近似[12，50，84]，以及它们的组合[3，9，92]。尽管这些方法将计算要求降低到线性或接近线性的序列长度，但其中许多方法并没有显示出与标准关注相比的挂钟加速，也没有得到广泛采用。一个主要原因是，他们专注于降低FLOP（这可能与墙上的时钟速度无关），并倾向于忽略来自内存访问（IO）的开销。

In this paper, we argue that a missing principle is making attention algorithms IO-aware [1]—that is, carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM [45], Figure 1 left). On modern GPUs, compute speed has out-paced memory speed [61, 62, 63], and most operations in Transformers are bottlenecked by memory accesses [43]. IO-aware algorithms have been critical for similar memory-bound operations, when reading and writing data can account for a large portion of the runtime—such as database joins [71], image processing [70], numerical linear algebra [4], and more [40, 85]. However, common Python interfaces to deep learning such as PyTorch and Tensorflow do not allow fine-grained control of memory access.

在本文中，我们认为一个缺失的原理是使注意力算法IO感知[1]，即仔细考虑对不同级别的快速和慢速存储器的读取和写入（例如，在快速GPU片上SRAM和相对较慢的GPU高带宽存储器之间，或HBM[45]，图1左）。在现代GPU上，计算速度超过了内存速度[61，62，63]，Transformers中的大多数操作都受到内存访问的限制[43]。IO感知算法对于类似的内存绑定操作至关重要，因为读写数据可以占运行时间的很大一部分，如数据库联接[71]、图像处理[70]、数值线性代数[4]等[40，85]。然而，深度学习的常见Python接口，如PyTorch和Tensorflow，不允许对内存访问进行细粒度控制。

Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large 𝑁 × 𝑁 attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large 𝑁 × 𝑁 attention matrix to HBM, resulting in an 7.6× speedup on the attention computation.
图1：左图：FlashAttention使用平铺来防止大型实体化𝑁 × 𝑁 在（相对）慢的GPU HBM上的注意力矩阵（虚线框）。在外循环（红色箭头）中，FlashAttention循环通过K和V矩阵块，并将它们加载到快速片上SRAM。在每个块中，FlashAttention在Q矩阵块（蓝色箭头）上循环，将它们加载到SRAM，并将注意力计算的输出写回HBM。右图：加快PyTorch对GPT-2的关注。FlashAttention不读写大号𝑁 × 𝑁 注意力矩阵到HBM，导致注意力计算的7.6倍加速。

We propose FlashAttention, a new attention algorithm that computes exact attention with far fewer memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from HBM. This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large intermediate attention matrix for the backward pass. We apply two well-established techniques to address these challenges. (i) We restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling). (ii) We store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the backward pass, which is faster than the standard approach of reading the intermediate attention matrix from HBM. We implement FlashAttention in CUDA to achieve fine-grained control over memory access and fuse all the attention operations into one GPU kernel. Even with the increased FLOPs due to recomputation, our algorithm both runs faster (up to 7.6x on GPT-2 [67], Figure 1 right) and uses less memory—linear in sequence length—than standard attention, thanks to the massively reduced amount of HBM access.

我们提出了FlashAttention，这是一种新的注意力算法，可以用更少的内存访问来计算精确的注意力。我们的主要目标是避免在HBM之间读写注意力矩阵。这需要（i）在不访问整个输入的情况下计算softmax约简（ii）不存储用于后向传递的大的中间注意力矩阵。我们应用两种公认的技术来解决这些挑战。（i） 我们重组注意力计算，将输入分成块，并在输入块上进行多次遍历，从而逐步执行softmax减少（也称为平铺）。（ii）我们存储来自前向传递的softmax归一化因子，以在后向传递中快速重新计算片上注意力，这比从HBM读取中间注意力矩阵的标准方法更快。我们在CUDA中实现了FlashAttention，以实现对内存访问的细粒度控制，并将所有注意力操作融合到一个GPU内核中。即使由于重新计算而增加了FLOP，由于HBM访问量大大减少，我们的算法运行速度更快（在GPT-2[67]上高达7.6倍，图1右侧），并且在序列长度上使用的内存线性比标准关注更少。

We analyze the IO complexity [1] of FlashAttention, proving that it requires 𝑂(𝑁 2𝑑 2𝑀−1 ) HBM accesses where 𝑑 is the head dimension and 𝑀 is the size of SRAM, as compared to Ω(𝑁 𝑑 + 𝑁 2 ) of standard attention. For typical values of 𝑑 and 𝑀, FlashAttention requires many times fewer HBM accesses compared to standard attention (up to 9× fewer, as shown in Fig. 2). Moreover, we provide a lower bound, showing that no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes.

我们分析了FlashAttention的IO复杂性[1]，证明它需要𝑂(𝑁 2.𝑑 2.𝑀−1）HBM接入，其中𝑑 是头部尺寸和𝑀 是SRAM的大小，与Ω相比(𝑁 𝑑 + 𝑁 2）标准注意力。对于的典型值𝑑 和𝑀, 与标准注意力相比，FlashAttention需要的HBM访问次数要少得多（最多减少9倍，如图2所示）。此外，我们提供了一个下界，表明在所有SRAM大小上，没有精确的注意力算法可以渐近地提高HBM访问的数量。

We also show that FlashAttention can serve as a useful primitive for realizing the potential of approximate attention algorithms by overcoming their issues with memory access overhead. As a proof of concept, we implement block-sparse FlashAttention, a sparse attention algorithm that is 2-4× faster than even FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than FlashAttention by a factor proportional to the sparsity ratio. We discuss further extensions to other operations (attention on multi-GPU, kernel regression, block-sparse matrix multiply) in Section 5. We open-source FlashAttention to make it easier to build on this primitive.1

我们还表明，FlashAttention可以作为一个有用的原语，通过克服近似注意力算法的内存访问开销问题，来实现它们的潜力。作为概念验证，我们实现了块稀疏FlashAttention，这是一种稀疏注意力算法，比FlashAttention快2-4倍，可扩展到64k的序列长度。我们通过与稀疏率成比例的因子证明了块稀疏FlashAttention比FlashAttention具有更好的IO复杂度。我们在第5节中讨论了对其他操作的进一步扩展（注意多GPU、内核回归、块稀疏矩阵乘法）。我们开源了FlashAttention，以便更容易地在此基础上进行构建。1

We empirically validate that FlashAttention speeds up model training and improves model quality by modeling longer context. We also benchmark the runtime and memory footprint of FlashAttention and block-sparse FlashAttention compared to prior attention implementations. 
* Faster Model Training. FlashAttention trains Transformer models faster in wall-clock time. We train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1 [58], GPT2 (seq. length 1K) 3× faster than baseline implementations from HuggingFace [87] and Megatron-LM [77], and long-range arena (seq. length 1K-4K) 2.4× faster than baselines. 
* Higher Quality Models. FlashAttention scales Transformers to longer sequences, which improves their quality and enables new capabilities. We observe a 0.7 improvement in perplexity on GPT-2 and 6.4 points of lift from modeling longer sequences on long-document classification [13]. FlashAttention enables the first Transformer that can achieve better-than-chance performance on the Path-X [80] challenge, solely from using a longer sequence length (16K). Block-sparse FlashAttention enables a Transformer to scale to even longer sequences (64K), resulting in the first model that can achieve better-than-chance performance on Path-256. 
* Benchmarking Attention. FlashAttention is up to 3× faster than the standard attention implementation across common sequence lengths from 128 to 2K and scales up to 64K. Up to sequence length of 512, FlashAttention is both faster and more memory-efficient than any existing attention method, whereas for sequence length beyond 1K, some approximate attention methods (e.g., Linformer) start to become faster. On the other hand, block-sparse FlashAttention is faster than all existing approximate attention methods that we know of. 

我们实证验证了FlashAttention通过建模更长的上下文来加快模型训练并提高模型质量。我们还对FlashAttention的运行时和内存占用进行了基准测试，并与之前的注意力实现进行了比较。
* 更快的模型训练。FlashAttention在挂钟时间内更快地训练Transformer模型。我们训练BERT大（序列号512）比MLPerf 1.1[58]中的训练速度记录快15%，GPT2（序列号1K）比HuggingFace[87]和Megatron LM[77]的基线实现快3倍，远程竞技场（序列号长度1K-4K）比基线快2.4倍。
* 更高质量的模型。FlashAttention将变形金刚扩展到更长的序列，这提高了它们的质量并启用了新的功能。我们观察到GPT-2的困惑度提高了0.7，对长文档分类的较长序列建模提高了6.4个百分点[13]。FlashAttention启用了第一个Transformer，它可以在Path-X[80]挑战中获得比偶然更好的性能，只需使用更长的序列长度（16K）。块稀疏FlashAttention使Transformer能够扩展到更长的序列（64K），从而产生了第一个可以在Path-256上获得更好性能的模型。
* 基准关注。FlashAttention在128到2K的公共序列长度上比标准注意力实现快3倍，可扩展到64K。当序列长度达到512时，FlashAttention比任何现有的注意力方法都更快，内存效率更高，而当序列长度超过1K时，一些近似的注意力方法（例如Linformer）开始变得更快。另一方面，块稀疏FlashAttention比我们所知道的所有现有的近似注意力方法都快。

## 2 Background
We provide some background on the performance characteristics of common deep learning operations on modern hardware (GPUs). We also describe the standard implementation of attention.

## 2.1 Hardware Performance
We focus here on GPUs. Performance on other hardware accelerators are similar [46, 48].

GPU Memory Hierarchy. The GPU memory hierarchy (Fig. 1 left) comprises multiple forms of memory of different sizes and speeds, with smaller memory being faster. As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [44, 45]. The on-chip

SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size. As compute has gotten faster relative to memory speed [61, 62, 63], operations are increasingly bottlenecked by memory (HBM) accesses. Thus exploiting fast SRAM becomes more important.

Execution Model. GPUs have a massive number of threads to execute an operation (called a kernel).

Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

Performance characteristics. Depending on the balance of computation and memory accesses, operations can be classified as either compute-bound or memory-bound. This is commonly measured by the arithmetic intensity [85], which is the number of arithmetic operations per byte of memory access.

## 1. Compute-bound: the time taken by the operation is determined by how many arithmetic operations there
 are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner dimension, and convolution with large number of channels.

## 2. Memory-bound: the time taken by the operation is determined by the number of memory accesses, while
 time spent in computation is much smaller. Examples include most other operations: elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).

Kernel fusion. The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations [53, 65, 75]. 1FlashAttention code is available at https://github.com/HazyResearch/flash-attention 3

However, in the context of model training, the intermediate values still need to be written to HBM to save for the backward pass, reducing the effectiveness of naive kernel fusion.

## 2.2 Standard Attention Implementation
Given input sequences Q, K, V ∈ R

𝑁 ×𝑑 where 𝑁 is the sequence length and 𝑑 is the head dimension, we want to compute the attention output O ∈ R

𝑁 ×𝑑 :

S = QK> ∈ R

𝑁 ×𝑁 , P = softmax(S) ∈ R

𝑁 ×𝑁 , O = PV ∈ R

𝑁 ×𝑑 , where softmax is applied row-wise.

Standard attention implementations materialize the matrices S and P to HBM, which takes 𝑂(𝑁 2 ) memory.

Often 𝑁  𝑑 (e.g., for GPT2, 𝑁 = 1024 and 𝑑 = 64). We describe the standard attention implementation in Algorithm 0. As some or most of the operations are memory-bound (e.g., softmax), the large number of memory accesses translates to slow wall-clock time.

This problem is exacerbated by other elementwise operations applied to the attention matrix, such as masking applied to S or dropout applied to P. As a result, there have been many attempts to fuse several elementwise operations, such as fusing masking with softmax [77].

In Section 3.2, we will show that the standard attention implementation performs HBM accesses quadratic in the sequence length 𝑁. We also compare the number of FLOPs and number of HBM accesses of standard attention and of our method (FlashAttention).

Algorithm 0 Standard Attention Implementation

Require: Matrices Q, K, V ∈ R

𝑁 ×𝑑 in HBM. 1: Load Q, K by blocks from HBM, compute S = QK> , write S to HBM. 2: Read S from HBM, compute P = softmax(S), write P to HBM. 3: Load P and V by blocks from HBM, compute O = PV, write O to HBM. 4: Return O. 

## 3 FlashAttention: Algorithm, Analysis, and Extensions
We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an attention algorithm that is both memory efficient and faster in wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses compared to standard attention. We further show that FlashAttention can serve as a useful primitive by extending it to handle block-sparse attention.

We focus here on the forward pass for ease of exposition; Appendix B contains details for the backward.

## 3.1 An Efficient Attention Algorithm With Tiling and Recomputation
Given the inputs Q, K, V ∈ R

𝑁 ×𝑑 in HBM, we aim to compute the attention output O ∈ R

𝑁 ×𝑑 and write it to

#### HBM. Our goal is to reduce the amount of HBM accesses (to sub-quadratic in 𝑁).
We apply two established techniques (tiling, recomputation) to overcome the technical challenge of computing exact attention in sub-quadratic HBM accesses. We describe this in Algorithm 1. The main idea is that we split the inputs Q, K, V into blocks, load them from slow HBM to fast SRAM, then compute the attention output with respect to those blocks. By scaling the output of each block by the right normalization factor before adding them up, we get the correct result at the end.

Tiling. We compute attention by blocks. Softmax couples columns of K, so we decompose the large softmax with scaling [51, 60, 66]. For numerical stability, the softmax of vector 𝑥 ∈ R

𝐵 is computed as: 𝑚(𝑥) := max 𝑖 𝑥𝑖 , 𝑓 (𝑥) := 𝑒 𝑥1−𝑚(𝑥) . . . 𝑒𝑥𝐵−𝑚(𝑥)  , ℓ(𝑥) := ∑︁ 𝑖 𝑓 (𝑥)𝑖 , softmax(𝑥) := ℓ 𝑓 ( ( 𝑥 𝑥 ) ) . 4

For vectors 𝑥 (1) , 𝑥(2) ∈ R

𝐵, we can decompose the softmax of the concatenated 𝑥 = 𝑥 (1) 𝑥 (2) ∈ R 2𝐵 as: 𝑚(𝑥) = 𝑚( 𝑥 (1) 𝑥 (2)  ) = max(𝑚(𝑥 (1) ), 𝑚(𝑥 (2) )), 𝑓 (𝑥) = h 𝑒 𝑚(𝑥 (1) )−𝑚(𝑥) 𝑓 (𝑥 (1) ) 𝑒 𝑚(𝑥 (2) )−𝑚(𝑥) 𝑓 (𝑥 (2) ) i , ℓ(𝑥) = ℓ( 𝑥 (1) 𝑥 (2)  ) = 𝑒 𝑚(𝑥 (1) )−𝑚(𝑥) ℓ(𝑥 (1) ) + 𝑒 𝑚(𝑥 (2) )−𝑚(𝑥) ℓ(𝑥 (2) ), softmax(𝑥) = ℓ 𝑓 ( ( 𝑥 𝑥 ) ) .

Therefore if we keep track of some extra statistics (𝑚(𝑥), ℓ(𝑥)), we can compute softmax one block at a time.2

We thus split the inputs Q, K, V into blocks (Algorithm 1 line 3), compute the softmax values along with extra statistics (Algorithm 1 line 10), and combine the results (Algorithm 1 line 12).

Recomputation. One of our goals is to not store 𝑂(𝑁 2 ) intermediate values for the backward pass. The backward pass typically requires the matrices S, P ∈ R

𝑁 ×𝑁 to compute the gradients with respect to Q, K, V.

However, by storing the output O and the softmax normalization statistics (𝑚, ℓ), we can recompute the attention matrix S and P easily in the backward pass from blocks of Q, K, V in SRAM. This can be seen as a form of selective gradient checkpointing [10, 34]. While gradient checkpointing has been suggested to reduce the maximum amount of memory required [66], all implementations (that we know off) have to trade speed for memory. In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses (Fig. 2). The full backward pass description is in Appendix B.

Implementation details: Kernel fusion. Tiling enables us to implement our algorithm in one

CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax, optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout in Appendix B). This avoids repeatedly reading and writing of inputs and outputs from and to HBM.

Algorithm 1 FlashAttention

Require: Matrices Q, K, V ∈ R

𝑁 ×𝑑 in HBM, on-chip SRAM of size 𝑀. 1: Set block sizes 𝐵𝑐 =  4

𝑀 𝑑  , 𝐵𝑟 = min   4

𝑀 𝑑  , 𝑑 . 2: Initialize O = (0)𝑁 ×𝑑 ∈ R

𝑁 ×𝑑 , ℓ = (0)𝑁 ∈ R

𝑁 , 𝑚 = (−∞)𝑁 ∈ R

𝑁 in HBM. 3: Divide Q into 𝑇𝑟 = l 𝐵

𝑁 𝑟 m blocks Q1, . . . , Q𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, and divide K, V in to 𝑇𝑐 = l 𝐵

𝑁 𝑐 m blocks

K1, . . . , K𝑇𝑐 and V1, . . . , V𝑇𝑐 , of size 𝐵𝑐 × 𝑑 each. 4: Divide O into 𝑇𝑟 blocks O𝑖 , . . . , O𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, divide ℓ into 𝑇𝑟 blocks ℓ𝑖 , . . . , ℓ𝑇𝑟 of size 𝐵𝑟 each, divide 𝑚 into 𝑇𝑟 blocks 𝑚1, . . . , 𝑚𝑇𝑟 of size 𝐵𝑟 each. 5: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do 6: Load K𝑗 , V𝑗 from HBM to on-chip SRAM. 7: for 1 ≤ 𝑖 ≤ 𝑇𝑟 do 8: Load Q𝑖 , O𝑖 , ℓ𝑖 , 𝑚𝑖 from HBM to on-chip SRAM. 9: On chip, compute S𝑖 𝑗 = Q𝑖K𝑇 𝑗 ∈ R

𝐵𝑟×𝐵𝑐 . 10: On chip, compute ˜𝑚𝑖 𝑗 = rowmax(S𝑖 𝑗) ∈ R

𝐵𝑟 , ˜P𝑖 𝑗 = exp(S𝑖 𝑗 − ˜𝑚𝑖 𝑗) ∈ R

𝐵𝑟×𝐵𝑐 (pointwise), ℓ˜ 𝑖 𝑗 = rowsum( ˜P𝑖 𝑗) ∈ R

𝐵𝑟 . 11: On chip, compute 𝑚𝑖 new = max(𝑚𝑖 , ˜𝑚𝑖 𝑗) ∈ R

𝐵𝑟 , ℓ 𝑖 new = 𝑒 𝑚𝑖−𝑚𝑖 new ℓ𝑖 + 𝑒 ˜𝑚𝑖 𝑗−𝑚𝑖 new ℓ˜ 𝑖 𝑗 ∈ R

𝐵𝑟 . 12: Write O𝑖 ← diag(ℓ 𝑖 new) −1 (diag(ℓ𝑖)𝑒 𝑚𝑖−𝑚𝑖 new O𝑖 + 𝑒 ˜𝑚𝑖 𝑗−𝑚𝑖 new ˜P𝑖 𝑗V𝑗) to HBM. 13: Write ℓ𝑖 ← ℓ 𝑖 new, 𝑚𝑖 ← 𝑚𝑖 new to HBM. 14: end for 15: end for 16: Return O.

We show FlashAttention’s correctness, runtime, and memory requirement (proof in Appendix C).

Theorem 1. Algorithm 1 returns O = softmax(QK> )V with 𝑂(𝑁 2𝑑) FLOPs and requires 𝑂(𝑁) additional memory beyond inputs and output.

## 3.2 Analysis: IO Complexity of FlashAttention
We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound, proving that no exact attention algorithm can 2This style of aggregation is called algebraic aggregation [33]. 5

Attention Standard FlashAttention

GFLOPs 66.6 75.2

HBM R/W (GB) 40.3 4.4

Runtime (ms) 41.7 7.3

Sparsity Speedup % Non-Zero Blocks 20 60 50 100 150

Eect of Block Size

Block Size 64 128 256 512 6 2

Dense

FlashAttention 2 4 6

Runtime

Figure 2: Left: Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor affecting runtime. Middle: Forward runtime of FlashAttention (seq. length 1024, head dim. 64, 16 heads, batch size 64) on

A100 GPU. Fewer HBM accesses result in faster runtime, up to a point. Right: The runtime (for seq. length 4K) of block-sparse FlashAttention is faster than FlashAttention by a factor proportional to the sparsity. asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in Appendix C.

Theorem 2. Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑 ≤ 𝑀 ≤ 𝑁 𝑑.

Standard attention (Algorithm 0) requires Θ(𝑁 𝑑 + 𝑁 2 ) HBM accesses, while FlashAttention (Algorithm 1) requires Θ(𝑁 2𝑑 2𝑀−1 ) HBM accesses.

For typical values of 𝑑 (64-128) and 𝑀 (around 100KB), 𝑑 2 is many times smaller than 𝑀, and thus

FlashAttention requires many times fewer HBM accesses than standard implementation. This leads to both faster execution and lower memory footprint, which we validate in Section 4.3.

The main idea of the proof is that given the SRAM size of 𝑀, we can load blocks of K, V of size Θ(𝑀) each (Algorithm 1 line 6). For each block of K and V, we iterate over all blocks of Q (Algorithm 1 line 8) to compute the intermediate values, resulting in Θ(𝑁 𝑑𝑀−1 ) passes over Q. Each pass loads Θ(𝑁 𝑑) elements, which amounts to Θ(𝑁 2𝑑 2𝑀−1 ) HBM accesses. We similarly prove that the backward pass of standard attention requires Θ(𝑁 𝑑 + 𝑁 2 ) HBM accesses while the backward pass of FlashAttention requires Θ(𝑁 2𝑑 2𝑀−1 )

HBM accesses (Appendix B).

We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all values of 𝑀 (the SRAM size) when computing exact attention.

Proposition 3. Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑 ≤ 𝑀 ≤ 𝑁 𝑑. There does not exist an algorithm to compute exact attention with 𝑜(𝑁 2𝑑 2𝑀−1 ) HBM accesses for all 𝑀 in the range [𝑑, 𝑁 𝑑].

The proof relies on the fact that for 𝑀 = Θ(𝑁 𝑑) any algorithm must perform Ω(𝑁 2𝑑 2𝑀−1 ) = Ω(𝑁 𝑑)

HBM accesses. This type of lower bound over a subrange of 𝑀 is common in the streaming algorithms literature [88]. We leave proving parameterized complexity [27] lower bounds in terms of 𝑀 as exciting future work.

We validate that the number of HBM accesses is the main determining factor of attention run-time.

In Fig. 2 (left), we see that even though FlashAttention has higher FLOP count compared to standard attention (due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much faster runtime. In Fig. 2 (middle), we vary the block size 𝐵𝑐 of FlashAttention, which results in different amounts of HBM accesses, and measure the runtime of the forward pass. As block size increases, the number of HBM accesses decreases (as we make fewer passes over the input), and runtime decreases. For large enough block size (beyond 256), the runtime is then bottlenecked by other factors (e.g., arithmetic operations).

Moreover, larger block size will not fit into the small SRAM size.

## 3.3 Extension: Block-Sparse FlashAttention
We extend FlashAttention to approximate attention: we propose block-sparse FlashAttention, whose

IO complexity is smaller than FlashAttention by a factor proportional to the sparsity.

Given inputs Q, K, V ∈ R

𝑁 ×𝑑 and a mask matrix ˜M ∈ {0, 1}

𝑁 ×𝑁 , we want to compute:

S = QK> ∈ R

𝑁 ×𝑁 , P = softmax(S  𝟙 ˜M) ∈ R

𝑁 ×𝑁 , O = PV ∈ R

𝑁 ×𝑑 , where (S  𝟙 ˜M)𝑘𝑙 = S𝑘𝑙 if ˜M𝑘𝑙 = 1 and −∞ if M𝑘𝑙 = 0. We require ˜M to have block form: for some block sizes

𝐵𝑟 , 𝐵𝑐, for all 𝑘, 𝑙, ˜M𝑘,𝑙 = M𝑖 𝑗 with 𝑖 = b 𝑘/𝐵𝑟 c , 𝑗 = b 𝑙/𝐵𝑐c for some M ∈ {0, 1}

𝑁 /𝐵𝑟×𝑁 /𝐵𝑐 . 6

Block-Sparse

FlashAttention

Fwd + Bwd (ms)

HBM Accesses (GB)

Fwd Runtime (ms)

HBM

Accesses

Given a predefined block sparsity mask M ∈ {0, 1}

𝑁 /𝐵𝑟×𝑁 /𝐵𝑐 we can easily adapt Algorithm 1 to only compute the nonzero blocks of the attention matrix. The algorithm is identical to Algorithm 1, except we skip zero blocks. We reproduce the algorithm description in Algorithm 5 in Appendix B.

We also analyze the IO complexity of block-sparse FlashAttention.

Proposition 4. Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑 ≤ 𝑀 ≤ 𝑁 𝑑. Block-sparse FlashAttention (Algorithm 5) requires Θ(𝑁 𝑑 + 𝑁 2𝑑 2𝑀−1 𝑠) HBM accesses where 𝑠 is the fraction of nonzero blocks in the block-sparsity mask.

We see that applying block-sparsity yields a direct improvement by the sparsity to the larger term in the

IO complexity. For large sequence lengths 𝑁, 𝑠 is often set to 𝑁 −1/2 [11] or 𝑁 −1 log 𝑁 [3, 17, 92], resulting in Θ(𝑁 √

𝑁) or Θ(𝑁 log 𝑁) IO complexity. For downstream experiments, we use the fixed butterfly sparsity pattern [17], which has been shown to be able to approximate arbitrary sparsity [16].

In Fig. 2 (right), we validate that as the sparsity increases, the runtime of block-sparse FlashAttention improves proportionally. On the LRA benchmark, block-sparse FlashAttention achieves 2.8× speedup, while performing on par with standard attention (Section 4). 

## 4 Experiments
We evaluate the impact of using FlashAttention to train Transformer models. We validate two claims about training time and model accuracy, and report attention runtime and memory benchmarks. 
* Training Speed. FlashAttention outperforms the MLPerf 1.1 [58] speed record for BERT by 15%, and speeds up GPT-2 up to 3× over HuggingFace [87] and 1.8× over Megatron [77] over standard Transformers.

FlashAttention speeds up the long-range arena (LRA) benchmark 2.4×. 
* Quality. FlashAttention scales Transformers to longer sequences, yielding higher quality. FlashAttention trains GPT-2 with context length 4K faster than Megatron trains GPT-2 with context length 1K, while achieving 0.7 better perplexity. Modeling longer sequences yields 6.4 points of lift on two longdocument classification tasks. Finally, FlashAttention yields the first Transformer that can achieve better-than-random performance on the challenging Path-X task (sequence length 16K), and block-sparse

FlashAttention yields the first sequence model that we know of that can achieve better-than-random performance on Path-256 (sequence length 64K). 
* Benchmarking Attention. We measure the runtime and memory performance of FlashAttention and block-sparse FlashAttention based on sequence length. We confirm that the memory footprint of FlashAttention scales linearly with seq. length and is up to 3× faster than standard attention for common seq. lengths (up to 2K). We confirm that runtime of block-sparse FlashAttention scales linearly in seq. length and is faster than all existing approximate attention baselines.

Additional experiment details are in Appendix E.

## 4.1 Faster Models with FlashAttention
#### BERT. FlashAttention yields the fastest single-node BERT training speed that we know of. We train a
BERT-large [22] model with FlashAttention on Wikipedia. Table 1 compares our training time to the implementation from Nvidia that set the training speed record for MLPerf 1.1 [58]. Our implementation is 15% faster.

Table 1: Training time of BERT-large, starting from the same initialization provided by the MLPerf benchmark, to reach the target accuracy of 72.0% on masked language modeling. Averaged over 10 runs on 8×A100 GPUs.

BERT Implementation Training time (minutes)

Nvidia MLPerf 1.1 [58] 20.0 ± 1.5

FlashAttention (ours) 17.4 ± 1.4

GPT-2. FlashAttention yields faster training times for GPT-2 [67] on the large OpenWebtext dataset [32] than the widely used HuggingFace [87] and Megatron-LM [77] implementations. Table 2 shows up to 3× endto-end speedup compared to Huggingface and 1.7× speedup compared to Megatron-LM. FlashAttention 7 achieves the same perplexity as the other two implementations, as we do not change the model definition.

Appendix E includes plots of the validation perplexity throughout training, confirming that FlashAttention is as numerically stable as the baselines and produces the same training / validation curves.

Table 2: GPT-2 small and medium using FlashAttention achieve up to 3× speed up compared to Huggingface implementation and up to 1.7× compared to Megatron-LM. Training time reported on 8×A100s GPUs.

Model implementations OpenWebText (ppl) Training time (speedup)

GPT-2 small - Huggingface [87] 18.2 9.5 days (1.0×)

GPT-2 small - Megatron-LM [77] 18.2 4.7 days (2.0×)

GPT-2 small - FlashAttention 18.2 2.7 days (3.5×)

GPT-2 medium - Huggingface [87] 14.2 21.0 days (1.0×)

GPT-2 medium - Megatron-LM [77] 14.3 11.5 days (1.8×)

GPT-2 medium - FlashAttention 14.3 6.9 days (3.0×)

Long-range Arena. We compare vanilla Transformer (with either standard implementation or FlashAttention) on the long-range arena (LRA [80]) benchmark. We measure accuracy, throughput, and training time of all models. Each task has a different sequence length varying between 1024 and 4096. We follow the implementation and experimental setting in Tay et al. [80]and Xiong et al. [90]. 3 Table 3 shows that FlashAttention achieves up 2.4× speed-up compared to standard attention. Block-sparse FlashAttention is faster than all of the approximate attention methods that we have tested.

Table 3: The performance of standard attention, FlashAttention, block-sparse FlashAttention, and approximate attention baselines on the Long-Range-Arena benchmarks.

Models ListOps Text Retrieval Image Pathfinder Avg Speedup

Transformer 36.0 63.6 81.6 42.3 72.7 59.3 -

FlashAttention 37.6 63.9 81.4 43.5 72.7 59.8 2.4×

Block-sparse FlashAttention 37.0 63.0 81.3 43.6 73.3 59.6 2.8×

Linformer [84] 35.6 55.9 77.7 37.8 67.6 54.9 2.5×

Linear Attention [50] 38.8 63.2 80.7 42.6 72.5 59.6 2.3×

Performer [12] 36.8 63.6 82.2 42.1 69.9 58.9 1.8×

Local Attention [80] 36.1 60.2 76.7 40.6 66.6 56.0 1.7×

Reformer [51] 36.5 63.8 78.5 39.6 69.4 57.6 1.3×

Smyrf [19] 36.1 64.1 79.0 39.6 70.5 57.9 1.7×

## 4.2 Better Models with Longer Sequences
Language Modeling with Long Context. The runtime and memory-efficiency of FlashAttention allow us to increase the context length of GPT-2 by 4× while still running faster than the optimized implementation from Megatron-LM. Table 4 shows that that GPT-2 with FlashAttention and context length 4K is still 30% faster than GPT-2 from Megatron with context length 1K, while achieving 0.7 better perplexity.

Table 4: GPT-2 small with FlashAttention, with 4× larger context length compared to Megatron-LM, is still 30% faster while achieving 0.7 better perplexity. Training time on 8×A100 GPUs is reported.

Model implementations Context length OpenWebText (ppl) Training time (speedup)

GPT-2 small - Megatron-LM 1k 18.2 4.7 days (1.0×)

GPT-2 small - FlashAttention 1k 18.2 2.7 days (1.7×)

GPT-2 small - FlashAttention 2k 17.6 3.0 days (1.6×)

GPT-2 small - FlashAttention 4k 17.5 3.6 days (1.3×)

Long Document Classification. Training Transformers with longer sequences with FlashAttention improves performance on the MIMIC-III [47] and ECtHR [6, 7] datasets. MIMIC-III contains intensive care unit patient discharge summaries, each annotated with multiple labels. ECtHR contains legal cases from the 3LRA accuracy results are known to be highly dependent on the tuning procedure [90]. Our reproduced baselines perform better than as reported in the original comparison [80]. 8

Attention Memory Usage

Sequence Length

Attention Runtime (Fwd Pass + Bwd Pass)

Sequence Length 256 8K 16K 32K 64K 128 256 512 1024 2048 4096 101 102 10 20

FlashAttention

Block-Sparse FlashAttention

PyTorch Attention

Megatron Attention

Linformer Attention

OpenAI Sparse Attention 8192 100

Crossover Points 20x 2x

Figure 3: Left: runtime of forward pass + backward pass. Right: attention memory usage.

European Court of Human Rights, each of which is mapped to articles of the Convention of Human Rights that were allegedly violaged. Both of these datasets contain very long text documents; the average number of tokens in MIMIC is 2,395 tokens, and the longest document contains 14,562 tokens, while the average and longest numbers in ECtHR are 2,197 and 49,392, respectively. We evaluate lift from increasing the sequence length of a pretrained RoBERTa model [56] (we repeat the positional embeddings, as in Beltagy et al. [3]).

Table 5 shows that sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and that length 8K outperforms length 512 by 8.5 points on ECtHR. The discrepancies may be due to subtle distribution shifts: MIMIC-III contains specialized medical text and thus may be more susceptible to a distribution shift in the document length, whereas ECtHR contains general language.

Table 5: Long Document performance (micro 𝐹1) at different sequence lengths using

FlashAttention. 512 1024 2048 4096 8192 16384

MIMIC-III [47] 52.8 50.7 51.7 54.6 56.4 57.1

ECtHR [6] 72.2 74.3 77.1 78.6 80.7 79.2

Table 6: We report the first Transformer model that can achieve non-random performance on Path-X and Path-256.

Model Path-X Path-256

Transformer ✗ ✗

Linformer [84] ✗ ✗

Linear Attention [50] ✗ ✗

Performer [12] ✗ ✗

Local Attention [80] ✗ ✗

Reformer [51] ✗ ✗

SMYRF [19] ✗ ✗

FlashAttention 61.4 ✗

Block-sparse FlashAttention 56.0 63.1

Path-X and Path-256. The Path-X and Path-256 benchmarks are challenging tasks from the long-range arena benchmark designed to test long context. The task is to classify whether two points in a black and white 128×128 (or 256×256) image have a path connecting them, and the images are fed to the transformer one pixel at a time. In prior work, all transformer models have either run out of memory, or only achieved random performance [80]. There has been a search for alternative architectures that can model such long context [37]. We present here the first result of Transformer models being able to solve Path-X and Path-256 (Table 6). We pretrain a transformer on Path-64, and then transfer to Path-X by spatially interpolating the positional embeddings. FlashAttention achieves 61.4 accuracy on Path-X. Additionally, block-sparse

FlashAttention enables the Transformers to scale to sequence length 64K, achieving 63.1 accuracy4 on

Path-256.

## 4.3 Benchmarking Attention
We vary sequence length and measure runtime and memory usage of FlashAttention and block-sparse

FlashAttention against various attention baselines on one A100 GPU with 40 GB HBM, with dropout and a padding mask. We compare against reference implementations for exact attention, approximate attention, and sparse attention. We report a subset of baselines in the main body; Appendix E contains more baselines and full details. 4Path-256 requires longer sequences but has relatively shorter paths than Path-X, so it is easier to obtain a higher accuracy. 9

Runtime (ms)

Memory Footprint (GB)

Runtime. Figure 3 (left) reports the runtime in milliseconds of the forward + backward pass of FlashAttention and block-sparse FlashAttention compared to the baselines in exact, approximate, and sparse attention (exact numbers in Appendix E). Runtime grows quadratically with sequence length, but FlashAttention runs significantly faster than exact attention baselines, up to 3× faster than the PyTorch implementation. The runtimes of many approximate/sparse attention mechanisms grow linearly with sequence length, but FlashAttention still runs faster than approximate and sparse attention for short sequences due to fewer memory accesses. The approximate attention runtimes begin to cross over with

FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention that we know of, across all sequence lengths.

Memory Footprint. Figure 3 (right) shows the memory footprint of FlashAttention and block-sparse

FlashAttention compared to various exact, approximate, and sparse attention baselines. FlashAttention and block-sparse FlashAttention have the same memory footprint, which grows linearly with sequence length. FlashAttention is up to 20× more memory efficient than exact attention baselines, and is more memory-efficient than the approximate attention baselines. All other algorithms except for Linformer run out of memory on an A100 GPU before 64K, and FlashAttention is still 2× more efficient than Linformer. 

## 5 Limitations and Future Directions
We discuss limitations of our approach and future directions. Related work is given in Appendix A.

Compiling to CUDA. Our current approach to building IO-aware implementations of attention requires writing a new CUDA kernel for each new attention implementation. This requires writing the attention algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort.

Implementations may also not be transferrable across GPU architectures. These limitations suggest the need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and compiling to IO-aware implementations in CUDA—similar to efforts such as Halide in image processing [70].

IO-Aware Deep Learning. We believe that the IO-aware approach can extend beyond attention.

Attention is the most memory-intensive computation in Transformers, but every layer in a deep network touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss these potential extensions in Appendix D.

Multi-GPU IO-Aware Methods. Our IO-aware implementation of attention is optimal within constants for computing attention on a single GPU. However, the attention computation may be parallelizable across multiple GPUs [72]. Using multiple GPUs adds an additional layer to IO analysis—accounting for data transfer between GPUs. We hope our work inspires future work in this direction.

## Acknowledgments
Our implementation uses Apex’s FMHA code (https://github.com/NVIDIA/apex/tree/master/apex/ contrib/csrc/fmha) as a starting point. We thank Young-Jun Ko for the in-depth explanation of his FMHA implementation and for his thoughtful answers to our questions about CUDA. We thank Sabri Eyuboglu,

Megan Leszczynski, Laurel Orr, Yuhuai Wu, Beidi Chen, and Xun Huang for their constructive feedback and suggestions on early drafts of the paper. We thank Markus Rabe and Charles Staats for helpful discussion of their attention algorithm.

We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos.

CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); ARL under

No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266 (Unifying Weak

Supervision); ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine

Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba,

TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce,

Total, the HAI-GCP & HAI-Azure Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), Department of Defense (DoD) through the National Defense Science and Engineering Graduate

Fellowship (NDSEG) Program, and members of the Stanford DAWN project: Facebook, Google, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes 10 notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government. Atri Rudra’s research is supported by NSF grant CCF-1763481.

## References
1. Alok Aggarwal and S Vitter, Jeffrey. The input/output complexity of sorting and related problems. Communications of the ACM, 31(9):1116–1127, 1988.
2. Irwan Bello. LambdaNetworks: Modeling long-range interactions without attention. arXiv preprint arXiv:2102.08602, 2021.
3. Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.
4. L Susan Blackford, Antoine Petitet, Roldan Pozo, Karin Remington, R Clint Whaley, James Demmel, Jack Dongarra, Iain Duff, Sven Hammarling, Greg Henry, et al. An updated set of basic linear algebra subprograms (blas). ACM Transactions on Mathematical Software, 28(2):135–151, 2002.
5. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
6. Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. Neural legal judgment prediction in English. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4317–4323, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1424. URL https://www.aclweb.org/anthology/P19-1424.
7. Ilias Chalkidis, Manos Fergadiotis, Dimitrios Tsarapatsanis, Nikolaos Aletras, Ion Androutsopoulos, and Prodromos Malakasiotis. Paragraph-level rationale extraction through regularization: A case study on european court of human rights cases. In Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics, Mexico City, Mexico, 2021. Association for Computational Linguistics.
8. Benjamin Charlier, Jean Feydy, Joan Alexis Glaunès, François-David Collin, and Ghislain Durif. Kernel operations on the gpu, with autodiff, without memory overflows. Journal of Machine Learning Research, 22(74):1–6, 2021. URL http://jmlr.org/papers/v22/20-275.html.
9. Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying sparse and low-rank attention. In Advances in Neural Information Processing Systems (NeurIPS), 2021.
10. Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016.
11. Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.
12. Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In International Conference on Learning Representations (ICLR), 2020.
13. Xiang Dai, Ilias Chalkidis, Sune Darkner, and Desmond Elliott. Revisiting transformer-based models for long document classification. arXiv preprint arXiv:2204.06683, 2022.
14. Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 2978–2988, 2019. 11
15. Tri Dao, Albert Gu, Matthew Eichhorn, Atri Rudra, and Christopher Ré. Learning fast algorithms for linear transforms using butterfly factorizations. In International Conference on Machine Learning (ICML), 2019.
16. Tri Dao, Nimit Sohoni, Albert Gu, Matthew Eichhorn, Amit Blonder, Megan Leszczynski, Atri Rudra, and Christopher Ré. Kaleidoscope: An efficient, learnable representation for all structured linear maps. In International Conference on Learning Representations (ICLR), 2020.
17. Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, Zhao Song, Atri Rudra, and Christopher Ré. Pixelated butterfly: Simple and efficient sparse training for neural network models. In International Conference on Learning Representations (ICLR), 2022.
18. Tri Dao, Beidi Chen, Nimit Sohoni, Arjun Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, and Christopher Ré. Monarch: Expressive structured matrices for efficient and accurate training. In International Conference on Machine Learning (ICML), 2022.
19. Giannis Daras, Nikita Kitaev, Augustus Odena, and Alexandros G Dimakis. Smyrf-efficient attention using asymmetric clustering. Advances in Neural Information Processing Systems, 33:6476–6489, 2020.
20. Christopher De Sa, Albert Gu, Rohan Puttagunta, Christopher Ré, and Atri Rudra. A two-pronged progress in structured dense matrix vector multiplication. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 1060–1079. SIAM, 2018.
21. Peter J Denning. The working set model for program behavior. Communications of the ACM, 11(5): 323–333, 1968.
22. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. 2019.
23. Xin Dong, Shangyu Chen, and Sinno Jialin Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. arXiv preprint arXiv:1705.07565, 2017.
24. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.
25. Y Eidelman and I Gohberg. On a new class of structured matrices. Integral Equations and Operator Theory, 34(3):293–324, 1999.
26. Jean Feydy, Joan Glaunès, Benjamin Charlier, and Michael Bronstein. Fast geometric learning with symbolic matrices. Advances in Neural Information Processing Systems, 33, 2020.
27. Jörg Flum and Martin Grohe. Parameterized Complexity Theory. Springer, 2006.
28. Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations, 2018.
29. Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M Roy, and Michael Carbin. Stabilizing the lottery ticket hypothesis. arXiv preprint arXiv:1903.01611, 2019.
30. Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning, pages 3259–3269. PMLR, 2020.
31. Karan Goel, Albert Gu, Chris Donahue, and Christopher Ré. It’s raw! audio generation with state-space models. In International Conference on Machine Learning (ICML), 2022.
32. Aaron Gokaslan, Vanya Cohen, Pavlick Ellie, and Stefanie Tellex. Openwebtext corpus, 2019. 12
33. Jim Gray, Surajit Chaudhuri, Adam Bosworth, Andrew Layman, Don Reichart, Murali Venkatrao, Frank Pellow, and Hamid Pirahesh. Data cube: A relational aggregation operator generalizing group-by, cross-tab, and sub-totals. Data mining and knowledge discovery, 1(1):29–53, 1997.
34. Andreas Griewank and Andrea Walther. Evaluating derivatives: principles and techniques of algorithmic differentiation. SIAM, 2008.
35. Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré. Hippo: Recurrent memory with optimal polynomial projections. In Advances in neural information processing systems (NeurIPS), 2020.
36. Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré. Combining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in Neural Information Processing Systems, 34, 2021.
37. Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In The International Conference on Learning Representations (ICLR), 2022.
38. Song Han, Jeff Pool, John Tran, and William J Dally. Learning both weights and connections for efficient neural networks. arXiv preprint arXiv:1506.02626, 2015.
39. Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations, 2016.
40. John Hennessy and David Patterson. Memory hierarchy design. Computer Architecture: A Quantitative Approach, pages 390–525, 2003.
41. Sara Hooker. The hardware lottery. arXiv preprint arXiv:2009.06489, 2020.
42. Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc V Le. Transformer quality in linear time. arXiv preprint arXiv:2202.10447, 2022.
43. Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shigang Li, and Torsten Hoefler. Data movement is all you need: A case study on optimizing transformers. Proceedings of Machine Learning and Systems, 3: 711–732, 2021.
44. Zhe Jia and Peter Van Sandt. Dissecting the Ampere GPU architecture via microbenchmarking. GPU Technology Conference, 2021.
45. Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. Dissecting the nvidia Volta GPU architecture via microbenchmarking. arXiv preprint arXiv:1804.06826, 2018.
46. Zhe Jia, Blake Tillman, Marco Maggioni, and Daniele Paolo Scarpazza. Dissecting the graphcore IPU architecture via microbenchmarking. arXiv preprint arXiv:1912.03413, 2019.
47. Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. Scientific data, 3(1):1–9, 2016.
48. Norman P Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, et al. In-datacenter performance analysis of a tensor processing unit. In Proceedings of the 44th annual international symposium on computer architecture, pages 1–12, 2017.
49. Thomas Kailath, Sun-Yuan Kung, and Martin Morf. Displacement ranks of matrices and linear equations. Journal of Mathematical Analysis and Applications, 68(2):395–407, 1979.
50. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156–5165. PMLR, 2020. 13
51. Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In The International Conference on Machine Learning (ICML), 2020.
52. Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite BEDRT for self-supervised learning of language representations. In The International Conference on Learning Representations (ICLR), 2020.
53. Mingzhen Li, Yi Liu, Xiaoyan Liu, Qingxiao Sun, Xin You, Hailong Yang, Zhongzhi Luan, Lin Gan, Guangwen Yang, and Depei Qian. The deep learning compiler: A comprehensive survey. IEEE Transactions on Parallel and Distributed Systems, 32(3):708–727, 2020.
54. Valerii Likhosherstov, Krzysztof Choromanski, Jared Davis, Xingyou Song, and Adrian Weller. Sub-linear memory: How to make performers slim. arXiv preprint arXiv:2012.11346, 2020.
55. Ji Lin, Yongming Rao, Jiwen Lu, and Jie Zhou. Runtime neural pruning. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.
56. Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
57. Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. Luna: Linear unified nested attention. Advances in Neural Information Processing Systems, 34, 2021.
58. Peter Mattson, Christine Cheng, Gregory Diamos, Cody Coleman, Paulius Micikevicius, David Patterson, Hanlin Tang, Gu-Yeon Wei, Peter Bailis, Victor Bittorf, et al. Mlperf training benchmark. Proceedings of Machine Learning and Systems, 2:336–349, 2020.
59. Frank McSherry, Michael Isard, and Derek G Murray. Scalability! but at what {COST}? In 15th Workshop on Hot Topics in Operating Systems (HotOS XV), 2015.
60. Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867, 2018.
61. NVIDIA. Nvidia Tesla V100 GPU architecture, 2017.
62. NVIDIA. Nvidia A100 tensor core GPU architecture, 2020.
63. NVIDIA. Nvidia H100 tensor core GPU architecture, 2022.
64. D Stott Parker. Random butterfly transformations with applications in computational linear algebra. 1995.
65. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems, 32, 2019.
66. Markus N Rabe and Charles Staats. Self-attention does not need 𝑂(𝑛 2 ) memory. arXiv preprint arXiv:2112.05682, 2021.
67. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
68. Jack Rae and Ali Razavi. Do transformers need deep long-range memory? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Online, July 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.acl-main.672.
69. Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. Compressive transformers for long-range sequence modelling. In The International Conference on Learning Representations (ICLR), 2020. 14
70. Jonathan Ragan-Kelley, Connelly Barnes, Andrew Adams, Sylvain Paris, Frédo Durand, and Saman Amarasinghe. Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines. Acm Sigplan Notices, 48(6):519–530, 2013.
71. Raghu Ramakrishnan, Johannes Gehrke, and Johannes Gehrke. Database management systems, volume 3. McGraw-Hill New York, 2003.
72. Benjamin Recht and Christopher Ré. Parallel stochastic gradient algorithms for large-scale matrix completion. Mathematical Programming Computation, 5(2):201–226, 2013.
73. Hongyu Ren, Hanjun Dai, Zihang Dai, Mengjiao Yang, Jure Leskovec, Dale Schuurmans, and Bo Dai. Combiner: Full attention transformer with sparse computation cost. Advances in Neural Information Processing Systems, 34, 2021.
74. Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9: 53–68, 2021.
75. Amit Sabne. XLA: Compiling machine learning for peak performance. 2020.
76. Victor Sanh, Thomas Wolf, and Alexander M Rush. Movement pruning: Adaptive sparsity by fine-tuning. arXiv preprint arXiv:2005.07683, 2020.
77. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.
78. Vikas Sindhwani, Tara Sainath, and Sanjiv Kumar. Structured transforms for small-footprint deep learning. In Advances in Neural Information Processing Systems, pages 3088–3096, 2015.
79. Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, and Armand Joulin. Adaptive attention span in transformers. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2019.
80. Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. In International Conference on Learning Representations, 2020.
81. Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey. arXiv preprint arXiv:2009.06732, 2020.
82. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
83. Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, and Furu Wei. Deepnet: Scaling transformers to 1,000 layers. arXiv preprint arXiv:2203.00555, 2022.
84. Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.
85. Samuel Williams, Andrew Waterman, and David Patterson. Roofline: an insightful visual performance model for multicore architectures. Communications of the ACM, 52(4):65–76, 2009.
86. Michael E Wolf and Monica S Lam. A data locality optimizing algorithm. In Proceedings of the ACM SIGPLAN 1991 conference on Programming language design and implementation, pages 30–44, 1991. 15
87. Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/2020.emnlp-demos.6.
88. David P Woodruff. Optimal space lower bounds for all frequency moments. In SODA, volume 4, pages 167–175. Citeseer, 2004.
89. Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin, and Michael Auli. Pay less attention with lightweight and dynamic convolutions. In The International Conference on Learning Representations (ICLR), 2019.
90. Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nyströmformer: A nystöm-based algorithm for approximating self-attention. In Proceedings of the AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence, volume 35, page 14138, 2021.
91. Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 558–567, 2021.
92. Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33, 2020.
93. Shuangfei Zhai, Walter Talbott, Nitish Srivastava, Chen Huang, Hanlin Goh, Ruixiang Zhang, and Josh Susskind. An attention free transformer. arXiv preprint arXiv:2105.14103, 2021.
94. Chen Zhu, Wei Ping, Chaowei Xiao, Mohammad Shoeybi, Tom Goldstein, Anima Anandkumar, and Bryan Catanzaro. Long-short transformer: Efficient transformers for language and vision. Advances in Neural Information Processing Systems, 34, 2021. 16 

## A Related Work
IO-Aware Runtime Optimization. The broad concept of optimizing for reading and writing to fast/slow memory has a long history in computer science and has been known by many names. We draw the most direct connection to the literature of analyzing I/O complexity in this work [1], but concepts of memory hierarchies are fundamental and has appeared in many forms, from the working set model [21], to data locality [86], to the Roofline model of arithmetic intensity [85], to analyses of scalability [59], to standard textbook treatments of computer architecture [40]. We hope that this work encourages the community to adopt these ideas in more parts of the deep learning stack.

Efficient ML Models with Structured Matrices. Matrix multiply is the core computational bottleneck of most machine learning models. To reduce the computational complexity, there have been numerous approaches to learn over a more efficient set of matrices. These matrices are called structured matrices, which have subquadratic (𝑜(𝑛 2 ) for dimension 𝑛 × 𝑛) number of parameters and runtime. Most common examples of structured matrices are sparse and low-rank matrices, along with fast transforms commonly encountered in signal processing (Fourier, Chebyshev, sine/cosine, orthogonal polynomials). There have been several more general classes of structured matrices proposed in machine learning: Toeplitz-like [78], low-displacement rank [49], quasi-separable [25]). The butterfly pattern we use for our block-sparse attention is motivated by the fact that butterfly matrices [15, 64] and their products have been shown to be able to express any structured matrices with almost optimal runtime and number of parameters [16, 20]. However, even though structured matrices are efficient in theory, they have not seen wide adoption since it is hard to translate their efficiency to wall-clock speedup since dense unconstrained matrix multiply has very optimize implementation, a phenomenon known as the hardware lottery [41]. Extensions of butterfly matrices [17, 18] aimed to make butterfly matrices more hardware-friendly.

Sparse Training. Our block-sparse FlashAttention can be seen as a step towards making sparse model training more efficient. Sparse models have seen success in compressing models for inference (pruning) by sparsifying the weight matrices [23, 38, 39, 55, 76]. For model training, the lottery tickets hypothesis [28, 29, 30] suggests that there are a set of small sub-networks derived from a larger dense network that performs as well as the original dense network. Out block-sparse FlashAttention can also be seen as a fixed lottery ticket in the context of attention: we fix the sparsity pattern to be the butterfly pattern through training, and observe that it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks.

Efficient Transformer. Transformer-based models have become the most widely-used architecture in natural language processing [22] and computer vision [24, 91]. However, one of their computational bottlenecks is that their time and memory scales quadratic in the sequence length. There are numerous approaches to overcome this bottleneck, including approximation with hashing (i.e., sparse) such as Reformer [51] and

Smyrf [19] and with low-rank approximation such as Performer [12, 54]. One can even combine sparse and low-rank approximation for better accuracy (e.g., Longformer [3], BigBird [92], Scatterbrain [9], Long-short transformer [94], Combiner [73]). Other approaches include compressing along the sequence dimension to attend to multiple tokens at once [52, 57, 79, 89]. One can also attend over the states from previous sequences to help lengthen the context (e.g., Transformer-XL [14] and Compressive Transformer [69]). We recommend the survey [81] for more details.

There are several lines of work on developing other modules instead of attention to model longer context.

HiPPO [35] and its extensions, most notably S4 [31, 36, 37] projects the history on a polynomial basis, allowing accurate reconstruction of the history through state-space models. They combine the strengths of

CNNs (efficient training), RNNs (efficient inference), and continuous models (robust to change in sampling rates). LambdaNetworks [2], AFT [93] and FLASH [42] are other attempts at replacing attention in the context of image classification and language modeling.

##　B Algorithm Details
We first derive the forward and backward passes of attention and show that they can be computed in a memory-efficient manner (requiring extra memory linear instead of quadratic in the sequence length). Though they reduce the amount of extra memory required, naively they still incur quadratic HBM accesses, resulting in slower execution speed. We describe the FlashAttention algorithm to implement both the forward 17 and the backward passes on GPUs that reduces HBM accesses, leading to both faster runtime and smaller memory footprint.

### B.1 Memory-efficient forward pass
The main challenge in making attention memory-efficient is the softmax that couples the columns of K (and columns of V). Our approach is to compute the softmax normalization constant separately to decouple the columns. This technique [60] has been used in the literature [51, 66] to show that attention computation does not need quadratic extra memory (though the number of HBM accesses is still quadratic, resulting in slow run-time).

For simplicity, we omit here the max-shifting step during softmax. The full algorithm in Appendix B.3 contains all the steps.

Recall that given input sequences Q, K, V ∈ R

𝑁 ×𝑑 , we want to compute the attention output O ∈ R

𝑁 ×𝑑 :

S = QK> ∈ R

𝑁 ×𝑁 , P = softmax(S) ∈ R

𝑁 ×𝑁 , O = PV ∈ R

𝑁 ×𝑑 .

We have that 𝑆𝑖 𝑗 = 𝑞

𝑇 𝑖 𝑘 𝑗 where 𝑞𝑖 and 𝑘 𝑗 are the 𝑖-th and 𝑗-th columns of Q and K respectively. Define the normalization constants of softmax:

𝐿𝑖 = ∑︁ 𝑗 𝑒 𝑞𝑖

𝑇 𝑘𝑗 . (1)

Let 𝑣 𝑗 be the 𝑗-th column of V, then the 𝑖-th columns of the output is 𝑜𝑖 = 𝑃𝑖:V = ∑︁ 𝑗

𝑃𝑖 𝑗𝑣 𝑗 = ∑︁ 𝑗 𝑒 𝑞

𝑇 𝑖 𝑘𝑗

𝐿𝑖 𝑣 𝑗 . (2)

We see that once 𝐿𝑖 is computed, we can compute 𝑜𝑖 without extra memory by repeatedly summing 𝑒 𝑞𝑖

𝑇 𝑘 𝑗

𝐿𝑖 𝑣 𝑗 . Therefore the forward pass can be computed with 𝑂(𝑛) extra memory:

#### 1. Compute 𝐿𝑖 for all 𝑖 according to Eq. (1), which takes 𝑂(𝑛) extra memory.
#### 2. Compute 𝑜𝑖 for all 𝑖 according to Eq. (2), which takes 𝑂(𝑑) extra memory.
### B.2 Memory-efficient backward pass
We derive the backward pass of attention and show that it can also be computed with linear memory. Rabe and Staats [66] suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly and show how it can be computed in a memory-efficient manner.

Suppose that there is a scalar loss function 𝜙, and let the output gradient be dO ∈ R 𝑛×𝑑 (where dO denotes 𝜕𝜙 𝜕O ). We want to compute the input gradients dQ, dK, dV ∈ R 𝑛×𝑑 (where dQ, dK, dV denote 𝜕𝜙 𝜕Q , 𝜕𝜙 𝜕K , 𝜕𝜙 𝜕V respectively).

The gradient dV is easy to see. Applying reverse-mode autodiff by hand (aka the chain rule), we obtain (in matrix notation) dV = P

𝑇 dO. Thus: 𝑑𝑣 𝑗 = ∑︁ 𝑖

𝑃𝑖 𝑗 𝑑𝑜𝑖 = ∑︁ 𝑖 𝑒 𝑞

𝑇 𝑖 𝑘𝑗

𝐿𝑖 𝑑𝑜𝑖 . (3)

Since we already computed 𝐿𝑖 , 𝑑𝑣 𝑗 can be computed without extra memory by repeated summing.

The gradients dQ and dK are a little more complicated. We go through the gradients dP and dS first.

From Eq. (2), we have that dP = dOV𝑇 , and so: 𝑑𝑃𝑖 𝑗 = 𝑑𝑜𝑇 𝑖 𝑣 𝑗 .

Recall that 𝑃𝑖: = softmax(𝑆𝑖:). Using the fact that the Jacobian of 𝑦 = softmax(𝑥) is diag(𝑦) − 𝑦𝑦𝑇 , we have that 𝑑𝑆𝑖: = (diag(𝑃𝑖:) − 𝑃𝑖:𝑃

𝑇 𝑖: )𝑑𝑃𝑖: = 𝑃𝑖: ◦ 𝑑𝑃𝑖: − (𝑃

𝑇 𝑖: 𝑑𝑃𝑖:)𝑃𝑖: , 18 where ◦ denotes pointwise multiplication.

Define

𝐷𝑖 = 𝑃

𝑇 𝑖: 𝑑𝑃𝑖: = ∑︁ 𝑗 𝑒 𝑞

𝑇 𝑖 𝑘𝑗

𝐿𝑖 𝑑𝑜𝑇 𝑖 𝑣 𝑗 = 𝑑𝑜𝑇 𝑖 ∑︁ 𝑗 𝑒 𝑞𝑖 > 𝑘𝑗

𝐿𝑖 𝑣 𝑗 = 𝑑𝑜𝑇 𝑖 𝑜𝑖 , (4) then 𝑑𝑆𝑖: = 𝑃𝑖: ◦ 𝑑𝑃𝑖: − 𝐷𝑖𝑃𝑖: .

Hence 𝑑𝑆𝑖 𝑗 = 𝑃𝑖 𝑗 𝑑𝑃𝑖 𝑗 − 𝐷𝑖𝑃𝑖 𝑗 = 𝑃𝑖 𝑗(𝑑𝑃𝑖 𝑗 − 𝐷𝑖).

Now we can get the gradients dQ and dK. Recall that 𝑆𝑖 𝑗 = 𝑞

𝑇 𝑖 𝑘 𝑗 , so 𝑑𝑞𝑖 = ∑︁ 𝑗 𝑑𝑆𝑖 𝑗 𝑘 𝑗 = ∑︁ 𝑗

𝑃𝑖 𝑗(𝑑𝑃𝑖 𝑗 − 𝐷𝑖)𝑘 𝑗 = ∑︁ 𝑗 𝑒 𝑞

𝑇 𝑖 𝑘𝑗

𝐿𝑖 (𝑑𝑜𝑇 𝑖 𝑣 𝑗 − 𝐷𝑖)𝑘 𝑗 . (5)

Similarly, 𝑑𝑘 𝑗 = ∑︁ 𝑖 𝑑𝑆𝑖 𝑗𝑞𝑖 = ∑︁ 𝑖

𝑃𝑖 𝑗(𝑑𝑃𝑖 𝑗 − 𝐷𝑖)𝑞𝑖 = ∑︁ 𝑖 𝑒 𝑞

𝑇 𝑖 𝑘𝑗

𝐿𝑖 (𝑑𝑜𝑇 𝑖 𝑣 𝑗 − 𝐷𝑖)𝑞𝑖 . (6)

Therefore the backward pass can also be computed with 𝑂(𝑛) extra memory:

#### 1. Compute 𝑑𝑣 𝑗 for all 𝑗 according to Eq. (3), which takes 𝑂(𝑑) extra memory.
#### 2. Compute 𝐷𝑖 for all 𝑖 according to Eq. (4), which takes 𝑂(𝑛) extra memory.
#### 3. Compute 𝑑𝑞𝑖 for all 𝑖 according to Eq. (5), which takes 𝑂(𝑑) extra memory.
#### 4. Compute 𝑑𝑘 𝑗 for all 𝑗 according to Eq. (6), which takes 𝑂(𝑑) extra memory.
### B.3 FlashAttention: Forward Pass
We describe the full details of FlashAttention forward pass. Given input sequences Q, K, V ∈ R

𝑁 ×𝑑 , we want to compute the attention output O ∈ R

𝑁 ×𝑑 :

S = 𝜏QK> ∈ R

𝑁 ×𝑁 , S masked = mask(𝑆) ∈ R

𝑁 ×𝑁 , P = softmax(S masked) ∈ R

𝑁 ×𝑁 ,

P dropped = dropout(P, 𝑝drop), O = P droppedV ∈ R

𝑁 ×𝑑 , where 𝜏 ∈ R is some softmax scaling (typically √ 1 𝑑 ), mask is some masking function that sets some entries of the input to −∞ and keep other entries the same (e.g., key padding mask when sequences in the batch don’t have the same lengths and are padded), and dropout(𝑥, 𝑝) applies dropout to 𝑥 elementwise (i.e., output 𝑥 1−𝑝 with probability 1 − 𝑝 and output 0 with probability 𝑝 for each element 𝑥).

The full algorithm is in Algorithm 2. We save the output O, the softmax statistics ℓ and 𝑚, and the pseudo-random number generator state R for the backward pass. 19

Algorithm 2 FlashAttention Forward Pass

Require: Matrices Q, K, V ∈ R

𝑁 ×𝑑 in HBM, on-chip SRAM of size 𝑀, softmax scaling constant 𝜏 ∈ R, masking function mask, dropout probability 𝑝drop. 1: Initialize the pseudo-random number generator state R and save to HBM. 2: Set block sizes 𝐵𝑐 =  4

𝑀 𝑑  , 𝐵𝑟 = min   4

𝑀 𝑑  , 𝑑 . 3: Initialize O = (0)𝑁 ×𝑑 ∈ R

𝑁 ×𝑑 , ℓ = (0)𝑁 ∈ R

𝑁 , 𝑚 = (−∞)𝑁 ∈ R

𝑁 in HBM. 4: Divide Q into 𝑇𝑟 = l 𝐵

𝑁 𝑟 m blocks Q1, . . . , Q𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, and divide K, V in to 𝑇𝑐 = l 𝐵

𝑁 𝑐 m blocks

K1, . . . , K𝑇𝑐 and V1, . . . , V𝑇𝑐 , of size 𝐵𝑐 × 𝑑 each. 5: Divide O into 𝑇𝑟 blocks O𝑖 , . . . , O𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, divide ℓ into 𝑇𝑟 blocks ℓ𝑖 , . . . , ℓ𝑇𝑟 of size 𝐵𝑟 each, divide 𝑚 into 𝑇𝑟 blocks 𝑚1, . . . , 𝑚𝑇𝑟 of size 𝐵𝑟 each. 6: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do 7: Load K𝑗 , V𝑗 from HBM to on-chip SRAM. 8: for 1 ≤ 𝑖 ≤ 𝑇𝑟 do 9: Load Q𝑖 , O𝑖 , ℓ𝑖 , 𝑚𝑖 from HBM to on-chip SRAM. 10: On chip, compute S𝑖 𝑗 = 𝜏Q𝑖K𝑇 𝑗 ∈ R

𝐵𝑟×𝐵𝑐 . 11: On chip, compute S masked 𝑖 𝑗 = mask(S𝑖 𝑗). 12: On chip, compute ˜𝑚𝑖 𝑗 = rowmax(S masked 𝑖 𝑗 ) ∈ R

𝐵𝑟 , ˜P𝑖 𝑗 = exp(S masked 𝑖 𝑗 − ˜𝑚𝑖 𝑗) ∈ R

𝐵𝑟×𝐵𝑐 (pointwise), ℓ˜ 𝑖 𝑗 = rowsum( ˜P𝑖 𝑗) ∈ R

𝐵𝑟 . 13: On chip, compute 𝑚 new 𝑖 = max(𝑚𝑖 , ˜𝑚𝑖 𝑗) ∈ R

𝐵𝑟 , ℓ new 𝑖 = 𝑒 𝑚𝑖−𝑚𝑖 new ℓ𝑖 + 𝑒 ˜𝑚𝑖 𝑗−𝑚𝑖 new ℓ˜ 𝑖 𝑗 ∈ R

𝐵𝑟 . 14: On chip, compute ˜P dropped 𝑖 𝑗 = dropout( ˜P𝑖 𝑗, 𝑝drop). 15: Write O𝑖 ← diag(ℓ 𝑖 new) −1 (diag(ℓ𝑖)𝑒 𝑚𝑖−𝑚𝑖 new O𝑖 + 𝑒 ˜𝑚𝑖 𝑗−𝑚new 𝑖 ˜P dropped 𝑖 𝑗 V𝑗) to HBM. 16: Write ℓ𝑖 ← ℓ 𝑖 new, 𝑚𝑖 ← 𝑚𝑖 new to HBM. 17: end for 18: end for 19: Return O, ℓ, 𝑚, R.

### B.4 FlashAttention: Backward Pass
We describe the full details of FlashAttention backward pass. Given input sequences Q, K, V ∈ R

𝑁 ×𝑑 , the output O ∈ R

𝑁 ×𝑑 , and the output gradient dO, we want to compute the input gradients dQ, dK, dV ∈ R

𝑁 ×𝑑 .

We first describe the standard attention backward pass in Algorithm 3 for completeness.

Algorithm 3 Standard Attention Backward Pass

Require: Matrices Q, K, V, dO ∈ R

𝑁 ×𝑑 , P ∈ R

𝑁 ×𝑁 in HBM. 1: Load P, dO by blocks from HBM, compute dV = P > dO ∈ R

𝑁 ×𝑑 , write dV to HBM. 2: Load dO, V by blocks from HBM, compute dP = dOV> ∈ R

𝑁 ×𝑁 , write dP to HBM. 3: Read P, dP from HBM, compute dS ∈ R

𝑁 ×𝑁 where 𝑑𝑆𝑖 𝑗 = 𝑃𝑖 𝑗(𝑑𝑃𝑖 𝑗 −

Í 𝑙 𝑃𝑖𝑙𝑑𝑃𝑖𝑙), write dS to HBM. 4: Load dS and K by blocks from HBM, compute dQ = dSK, write dQ to HBM. 5: Load dS and Q by blocks from HBM, compute dK = dS> Q, write dK to HBM. 6: Return dQ, dK, dV.

We now make two observations about FlashAttention backward pass:

## 1. We do not need to store the dropout mask of size 𝑂(𝑁
 2 ) from the forward pass. Instead, we can save the pseudo-random number generator states from the forward pass and re-generate the dropout mask in the backward pass. This allows us to only use 𝑂(𝑁) extra memory.

### 2. When computing the softmax gradient, we use Eq. (4) to compute 𝐷𝑖 = 𝑃𝑖
 >: 𝑑𝑃𝑖: without reducing over

𝑃𝑖: and 𝑑𝑃𝑖: of size 𝑁 (they might not fit into SRAM). Instead we can rewrite 𝐷𝑖 = 𝑑𝑜>𝑖 𝑜𝑖 and compute the dot product between vectors of size 𝑑. 20

The full FlashAttention backward pass algorithm is in Algorithm 4. Conceptually it is just a block version of the derivation in Appendix B.2.

Algorithm 4 FlashAttention Backward Pass

Require: Matrices Q, K, V, O, dO ∈ R

𝑁 ×𝑑 in HBM, vectors ℓ, 𝑚 ∈ R

𝑁 in HBM, on-chip SRAM of size 𝑀, softmax scaling constant 𝜏 ∈ R, masking function mask, dropout probability 𝑝drop, pseudo-random number generator state R from the forward pass. 1: Set the pseudo-random number generator state to R. 2: Set block sizes 𝐵𝑐 =  4

𝑀 𝑑  , 𝐵𝑟 = min   4

𝑀 𝑑  , 𝑑 . 3: Divide Q into 𝑇𝑟 = l 𝐵

𝑁 𝑟 m blocks Q1, . . . , Q𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, and divide K, V in to 𝑇𝑐 = l 𝐵

𝑁 𝑐 m blocks

K1, . . . , K𝑇𝑐 and V1, . . . , V𝑇𝑐 , of size 𝐵𝑐 × 𝑑 each. 4: Divide O into 𝑇𝑟 blocks O𝑖 , . . . , O𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, divide dO into 𝑇𝑟 blocks dO𝑖 , . . . , dO𝑇𝑟 of size

𝐵𝑟 × 𝑑 each, divide ℓ into 𝑇𝑟 blocks ℓ𝑖 , . . . , ℓ𝑇𝑟 of size 𝐵𝑟 each, divide 𝑚 into 𝑇𝑟 blocks 𝑚1, . . . , 𝑚𝑇𝑟 of size

𝐵𝑟 each. 5: Initialize dQ = (0)𝑁 ×𝑑 in HBM and divide it into 𝑇𝑟 blocks dQ1 , . . . , dQ𝑇𝑟 of size 𝐵𝑟 × 𝑑 each. Initialize dK = (0)𝑁 ×𝑑, dV = (0)𝑁 ×𝑑 in HBM and divide dK, dV in to 𝑇𝑐 blocks dK1, . . . , dK𝑇𝑐 and dV1, . . . , dV𝑇𝑐 , of size 𝐵𝑐 × 𝑑 each. 6: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do 7: Load K𝑗 , V𝑗 from HBM to on-chip SRAM. 8: Initialize ˜dK𝑗 = (0)𝐵𝑐×𝑑, ˜dV𝑗 = (0)𝐵𝑐×𝑑 on SRAM. 9: for 1 ≤ 𝑖 ≤ 𝑇𝑟 do 10: Load Q𝑖 , O𝑖 , dO𝑖 , dQ𝑖 , ℓ𝑖 , 𝑚𝑖 from HBM to on-chip SRAM. 11: On chip, compute S𝑖 𝑗 = 𝜏Q𝑖K𝑇 𝑗 ∈ R

𝐵𝑟×𝐵𝑐 . 12: On chip, compute S masked 𝑖 𝑗 = mask(S𝑖 𝑗). 13: On chip, compute P𝑖 𝑗 = diag(𝑙𝑖) −1 exp(S masked 𝑖 𝑗 − 𝑚𝑖) ∈ R

𝐵𝑟×𝐵𝑐 . 14: On chip, compute dropout mask Z𝑖 𝑗 ∈ R

𝐵𝑟×𝐵𝑐 where each entry has value 1−𝑝 1 drop with probability 1 − 𝑝drop and value 0 with probability 𝑝drop. 15: On chip, compute P dropped 𝑖 𝑗 = P𝑖 𝑗 ◦ Z𝑖 𝑗 (pointwise multiply). 16: On chip, compute ˜dV𝑗 ← ˜dV𝑗 + (P dropped 𝑖 𝑗 ) > dO𝑖 ∈ R

𝐵𝑐×𝑑 . 17: On chip, compute dPdropped 𝑖 𝑗 = dO𝑖V>𝑗 ∈ R

𝐵𝑟×𝐵𝑐 . 18: On chip, compute dP𝑖 𝑗 = dPdropped 𝑖 𝑗 ◦ Z𝑖 𝑗 (pointwise multiply). 19: On chip, compute 𝐷𝑖 = rowsum(dO𝑖 ◦ O𝑖) ∈ R

𝐵𝑟 . 20: On chip, compute dS𝑖 𝑗 = P𝑖 𝑗 ◦ (dP𝑖 𝑗 − 𝐷𝑖) ∈ R

𝐵𝑟×𝐵𝑐 . 21: Write dQ𝑖 ← dQ𝑖 + 𝜏dS𝑖 𝑗K𝑗 ∈ R

𝐵𝑟×𝑑 to HBM. 22: On chip, compute ˜dK𝑗 ← ˜dK𝑗 + 𝜏dS>𝑖 𝑗Q𝑖 ∈ R

𝐵𝑐×𝑑 . 23: end for 24: Write dK𝑗 ← ˜dK𝑗 , dV𝑗 ← ˜dV𝑗 to HBM. 25: end for 26: Return dQ, dK, dV.

𝑂

We see that similar to the forward pass, the backward pass performs 𝑂(𝑁 2 ) FLOPs and only requires (𝑁) extra memory beyond inputs, output, output gradient, and input gradients.

We analyze the IO-complexity of the backward pass, similar to the forward pass (Theorem 2).

Theorem 5. Let 𝑁 be the sequence length, 𝑑 be the head dimension, and 𝑀 be size of SRAM with 𝑑 ≤ 𝑀 ≤ 𝑁 𝑑.

Standard attention (Algorithm 0) backward pass requires Θ(𝑁 𝑑 + 𝑁 2 ) HBM accesses, while FlashAttention backward pass (Algorithm 4) requires Θ(𝑁 2𝑑 2𝑀−1 ) HBM accesses.

The proof is in Appendix C. 21

### B.5 Comparison with Rabe and Staats [66]
We describe here some similarities and differences between our FlashAttention algorithm and the algorithm of Rabe and Staats [66].

Conceptually, both FlashAttention and Rabe and Staats [66] operate on blocks of the attention matrix using the well-established technique of tiling (or softmax scaling) [51, 60]. To reduce the memory footprint, both methods avoid storing the large attention matrix in the forward pass and recompute it in the backward pass.

The first major difference is that Rabe and Staats [66] focuses on the reducing the total memory footprint (maximum amount of GPU memory required) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in Section 2, the amount of memory access is the primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amount of memory required (e.g., if an operation incurs 𝐴 memory accesses, then its total memory requirement is at most 𝐴). As a result, FlashAttention is faster than standard attention (2-4×) while Rabe and Staats [66] is around the same speed or slightly slower than standard attention. In terms of total memory required, both methods offer substantial memory saving.

The second difference between the two methods is the way information is summarized from each block to pass to the next block. Rabe and Staats [66] summarizes each block with its temporary output along with the softmax normalization statistics. At the end of the forward pass, the temporary outputs of all the blocks are combined using the statistics to produce the final output. FlashAttention instead incrementally updates the output (Algorithm 1 line 12) after processing each block, so only one copy of the output is needed (instead of 𝐾 copies for 𝐾 blocks). This means that FlashAttention has smaller total memory requirement compared to Rabe and Staats [66].

The final major difference is the way the backward pass is computed. Rabe and Staats [66] uses gradient checkpointing to recompute the attention matrix and the temporary output of each block. FlashAttention instead simplifies the backward pass analytically (Appendices B.2 and B.4). It only recomputes the attention matrix and does not recompute the temporary output of each block. This reduces the memory requirement for the backward pass and yields speedup.

## C Proofs

Proof of Theorem 1. We first count the number of FLOPs and extra memory required.

The dominating FLOPs are from matrix multiplication. In the inner loop, (Algorithm 1 line 9), we compute Q𝑖K>𝑗 ∈ R

𝐵𝑟×𝐵𝑐 for Q𝑖 ∈ R

𝐵𝑟×𝑑 and K𝑗 ∈ R

𝐵𝑐×𝑑 , which takes 𝑂(𝐵𝑟 𝐵𝑐 𝑑) FLOPs. We also compute (Algorithm 1 line 12) ˜P𝑖 𝑗V𝑗 ∈ R

𝐵𝑟×𝑑 for ˜P𝑖 𝑗 ∈ R

𝐵𝑟×𝐵𝑐 and V𝑗 ∈ R

𝐵𝑐×𝑑 , which takes 𝑂(𝐵𝑟 𝐵𝑐 𝑑) FLOPs. We execute the inner loops 𝑇𝑐𝑇𝑟 = l

𝑁

𝐵𝑐 m l

𝑁

𝐵𝑟 m times. Therefore the total number of FLOPs is

𝑂 

𝑁 2

𝐵𝑐𝐵𝑟

𝐵𝑟 𝐵𝑐 𝑑  = 𝑂(𝑁 2 𝑑).

In terms of extra memory required, we see that we need 𝑂(𝑁) memory to store the statistics (ℓ, 𝑚).

We now prove the algorithm’s correctness by induction on 𝑗 for 0 ≤ 𝑗 ≤ 𝑇𝑐. Let K:𝑗 ∈ R 𝑗𝐵𝑐×𝑑 be the first 𝑗 𝐵𝑐 rows of K, and similarly V:𝑗 ∈ R 𝑗𝐵𝑐×𝑑 the the first 𝑗 𝐵𝑐 rows of V. Let S:,:𝑗 = QK>:𝑗 ∈ R

𝑁 ×𝑗𝐵𝑐 , and

P:,:𝑗 = softmax(S:,:𝑗) ∈ R

𝑁 ×𝑗𝐵𝑐 (softmax applied row-wise). Let 𝑚 𝑗 , ℓ( 𝑗) , O( 𝑗) be the values of 𝑚, ℓ, O in HBM after the 𝑗-th iteration of the outer loop (Algorithm 1 line 5). (Note that these values of 𝑚, ℓ, O are updated after each iteration of the outer loop.) We want to show that after the 𝑗-th iteration of the outer loop, we have computed in HBM: 𝑚 ( 𝑗) = rowmax(S:,:𝑗) ∈ R

𝑁 , ℓ( 𝑗) = rowsum(exp(S:,:𝑗 − 𝑚 ( 𝑗) )) ∈ R

𝑁 , O ( 𝑗) = P:,:𝑗V:𝑗 ∈ R

𝑁 ×𝑑 .

Based on our initialization (Algorithm 1 line 2), this claim is true for 𝑗 = 0 (i.e., before the any iteration of the outer loop is executed). Suppose that the claim holds for some 𝑗 = 0, . . . , 𝑇𝑐 − 1. We want to show that the claim also holds for 𝑗 + 1. Indeed, when we update the statistics in the inner loop (Algorithm 1 line 10) 22 on the ( 𝑗 + 1)-th iteration of the outer loop, we update 𝑚 ( 𝑗+1) = max(𝑚 ( 𝑗) , ˜𝑚) where ˜𝑚 ∈ R

𝑁 is the row-max of S:, 𝑗:𝑗+1, the slice of S from column 𝑗 𝐵𝑐 to column ( 𝑗 + 1)𝐵𝑐 − 1. This implies that 𝑚 ( 𝑗+1) = rowmax(S:,:𝑗+1) ∈ R

𝑁 .

Similarly, we update ℓ ( 𝑗+1) = 𝑒 𝑚( 𝑗)−𝑚( 𝑗+1) ℓ ( 𝑗) + 𝑒 ˜𝑚−𝑚( 𝑗+1) ˜ℓ, where ℓ˜= rowsum(exp(S:, 𝑗:𝑗+1 − ˜𝑚)) ∈ R

𝑁 . By the same algebraic manipulation in Section 3.1, we obtain: ℓ ( 𝑗+1) = rowsum(exp(S:,:𝑗+1 − 𝑚 ( 𝑗+1) )) ∈ R

𝑁 .

Let V𝑗:𝑗+1 be the slice of V from column 𝑗 𝐵𝑐 to column ( 𝑗 + 1)𝐵𝑐 − 1, we also update:

O ( 𝑗+1) = diag(ℓ ( 𝑗+1) ) −1 (diag(ℓ ( 𝑗) )𝑒 𝑚( 𝑗)−𝑚( 𝑗+1)

O ( 𝑗) + 𝑒 ˜𝑚−𝑚( 𝑗+1) exp(S𝑗:𝑗+1 − ˜𝑚)V𝑗:𝑗+1) = diag(ℓ ( 𝑗+1) ) −1 (diag(ℓ ( 𝑗) )𝑒 𝑚( 𝑗)−𝑚( 𝑗+1)

P:,:𝑗V:𝑗 + 𝑒 −𝑚( 𝑗+1) exp(S𝑗:𝑗+1)V𝑗:𝑗+1) = diag(ℓ ( 𝑗+1) ) −1 (diag(ℓ ( 𝑗) )𝑒 𝑚( 𝑗)−𝑚( 𝑗+1) diag(ℓ ( 𝑗) ) exp(S:,:𝑗 − 𝑚 ( 𝑗) )V:𝑗 + 𝑒 −𝑚( 𝑗+1) exp(S𝑗:𝑗+1)V𝑗:𝑗+1) = diag(ℓ ( 𝑗+1) ) −1 (𝑒 −𝑚( 𝑗+1) exp(S:,:𝑗)V:𝑗 + 𝑒 −𝑚( 𝑗+1) exp(S𝑗:𝑗+1)V𝑗:𝑗+1) = diag(ℓ ( 𝑗+1) ) −1 (exp(S:,:𝑗 − 𝑚 ( 𝑗+1) )V:𝑗 + exp(S𝑗:𝑗+1 − 𝑚 ( 𝑗+1) )V𝑗:𝑗+1) = diag(ℓ ( 𝑗+1) ) −1  exp   S:,:𝑗 S𝑗:𝑗+1 − 𝑚 ( 𝑗+1)    V:𝑗

V𝑗:𝑗+1  = softmax(S:𝑗+1)V:𝑗+1.

We then see that the claim is also true for 𝑗 + 1. By induction, the claim is true for all 𝑗 = 0, . . . , 𝑇𝑐.

When 𝑗 = 𝑇𝑐, we conclude that the final value of O in HBM is softmax(S)V = softmax(QK> )V. 

Proof of Theorem 2. We first analyze the IO complexity of standard attention implementation. The inputs

Q, K, V ∈ R

𝑁 ×𝑑 reside in HBM, and the at the end of the algorithm the output O ∈ R

𝑁 ×𝑑 is written to HBM.

In the first step of computing the matrix multiply S = QK> , the inputs Q, K are read from HBM and the output S ∈ R

𝑁 ×𝑁 is written to HBM (Algorithm 0 line 1). This incurs Θ(𝑁 𝑑 + 𝑁 2 ) HBM accesses.

In the second step of computing P = softmax(S), the input S is read from HBM and the output P is written to HBM (Algorithm 0 line 2). This incurs Θ(𝑁 2 ) HBM accesses.

In the last step of computing O = PV, the inputs P, V are read from global memory and the output O is written to HBM (Algorithm 0 line 3). This incurs Θ(𝑁 𝑑 + 𝑁 2 ) HBM accesses.

Overall, standard attention implementation requires Θ(𝑁 𝑑 + 𝑁 2 ) global memory accesses.

We now analyze the IO complexity of streaming attention.

Following Algorithm 1, we see that each element of K and V is loaded from HBM once (Algorithm 1 line 6). We make 𝑇𝑐 passes over Q and O, each pass loading all of Q and all of O to HBM (Algorithm 1 line 8). Therefore the number of HBM accesses is Θ (𝑁 𝑑 + 𝑁 𝑑𝑇𝑐) = Θ(𝑁 𝑑𝑇𝑐).

We derive the conditions on the block sizes 𝐵𝑐 and 𝐵𝑟 . We need the blocks K𝑗 and V𝑗 of size 𝐵𝑐 × 𝑑 to fit into on-chip memory, which translates to:

𝐵𝑐 𝑑 = 𝑂(𝑀) ⇔ 𝐵𝑐 = 𝑂 

𝑀 𝑑  .

Similarly, we need the blocks Q𝑖 , O𝑖 of size 𝐵𝑟 × 𝑑 to fit into on-chip memory, which translates to:

𝐵𝑟 𝑑 = 𝑂(𝑀) ⇔ 𝐵𝑟 = 𝑂 

𝑀 𝑑  .

Finally, we need the block S𝑖 𝑗 of size 𝐵𝑟 × 𝐵𝑐 to fit into on-chip memory, which translates to:

𝐵𝑟 𝐵𝑐 = 𝑂(𝑀). 23

We therefore set:

𝐵𝑐 = Θ 

𝑀 𝑑  , 𝐵𝑟 = Θ  min 

𝑀 𝑑 ,

𝑀

𝐵𝑐   = Θ  min 

𝑀 𝑑 , 𝑑  .

We then have:

𝑇𝑐 =

𝑁

𝐵𝑐 = Θ 

𝑁 𝑑

𝑀  .

As a result, the number of HBM accesses is:

Θ (𝑁 𝑑𝑇𝑐) = Θ 

𝑁 2𝑑 2

𝑀  . 

Proof of Proposition 3. For contradiction, suppose that there exists an algorithm that computes exact attention where the number for HBM access for all 𝑀 ∈ [𝑑, 𝑁 𝑑] is 𝑜 

𝑁 2𝑑 2

𝑀  .

In the regime of 𝑀 = Θ(𝑁 𝑑), this results in the number of HBM accesses: 𝑜 

𝑁 2𝑑 2

𝑁 𝑑  = 𝑜(𝑁 𝑑).

However, the input to attention (matrices Q, K, V) and the output O have size 𝑁 𝑑 and they start out being in HBM, so if the algorithm computes exact attention it must incur at least Ω(𝑁 𝑑) HBM accesses. This is a contradiction.

Proof of Theorem 5. The IO complexity of the attention backward is very similar to the IO complexity of the attention forward (Theorem 2). Here we provide a sketch of the proof.

We first analyze the IO complexity of standard attention backward pass. The inputs Q, K, V, dO ∈ R

𝑁 ×𝑑 reside in HBM, and the at the end of the algorithm the outputs dQ, dK, dV ∈ R

𝑁 ×𝑑 are written to HBM.

At each step of the standard attention backward pass, one needs to load inputs of size 𝑁 𝑑 or 𝑁 2 from

HBM, and needs to write the outputs of size 𝑁 2 or 𝑁 𝑑 to HBM. This incurs Θ(𝑁 𝑑 + 𝑁 2 ) HBM accesses.

We now analyze the IO complexity of FlashAttention backward pass.

Similar to Theorem 2, we see that each element of K and V is loaded from HBM once. Each element of dK and dV is only written to HBM once. We make 𝑇𝑐 passes over Q, O, dO, each pass loading all of Q, O, dO to HBM. We also make 𝑇𝑐 passes over dQ, each pass reading/writing all of dQ from/to HBM. Therefore the number of HBM accesses is Θ (𝑁 𝑑 + 𝑁 𝑑𝑇𝑐) = Θ(𝑁 𝑑𝑇𝑐).

As in the proof of Theorem 2, the constraints on the block sizes are that:

𝐵𝑐 = Θ 

𝑀 𝑑  , 𝐵𝑟 = Θ  min 

𝑀 𝑑 , 𝑑  .

We then have:

𝑇𝑐 =

𝑁

𝐵𝑐 = Θ 

𝑁 𝑑

𝑀  .

As a result, the number of HBM accesses is:

Θ (𝑁 𝑑𝑇𝑐) = Θ 

𝑁 2𝑑 2

𝑀  .  24

Algorithm 5 Block-Sparse FlashAttention Forward Pass

Require: Matrices Q, K, V ∈ R

𝑁 ×𝑑 in HBM, on-chip SRAM of size 𝑀, softmax scaling constant 𝜏 ∈ R, masking function mask, dropout probability 𝑝drop, block sizes 𝐵𝑐 =  4

𝑀 𝑑  , 𝐵𝑟 = min   4

𝑀 𝑑  , 𝑑 , block sparsity mask 𝑀 ∈ {0, 1}

𝑁 /𝐵𝑟×𝑁 /𝐵𝑐 .. 1: Initialize the pseudo-random number generator state R and save to HBM. 2: Initialize O = (0)𝑁 ×𝑑 ∈ R

𝑁 ×𝑑 , ℓ = (0)𝑁 ∈ R

𝑁 , 𝑚 = (−∞)𝑁 ∈ R

𝑁 in HBM. 3: Divide Q into 𝑇𝑟 = l 𝐵

𝑁 𝑟 m blocks Q1, . . . , Q𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, and divide K, V in to 𝑇𝑐 = l 𝐵

𝑁 𝑐 m blocks

K1, . . . , K𝑇𝑐 and V1, . . . , V𝑇𝑐 , of size 𝐵𝑐 × 𝑑 each. 4: Divide O into 𝑇𝑟 blocks O𝑖 , . . . , O𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, divide ℓ into 𝑇𝑟 blocks ℓ𝑖 , . . . , ℓ𝑇𝑟 of size 𝐵𝑟 each, divide 𝑚 into 𝑇𝑟 blocks 𝑚1, . . . , 𝑚𝑇𝑟 of size 𝐵𝑟 each. 5: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do 6: Load K𝑗 , V𝑗 from HBM to on-chip SRAM. 7: for 1 ≤ 𝑖 ≤ 𝑇𝑟 do 8: if 𝑀𝑖 𝑗 ≠ 0 then 9: Load Q𝑖 , O𝑖 , ℓ𝑖 , 𝑚𝑖 from HBM to on-chip SRAM. 10: On chip, compute S𝑖 𝑗 = 𝜏Q𝑖K𝑇 𝑗 ∈ R

𝐵𝑟×𝐵𝑐 . 11: On chip, compute S masked 𝑖 𝑗 = mask(S𝑖 𝑗). 12: On chip, compute ˜𝑚𝑖 𝑗 = rowmax(S masked 𝑖 𝑗 ) ∈ R

𝐵𝑟 , ˜P𝑖 𝑗 = exp(S masked 𝑖 𝑗 − ˜𝑚𝑖 𝑗) ∈ R

𝐵𝑟×𝐵𝑐 (pointwise), ℓ˜ 𝑖 𝑗 = rowsum( ˜P𝑖 𝑗) ∈ R

𝐵𝑟 . 13: On chip, compute 𝑚 new 𝑖 = max(𝑚𝑖 , ˜𝑚𝑖 𝑗) ∈ R

𝐵𝑟 , ℓ new 𝑖 = 𝑒 𝑚𝑖−𝑚𝑖 new ℓ𝑖 + 𝑒 ˜𝑚𝑖 𝑗−𝑚𝑖 new ℓ˜ 𝑖 𝑗 ∈ R

𝐵𝑟 . 14: On chip, compute ˜P dropped 𝑖 𝑗 = dropout( ˜P𝑖 𝑗, 𝑝drop). 15: Write O𝑖 ← diag(ℓ 𝑖 new) −1 (diag(ℓ𝑖)𝑒 𝑚𝑖−𝑚𝑖 new O𝑖 + 𝑒 ˜𝑚𝑖 𝑗−𝑚new 𝑖 ˜P dropped 𝑖 𝑗 V𝑗) to HBM. 16: Write ℓ𝑖 ← ℓ new 𝑖 , 𝑚𝑖 ← 𝑚 new 𝑖 to HBM. 17: end if 18: end for 19: end for 20: Return O, ℓ, 𝑚, R.

## D Extension Details
### D.1 Block-sparse FlashAttention
We describe the full block-sparse FlashAttention algorithm in Algorithm 5. The algorithm is identical to Algorithm 2, except that we skip zero blocks.

We prove the IO-complexity of block-sparse FlashAttention.

Proof of Proposition 4. The proof is very similar to the proof of Theorem 2. For the block-sparse case, notice that we only need to load blocks corresponding to nonzero blocks. As a result, the number of HBM accesses are scaled by 𝑠, the fraction of nonzero blocks in the block-sparsity mask. However, for small values of 𝑠, we would still need to write the result O ∈ R

𝑁 ×𝑑 . Therefore the number of HBM accesses is

Θ  𝑁 𝑑 +

𝑁 2𝑑 2

𝑀 𝑠  . 

### D.2 Potential Extensions
We discuss here a few potential extensions of the IO-aware approach to speed up deep learning training.

Multi-GPU Attention. Large language models are trained on hundreds or thousands of GPUs, and one typically splits the attention computation between 4-8 GPUs on the same node [77]. This introduces another level of memory hierarchy: beside GPU SRAM and GPU HBM, we also have the HBM of other 25

GPUs. For very long sequences, the different GPUs on the same node can cooperate to compute attention by taking into account the asymmetry of different levels of memory hierarchy.

Sparse MLP layers. Typical dense MLP layers are compute-bound and not memory-bound. To improve their efficiency, MLP layers with sparse weight matrices can be used [17]. However, many sparse MLP layers are instead memory-bound, and their speedup is often not proportional to the sparsity. We believe that an

IO-aware implementation can alleviate this issue and realize the benefits of sparsity. We are excited about future work in this direction, to reduce the computational requirement of large models and improve their wall-block runtime.

Kernel machine learning. Our approach in FlashAttention relies on the fact that the 𝑁 × 𝑁 attention matrix is a function of a low-rank matrix QK> (of rank 𝑑  𝑁). As a result, we can repeatedly load the inputs Q, K and recompute the block of the attention matrix that we need, significantly reducing

HBM access. As similar scenario happens in kernel machine learning: each element 𝐾𝑖 𝑗 of the 𝑁 × 𝑁 kernel matrix K is a function of two vectors of size 𝑑  𝑁, as it measures the similarity between two datapoints 𝑥𝑖 and 𝑥 𝑗 . The KeOps library [8, 26] is a successful example of how reducing memory reads/writes can speed up kernel operations. We hope that this will motivate kernel methods that focus more on reducing IOs instead of just FLOPs.

## E Full Experimental Results
### E.1 BERT
We train BERT-large following the training procedure and hyperparameters of the reference MLPerf 1.1 implementation. In particular, we use the LAMB optimizer with learning rate 3.75e-3, with batch size 448, trained for at most 7100 steps. The training is stopped once the validation accuracy (for masked language modeling) reaches the target 72.0%, and the wall-clock run-time is measured. We train with FP16 precision using Apex AMP (with O2 optimization level).

We compare our results with the reported training speed from Nvidia that was submitted to MLPerf 1.1 (Table 1).

We use the same train / validation data split provided by MLPerf 1.1 reference implementation. In particular, we evaluate on the same 10000 validation examples as the baseline from Nvidia.

We train the model on 8×A100-80GB GPUs. Each training run takes between 16 and 19 minutes, and we average the results of 10 runs.

### E.2 GPT-2
We use the standard implementations of GPT-2 [67] from Huggingface transformers library and from

Nvidia’s Megatron-LM repo. We follow the training recipe of the Megatron-LM repo.

We use an effective batch size of 512, and use gradient accumulation to fit into available GPU memory.

We use the AdamW optimizer, with learning rate 6e-4 for GPT-2 small and 1.5e-4 for GPT-2 medium, and weight decay of 0.1. All models are trained with the same hyperparameters for 400K steps. We run all implementations with mixed-precision training (PyTorch AMP).

We use the Openwebtext dataset, with the GPT-2 BPE tokenizer. We randomly select 0.5% of the dataset as the validation set, with the rest being used as training set. This random selection of validation set is done once, and all models are evaluated on the same validation set.

We train the model on 8×A100-40GB GPUs, and we measure the wall-clock training time. Training

GPT-2 small takes between 2.7-9.5 days, and training GPT-2 medium takes between 6.9-21.0 days (Table 2).

In Fig. 4, we plot of the validation perplexity throughout training of GPT-2 small/medium, using either

HuggingFace implementation or our FlashAttention implementation. We see that FlashAttention behaves the same as the baseline implementation and the validation perplexity curves of the two implementations almost lie on top of each other.

Long Document Classification. For MIMIC-III and ECtHR, we follow the hyperparameters of Dai et al. [13]. 26 100k 200k 300k

Training steps 10 15 20 25 30

GPT-2-small HuggingFace

GPT-2-small FlashAttention

GPT-2-medium HuggingFace

GPT-2-medium FlashAttention

Figure 4: Validation perplexity of GPT-2 small/medium using two implementations. We confirm that

FlashAttention yields the same validation curves as the baseline implementation from HuggingFace.

### E.3 LRA details
We follow the hyperparameters from the Long-range arena paper [80], the Long-range arena repo (https: //github.com/google-research/long-range-arena), and the Nyströmformer reproduction [90]. To be generous to the baseline methods, if we are unable to reproduce the performance of any baseline for any of the five tasks, we report the better performance from Tay et al. [80] or Xiong et al. [90] for that baseline on that task.

After hyperparameter tuning, almost all of the attention methods achieve similar accuracy on all of the five LRA tasks.

We run all methods with mixed-precision training, except for Performer (not stable with mixed precision) and Local Attention (implementation does not support FP16).

To calculate the overall wallclock-time speedup, we take the geometric mean of the wallclock-time speedup of each of the five tasks.

Path-X For Path-X and Path-256, we follow the hyperparameters from the PathFinder-32 experiments from the long-range arena paper[80]. For both, we first pretrain a model on Path-64. We take the checkpoint after 200 epochs, upsample its positional embedding (we duplicate the positional embeddings gridwise in space), and fine-tune it on the downstream task for 200 epochs with one epoch of linear warmup, and cosine decay of the learning rate. For Path-X, we take the best performing checkpoint (according to val accuracy), and additionally fine-tune it for 200 epochs with the same warmup and learning rate (this adds roughly 4 points of accuracy to FlashAttention for Path-X, but the model starts overfitting afterwards).

### E.4 Comparison with Apex FMHA
We compare our method/implementation with Apex FMHA (https://github.com/NVIDIA/apex/tree/ master/apex/contrib/csrc/fmha).

When we started this project, Apex FMHA was the fastest implementation of attention (that we knew of), tailored for short sequences of length at most 512. In fact, almost all MLPerf submissions for BERT training benchmark running on Nvidia GPUs use FMHA for their model code, as of MLPerf 1.1 [58]. Since 27

Val perplexity

Table 7: Runtime (ms) of FlashAttention compared to FMHA by sequence length, with masking and dropout, measured on an A100-SXM4-40GB GPU. Batch size 64, 16 heads, head dimension 64 (i.e., BERT-large size).

Attention Method 128 256 512

Apex FMHA forward 0.10 0.29 1.14

FlashAttention forward 0.08 0.22 0.81

Apex FMHA backward 0.17 0.52 1.81

FlashAttention backward 0.20 0.53 2.00

Apex FMHA forward + backward 0.27 0.81 2.95

FlashAttention forward + backward 0.28 0.75 2.81

FMHA targets BERT models, it only supports head dimension 64, and only runs on A100 GPUs. FMHA fuses the attention computation dropout(softmax(mask(QK> )))V into one CUDA kernel. In the forward pass, it stores the attention matrix softmax(mask(QK𝑇 )) to HBM to be used in gradient computation. As a result, it does not offer substantial memory saving (though for shorter sequences memory footprint is often not a primary concern).

We use FMHA code as a starting point, and apply two well-established techniques (tiling and recomputation) to deal with long sequences and to save memory as mentioned in Section 3. As a result, we can support much longer sequences (e.g., up to length 64K). We also support more head dimensions (16, 32, 64, 128) and broader GPU types (all Turing and Ampere GPUs at the time of writing).

In Table 7, we compare the performance of FlashAttention and Apex FMHA for short sequences (as

FMHA only supports sequence length at most 512). Generally FlashAttention is slightly faster than

FMHA in the forward pass and slightly slower than FMHA in the backward pass. This is because we do not store the attention matrix in the forward pass and recompute it in the backward pass. Compared to FMHA, the overall runtime of FlashAttention is about 4% slower for sequence length 128, 8% faster for sequence length 256, and 5% faster for sequence length 512.

### E.5 Speedup On Different Hardware and Configurations
Speedup varies between different types of GPU types and generations depending on HBM bandwidth and

SRAM size. In this section, we profile FlashAttention speedup on different GPUs and configurations.

Figure 5: Speedup over standard PyTorch attention at different sequence lengths, on A100.

A100 Figure 5 shows speedup on an A100 GPU with batch size 8, head dimension 64, and 12 attention heads, across different sequence lengths. We generally see 2-4× speedup, and we see more speedup when using dropout and masking due to kernel fusion. 28

Figure 6: Speedup over standard PyTorch attention at different sequence lengths, on A100, with head dimension 128.

A100, Head Dimension 128 Speedup also changes when we increase the head dimension. Each block requires more memory, so we need to use smaller block sizes to fit into SRAM. Figure 6 shows speedup with head dimension 128 on an A100 (batch size 16, 12 heads). We see less speedup overall—but we can still see significant speedup (up to 3×) with a causal mask, where half the blocks are masked out.

Figure 7: Speedup over standard PyTorch attention at different sequence lengths, on RTX 3090.

RTX 3090 Figure 7 shows speedup on an RTX 3090 GPU. Here, we use batch size 12 with 12 attention heads. We observe slightly higher speedups on the RTX 3090 (between 2.5-4.5×), since the memory bandwidth on an RTX 3090 is lower than on an A100 (roughly 900 GB/s vs. 1.5 TB/s).

T4 Figure 8 shows speedup on a T4 GPU. T4 SRAM is smaller than A100, so we need to make the block sizes smaller in FlashAttention. As a result, we observe less speedup on T4, which matches the IO complexity analysis in Section 3.2. T4 GPUs are commonly used for inference, so we also report speedup on the forward pass only. 29

Figure 8: Speedup over standard PyTorch attention at different sequence lengths, on T4. Top: Combined forward pass + backward pass. Bottom: Forward pass only.

### E.6 Full Benchmarking Results
We report the full benchmarking results and experimental details on A100.

Baselines We compare against reference implementations for exact attention from PyTorch/HuggingFace and Megatron, approximate attention, and sparse attention. For approximate attention, we compare against reference implementations of Reformer [51], Local Attention [68], Linformer Attention [84], Smyrf [19], and

LongShortFormer (LSFormer) [94]. For sparse attention, we compare against reference implementations of

Block-Sparse Attention form OpenAI [11], Longformer[3], and BigBird Attention [92]. For the approximate and sparse attention, we use a compression ratio of 1/8, or a compressed sequence length of 256, whichever is smaller.

Setup We measure runtime and memory usage of the attention computation with 8 heads of dimension 64, and batch size 16 on a machine with one A100 GPU with 40 GB of GPU HBM. We vary sequence length in our experiments. We compute attention on random vectors for Q, K, and V (we do not measure the projection from the hidden layer). For dropout, we use dropout 0.1; for masking, we use a padding mask with uniformly-random mask lengths between the total sequence length and the total sequence length minus

### 20. To measure runtime, we take the average of 100 measurements of the attention call. We only measure
 memory footprint once, since it does not vary between runs. 30

Table 8: Pointers to results tables.

Dropout Masking Pass Table

Yes Yes Forward Table 9

Yes Yes Backward Table 10

Yes Yes Combined Table 11

No Yes Forward Table 12

No Yes Backward Table 13

No Yes Combined Table 14

Yes No Forward Table 15

Yes No Backward Table 16

Yes No Combined Table 17

No No Forward Table 18

No No Backward Table 19

No No Combined Table 20

No No Memory Usage (Combined) Table 21

Table 9: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.36 0.34 0.78 2.54 9.33 36.33 - - - -

Megatron 0.40 0.40 1.10 3.65 16.19 - - - - -

Reformer 2.03 3.15 5.67 11.02 22.59 46.14 97.38 212.13 - -

Local Attention 0.83 0.86 1.01 2.20 7.13 14.32 28.60 57.79 117.67 -

Linformer 0.67 0.52 0.69 0.71 1.65 3.18 6.15 12.16 24.17 52.39

Smyrf 2.27 2.34 3.91 7.44 14.71 29.22 58.27 116.41 - -

LSformer 1.18 1.27 1.34 3.38 11.40 22.55 44.95 89.76 179.66 -

Block Sparse 1.12 1.11 2.13 2.77 6.95 20.91 - - - -

Longformer 1.22 1.14 1.08 1.95 5.72 12.98 - - - -

BigBird 1.13 1.12 1.12 1.77 6.03 13.68 - - - -

FlashAttention 0.04 0.06 0.21 0.82 2.85 10.41 41.74 167.19 670.76 2682.35

Block-Sparse FlashAttention 0.06 0.06 0.06 0.12 0.44 0.86 1.70 3.29 6.55 13.34

We report timing results on the forward pass, backward pass, and combined forward + backward pass.

We measure each method with and without dropout, masking, or both—except for Block Sparse, Longformer, and BigBird. These methods did not successfully run the backward pass with masking due to a bug in external libraries, so we measured them without masking to be generous. We use FP16 for all measurements, except for Local Attention, whose implementation only supports FP32.

For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not support sequence lengths longer than 2048.

Block-Sparse (OpenAI) does not support sequence lengths longer than 4096. Longformer and BigBird do not support sequence lengths longer than 8092.

We measure memory usage on the combined forward + backward pass, without dropout or masking.

Results Table 8 summarizes all the experimental configurations and contains pointers to the results tables. 31

Table 10: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.37 0.49 1.66 5.81 22.32 87.67 - - - -

Megatron 0.35 0.32 0.77 2.42 8.43 - - - - -

Reformer 2.37 4.59 8.91 17.68 35.13 70.05 140.01 - - -

Local Attention 0.55 0.62 1.49 4.03 13.78 27.61 55.20 110.27 221.40 -

Linformer 0.89 0.80 0.81 0.93 2.48 4.75 9.29 18.27 36.53 -

Smyrf 1.41 2.83 5.43 10.72 21.25 42.31 84.48 168.95 - -

LSformer 1.75 1.76 3.01 7.50 20.07 39.08 76.39 150.82 - -

Block Sparse 1.29 1.28 2.18 3.04 7.27 21.16 - - - -

Longformer 1.27 1.31 1.29 2.04 5.24 10.74 25.95 - - -

BigBird 1.33 1.28 1.32 1.81 5.55 11.44 27.45 - - -

FlashAttention 0.30 0.26 0.68 2.02 6.84 26.89 105.70 418.96 1666.89 6660.44

Block-Sparse FlashAttention 0.30 0.27 0.29 0.59 1.50 2.94 5.82 11.85 23.98 47.61

Table 11: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.84 0.86 2.35 8.29 31.75 124.19 - - - -

Megatron 0.87 0.89 1.33 4.21 16.50 - - - - -

Reformer 4.30 7.76 14.60 28.74 57.79 116.34 237.57 - - -

Local Attention 1.40 1.60 2.06 6.06 20.94 42.01 84.08 168.48 339.45 -

Linformer 1.57 1.49 1.55 1.60 4.19 8.04 15.71 30.92 61.47 -

Smyrf 3.41 5.08 9.35 18.18 36.03 71.68 143.04 285.87 - -

LSformer 3.08 3.10 4.26 10.90 31.59 61.72 121.51 241.18 - -

Block Sparse 2.54 2.52 3.71 5.44 13.29 39.19 - - - -

Longformer 2.47 2.49 2.51 3.10 10.39 22.49 60.44 - - -

BigBird 2.51 2.49 2.52 3.40 10.97 23.89 63.28 - - -

FlashAttention 0.43 0.41 0.95 2.55 9.56 37.49 147.75 586.61 2339.11 9341.30

Block-Sparse FlashAttention 0.44 0.44 0.45 0.89 1.95 4.12 7.64 16.60 32.73 64.11

Table 12: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.30 0.30 0.63 1.93 7.08 27.45 112.90 - - -

Megatron 0.45 0.41 0.43 1.52 5.80 - - - - -

Reformer 1.87 3.00 5.37 10.43 21.40 43.83 92.80 203.24 - -

Local Attention 0.70 0.81 1.02 2.09 6.64 13.34 26.77 54.02 110.11 -

Linformer 0.63 0.50 0.67 0.65 1.36 2.60 5.04 9.92 19.69 43.47

Smyrf 2.38 2.32 3.76 7.16 14.14 28.09 55.98 111.73 - -

LSformer 1.22 1.29 1.44 3.28 10.99 21.72 43.29 86.32 172.76 -

Block Sparse 0.96 1.04 1.66 2.16 5.41 16.15 - - - -

Longformer 0.99 0.98 0.99 1.56 4.79 11.07 32.98 - - -

BigBird 0.96 1.02 1.02 1.48 5.05 11.59 34.16 - - -

FlashAttention 0.03 0.04 0.17 0.68 2.28 8.40 33.55 134.14 537.50 2150.88

Block-Sparse FlashAttention 0.05 0.04 0.05 0.11 0.35 0.68 1.33 2.54 5.34 10.73

Table 13: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.44 0.46 1.53 5.33 20.34 79.87 - - - -

Megatron 0.29 0.31 0.65 1.95 6.49 - - - - -

Reformer 2.31 4.47 8.68 17.20 34.14 68.09 136.02 - - -

Local Attention 0.51 0.62 1.30 3.81 13.33 26.72 53.41 106.82 214.15 -

Linformer 0.76 0.81 0.94 0.87 2.24 4.25 8.35 16.38 32.67 72.11

Smyrf 1.34 2.77 5.30 10.46 20.73 41.27 82.41 164.86 - -

LSformer 1.66 1.61 3.09 7.42 19.68 38.35 74.92 147.86 - -

Block Sparse 1.24 1.25 2.04 2.91 6.78 19.67 - - - -

Longformer 1.27 1.23 1.24 1.85 4.99 10.21 24.89 - - -

BigBird 1.43 1.50 1.44 1.69 5.25 10.86 26.26 - - -

FlashAttention 0.21 0.22 0.62 1.84 5.77 22.25 86.21 338.91 1343.91 5361.09

Block-Sparse FlashAttention 0.22 0.22 0.26 0.57 1.55 3.13 5.98 12.21 23.49 47.85 32

Table 14: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.80 0.81 2.08 7.23 27.51 107.58 - - - -

Megatron 0.81 0.83 1.09 3.36 12.39 - - - - -

Reformer 4.16 7.46 14.06 27.68 55.66 112.15 229.37 - - -

Local Attention 1.39 1.68 2.08 5.83 20.04 40.16 80.44 161.35 325.11 -

Linformer 1.51 1.42 1.56 1.67 3.67 6.99 13.63 26.77 53.36 117.56

Smyrf 3.38 4.93 9.07 17.66 34.94 69.55 138.72 277.41 - -

LSformer 3.08 3.10 4.26 10.90 31.59 61.72 121.51 241.18 - -

Block Sparse 2.39 2.40 3.31 5.02 12.25 35.94 - - - -

Longformer 2.36 2.34 2.38 2.94 9.83 21.35 58.12 - - -

BigBird 2.35 2.35 2.37 3.25 10.36 22.57 60.63 - - -

FlashAttention 0.32 0.30 0.83 2.37 7.95 30.77 119.98 473.65 1883.43 7513.01

Block-Sparse FlashAttention 0.34 0.34 0.36 0.69 1.85 3.89 7.16 14.85 30.46 60.03

Table 15: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.26 0.24 0.57 1.80 6.56 25.34 - - - -

Megatron 0.27 0.27 0.56 1.88 6.56 - - - - -

Reformer 1.83 2.96 5.31 10.33 21.19 43.42 91.96 201.34 - -

Local Attention 0.51 0.60 0.78 2.01 6.23 12.52 25.07 50.50 102.18 -

Linformer 0.47 0.37 0.49 0.52 1.37 2.65 5.12 10.13 20.25 44.16

Smyrf 2.12 2.01 3.15 5.97 11.83 23.36 46.48 92.72 - -

LSformer 1.28 1.33 1.51 3.39 11.40 22.54 44.96 89.85 179.73 -

Block Sparse 1.03 1.00 1.72 2.39 5.96 17.88 - - - -

Longformer 1.02 1.03 1.03 1.73 5.10 11.63 34.22 - - -

BigBird 0.99 1.03 1.01 1.58 5.36 12.27 35.56 - - -

FlashAttention 0.10 0.10 0.22 0.83 2.81 10.38 41.63 167.01 668.74 2678.11

Block-Sparse FlashAttention 0.54 0.51 0.68 0.61 0.67 1.10 1.89 3.71 7.18 14.41

Table 16: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.44 0.35 0.90 2.94 10.77 41.67 - - - -

Megatron 0.28 0.33 0.92 2.94 10.80 - - - - -

Reformer 2.24 4.34 8.39 16.62 33.02 65.77 131.52 - - -

Local Attention 0.51 0.58 1.41 3.71 12.96 25.98 51.94 103.72 207.78 -

Linformer 0.84 0.74 0.79 0.85 2.28 4.37 8.66 17.02 33.78 -

Smyrf 1.27 2.56 4.90 9.66 19.16 38.13 76.17 152.39 - -

LSformer 1.67 1.77 3.03 7.52 20.10 39.13 76.35 150.83 - -

Block Sparse 1.27 1.36 2.15 3.04 7.27 21.18 - - - -

Longformer 1.28 1.34 1.38 1.98 5.24 10.74 25.95 - - -

BigBird 1.48 1.47 1.50 1.81 5.57 11.38 27.43 - - -

FlashAttention 0.15 0.18 0.58 1.86 6.50 26.21 104.27 416.10 1661.92 6643.01

Block-Sparse FlashAttention 0.17 0.17 0.17 0.40 1.10 2.04 4.43 9.33 18.28 37.31

Table 17: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.66 0.67 1.43 4.82 17.47 67.29 - - - -

Megatron 0.88 0.90 1.49 4.73 17.41 - - - - -

Reformer 4.06 7.28 13.68 26.98 54.27 109.39 223.80 - - -

Local Attention 1.09 1.40 1.99 5.61 19.23 38.62 77.30 154.63 311.12 -

Linformer 1.31 1.21 1.30 1.39 3.73 7.15 14.05 27.69 55.00 -

Smyrf 3.00 4.37 8.05 15.66 31.04 61.64 123.04 245.65 - -

LSformer 3.07 3.17 4.31 10.89 31.54 61.78 121.56 240.94 - -

Block Sparse 2.54 2.52 3.71 5.44 13.29 39.19 - - - -

Longformer 2.47 2.49 2.51 3.10 10.39 22.49 60.44 - - -

BigBird 2.51 2.49 2.52 3.40 10.97 23.89 63.28 - - -

FlashAttention 0.35 0.36 0.80 2.52 9.16 36.70 146.13 583.45 2332.01 9323.63

Block-Sparse FlashAttention 0.91 0.83 0.94 0.92 1.83 3.50 7.02 13.56 26.71 53.92 33

Table 18: Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length.

Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.21 0.22 0.43 1.27 4.32 16.47 67.77 - - -

Megatron 0.24 0.26 0.42 1.33 4.28 - - - - -

Reformer 1.77 2.82 5.01 9.74 20.03 41.11 87.39 192.40 - -

Local Attention 0.48 0.57 0.80 1.90 5.76 11.56 23.13 46.65 94.74 -

Linformer 0.46 0.36 0.45 0.50 1.09 2.09 4.01 7.90 15.70 35.40

Smyrf 1.94 1.96 3.01 5.69 11.26 22.23 44.21 88.22 - -

LSformer 1.21 1.34 1.34 3.31 11.01 21.71 43.27 86.32 172.85 -

Block Sparse 0.96 1.04 1.66 2.16 5.41 16.15 - - - -

Longformer 0.99 0.98 0.99 1.56 4.79 11.07 32.98 - - -

BigBird 0.96 1.02 1.02 1.48 5.05 11.59 34.16 - - -

FlashAttention 0.08 0.09 0.18 0.68 2.40 8.42 33.54 134.03 535.95 2147.05

Block-Sparse FlashAttention 0.56 0.52 0.63 0.65 0.61 0.96 1.69 3.02 5.69 11.77

Table 19: Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length.

Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.26 0.29 0.78 2.44 8.82 33.87 - - - -

Megatron 0.29 0.30 0.80 2.59 8.86 - - - - -

Reformer 2.18 4.21 8.14 16.12 32.02 63.84 127.60 - - -

Local Attention 0.51 0.64 1.28 3.60 12.52 25.08 50.22 100.23 200.66 -

Linformer 0.69 0.76 0.69 0.80 2.04 3.88 7.67 15.04 30.11 63.15

Smyrf 1.24 2.49 4.77 9.42 18.65 37.12 74.15 148.35 - -

LSformer 1.68 1.61 3.02 7.40 19.72 38.27 74.89 147.99 - -

Block Sparse 1.24 1.25 2.04 2.91 6.78 19.67 - - - -

Longformer 1.27 1.23 1.24 1.85 4.99 10.21 24.89 - - -

BigBird 1.43 1.50 1.44 1.69 5.25 10.86 26.26 - - -

FlashAttention 0.11 0.16 0.52 1.62 5.45 21.57 84.75 336.00 1338.56 5343.19

Block-Sparse FlashAttention 0.11 0.12 0.16 0.38 1.20 2.34 4.69 9.10 18.74 37.04

Table 20: Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 0.67 0.70 1.18 3.67 13.22 50.44 - - - -

Megatron 0.74 0.65 1.23 3.80 13.21 - - - - -

Reformer 3.93 7.01 13.15 25.89 52.09 105.00 215.13 - - -

Local Attention 1.09 1.27 1.99 5.38 18.32 36.77 73.67 147.29 296.35 -

Linformer 1.31 1.25 1.30 1.29 3.20 6.10 11.93 23.39 46.72 100.52

Smyrf 2.98 4.23 7.78 15.12 29.96 59.45 118.60 237.02 - -

LSformer 3.03 3.05 4.26 10.70 30.77 60.15 118.33 234.94 - -

Block Sparse 2.39 2.40 3.31 5.02 12.25 35.94 - - - -

Longformer 2.36 2.34 2.38 2.94 9.83 21.35 58.12 - - -

BigBird 2.35 2.35 2.37 3.25 10.36 22.57 60.63 - - -

FlashAttention 0.31 0.31 0.73 2.29 7.64 30.09 118.50 470.51 1876.08 7492.85

Block-Sparse FlashAttention 0.74 0.77 0.82 0.88 1.71 3.21 6.56 12.60 24.93 50.39

Table 21: Memory usage (MB) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

Attention Method 128 256 512 1024 2048 4096 8192 16384 32768 65536

PyTorch Attention 36 104 336 1184 4416 17024 - - - -

Megatron 36 104 336 1184 4416 - - - - -

Reformer 377 754 1508 3016 6033 12067 24134 - - -

Local Attention 53 110 232 592 1696 3392 6784 13568 27136 -

Linformer 25 52 114 287 832 1652 3292 6572 13132 26252

Smyrf 217 434 868 1737 3474 6947 13894 27788 - -

LSformer 72 152 333 796 2540 5068 10125 20240 - -

Block Sparse 33 82 228 408 910 2401 - - - -

Longformer 30 61 124 277 681 1370 2748 - - -

BigBird 33 66 131 294 708 1431 2872 - - -

FlashAttention 22 44 104 209 418 836 1672 3344 6688 13376

Block-Sparse FlashAttention 22 44 104 209 418 836 1672 3344 6690 13384
