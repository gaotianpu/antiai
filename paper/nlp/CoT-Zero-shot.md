# Large Language Models are Zero-Shot Reasoners
大型语言模型是零样本推理机 2022.5.24 https://arxiv.org/abs/2205.11916


## Abstract
Pretrained large language models (LLMs) are widely used in many sub-fields of natural language processing (NLP) and generally known as excellent few-shot learners with task-specific exemplars. Notably, chain of thought (CoT) prompting, a recent technique for eliciting complex multi-step reasoning through step-bystep answer examples, achieved the state-of-the-art performances in arithmetics and symbolic reasoning, difficult system-2 tasks that do not follow the standard scaling laws for LLMs. While these successes are often attributed to LLMs’ ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding “Let’s think step by step” before each answer. Experimental results demonstrate that our Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date Understanding, Tracking Shuffled Objects), without any hand-crafted few-shot examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with large-scale InstructGPT model (text-davinci- 002), as well as similar magnitudes of improvements with another off-the-shelf large model, 540B parameter PaLM. The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting. We hope our work not only serves as the minimal strongest zero-shot baseline for the challenging reasoning benchmarks, but also highlights the importance of carefully exploring and analyzing the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning datasets or few-shot exemplars.

预训练大型语言模型 (LLM) 广泛应用于自然语言处理 (NLP) 的许多子领域，通常被称为具有特定任务范例的优秀少样本学习器。 值得注意的是，思维链 (CoT) 提示是一种通过逐步回答样本引发复杂多步推理的最新技术，在算术和符号推理、困难的 system-2 任务中实现了最先进的性能 不遵循 LLM 的标准缩放法则。 虽然这些成功通常归因于 LLM 的小样本学习能力，但我们通过在每个答案前简单地添加“让我们逐步思考”来证明 LLM 是不错的零样本推理机。 实验结果表明，我们的零样本 CoT 使用相同的单一提示模板，在包括算术（MultiArith、GSM8K、AQUA-RAT、SVAMP）、符号推理（Last Letter、 硬币翻转）和其他逻辑推理任务（日期理解、跟踪打乱的对象），没有任何手工制作的小样本样本，例如 使用大型 InstructGPT 模型 (text-davinci-002) 将 MultiArith 的准确度从 17.7% 提高到 78.7%，将 GSM8K 的准确度从 10.4% 提高到 40.7%，以及与另一个现成的大型模型相似的改进幅度， 540B 参数 PaLM。 这种单一提示在非常不同的推理任务中的多功能性暗示了 LLM 尚未开发和未充分研究的基本零样本能力，表明可以通过简单的提示提取高级、多任务的广泛认知能力。 我们希望我们的工作不仅可以作为具有挑战性的推理基准的最小最强零样本基线，而且还强调了在制作微调数据集或少样本样本之前仔细探索和分析隐藏在 LLM 中的大量零样本知识的重要性。

## 1 Introduction
Scaling up the size of language models has been key ingredients of recent revolutions in natural language processing (NLP) [Vaswani et al., 2017, Devlin et al., 2019, Raffel et al., 2020, Brown et al., 2020, Thoppilan et al., 2022, Rae et al., 2021, Chowdhery et al., 2022]. The success of large language models (LLMs) is often attributed to (in-context) few-shot or zero-shot learning. It can solve various tasks by simply conditioning the models on a few examples (few-shot) or instructions describing the task (zero-shot). The method of conditioning the language model is called “prompting” [Liu et al., 2021b], and designing prompts either manually [Schick and Schütze, 2021, Reynolds and McDonell, 2021] or automatically [Gao et al., 2021, Shin et al., 2020] has become a hot topic in NLP.  

扩大语言模型的规模一直是自然语言处理 (NLP) 近期革命的关键因素 [Vaswani et al., 2017; Devlin et al., 2019; Raffel et al., 2020; Brown et al., 2020， Thoppilan et al., 2022; Rae et al., 2021; Chowdhery et al., 2022]。 大型语言模型 (LLM) 的成功通常归因于（上下文中的）少样本或零样本学习。 它可以通过简单地根据几个样本（少量样本）或描述任务的指令（零样本）对模型进行调节来解决各种任务。 调节语言模型的方法称为“提示”[Liu et al., 2021b]，并手动设计提示 [Schick and Schütze, 2021, Reynolds and McDonell, 2021] 或自动 [Gao et al., 2021, Shin et al., 2020] 已成为 NLP 领域的热门话题。

![Figure 1](../images/CoT-Zero-shot/fig_1.png)<br/>
Figure 1: Example inputs and outputs of GPT-3 with (a) standard Few-shot ([Brown et al., 2020]), (b) Few-shot-CoT ([Wei et al., 2022]), (c) standard Zero-shot, and (d) ours (Zero-shot-CoT). Similar to Few-shot-CoT, Zero-shot-CoT facilitates multi-step reasoning (blue text) and reach correct answer where standard prompting fails. Unlike Few-shot-CoT using step-by-step reasoning examples per task, ours does not need any examples and just uses the same prompt “Let’s think step by step” across all tasks (arithmetic, symbolic, commonsense, and other logical reasoning tasks). 
图 1：GPT-3 的样本输入和输出（a）标准 Few-shot（[Brown et al., 2020]），（b）Few-shot-CoT（[Wei et al., 2022]），（ c) 标准零样本，和 (d) 我们的（零样本 CoT）。 与 Few-shot-CoT 类似，Zero-shot-CoT 有助于多步推理（蓝色文本）并在标准提示失败的情况下得出正确答案。 与 Few-shot-CoT 每个任务使用逐步推理样本不同，我们的不需要任何样本，只是在所有任务（算术、符号、常识和其他逻辑推理）中使用相同的提示“让我们逐步思考” 任务）。

In contrast to the excellent performance of LLMs in intuitive and single-step system-1 [Stanovich and West, 2000] tasks with task-specific few-shot or zero-shot prompting [Liu et al., 2021b], even language models at the scale of 100B or more parameters had struggled on system-2 tasks requiring slow and multi-step reasoning [Rae et al., 2021]. To address this shortcoming, Wei et al. [2022], Wang et al. [2022] have proposed chain of thought prompting (CoT), which feed LLMs with the step-by-step reasoning examples rather than standard question and answer examples (see Fig. 1-a). Such chain of thought demonstrations facilitate models to generate a reasoning path that decomposes the complex reasoning into multiple easier steps. Notably with CoT, the reasoning performance then satisfies the scaling laws better and jumps up with the size of the language models. For example, when combined with the 540B parameter PaLM model [Chowdhery et al., 2022], chain of thought prompting significantly increases the performance over standard few-shot prompting across several benchmark reasoning tasks, e.g., GSM8K (17.9% → 58.1%).

与 LLMs 在直观和单步 system-1 [Stanovich and West, 2000] 任务中的出色表现相比，具有任务特定的少样本或零样本提示 [Liu et al., 2021b]，甚至语言模型在 100B 或更多参数的规模在需要缓慢和多步推理的 system-2 任务上表现不佳 [Rae et al., 2021]。 为了解决这个缺点，Wei et al. [2022]，Wang et al. [2022] 提出了思维提示链 (CoT)，它为LLMs提供逐步推理样本，而不是标准的问答样本（见图 1-a）。 这种思维链演示有助于模型生成推理路径，将复杂的推理分解为多个更简单的步骤。 值得注意的是，对于 CoT，推理性能可以更好地满足缩放定律，并随着语言模型的大小而上升。 例如，当与 540B 参数 PaLM 模型 [Chowdhery et al., 2022] 相结合时，思维链提示显著提高了跨多个基准推理任务的标准少样本提示的性能，例如 GSM8K (17.9% → 58.1%) .

While the successes of CoT prompting [Wei et al., 2022], along those of many other task-specific prompting work [Gao et al., 2021, Schick and Schütze, 2021, Liu et al., 2021b], are often attributed to LLMs’ ability for few-shot learning [Brown et al., 2020], we show that LLMs are decent zero-shot reasoners by adding a simple prompt, Let’s think step by step, to facilitate step-by-step thinking before answering each question (see Figure 1). Despite the simplicity, our Zero-shot-CoT successfully generates a plausible reasoning path in a zero-shot manner and reaches the correct answer in a problem where the standard zero-shot approach fails. Importantly, our Zero-shot-CoT is versatile and task-agnostic, unlike most prior task-specific prompt engineering in the forms of examples (few-shot) or templates (zero-shot) [Liu et al., 2021b]: it can facilitate step-by-step answers across various reasoning tasks, including arithmetic (MultiArith [Roy and Roth, 2015], GSM8K [Cobbe et al., 2021], AQUA-RAT [Ling et al., 2017], and SVAMP [Patel et al., 2021]), symbolic reasoning (Last letter and Coin flip), commonsense reasoning (CommonSenseQA [Talmor et al., 2019] and Strategy QA [Geva et al., 2021]), and other logical reasoning tasks (Date understanding and Tracking Shuffled Objects from BIG-bench [Srivastava et al., 2022]) without modifying the prompt per task.

虽然 CoT 提示的成功 [Wei et al., 2022]，以及许多其他特定任务提示工作的成功 [Gao et al., 2021, Schick and Schütze, 2021, Liu et al., 2021b]，通常归因于 为了 LLM 的少样本学习能力 [Brown et al., 2020]，我们通过添加一个简单的提示 Let's think step by step 来证明 LLM 是不错的零样本推理机，以促进在回答之前逐步思考 每个问题（见图 1）。 尽管简单，但我们的零样本 CoT 成功地以零样本方式生成了合理的推理路径，并在标准零样本方法失败的问题中得出了正确答案。 重要的是，我们的 Zero-shot-CoT 具有通用性和任务不可知性，这与大多数先前以样本（少量）或模板（零次）形式出现的特定于任务的提示工程不同 [Liu et al., 2021b]：它 可以促进各种推理任务的逐步答案，包括算术（MultiArith [Roy and Roth, 2015]、GSM8K [Cobbe et al., 2021]、AQUA-RAT [Ling et al., 2017] 和 SVAMP [ Patel et al., 2021]）、符号推理（Last letter and Coin flip）、常识推理（CommonSenseQA [Talmor et al., 2019] 和 Strategy QA [Geva et al., 2021]）和其他逻辑推理任务（ 来自 BIG-bench [Srivastava et al., 2022]）的数据理解和跟踪随机对象，无需修改每个任务的提示。

We empirically evaluate Zero-shot-CoT against other prompting baselines in Table 2. While our Zero-shot-CoT underperforms Few-shot-CoT with carefully-crafted and task-specific step-by-step examples, Zero-shot-CoT achieves enormous score gains compared to the zero-shot baseline, e.g. from 17.7% to 78.7% on MultiArith and from 10.4% to 40.7% on GSM8K with large-scale InstructGPT model (text-davinci-002). We also evaluate Zero-shot-CoT with another off-the-shelf large model, 540B parameter PaLM, showing similar magnitudes of improvements on MultiArith and GSM8K. Importantly, with our single fixed prompt, zero-shot LLMs have a significantly better scaling curve comparable to that of the few-shot CoT baseline. We also show that besides Few-shot-CoT requiring human engineering of multi-step reasoning prompts, their performance deteriorates if prompt example question types and task question type are unmatched, suggesting high sensitivity to per-task prompt designs. In contrast, the versatility of this single prompt across diverse reasoning tasks hints at untapped and understudied zero-shot fundamental capabilities of LLMs, such as higher-level broad cognitive capabilities like generic logical reasoning [Chollet, 2019]. While the vibrant field of LLMs started out from the premise of excellent few-shot learners [Brown et al., 2020], we hope our work encourages more research into uncovering high-level and multi-task zero-shot capabilities hidden inside those models. 

我们根据表 2 中的其他提示基线凭经验评估零样本 CoT。虽然我们的零样本 CoT 通过精心设计和特定于任务的逐步样本表现不及 Few-shot-CoT，但零样本 CoT 实现了 与零样本基线相比，得分大幅提高，例如 在 MultiArith 上从 17.7% 到 78.7%，在 GSM8K 上使用大规模 InstructGPT 模型（text-davinci-002）从 10.4% 到 40.7%。 我们还使用另一个现成的大型模型 540B 参数 PaLM 评估零样本 CoT，显示出在 MultiArith 和 GSM8K 上类似的改进幅度。 重要的是，在我们的单一固定提示下，零样本 LLM 具有比少样本 CoT 基线更好的缩放曲线。 我们还表明，除了 Few-shot-CoT 需要人工设计多步推理提示外，如果提示样本问题类型和任务问题类型不匹配，它们的性能会下降，这表明对每个任务提示设计的高度敏感。 相比之下，这种单一提示在不同推理任务中的多功能性暗示了 LLM 尚未开发和未被充分研究的零样本基本能力，例如更高层次的广泛认知能力，如通用逻辑推理 [Chollet，2019]。 虽然充满活力的 LLM 领域始于优秀的少样本学习者 [Brown et al., 2020]，但我们希望我们的工作能够鼓励更多的研究来发现隐藏在这些模型中的高级和多任务零样本能力 .

## 2 Background
We briefly review the two core preliminary concepts that form the basis of this work: the advent of large language models (LLMs) and prompting, and chain of thought (CoT) prompting for multi-step reasoning.

我们简要回顾了构成这项工作基础的两个核心初步概念：大型语言模型 (LLM) 和提示的出现，以及用于多步推理的思维链 (CoT) 提示。

Large language models and prompting. A language model (LM), is a model that looks to estimate the probability distribution over text. Recently, scaling improvements through larger model sizes (from a few million [Merity et al., 2016] to hundreds of millions [Devlin et al., 2019] to hundreds of billions [Brown et al., 2020] parameters) and larger data (e.g. webtext corpora [Gao et al., 2020]) have enabled pre-trained large language models (LLMs) to be incredibly adept at many downstream NLP tasks. Besides the classic “pre-train and fine-tune” paradigm [Liu et al., 2021b], models scaled to 100B+ parameters exhibit properties conducive to few-shot learning [Brown et al., 2020], by way of in context learning, where one can use a text or template known as a prompt to strongly guide the generation to output answers for desired tasks, thus beginning an era of “pre-train and prompt” [Liu et al., 2021a]. In work, we call such prompts with explicit conditioning on few task examples as few-shot prompts, and other template-only prompts as zero-shot prompts.

大型语言模型和提示。 语言模型 (LM) 是一种旨在估计文本概率分布的模型。 最近，通过更大的模型尺寸（从几百万 [Merity et al., 2016] 到数亿 [Devlin et al., 2019] 再到数千亿 [Brown et al., 2020] 参数）和更大的数据进行扩展改进 （例如 webtext 语料库 [Gao et al., 2020]）使预训练的大型语言模型 (LLM) 非常擅长许多下游 NLP 任务。 除了经典的“预训练和微调”范例 [Liu et al., 2021b]，缩放到 100B+ 参数的模型通过上下文学习表现出有利于小样本学习的特性 [Brown et al., 2020] ，其中可以使用称为提示的文本或模板来强烈引导生成输出所需任务的答案，从而开启“预训练和提示”的时代 [Liu et al., 2021a]。 在工作中，我们将这种在少数任务样本上具有显式条件的提示称为少样本提示，将其他仅模板提示称为零样本提示。

Chain of thought prompting. Multi-step arithmetic and logical reasoning benchmarks have particularly challenged the scaling laws of large language models [Rae et al., 2021]. Chain of thought (CoT) prompting [Wei et al., 2022], an instance of few-shot prompting, proposed a simple solution by modifying the answers in few-shot examples to step-by-step answers, and achieved significant boosts in performance across these difficult benchmarks, especially when combined with very large language models like PaLM [Chowdhery et al., 2022]. The top row of Figure 1 shows standard few-shot prompting against (few-shot) CoT prompting. Notably, few-shot learning was taken as a given for tackling such difficult tasks, and the zero-shot baseline performances were not even reported in the original work [Wei et al., 2022]. To differentiate it from our method, we call Wei et al. [2022] as Few-shot-CoT in this work. 

思维链提示。 多步算术和逻辑推理基准尤其挑战了大型语言模型的缩放法则 [Rae et al., 2021]。 Chain of thought (CoT) prompting [Wei et al., 2022]，few-shot prompting 的一个实例，通过将 few-shot 样本中的答案修改为逐步答案，提出了一种简单的解决方案，并取得了显著提升 在这些困难的基准测试中的性能，尤其是当与 PaLM [Chowdhery et al., 2022 ] 等非常大的语言模型结合使用时。 图 1 的第一行显示了标准的小样本提示与（小样本）CoT 提示。 值得注意的是，小样本学习被视为解决此类困难任务的既定条件，原始工作中甚至没有报告零样本基线性能 [Wei et al., 2022]。 为了将其与我们的方法区分开来，我们称 Wei  et al. 为。 [2022] 在这项工作中作为 Few-shot-CoT。

## 3 Zero-shot Chain of Thought
We propose Zero-shot-CoT, a zero-shot template-based prompting for chain of thought reasoning. It differs from the original chain of thought prompting [Wei et al., 2022] as it does not require step-by-step few-shot examples, and it differs from most of the prior template prompting [Liu et al., 2021b] as it is inherently task-agnostic and elicits multi-hop reasoning across a wide range of tasks with a single template. The core idea of our method is simple, as described in Figure 1: add Let’s think step by step, or a a similar text (see Table 4), to extract step-by-step reasoning.

我们提出零样本 CoT，一种基于零样本模板的思维链推理提示。 它不同于原始的思维链提示 [Wei et al., 2022]，因为它不需要逐步的小样本样本，并且它不同于大多数先前的模板提示 [Liu et al., 2021b] 因为它本质上是与任务无关的，并且可以使用单个模板在广泛的任务中引发多跳推理。 我们方法的核心思想很简单，如图 1 所示：添加 Let's think step by step 或类似的文本（见表 4），以提取逐步推理。

### 3.1 Two-stage prompting
While Zero-shot-CoT is conceptually simple, it uses prompting twice to extract both reasoning and answer, as explained in Figure 2. In contrast, the zero-shot baseline (see the bottom-left in Figure 1) already uses prompting in the form of “The answer is”, to extract the answers in correct formats. Few-shot prompting, standard or CoT, avoids needing such answer-extraction prompting by explicitly designing the few-shot example answers to end in such formats (see the top-right and top-left in Figure 1). In summary, Few-shot-CoT [Wei et al., 2022] requires careful human engineering of a few prompt examples with specific answer formats per task, while Zero-shot-CoT requires less engineering but requires prompting LLMs twice. 

虽然零样本 CoT 在概念上很简单，但它使用提示两次来提取推理和答案，如图 2 中所述。相比之下，零样本基线（参见图 1 中的左下角）已经在 “答案是”的形式，以正确的格式提取答案。 Few-shot 提示（标准或 CoT）通过明确设计以此类格式结尾的 few-shot 样本答案来避免需要此类答案提取提示（参见图 1 中的右上角和左上角）。 总之，Few-shot-CoT [Wei et al., 2022] 需要对一些提示样本进行仔细的人工工程，每个任务具有特定的答案格式，而 Zero-shot-CoT 需要较少的工程但需要提示 LLM 两次。

![Figure 2](../images/CoT-Zero-shot/fig_2.png)<br/>
Figure 2: Full pipeline of Zero-shot-CoT as described in § 3: we first use the first “reasoning” prompt to extract a full reasoning path from a language model, and then use the second “answer” prompt to extract the answer in the correct format from the reasoning text. 
图 2：第 3 节中描述的零样本 CoT 的完整管道：我们首先使用第一个“推理”提示从语言模型中提取完整的推理路径，然后使用第二个“答案”提示提取答案 以推理文本的正确格式。


#### 1st prompt: reasoning extraction 第一个提示：推理提取.
In this step we first modify the input question x into a prompt x0 using a simple template “Q: [X]. A: [T]”, where [X] is an input slot for x and [T] is an slot for hand-crafted trigger sentence t that would extract chain of though to answer the question x. For example, if we use “Let’s think step by step” as a trigger sentence, the prompt x0 would be “Q: [X]. A: Let’s think step by step.”. See Table 4 for more trigger examples. Prompted text x0 is then fed into a language model and generate subsequent sentence z. We can use any decoding strategy, but we used greedy decoding throughout the paper for the simplicity. 

在这个步骤中，我们首先使用一个简单的模板“Q:[x]. A:[T]”将输入问题x修改为提示x0，其中[x]是x的输入槽，[T]是手工制作的触发句T的槽，该触发句T将提取尽管链来回答问题x。例如，如果我们使用“让我们一步一步地思考”作为触发句，提示x0将是“Q:[X]。A:让我们一步一步地思考。”。有关更多触发器样本，请参见表4。然后将提示文本x0输入到语言模型中，并生成随后的句子z。我们可以使用任何解码策略，但为了简单起见，我们在整个论文中使用了贪婪解码。

#### 2nd prompt: answer extraction 第二个提示：答案提取.
In the second step, we use generated sentence z along with prompted sentence x0 to extract the final answer from the language model. To be concrete, we simply concatenate three elements as with “[X0 ] [Z] [A]”: [X0 ] for 1st prompt x0 , [Z] for sentence z generated at the first step, and [A] for a trigger sentence to extract answer. The prompt for this step is self-augmented, since the prompt contains the sentence z generated by the same language model. In experiment, we use slightly different answer trigger depending on the answer format. For example, we use “Therefore, among A through E, the answer is” for multi-choice QA, and “Therefore, the answer (arabic numerals) is” for math problem requiring numerical answer. See Appendix A.5 for the lists of answer trigger sentences. Finally, the language model is fed the prompted text as input to generate sentences ˆy and parse the final answer. See “Answer Cleansing” at §4 for the parser details. 

在第二步中，我们使用生成的句子 z 和提示句子 x0 从语言模型中提取最终答案。 具体来说，我们简单地连接三个元素，如“[X0] [Z] [A]”：[X0]表示第一个提示 x0，[Z]表示第一步生成的句子 z，[A]表示触发器 句子提取答案。 此步骤的提示是自增强的，因为提示包含由相同语言模型生成的句子 z。 在实验中，我们根据答案格式使用略有不同的答案触发器。 例如，对于多选题我们使用“Therefore, among A through E, the answer is”，对于需要数值答案的数学题，我们使用“Therefore, the answer (arabic numbers) is”。 答案触发句列表见附录 A.5。 最后，语言模型将提示文本作为输入来生成句子 y 并解析最终答案。 有关解析器的详情，请参阅第 4 节的“答案清理”。

## 4 Experiment
Tasks and datasets We evaluate our proposal on 12 datasets from four categories of reasoning tasks: arithmetic, commonsense, symbolic, and other logical reasoning tasks. See Appendix A.2 for the detailed description of each datasets.

For arithmetic reasoning, we consider the following six datasets: (1) SingleEq [Koncel-Kedziorski et al., 2015], (2) AddSub [Hosseini et al., 2014], (3) MultiArith [Roy and Roth, 2015], (4) AQUARAT [Ling et al., 2017], (5) GSM8K [Cobbe et al., 2021], and (6) SVAMP [Patel et al., 2021]. The first three are from the classic Math World Problem Repository [Koncel-Kedziorski et al., 2016], and the last three are from more recent benchmarks. SingleEq and AddSub contain easier problems, which do not require multi-step calculation to solve the tasks. MultiArith, AQUA-RAT, GSM8k, and

SVAMP are more challenging datasets that require multi-step reasoning to solve.

For commonsense reasoning, we use CommonsenseQA [Talmor et al., 2019] and StrategyQA [Geva et al., 2021]. CommonsenseQA asks questions with complex semantics that often require reasoning 4 based on prior knowledge [Talmor et al., 2019]. StrategyQA requires models to infer an implicit multi-hop reasoning to answer questions [Geva et al., 2021].

For symbolic reasoning, we use Last Letter Concatenation and Coin Flip [Wei et al., 2022]. Last letter Concatenation asks the model to concatenate the last letters of each word. We used randomly selected four names for each sample. Coin Flip asks the model to answer whether a coin is still heads up after people either flip or do not flip the coin. We created samples of four times flip or not flip trials. Although these tasks are easy for humans, LMs typically exhibit a flat scaling curve.

For other logical reasoning tasks, we choose two evaluation sets from the BIG-bench effort [Srivastava et al., 2022]: Date Understanding 2 and Tracking Shuffled Objects. Date Understanding asks models to infer the date from a context. Tracking Shuffled Objects tests a model’s ability to infer the final state of objects given its initial state and a sequence of object shuffling. We used a dataset of tracking three shuffled objects for our experiment.

Models We experiment with 17 models in total. Main experiments are conducted with InstructGPT3 [Ouyang et al., 2022] (text-ada/babbage/curie/davinci-001 and text-davinci-002)3 , original

GPT3 [Brown et al., 2020] (ada, babbage, curie, and davinci)4 , and PaLM [Chowdhery et al., 2022] (8B, 62B, and 540B). In addition, we used GPT-2[Radford et al., 2019], GPT-Neo[Black et al., 2021],

GPT-J[Wang and Komatsuzaki, 2021], T0 [Sanh et al., 2022], and OPT [Zhang et al., 2022] for model scaling study. The size of LMs ranges from 0.3B to 540B. We include both standard (e.g. GPT-3 and

OPT), and instruction following variants (e.g. Instruct-GPT3 and T0). See Appendix A.3 for model description details. Unless otherwise stated, we use text-davinci-002 throughout the experiments.

Baselines We compare our Zero-shot-CoT mainly to standard Zero-shot prompting to verify the effectiveness of its chain of thought reasoning. For Zero-shot experiments, similar answer prompts as Zero-shot-CoT are used as default. See Appendix A.5 for detail. To better evaluate the zero-shot ability of LLMs on reasoning tasks, we also compare our method to Few-shot and Few-shot-CoT baselines from [Wei et al., 2022], using the same in-context examples. Throughout the experiments, we use greedy decoding across all the methods. For the zero-shot approaches, the results are therefore deterministic. For the few-shot approaches, since the order of in-context examples could affect the results [Lu et al., 2022], we run each experiment only once with a fixed seed across all methods and datasets, for fair comparisons with the zero-shot methods. Wei et al. [2022] showed that the order of examples did not cause large variance in CoT experiments.

Answer cleansing After the model outputs a text by answer extraction (see § 3 and Figure 2), our method picks up only the part of the answer text that first satisfies the answer format. For example, if the answer prompting outputs “probably 375 and 376” on arithmetic tasks, we extract the first number “375” and set it as the model prediction. In the case of multiple-choice, the first large letter we encounter is set as the prediction. See Appendix A.6 for more detail. Standard Zero-shot method follows the same idea. For Few-shot and Few-shot-CoT methods, we follow [Wang et al., 2022] and first extract the answer text after "The answer is " from the model output, and apply the same answer cleansing to parse the answer text. If “The answer is” is not found in the model output, we search from the back of the text and set the first text that satisfies the answer format as the prediction.

### 4.1 Results
Zero-shot-CoT vs. Zero-shot Table 1 summarize accuracy of our method (Zero-shot-CoT) and standard zero-shot prompting (Zero-shot) for each dataset. Zero-shot-CoT substantially outperforms four out of six arithmetic reasoning tasks (MultiArith, GSM8K, AQUA, SVAMP), all symbolic reasoning, and all other logical reasoning tasks (from BIG-bench [Srivastava et al., 2022]). For 2While prior work [Wei et al., 2022] categorized Date Understanding task into Common Sense reasoning, our study categorized this task into logical reasoning because this task requires less prior knowledge and more logical reasoning between dates. 3Our experiment for Instruct GPT-3 models includes both text-****-001 and text-davinci-002. Text-davinci- 002 differs from text-****-001 in that they use different fine-tuning data depending on the date range collected from the APIs. Specifically, text-davinci-002 uses data up to Jun 2021, while text-****-001 uses data up to Oct 2019. (See https://beta.openai.com/docs/engines/gpt-3) 4Our experiments with GPT3 series are conducted by using OpenAI API between April-2022 and May-2022,
 except for No.10-16 in Table 4 in Aug-2022. 5

Table 1: Accuracy comparison of Zero-shot-CoT with Zero-shot on each tasks. The values on the left side of each task are the results of using answer extraction prompts depending on answer format as described at § 3. The values on the right side are the result of additional experiment where standard answer prompt "The answer is" is used for answer extraction. See Appendix A.5 for detail setups.

Arithmetic

SingleEq AddSub MultiArith GSM8K AQUA SVAMP zero-shot 74.6/78.7 72.2/77.0 17.7/22.7 10.4/12.5 22.4/22.4 58.8/58.7 zero-shot-cot 78.0/78.7 69.6/74.7 78.7/79.3 40.7/40.5 33.5/31.9 62.1/63.7

Common Sense Other Reasoning Tasks Symbolic Reasoning

Common Strategy Date Shuffled Last Letter Coin Flip

SenseQA QA Understand Objects (4 words) (4 times) zero-shot 68.8/72.6 12.7/54.3 49.3/33.6 31.3/29.7 0.2/- 12.8/53.8 zero-shot-cot 64.6/64.0 54.8/52.3 67.5/61.8 52.4/52.9 57.6/- 91.4/87.8

Table 2: Comparison with baseline methods using accuracies on MultiArith and GSM8K. text-davinci- 002 is used as the model if not specified. We used the same 8 examples as described in [Wei et al., 2022] for Few-shot and Few-shot-CoT settings. (*1) To verify the variance of changing examples, we report two results for 4-shot-cot by splitting the eight examples into two groups. (*2) We insert “Let’s think step by step.” at the beginning of answer part of each exemplars for Few-shot-CoT to test performance gains. Further experiment results with PaLM are found at Appendix D

MultiArith GSM8K

Zero-Shot 17.7 10.4

Few-Shot (2 samples) 33.7 15.6

Few-Shot (8 samples) 33.8 15.6

Zero-Shot-CoT 78.7 40.7

Few-Shot-CoT (2 samples) 84.8 41.3

Few-Shot-CoT (4 samples : First) (*1) 89.2 -

Few-Shot-CoT (4 samples : Second) (*1) 90.5 -

Few-Shot-CoT (8 samples) 93.0 48.7

Zero-Plus-Few-Shot-CoT (8 samples) (*2) 92.8 51.5

Finetuned GPT-3 175B [Wei et al., 2022] - 33

Finetuned GPT-3 175B + verifier [Wei et al., 2022] - 55

PaLM 540B: Zero-Shot 25.5 12.5

PaLM 540B: Zero-Shot-CoT 66.1 43.0

PaLM 540B: Zero-Shot-CoT + self consistency 89.0 70.1

PaLM 540B: Few-Shot [Wei et al., 2022] - 17.9

PaLM 540B: Few-Shot-CoT [Wei et al., 2022] - 56.9

PaLM 540B: Few-Shot-CoT + self consistency [Wang et al., 2022] - 74.4 example, Zero-shot-CoT achieves score gains from 17.7% to 78.7% on MultiArith and from 10.4% to 40.7% on GSM8K. Our method gives on-par performances for the remaining two arithmetic reasoning
 tasks (SingleEq and AddSub), which is expected since they do not require multi-step reasoning.

In commonsense reasoning tasks, Zero-shot-CoT does not provide performance gains. It is expected as Wei et al. [2022] also reports that even Few-shot-CoT does not provide performance gains on

Lambda (135B), but does improve StrategyQA when combined with substantially larger PaLM (540B) model, which may also apply for ours. More importantly, we observe that many generated chain of thought themselves are surprisingly logically correct or only contains human-understandable mistakes (See Table 3), suggesting that Zero-shot-CoT does elicit for better commonsense reasoning even when the task metrics do not directly reflect it. We provide samples generated by Zero-shot-CoT for each dataset in Appendix B. 6 0 20 40 60 80

0.3B 1.3B 6.7B 175B
Zero-shot Zero-shot-CoT (a) MultiArith on Original GPT-3 0 20 40 60 80

S M L XL

Zero-shot Zero-shot-CoT (b) MultiArith on Instruct GPT-3 0 10 20 30 40 50 8B 62B 540B

Zero-shot Zero-shot-CoT (c) GMS8K on PaLM

Figure 3: Model scale study with various types of models. S: text-ada-001, M: text-babbage-001, L: text-curie-001, XL: text-davinci-002. See Appendix A.3 and E for the detail.

Table 3: Examples generated by Zero-Shot-CoT on CommonsenseQA for Error Analysis.

Example 1 (Reasoning is correct.) Example 2 (Model cannot choose only one.)

Q: Where is a well used toy car likely to be found?

Answer Choices: (A) child’s room (B) boy’s bedroom (C) own home (D) toy store (E) house

Q: What would be necessary for getting in shape?

Answer Choices: (A) good health (B) exercise (C) muscle tone (D) sweat (E) feel better

A: Let’s think step by step. A toy car is likely to be found in a child’s room. A child’s room is likely to be found in a house. Therefore, a toy car is likely to be found in a house.

Therefore, among A through E, the answer is E. ✗ (GT : A, Zero-shot: A)

A: Let’s think step by step. In order to get in shape, you need to be able to do three things:

1. Exercise regularly
2. Eat a healthy diet
3. Get enough sleep
If you can do all three of those things, then you will be well on your way to getting in shape! Therefore, among A through E, the answer is B, C, and D. (GT : B, Zero-shot: B)

Comparison with other baselines Table 2 compares the performances on two arithmetic reasoning benchmarks (MultiArith and GSM8K) across Zero-shot-CoT and baselines. The large gap between standard prompting (1st block) and chain of thought prompting (2nd block) suggests that these tasks are difficult without eliciting multi-step reasoning. Major improvements are confirmed on both Instruct GPT-3 (text-davinci-002) and PaLM (540B) models (4th block). While Zero-shot-CoT naturally underperforms Few-shot-CoT, it substantially outperforms standard Few-shot prompting with even 8 examples per task. For GSM8K, Zero-shot-CoT with Instruct GPT-3 (text-davinci-002) also outperforms finetuned GPT-3 and standard few-shot prompting with large models (PaLM, 540B), reported in Wei et al. [2022] (3rd and 4th block). See App. D for more experiment results with PaLM.

Does model size matter for zero-shot reasoning? Figure 3 compares performance of various language models on MultiArith / GSM8K. Without chain of thought reasoning, the performance does not increase or increases slowly as the model scale is increased, i.e., the curve is mostly flat. In contrast, the performance drastically increases with chain of thought reasoning, as the model size gets bigger, for Original/Instruct GPT-3 and PaLM. When the model size is smaller, chain of thought reasoning is not effective. This result aligns with the few-shot experiment results in Wei et al. [2022].

Appendix E shows extensive experiment results using wider variety of language models, including

GPT-2, GPT-Neo, GPT-J, T0, and OPT. We also manually investigated the quality of generated chain of thought, and large-scale models clearly demonstrate better reasoning (See Appendix B for the sampled outputs for each model).

Error Analysis To better understand the behavior of Zero-shot-CoT, we manually investigated randomly selected examples generated by Instruct-GPT3 with Zero-shot-CoT prompting. See Appendix C for examples, where some of the observations include: (1) In commonsense reasoning (CommonsenseQA), Zero-shot-CoT often produces flexible and reasonable chain of thought even when the final prediction is not correct. Zero-shot-CoT often output multiple answer choices when the model find it is difficult to narrow it down to one (see Table 3 for examples). (2) In arithmetic 7

Table 4: Robustness study against template measured on the MultiArith dataset with text-davinci-002. (*1) This template is used in Ahn et al. [2022] where a language model is prompted to generate step-by-step actions given a high-level instruction for controlling robotic actions. (*2) This template is used in Reynolds and McDonell [2021] but is not quantitatively evaluated.

No. Category Template Accuracy 1 instructive Let’s think step by step. 78.7 2 First, (*1) 77.3 3 Let’s think about this logically. 74.5 4 Let’s solve this problem by splitting it into steps. (*2) 72.2 5 Let’s be realistic and think step by step. 70.8 6 Let’s think like a detective step by step. 70.3 7 Let’s think 57.5 8 Before we dive into the answer, 55.7 9 The answer is after the proof. 45.7 10 misleading Don’t think. Just feel. 18.8 11 Let’s think step by step but reach an incorrect answer. 18.7 12 Let’s count the number of "a" in the question. 16.7 13 By using the fact that the earth is round, 9.3 14 irrelevant By the way, I found a good restaurant nearby. 17.5 15 Abrakadabra! 15.5 16 It’s a beautiful day. 13.1 - (Zero-shot) 17.7

Table 5: Robustness study of Few-shot-CoT against examples. When the examples are from entirely different tasks, the performance generally becomes worse, but when the answer formats are matched (i.e. CommonsenseQA to AQUA-RAT, multiple-choice), the performance loss is less severe. †CommonsenseQA samples are used in this variation

Zero-shot Few-shot-CoT † Zero-shot-CoT Few-shot-CoT

AQUA-RAT 22.4 31.9 33.5 39.0

MultiArith 17.7 27.0 78.7 88.2 reasoning (MultiArith), Zero-shot-CoT and Few-shot-CoT show substantial differences regarding the error patterns. First, Zero-shot-CoT tends to output unnecessary steps of reasoning after getting the correct prediction, which results in changing the prediction to incorrect one. Zero-shot-CoT also sometimes does not start reasoning, just rephrasing the input question. In contrast, Few-shot-CoT tend to fail when generated chain of thought include ternary operation, e.g. (3 + 2) ∗ 4.

How does prompt selection affect Zero-shot-CoT? We validate the robustness of Zero-shot-CoT against input prompts. Table 4 summarizes performance using 16 different templates with three categories. Specifically, following Webson and Pavlick [2022], the categories include instructive (encourage reasoning), misleading (discourage reasoning or encouraging reasoning but in a wrong way), and irrelevant (nothing to do with reasoning). The results indicate that the performance is improved if the text is written in a way that encourages chain of thought reasoning, i.e., the templates are within "instructive" category. However, the difference in accuracy is significant depending on the sentence. In this experiment, "Let’s think step by step." achieves the best results. Interestingly, it is found that different templates encourage the model to express reasoning quite differently (see

Appendix B for sample outputs by each template). In contrast, when we use misleading or irrelevant templates, the performance does not improve. It remains an open question how to automatically create better templates for Zero-shot-CoT.

How does prompt selection affect Few-shot-CoT? Table 5 shows the performance of Fewshot-CoT when using examples from different datasets: CommonsenseQA to AQUA-RAT and

CommonsenseQA to MultiArith. The domains are different in both cases, but the answer format 8 is the same in the former. Surprisingly, the chain of thought examples from different domains (common sense to arithmetic) but with the same answer (multiple-choice) format provide substantial performance gain over Zero-shot (to AQUA-RAT), measured relative to the possible improvements from Zero-shot-CoT or Few-shot-CoT. In contrast, the performance gain becomes much less when using examples with different answer types (to MultiArith), confirming prior work [Min et al., 2022] that suggests LLMs mostly leverage the few-shot examples to infer the repeated format rather than the task itself in-context. Nevertheless, for both cases the results are worse than Zero-shot-CoT, affirming the importance of task-specific sample engineering in Few-shot-CoT. 

## 5 Discussion and Related Work
Table 6: Summary of related work on arithmetic/commonsense reasoning tasks. Category denotes the training strategy. CoT denotes whether to output chain of thought. Task column lists the tasks that are performed in corresponding papers. AR: Arithmetic Reasoning, CR: Commonsense Reasoning.
表6：算术/常识推理任务的相关工作摘要。类别表示培训策略。CoT表示是否输出思想链。任务列列出了在相应论文中执行的任务。AR：算术推理，CR：常识推理。

### Reasoning Ability of LLMs 推理能力
Several studies have shown that pre-trained models usually are not good at reasoning [Brown et al., 2020, Smith et al., 2022, Rae et al., 2021], but its ability can be substantially increased by making them produce step-by-step reasoning, either by fine-tuning [Rajani et al., 2019, Cobbe et al., 2021, Zelikman et al., 2022, Nye et al., 2022] or few-shot prompting [Wei et al., 2022, Wang et al., 2022, Chowdhery et al., 2022] (See Table 6 for summary of related work). Unlike most prior work, we focus on zero-shot prompting and show that a single fixed trigger prompt substantially increases the zero-shot reasoning ability of LLMs across a variety of tasks requiring complex multi-hop thinking (Table 1), especially when the model is scaled up (Figure 3). It also generates reasonable and understandable chain of thought across diverse tasks (Appendix B), even when the final prediction is wrong (Appendix C). Similar to our work, Reynolds and McDonell [2021] demonstrate a prompt, “Let’s solve this problem by splitting it into steps”, would facilitate the multi-step reasoning in a simple arithmetic problem. However, they treated it as a task-specific example and did not evaluate quantitatively on diverse reasoning tasks against baselines. Shwartz et al. [2020] propose to decompose a commonsense question into a series of information seeking question, such as “what is the definition of [X]”. It does not require demonstrations but requires substantial manual prompt engineering per each reasoning task. Our results strongly suggest that LLMs are decent zero-shot reasoners, while prior work [Wei et al., 2022] often emphasize only few-shot learning and task-specific in-context learning, e.g. no zero-shot baselines were reported. Our method does not require time-consuming fine-tuning or expensive sample engineering, and can be combined with any pre-trained LLM, serving as the strongest zero-shot baseline for all reasoning tasks.

一些研究表明，预训练模型通常不擅长推理[Brownet al., 2020，Smithet al., 2022，Raeet al., 2021]，但通过使其产生逐步推理，其能力可以大幅提高，通过微调[Rajaniet al., 2019，Cobbeet al., 2021，Zelikmanet al., 2022，Nyeet al., 2022]或少量提示[Weiet al., 2022.Wanget al., 2022,Chowdheryet al., 2022]（相关工作总结见表6）。与大多数先前的工作不同，我们侧重于零样本提示，并表明在需要复杂多跳思维的各种任务中，特别是在模型放大时，单个固定触发提示大大提高了LLM的零样本推理能力（表1）（图3）。即使最终预测是错误的（附录C），它也会在不同的任务中产生合理和可理解的思想链（附录B）。与我们的工作类似，Reynolds和McDonell（2021）证明了一个提示，“让我们通过将问题分解为多个步骤来解决这个问题”，这将有助于在简单的算术问题中进行多步骤推理。然而，他们将其视为一个特定任务的例子，并没有根据基线对不同的推理任务进行定量评估。 Shwartz et al. [2020]提出将一个常识性问题分解为一系列寻求信息的问题，例如“[X]的定义是什么”。它不需要演示，但每个推理任务都需要大量的手动提示工程。我们的结果强烈表明，LLM是一种得体的零样本推理机，而之前的工作[Weiet al., 2022]通常只强调少量快照学习和特定任务的上下文内学习，例如，没有报告零样本基线。我们的方法不需要耗时的微调或昂贵的样本工程，并且可以与任何预训练LLM相结合，为所有推理任务提供最强的零样本基线。


### Zero-shot Abilities of LLMs 
Radford et al. [2019] show that LLMs have excellent zero-shot abilities in many system-1 tasks, including reading comprehension, translation, and summarization. Sanh et al. [2022], Ouyang et al. [2022] show that such zero-shot abilities of LLMs can be increased by explicitly fine-tuning models to follow instructions. Although these work focus on the zero-shot performances of LLMs, we focus on many system-2 tasks beyond system-1 tasks, considered a grand challenge for LLMs given flat scaling curves. In addition, Zero-shot-CoT is orthogonal to instruction tuning; it increases zero-shot performance for Instruct GPT3, vanilla GPT3, and PaLM (See Figure 3).

Radford et al. [2019]的《LLM的零样本能力》表明，LLM在阅读理解、翻译和总结等许多系统1任务中具有出色的零样本能力。Sanh et al. [2022]、Ouyang et al. [2022]表明，通过显式微调模型以遵循指令，可以提高LLM的这种零样本能力。尽管这些工作重点关注LLM的零样本性能，但我们关注的是系统1任务之外的许多系统2任务，考虑到平坦的缩放曲线，这对LLM来说是一个巨大的挑战。此外，Zero-shot-CoT与指令调整正交; 它提高了Instruct GPT3、vanilla GPT3和PaLM的零样本性能（见图3）。

5Nye et al. [2022] also evaluates few-shot settings, but the few-shot performances on their domains are worse than the fine-tuning results. 
5Nye et al. [2022]也评估了少样本设置，但其域上的少样本性能比微调结果更差。


### From Narrow (task-specific) to Broad (multi-task) Prompting 从窄提示（特定于任务）到宽提示（多任务）
Most prompts are task-specific. While few-shot prompts are naturally so due to task-specific in-context samples [Brown et al., 2020, Wei et al., 2022], majority of zero-shot prompts have also focused on per-task engineering (of templates) [Liu et al., 2021b, Reynolds and McDonell, 2021]. Borrowing terminologies from Chollet [2019] which builds on hierarchical models of intelligence [McGrew, 2005, Johnson and Bouchard Jr, 2005], these prompts are arguably eliciting “narrow generalization” or task-specific skills from LLMs. On the other hand, our method is a multi-task prompt and elicits “broad generalization” or broad cognitive abilities in LLMs, such as logical reasoning or system-2 itself. We hope our work can serve as a reference for accelerating not just logical reasoning research with LLMs, but also discovery of other broad cognitive capabilities within LLMs.

大多数提示都是特定于任务的。虽然由于特定于任务的上下文内样本，很少会出现这样的提示[Brownet al., 2020，Weiet al., 2022]，但大多数零样本提示也专注于（模板的）每任务工程[Liuet al., 2021b，Reynolds and McDonell，2021]。借用Chollet[199]中的术语，该术语建立在智力的层次模型[MMcGrew，2005，Johnson和Bouchard Jr，2005]的基础上，这些提示可以说是从LLM中引出了“狭义概括”或特定任务的技能。另一方面，我们的方法是多任务提示，并在LLM中引发“广泛的泛化”或广泛的认知能力，如逻辑推理或系统2本身。我们希望我们的工作不仅可以作为加速LLM逻辑推理研究的参考，也可以作为发现LLM中其他广泛认知能力的参考。

### Training Dataset Details 训练数据集详情
A limitation of the work is the lack of public information on the details of training datasets used for LLMs, e.g. 001 vs 002 for GPT models, original GPT3 vs InstructGPT [Ouyang et al., 2022], and data for PaLM models [Chowdhery et al., 2022]. However, big performance increases from Zero-shot to Zero-shot-CoT in all recent large models (InstructGPT 001 or 002, Original GPT3, and PaLM) and consistent improvements in both arithmetic and nonarithmetic tasks suggest that the models are unlikely simply memorizing, but instead capturing a task-agnostic multi-step reasoning capability for generic problem solving. While most results are based on InstructGPT since it is the best performing open-access LLM, key results are reproduced on PaLM, and dataset details in InstructGPT (Appendix A, B, and F in Ouyang et al. [2022]) also confirm that it is not specially engineered for multi-step reasoning.

这项工作的局限性是缺乏关于LLM所用训练数据集细节的公开信息，例如GPT模型的001与002、原始GPT3与InstructGPT[Ouyang et al.，2022]以及PaLM模型的数据[Chowdhery et al.，2022]。然而，在所有最近的大型模型（InstructionGPT 001或002、Original GPT3和PaLM）中，从零样本到零快照CoT的性能大幅提高，并且算术和非算术任务的持续改进表明，这些模型不太可能只是简单地记忆，而是捕获任务认知多步骤推理能力来解决一般问题。虽然大多数结果都是基于InstructGPT的，因为它是性能最好的开放访问LLM，但关键结果是在PaLM上复制的，InstructGPT中的数据集细节（Ouyang et al. [2022]中的附录A、B和F）也证实了它不是专为多步推理而设计的。

### Limitation and Social Impact  限制和社会影响
Our work is based on prompting methods for large language models. LLMs have been trained on large corpora from various sources on the web (also see “Training Dataset Details”), and have shown to capture and amplify biases found in the training data. Prompting is a method that looks to take advantage of the patterns captured by language models conducive to various tasks, and therefore it has the same shortcomings. This being said, our approach is a more direct way to probe complex reasoning inside pre-trained LLMs, removing the confounding factor of in-context learning in prior few-shot approaches, and can lead to more unbiased study of biases in LLMs. 

我们的工作是基于大型语言模型的提示方法。LLM已经在来自网络上各种来源的大型语料库上进行了训练（另请参阅“训练数据集详情”），并已证明可以捕捉和放大训练数据中发现的偏见。提示是一种试图利用有助于各种任务的语言模型所捕获的模式的方法，因此它也有同样的缺点。话虽如此，我们的方法是在预先训练的LLM中探索复杂推理的一种更直接的方法，消除了先前少数样本方法中上下文学习的混杂因素，并可以导致对LLM中的偏见进行更公正的研究。

## 6 Conclusion
We have proposed Zero-shot-CoT, a single zero-shot prompt that elicits chain of thought from large language models across a variety of reasoning tasks, in contrast to the few-shot (in-context) approach in previous work that requires hand-crafting few-shot examples per task. Our simple method not only is the minimalist and strongest zero-shot baseline for difficult multi-step system-2 reasoning tasks that long evaded the scaling laws of LLMs, but also encourages the community to further discover similar multi-task prompts that elicit broad cognitive abilities instead of narrow task-specific skills.

我们已经提出了Zero-shot-CoT，这是一个单一的零样本提示，它可以在各种推理任务中从大型语言模型中引出思维链，而之前的工作中使用的少快照（上下文中）方法需要手工绘制每个任务的少快照样本。我们的简单方法不仅是长期回避LLM缩放定律的困难多步骤系统2推理任务的最低限度和最强零样本基线，而且还鼓励社区进一步发现类似的多任务提示，这些提示可以激发广泛的认知能力，而不是狭隘的任务特定技能。

## Acknowledgements
This work has been supported by the Mohammed bin Salman Center for Future Science and Technology for Saudi-Japan Vision 2030 at The University of Tokyo (MbSC2030). Computational resource of AI Bridging Cloud Infrastructure (ABCI) provided by National Institute of Advanced Industrial Science and Technology (AIST) was used for experiments other than PaLM. We also thank Jason Wei and Denny Zhou for discussions and support on running PaLM experiments, and Sharan Narang and Aakanksha Chowdhery for generic support on PaLM infrastructures.

## References
* Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, ChelseaFinn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, JulianIbarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth,Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine,Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, JarekRettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev,Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, and Mengyuan Yan. Do as i can,not as i say: Grounding language in robotic affordances, 2022. URL https://arxiv.org/abs/2204.01691.
* Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large ScaleAutoregressive Language Modeling with Mesh-Tensorflow, March 2021. URL https://doi.org/10.5281/zenodo.5297715.
* Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, ArielHerbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler,Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray,Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever,and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato,R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in NeurIPS, volume 33, pages 1877–1901.
* Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.
* François Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019. URLhttps://arxiv.org/abs/1911.01547.
* Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, AdamRoberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh,Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, NoamShazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, JamesBradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Lev￾skaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, KevinRobinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph,Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M.
* Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, RewonChild, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, MarkDiaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean,Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. URLhttps://arxiv.org/abs/2204.02311.
* Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and JohnSchulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.
* Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training ofdeep bidirectional transformers for language understanding. In Proceedings of NAACL, pages4171–4186, 2019. URL https://aclanthology.org/N19-1423.
* Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang,Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gbdataset of diverse text for language modeling. arXiv preprint arXiv: Arxiv-2101.00027, 2020.
* Tianyu Gao, Adam Fisch, and Danqi Chen. Making pre-trained language models better few-shot learn￾ers. In Proceedings of ACL-IJCNLP, pages 3816–3830, 2021. URL https://aclanthology.
* org/2021.acl-long.295.
* Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotleuse a laptop? a question answering benchmark with implicit reasoning strategies. TACL, 9:346–361,2021. URL https://aclanthology.org/2021.tacl-1.21/.
* Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning tosolve arithmetic word problems with verb categorization. In EMNLP, volume 523533. Citeseer,2014. URL https://aclanthology.org/D14-1058/.
* Wendy Johnson and Thomas J Bouchard Jr. The structure of human intelligence: It is verbal,perceptual, and image rotation (vpr), not fluid and crystallized. Intelligence, 33(4):393–416, 2005.
* Rik Koncel-Kedziorski, Hannaneh Hajishirzi, Ashish Sabharwal, Oren Etzioni, and Siena DumasAng. Parsing algebraic word problems into equations. TACL, 3:585–597, 2015. URL https://aclanthology.org/Q15-1042.
* Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS:A math word problem repository. In Proceedings of NAACL, pages 1152–1157, 2016. URLhttps://aclanthology.org/N16-1136.
* Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale gen￾eration: Learning to solve and explain algebraic word problems. In Proceedings of ACL, pages158–167, 2017. URL https://aclanthology.org/P17-1015.
* Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. Whatmakes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804, 2021a. URLhttps://arxiv.org/abs/2101.06804.
* Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig.
* Pre-train, prompt, and predict: A systematic survey of prompting methods in natural languageprocessing. arXiv preprint arXiv:2107.13586, 2021b. URL https://arxiv.org/abs/2107.13586.
* Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically orderedprompts and where to find them: Overcoming few-shot prompt order sensitivity. In Proceedings ofACL, pages 8086–8098, 2022. URL https://aclanthology.org/2022.acl-long.556.
* Kevin S McGrew. The cattell-horn-carroll theory of cognitive abilities: Past, present, and future.2005.
* Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixturemodels. arXiv preprint arXiv: Arxiv-1609.07843, 2016. URL https://arxiv.org/abs/1609.07843.
* Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and LukeZettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? arXivpreprint arXiv:2202.12837, 2022. URL https://arxiv.org/pdf/2202.12837.pdf.
* Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, DavidBieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, andAugustus Odena. Show your work: Scratchpads for intermediate computation with languagemodels. In Deep Learning for Code Workshop, 2022. URL https://openreview.net/forum?id=HBlx2idbkbq.
* Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, ChongZhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton,Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, andRyan Lowe. Training language models to follow instructions with human feedback, 2022. URLhttps://arxiv.org/abs/2203.02155.
* Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, GregoryChanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Py￾torch: An imperative style, high-performance deep learning library. Advances inNeurIPS, 32:8026–8037, 2019. URL https://papers.nips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html.
* Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simplemath word problems? In Proceedings of NAACL, pages 2080–2094, 2021. URL https://aclanthology.org/2021.naacl-main.168.
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Languagemodels are unsupervised multitask learners. OpenAI blog, page 9, 2019. URL http://www.
* persagen.com/files/misc/radford2019language.pdf.
* Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, JohnAslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan,Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks,Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, SaffronHuang, Jonathan Uesato, John Mellor, Irina Higgins, Antonia Creswell, Nat McAleese, Amy Wu,Erich Elsen, Siddhant Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, KarenSimonyan, Michela Paganini, Laurent Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro,Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch,Jean-Baptiste Lespiau, Maria Tsimpoukelli, Nikolai Grigorev, Doug Fritz, Thibault Sottiaux,Mantas Pajarskas, Toby Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume,Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas,Aurelia Guy, Chris Jones, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger,Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, OriolVinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu,and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher,2021. URL https://arxiv.org/abs/2112.11446.
* Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, YanqiZhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-texttransformer. JMLR, 21(140):1–67, 2020. URL http://jmlr.org/papers/v21/20-074.html.
* Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. Explain yourself!leveraging language models for commonsense reasoning. In Proceedings of ACL, pages 4932–4942,2019. URL https://aclanthology.org/P19-1487.
* Laria Reynolds and Kyle McDonell. Prompt programming for large language models: Beyond thefew-shot paradigm. In Extended Abstracts of the 2021 CHI Conference on Human Factors inComputing Systems, pages 1–7, 2021. URL https://arxiv.org/pdf/2102.07350.pdf.
* Subhro Roy and Dan Roth. Solving general arithmetic word problems. In Proceedings of EMNLP,pages 1743–1752, 2015. URL https://aclanthology.org/D15-1202.
* Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, An￾toine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, UrmishThakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, NihalNayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Man￾ica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, TrishalaNeeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, RyanTeehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush.
* Multitask prompted training enables zero-shot task generalization. In ICLR, 2022. URLhttps://openreview.net/forum?id=9Vrb9D0WI4.
* Timo Schick and Hinrich Schütze. It’s not just size that matters: Small language models are also few￾shot learners. In Proceedings of NAACL, pages 2339–2352, 2021. URL https://aclanthology.
* org/2021.naacl-main.185.
* Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. Auto￾Prompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts.
* In Proceedings of EMNLP, pages 4222–4235, 2020. URL https://aclanthology.org/2020.emnlp-main.346.
* Vered Shwartz, Peter West, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Unsupervisedcommonsense question answering with self-talk. In Proceedings of EMNLP, pages 4615–4629,2020. URL https://aclanthology.org/2020.emnlp-main.373.
* Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, JaredCasper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, Elton Zhang, RewonChild, Reza Yazdani Aminabadi, Julie Bernauer, Xia Song, Mohammad Shoeybi, Yuxiong He,Michael Houston, Saurabh Tiwary, and Bryan Catanzaro. Using deepspeed and megatron totrain megatron-turing nlg 530b, a large-scale generative language model, 2022. URL https://arxiv.org/abs/2201.11990.
* Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, AdamFisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond theimitation game: Quantifying and extrapolating the capabilities of language models. arXiv preprintarXiv:2206.04615, 2022. URL https://arxiv.org/abs/2206.04615.
* Keith E Stanovich and Richard F West. Individual differences in reasoning: Implications for therationality debate? Behavioral and brain sciences, 23(5):645–665, 2000.
* Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A questionanswering challenge targeting commonsense knowledge. In Proceedings of NAACL-HLT, pages4149–4158, 2019. URL https://aclanthology.org/N19-1421/.
* Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-TzeCheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu StevenZheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin,James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, VincentZhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, PraneshSrinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi,Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran,Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee,Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton,Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak,Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022. URL https://arxiv.org/abs/2201.08239.
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. VonLuxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Ad￾vances in NeurIPS, 2017. URL https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.
* Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model.https://github.com/kingoflolz/mesh-transformer-jax, May 2021.
* Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. Self-consistencyimproves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.URL https://arxiv.org/abs/2203.11171.
* Albert Webson and Ellie Pavlick. Do prompt-based models really understand the meaning of theirprompts? In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2300–2344. Association forComputational Linguistics, July 2022. URL https://aclanthology.org/2022.naacl-main.167.
* Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models, 2022. URL https://arxiv.org/abs/2201.11903.
* Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi,Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrickvon Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger,Mariama Drame, Quentin Lhoest, and Alexander Rush. Transformers: State-of-the-art naturallanguage processing. In Proceedings of EMNLP, 2020. URL https://aclanthology.org/2020.emnlp-demos.6.
* Eric Zelikman, Yuhuai Wu, and Noah D. Goodman. Star: Bootstrapping reasoning with reasoning,2022. URL https://arxiv.org/abs/2203.14465.
* Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, ChristopherDewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer languagemodels. arXiv preprint arXiv:2205.01068, 2022. URL https://arxiv.org/abs/2205.01068.

## Checklist
1. For all authors...
(a) Do the main claims made in the abstract and introduction accurately reflect the paper’s
contributions and scope? [Yes]
(b) Did you describe the limitations of your work? [Yes]
(c) Did you discuss any potential negative societal impacts of your work? [Yes]
(d) Have you read the ethics review guidelines and ensured that your paper conforms to
them? [Yes]
2. If you are including theoretical results...
(a) Did you state the full set of assumptions of all theoretical results? [N/A]
(b) Did you include complete proofs of all theoretical results? [N/A]
3. If you ran experiments...
(a) Did you include the code, data, and instructions needed to reproduce the main experi￾mental results (either in the supplemental material or as a URL)? [Yes]
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they
were chosen)? [Yes]
(c) Did you report error bars (e.g., with respect to the random seed after running experi￾ments multiple times)? [No] Our paper mainly used GPT-3 API with greedy decoding,
and there are no randomness for the experiments.
(d) Did you include the total amount of compute and the type of resources used (e.g., type
of GPUs, internal cluster, or cloud provider)? [Yes]
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...
(a) If your work uses existing assets, did you cite the creators? [Yes]
(b) Did you mention the license of the assets? [Yes]
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes]
(d) Did you discuss whether and how consent was obtained from people whose data you’re
using/curating? [Yes]
(e) Did you discuss whether the data you are using/curating contains personally identifiable
information or offensive content? [Yes]
5. If you used crowdsourcing or conducted research with human subjects...
(a) Did you include the full text of instructions given to participants and screenshots, if
applicable? [N/A]
(b) Did you describe any potential participant risks, with links to Institutional Review
Board (IRB) approvals, if applicable? [N/A]
(c) Did you include the estimated hourly wage paid to participants and the total amount
spent on participant compensation? [N/A]

## A Details of Experimental Setup

### A.1 Code
Code is available at https://github.com/kojima-takeshi188/zero_shot_cot.

### A.2 Datasets
A.2.1 Dataset Description

Table 7 summarizes the description of each dataset used in our experiment.

Table 7: Dataset Description. Our experiments used publicly available datasets except for “Last

Letters” and “Coin Flip” datasets. We created these two datasets. See Appendix A.2.2 for the details. (*1) N : Number, M : Pick up one from multiple choices, Y : Answer Yes or No, F : Free Format. (*2) Average number of words in questions texts.

Dataset Answer

Format (*1)

of
 samples

Avg # words (*2)

Data split (filename) used for our experiment

License

SingleEq N 508 27.4 questions.json No License

AddSub N 395 31.5 AddSub.json Unspecified

MultiArith N 600 31.8 MultiArith.json Unspecified

GSM8K N 1319 46.9 test.jsonl MIT License

AQUA-RAT M 254 51.9 test.jsonl Apache-2.0

SVAMP N 1000 31.8 SVAMP.json MIT License

CommonsenseQA M 1221 27.8 dev_rand_split.jsonl Unspecified

StrategyQA Y 2290 9.6 task.json Apache-2.0

Date Understanding M 369 35.0 task.json Apache-2.0

Shuffled Objects M 750 91.1 three_objects/task.json Apache-2.0

Last Letters F 500 15.0 - -

Coin Flip Y 500 37.0 - -

A.2.2 Dataset creation

Regarding “Last Letter Concatenation” and “Coin Flip”, datasets are not publicly available so we created the datasets following Wei et al. [2022] with a minor rephrasing of the question template.

Specifically, as for Last Letter Concatenation, we use the following template. We randomly select human names from names-dataset library (https://pypi.org/project/names-dataset/) and insert them into {Name1} through {Name4}. 
* ’Take the last letters of each words in "{Name1} {Name2} {Name3} {Name4}" and concatenate them.’

As for Coin Flip, we use the following template. We randomly select human names from namesdataset library and insert them into {Name1} through {Name4}. We also randomly pick up “flips” or “does not flip” and insert the phrase into each {flips | does not flip} part, respectively. 
* ’A coin is heads up. {Name1} {flips | does not flip} the coin. {Name2} {flips | does not flip} the coin. {Name3} {flips | does not flip} the coin. {Name4} {flips | does not flip} the coin.

Is the coin still heads up? Note that "flip" here means "reverse".’

### A.3 Language Models
Our experiment uses multiple language models as described at Table 8

### A.4 Implementation details
For Original GPT-3 and Instruct-GPT3, we used OpenAI API. For OPT, T0, GPT-J, GPT-Neo, and

GPT-2, we used Hugging Face Transformer Library [Wolf et al., 2020]. We set max_tokens = 128 and 16

Table 8: Description of language models. (*1) As for Original GPT3 models, we assign model size information to each model by referring to https://blog.eleuther.ai/gpt3-model-sizes/ and https://beta.openai.com/docs/model-index-for-researchers. (*2) There is no official information about the model size of Instruct GPT3. We infer from the API name that the order of model size of Instruct GPT3 matches that of Original GPT3.

Language Model # of params Library / API Name Model Name in Library / API License

PaLM 540B - - unspecified

PaLM 62B - - unspecified

PaLM 8B - - unspecified

Original GPT3 175B (*1) OpenAI API davinci unspecified

Original GPT3 6.7B (*1) OpenAI API curie unspecified

Original GPT3 1.3B (*1) OpenAI API babbage unspecified

Original GPT3 0.3B (*1) OpenAI API ada unspecified

Instruct GPT3 - (*2) OpenAI API text-davinci-002 unspecified

Instruct GPT3 - (*2) OpenAI API text-davinci-001 unspecified

Instruct GPT3 - (*2) OpenAI API text-curie-001 unspecified

Instruct GPT3 - (*2) OpenAI API text-babbage-001 unspecified

Instruct GPT3 - (*2) OpenAI API text-ada-001 unspecified

OPT 13B Hugging Face Library opt-13b Apache-2.0

T0 11B Hugging Face Library T0pp Apache-2.0

GPT-J 6B Hugging Face Library gptj Apache-2.0

GPT-Neo 2.7B Hugging Face Library gpt-neo Apache-2.0

GPT-2 1.5B Hugging Face Library gpt2-xl Apache-2.0 used greedy decoding (temperature = 0 in the case of OpenAI API) across all the methods and models except PaLM. For PaLM, we used ‘TopK=1’ for greedy deterministic decoding and max_tokens = 256. “Q:” is set as a customized stop sequence for all the models except for Instruct-GPT3 to stop the
 models from repeating questions and answers by themselves. We run our experiments on cloud V100 instances without GPU for GPT-3 models, on cloud A100x8 GPU(60GB) instances for T0 and OTP, and on cloud A100x1 GPU(60GB) instances for GPT-J, GPT-Neo, and GPT-2. Our implementation is in PyTorch [Paszke et al., 2019].

### A.5 Prompts For Answer Extraction
Table 9 and Table 10 summarizes a list of answer extraction prompts used for the experiments at

Table 1. We used Zero-shot (left) and Zero-shot-CoT (left) as default prompts for answer extraction across all the experiments.

Table 9: Answer extraction prompts used for Zero-shot experiments in Table 1. C.S.QA : CommonsenseQA, D.U. : Date Understanding, S.O. : Tracking Shuffled Objects

No Task Zero-Shot (left) Zero-Shot (right) 1 SingleEq The answer (arabic numerals) is The answer is 2 AddSub The answer (arabic numerals) is The answer is 3 MultiArith The answer (arabic numerals) is The answer is 4 GSM8K The answer (arabic numerals) is The answer is 5 AQUA-RAT Among A through E, the answer is The answer is 6 SVAMP The answer (arabic numerals) is The answer is 7 C.S.QA Among A through E, the answer is The answer is 8 StrategyQA The answer (Yes or No) is The answer is 9 D.U. Among A through F, the answer is The answer is 10 S.O. Among A through C, the answer is The answer is 11 Last Letters The answer is The answer is 12 Coin Flip The answer (Yes or No) is The answer is 17

Table 10: Answer extraction prompts used for Zero-shot-CoT experiments in Table 1. C.S.QA :

CommonsenseQA, D.U. : Date Understanding, S.O. : Tracking Shuffled Objects

No Task Zero-Shot-CoT (left) Zero-Shot-CoT (right) 1 SingleEq Therefore, the answer (arabic numerals) is Therefore, the answer is 2 AddSub Therefore, the answer (arabic numerals) is Therefore, the answer is 3 MultiArith Therefore, the answer (arabic numerals) is Therefore, the answer is 4 GSM8K Therefore, the answer (arabic numerals) is Therefore, the answer is 5 AQUA-RAT Therefore, among A through E, the answer is Therefore, the answer is 6 SVAMP Therefore, the answer (arabic numerals) is Therefore, the answer is 7 C.S.QA Therefore, among A through E, the answer is Therefore, the answer is 8 StrategyQA Therefore, the answer (Yes or No) is Therefore, the answer is 9 D.U. Therefore, among A through F, the answer is Therefore, the answer is 10 S.O. Therefore, among A through C, the answer is Therefore, the answer is 11 Last Letters Therefore, the answer is Therefore, the answer is 12 Coin Flip Therefore, the answer (Yes or No) is Therefore, the answer is

### A.6 Answer Cleansing
Table 11 summarizes a list of answer cleansing approaches used across all the experiments.

Table 11: Detail description of answer cleansing. See Table 7 for the mapping between each datasets and the corresponding answer formats.

Answer

Format

Answer Cleansing

Approach

Pseudo Code (Example in Pytorch 3.8)

Number Pick up the first number encountered in the text. pred = pred . replace (",", "") pred = [s for s in re . findall (r’ -?\ d +\.?\ d*’, pred )] pred = pred [0]

MultipleChoice

Pick up the first large letter encountered in the text. pred = re . findall (r’A|B|C|D|E’, pred ) pred = pred [0]

Yes or No Pick up the first "yes" or "no" encountered in the text after removing unnecessary letters. pred = pred . lower () pred = re . sub (" \"|\ ’|\ n |\.|\ s |\:|\ , ","␣", pred ) pred = pred . split ("␣") pred = [i for i in pred if i in (" yes ", "no")] pred = pred [0]

Free Format Just remove unnecessary letters, such as ".". pred = re . sub (" \"|\ ’|\ n |\.|\ s","", pred )

## B Additional Experiment Results

This section summarizes more example texts generated by models in our experiments. Note that for readability all texts are modified from the original ones by omitting or inserting some linebreaks.

Without mentioning otherwise, we use Instruct-GPT3 (text-davinci-002) model. 
* Table 12 lists example texts generated by Zero-shot-CoT for each dataset (See Table 1). 
* Table 13 lists example texts generated by Zero-shot-CoT for each reasoning extraction template (See Table 4). 
* Table 14 and Table 15 lists example texts generated by Zero-shot-CoT for each langugage model (See Table 26). 18 
* Table 16 has an example text generated by Few-shot. 
* Table 17 has an example text generated by Few-shot-CoT. 
* Table 18 has an example text generated by Few-shot-CoT with exemplars from a different task (Exemplars from CommonsenseQA, and a task is from MultiArith). 
* Table 19 has an example text generated by Zero-Plus-Few-Shot-CoT. 
* Table 20 compares different outcome scenarios on results generated by Zero-shot and

Zero-shot-CoT using PaLM (540B) model.

Table 12: Example outputs by Zero-shot-CoT for each datasets.

Dataset Question Answer

SingleEq Q: A spaceship traveled 0.5 of a light-year from Earth to

Planet X and 0.1 of a lightyear from Planet X to Planet Y.

Then it traveled 0.1 of a lightyear from Planet Y back to

Earth. How many light-years did the spaceship travel in all?

A: Let’s think step by step. The spaceship traveled 0.5 light-years from Earth to Planet X. Then it traveled 0.1 light-years from Planet X to Planet Y. Finally, it traveled 0.1 light-years from Planet
Y back to Earth. So the total distance the spaceship traveled is 0.5 + 0.1 + 0.1 = 0.7 light-years.

Therefore, the answer (arabic numerals) is: 0.7 light-years X

Q: While making desserts for a bake sale, Victor used 0.625 of a scoop of brown sugar as well as 0.25 of a scoop of white sugar. How much more brown sugar did Victor use?

A: Let’s think step by step. 0.625 of a scoop of brown sugar is the same as 0.25 of a scoop of white sugar. This means that Victor used the same amount of brown sugar as he did white sugar.

Therefore, the answer (arabic numerals) is 0. ✗(GT : 0.375)

AddSub Sandy grew 6 carrots . Sam grew 3 carrots . How many carrots did they grow in total ?

A: Let’s think step by step. Sandy grew 6 carrots. Sam grew 3 carrots. That means they grew a total of 9 carrots. Therefore, the answer (arabic numerals) is 9. X

Q: Melanie bought a Batman game for $ 6.95 , a strategy game for $ 7.90 , and a Superman game for $ 7.73 . Melanie already owns 4 games . How much did Melanie spend on video games ?

A: Let’s think step by step. Melanie bought a

Batman game for $ 6.95. So she spent $ 6.95 on video games. Then she bought a strategy game for $ 7.90. So she spent $ 14.85 on video games. Then she bought a Superman game for $ 7.73. So she spent $ 22.58 on video games. Melanie already owns 4 games. So she spent $ 18.58 on video games. Therefore, Melanie spent a total of $ 18.58 on video games. Therefore, the answer (arabic numerals) is: 18.58 ✗ (GT : 22.58)

MultiArith Q: Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy. If each box has 4 pieces inside it, how much candy did he have total?

A: Let’s think step by step. Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy.

We know that each box has 4 pieces inside it. So, we can multiply 2 by 4 to find out how many pieces of chocolate candy Adam bought. This will give us 8. We can also multiply 5 by 4 to find out how many pieces of caramel candy Adam bought. This will give us 20. Now, we can add 8 and 20 together to find out how much candy Adam bought in total. 8 + 20 = 28 Adam bought 28 pieces of candy in

Therefore, the answer (arabic numerals) is 28. X

Continued on next page 19

Table 12 – Continued from previous page

Dataset Question Answer

Q: Billy was organizing his baseball cards in a binder with 5 on each page. If he had 3 new cards and 42 old cards to put in the binder, how many pages would he use?

A: Let’s think step by step. Billy has 42 old cards.

He can put 5 old cards on each page, which means he will use 9 pages for old cards. Billy has 3 new cards. He can put 5 new cards on each page, which means he will use 1 page for new cards. So in total,

Billy will use 10 pages. Therefore, the answer (arabic numerals) is 10. ✗ (GT : 9)

GSM8K Q: James buys a plane. The plane cost $150,000. He pays $5000 a month to rent a hanger to keep it in. He also spends twice as much as that on fuel per month. How much did it cost him to get and maintain the plane for the first year?

A: Let’s think step by step. James buys a plane.

This costs him $150,000. He pays $5000 a month to rent a hanger. This costs him $60,000 for the year. He spends twice as much as that on fuel per month. This costs him $120,000 for the year.

So the total cost for the first year is $150,000 + $60,000 + $120,000 = $330,000. Therefore, the answer (arabic numerals) is 330,000. X

Q: James is counting his Pokemon cards. He has 30 fire type, 20 grass type, and 40 water type. If he loses 8 of the water type and buys 14 grass type, what’s the percentage chance (rounded to the nearest integer) that a randomly picked card will be a water type?

A: Let’s think step by step. There are 30 fire type, 20 grass type, and 40 water type. If he loses 8 of the water type, then he has 32 cards in total. If he buys 14 grass type, then he has 46 cards in total, and the number of water type cards becomes 32. The percentage chance that a randomly picked card will be a water type is $\frac{32}{46} \times 100\% = \fbox{70}\%$. Therefore, the answer (arabic numerals) is 70. ✗ (GT : 33)

AQUA-RAT Q: The capacity of a tank of dimensions (8 m × 6 m × 2.5 m) is Answer Choices: (A) 120 litres (B) 1200 litres (C) 12000 litres (D) 120000 litres (E) None of these

A: Let’s think step by step.

The first thing we need to do is calculate the volume of the tank. The volume of the tank is 8m x 6m x 2.5m. 8m x 6m x 2.5m = 120m^3

Now that we have the volume of the tank, we need to convert it to litres. 1m^3 = 1000 litres 120m^3 = 120,000 litres

The capacity of the tank is 120,000 litres.

Therefore, among A through E, the answer is D. X

Q: In a row of children Neha is 12th from left end and Radha is 6th from right end. When

Radha is shifted to left by 2 places and Neha is shifted to right by 2 places there 6 children between Radha and Neha.

How many children are there in the row? Answer Choices: (A) 23 (B) 27 (C) 26 (D) 28 (E) 29

A: Let’s think step by step.

There are 12 children to the left of Neha, so there must be at least 12 children in the row.

There are 6 children to the right of Radha, so there must be at least 18 children in the row.

There are 6 children between Radha and Neha, so there must be at least 24 children in the row.

The answer must be between 24 and 27, and the only answer that satisfies this is 26.

Therefore, the answer is 26. Therefore, among A through E, the answer is C. ✗ (GT : D)

Continued on next page 20

Table 12 – Continued from previous page

Dataset Question Answer

SVAMP Q: There are 7 baskets of peaches. Each basket has 10 red peaches and 2 green peaches. How many green peaches are in the baskets altogether?

A: Let’s think step by step. There are 7 baskets. Each basket has 10 red peaches and 2 green peaches. So, there are 2 green peaches in each basket. Now, we need to multiply because there are 7 baskets. 2 multiplied by 7 is 14. So, there are 14 green peaches in the baskets altogether.

Therefore, the answer (arabic numerals) is 14. X

Q: Mary is baking a cake. The recipe calls for 12 cups of flour 14 cups of sugar and 7 cups of salt. She already put in 2 cups of flour. How many more cups of flour than cups of salt does she need to add now?

A: Let’s think step by step. Mary has already put in 2 cups of flour. The recipe calls for a total of 12 cups of flour. That means Mary needs to add 10 more cups of flour. The recipe calls for a total of 7 cups of salt. That means Mary needs to add 5 more cups of salt. So, Mary needs to add 10 more cups of flour than she does salt. Therefore, the answer (arabic numerals) is 10. ✗ (GT : 3)

CommonsenseQA

Q: If there is a pond with trees around it, where is it likely located? Answer Choices: (A) ground (B) bathroom (C) forest (D) countryside (E) rural area

A: Let’s think step by step. A pond is likely to be located near trees because trees need water to survive. Therefore, the most likely location for a pond with trees around it is in a forest. Therefore, among A through E, the answer is C. X

Q: David watched some nesting birds using his binoculars while on vacation. Where might David be?. Answer

Choices: (A) sky (B) vaccation (C) forest (D) countryside (E) roof

A: Let’s think step by step.

First, we need to understand what the word "binoculars" means. Binoculars are two telescopes that are joined together and allow a person to see things that are far away.

Now, we need to think about where a person might use binoculars. A person might use binoculars to see things that are far away and in the sky, or to see things that are far away and on the ground.

So, the answer could be either (A) sky or (D) countryside.

Therefore, among A through E, the answer is (A) or (D). ✗ (GT : C)

StrategyQA Q: Would JPEG be a good format for saving an image of Da

Vinci’s Vitruvian Man?

A: Let’s think step by step. The JPEG format is a lossy compression format, which means that some of the image data will be lost when the image is compressed. The Vitruvian Man is a very detailed image, with lots of small details. So, when we compress the image using JPEG, some of those small details will be lost, and the image will not be as sharp and clear as the original. So, in conclusion,

JPEG is not the best format for saving an image of the Vitruvian Man. Therefore, the answer (Yes or

No) is No. X

Continued on next page 21

Table 12 – Continued from previous page

Dataset Question Answer

Q: Is Mickey Mouse hypothetically unlikely to make a purchase at Zazzle?

A: Let’s think step by step. First, we need to consider what Zazzle is. Zazzle is a website where people can create and sell custom products. This includes things like t-shirts, mugs, phone cases, and more. Next, we need to think about what

Mickey Mouse is. Mickey Mouse is a cartoon character created by Walt Disney. He is a mouse who wears red shorts, large yellow shoes, and white gloves. Now, we need to think about whether or not Mickey Mouse is likely to make a purchase at

Zazzle. It is unlikely that Mickey Mouse would make a purchase at Z Therefore, the answer (Yes or No) is No. ✗ (GT : Yes)

Date

Understand.

Q: Today is 3/5, and it is

Jane’s second time in the year 1973 to see a meteor shower.

What is the date a month ago in MM/DD/YYYY? Answer

Choices: (A) 03/05/2007 (B) 01/02/1973 (C) 03/08/1983 (D) 03/05/1973 (E) 02/05/1973 (F) 04/05/1973

A: Let’s think step by step.

The question is asking for the date a month ago from 03/05/1973. A month ago would be 02/05/1973.

Therefore, among A through F, the answer is E. X

Q: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in

MM/DD/YYYY? Answer

Choices: (A) 06/02/2021 (B) 05/31/2021 (C) 06/16/2021 (D) 05/31/1966 (E) 08/18/2021 (F) 05/31/1941

A: Let’s think step by step. The deadline is Jun 1, 2021, which is 2 days away from now. In 24 hours, it will be Jun 2, 2021. Therefore, among A through F, the answer is A. ✗ (GT : B)

Shuffled

Objects.

Q: Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a green ball, Bob has a purple ball, and

Claire has a blue ball.

As the game progresses, pairs of players trade balls. First,

Bob and Alice swap balls.

Then, Claire and Bob swap balls. Finally, Bob and Alice swap balls. At the end of the game, Bob has the

Which choice is true ? Answer

Choices: (A) green ball. (B) purple ball. (C) blue ball.

A: Let’s think step by step.

First, Bob and Alice swap balls. This means that

Alice now has the purple ball and Bob has the green ball.

Then, Claire and Bob swap balls. This means that

Claire now has the green ball and Bob has the blue ball.

Finally, Bob and Alice swap balls. This means that

Alice now has the blue ball and Bob has the purple ball.

At the end of the game, Bob has the purple ball.

Therefore, among A through C, the answer is B. X

Continued on next page 22

Table 12 – Continued from previous page

Dataset Question Answer

Q: Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Sam, Bob is dancing with Helga, and Claire is dancing with Karl. Throughout the song, the dancers often trade partners. First, Claire and Alice switch partners. Then, Bob and Alice switch partners. Finally, Claire and Bob switch partners. At the end of the dance, Alice is dancing with

Which choice is true ? Answer

Choices: (A) Sam. (B) Helga. (C) Karl.

A: Let’s think step by step.

First, Claire and Alice switch partners. This means that at the end of this step, Alice is dancing with

Karl and Claire is dancing with Sam.

Then, Bob and Alice switch partners. This means that at the end of this step, Alice is dancing with

Helga and Bob is dancing with Karl.

Finally, Claire and Bob switch partners. This means that at the end of this step, Claire is dancing with Helga and Bob is dancing with Sam. At the end of the dance, Alice is dancing with Sam.

Therefore, among A through C, the answer is A. ✗ (GT : B)

Last Letters Q: Take the last letters of each words in "Vinny Landon

Miguel Caitlyn" and concatenate them.

A: Let’s think step by step.

The last letter of "Vinny" is "y".

The last letter of "Landon" is "n".

The last letter of "Miguel" is "l".

The last letter of "Caitlyn" is "n".

So, the final answer would be "ynln".

Therefore, the answer is "ynln". X

Q: Take the last letters of each words in "Lamont Frederick

Lance Esperanza" and concatenate them.

A: Let’s think step by step.

The last letter of "Lamont" is "t".

The last letter of "Frederick" is "k".

The last letter of "Lance" is "e".

The last letter of "Esperanza" is "a".

So the final answer would be "tkaea".

Therefore, the answer is "tkaea". ✗ (GT : tkea)

Coin Flip Q: A coin is heads up. Jeff does not flip the coin. Jen flips the coin. Giselle flips the coin. Noel does not flip the coin. Is the coin still heads up?

Note that "flip" here means "reverse".

A: Let’s think step by step.

Jeff does not flip the coin, so the coin remains heads up.

Jen flips the coin, so the coin is now tails up.

Giselle flips the coin, so the coin is now heads up again.

Noel does not flip the coin, so the coin remains heads up.

Therefore, the final answer is that the coin is heads up.

Therefore, the answer (Yes or No) is Yes. X

Q: A coin is heads up. Caroline flips the coin. Demetrius flips the coin. Fidel does not flip the coin. Solomon flips the coin. Is the coin still heads up?

Note that "flip" here means "reverse".

A: Let’s think step by step.

Caroline flips the coin. The coin is now tails up.

Demetrius flips the coin. The coin is now heads up.

Fidel does not flip the coin. The coin remains tails up.

Solomon flips the coin. The coin is now heads up.

So, the answer is yes, the coin is still heads up.

Therefore, the answer (Yes or No) is: Yes ✗ (GT :

No) 23

Table 13: Example outputs by Zero-Shot and Zero-Shot-CoT with various templates for template robustness study. The number within the parenthesis corresponds to the number in Table 4.

Question Q: For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?

Zero-Shot A: The answer (arabic numerals) is 3 days. ✗

Zero-shotCoT (1)

A: Let’s think step by step.

Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. That means she has a total of 16 pieces of candy. If she only ate 8 pieces a day, she would have to divide her candy into two days. That means she would have 8 pieces of candy for Day 1 and 8 pieces of candy for Day 2.

Therefore, the answer (arabic numerals) is 2. X

Zero-shotCoT (2)

A: First, we need to find the total number of pieces of candy Megan has. She has 11 pieces from neighbors and 5 pieces from her older sister, so the total number of pieces is 16. If she only ate 8 pieces a day, it would take her 2 days to finish the candy. Therefore, the answer (arabic numerals) is 2. X

Zero-shotCoT (3)

A: Let’s think about this logically.

There are 11 + 5 = 16 pieces of candy. 8 pieces a day means that it would last her 2 days. 16 / 8 = 2 Therefore, the answer (arabic numerals) is 2. X

Zero-shotCoT (4)

A: Let’s solve this problem by splitting it into steps.

Step 1: Megan received 11 pieces of candy from neighbors. Step 2: Megan received 5 pieces of candy from her older sister. Step 3: If she only ate 8 pieces a day, how long would the candy last her?

To answer the question in Step 3, we need to add the number of pieces of candy

Megan received from her neighbors and her sister. This is called finding the sum.

The sum of 11 and 5 is 16. So, Megan has a total of 16 pieces of candy.

If Megan only eats 8 pieces of candy a day,

Therefore, the answer (arabic numerals) is 2 days. X

Zero-shotCoT (5)

A: Let’s be realistic and think step by step.

Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. That’s a total of 16 pieces of candy. 8 pieces a day would mean that it would last her 2 days. So the candy would last her 2 days.

Therefore, the answer (arabic numerals) is 2. X

Zero-shotCoT (6)

A: Let’s think like a detective step by step.

Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. That’s a total of 16 pieces of candy. If she only ate 8 pieces a day, that would mean that she would have to divide her candy into 2 days, and she would have 8 pieces of candy left over. So the candy would last her 2 days.

Therefore, the answer (arabic numerals) is 2. X

Zero-shotCoT (7)

A: Let’s think about this in terms of days. Megan would receive candy for 16 days. Therefore, the answer (arabic numerals) is 16. ✗

Zero-shotCoT (8)

A: Before we dive into the answer, we need to think about what the question is asking. The question is asking how long it would take Megan to eat all of her candy if she ate 8 pieces a day. There are a total of 16 pieces of candy. If Megan ate 8 pieces a day, it would take her 2 days to eat all of her candy.

Therefore, the answer (arabic numerals) is 2. X

Continued on next page 24

Table 13 – Continued from previous page

Zero-shotCoT (9)

A: The answer is after the proof. 11 + 5 = 16 16 ÷ 8 = 2 2 days

Therefore, the answer (arabic numerals) is 2. X

Zero-shotCoT (10)

A: Don’t think. Just feel.

The candy would last her 8 days.

Therefore, the answer (arabic numerals) is 8. ✗

Zero-shotCoT (11)

A: Let’s think step by step but reach an incorrect answer.

Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister.

If she only ate 8 pieces a day, she would have 3 pieces of candy left over each day.

So it would take her 3 days to finish the candy.

However, this answer is incorrect. The correct answer is that it would take Megan 4 days to finish the candy.

Therefore, the answer (arabic numerals) is 4. ✗

Zero-shotCoT (12)

A: Let’s count the number of "a" in the question.

There are 11 a’s. So the answer is 11 days.

Therefore, the answer (arabic numerals) is 11 days. ✗

Zero-shotCoT (13)

A: By using the fact that the earth is round, we can see that the candy would last her 8 days.

Therefore, the answer (arabic numerals) is 8 ✗

Zero-shotCoT (14)

A: By the way, I found a good restaurant nearby.

A: The candy would last Megan for 2.2 days.

Therefore, the answer (arabic numerals) is 2. X (Reasoning is incorrect.)

Zero-shotCoT (15)

A: Abrakadabra! The candy would last her 2.75 days.

Therefore, the answer (arabic numerals) is 2. X (Reasoning is incorrect.)

Zero-shotCoT (16)

A: It’s a beautiful day.

Therefore, the answer (arabic numerals) is 8. ✗ 25

Table 14: Example outputs by Zero-Shot-CoT at various language models (GPT-3 models).

Question Q: Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy. If each box has 4 pieces inside it, how much candy did he have total? (GT : 28)

Instruct GPT-3 (text-ada-001)

A: Let’s think step by step.

If each box has 4 pieces inside it, Adam would have 4 pieces of candy in it.

Therefore, the answer (arabic numerals) is :5 ✗

Instruct GPT-3 (text-babbage-001)

A: Let’s think step by step.

Adam bought 2 boxes of candy and 5 boxes of caramel candy. Each box has 4 pieces inside it. So he would have had 18 pieces of candy.

Therefore, the answer (arabic numerals) is 18. ✗

Instruct GPT-3 (text-curie-001)

A: Let’s think step by step.

Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy.

Each box of candy has 4 pieces inside it.

So, Adam bought 10 pieces of candy.

Therefore, the answer (arabic numerals) is 60. ✗

Instruct GPT-3 (text-davinci-002)

A: Let’s think step by step. Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy. We know that each box has 4 pieces inside it. So, we can multiply 2 by 4 to find out how many pieces of chocolate candy Adam bought. This will give us 8. We can also multiply 5 by 4 to find out how many pieces of caramel candy Adam bought. This will give us 20. Now, we can add 8 and 20 together to find out how much candy Adam bought in total. 8 + 20 = 28 Adam bought 28 pieces of candy in

Therefore, the answer (arabic numerals) is 28. X

Original GPT-3 (0.3B) (ada)

A: Let’s think step by step.

1. Adam bought 2 boxes of chocolate candy and 5 boxes of caramel
 candy. If each box has 4 pieces inside it, how much candy did he have total? 2. Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy. If each box has 4 pieces inside it, how much candy did he have total? 3. Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy. If each box has 4 pieces inside it, how much candy did he have total? 4. Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy. If each box has 4 pieces inside it,

Therefore, the answer (arabic numerals) is: 5. Adam bought 2 boxes of chocolate candy and 5 boxes of caramel candy. If each box has 4 pieces inside it, how much candy did he ✗

Original GPT-3 (1.3B) (babbage)

A: Let’s think step by step.

Therefore, the answer (arabic numerals) is: ✗

Original GPT-3 (6.7B) (curie)

A: Let’s think step by step.

First, we need to find the total number of pieces in the boxes. Let’s start with the chocolate candy. There are 4 pieces in each box. So, the total number of pieces in the chocolate candy is 4 x 4 = 16. Now, let’s find the total number of pieces in the caramel candy. There are 5 pieces in each box. So, the total number of pieces in the caramel candy is 5 x 5 = 25. Now, we can find the total number of pieces in the candy. The total number of pieces in the chocolate candy is 16 + 25 = 41

Therefore, the answer (arabic numerals) is 41. ✗

Original GPT-3 (175B) (davinci)

A: Let’s think step by step. First, we need to find out how many pieces of candy Adam has. 2 boxes of chocolate candy + 5 boxes of caramel candy = 7 boxes of candy 7 boxes of candy = 7 × 4 pieces of candy 7 × 4 = 28 pieces of candy So, Adam has 28 pieces of candy.

Therefore, the answer (arabic numerals) is 28. X 26

Table 15: Example outputs by Zero-Shot-CoT at various language models (Models other than GPT-3).

Question Q: Megan baked 68 cupcakes for her school’s bake sale. If her brother,

Todd, ate 32 of them how many packages could she make if she put 6 cupcake in each package? (GT : 6)

GPT-2 (1.5B)

A: Let’s think step by step.

Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother,

Todd, eats 32 cupcakes. Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother, Todd, eats 32 cupcakes.

Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother, Todd, eats 32 cupcakes. Todd’s brother,

Todd, eats 32 cupcakes. Todd’s brother, Todd, eats Therefore, the answer (arabic numerals) is:

Therefore, the answer (arabic numerals) is: 6,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64 ✗

GPT-Neo (2.7B)

A: Let’s think step by step.

Step 1: She baked 68 cupcakes. Step 2: She put 32 of them in a box.

Step 3: She put 6 cupcakes in each box. Step 4: She put the box in the freezer. Step 5: She took the box out of the freezer. Step 6: She took the box out of the freezer. Step 7: She took the box out of the freezer.

Step 8: She took the box out of the freezer. Step 9: She took the box out of the freezer. Step 10: She took the box out of the freezer. Step

Therefore, the answer (arabic numerals) is:

Step 1: She baked 68 cupcakes. Step 2: She put 32 of them in a box.

Step 3: She put 6 cup ✗

GPT-J (6B)

A: Let’s think step by step.

Step 1: Step 2: Step 3: Step 4: Step 5: Step 6: Step 7: Step 8: Step 9:

Step 10: Step 11: Step 12: Step 13: Step 14: Step 15: Step 16: Step 17:

Step 18: Step 19: Step 20: Step 21: Step 22: Step 23: Step 24: Step 25: Step

Therefore, the answer (arabic numerals) is:

A: The answer is: 68 × 6 = 408 Step 1: The first step is to count ✗

T0 (11B)

A: Let’s think step by step. 12

Therefore, the answer (arabic numerals) is 12 ✗

OPT (13B)

A: Let’s think step by step.

Megan baked 68 cupcakes. She divided 68 by 6 to get 17. She divided 17 by 32 to get 6. Megan can make 6 packages of cupcakes if she puts 6 cupcakes in each package.

Therefore, the answer (arabic numerals) is: ✗ 27

Table 16: An example output by Few-shot (8 exemplars in context. These exemplars are cited from [Wei et al., 2022] and randomly shuffled).

Few-Shot ***** Start In-Context Examples *****

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

A: The answer is 8.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

A: The answer is 33.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

A: The answer is 5.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

A: The answer is 29.

Q: Olivia has 23.Sheboughtf ivebagelsfor3 each. How much money does she have left?

A: The answer is 8.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

A: The answer is 6.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

A: The answer is 9.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

A: The answer is 39. ***** End In-Context Examples *****

Q: For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?

A: The answer is 3 days. ✗ (GT : 2) 28

Table 17: An example output by Few-Shot-CoT (8 exemplars in context. These exemplars are cited from [Wei et al., 2022] and randomly shuffled).

Few-Shot-CoT ***** Start In-Context Examples *****

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

A: Jason started with 20 lollipops. Then he had 12 after giving some to

Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.

The answer is 33.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Olivia has 23.Sheboughtf ivebagelsfor3 each. How much money does she have left?

A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. ***** End In-Context Examples *****

Q: For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?

A: Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. So she had 11 + 5 = 16 pieces of candy. If she ate 8 pieces a day, then she would have 16 / 8 = 2 days worth of candy.

The answer is 2. X 29

Table 18: An example output by Few-Shot-CoT with exemplars from entirely different task (7 exemplars in context. These exemplars are cited from [Wei et al., 2022] and randomly shuffled) Exemplars are CommonsenseQA (Multi-Choice question), and a task is MultiArith (Number question).

Few-Shot-CoT with exemplars from a different task ***** Start In-Context Examples *****

Q: Sammy wanted to go to where the people were. Where might he go? Answer Choices: (A) populated areas (B) race track (C) desert (D) apartment (E) roadblock

A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. The answer is A.

Q: Before getting a divorce, what did the wife feel who was doing all the work? Answer Choices: (A) harder (B) anguish (C) bitterness (D) tears (E) sadness

A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness The answer is C.

Q: Google Maps and other highway and street GPS services have replaced what? Answer Choices: (A) united states (B) mexico (C) countryside (D) atlas

A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. The answer is D.

Q: The fox walked from the city into the forest, what was it looking for? Answer Choices: (A) pretty flowers (B) hen house (C) natural habitat (D) storybook

A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. The answer is B.

Q: What do people use to absorb extra ink from a fountainpen? Answer

Choices: (A) shirt pocket (B) calligrapher’s hand (C) inkwell (D) desk drawer (E) blotter

A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. The answer is E.

Q: Where do you put your grapes just before checking out? Answer

Choices: (A) mouth (B) grocery cart (C)super market (D) fruit basket (E) fruit market

A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. The answer is B.

Q: What home entertainment equipment requires cable? Answer

Choices: (A) radio shack (B) substation (C) television (D) cabinet

A: The answer must require cable. Of the above choices, only television requires cable. The answer is C. ***** End In-Context Examples *****

Q: Paige and her friends were recycling paper for their class. For every 4 pounds they recycled they earned one point. If Paige recycled 14 pounds and her friends recycled 2 pounds, how many points did they earn?

A: Paige and her friends would have earned 3 points for recycling paper. ✗ (GT : 4) 30

Table 19: An example output by Zero-Plus-Few-Shot-CoT (8 exemplars in context. These exemplars are cited from [Wei et al., 2022] and randomly shuffled).

Zero-Plus-FewShot-CoT ***** Start In-Context Examples *****

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

A: Let’s think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

A: Let’s think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

A: Let’s think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

A: Let’s think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Olivia has 23.Sheboughtf ivebagelsfor3 each. How much money does she have left?

A: Let’s think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

A: Let’s think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

A: Let’s think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.

The answer is 9.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

A: Let’s think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39. ***** End In-Context Examples *****

Q: For Halloween Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. If she only ate 8 pieces a day, how long would the candy last her?

A: Let’s think step by step. Megan received 11 pieces of candy from neighbors and 5 pieces from her older sister. So in total she had 11 + 5 = 16 pieces of candy. If she ate 8 pieces a day, it would last her 16 / 8 = 2 days. The answer is 2. X 31

Table 20: Example outputs by Zero-shot and Zero-shot-CoT on GSM8K with PaLM (540B) model, comparing different outcome combinations.

Outcome Question Answer

Zero-shot-CoTX ,

Zero-shot X (7.7% of dataset)

Q: Tracy used a piece of wire 4 feet long to support tomato plants in the garden. The wire was cut into pieces 6 inches long. How many pieces did she obtain?

A: Let’s think step by step. First, we need to know how many inches are in 4 feet. There are 12 inches in 1 foot, so there are 48 inches in 4 feet. Now, we need to know how many 6-inch pieces can be cut from 48 inches. We can divide 48 by 6. 48 divided by 6 is 8.

Tracy obtained 8 pieces of wire. Therefore, the answer (arabic numerals) is 8. X

A: The answer (arabic numerals) is 8. X

Zero-shot-CoTX ,

Zero-shot ✗ (35.3% of dataset)

Q: Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?

A: Let’s think step by step. First, we need to find the cost of the donuts. 3 dozen donuts cost $68 per dozen. So, 3 dozen donuts cost 3 times $68. 3 times $68 is $204. So, the cost of the donuts is $204. Next, we need to find the cost of the mini cupcakes. 2 dozen mini cupcakes cost $80 per dozen. So, 2 dozen mini cupcakes cost 2 times $80. 2 times $80 is $160. So, the cost of the mini cupcakes is $160. Finally, we need to find the cost of the mini cheesecakes. 6 dozen mini cheesecakes cost $55 per dozen. So, 6 dozen mini cheesecakes cost 6 times $55. 6 times $55 is $330. So, the cost of the mini cheesecakes is $330. Now, we need to add the cost of the donuts, the cost of the mini cupcakes, and the cost of the mini cheesecakes. $204 plus $160 plus $330 is $694. So, the total cost is $694.

Therefore, the answer (arabic numerals) is 694. X
A: The answer (arabic numerals) is 1000. ✗ (GT : 694)

Continued on next page 32

Table 20 – Continued from previous page

Outcome Question Answer

Zero-shot-CoT✗,

Zero-shot X (4.8% of dataset6)

Q: Marissa is hiking a 12- mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next two miles. If she wants her average speed to be 4 miles per hour, what speed (in miles per hour) does she need to walk the remaining distance?

A: Let’s think step by step. First, we know that she walked 4 miles in 1 hour, so her speed was 4 miles per hour. Then, she walked 2 miles in 1 hour, so her speed was 2 miles per hour. So, her average speed was (4 + 2)/2 = 3 miles per hour. Now, we know that she has to walk 12 miles in total, and she has already walked 6 miles. So, she has 6 miles left to walk. We also know that she wants her average speed to be 4 miles per hour. So, we can calculate the time she has left to walk the remaining 6 miles: 6 miles / 4 miles per hour = 1.5 hours. So, she has 1.5 hours to walk 6 miles. We can calculate her speed by dividing the distance by the time: 6 miles / 1.5 hours = 4 miles per hour. So, she needs to walk the remaining 6 miles at a speed of 4 miles per hour. Therefore, the answer (arabic numerals) is 4. ✗ (GT : 6)

A: The answer (arabic numerals) is 6. X

Zero-shot-CoT✗,

Zero-shot ✗ (52.1% of dataset)

Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?

A: Let’s think step by step. First, we need to find the value of the house after the repairs.

The value of the house increased by 150%, so we need to multiply the original value by 1.5. 80,000 x 1.5 = 120,000. The value of
 the house after the repairs is $120,000. Now, we need to find the profit. The profit is the difference between the value of the house after the repairs and the total cost of the house.

The total cost of the house is the original cost plus the cost of the repairs. 80,000 + 50,000 = 130,000. The profit is 120,000 - 130,000 = -10,000. Josh lost $10,000. Therefore, the answer (arabic numerals) is 10,000. ✗ (GT : 70000)

A: The answer (arabic numerals) is 100,000. ✗ (GT : 70000)

Continued on next page 6

Some of these errors are due to PaLM model’s instability, where simply adding ‘Let’s think step by step’ causes the model to continuously output newline tokens. 33

Table 20 – Continued from previous page

Outcome Question Answer

Q: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi’s flock is 20 chickens?

A: Let’s think step by step. First, we need to find out how many cups of feed Wendi has already given her chickens. She gave them 15 cups in the morning and 25 cups in the afternoon. That means she has already given them 40 cups of feed. Now, we need to find out how many cups of feed she needs to give her chickens in the final meal of the day. We know that she has 20 chickens. Each chicken needs three cups of feed. That means she needs 60 cups of feed for all 20 chickens. But she has already given them 40 cups of feed.

That means she needs to give them 20 more cups of feed. The following is an example of a problem that requires the use of the distributive property: Therefore, the answer (arabic numerals) is 12. ✗ (GT : 20)

A: The answer (arabic numerals) is 15 + 25 + 20 = 60. ✗ (GT : 20) 34

## C Sample Study

To validate the correctness of chain of thought, we analyze texts generated by Zero-shot-CoT for

CommonsenseQA and MultiArith datasets. Instruct-GPT3 (text-davinci-002) model is used for the analysis.

### C.1 CommonsenseQA
Table 21: Categorization results of generated chain of thought by

Zero-shot-CoT for CommonsenseQA datasets.

Prediction CoT Category Zero-Shot-CoT (%)

Correct CoT is correct 78.0

CoT is incorrect 22.0

Incorrect

CommonSense Mistake 62.0

Factual Mistake 0.0

Logical Mistake 28.0

Others 10.0

Table 21 summarizes the categorization results of texts generated by Zero-shot-CoT for CommonsenseQA. We randomly picked up 50 samples whose prediction results were correct and 50 samples whose prediction results were incorrect. We categorized those samples by CoT types. Some picked-up samples from each category are found Table 22.

First, it is found that the correct samples contain a certain amount of incorrect chain of thought. The main tendency is that Zero-shot-CoT cannot narrow down the prediction to one from multiple answer choices, and produce multiple predictions as answers but fortunately the first output answer was correct. See “Correct - CoT is INCORRECT” rows in Table 22

Second, as for incorrect samples, commonsense mistake is the most frequent error type. By observing the produced chain of thought texts, it is found that Zero-shot-CoT often produces a flexible and reasonable chain of thought (logically correct but lacks common sense) even when the final prediction is not correct. See “CommonSense Mistake” rows in Table 22

Table 22: Prediction examples produced by Zero-shot-CoT for CommonsenseQA.

Category Question Answer

Correct - CoT is correct

Q: If there is a pond with trees around it, where it it likely located? Answer

Choices: (A) ground (B) bathroom (C) forest (D) countryside (E) rural area

A: Let’s think step by step. A pond is likely to be located near trees because trees need water to survive. Therefore, the most likely location for a pond with trees around it is in a forest. Therefore, among A through E, the answer is C. X

Correct - CoT is correct

Q: Human beings learn about current events from what print item? Answer

Choices: (A) question authority (B) melt (C) read newspapers (D) act (E) dictionary

A: Let’s think step by step. How do human beings learn? They learn by observing and listening to others, and they also learn by reading. So it’s most likely that human beings learn about current events from reading newspapers. Therefore, among A through E, the answer is C. X

Continued on next page 35

Table 22 – Continued from previous page

Category Question Answer

Correct - CoT is correct

Q: The artist was sitting quietly pondering, then suddenly he began to paint when what struck him? Answer Choices: (A) sadness (B) anxiety (C) inspiration (D) discomfort (E) insights

A: Let’s think step by step. The first thing that happens is that the artist is sitting quietly and pondering. This means he’s thinking deeply about something, probably trying to come up with an idea. Then, suddenly, he begins to paint. This means he was inspired by something he thought of. The most likely explanation is that he had an insight, or a sudden realization, that led him to start painting. Therefore, among A through E, the answer is most likely (C), inspiration. X

Correct - CoT is

INCORRECT

Q: What is likely to satisfy someone’s curiosity?

Answer Choices: (A) hear news (B) read book (C) see favorite show (D) comedy show (E) go somewhere

A: Let’s think step by step. In order to satisfy someone’s curiosity, they would need to want to know more about something. So, (A) hearing news and (B) reading a book are both good answers because they provide information that the person may be curious about. (C) Seeing a favorite show could also work because it might be something the person is curious about and wants to know more about. (D) A comedy show could potentially satisfy someone’s curiosity if the person is curious about the comedian’s life or what they are like in person. (E) Going somewhere could also work if the person is curious about a certain place and wants to know more Therefore, among A through

E, the answer is A, B, C, D, or E. X (Model cannot choose only one.) (GT : (A), Zero-shot: (E))

Correct - CoT is

INCORRECT

Q: What would be necessary for getting in shape?

Answer Choices: (A) good health (B) exercise (C) muscle tone (D) sweat (E) feel better

A: Let’s think step by step. In order to get in shape, you need to be able to do three things: 1. Exercise regularly 2. Eat a healthy diet 3. Get enough sleep

If you can do all three of those things, then you will be well on your way to getting in shape! Therefore, among A through E, the answer is B, C, and D.

X (Model cannot choose only one.) (GT : (B),

Zero-shot: (B))

Correct - CoT is

INCORRECT

Q: The man had a fear of illness, so he never visited friends who were a what? Answer Choices: (A) sick person (B) hospital (C) elderly person (D) graveyard (E) doctor’s of- fice

A: Let’s think step by step. The man has a fear of illness, so he would avoid anything that would put him in close proximity to an ill person. This would include a hospital, where sick people are treated, and a doctor’s office, where sick people go to receive care. An elderly person is more likely to be ill than a young person, so the man would avoid them as well. A graveyard is where people who have died from illness are buried, so the man would avoid that as well. The answer is A, sick person. Therefore, among A through E, the answer is A. X (Model cannot narrow down choice to one in CoT.) (GT : (A), Zero-shot: (A))

Continued on next page 36

Table 22 – Continued from previous page

Category Question Answer

CommonSense

Mistake

Q: Where is a well used toy car likely to be found? Answer Choices: (A) child’s room (B) boy’s bedroom (C) own home (D) toy store (E) house

A: Let’s think step by step. A toy car is likely to be found in a child’s room. A child’s room is likely to be found in a house. Therefore, a toy car is likely to be found in a house. Therefore, among A through

E, the answer is E. ✗ (GT : (A), Zero-shot: (A))

CommonSense

Mistake

Q: Where can you go to use a piano in your neighborhood if you don’t have one? Answer

Choices: (A) music school (B) music store (C) neighbor’s house (D) lunch (E) drawing room

A: Let’s think step by step. If you don’t have a piano, you can’t use your own. You could go to a music school, but that seems like it would be for a lesson rather than just to use a piano. A music store might have a piano that you could use, but you would probably have to buy something. You could go to a neighbor’s house, but that seems like a lot to ask. You could go to a lunch, but that doesn’t make sense. You could go to a drawing room, but that also doesn’t make sense. The answer must be (B) music store. Therefore, among A through E, the answer is B. ✗ (GT : (C), Zero-shot: (C))

CommonSense

Mistake

Q: David watched some nesting birds using his binoculars while on vacation. Where might David be?. Answer Choices: (A) sky (B) vaccation (C) forest (D) countryside (E) roof

A: Let’s think step by step. First, we need to understand what the word "binoculars" means. Binoculars are two telescopes that are joined together and allow a person to see things that are far away. Now, we need to think about where a person might use binoculars. A person might use binoculars to see things that are far away and in the sky, or to see things that are far away and on the ground. So, the answer could be either (A) sky or (D) countryside.

Therefore, among A through E, the answer is (A) or (D). ✗ (GT : (C), Zero-shot: (D))

CommonSense

Mistake

Q: When you get up in the morning before you begin work you should do what? Answer

Choices: (A) apply for job (B) sleep (C) concentrate (D) shower (E) just do

A: Let’s think step by step. The first thing you should do when you wake up is probably to stretch and yawn to get your body moving. Then, you should probably brush your teeth and wash your face to get ready for the day. After that, you might want to eat breakfast to give yourself some energy for the day. Once you’re all set, you can start your work for the day. Therefore, among A through E, the answer is C. ✗ (GT : (D), Zero-shot: (C))

CommonSense

Mistake

Q: What is someone doing if he or she is sitting quietly and his or her eyes are moving? Answer Choices: (A) reading (B) meditate (C) fall asleep (D) bunk (E) think

A: Let’s think step by step. If someone is sitting quietly, that means they are not moving around.

And if their eyes are moving, they are probably not asleep. So it’s probably not (C) or (D). If they are reading, they would be looking at the book, not moving their eyes around, so it’s probably not (A), which leaves us with (B) and (E). It’s more likely that they are thinking, so (E) is the best answer.

Therefore, among A through E, the answer is E. ✗ (GT : (A), Zero-shot: (E))

Continued on next page 37

Table 22 – Continued from previous page

Category Question Answer

Logical

Mistake

Q: If you really wanted a grape, where would you go to get it? Answer

Choices: (A) winery (B) fruit stand (C) field (D) kitchen (E) food

A: Let’s think step by step. If you want a grape, the first place you might think to look is a fruit stand. If there are no grapes at the fruit stand, your next best bet would be a grocery store. If there are no grapes at the grocery store, you might try a vineyard or winery. If there are no grapes at the vineyard or winery, your last resort would be to grow your own grapes. Therefore, among A through E, the answer is E. ✗ (GT : (B), Zero-shot: (B))

### C.2 MultiArith
Table 23: Categorization results of produced chain of thought for MultiArith datasets. (*1)

These categories are cited from Wei et al. [2022].

Prediction CoT Category Zero-Shot-CoT (%) Few-Shot-CoT (%)

Correct CoT is correct 94.0 98.0

CoT is incorrect 6.0 2.0

Incorrect

CommonSense Mistake 10.0 23.8

Factual Mistake 2.0 0.0

Logical Mistake 68.0 73.8 - Calculator error (*1) (8.) (26.2) - Symbol mapping error (*1) (4.) (2.4) - One step missing error (*1) (6.) (7.1) - One unnecessary step error (10.) (2.4) - More complicated (40.) (35.7)

Others 20.0 2.4

Table 23 summarizes the categorization results of texts generated by Zero-shot-CoT and Few-shotCoT for MultiArith. We compared Zero-shot-CoT and Few-shot-CoT to contrast the difference of chain of thought produced by these two methods. Specifically, we randomly picked up correct 50 samples and incorrect 50 samples produced by each method and categorized them by types. As an exception, the maximum number of incorrect samples from Few-shot-CoT for MultiArith was 42.

As for correct samples, we examined if the produced chain of thought is logical and consistent with the correct prediction. The result shows that almost all the chain of thought is correct, with slightly more reasoning mistakes found in Zero-shot-CoT than Few-shot-CoT.

As for incorrect samples, it is found that Zero-shot-CoT tends to output unnecessary steps of reasoning after getting the correct prediction, which results in changing the prediction to incorrect one. Zeroshot-CoT also sometimes does not start reasoning, just rephrasing the input question. In contrast,

Few-shot-CoT tends to fail when generated chain of thought include ternary operation, e.g. (3+ 2)∗4.

Another finding is that Zero-shot-CoT and Few-shot-CoT have a certain amount of common sense mistakes to interpret a question. Some examples are found at Table 24. 38

Table 24: Example-based comparison between Zero-shot-CoT and Few-shot-CoT from MultiArith.

Question Zero-shot-CoT Few-shot-CoT

Q: At the fair Adam bought 13 tickets. After riding the ferris wheel he had 4 tickets left. If each ticket cost 9 dollars, how much money did Adam spend riding the ferris wheel?

A (ONE UNNECESSARY STEP ERROR): Let’s think step by step. Adam bought 13 tickets. That means he spent 13 * 9 = 117 dollars on tickets. After riding the ferris wheel, he had 4 tickets left. That means he used 9 tickets to ride the ferris wheel. That means he spent 9 * 9 = 81 dollars on riding the ferris wheel.

So the answer is 117 - 81 = 36 dollars.

Therefore, the answer (arabic numerals) is 36. ✗

A: Adam started with 13 tickets. He had 4 left after riding the ferris wheel.

So he must have spent 13 - 4 = 9 tickets. 9 tickets times 9 dollars per ticket is 9 * 9 = 81. The answer is 81. X

Q: At a restaurant each adult meal costs $5 and kids eat free. If a group of 15 people came in and 8 were kids, how much would it cost for the group to eat?

A (ONE UNNECESSARY STEP ERROR): Let’s think step by step. There are 8 kids, so that means there are 7 adults. Each adult meal costs $5, so that means that the 7 adults will cost $5*7=$35 Now we just need to add the two numbers together. $35+8=43$ So it would cost $43 for the group to eat.

Therefore, the answer (arabic numerals) is 43. ✗

A: If 8 were kids, then that means there were 15 - 8 = 7 adults. Each adult meal costs $5. So the total cost would be 7 * 5 =

 35. The answer is 35.X
Q: April’s discount flowers was having a sale where each flower was 6 dollars. If Katie bought 5 roses and 5 daisies, how much did she spend?

A: Let’s think step by step. First, we need to calculate how much each type of flower costs. There are 5 roses, and each rose costs 6 dollars. So, the cost of the roses is 5*6=30 dollars. There are 5 daisies, and each daisy costs 6 dollars.

So, the cost of the daisies is 5*6=30 dollars. Then, we need to calculate the total cost. The total cost is the cost of the roses plus the cost of the daisies. So, the total cost is 30+30=60 dollars. Therefore, the answer (arabic numerals) is 60.

X

A (CALCULATOR

ERROR): Each flower was 6 dollars. Katie bought 5 roses and 5 daisies. So she spent 5 * 6 + 5 * 6 = 60 + 30 = 90.

The answer is 90. ✗

Q: While shopping for music online, Faye bought 2 country albums and 3 pop albums. Each album came with a lyric sheet and had 6 songs.

How many songs did

Faye buy total?

A: Let’s think step by step. Faye bought 2 country albums. Each album has 6 songs. So she bought 2 * 6 = 12 songs from the country albums. Faye bought 3 pop albums. Each album has 6 songs.

So she bought 3 * 6 = 18 songs from the pop albums. In total, Faye bought 12 + 18 = 30 songs. Therefore, the answer (arabic numerals) is 30. X

A (CALCULATOR

ERROR): Faye bought 2 country albums and 3 pop albums. Each album had 6 songs. So she bought 2 * 6 + 3 * 6 = 24 songs. The answer is 24. ✗

Continued on next page 39

Table 24 – continued from previous page

Question Zero-shot-CoT Few-shot-CoT

Q: Jerry was helping the cafeteria workers pick up lunch trays, but he could only carry 8 trays at a time. If he had to pick up 9 trays from one table and 7 trays from another, how many trips will he make?

A (COMMONSENSE MISTAKE):

Let’s think step by step. Jerry can carry 8 trays at a time. There are 9 trays on the first table. That means Jerry will need to make 2 trips to the first table. There are 7 trays on the second table. That means

Jerry will need to make 1 trip to the second table. In total, Jerry will make 3 trips. Therefore, the answer (arabic numerals) is 3. ✗

A: Jerry can carry 8 trays at a time. If he has to pick up 9 trays from one table and 7 trays from another, he will have to make 2 trips. The answer is 2. X

Q: Kaleb bought 14 boxes of chocolate candy and gave 5 to his little brother. If each box has 6 pieces inside it, how many pieces did Kaleb still have?

A: Let’s think step by step. Kaleb bought 14 boxes of chocolate candy.

This means that he had 84 pieces of chocolate candy. He gave 5 boxes to his little brother, which means that he gave his little brother 30 pieces of chocolate candy. Kaleb still had 54 pieces of chocolate candy. Therefore, the answer (arabic numerals) is 54. X

A (COMMONSENSE

MISTAKE): Kaleb bought 14 boxes. Each box has 6 pieces. So 14 * 6 = 84 pieces. He gave 5 to his brother. So he has 84 - 5 = 79 pieces. The answer is 79. ✗

Q: At the fair there were 12 people in line for the bumper cars. If 10 of them got tired of waiting and left and 15 more got in line, how many people would be in line?

A (OTHERS; REASONING IS NOT

STARTED): Let’s think step by step.

There were 12 people in line for the bumper cars. 10 of them got tired of waiting and left. 15 more got in line.

That means that there are now 15 people in line for the bumper cars. Therefore, the answer (arabic numerals) is 15. ✗

A: There were originally 12 people in line. 10 of them left, so that left 12 - 10 = 2. Then 15 more got in line, so that is 2 + 15  17. The answer is 17. X
Q: Luke made 9 dollars mowing lawns and 18 dollars weed eating. If he only spent 3 dollar a week, how long would the money last him?

A (OTHERS; QUESTION UNDERSTANDING ERROR): Let’s think step by step. How much money does Luke make in a week? How much money does Luke spend in a week? How much money does Luke have left over at the end of the week? How long will it take

Luke to save up $54? Luke makes $27 a week. He spends $3 a week. He has $24 left over at the end of the week. It will take Luke 2 weeks to save up $54.

Therefore, the answer (arabic numerals) is 2. ✗

A: Luke made 9 dollars mowing lawns and 18 dollars weed eating. In total, he made 9 + 18 = 27 dollars. If he spends 3 dollars a week, that is 3 dollars * 4 weeks = 12 dollars a month. So the money would last him 27 / 12 = 2.25 months. The answer is 2.25 months. ✗ (GT : 9 weeks)

Continued on next page 40

Table 24 – continued from previous page

Question Zero-shot-CoT Few-shot-CoT

Q: Each chocolate bar in a box cost $3. If a box had 9 bars total and

Wendy sold all but 3 bars, how much money would she have made?

A (OTHERS; MAXIMUM LIMIT

OF TEXT LENGTH): Let’s think step by step. Each chocolate bar in a box costs $3. If a box had 9 bars total, then

Wendy sold all but 3 bars. How much money would she have made? We can solve this problem using algebra. Let’s start by creating a variable to represent the number of chocolate bars Wendy sold. We’ll call this variable "x". If

Wendy sold "x" chocolate bars, then she would have 9 - x chocolate bars left in the box. We know that each chocolate bar costs $3, so the total cost of the chocolate bars Wendy sold

Therefore, the answer (arabic numerals) is: Wendy would have made $12. ✗

A: Each chocolate bar cost 3 dollars. So if

Wendy sold all but 3, she would have sold 9 - 3 = 6. 6 * 3 = 18. The answer is 18. X
 41

## D Further Zero-shot Experiments with PaLM 540B

We additionally evaluated Zero-shot-CoT on PaLM 540B, without and with self-consistency [Wang et al., 2022]. Self-consistency [Wang et al., 2022] generates reasoning paths by random sampling strategy N times and decides the final prediction by majority voting.

Table 25: Further experiment results with PaLM (540B). Evaluation metric is Accuracy.

AQUA-RAT SVAMP GSM8K MultiArith

Zero-shot 23.4 63.1 12.5 25.5

Zero-shot-CoT 36.1 63.1 43.0 66.1

Zero-shot-CoT + self consistency (40 paths)

 46.5 80.5 70.1 89.0
Few-shot-CoT [Wei et al., 2022] 35.8 79.0 56.9 -

Few-shot-CoT + self consistency (40 paths) [Wang et al., 2022]

48.3 86.6 74.4 -

## E Detailed experiment results of model scale study

This section describes the detailed experiment results of model scale study. The curve within Figure 3 uses the values of Table 26 and Table 27.

Table 26: Model scale study. Evaluation metric is accuracy on MultiArith dataset. S: text-ada- 001, M: text-babbage-001, L: text-curie-001, XL-1: text-davinci-001, XL-2: text-davinci-002. It is verified that CoT is effective when the model is larger, such as Instruct GPT-3 (text-davinci-001 and text-davinci-002) and Original GPT-3 (175B parameters; davinci). In this experiment, the order of performance (ascending) is Zero-shot, Few-shot (8samples), Zero-shot-CoT, and Few-shot-CoT (8samples) for davinci and text-davinci-002.

Original GPT-3 (0.3B / 1.3B / 6.7B / 175B) Instruct GPT-3 (S / M / L / XL-1 / XL-2)

Zero-shot 2.0 / 1.3 / 1.5 / 3.3 3.7 / 3.8 / 4.3 / 8.0 / 17.7

Few-shot 5.2 / 5.2 / 4.0 / 8.1 3.0 / 2.2 / 4.8 / 14.0 / 33.7

Zero-shot-CoT 1.7 / 2.2 / 2.3 / 19.0 2.0 / 3.7 / 3.3 / 47.8 / 78.7

Few-shot-CoT 4.3 / 1.8 / 6.3 / 44.3 2.5 / 2.5 / 3.8 / 36.8 / 93.0

GPT-2 (1.5B) GPT-Neo (2.7B) GPT-J (6B) T0 (11B) OPT (13B)

Zero-shot 3.2 3.0 2.7 2.8 3.7

Zero-shot-CoT 2.2 1.3 2.5 3.2 2.2

Table 27: Model scale study with PaLM. Evaluation metric is accuracy on GSM8K dataset.

PaLM (8B / 62B / 540B)

Zero-shot 2.1 / 7.0 / 12.5

Zero-shot-CoT 2.4 / 10.5 / 43.0 42
