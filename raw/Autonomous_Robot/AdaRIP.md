# Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?
自动驾驶汽车能否识别、恢复并适应分布变化？ https://arxiv.org/abs/2006.14911

## Abstract
Out-of-training-distribution (OOD) scenarios are a common challenge of learning agents at deployment, typically leading to arbitrary deductions and poorly-informed decisions. In principle, detection of and adaptation to OOD scenes can mitigate their adverse effects. In this paper, we highlight the limitations of current approaches to novel driving scenes and propose an epistemic uncertainty-aware planning method, called robust imitative planning (RIP). Our method can detect and recover from some distribution shifts, reducing the overconfident and catastrophic extrapolations in OOD scenes. If the model’s uncertainty is too great to suggest a safe course of action, the model can instead query the expert driver for feedback, enabling sample-efficient online adaptation, a variant of our method we term adaptive robust imitative planning (AdaRIP). Our methods outperform current state-of-the-art approaches in the nuScenes prediction challenge, but since no benchmark evaluating OOD detection and adaption currently exists to assess control, we introduce an autonomous car novel-scene benchmark,CARNOVEL, to evaluate the robustness of driving agents to a suite of tasks with distribution shifts, where our methods outperform all the baselines.

训练分布外 (OOD) 场景是学习智能体在部署时面临的常见挑战，通常会导致任意推论和不明智的决策。 原则上，检测和适应 OOD 场景可以减轻它们的不利影响。 在本文中，我们强调了当前新颖驾驶场景方法的局限性，并提出了一种认知不确定性感知规划方法，称为稳健模仿规划 (RIP)。 我们的方法可以检测到一些分布变化并从中恢复，减少 OOD 场景中的过度自信和灾难性外推。 如果模型的不确定性太大而无法建议安全的行动方案，则该模型可以向专家司机查询反馈，从而实现样本有效的在线适应，这是我们称为自适应稳健模仿规划 (AdaRIP) 方法的一种变体。 我们的方法在 nuScenes 预测挑战中优于当前最先进的方法，但由于目前没有评估 OOD 检测和适应的基准来评估控制，我们引入了一个自动驾驶汽车新场景基准 CARNOVEL 来评估 驱动智能体执行一组具有分布变化的任务，我们的方法在这些任务中优于所有基线。

## 1. Introduction
Autonomous agents hold the promise of systematizing decision-making to reduce catastrophes due to human mistakes. Recent advances in machine learning (ML) enable the deployment of such agents in challenging, real-world, safety-critical domains, such as autonomous driving (AD) in urban areas. However, it has been repeatedly demonstrated that the reliability of ML models degrades radically when they are exposed to novel settings (i.e., under a shift away from the distribution of observations seen during their training) due to their failure to generalise, leading to catastrophic outcomes (Sugiyama & Kawanabe, 2012; Amodei et al., 2016; Snoek et al., 2019). The diminishing performance of ML models to out-of-training distribution (OOD) regimes is concerning in life-critical applications, such as AD (Quionero-Candela et al., 2009; Leike et al., 2017).

自主智能体有望将决策系统化，以减少因人为错误造成的灾难。 机器学习 (ML) 的最新进展使此类智能体能够部署在具有挑战性的、现实世界的、安全关键领域，例如城市地区的自动驾驶 (AD)。 然而，已经反复证明，当 ML 模型暴露于新环境时(即偏离训练期间观察到的分布)，由于它们无法泛化，可靠性会急剧下降，从而导致灾难性的后果 (Sugiyama 和 Kawanabe，2012 ;Amodei et al., 2016 ;Snoek et al., 2019)。 ML 模型对训练外分布 (OOD) 机制的性能下降在生命关键应用程序中引起关注，例如 AD(Quionero-Candela et al., 2009 ;Leike et al., 2017)。

Figure 1. Didactic example: (a) in a novel, out-of-training distribution (OOD) driving scenario, candidate plans/trajectories y1, y2, y3 are (b) evaluated (row-wise) by an ensemble of expertlikelihood models q1, q2, q3. Under models q1 and q2 the best plans are the catastrophic trajectories y1 and y2 respectively. Our epistemic uncertainty-aware robust (RIP) planning method aggregates the evaluations of the ensemble and proposes the safe plan y3 . RIP considers the disagreement between the models and avoid overconfident but catastrophic extrapolations in OOD tasks. 
图 1. 教学样本：(a) 在新颖的训练外分布 (OOD) 驾驶场景中，候选计划/轨迹 y1、y2、y3 (b) 由一组专家似然模型进行评估(逐行) 问题 1、问题 2、问题 3。 在模型 q1 和 q2 下，最佳计划分别是灾难性轨迹 y1 和 y2。 我们的认知不确定性感知稳健 (RIP) 规划方法汇总了整体的评估并提出了安全计划 y3。 RIP 考虑模型之间的分歧，避免在 OOD 任务中过度自信但灾难性的外推。

Although there are relatively simple strategies (e.g., stay within the lane boundaries, avoid other cars and pedestrians) that generalise, perception-based, end-to-end approaches, while flexible, they are also susceptible to spurious correlations. Therefore, they can pick up non-causal features that lead to confusion in OOD scenes (de Haan et al., 2019).

尽管有相对简单的策略(例如，留在车道边界内，避开其他汽车和行人)可以概括、基于感知的端到端方法，虽然灵活，但它们也容易受到虚假相关的影响。 因此，他们可以拾取导致 OOD 场景混淆的非因果特征(de Haan et al., 2019)。

Due to the complexity of the real-world and its everchanging dynamics, the deployed agents inevitably face novel situations and should be able to cope with them, to at least (a) identify and ideally (b) recover from them, without failing catastrophically. These desiderata are not captured by the existing benchmarks (Ros et al., 2019; Codevilla et al., 2019) and as a consequence, are not satisfied by the current state-of-the-art methods (Chen et al., 2019; Tang et al., 2019; Rhinehart et al., 2020), which are prone to fail in unpredictable ways when they experience OOD scenarios (depicted in Figure 1 and empirically verified in Section 4).

由于现实世界的复杂性和不断变化的动态，部署的智能体不可避免地会面临新情况，并且应该能够应对这些情况，至少 (a) 识别并理想地 (b) 从中恢复，而不会出现灾难性的失败。 现有基准没有捕捉到这些迫切需求(Ros et al., 2019 ;Codevilla et al., 2019)，因此，当前最先进的方法无法满足这些要求(Chen et al., 2019) ; Tang et al., 2019 ;Rhinehart et al., 2020)，当它们遇到 OOD 场景时，它们很容易以不可预测的方式失败(如图 1 所示，并在第 4 节中进行了经验验证)。

Figure 2. The robust imitative planning (RIP) framework. (a) Expert demonstrations. We assume access to observations x and expert state y pairs, collected either in simulation (Dosovitskiy et al., 2017) or in real-world (Caesar et al., 2019; Sun et al., 2019; Kesten et al., 2019). (b) Learning algorithm (cf. Section 3.1). We capture epistemic model uncertainty by training an ensemble of density estimators {q(y|x; θk)}Kk=1, via maximum likelihood. Other approximate Bayesian deep learning methods (Gal & Ghahramani, 2016) are also tested. (c) Planning paradigm (cf. Section 3.3). The epistemic uncertainty is taken into account at planning via the aggregation operator ⊕ (e.g., mink), and the optimal plan y∗ is calculated online with gradient-based optimization through the learned likelihood models. 
图 2. 稳健的模拟规划 (RIP) 框架。 (一)专家论证。 我们假设可以访问在模拟(Dosovitskiy et al., 2017)或现实世界(Caesar et al., 2019 ;Sun et al., 2019 ;Kesten et al., 2019)中收集的观察结果 x 和专家状态 y 对 ). (b) 学习算法(参见第 3.1 节)。 我们通过最大似然训练一组密度估计量 {q(y|x; θk)}Kk=1 来捕捉认知模型的不确定性。 还测试了其他近似贝叶斯深度学习方法 (Gal & Ghahramani, 2016)。 (c) 规划范例(参见第 3.3 节)。 通过聚合运算符⊕(例如，mink)在规划时考虑了认知不确定性，并通过学习的似然模型通过基于梯度的优化在线计算最优计划 y*。

In this paper, we demonstrate the practical importance of OOD detection in AD and its importance for safety. The key contributions are summarised as follows:
1. Epistemic uncertainty-aware planning: We present an epistemic uncertainty-aware planning method, called robust imitative planning (RIP) for detecting and recovering from distribution shifts. Simple quantification of epistemic uncertainty with deep ensembles enables detection of distribution shifts. By employing Bayesian decision theory and robust control objectives, we show how we can act conservatively in unfamiliar states which often allows us to recover from distribution shifts (didactic example depicted in Figure 1).
2. Uncertainty-driven online adaptation: Our adaptive, online method, called adaptive robust imitative planning (AdaRIP), uses RIP’s epistemic uncertainty estimates to efficiently query the expert for feedback which is used to adapt on-the-fly, without compromising safety. Therefore, AdaRIP could be deployed in the real world: it can reason about what it does not know and in these cases ask for human guidance to guarantee current safety and enhance future performance.
3. Autonomous car novel-scene benchmark: We introduce an autonomous car novel-scene benchmark, called CARNOVEL, to assess the robustness of AD methods to a suite of out-of-distribution tasks. In particular, we evaluate them in terms of their ability to: (a) detect OOD events, measured by the correlation of infractions and model uncertainty; (b) recover from distribution shifts, quantified by the percentage of successful manoeuvres in novel scenes and (c) efficiently adapt to OOD scenarios, provided online supervision.

在本文中，我们展示了 OOD 检测在 AD 中的实际重要性及其对安全的重要性。 主要贡献总结如下：
1. 认知不确定性感知规划：我们提出了一种认知不确定性感知规划方法，称为稳健模仿规划 (RIP)，用于检测分布变化并从中恢复。 使用深度集合对认知不确定性进行简单量化可以检测分布变化。 通过采用贝叶斯决策理论和稳健控制目标，我们展示了我们如何在不熟悉的状态下采取保守行动，这通常使我们能够从分布变化中恢复过来(图 1 中描述的教学样本)。
2. 不确定性驱动的在线适应：我们的自适应在线方法称为自适应稳健模仿规划 (AdaRIP)，它使用 RIP 的认知不确定性估计来有效地查询专家的反馈，用于即时适应，而不会影响安全性。 因此，AdaRIP 可以部署在现实世界中：它可以推理它不知道的事情，并在这些情况下请求人类指导以保证当前的安全并提高未来的性能。
3. 自动驾驶汽车新颖场景基准：我们引入了一种称为 CARNOVEL 的自动驾驶汽车新颖场景基准，以评估 AD 方法对一组分布外任务的稳健性。 特别是，我们根据它们的能力来评估它们：(a) 检测 OOD 事件，通过违规和模型不确定性的相关性来衡量;  (b) 从分布变化中恢复，通过新场景中成功机动的百分比量化，以及 (c) 有效适应 OOD 场景，提供在线监督。

## 2. Problem Setting and Notation
We consider sequential decision-making in safety-critical domains. A method is considered safety when it is accurate, with respect to some metric (cf. Sections 4, 6), and certain.

Assumption 1 (Expert demonstrations). We assume access to a dataset, D = {(xi, yi)}Ni=1, of time-profiled expert trajectories (i.e., plans), y, paired with high-dimensional observations, x, of the corresponding scenes. The trajectories are drawn from the expert policy, y ∼ πexpert(·|x).

Our goal is to approximate the (i.e., near-optimal) unknown expert policy, πexpert, using imitation learning (Widrow &

Smith, 1964; Pomerleau, 1989, IL), based only on the demonstrations, D. For simplicity, we also make the following assumptions, common in the autonomous driving and robotics literature (Rhinehart et al., 2020; Du et al., 2019).

Assumption 2 (Inverse dynamics). We assume access to an inverse dynamics model (Bellman, 2015, PID controller, I), which performs the low-level control – inverse planning – at (i.e., steering, braking and throttling), provided the current and next states (i.e., positions), st and st+1, respectively.

Therefore, we can operate directly on state-only trajectories, y = (s1, . . . , sT ), where the actions are determined by the local planner, at = I(st, st+1), ∀t = 1, . . . , T − 1.

Assumption 3 (Global planner). We assume access to a global navigation system that we can use to specify highlevel goal locations G or/and commands C (e.g., turn left/right at the intersection, take the second exit).

Assumption 4 (Perfect localization). We consider the provided locations (e.g., goal, ego-vehicle positions) as accurate, i.e., filtered by a localization system.

These are benign assumptions for many applications in robotics. If required, these quantities can also be learned from data, and are typically easier to learn than πexpert.




## 3. Robust Imitative Planning
We seek an imitation learning method that (a) provides a distribution over expert plans; (b) quantifies epistemic uncertainty to allow for detection of OOD observations and (c) enables robustness to distribution shift with an explicit mechanism for recovery. Our method is shown in Figure 2.

First, we present the model used for imitating the expert.

### 3.1. Bayesian Imitative Model
We perform context-conditioned density estimation of the distribution over future expert trajectories (i.e., plans), using a probabilistic “imitative” model q(y|x; θ), trained via maximum likelihood estimation (MLE): θMLE = arg max θ E(x,y)∼D [log q(y|x; θ)] . (1)

Contrary to existing methods in AD (Rhinehart et al., 2020;

Chen et al., 2019), we place a prior distribution p(θ) over possible model parameters θ, which induces a distribution over the density models q(y|x; θ). After observing data D, the distribution over density models has a posterior p(θ|D).

Practical implementation. We use an autoregressive neural density estimator (Rhinehart et al., 2018), depicted in

Figure 2b, as the imitative model, parametrised by learnable parameters θ. The likelihood of a plan y in context x to come from an expert (i.e., imitation prior) is given by: q(y|x; θ) =

TYt=1 p(st|y<\t, x; θ) = TYt=1

N (st; µ(y<\t, x; θ), Σ(y<\t, x; θ)), (2) where µ(·; θ) and Σ(·; θ) are two heads of a recurrent neural network, with shared torso. We decompose the imitation prior as a telescopic product (cf. Eqn. (2)), where conditional densities are assumed normally distributed, and the distribution parameters are learned (cf. Eqn. (1)). Despite the unimodality of normal distributions, the autoregression (i.e., sequential sampling of normal distributions where the future samples depend on the past) allows to model multimodal distributions (Uria et al., 2016). Although more expressive alternatives exist, such as the mixture of density networks (Bishop, 1994) and normalising flows (Rezende &

Mohamed, 2015), we empirically find Eqn. (2) sufficient.

The estimation of the posterior of the model parameters, p(θ|D), with exact inference is intractable for non-trivial models (Neal, 2012). We use ensembles of deep imitative models as a simple approximation to the posterior p(θ|D).

We consider an ensemble of K components, using θk to refer to the parameters of our k-th model qk, trained with via maximum likelihood (cf. Eqn. (1) and Figure 2b). However, any (approximate) inference method to recover the posterior p(θ|D) would be applicable. To that end, we also try Monte

Carlo dropout (Gal & Ghahramani, 2016).

### 3.2. Detecting Distribution Shifts
The log-likelihood of a plan log q(y|x; θ) (i.e., imitation prior) is a proxy of the quality of a plan y in context x under model θ. We detect distribution shifts by looking at the disagreement of the qualities of a plan under models coming from the posterior, p(θ|D). We use the variance of the imitation prior with respect to the model posterior, i.e., u(y) , Varp(θ|D) [log q(y|x; θ)] (3) to quantify the model disagreement: Plans at in-distribution scenes have low variance, but high variance in OOD scenes.

We can efficiently calculate Eqn. (3) when we use ensembles, or Monte Carlo, sampling-based methods for p(θ|D).

Having to commit to a decision, just the detection of distribution shifts via the quantification of epistemic uncertainty is insufficient for recovery. Next, we introduce an epistemic uncertainty-aware planning objective that allows for robustness to distribution shifts.

### 3.3. Planning Under Epistemic Uncertainty
We formulate planning to a goal location G under epistemic uncertainty, i.e., posterior over model parameters p(θ|D), as the optimization (Barber, 2012) of the generic objective, which we term robust imitative planning (RIP): yG

RIP , arg max y aggregation operator z }| { ⊕ θ∈supp p(θ|D) log p(y|G, x; θ) | {z} imitation posterior = arg max y ⊕ θ∈supp p(θ|D) logq(y|x; θ) | {z} imitation prior + log p(G|y) | {z} goal likelihood , (4) where ⊕ is an operator (defined below) applied on the posterior p(θ|D) and the goal-likelihood is given, for example, by a Gaussian centred at the final goal location sGT and a pre-specified tolerance  , p(G|y) = N (yT ; yGT , 2I).

Intuitively, we choose the plan yG

RIP that maximises the likelihood to have come from an expert demonstrator (i.e., “imitation prior”) and is “close” to the goal G. The model posterior p(θ|D) represents our belief (uncertainty) about the true expert model, having observed data D and from prior p(θ) and the aggregation operator ⊕ determines our level of awareness to uncertainty under a unified framework.

For example, a deep imitative model (Rhinehart et al., 2020) is a particular instance of the more general family of objectives described by Eqn. (4), where the operator ⊕ selects a


? single θk from the posterior (point estimate). However, this approach is oblivious to the epistemic uncertainty and prone to fail in unfamiliar scenes (cf. Section 4).

In contrast, we focus our attention on two aggregation operators due to their favourable properties, which take epistemic uncertainty into account: (a) one inspired by robust control (Wald, 1939) which encourages pessimism in the face of uncertainty and (b) one from Bayesian decision theory, which marginalises the epistemic uncertainty. Table 1 summarises the different operators considered in our experiments. Next, we motivate the used operators.

#### 3.3.1. WORST CASE MODEL (RIP-WCM)
In the face of (epistemic) uncertainty, robust control (Wald, 1939) suggests to act pessimistically – reason about the worst case scenario and optimise it. All models with nonzero posterior probability p(θ|D) are likely and hence our robust imitative planning with respect to the worst case model (RIP-WCM) objective acts with respect to the most pessimistic model, i.e., sRIP-WCM , arg max y min θ∈supp p(θ|D) log q(y|x; θ). (5)

The solution of the arg maxy minθ optimization problem in Eqn. (5) is generally not tractable, but our deep ensemble approximation enables us to solve it by evaluating the minimum over a finite number of K models. The maximization over plans, y, is solved with online gradient-based adaptive optimization, specifically ADAM (Kingma & Ba, 2014).

An alternative online planning method with a trajectory library (Liu & Atkeson, 2009) (c.f. Appendix D) is used too but its performance in OOD scenes is noticeably worse than online gradient descent.

Alternative, “softer” robust operators can be used instead of the minimum, including the Conditional Value at Risk (Embrechts et al., 2013; Rajeswaran et al., 2016, CVaR) that employs quantiles. CVaR may be more useful in cases of full support model posterior, where there may be a pessimistic but trivial model, for example, due to misspecification of the prior, p(θ), or due to the approximate inference procedure. Mean-variance optimization (Kahn et al., 2017;

Kenton et al., 2019) can be also used, aiming to directly minimise the distribution shift metric, as defined in Eqn. (3).

Next, we present a different aggregator for epistemic uncertainty that is not as pessimistic as RIP-WCM and, as found empirically, works sufficiently well too.

#### 3.3.2. MODEL AVERAGING (RIP-MA)
In the face of (epistemic) uncertainty, Bayesian decision theory (Barber, 2012) uses the predictive posterior (i.e., model averaging), which weights each model’s contribution according to its posterior probability, i.e., sRIP-MA , arg max y Z p(θ|D)log q(y|x; θ)dθ . (6)

Despite the intractability of the exact integration, the ensemble approximation used allows us to efficiently estimate and optimise the objective. We call this method robust imitative planning with model averaging (RIP-MA), where the more likely models’ impacts are up-weighted according to the predictive posterior.

From a multi-objective optimization point of view, we can interpret the log-likelihood, log q(y|x; θ), as the utility of a task θ, with importance p(θ|D), given by the posterior density. Then RIP-MA in Eqn. (6) gives the Pareto efficient solution (Barber, 2012) for the tasks θ ∈ supp p(θ|D) .

Table 1. Robust imitative planning (RIP) unified framework. The different aggregation operators applied on the posterior distribution p(θ|D), approximated with the deep ensemble (Lakshminarayanan et al., 2017) components θk.

Methods Operator ⊕ Interpretation

Imitative Models log qk=1 Sample

Best Case (RIP-BCM) maxk log qk Max

Robust Imitative Planning (ours)

Model Average (RIP-MA) P k log qk Geometric Mean

Worst Case (RIP-WCM) mink log qk Min (a) nuScenes (b) CARNOVEL

Figure 3. RIP’s (ours) robustness to OOD scenarios, compared to (Codevilla et al., 2018, CIL) and (Rhinehart et al., 2020, DIM).

## 4. Benchmarking Robustness to Novelty
We designed our experiments to answer the following questions: Q1. Can autonomous driving, imitation-learning, epistemic-uncertainty unaware methods detect distribution shifts? Q2. How robust are these methods under distribution shifts, i.e., can they recover? Q3. Does RIP’s epistemic uncertainty quantification enable identification of novel scenes? Q4. Does RIP’s explicit mechanism for recovery from distribution shifts lead to improved performance?

To that end, we conduct experiments both on real data, in




Table 2. We evaluate different autonomous driving prediction methods in terms of their robustness to distribution scene, in the nuScenes

ICRA 2020 challenge (Phan-Minh et al., 2019). We use the provided train–val–test splits and report performance on the test (i.e., out-of-sample) scenarios. A “♣” indicates methods that use LIDAR observation, as in (Rhinehart et al., 2019), and a “♦” methods that use bird-view privileged information, as in (Phan-Minh et al., 2019). A “F ” indicates that we used the results from the original paper, otherwise we used our implementation. Standard errors are in gray (via bootstrap sampling). The outperforming method is in bold.

Boston Singapore minADE1 ↓ minADE5 ↓ minFDE1 ↓ minADE1 ↓ minADE5 ↓ minFDE1 ↓

Methods (2073 scenes, 50 samples, open-loop planning) (1189 scenes, 50 samples, open-loop planning)

MTP♦F (Cui et al., 2019) 4.13 3.24 9.23 4.13 3.24 9.23

MultiPath♦F (Chai et al., 2019) 3.89 3.34 9.19 3.89 3.34 9.19

CoverNet♦F (Phan-Minh et al., 2019) 3.87 2.41 9.26 3.87 2.41 9.26

DIM♣ (Rhinehart et al., 2020) 3.64±0.05 2.48±0.02 8.22±0.13 3.82±0.04 2.95±0.01 8.91±0.08

RIP-BCM♣ (baseline, cf. Table 1) 3.53±0.04 2.37±0.01 7.92±0.09 3.57±0.02 2.70±0.01 8.39±0.03

RIP-MA♣ (ours, cf. Section 3.3.2) 3.39±0.03 2.33±0.01 7.62±0.07 3.48±0.01 2.69±0.02 8.19±0.02

RIP-WCM♣ (ours, cf. Section 3.3.1) 3.29±0.03 2.28±0.00 7.45±0.05 3.43±0.01 2.66±0.01 8.09±0.04

Section 4.1, and on simulated scenarios, in Section 4.2, comparing our method (RIP) against current state-of-the-art driving methods.

### 4.1. nuScenes
We first compare our robust planning objectives (cf. Eqn. (5– 6)) against existing state-of-the-art imitation learning methods in a prediction task (Phan-Minh et al., 2019), based on nuScenes (Caesar et al., 2019), the public, real-world, large-scale dataset for autonomous driving. Since we do not have control over the scenes split, we cannot guarantee that the evaluation is under distribution shifts, but only test out-of-sample performance, addressing question Q4.

#### 4.1.1. METRICS
For fair comparison with the baselines, we use the metrics from the ICRA 2020 nuScenes prediction challenge.

Displacement error. The quality of a plan, y, with respect to the ground truth prediction, y∗ is measured by the average displacement error, i.e.,

ADE(y) , 1T TXt=1 k st − s∗t k , (7) where y = (s1, . . . , sT ). Stochastic models, such as our imitative model, q(y|x; θ), can be evaluated based on their samples, using the minimum (over k samples) ADE (i.e., minADEk), i.e., minADEk(q) , min {yi}ki=1∼q(y|x)

ADE(yi). (8)

In prior work, Phan-Minh et al. (2019) studied minADEk for k > 1 in order to assess the quality of the generated samples from a model, q. Although we report minADEk for k = {1, 5}, we are mostly interested in the decision-making (planning) task, where the driving agent commits to a single plan, k = 1. We also study the final displacement error (FDE), or equivalently minFDE1, i.e., minFDE1(y) , k sT − s∗T k . (9)

#### 4.1.2. BASELINES
We compare our contribution to state-of-the-art methods in the nuScenes dataset: the Multiple-Trajectory Prediction (Cui et al., 2019, MTP), MultiPath (Chai et al., 2019) and CoverNet (Phan-Minh et al., 2019), all of which score a (fixed) set of trajectories, i.e., trajectory library (Liu &

Atkeson, 2009). Moreover, we implement the Deep Imitative Model (Rhinehart et al., 2020, DIM) and an optimistic variant of RIP, termed RIP-BCM and described in Table 1.

#### 4.1.3. OFFLINE FORECASTING EXPERIMENTS
We use the provided train-val-test splits from (Phan-Minh et al., 2019), for towns Boston and Singapore. For all methods we use N = 50 trajectories, and in case of both

DIM and RIP, we only optimise the “imitation prior” (cf.

Eqn. 4), since goals are not provided, running N planning procedures with different random initializations. The performance of the baselines and our methods are reported on

Table 2. We can affirmatively answer Q4 since RIP consistently outperforms the current state-of-the-art methods in out-of-sample evaluation. Moreover, Q2 can be partially answered, since the epistemic-uncertainty-unaware baselines underperformed compared to RIP.

Nonetheless, since we do not have full control over train and test splits at the ICRA 2020 challenge and hence we cannot introduce distribution shifts, we are not able to address questions Q1 and Q3 with the nuScenes benchmark. To that end, we now introduce a control benchmark based on the

CARLA driving simulator (Dosovitskiy et al., 2017).




Table 3. We evaluate different autonomous driving methods in terms of their robustness to distribution shifts, in our new benchmark,CARNOVEL. All methods are trained on CARLA Town01 using imitation learning on expert demonstrations from the autopilot (Dosovitskiy et al., 2017). A “†” indicates methods that use first-person camera view, as in (Chen et al., 2019), a “♣” methods that use LIDAR
 observation, as in (Rhinehart et al., 2020) and a “♦” methods that use the ground truth game engine state, as in (Chen et al., 2019). A “F ” indicates that we used the reference implementation from the original paper, otherwise we used our implementation. For all the scenes we chose pairs of start-destination locations and ran 10 trials with randomised initial simulator state for each pair. Standard errors are in gray (via bootstrap sampling). The outperforming method is in bold. The complete CARNOVEL benchmark results are in Appendix B.

AbnormalTurns Hills Roundabouts

Success ↑ Infra/km ↓ Success ↑ Infra/km ↓ Success ↑ Infra/km ↓

Methods (7 × 10 scenes, %) (×1e−3) (4 × 10 scenes, %) (×1e−3) (5 × 10 scenes, %) (×1e−3)

CIL♣F (Codevilla et al., 2018) 65.71±07.37 7.04±5.07 60.00±29.34 4.74±3.02 20.00±00.00 4.60±3.23

LbC†F (Chen et al., 2019) 00.00±00.00 5.81±0.58 50.00±00.00 1.61±0.15 08.00±10.95 3.70±0.72

LbC-GT♦F (Chen et al., 2019) 02.86±06.39 3.68±0.34 05.00±11.18 3.36±0.26 00.00±00.00 6.47±0.99

DIM♣ (Rhinehart et al., 2020) 74.28±11.26 5.56±4.06 70.00±10.54 6.87±4.09 20.00±09.42 6.19±4.73

RIP-BCM♣ (baseline, cf. Table 1) 68.57±09.03 7.93±3.73 75.00±00.00 5.49±4.03 06.00±09.66 6.78±7.05

RIP-MA♣ (ours, cf. Section 3.3.2) 84.28±14.20 7.86±5.70 97.50±07.90 0.26±0.54 38.00±06.32 5.48±5.56

RIP-WCM♣ (ours, cf. Section 3.3.1) 87.14±14.20 4.91±3.60 87.50±13.17 1.83±1.73 42.00±06.32 4.32±1.91

### 4.2. CARNOVEL
In order to access the robustness of AD methods to novel,

OOD driving scenarios, we introduce a benchmark, called

CARNOVEL. In particular, CARNOVEL is built on the
CARLA simulator (Dosovitskiy et al., 2017). Offline expert demonstrations1 from Town01 are provided for training.

Then, the driving agents are evaluated on a suite of OOD navigation tasks, including but not limited to roundabouts, challenging non-right-angled turns and hills, none of which are experienced during training. The CARNOVEL tasks are summarised in Appendix A. Next, we introduce metrics that quantify and help us answer questions Q1, Q3.

#### 4.2.1. METRICS
Since we are studying navigation tasks, agents should be able to reach safely pre-specified destinations. As done also in previous work (Codevilla et al., 2018; Rhinehart et al., 2020; Chen et al., 2019), the infractions per kilometre metric (i.e., violations of rules of the road and accidents per driven kilometre) measures how safely the agent navigates.

The success rate measures the percentage of successful navigations to the destination, without any infraction. However, these standard metrics do not directly reflect the methods’ performance under distribution shifts. As a result, we introduce two new metrics for quantifying the performance in out-of-training distribution tasks:

Detection score. The correlation of infractions and model’s uncertainty termed detection score is used to measure a method’s ability to predict the OOD scenes that lead to catastrophic events. As discussed by Michelmore et al. 1 using the CARLA rule-based autopilot (Dosovitskiy et al., 2017) without actuator noise. (2018), we look at time windows of 4 seconds (Taoka, 1989;

Coley et al., 2009). A method that can detect potential infractions should have high detection score.

Recovery score. The percentage of successful manoeuvres in novel scenes — where the uncertainty-unaware methods fail — is used to quantify a method’s ability to recover from distribution shifts. We refer to this metric as recovery score.

A method that is oblivious to novelty should have 0 recovery score, but positive otherwise.

#### 4.2.2. BASELINES
We compare RIP against the current state-of-the-art imitation learning methods in the CARLA benchmark (Codevilla et al., 2018; Rhinehart et al., 2020; Chen et al., 2019). Apart from DIM and RIP-BCM, discussed in Section 4.1.2, we also benchmark:

Conditional imitation learning (Codevilla et al., 2018,

CIL) is a discriminative behavioural cloning method that conditions its predictions on contextual information (e.g., LIDAR) and high-level commands (e.g., turn left, go straight).

Learning by cheating (Chen et al., 2019, LbC) is a method that builds on CIL and uses (cross-modal) distillation of privileged information (e.g., game state, rich, annotated bird-eye-view observations) to a sensorimotor agent. For reference, we also evaluate the agent who has uses privileged information directly (i.e., teacher), which we term LbC-GT.

#### 4.2.3. ONLINE PLANNING EXPERIMENTS
All the methods are trained on offline expert demonstrations from CARLA Town01. We perform 10 trials per

CARNOVEL task with randomised initial simulator state and the results are reported on Table 3 and Appendix B.


? 0 1 2 3

Number of Demos 0 50 100 (a) AbnormalTurns4-v0 0 1 2 3

Number of Demos 0 50 100 (b) BusyTown2-v0 0 1 2 3

Number of Demos 0 50 100 (c) Hills1-v0 0 1 2 3

Number of Demos 0 50 100 (d) Roundabouts1-v0

Figure 4. Adaptation scores of AdaRIP (cf. Section 5) on CARNOVEL tasks that RIP-WCM and RIP-MA (cf. Section 3) do worst. We observe that as the number of online expert demonstrations increases, the success rate improves thanks to online model adaptation.

Our robust imitative planning (i.e., RIP-WCM and RIPMA) consistently outperforms the current state-of-the-art imitation learning-based methods in novel, OOD driving scenarios. In alignment with the experimental results from nuScenes (cf. Section 4.1), we address questions Q4 and

Q2, reaching the conclusion that RIP’s epistemic uncertainty explicit mechanism for recovery improves its performance under distribution shifts, compared to epistemic uncertaintyunaware methods. As a result, RIP’s recovery score (cf.

Section 4.2.1) is higher than the baselines.

Towards distribution shift detection and answering questions

Q1 and Q3, we collect 50 scenes for each method that led to a crash, record the uncertainty 4 seconds (Taoka, 1989) before the accident and assert if the uncertainties can be used for detection. RIP’s (ours) predictive variance (cf.

Eqn. (3)) serves as a useful detector, while DIM’s (Rhinehart et al., 2020) negative log-likelihood was unable to detect catastrophes. The results are illustrated on Figure 5.

Despite RIP’s improvement over current state-of-the-art methods with 97.5% success rate and 0.26 infractions per driven kilometre (cf. Table 3), the safety-critical nature of the task mandates higher performance. Towards this goal, we introduce an online adaptation variant of RIP. 0 20 40 60 80 100



Figure 5. Uncertainty estimators as indicators of catastrophes on

CARNOVEL. We collect 50 scenes for each model that led to a
 crash, record the uncertainty 4 seconds (Taoka, 1989) before the accident and assert if the uncertainties can be used for detection.

RIP’s (ours) predictive variance (in blue, cf. Eqn. (3)) serves as a useful detector, while DIM’s (Rhinehart et al., 2020) negative loglikelihood (in orange) cannot be used for detecting catastrophes.

## 5. Adaptive Robust Imitative Planning
We empirically observe that the quantification of epistemic uncertainty and its use in the RIP objectives is not always sufficient to recover from shifts away from the training distribution (cf. Section 4.2.3). However, we can use uncertainty estimates to ask the human driver to take back control or default to a safe policy, avoiding potential infractions. In the former case, the human driver’s behaviors can be recorded and used to reduce RIP’s epistemic uncertainty via online adaptation. The epistemic uncertainty is reducible and hence it can be eliminated, provided enough demonstrations.

We propose an adaptive variant of RIP, called AdaRIP, which uses the epistemic uncertainty estimates to decide when to query the human driver for feedback, which is used to update its parameters online, adapting to arbitrary new driving scenarios. AdaRIP relies on external, online feedback from an expert demonstrator2 , similar to DAgger (Ross et al., 2011) and its variants (Zhang & Cho, 2016; Cronrath et al., 2018).

However, unlike this prior work, AdaRIP uses an epistemic uncertainty-aware acquisition mechanism. AdaRIP’s pseudocode is given in Algorithm 1.

The uncertainty (i.e., variance) threshold, τ , is calibrated on a validation dataset, such that it matches a pre-specified level of false negatives, using a similar analysis to Figure 5.

## 6. Benchmarking Adaptation
The goal of this section is to provide experimental evidence for answering the following questions: Q5. Can RIP’s epistemic-uncertainty estimation be used for efficiently querying an expert for online feedback (i.e., demonstrations)? Q6. Does AdaRIP’s online adaptation mechanism improve success rate?

We evaluate AdaRIP on CARNOVEL tasks, where the

CARLA autopilot (Dosovitskiy et al., 2017) is queried for demonstrations online when the predictive variance (cf.

Eqn. (3)) exceeds a threshold, chosen according to RIP’s detection score, (cf. Figure 5). We measure performance 2AdaRIP is also compatible with other feedback mechanisms, such as expert preferences (Christiano et al., 2017) or explicit reward functions (de Haan et al., 2019).

Success, % Binary Accuracy


? novel (OOD) in-distribution (a) Data distribution (b) Domain Randomization

ENC (c) Domain adaptation (d) Online adaptation

Figure 6. Common approaches to distribution shift, as in (a) there are novel (OOD) points that are outside the support of the training data: (b) domain randomization (e.g., Sadeghi & Levine (2016)) covers the data distribution by exhaustively sampling configurations from a simulator; (c) domain adaptation (e.g., McAllister et al. (2019)) projects (or encodes) the (OOD) points to the in-distribution space and (d) online adaptation (e.g., Ross et al. (2011)) progressively expands the in-distribution space by incorporating online, external feedback. according to the:

Adaptation score. The improvement in success rate as a function of number of online expert demonstrations is used to measure a method’s ability to adapt efficiently online. We refer to this metric as adaptation score. A method that can adapt online should have a positive adaptation score.

AdaRIP’s performance on the most challenging CARNOVEL tasks is summarised in Figure 4, where, as expected, the success rate improves as the number of online demonstrations increases. Qualitative examples are illustrated in Appendix C.

Although AdaRIP can adapt to any distribution shift, it is prone to catastrophic forgetting and sample-inefficiency, as many online methods (French, 1999). In this paper, we only demonstrate AdaRIP’s efficacy to adapt under distribution shifts and do not address either of these limitations.

Future work lies in providing a practical, sample-efficient algorithm to be used in conjunction with the AdaRIP framework. Methods for efficient (e.g., few-shot or zero-shot) and safe adaptation (Finn et al., 2017; Zhou et al., 2019) are orthogonal to AdaRIP and hence any improvement in these fields could be directly used for AdaRIP.

## 7. Related Work
### Imitation learning. 
Learning from expert demonstrations (i.e., imitation learning (Widrow & Smith, 1964; Pomerleau, 1989, IL)) is an attractive framework for sequential decisionmaking in safety-critical domains such as autonomous driving, where trial and error learning has little to no safety guarantees during training. A plethora of expert driving demonstrations has been used for IL (Caesar et al., 2019; Sun et al., 2019; Kesten et al., 2019) since a model mimicking expert demonstrations can simply learn to stay in “safe”, expert-like parts of the state space and no explicit reward function need be specified.

模仿学习。 从专家示范中学习(即模仿学习(Widrow & Smith，1964 ;Pomerleau，1989，伊利诺伊州))是在自动驾驶等安全关键领域进行顺序决策的一个有吸引力的框架，在这些领域，试错学习几乎没有安全性 训练期间的保证。 大量专家驾驶演示已用于 IL(Caesar et al., 2019 ;Sun et al., 2019 ;Kesten et al., 2019)，因为模仿专家演示的模型可以简单地学会保持“安全”，专家 -类似于状态空间的部分，不需要指定明确的奖励函数。

On the one hand, behavioural cloning approaches (Liang et al., 2018; Sauer et al., 2018; Li et al., 2018; Codevilla et al., 2018; 2019; Chen et al., 2019) fit commandconditioned discriminative sequential models to expert demonstrations, which are used in deployment to produce expert-like trajectories. On the other hand, Rhinehart et al. (2020) proposed command-unconditioned expert trajectory density models which are used for planning trajectories that both satisfy the goal constraints and are likely under the expert model. However, both of these approaches fit pointestimates to their parameters, thus do not quantify their model (epistemic) uncertainty, as explained next. This is especially problematic when estimating what an expert would or would not do in unfamiliar, OOD scenes. In contrast, our methods, RIP and AdaRIP, does quantify epistemic uncertainty in order to both improve planning performance and triage situations in which an expert should intervene.

一方面，行为克隆方法(Liang et al., 2018 ;Sauer et al., 2018 ;Li et al., 2018 ;Codevilla et al., 2018 ;2019 ;Chen et al., 2019)适合命令条件判别顺序 模型到专家演示，用于部署以产生类似专家的轨迹。 另一方面，Rhinehart et al. (2020) 提出了命令无条件专家轨迹密度模型，用于规划既满足目标约束又可能在专家模型下的轨迹。 然而，这两种方法都将点估计拟合到它们的参数，因此不量化它们的模型(认知)不确定性，如下所述。 在估计专家在不熟悉的 OOD 场景中会做什么或不会做什么时，这尤其成问题。 相比之下，我们的方法 RIP 和 AdaRIP 确实量化了认知不确定性，以提高规划绩效和专家应该干预的分类情况。

### Novelty detection & epistemic uncertainty. 
A principled means to capture epistemic uncertainty is with Bayesian inference to compute the predictive distribution. However, evaluating the posterior p(θ|D) with exact inference is intractable for non-trivial models (Neal, 2012). Approximate inference methods (Graves, 2011; Blundell et al., 2015; Gal & Ghahramani, 2016; Hern´andez-Lobato & Adams, 2015) have been introduced that can efficiently capture epistemic uncertainty. One approximation for epistemic uncertainty in deep models is model ensembles (Lakshminarayanan et al., 2017; Chua et al., 2018). Prior work by Kahn et al. (2017) and Kenton et al. (2019) use ensembles of deep models to detect and avoid catastrophic actions in navigation tasks, although they can not recover from or adapt to distribution shifts. Our epistemic uncertainty-aware planning objective, RIP, instead, managed to recover from some distribution shifts, as shown experimentally in Section 4.

新颖性检测和认知不确定性。 捕获认知不确定性的一种原则性方法是使用贝叶斯推理来计算预测分布。 然而，对于非平凡模型，使用精确推理评估后验 p(θ|D) 是很棘手的 (Neal, 2012)。 近似推理方法 (Graves, 2011; Blundell et al., 2015; Gal & Ghahramani, 2016; Hern´andez-Lobato & Adams, 2015) 已经被引入，可以有效地捕捉认知不确定性。 深度模型中认知不确定性的一种近似是模型集成(Lakshminarayanan et al., 2017 ;Chua et al., 2018)。 Kahn et al 之前的工作。 (2017) 和 Kenton et al. (2019) 使用深度模型的集合来检测和避免导航任务中的灾难性行为，尽管它们无法从分布变化中恢复或适应分布变化。 相反，我们的认知不确定性感知规划目标 RIP 设法从一些分布变化中恢复，如第 4 节中的实验所示。

### Coping with distribution shift. 
Strategies to cope with distribution shift include (a) domain randomization; (b) domain adaptation and (c) online adaptation. Domain randomization assumes access to a simulator and exhaustively searches for configurations that cover all the data distribution support in order to eliminate OOD scenes, as illustrated in Figure 6b. This approach has been successfully used in simple robotic tasks (Sadeghi & Levine, 2016; OpenAI et al., 2018; Akkaya et al., 2019) but it is impractical for use in large, real-world tasks, such as AD. Domain adaptation and bisimulation (Castro & Precup, 2010), depicted in Figure 6c, tackle OOD points by projecting them back to in-distribution points, that are “close” to training points according to some metric. Despite its success in simple visual tasks (McAllister et al., 2019), it has no guarantees under arbitrary distribution shifts. In contrast, online learning methods (Cesa-Bianchi & Lugosi, 2006; Ross et al., 2011; Zhang & Cho, 2016; Cronrath et al., 2018) have no-regret guarantees and, provided frequent expert supervision, they asymptotically cover the whole data distribution’s support, adaptive to any distribution shift, as shown in Figure 6d. In order to continually cope with distribution shift, a learner must receive interactive feedback (Ross et al., 2011), however, the frequency of this costly feedback should be minimised. Our epistemic-uncertainty-aware method, Robust Imitative Planning can cope with some OOD events, thereby reducing the system’s dependency on expert feedback, and can use this uncertainty to decide when it cannot cope–when the expert must intervene.

应对分配转变。 应对分布转变的策略包括(a)域随机化;  (b) 域适应和 (c) 在线适应。 域随机化假定访问模拟器并详尽地搜索涵盖所有数据分布支持的配置，以消除 OOD 场景，如图 6b 所示。 这种方法已成功用于简单的机器人任务(Sadeghi & Levine，2016 ;OpenAI et al., 2018 ;Akkaya et al., 2019)，但用于大型现实世界任务(例如 AD)是不切实际的。 域适应和双向模拟(Castro & Precup，2010)，如图 6c 所示，通过将 OOD 点投影回分布点来解决这些点，这些点根据某些指标“接近”训练点。 尽管它在简单的视觉任务中取得了成功(McAllister et al., 2019)，但它在任意分布变化下无法保证。 相比之下，在线学习方法 (Cesa-Bianchi & Lugosi, 2006; Ross et al., 2011; Zhang & Cho, 2016; Cronrath et al., 2018) 具有无悔保证，并且提供频繁的专家监督，它们渐近覆盖 整个数据分布的支持，适应任何分布变化，如图6d所示。 为了持续应对分布转变，学习者必须接收交互式反馈(Ross et al., 2011)，但是，应该尽量减少这种代价高昂的反馈的频率。 我们的认知不确定性感知方法 Robust Imitative Planning 可以应对一些 OOD 事件，从而减少系统对专家反馈的依赖，并可以利用这种不确定性来决定何时无法应对——何时专家必须干预。

Algorithm 1: Adaptive Robust Imitative Planning
算法 1：自适应稳健模仿规划

### Current benchmarks. 
We are interested in the control problem, where AD agents get deployed in reactive environments and make sequential decisions. The CARLA Challenge (Ros et al., 2019; Dosovitskiy et al., 2017; Codevilla et al., 2019) is an open-source benchmark for control in AD. It is based on 10 traffic scenarios from the NHTSA pre-crash typology (National Highway Traffic Safety Administration, 2007) to inject challenging driving situations into traffic patterns encountered by AD agents. The methods are only assessed in terms of their generalization to weather conditions, the initial state of the simulation (e.g., the start and goal locations, and the random seed of other agents.) and the traffic density (i.e., empty town, regular traffic and dense traffic).

当前基准。 我们对控制问题感兴趣，其中 AD 智能体部署在反应性环境中并做出顺序决策。 CARLA 挑战赛(Ros et al., 2019 ;Dosovitskiy et al., 2017 ;Codevilla et al., 2019)是 AD 控制的开源基准。 它基于 NHTSA 预碰撞类型学(国家公路交通安全管理局，2007)的 10 种交通场景，将具有挑战性的驾驶情况注入 AD 智能体遇到的交通模式中。 这些方法仅根据它们对天气条件的泛化、模拟的初始状态(例如，开始和目标位置，以及其他智能体的随机种子)和交通密度(即空城、正常交通)进行评估 和密集的交通)。

Despite these challenging scenarios selected in the CARLA Challenge, the agents are allowed to train on the same scenarios in which they evaluated, and so the robustness to distributional shift is not assessed. Consequently, both Chen et al. (2019) and Rhinehart et al. (2020) manage to solve the CARLA Challenge with almost 100% success rate, when trained in Town01 and tested in Town02. However, both methods score almost 0% when evaluated in Roundabouts due to the presence of OOD road morphologies, as discussed in Section 4.2.3.

尽管在 CARLA 挑战赛中选择了这些具有挑战性的场景，但允许智能体在他们评估的相同场景中进行训练，因此不评估对分布转变的稳健性。 因此，陈et al. (2019) 和 Rhinehart et al. (2020) 在 Town01 中训练并在 Town02 中测试时，以几乎 100% 的成功率成功解决了 CARLA 挑战。 然而，由于存在 OOD 道路形态，这两种方法在 Roundabouts 中进行评估时得分几乎为 0%，如第 4.2.3 节所述。

## 8. Summary and Conclusions
To summarise, in this paper, we studied autonomous driving agents in out-of-training distribution tasks (i.e. under distribution shifts). We introduced an epistemic uncertaintyaware planning method, called robust imitative planning (RIP), which can detect and recover from distribution shifts, as shown experimentally in a real prediction task, nuScenes, and a driving simulator, CARLA. We presented an adaptive variant (AdaRIP) which uses RIP’s epistemic uncertainty estimates to efficiently query the expert for online feedback and adapt its model parameters online.

总而言之，在本文中，我们研究了训练外分配任务(即分配转移)中的自动驾驶智能体。 我们引入了一种认知不确定性感知规划方法，称为稳健模仿规划 (RIP)，它可以检测分布变化并从中恢复，如真实预测任务 nuScenes 和驾驶模拟器 CARLA 中的实验所示。 我们提出了一种自适应变体 (AdaRIP)，它使用 RIP 的认知不确定性估计来有效地查询专家的在线反馈并在线调整其模型参数。

We also introduced and open-sourced an autonomous car novel-scene benchmark, termed CARNOVEL, to assess the robustness of driving agents to a suite of OOD tasks.

我们还引入并开源了一个名为 CARNOVEL 的自动驾驶汽车新场景基准，以评估驾驶智能体对一组 OOD 任务的稳健性。

## Acknowledgements
This work was supported by the UK EPSRC CDT in Autonomous Intelligent Machines and Systems (grant reference EP/L015897/1). This project has received funding from the Office of Naval Research, the DARPA Assured Autonomy Program, and ARL DCIST CRA W911NF-17-2- 0181, Microsoft Azure and Intel AI Labs.

这项工作得到了英国 EPSRC CDT 在自主智能机器和系统方面的支持(授权参考 EP/L015897/1)。 该项目已获得海军研究办公室、DARPA 保证自治计划和 ARL DCIST CRA W911NF-17-2-0181、Microsoft Azure 和英特尔人工智能实验室的资助。

## References
* Akkaya, I., Andrychowicz, M., Chociej, M., Litwin, M.,McGrew, B., Petron, A., Paino, A., Plappert, M., Powell,G., Ribas, R., et al. Solving Rubik’s cube with a robothand. arXiv preprint arXiv:1910.07113, 2019.
* Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schul￾man, J., and Man´e, D. Concrete problems in AI safety.
* arXiv preprint arXiv:1606.06565, 2016.
* Barber, D. Bayesian reasoning and machine learning. Cam￾bridge University Press, 2012.
* Bellman, R. E. Adaptive control processes: a guided tour.
* Princeton university press, 2015.
* Bishop, C. M. Mixture density networks. 1994.
* Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra,D. Weight uncertainty in neural networks. arXiv preprintarXiv:1505.05424, 2015.
* Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E.,Xu, Q., Krishnan, A., Pan, Y., Baldan, G., and Beijbom, O.
* nuscenes: A multimodal dataset for autonomous driving.
* arXiv preprint arXiv:1903.11027, 2019.
* Castro, P. S. and Precup, D. Using bisimulation for pol￾icy transfer in MDPs. In AAAI Conference on ArtificialIntelligence, 2010.
* Cesa-Bianchi, N. and Lugosi, G. Prediction, learning, andgames. Cambridge University Press, 2006.
* Chai, Y., Sapp, B., Bansal, M., and Anguelov, D. Multipath:Multiple probabilistic anchor trajectory hypotheses forbehavior prediction. arXiv preprint arXiv:1910.05449,2019.
* Chen, D., Zhou, B., Koltun, V., and Kr¨ahenb¨uhl, P. Learningby cheating. arXiv preprint arXiv:1912.12294, 2019.
* Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg,S., and Amodei, D. Deep reinforcement learning fromhuman preferences. In Advances in Neural InformationProcessing Systems, pp. 4299–4307, 2017.
* Chua, K., Calandra, R., McAllister, R., and Levine, S. Deepreinforcement learning in a handful of trials using proba￾bilistic dynamics models. In Neural Information Process￾ing Systems (NeurIPS), pp. 4754–4765, 2018.
* Codevilla, F., Miiller, M., L´opez, A., Koltun, V., and Doso￾vitskiy, A. End-to-end driving via conditional imitationlearning. In International Conference on Robotics andAutomation (ICRA), pp. 1–9. IEEE, 2018.
* Codevilla, F., Santana, E., L´opez, A. M., and Gaidon, A.
* Exploring the limitations of behavior cloning for au￾tonomous driving. In International Conference on Com￾puter Vision (ICCV), pp. 9329–9338, 2019.
* Coley, G., Wesley, A., Reed, N., and Parry, I. Driver reactiontimes to familiar, but unexpected events. TRL PublishedProject Report, 2009.
* Cronrath, C., Jorge, E., Moberg, J., Jirstrand, M.,and Lennartson, B. BAgger: A Bayesian al￾gorithm for safe and query-efficient imitationlearning. https://personalrobotics.cs.
* washington.edu/workshops/mlmp2018/assets/docs/24_CameraReadySubmission_180928_BAgger.pdf, 2018.
* Cui, H., Radosavljevic, V., Chou, F.-C., Lin, T.-H., Nguyen,T., Huang, T.-K., Schneider, J., and Djuric, N. Mul￾timodal trajectory predictions for autonomous drivingusing deep convolutional networks. In 2019 Interna￾tional Conference on Robotics and Automation (ICRA),pp. 2090–2096. IEEE, 2019.
* de Haan, P., Jayaraman, D., and Levine, S. Causal confusionin imitation learning. In Neural Information ProcessingSystems (NeurIPS), pp. 11693–11704, 2019.
* Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., andKoltun, V. CARLA: An open urban driving simulator.
* arXiv preprint arXiv:1711.03938, 2017.
* Du, Y., Lin, T., and Mordatch, I. Model based planning withenergy based models. arXiv preprint arXiv:1909.06878,2019.
* Embrechts, P., Kl¨uppelberg, C., and Mikosch, T. Modellingextremal events: for insurance and finance, volume 33.
* Springer Science & Business Media, 2013.
* Finn, C., Abbeel, P., and Levine, S. Model-agnostic meta￾learning for fast adaptation of deep networks. In Inter￾national Conference on Machine Learning (ICML), pp.
* 1126–1135, 2017.
* French, R. M. Catastrophic forgetting in connectionist net￾works. Trends in cognitive sciences, 3(4):128–135, 1999.
* Gal, Y. and Ghahramani, Z. Dropout as a Bayesian approx￾imation: Representing model uncertainty in deep learn￾ing. In International Conference on Machine Learning(ICML), pp. 1050–1059, 2016.
* Graves, A. Practical variational inference for neuralnetworks. In Neural Information Processing Systems(NeurIPS), pp. 2348–2356, 2011.
* Hern´andez-Lobato, J. M. and Adams, R. Probabilistic back￾propagation for scalable learning of Bayesian neural net￾works. In International Conference on Machine Learning(ICML), pp. 1861–1869, 2015.
* Kahn, G., Villaflor, A., Pong, V., Abbeel, P., and Levine, S.
* Uncertainty-aware reinforcement learning for collisionavoidance. arXiv preprint arXiv:1702.01182, 2017.
* Kenton, Z., Filos, A., Evans, O., and Gal, Y. Generalizingfrom a few environments in safety-critical reinforcementlearning. arXiv preprint arXiv:1907.01475, 2019.
* Kesten, R., Usman, M., Houston, J., Pandya, T., Nadhamuni,K., Ferreira, A., Yuan, M., Low, B., Jain, A., Ondruska,P., Omari, S., Shah, S., Kulkarni, A., Kazakova, A., Tao,C., Platinsky, L., Jiang, W., and Shet, V. Lyft level 5 avdataset 2019, 2019. URL https://level5.lyft.
* com/dataset/.
* Kingma, D. P. and Ba, J. Adam: A method for stochasticoptimization. arXiv preprint arXiv:1412.6980, 2014.
* Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simpleand scalable predictive uncertainty estimation using deepensembles. In Neural Information Processing Systems(NeurIPS), pp. 6402–6413, 2017.
* Leike, J., Martic, M., Krakovna, V., Ortega, P. A., Everitt,T., Lefrancq, A., Orseau, L., and Legg, S. AI safetygridworlds. arXiv preprint arXiv:1711.09883, 2017.
* Li, Z., Motoyoshi, T., Sasaki, K., Ogata, T., and Sugano, S.
* Rethinking self-driving: Multi-task knowledge for bettergeneralization and accident explanation ability. arXivpreprint arXiv:1809.11100, 2018.
* Liang, X., Wang, T., Yang, L., and Xing, E. Cirl: Control￾lable imitative reinforcement learning for vision-basedself-driving. In European Conference on Computer Vision(ECCV), pp. 584–599, 2018.
* Liu, C. and Atkeson, C. G. Standing balance control usinga trajectory library. In 2009 IEEE/RSJ International Con￾ference on Intelligent Robots and Systems, pp. 3031–3036.
* IEEE, 2009.
* McAllister, R., Kahn, G., Clune, J., and Levine, S. Robust￾ness to out-of-distribution inputs via task-aware genera￾tive uncertainty. In International Conference on Roboticsand Automation (ICRA), pp. 2083–2089. IEEE, 2019.
* Michelmore, R., Kwiatkowska, M., and Gal, Y. Evaluat￾ing uncertainty quantification in end-to-end autonomousdriving control. arXiv preprint arXiv:1811.06817, 2018.
* National Highway Traffic Safety Administration. Pre-crashscenario typology for crash avoidance research, 2007.
* URL https://www.nhtsa.gov/sites/nhtsa.
* dot.gov/files/pre-crash_scenario_typology-final_pdf_version_5-2-07.
* pdf.
* Neal, R. M. Bayesian learning for neural networks, volume118. Springer Science & Business Media, 2012.
* OpenAI, M. A., Baker, B., Chociej, M., J´ozefowicz, R., Mc￾Grew, B., Pachocki, J., Petron, A., Plappert, M., Powell,G., Ray, A., et al. Learning dexterous in-hand manipula￾tion. arXiv preprint arXiv:1808.00177, 2018.
* Phan-Minh, T., Grigore, E. C., Boulton, F. A., Beijbom,O., and Wolff, E. M. Covernet: Multimodal behav￾ior prediction using trajectory sets. arXiv preprintarXiv:1911.10298, 2019.
* Pomerleau, D. A. Alvinn: An autonomous land vehiclein a neural network. In Neural Information ProcessingSystems (NeurIPS), pp. 305–313, 1989.
* Quionero-Candela, J., Sugiyama, M., Schwaighofer, A., andLawrence, N. D. Dataset shift in machine learning. MITPress, 2009.
* Rajeswaran, A., Ghotra, S., Ravindran, B., and Levine,S. Epopt: Learning robust neural network policies us￾ing model ensembles. arXiv preprint arXiv:1610.01283,2016.
* Rezende, D. J. and Mohamed, S. Variational inference withnormalizing flows. arXiv preprint arXiv:1505.05770,2015.
* Rhinehart, N., Kitani, K. M., and Vernaza, P. R2P2: Areparameterized pushforward policy for diverse, precisegenerative path forecasting. In European Conference onComputer Vision (ECCV), pp. 772–788, 2018.
* Rhinehart, N., McAllister, R., Kitani, K., and Levine, S.
* PRECOG: Prediction conditioned on goals in visualmulti-agent settings. International Conference on Com￾puter Vision, 2019.
* Rhinehart, N., McAllister, R., and Levine, S. Deep imitativemodels for flexible inference, planning, and control. InInternational Conference on Learning Representations(ICLR), April 2020.
* Ros, G., Koltun, V., Codevilla, F., and Lopez, M. A. CARLAchallenge, 2019. URL https://carlachallenge.
* org.
* Ross, S., Gordon, G., and Bagnell, D. A reduction ofimitation learning and structured prediction to no-regretonline learning. In Artificial Intelligence and Statistics(AISTATS), pp. 627–635, 2011.
* Sadeghi, F. and Levine, S. Cad2rl: Real single-imageflight without a single real image. arXiv preprintarXiv:1611.04201, 2016.
* Sauer, A., Savinov, N., and Geiger, A. Conditional affor￾dance learning for driving in urban environments. arXivpreprint arXiv:1806.06498, 2018.
* Snoek, J., Ovadia, Y., Fertig, E., Lakshminarayanan, B.,Nowozin, S., Sculley, D., Dillon, J., Ren, J., and Nado,Z. Can you trust your model’s uncertainty? evaluatingpredictive uncertainty under dataset shift. In Neural Infor￾mation Processing Systems (NeurIPS), pp. 13969–13980,2019.
* Sugiyama, M. and Kawanabe, M. Machine learning in non￾stationary environments: Introduction to covariate shiftadaptation. MIT press, 2012.
* Sun, P., Kretzschmar, H., Dotiwalla, X., Chouard, A., Pat￾naik, V., Tsui, P., Guo, J., Zhou, Y., Chai, Y., Caine,B., et al. Scalability in perception for autonomousdriving: An open dataset benchmark. arXiv preprintarXiv:1912.04838, 2019.
* Tang, Y. C., Zhang, J., and Salakhutdinov, R. Worst casespolicy gradients. arXiv preprint arXiv:1911.03618, 2019.
* Taoka, G. T. Brake reaction times of unalerted drivers. ITEjournal, 59(3):19–21, 1989.
* Uria, B., Cˆot´e, M.-A., Gregor, K., Murray, I., andLarochelle, H. Neural autoregressive distribution esti￾mation. The Journal of Machine Learning Research, 17(1):7184–7220, 2016.
* Wald, A. Contributions to the theory of statistical estimationand testing hypotheses. The Annals of MathematicalStatistics, 10(4):299–326, 1939.
* Widrow, B. and Smith, F. W. Pattern-recognizing controlsystems, 1964.
* Zhang, J. and Cho, K. Query-efficient imitation learn￾ing for end-to-end autonomous driving. arXiv preprintarXiv:1605.06450, 2016.
* Zhou, A., Jang, E., Kappler, D., Herzog, A., Khansari, M.,Wohlhart, P., Bai, Y., Kalakrishnan, M., Levine, S., andFinn, C. Watch, try, learn: Meta-learning from demon￾strations and reward. arXiv preprint arXiv:1906.03352,2019.


## Appendix
### A. CARNOVEL: Suite of Tasks Under Distribution Shift
 (a) AbnormalTurns0-v0 (b) AbnormalTurns1-v0 (c) AbnormalTurns2-v0 (d) AbnormalTurns3-v0 (e) AbnormalTurns4-v0 (f) AbnormalTurns5-v0 (g) AbnormalTurns6-v0 (h) BusyTown0-v0 (i) BusyTown1-v0 (j) BusyTown2-v0 (k) BusyTown3-v0 (l) BusyTown4-v0 (m) BusyTown5-v0 (n) BusyTown6-v0 (o) BusyTown7-v0  (p) BusyTown8-v0 (q) BusyTown9-v0 (r) BusyTown10-v0 (s) Hills0-v0 (t) Hills1-v0 (u) Hills2-v0 (v) Hills3-v0 (w) Roundabouts0-v0 (x) Roundabouts1-v0 (y) Roundabouts2-v0 (z) Roundabouts3-v0 (aa) Roundabouts4-v0 

### B. Experimental Results on CARNOVEL
Table 4. We evaluate different autonomous driving methods in terms of their robustness to distribution shifts, in our new benchmark,

CARNOVEL. All methods are trained on CARLA Town01 using imitation learning on expert demonstrations from the autopilot (Dosovitskiy et al., 2017). A “†” indicates methods that use first-person camera view, as in (Chen et al., 2019), a “♣” methods that use LIDAR
 observation, as in (Rhinehart et al., 2020) and a “♦” methods that use the ground truth game engine state, as in (Chen et al., 2019). A “F ” indicates that we used the reference implementation from the original paper, otherwise we used our implementation. For all the scenes we chose pairs of start-destination locations and ran 10 trials with randomized initial simulator state for each pair. Standard errors are in gray (via bootstrap sampling). The outperforming method is in bold.

AbnormalTurns BusyTown

Success ↑ Infra/km ↓ Distance ↑ Success ↑ Infra/km ↓ Distance ↑

Methods (7 × 10 scenes, %) (×1e−3) (m) (11 × 10 scenes, %) (×1e−3) (m)

CIL♣F (Codevilla et al., 2018) 65.71±07.37 07.04±05.07 128±020 05.45±06.35 11.49±03.66 217±033

LbC†F (Chen et al., 2019) 00.00±00.00 05.81±00.58 208±004 20.00±13.48 03.96±00.15 374±016

LbC-GT♦F (Chen et al., 2019) 02.86±06.39 03.68±00.34 217±033 65.45±07.60 02.59±00.02 400±006

DIM♣ (Rhinehart et al., 2020) 74.28±11.26 05.56±04.06 108±017 47.13±14.54 08.47±05.22 175±026

RIP-BCM♣ (baseline, cf. Table 1) 68.57±09.03 07.93±03.73 096±017 50.90±20.64 03.74±05.52 175±031

RIP-MA♣ (ours, cf. Section 3.3.2) 84.28±14.20 07.86±05.70 102±015 64.54±23.25 05.86±03.99 170±033

RIP-WCM♣ (ours, cf. Section 3.3.1) 87.14±14.20 04.91±03.60 102±021 62.72±05.16 03.17±02.04 167±021

Hills Roundabouts

Success ↑ Infra/km ↓ Distance ↑ Success ↑ Infra/km ↓ Distance ↑

Methods (4 × 10 scenes, %) (×1e−3) (m) (5 × 10 scenes, %) (×1e−3) (m)

CIL♣F (Codevilla et al., 2018) 60.00±29.34 04.74±03.02 219±034 20.00±00.00 03.60±03.23 269±021

LbC†F (Chen et al., 2019) 50.00±00.00 01.61±00.15 541±101 08.00±10.95 03.70±00.72 323±043

LbC-GT♦F (Chen et al., 2019) 05.00±11.18 03.36±00.26 312±020 00.00±00.00 06.47±00.99 123±018

DIM♣ (Rhinehart et al., 2020) 70.00±10.54 06.87±04.09 195±012 20.00±09.42 06.19±04.73 240±044

RIP-BCM♣ (baseline, cf. Table 1) 75.00±00.00 05.49±04.03 191±013 06.00±09.66 06.78±07.05 251±027

RIP-MA♣ (ours, cf. Section 3.3.2) 97.50±07.90 00.26±00.54 196±013 38.00±06.32 05.48±05.56 271±047

RIP-WCM♣ (ours, cf. Section 3.3.1) 87.50±13.17 01.83±01.73 191±006 42.00±06.32 04.32±01.91 217±030 

### C. AdaRIP Examples
 (Normalized) Uncertainty (a) RIP (b) AdaRIP

Figure 8. Examples where the non-adaptive method (a) fails to recover from a distribution shift, despite it being able to detect it. The adaptive method (b) queries the human driver when uncertain (dark red), then uses the online demonstrations for updating its model, resulting into confident (light red, white) and safe trajectories. 

### D. Online Planning with a Trajectory Library
In the absence of scalable global optimizers, we search the trajectory space in Eqn. (4) by restricting the search space to a trajectory library (Liu & Atkeson, 2009), TY, a finite set of fixed trajectories. In this work, we perform K-means clustering of the expert plan’s from the training distribution and keep 64 of the centroids, as illustrated in Figure 9. Therefore we efficiently solve a search problem over a discrete space rather than an optimization problem of continuous variables. The modified objective is: yG

RIP ≈ arg max y∈TY ⊕ θ∈supp p(θ|D) log p(y|G, x; θ) (10)

Solving for Eqn. (10) results in ×20 improvement in runtime compared to the gradient descent alternative. Although in in-distribution scenes solving Eqn. (10) over Eqn. (4) does not deteriorate perfomance, in out-of-distribution scenes the trajectory library, TY, is not useful. Therefore in the experiments (c.f. Section 4.2.3) we used online gradient-descent.

Future work lies in developing a hybrid optimization method that takes advantage of the speedup the trajectory library provides without a decrease in performance in out-of-distribution scenarios. (a) Trajectories (b) K = 64 (c) K = 128 (d) K = 1024

Figure 9. Our trajectory library from CARLA’s autopilot demonstrations, 4 seconds.   
