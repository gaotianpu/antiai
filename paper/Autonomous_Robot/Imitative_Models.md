# Deep Imitative Models for Flexible Inference, Planning, and Control
用于灵活推理、规划和控制的深度模仿模型 https://arxiv.org/abs/1810.06544

## Abstract
Imitation Learning (IL) is an appealing approach to learn desirable autonomous behavior. However, directing IL to achieve arbitrary goals is difficult. In contrast, planning-based algorithms use dynamics models and reward functions to achieve goals. Yet, reward functions that evoke desirable behavior are often difficult to specify. In this paper, we propose “Imitative Models” to combine the benefits of IL and goal-directed planning. Imitative Models are probabilistic predictive models of desirable behavior able to plan interpretable expert-like trajectories to achieve specified goals. We derive families of flexible goal objectives, including constrained goal regions, unconstrained goal sets, and energy-based goals. We show that our method can use these objectives to successfully direct behavior. Our method substantially outperforms six IL approaches and a planning-based approach in a dynamic simulated autonomous driving task, and is efficiently learned from expert demonstrations without online data collection. We also show our approach is robust to poorly specified goals, such as goals on the wrong side of the road.

模仿学习 (IL) 是学习理想的自主行为的一种有吸引力的方法。 然而，指导 IL 实现任意目标是困难的。 相比之下，基于规划的算法使用动态模型和奖励函数来实现目标。 然而，唤起理想行为的奖励函数通常很难指定。 在本文中，我们提出了“模仿模型”以结合 IL 和目标导向规划的优势。 模仿模型是期望行为的概率预测模型，能够规划可解释的类似专家的轨迹以实现指定目标。 我们派生了一系列灵活的目标目标，包括约束目标区域、无约束目标集和基于能量的目标。 我们表明我们的方法可以使用这些目标来成功地指导行为。 我们的方法在动态模拟自动驾驶任务中大大优于六种 IL 方法和基于规划的方法，并且无需在线数据收集即可从专家演示中有效地学习。 我们还表明我们的方法对于指定不当的目标是稳健的，例如道路错误一侧的目标。

## 1 Introduction
Imitation learning (IL) is a framework for learning a model to mimic behavior. At test-time, the model pursues its best-guess of desirable behavior. By letting the model choose its own behavior, we cannot direct it to achieve different goals. While work has augmented IL with goal conditioning (Dosovitskiy & Koltun, 2016; Codevilla et al., 2018), it requires goals to be specified during training, explicit goal labels, and are simple (e.g., turning). In contrast, we seek flexibility to achieve general goals for which we have no demonstrations.

模仿学习 (IL) 是学习模型以模仿行为的框架。 在测试时，该模型追求其对理想行为的最佳猜测。 通过让模型选择自己的行为，我们不能指导它实现不同的目标。 虽然工作通过目标调节增强了 IL(Dosovitskiy & Koltun，2016 ;Codevilla et al., 2018)，但它需要在训练期间指定目标、明确的目标标签，并且很简单(例如，转弯)。 相反，我们寻求灵活性以实现我们没有示范的总体目标。

In contrast to IL, planning-based algorithms like model-based reinforcement learning (MBRL) methods do not require expert demonstrations. MBRL can adapt to new tasks specified through reward functions (Kuvayev & Sutton, 1996; Deisenroth & Rasmussen, 2011). The “model” is a dynamics model, used to plan under the user-supplied reward function. Planning enables these approaches to perform new tasks at test-time. The key drawback is that these models learn dynamics of possible behavior rather than dynamics of desirable behavior. This means that the responsibility of evoking desirable behavior is entirely deferred to engineering the input reward function. Designing reward functions that cause MBRL to evoke complex, desirable behavior is difficult when the space of possible undesirable behaviors is large. In order to succeed, the rewards cannot lead the model astray towards observations significantly different than those with which the model was trained.

与 IL 相比，基于规划的算法(如基于模型的强化学习 (MBRL) 方法)不需要专家演示。 MBRL 可以适应通过奖励函数指定的新任务 (Kuvayev & Sutton, 1996; Deisenroth & Rasmussen, 2011)。 “模型”是一个动态模型，用于在用户提供的奖励函数下进行规划。 规划使这些方法能够在测试时执行新任务。 主要缺点是这些模型学习可能行为的动态而不是理想行为的动态。 这意味着唤起理想行为的责任完全推迟到设计输入奖励函数。 当可能的不良行为空间很大时，设计导致 MBRL 唤起复杂、理想行为的奖励函数是困难的。 为了成功，奖励不能使模型误入歧途，观察结果与训练模型时所用的观察结果有显著差异。

Our goal is to devise an algorithm that combines the advantages of MBRL and IL by offering MBRL’s flexibility to achieve new tasks at test-time and IL’s potential to learn desirable behavior entirely from offline data. To accomplish this, we first train a model to forecast expert trajectories with a density function, which can score trajectories and plans by how likely they are to come from the expert. A probabilistic model is necessary because expert behavior is stochastic: e.g. at an intersection, the expert could choose to turn left or right. Next, we derive a principled probabilistic inference objective to create plans that incorporate both (1) the model and (2) arbitrary new tasks. Finally, we derive families of tasks that we can provide to the inference framework. Our method can accomplish new tasks specified as complex goals without having seen an expert complete these tasks before. 

我们的目标是设计一种算法，通过提供 MBRL 在测试时完成新任务的灵活性和 IL 完全从离线数据中学习理想行为的潜力，结合 MBRL 和 IL 的优势。 为了实现这一点，我们首先训练一个模型来预测具有密度函数的专家轨迹，它可以根据来自专家的可能性对轨迹和计划进行评分。 概率模型是必要的，因为专家行为是随机的：例如 在十字路口，专家可以选择左转或右转。 接下来，我们推导出一个有原则的概率推理目标，以创建包含 (1) 模型和 (2) 任意新任务的计划。 最后，我们推导出可以提供给推理框架的任务系列。 我们的方法可以完成指定为复杂目标的新任务，而无需看到专家之前完成这些任务。

Figure 1: Our method: deep imitative models. Top Center. We use demonstrations to learn a probability density function q of future behavior and deploy it to accomplish various tasks. Left: A region in the ground plane is input to a planning procedure that reasons about how the expert would achieve that task. It coarsely specifies a destination, and guides the vehicle to turn left. Right: Goal positions and potholes yield a plan that avoids potholes and achieves one of the goals on the right.
图 1：我们的方法：深度模仿模型。 顶部中心。 我们使用演示来学习未来行为的概率密度函数 q 并将其部署以完成各种任务。 左图：地平面中的一个区域被输入到规划程序中，该程序推断专家将如何完成该任务。 它粗略地指定一个目的地，并引导车辆左转。 右图：目标位置和坑洼产生了一个避免坑洼并实现右侧目标之一的计划。

We investigate properties of our method on a dynamic simulated autonomous driving task (see Fig. 1). Videos are available at https://sites.google.com/view/imitative-models. Our contributions are as follows:
1. Interpretable expert-like plans without reward engineering. Our method outputs multi-step expert-like plans, offering superior interpretability to one-step imitation learning models. In contrast to MBRL, our method generates expert-like behaviors without reward function crafting.
2. Flexibility to new tasks: In contrast to IL, our method flexibly incorporates and achieves goals not seen during training, and performs complex tasks that were never demonstrated, such as navigating to goal regions and avoiding test-time only potholes, as depicted in Fig. 1.
3. Robustness to goal specification noise: We show that our method is robust to noise in the goal specification. In our application, we show that our agent can receive goals on the wrong side of the road, yet still navigate towards them while staying on the correct side of the road.
4. State-of-the-art CARLA performance: Our method substantially outperforms MBRL, a custom IL method, and all five prior CARLA IL methods known to us. It learned near-perfect driving through dynamic and static CARLA environments from expert observations alone. 

我们研究了我们的方法在动态模拟自动驾驶任务中的特性(见图 1)。 视频可在 https://sites.google.com/view/imitative-models 获取。 我们的贡献如下：
1. 没有奖励工程的可解释的类似专家的计划。 我们的方法输出类似专家的多步计划，为一步模仿学习模型提供卓越的可解释性。 与 MBRL 相比，我们的方法在没有奖励函数的情况下生成类似专家的行为。
2. 新任务的灵活性：与 IL 相比，我们的方法灵活地结合并实现了训练期间未见的目标，并执行了从未展示过的复杂任务，例如导航到目标区域和避免仅测试时间的坑洼，如中所述 图。1。
3. 对目标规范噪声的稳健性：我们表明我们的方法对目标规范中的噪声具有稳健性。 在我们的应用程序中，我们展示了我们的智能体可以在道路的错误一侧接收目标，但仍然在保持在道路正确的一侧的同时导航到它们。
4. 最先进的 CARLA 性能：我们的方法大大优于 MBRL、自定义 IL 方法和我们已知的所有五种先前的 CARLA IL 方法。 它仅通过专家观察就学会了在动态和静态 CARLA 环境中近乎完美的驾驶。

## 2 DEEP IMITATIVE MODELS
We begin by formalizing assumptions and notation. We model continuous-state, discrete-time, partially-observed Markov processes. Our agent’s state at time t is st ∈ RD; t = 0 refers to the current time step, and φ is the agent’s observations. Variables are bolded. Random variables are capitalized. Absent subscripts denote all future time steps, e.g. S .= S1:T ∈ RT ×D. We denote a probability density function of a random variable S as p(S), and its value as p(s) .= p(S=s).

To learn agent dynamics that are possible and preferred, we construct a model of expert behavior.

We fit an “Imitative Model” q(S1:T |φ) = Q Tt=1 q(St|S1:t−1, φ) to a dataset of expert trajectories

D = {(si , φi)}Ni=1 drawn from a (unknown) distribution of expert behavior si ∼ p(S|φi). By training q(S|φ) to forecast expert trajectories with high likelihood, we model the scene-conditioned expert dynamics, which can score trajectories by how likely they are to come from the expert. 2

### 2.1 IMITATIVE PLANNING TO GOALS
After training, q(S|φ) can generate trajectories that resemble those that the expert might generate – e.g. trajectories that navigate roads with expert-like maneuvers. However, these maneuvers will not have a specific goal. Beyond generating human-like behaviors, we wish to direct our agent to goals and have the agent automatically reason about the necessary mid-level details. We define general tasks by a set of goal variables G. The probability of a plan s conditioned on the goal G is modelled by a posterior p(s|G, φ). This posterior is implemented with q(s|φ) as a learned imitation prior and p(G|s, φ) as a test-time goal likelihood. We give examples of p(G|s, φ) after deriving a maximum a posteriori inference procedure to generate expert-like plans that achieve abstract goals: s∗ .= arg max s log p(s|G, φ) = arg max s log q(s|φ) + log p(G|s, φ) − log p(G|φ) = arg max s log q(s|φ) | {z} imitation prior + log p(G|s, φ) | {z} goal likelihood . (1)

We perform gradient-based optimization of Eq. 1, and defer this discussion to Appendix A. Next, we discuss several goal likelihoods, which direct the planning in different ways. They communicate goals they desire the agent to achieve, but not how to achieve them. The planning procedure determines how to achieve them by producing paths similar to those an expert would have taken to reach the given goal. In contrast to black-box one-step IL that predicts controls, our method produces interpretable multi-step plans accompanied by two scores. One estimates the plan’s “expertness”, the second estimates its probability to achieve the goal. Their sum communicates the plan’s overall quality.

### 2.2 CONSTRUCTING GOAL LIKELIHOODS
Constraint-based planning to goal sets (hyperparameter-free): Consider the setting where we have access to a set of desired final states, one of which the agent should achieve. We can model this by applying a Dirac-delta distribution on the final state, to ensure it lands in a goal set G ⊂RD: p(G|s, φ) ← δsT (G), δsT (G) = 1 if sT ∈ G, δsT (G) = 0 if sT 6∈ G. (2) δsT (G)’s partial support of sT ∈ G ⊂ RD constrains sT and introduces no hyperparameters into p(G|s, φ). For each choice of G, we have a different way to provide high-level task information to the agent. The simplest choice for G is a finite set of points: a (A) Final-State Indicator likelihood.

We applied (A) to a sequence of waypoints received from a standard A∗ planner (provided by the

CARLA simulator), and outperformed all prior dynamic-world CARLA methods known to us. We can also consider providing an infinite number of points. Providing a set of line-segments as G yields a (B) Line-Segment Final-State Indicator likelihood, which encourages the final state to land along one of the segments. Finally, consider a (C) Region Final-State Indicator likelihood in which G is a polygon (see Figs. 1 and 4). Solving Eq. 1 with (C) amounts to planning the most expert-like trajectory that ends inside a goal region. Appendix B provides derivations, implementation details, and additional visualizations. We found these methods to work well when G contains “expert-like” final position(s), as the prior strongly penalizes plans ending in non-expert-like positions.

Unconstrained planning to goal sets (hyperparameter-based): Instead of constraining that the final state of the trajectory reach a goal, we can use a goal likelihood with full support (sT ∈RD), centered at a desired final state. This lets the goal likelihood encourage goals, rather than dictate them.

If there is a single desired goal (G={gT }), the (D) Gaussian Final-State likelihood p(G|s, φ) ← N (gT ; sT , I) treats gT as a noisy observation of a final future state, and encourages the plan to arrive at a final state. We can also plan to K successive states G = (gT −K+1, . . . , gT ) with a (E)

Gaussian State Sequence: p(G|s, φ) ← Q Tk=T −K+1 N (gk; sk, I) if a program wishes to specify a desired end velocity or acceleration when reaching the final state gT (Fig. 2). Alternatively, a planner may propose a set of states with the intention that the agent should reach any one of them. This is possible by using a (F) Gaussian Final-State Mixture: p(G|s, φ) ← 1K P Kk=1 N (gkT ; sT , I) and is useful if some of those final states are not reachable with an expert-like plan. Unlike A–C, D–F introduce a hyperparameter “ ”. However, they are useful when no states in G correspond to observed expert behavior, as they allow the imitation prior to be robust to poorly specified goals.

Costed planning: Our model has the additional flexibility to accept arbitrary user-specified costs c at test-time. For example, we may have updated knowledge of new hazards at test-time, such as a given map of potholes or a predicted cost map. Cost-based knowledge c(si|φ) can be incorporated as 3 an (G) Energy-based likelihood: p(G|s, φ) ∝ Q Tt=1 e−c(st|φ) (Todorov, 2007; Levine, 2018). This can be combined with other goal-seeking objectives by simply multiplying the likelihoods together.

Examples of combining G (energy-based) with F (Gaussian mixture) were shown in Fig. 1 and are shown in Fig. 3. Next, we describe instantiating q(S|φ) in CARLA (Dosovitskiy et al., 2017).

Figure 2: Imitative planning with the

Gaussian State Sequence enables finegrained control of the plans.

Figure 3: Costs can be assigned to “potholes” only seen at test-time. The planner prefers routes avoiding potholes.

Figure 4: Goal regions can be coarsely specified to give directions.

### 2.3 APPLYING DEEP IMITATIVE MODELS TO AUTONOMOUS DRIVING
Figure 5: Architecture of mθ and σθ, which parameterize qθ(S|φ={χ, s−τ:0,λ}). Inputs: LIDAR χ, past-states s−τ:0, light-state λ, and latent noise Z1:T . Output: trajectory S1:T . Details in Appendix C.

In our autonomous driving application, we model the agent’s state at time t as st ∈ RD with D = 2; st represents our agent’s location on the ground plane. The agent has access to environment perception φ ← {s−τ:0, χ,λ}, where τ is the number of past positions we condition on, χ is a high-dimensional observation of the scene, and λ is a low-dimensional traffic light signal. χ could represent either

LIDAR or camera images (or both), and is the agent’s observation of the world. In our setting, we featurize LIDAR to χ = R 200×200×2 , with χij representing a 2-bin histogram of points above and at ground level in a 0.5m2 cell at position (i, j). CARLA provides ground-truth s−τ:0 and λ. Their availability is a realistic input assumption in perception-based autonomous driving pipelines.

Model requirements: A deep imitative model forecasts future expert behavior. It must be able to compute q(s|φ)∀s ∈ RT ×D. The ability to compute ∇sq(s|φ) enables gradient-based optimization for planning. Rudenko et al. (2019) provide a recent survey on forecasting agent behavior. As many forecasting methods cannot compute trajectory probabilities, we must be judicious in choosing q(S|φ).

A model that can compute probabilities R2P2 (Rhinehart et al., 2018), a generative autoregressive flow (Rezende & Mohamed, 2015; Oord et al., 2017). We extend R2P2 to instantiate the deep imitative model q(S|φ). R2P2 was previously used to forecast vehicle trajectories: it was not demonstrated or developed to plan or execute controls. Although we used R2P2, other futuretrajectory density estimation techniques could be used – designing q(s|φ) is not the primary focus of this work. In R2P2, qθ(S|φ) is induced by an invertible, differentiable function: S = fθ(Z; φ) :

RT ×2 7→RT ×2; fθ warps a latent sample from a base distribution Z∼q0 =N (0, I) to S. θ is trained to maximize qθ(S|φ) of expert trajectories. fθ is defined for 1..T as follows:

St = ft(Z1:t) = µθ(S1:t−1, φ) + σθ(S1:t−1, φ)Zt, (3) where µθ(S1:t−1, φ)= 2St−1−St−2+mθ(S1:t−1, φ) encodes a constant-velocity inductive bias. The mθ ∈ R2 and σθ ∈ R2×2 are computed by expressive neural networks. The resulting trajectory 4

Figure 6: Illustration of our method applied to autonomous driving. Our method trains an imitative model from a dataset of expert examples. After training, the model is repurposed as an imitative planner. At test-time, a route planner provides waypoints to the imitative planner, which computes expert-like paths to each goal. The best plan is chosen according to the planning objective and provided to a low-level PID-controller in order to produce steering and throttle actions. This procedure is also described with pseudocode in Appendix A. distribution is complex and multimodal (Appendix C.1 depicts samples). Because traffic light state was not included in the φ of R2P2’s “RNN” model, it could not react to traffic lights. We created a new model that includes λ. It fixed cases where q(S|φ) exhibited no forward-moving preference when the agent was already stopped, and improved q(S|φ)’s stopping preference at red lights. We used T = 40 trajectories at 10Hz (4 seconds), and τ = 3. Fig. 5 depicts the architecture of µθ and σθ.

### 2.4 IMITATIVE DRIVING
We now instantiate a complete autonomous driving framework based on imitative models to study in our experiments, seen in Fig. 6. We use three layers of spatial abstraction to plan to a faraway destination, common to autonomous vehicle setups: coarse route planning over a road map, path planning within the observable space, and feedback control to follow the planned path (Paden et al., 2016; Schwarting et al., 2018). For instance, a route planner based on a conventional GPS-based navigation system might output waypoints roughly in the lanes of the desired direction of travel, but not accounting for environmental factors such as the positions of other vehicles. This roughly communicates possibilities of where the vehicle could go, but not when or how it could get to them, or any environmental factors like other vehicles. A goal likelihood from Sec. 2.2 is formed from the route and passed to the planner, which generates a state-space plan according to the optimization in Eq. 1. The resulting plan is fed to a simple PID controller on steering, throttle, and braking. In Pseudocode of the driving, inference, and PID algorithms are given in Appendix A. 

## 3 RELATED WORK
A body of previous work has explored offline IL (Behavior Cloning – BC) in the CARLA simulator (Li et al., 2018; Liang et al., 2018; Sauer et al., 2018; Codevilla et al., 2018; 2019). These BC approaches condition on goals drawn from a small discrete set of directives. Despite BC’s theoretical drift shortcomings (Ross et al., 2011), these methods still perform empirically well. These approaches and ours share the same high-level routing algorithm: an A∗ planner on route nodes that generates waypoints. In contrast to our approach, these approaches use the waypoints in a Waypoint Classifier, which reasons about the map and the geometry of the route to classify the waypoints into one of several directives: {Turn left, Turn right, Follow Lane, Go Straight}. One of the original motivations for these type of controls was to enable a human to direct the robot (Codevilla et al., 2018). However, in scenarios where there is no human in the loop (i.e. autonomous driving), we advocate for approaches to make use of the detailed spatial information inherent in these waypoints. Our approach and several others we designed make use of this spatial information. One of these is CIL-States (CILS): whereas the approach in Codevilla et al. (2018) uses images to directly generate controls, CILS uses identical inputs and PID controllers as our method. With respect to prior conditional IL methods, our main approach has more flexibility to handle more complex directives post-training, the ability to learn without goal labels, and the ability to generate interpretable planned and unplanned trajectories. These contrasting capabilities are illustrated in Table 1.

Our approach is also related to MBRL. MBRL can also plan, but with a one-step predictive model of possible dynamics. The task of evoking expert-like behavior is offloaded to the reward function, which can be difficult and time-consuming to craft properly. We know of no MBRL approach 

Table 1: Desirable attributes of each approach. A green check denotes that a method has a desirable attribute, whereas a red cross denotes the opposite. A “† ” indicates an approach we implemented.

Approach Flexible to New Goals Trains without goal labels Outputs Plans Trains Offline Has Expert P.D.F.

CIRL∗ (Liang et al., 2018) ✗ ✗ ✗ ✗ ✗

CAL∗ (Sauer et al., 2018) ✗ ✗ ✗ ✓ ✗

MT∗ (Li et al., 2018) ✗ ✗ ✗ ✓ ✗

CIL∗ (Codevilla et al., 2018) ✗ ✗ ✗ ✓ ✗

CILRS∗ (Codevilla et al., 2019) ✗ ✗ ✗ ✓ ✗

CILS† ✗ ✓ ✗ ✓ ✗

MBRL† ✓ ✓ ✓ ✗ ✗

Imitative Models (Ours)† ✓ ✓ ✓ ✓ ✓

Table 2: Algorithmic components of each approach. A “† ” indicates an approach we implemented.

Approach Control Algorithm ← Learning Algorithm ← Goal-Generation Algorithm ← Routing Algorithm High-Dim. Obs.

CIRL∗ (Liang et al., 2018) Policy Behavior Cloning+RL Waypoint Classifier A∗ Waypointer Image

CAL∗ (Sauer et al., 2018) PID Affordance Learning Waypoint Classifier A∗ Waypointer Image MT∗ (Li et al., 2018) Policy Behavior Cloning Waypoint Classifier A∗ Waypointer Image

CIL∗ (Codevilla et al., 2018) Policy Behavior Cloning Waypoint Classifier A∗ Waypointer Image

CILRS∗ (Codevilla et al., 2019) Policy Behavior Cloning Waypoint Classifier A∗ Waypointer Image

CILS† PID Trajectory Regressor Waypoint Classifier A∗ Waypointer LIDAR

MBRL† Reachability Tree State Regressor Waypoint Selector A∗ Waypointer LIDAR

Imitative Models (Ours)†

Imitative Plan+PID Traj. Density Est. Goal Likelihoods A∗ Waypointer LIDAR previously applied to CARLA, so we devised one for comparison. This MBRL approach also uses identical inputs to our method, instead to plan a reachability tree (LaValle, 2006) over an dynamic obstacle-based reward function. See Appendix D for further details of the MBRL and CILS methods, which we emphasize use the same inputs as our method.

Several prior works (Tamar et al., 2016; Amos et al., 2018; Srinivas et al., 2018) used imitation learning to train policies that contain planning-like modules as part of the model architecture. While our work also combines planning and imitation learning, ours captures a distribution over possible trajectories, and then plan trajectories at test-time that accomplish a variety of given goals with high probability under this distribution. Our approach is suited to offline-learning settings where interactively collecting data is costly (time-consuming or dangerous). However, there exists online IL approaches that seek to be safe (Menda et al., 2017; Sun et al., 2018; Zhang & Cho, 2017). 

## 4 EXPERIMENTS
We evaluate our method using the CARLA driving simulator (Dosovitskiy et al., 2017). We seek to answer four primary questions: (1) Can we generate interpretable, expert-like plans with offline learning and no reward engineering? Neither IL nor MBRL can do so. It is straightforward to interpret the trajectories by visualizing them on the ground plane; we thus seek to validate whether these plans are expert-like by equating expert-like behavior with high performance on the CARLA benchmark. (2) Can we achieve state-of-the-art CARLA performance using resources commonly available in real autonomous vehicle settings? There are several differences between the approaches, as discussed in Sec 3 and shown in Tables 1 and 2. Our approach uses the CARLA toolkit’s resources that are commonly available in real autonomous vehicle settings: waypoint-based routes (all prior approaches use these) and LIDAR (CARLA-provided, but only the approaches we implemented use it). Furthermore, the two additional methods of comparison we implemented (CILS and MBRL) use the exact same inputs as our algorithm. These reasons justify an overall performance comparison to answer (2): whether we can achieve state-of-the-art performance using commonly available resources. We advocate that other approaches also make use of such resources. (3) How flexible is our approach to new tasks? We investigate (3) by applying each of the goal likelihoods we derived and observing the resulting performance. (4) How robust is our approach to error in the provided goals? We do so by injecting two different types of error into the waypoints and observing the resulting performance.

We begin by training q(S|φ) on a dataset of 25 hours of driving we collected in Town01, detailed in Appendix C.2. Following existing protocol, each test episode begins with the vehicle randomly positioned on a road in the Town01 or Town02 maps in one of two settings: static-world (no other vehicles) or dynamic-world (with other vehicles). We construct the goal set G for the Final-State

Indicator (A) directly from the route output by CARLA’s waypointer. B’s line segments are formed by connecting the waypoints to form a piecewise linear set of segments. C’s regions are created a polygonal goal region around the segments of (B). Each represents an increasing level of coarseness of direction. Coarser directions are easier to specify when there is ambiguity in positions (both the position of the vehicle and the position of the goals). Further details are discussed in Appendix B.3. 6

We use three metrics: (a) success rate in driving to the destination without any collisions (which all prior work reports); (b) red-light violations; and (c) proportion of time spent driving in the wrong lane and off road. With the exception of metric (a), lower numbers are better.

Results: Towards questions (1) and (3) (expert-like plans and flexibility), we apply our approach with a variety of goal likelihoods to the CARLA simulator. Towards question (2), we compare our methods against CILS, MBRL, and prior work. These results are shown in Table 3. The metrics for the methods we did not implement are from the aggregation reported in Codevilla et al. (2019).

We observe our method to outperform all other approaches in all settings: static world, dynamic world, training conditions, and test conditions. We observe the Goal Indicator methods are able to perform well, despite having no hyperparameters to tune. We found that we could further improve our approach’s performance if we use the light state to define different goal sets, which defines a “smart” waypointer. The settings where we use this are suffixed with “S.” in the Tables. We observed the planner prefers closer goals when obstructed, when the vehicle was already stopped, and when a red light was detected; we observed the planner prefers farther goals when unobstructed and when green lights or no lights were observed. Examples of these and other interesting behaviors are best seen in the videos on the website (https://sites.google.com/view/imitative-models). These behaviors follow from the method leveraging q(S|φ)’s internalization of aspects of expert behavior in order to reproduce them in new situations. Altogether, these results provide affirmative answers to questions (1) and (2). Towards question (3), these results show that our approach is flexible to different directions defined by these goal likelihoods.

Table 3: We evaluate different autonomous driving methods on CARLA’s Dynamic Navigation task.

A “† ” indicates methods we have implemented (each observes the same waypoints and LIDAR as input). A “∗ ” indicates results reported in Codevilla et al. (2019). A “–” indicates an unreported statistic. A “‡ ” indicates an optimistic estimate in transferring a result from the static setting to the dynamic setting. “S.” denotes a “smart” waypointer reactive to light state, detailed in Appendix B.2.

Town01 (training conditions) Town02 (test conditions)

Dynamic Nav. Method Success Ran Red Light Wrong lane Off road Success Ran Red Light Wrong lane Off road

CIRL∗(Liang et al., 2018) 82% – – – 41% – – –

CAL∗(Sauer et al., 2018) 83% – – – 64% – – – MT∗ (Li et al., 2018) 81% – – – 53% – – –

CIL∗(Codevilla et al., 2018) 83% 83%‡ – – 38% 82%‡ – –

CILRS∗(Codevilla et al., 2019) 92% 27%‡ – – 66% 64%‡ – –

CILS, Waypoint Input† 17% 0.0% 0.20% 12.1% 36% 0.0% 1.11% 11.70%

MBRL, Waypoint Input† 64% 72% 11.1% 2.96% 48% 54% 20.6% 13.3 %

Our method, Final-State Indicator† 92% 26% 0.05% 0.012% 84% 35% 0.13% 0.38%

Our method, Line Segment Final-St. Indicator† 84% 42% 0.03% 0.295% 88% 33% 0.12% 0.14%

Our method, Region Final-St. Indicator† 84% 56% 0.03% 0.131% 88% 54% 0.13% 0.22%

Our method, Gaussian Final-St. Mix.† 92% 6.3% 0.04% 0.005% 100% 12% 0.11% 0.04%

Our method, Region Final-St. Indicator S.† 92% 2.8% 0.021% 0.099% 92% 4.0% 0.11% 1.85%

Our method, Gaussian Final-St. Mix. S.† 100% 1.7% 0.03% 0.005% 92% 0.0% 0.05% 0.15%

Town01 (training conditions) Town02 (test conditions)

Static Nav. Method Success Ran Red Light Wrong lane Off road Success Ran Red Light Wrong lane Off road

CIRL∗ (Liang et al., 2018) 93% – – – 68% – – –

CAL∗ (Sauer et al., 2018) 92% – – – 68% – – –

MT∗ (Li et al., 2018) 81% – – – 78% – – –

CIL∗ (Codevilla et al., 2018) 86% 83% – – 44% 82% – –

CILRS∗(Codevilla et al., 2019) 95% 27% – – 90% 64% – –

CILS, Waypoint Input† 28% 0.0% 0.38% 10.23% 36% 0.0% 1.69% 16.82%

MBRL, Waypoint Input† 96% 78% 14.3% 1.94% 96% 73% 19.6 % 0.75%

Our method, Final-State Indicator† 100% 48% 0.05% 0.002% 100% 52% 0.10% 0.13%

Our method, Gaussian Final-St. Mixture† 96% 0.83% 0.01% 0.08% 96% 0.0% 0.03% 0.14%

Our method, Gaussian Final-St. Mix. S.† 96% 0.0% 0.04% 0.07% 92% 0.0% 0.18% 0.27%

### 4.1 ROBUSTNESS TO ERRORS IN GOAL-SPECIFICATION
Towards questions (3) (flexibility) and (4) (noise-robustness), we analyze the performance of our method when the path planner is heavily degraded, to understand its stability and reliability. We use the Gaussian Final-State Mixture goal likelihood.

Navigating with high-variance waypoints. As a test of our model’s capability to stay in the distribution of demonstrated behavior, we designed a “decoy waypoints” experiment, in which half of the waypoints are highly perturbed versions of the other half, serving as distractions for our Gaussian Final-State Mixture imitative planner. We observed surprising robustness to decoy waypoints. Examples of this robustness are shown in Fig. 7. In Table 4, we report the success rate and the mean number of planning rounds for failed episodes in the “1/2 distractors” row. These 7

Figure 7: Tolerating bad goals. The planner prefers goals in the distribution of expert behavior (on the road at a reasonable distance). Left: Planning with 1/2 decoy goals. Right: Planning with all goals on the wrong side of the road.

Figure 8: Testtime plans steering around potholes. numbers indicate our method can execute dozens of planning rounds without decoy waypoints causing a catastrophic failure, and often it can execute the hundreds necessary to achieve the goal.

See Appendix E for details.

Navigating with waypoints on the wrong side of the road. We also designed an experiment to test our method under systemic bias in the route planner. Our method is provided waypoints on the wrong side of the road (in CARLA, the left side), and tasked with following the directions of these waypoints while staying on the correct side of the road (the right side). In order for the value of q(s|φ) to outweigh the influence of these waypoints, we increased the  hyperparameter. We found our method to still be very effective at navigating, and report results in Table 4. We also investigated providing very coarse 8-meter wide regions to the Region Final-State likelihood; these always include space in the wrong lane and off-road (Fig. 12 in Appendix B.4 provides visualization). Nonetheless, on

Town01 Dynamic, this approach still achieved an overall success rate of 48%. Taken together towards question (4), our results indicate that our method is fairly robust to errors in goal-specification.

### 4.2 PRODUCING UNOBSERVED BEHAVIORS TO AVOID NOVEL OBSTACLES
Table 4: Robustness to waypoint noise and test-time pothole adaptation. Our method is robust to waypoints on the wrong side of the road and fairly robust to decoy waypoints. Our method is flexible enough to safely produce behavior not demonstrated (pothole avoidance) by incorporating a test-time cost. Ten episodes are collected in each Town.

Town01 (training conditions) Town02 (test conditions)

Waypointer Extra Cost Success Wrong lane Potholes hit Success Wrong lane Potholes hit

Noiseless waypointer 100% 0.00% 177/230 100% 0.41% 82/154

Waypoints wrong lane 100% 0.34% – 70% 3.16% – 1/2 waypoints distracting 70% – 50% – –

Noiseless waypointer Pothole 90% 1.53% 10/230 70% 1.53% 35/154

To further investigate our model’s flexibility to test-time objectives (question 3), we designed a pothole avoidance experiment. We simulated potholes in the environment by randomly inserting them in the cost map near waypoints. We ran our method with a test-time-only cost map of the simulated potholes by combining goal likelihoods (F) and (G), and compared to our method that did not incorporate the cost map (using (F) only, and thus had no incentive to avoid potholes). We recorded the number of collisions with potholes. In Table 4, our method with cost incorporated avoided most potholes while avoiding collisions with the environment. To do so, it drove closer to the centerline, and occasionally entered the opposite lane. Our model internalized obstacle avoidance by staying on the road and demonstrated its flexibility to obstacles not observed during training. Fig. 8 shows an example of this behavior. See Appendix F for details of the pothole generation. 

## 5 DISCUSSION
We proposed “Imitative Models” to combine the benefits of IL and MBRL. Imitative Models are probabilistic predictive models able to plan interpretable expert-like trajectories to achieve new goals.

Inference with an Imitative Model resembles trajectory optimization in MBRL, enabling it to both incorporate new goals and plan to them at test-time, which IL cannot. Learning an Imitative Model resembles offline IL, enabling it to circumvent the difficult reward-engineering and costly online data collection necessities of MBRL. We derived families of flexible goal objectives and showed 8 our model can successfully incorporate them without additional training. Our method substantially outperformed six IL approaches and an MBRL approach in a dynamic simulated autonomous driving task. We showed our approach is robust to poorly specified goals, such as goals on the wrong side of the road. We believe our method is broadly applicable in settings where expert demonstrations are available, flexibility to new situations is demanded, and safety is paramount.

## References
* Brandon Amos, Ivan Dario Jimenez Rodriguez, Jacob Sacks, Byron Boots, and Zico Kolter. Differentiable MPC for end-to-end planning and control. In Neural Information Processing Systems(NeurIPS), 2018.
* Felipe Codevilla, Matthias Miiller, Antonio López, Vladlen Koltun, and Alexey Dosovitskiy. Endto-end driving via conditional imitation learning. In International Conference on Robotics andAutomation (ICRA), pp. 1–9. IEEE, 2018.
* Felipe Codevilla, Eder Santana, Antonio M López, and Adrien Gaidon. Exploring the limitations ofbehavior cloning for autonomous driving. arXiv preprint arXiv:1904.08980, 2019.
* Marc Deisenroth and Carl E Rasmussen. PILCO: A model-based and data-efficient approach topolicy search. In International Conference on Machine Learning (ICML), pp. 465–472, 2011.
* Alexey Dosovitskiy and Vladlen Koltun. Learning to act by predicting the future. arXiv preprintarXiv:1611.01779, 2016.
* Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA:An open urban driving simulator. In Conference on Robot Learning (CoRL), pp. 1–16, 2017.
* Leonid Kuvayev and Richard S. Sutton. Model-based reinforcement learning with an approximate,learned model. In Yale Workshop on Adaptive and Learning Systems, pp. 101–105, 1996.
* Steven M LaValle. Planning algorithms. Cambridge University Press, 2006.
* Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review.
* arXiv preprint arXiv:1805.00909, 2018.
* Zhihao Li, Toshiyuki Motoyoshi, Kazuma Sasaki, Tetsuya Ogata, and Shigeki Sugano. Rethinkingself-driving: Multi-task knowledge for better generalization and accident explanation ability. arXivpreprint arXiv:1809.11100, 2018.
* Xiaodan Liang, Tairui Wang, Luona Yang, and Eric Xing. CIRL: Controllable imitative reinforcementlearning for vision-based self-driving. arXiv preprint arXiv:1807.03776, 2018.
* Kunal Menda, Katherine Driggs-Campbell, and Mykel J Kochenderfer. DropoutDAgger: A Bayesianapproach to safe imitation learning. arXiv preprint arXiv:1709.06166, 2017.
* Aaron van den Oord, Yazhe Li, Igor Babuschkin, Karen Simonyan, Oriol Vinyals, Koray Kavukcuoglu,George van den Driessche, Edward Lockhart, Luis C Cobo, Florian Stimberg, et al. ParallelWaveNet: Fast high-fidelity speech synthesis. arXiv preprint arXiv:1711.10433, 2017.
* Brian Paden, Michal ˇCáp, Sze Zheng Yong, Dmitry Yershov, and Emilio Frazzoli. A survey ofmotion planning and control techniques for self-driving urban vehicles. Transactions on IntelligentVehicles, 1(1):33–55, 2016.
* Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In InternationalConference on Machine Learning (ICML), pp. 1530–1538, 2015.
* Nicholas Rhinehart, Kris M. Kitani, and Paul Vernaza. R2P2: A reparameterized pushforward policyfor diverse, precise generative path forecasting. In European Conference on Computer Vision(ECCV), September 2018.
* Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structuredprediction to no-regret online learning. In International Conference on Artificial Intelligence andStatistics (AISTATS), pp. 627–635, 2011.
* 9Andrey Rudenko, Luigi Palmieri, Michael Herman, Kris M Kitani, Dariu M Gavrila, and Kai O Arras.
* Human motion trajectory prediction: A survey. arXiv preprint arXiv:1905.06113, 2019.
* Axel Sauer, Nikolay Savinov, and Andreas Geiger. Conditional affordance learning for driving inurban environments. arXiv preprint arXiv:1806.06498, 2018.
* Wilko Schwarting, Javier Alonso-Mora, and Daniela Rus. Planning and decision-making for autonomous vehicles. Annual Review of Control, Robotics, and Autonomous Systems, 1:187–210,2018.
* Aravind Srinivas, Allan Jabri, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Universal planning networks: Learning generalizable representations for visuomotor control. In InternationalConference on Machine Learning (ICML), pp. 4739–4748, 2018.
* Liting Sun, Cheng Peng, Wei Zhan, and Masayoshi Tomizuka. A fast integrated planning andcontrol framework for autonomous driving via imitation learning. In Dynamic Systems and ControlConference. American Society of Mechanical Engineers, 2018.
* Aviv Tamar, Yi Wu, Garrett Thomas, Sergey Levine, and Pieter Abbeel. Value iteration networks. InNeural Information Processing Systems (NeurIPS), pp. 2154–2162, 2016.
* Emanuel Todorov. Linearly-solvable Markov decision problems. In Neural Information ProcessingSystems (NeurIPS), pp. 1369–1376, 2007.
* Jiakai Zhang and Kyunghyun Cho. Query-efficient imitation learning for end-to-end simulateddriving. In Association for the Advancement of Artificial Intelligence (AAAI), pp. 2891–2897,2017.

## A ALGORITHMS

Algorithm 1 IMITATIVEDRIVING(ROUTEPLAN, IMITATIVEPLAN, PIDCONTROLLER, qθ, H) 1: φ ← ENVIRONMENT(∅) {Initialize the robot} 2: while not at destination do 3: G ← ROUTEPLAN(φ) {Generate goals from a route} 4: sG 1:T ← IMITATIVEPLANR2P2(qθ, G, φ) {Plan path} 5: for h = 0 to H do 6: u ← PIDCONTROLLER(φ, sG 1:T , h, H) 7: φ ← ENVIRONMENT(u) {Execute control} 8: end for 9: end while

In Algorithm 1, we provide pseudocode for receding-horizon control via our imitative model. In

Algorithm 2 we provide pesudocode that describes how we plan in the latent space of the trajectory.

Since s1:T = f(z1:T ) in our implementation, and f is differentiable, we can perform gradient descent of the same objective in terms of z1:T . Since q is trained with z1:T ∼ N (0, I), the latent space is likelier to be better numerically conditioned than the space of s1:T , although we did not compare the two approaches formally.

Algorithm 2 IMITATIVEPLANR2P2(qθ, G, φ, f ) 1: Define MAP objective L with qθ according to Eq. 1 {Incorporate the Imitative Model} 2: Initialize z1:T ∼ q0 3: while not converged do 4: z1:T ← z1:T + ∇z1:T L(s1:T = f(z1:T ), G, φ) 5: end while 6: return s1:T = f(z1:T )

In Algorithm 3, we detail the speed-based throttle and position-based steering PID controllers.

Algorithm 3 PIDCONTROLLER(φ = {s0, s−1, . . . }, sG 1:T , h, H; K ˙ps , Kpα) 1: i ← T − H + h {Compute the index of the target position} 2: ˙sprocess-speed ← (s0,x − s−1,x) {Compute the current forward speed from the observations} 3: ssetpoint-position ← sG i,x {Retrieve the target position x-coordinate from the plan} 4: ˙ssetpoint-speed ← ssetpoint-position/i {Compute the forward target speed} 5: e ˙s ← ˙ssetpoint-speed − ˙sprocess-speed {Compute the forward speed error} 6: u˙s ← K ˙pse ˙s {Compute the accelerator control with a nonzero proportional term} 7: throttle ← ✶(e > 0) · u + ✶(e ≤ 0) · 0 {Use the control as throttle if the speed error is positive} 8: brake ← ✶(e > 0) · 0 + ✶(e ≤ 0) · u {Use the control as brake if the speed error is negative} 9: αprocess ← arctan(s0,y − s−1,y, s0,x − s−1,x) {Compute current heading} 10: αsetpoint ← arctan(sG i,y − s0,y, |sG i,x − s0,x|) {Compute target forward heading} 11: eα ← αsetpoint − αprocess {Compute the heading error} 12: steering ← Kpαeα {Compute the steering with a nonzero proportional term} 13: u ← [throttle,steering, brake] 14: return u

## B GOAL DETAILS

### B.1 OPTIMIZING GOAL LIKELIHOODS WITH SET CONSTRAINTS
We now derive an approach to optimize our main objective with set constraints. Although we could apply a constrained optimizer, we find that we are able to exploit properties of the model  and constraints to derive differentiable objectives that enable approximate optimization of the corresponding closed-form optimization problems. These enable us to use the same straightforward gradient-descent-based optimization approach described in Algorithm 2.

Shorthand notation: In this section we omit dependencies on φ for brevity, and use short hand µt .= µθ(s1:t−1) and Σt .= Σθ(s1:t−1). For example, q(st|s1:t−1) = N (st; µt, Σt).

Let us begin by defining a useful delta function: δsT (G) .=  1 if sT ∈ G 0 if sT 6∈ G, (4) which serves as our goal likelihood when using goal with set constraints: p(G|s1:T ) ← δST (G). We now derive the corresponding maximum a posteriori optimization problem: s∗ 1:T .= arg max s1:T ∈R2T p(s1:T |G) = arg max s1:T ∈R2T p(G|s1:T ) · q(s1:T ) · p−1(G) = arg max s1:T ∈R2T p(G|s1:T ) | {z} goal likelihood · q(s1:T ) | {z} imitation prior = arg max s1:T ∈R2T δST (G) | {z} set constraint · q(s1:T ) | {z} imitation prior = arg max s1:T ∈R2T  q(s1:T ) if sT ∈ G 0 if sT 6∈ G = arg max s1:T−1∈R2(T−1),sT ∈G q(s1:T ) = arg max s1:T−1∈R2(T−1) arg max sT ∈G q(sT |s1:T −1) T −1 Y t=1 q(st|s1:t−1) = arg max s1:T−1∈R2(T−1) arg max sT ∈G

N (sT ; µT , ΣT ) T −1 Y t=1

N (st; µt, Σt). (5)

By exploiting the fact that q(sT |s1:T −1) = N (sT ; µT , ΣT ), we can derive closed-form solutions for s∗T = arg max sT ∈G N (sT ; µT , ΣT ) (6) when G has special structure, which enables us to apply gradient descent to solve this constrainedoptimization problem (examples below). With a closed form solution to equation 6, we can easily compute equation 5 using unconstrained-optimization as follows: s∗ 1:T = arg max s1:T−1∈R2(T−1) arg max sT ∈Gline-segment q(sT |s1:T −1) T −1 Y t=1 q(st|s1:t−1) (7) s∗ 1:T −1 = arg max s1:T−1∈R2(T−1) | {z } unconstrained optimization q(s∗T |s1:t−1) T −1 Y t=1 q(st|s1:t−1). | {z} objective function of s1:T−1 (8)

Note that equation 8 only helps solve equation 5 if equation 6 has a closed-form solution. We detail example of goal-sets with such closed-form solutions in the following subsections.

#### B.1.1 POINT GOAL-SET

The solution to equation 6 in the case of a single desired goal g ∈ RD is simply:

Gpoint .= {gT }, (9) s∗

T ,point .= arg max sT ∈Gpoint

N (sT ; µT , ΣT ) = gT . (10) 

More generally, multiple point goals help define optional end points for planning: where the agent only need reach one valid end point (see Fig. 9 for examples), formulated as:

Gpoints .= {gkT }Kk=1, (11) s∗

T ,points .= arg max gkT ∈Gpoints

N  gkT ; µT , ΣT  . (12)

#### B.1.2 LINE-SEGMENT GOAL-SET

We can form a goal-set as a finite-length line segment, connecting point a ∈ RD to point b ∈ RD: gline(u) .= a + u · (b − a), u ∈ R, (13)

Ga→b line-segment .= {gline(u) : u ∈ [0, 1]}. (14)

The solution to equation 6 in the case of line-segment goals is: s∗

T ,line-segment .= arg max sT ∈Ga→b line-segment

N (sT ; µT , ΣT ) (15) = a + min  1, max  0, (b − a)> Σ−1 T (µT − a) (b − a)> Σ−1 T (b − a)  · (b − a). (16)

Proof:

To solve equation 15 is to find which point along the line gline(u) maximizes N (·; µT , ΣT ) subject to the constraint 0 ≤ u ≤ 1: u∗ .= arg max u∈[0,1]

N (gline(u); µT , ΣT )) = arg min u∈[0,1] (gline(u) − µT )> Σ−T 1(gline(u) − µT ) | {z} Lu(u) . (17)

Since Lu is convex, the optimal value u∗ is value closest to the unconstrained arg max of Lu(u), subject to 0 ≤ u ≤ 1: u∗R .= arg max u∈R Lu(u), (18) u∗ = arg min u∈[0,1]

Lu(u) = min (1, max (0, u∗R)). (19)

We now solve for u∗R: u∗R = u : 0 = dL(u) du = d  (gline(u) − µT )> Σ−1 T (gline(u) − µT ) du = 2 · d(gline(u) − µT )> du Σ−1 T (gline(u) − µT ) = 2 · d(a + u · (b − a) − µT )> du Σ−1 T (a + u · (b − a) − µT ) = 2 · (b − a)> Σ−T 1(a + u · (b − a) − µT ), u∗R = (b − a)> Σ−1 T (µT − a) (b − a)> Σ−1 T (b − a) , (20) which gives us: s∗

T ,line-segment = gline(u∗) = a + u∗ · (b − a) = a + min (1, max (0, u∗R)) · (b − a) = a + min  1, max  0, (b − a)> Σ−1 T (µT − a) (b − a)> Σ−1 T (b − a)  · (b − a). (21) 

B.1.3 MULTIPLE-LINE-SEGMENT GOAL-SET:

More generally, we can combine multiple line-segments to form piecewise linear “paths” we wish to follow. By defining a path that connects points (p0, p1, ..., pN ), we can evaluate Liu for each

Gpi→pi+1 line-segment, select the optimal segment i∗ = arg maxi Liu , and use the segment i∗ ’s solution to u∗ to compute s∗T . Examples shown in Fig. 10.

####  B.1.4 POLYGON GOAL-SET

Instead of a route or path, a user (or program) may wish to provide a general region the agent should go to, and state within that region being equally valid. Polygon regions (including both boundary and interior) offer closed form solution to equation 6 and are simple to specify. A polygon can be specified by an ordered sequence of vertices (p0, p1, ..., pN ) ∈ RN×2 . Edges are then defined as the sequence of line-segments between successive vertices (and a final edge between first and last vertex): ((p0, p1), ...,(pN−1, pN ),(pN , p0)). Examples shown in Fig. 11 and 12.

Solving equation 6 with a polygon has two cases: depending whether µT is inside the polygon, or outside. If µT lies inside the polygon, then the optimal value for s∗T that maximizes N (s∗T ; µT , ΣT ) is simply µT : the mode of the Gaussian distribution. Otherwise, if µT lies outside the polygon, then the optimal value s∗T will lie on one of the polygon’s edges, solved using B.1.3.

### B.2 WAYPOINTER DETAILS
The waypointer uses the CARLA planner’s provided route to generate waypoints. In the constrainedbased planning goal likelihoods, we use this route to generate waypoints without interpolating between them. In the relaxed goal likelihoods, we interpolate this route to every 2 meters, and use the first 20 waypoints. As mentioned in the main text, one variant of our approach uses a “smart” waypointer. This waypointer simply removes nearby waypoints closer than 5 meters from the vehicle when a green light is observed in the measurements provided by CARLA, to encourage the agent to move forward, and removes far waypoints beyond 5 meters from the vehicle when a red light is observed in the measurements provided by CARLA. Note that the performance differences between our method without the smart waypointer and our method with the smart waypointer are small: the only signal in the metrics is that the smart waypointer improves the vehicle’s ability to stop for red lights, however, it is quite adept at doing so without the smart waypointer.

### B.3 CONSTRUCTING GOAL SETS
Given the in-lane waypoints generated by CARLA’s route planner, we use these to create Point goal sets, Line-Segment goal sets, and Polygon Goal-Sets, which respectively correspond to the (A)

Final-State Indicator, (B) Line-Segment Final-State Indicator, and (C) Final-State Region Indicator described in Section 2.2. For (A), we simply feed the waypoints directly into the Final-State Indicator, which results in a constrained optimization to ensure that ST ∈ G = {gkT }Kk=1. We also included the vehicle’s current position in the goal set, in order to allow it to stop. The gradient-descent based optimization is then formed from combining Eq. 8 with Eq. 12. The gradient to the nearest goal of the final state of the partially-optimized plan encourage the optimization to move the plan closer to that goal. We used K = 10. We applied the same procedure to generate the goal set for the (B)

Line Segment indicator, as the waypoints returned by the planner are ordered. Finally, for the (C)

Final-State Region Indicator (polygon), we used the ordered waypoints as the “skeleton” of a polygon that surrounds. It was created by adding a two vertices for each point vt in the skeleton at a distance 1 meter from vt perpendicular to the segment connecting the surrounding points (vt−1, vt+1). This resulted in a goal set Gpolygon ⊃ Gline-segment, as it surrounds the line segments. The (F) Gaussian

Final-State Mixture goal set was constructed in the same way as (A), and also used when the pothole costs were added.

For the methods we implemented, the task is to drive the furthest road location from the vehicle’s initial position. Note that this protocol more difficult than the one used in prior work Codevilla et al. (2018); Liang et al. (2018); Sauer et al. (2018); Li et al. (2018); Codevilla et al. (2019), which has no distance guarantees between start positions and goals, and often results in shorter paths. 

Figure 9: Planning with the Final State Indicator yields plans that end at one of the provided locations.

The orange diamonds indicate the locations in the goal set. The red circles indicate the chosen plan.

Figure 10: Planning with the Line Segment Final State Indicator yields plans that end along one of the segments. The orange diamonds indicate the endpoints of each line segment. The red circles indicate the chosen plan.

Figure 11: Planning with the Region Final State Indicator yields plans that end inside the region. The orange polygon indicates the region. The red circles indicate the chosen plan. 

Figure 12: Planning with the Region Final State Indicator yields plans that end inside the region.

The orange polygon indicates the region. The red circles indicate the chosen plan. Note even with a wider goal region than Fig. 11, the vehicle remains in its lane, due to the imitation prior. Despite their coarseness, these wide goal regions still provide useful guidance to the vehicle.

### B.4 PLANNING VISUALIZATIONS
Visualizations of examples of our method deployed with different goal likelihoods are shown in

Fig. 9, Fig. 10, Fig. 11, and Fig. 12.

## C ARCHITECTURE AND TRAINING DETAILS

The architecture of q(S|φ) is shown in Table 5.

Table 5: Detailed Architecture that implements s1:T = f(z1:T , φ). Typically, T = 40, D = 2,H = W = 200.

Component Input [dimensionality] Layer or Operation Output [dimensionality] Details

Static featurization of context: φ = {χ, s 1:A−τ:0}.

MapFeat χ [H, W, 2] 2D Convolution 1χ [H, W, 32] 3 × 3 stride 1, ReLu

MapFeat i−1χ [H, W, 32] 2D Convolution iχ [H, W, 32] 3 × 3 stride 1, ReLu, i ∈ [2, . . . , 8]

MapFeat 8χ [H, W, 32] 2D Convolution Γ [H, W, 8] 3 × 3 stride 1, ReLu

PastRNN s−τ:0 [τ + 1, D] RNN α [32] GRU across time dimension

Dynamic generation via loop: for t ∈ {0, . . . , T − 1}.

MapFeat st [D] Interpolate γt = Γ(st) [8] Differentiable interpolation

JointFeat γt, s0, 2η, α, λ γt ⊕ s0 ⊕ 2η ⊕ α ⊕ λ ρt [D + 50 + 32 + 1] Concatenate (⊕)

FutureRNN ρt [D + 50 + 32 + 1] RNN 1ρt [50] GRU

FutureMLP 1ρt [50] Affine (FC) 2ρt [200] Tanh activation

FutureMLP 2ρt[200] Affine (FC) mt [D], ξt [D, D] Identity activation

MatrixExp ξt [D, D] expm(ξt + ξt a,transpose) σt [D, D] Differentiable Matrix Exponential Rhinehart et al. (2018)

VerletStep st, st−1, mt,σt, zt 2st − st−1 + mt + σtzt st+1 [D]

### C.1 PRIOR VISUALIZATION AND STATISTICS
We show examples of the priors multimodality in Fig. 13

C.1.1 STATISTICS OF PRIOR AND GOAL LIKELIHOODS

Following are the values of the planning criterion on N ≈ 8 · 103 rounds from applying the “Gaussian

Final-State Mixture” to Town01 Dynamic. Mean of log q(s∗|φ) ≈ 104. Mean of log p(G|s∗ , φ) = −4.

This illustrates that while the prior’s value mostly dominates the values of the final plans, the Gaussian

Final-State Goal Mixture likelihood has a moderate amount of influence on the value of the final plan. 

Figure 13: Left: Samples from the prior, q(S|φ), go left or right. Right: Samples go forward or right.

### C.2 DATASET
Before training q(S|φ), we ran CARLA’s expert in the dynamic world setting of Town01 to collect a dataset of examples. We have prepared the dataset of collected data for public release upon publication. We ran the autopilot in Town01 for over 900 episodes of 100 seconds each in the presence of 100 other vehicles, and recorded the trajectory of every vehicle and the autopilot’s LIDAR observation. We randomized episodes to either train, validation, or test sets. We created sets of 60,701 train, 7586 validation, and 7567 test scenes, each with 2 seconds of past and 4 seconds of future position information at 10Hz. The dataset also includes 100 episodes obtained by following the same procedure in Town02.

D BASELINE DETAILS

### D.1 CONDITIONAL IMITATION LEARNING OF STATES (CILS):
We designed a conditional imitation learning baselines that predicts the setpoint for the PID-controller.

Each receives the same scene observations (LIDAR) and is trained with the same set of trajectories as our main method. It uses nearly the same architecture as that of the original CIL, except it outputs setpoints instead of controls, and also observes the traffic light information. We found it very effective for stable control on straightaways. When the model encounters corners, however, prediction is more difficult, as in order to successfully avoid the curbs, the model must implicitly plan a safe path. We found that using the traffic light information allowed it to stop more frequently.

### D.2 MODEL-BASED REINFORCEMENT LEARNING:
Static-world To compare against a purely model-based reinforcement learning algorithm, we propose a model-based reinforcement learning baseline. This baseline first learns a forwards dynamics model st+1 = f(st−3:t, at) given observed expert data (at are recorded vehicle actions). We use an MLP with two hidden layers, each 100 units. Note that our forwards dynamics model does not imitate the expert preferred actions, but only models what is physically possible. Together with the same LIDAR map χ our method uses to locate obstacles, this baseline uses its dynamics model to plan a reachability tree LaValle (2006) through the free-space to the waypoint while avoiding obstacles. The planner opts for the lowest-cost path that ends near the goal C(s1:T ; gT ) = ||sT − gT ||2 + P Tt=1 c(st), where cost of a position is determined by c(st) = 1.5✶(st < 1 meters from any obstacle) + 0.75✶(1 <= st < 2 meters from any obstacle) + ... st.

We plan forwards over 20 time steps using a breadth-first search search over CARLA steering angle {−0.3, −0.1, 0., 0.1, 0.3}, noting valid steering angles are normalized to [−1, 1], with constant throttle at 0.5, noting the valid throttle range is [0, 1]. Our search expands each state node by the available actions and retains the 50 closest nodes to the waypoint. The planned trajectory efficiently reaches the waypoint, and can successfully plan around perceived obstacles to avoid getting stuck. To convert the LIDAR images into obstacle maps, we expanded all obstacles by the approximate radius of the car, 1.5 meters. 

Dynamic-world We use the same setup as the Static-MBRL method, except we add a discrete temporal dimension to the search space (one R2 spatial dimension per T time steps into the future).

All static obstacles remain static, however all LIDAR points that were known to collide with a vehicle are now removed: and replaced at every time step using a constant velocity model of that vehicle. We found that the main failure mode was due to both to inaccuracy in constant velocity prediction as well as the model’s inability to perceive lanes in the LIDAR. The vehicle would sometimes wander into the opposing traffic’s lane, having failed to anticipate an oncoming vehicle blocking its path.

E ROBUSTNESS EXPERIMENTS DETAILS

### E.1 DECOY WAYPOINTS
In the decoy waypoints experiment, the perturbation distribution is N (0, σ = 8m): each waypoint is perturbed with a standard deviation of 8 meters. One failure mode of this approach is when decoy waypoints lie on a valid off-route path at intersections, which temporarily confuses the planner about the best route. Additional visualizations are shown in Fig. 14.

Figure 14: Tolerating bad waypoints. The planner prefers waypoints in the distribution of expert behavior (on the road at a reasonable distance). Columns 1,2: Planning with 1/2 decoy waypoints.

Columns 3,4: Planning with all waypoints on the wrong side of the road.

### E.2 PLAN RELIABILITY ESTIMATION
Besides using our model to make a best-effort attempt to reach a user-specified goal, the fact that our model produces explicit likelihoods can also be leveraged to test the reliability of a plan by evaluating whether reaching particular waypoints will result in human-like behavior or not. This capability can be quite important for real-world safety-critical applications, such as autonomous driving, and can be used to build a degree of fault tolerance into the system. We designed a classification experiment to evaluate how well our model can recognize safe and unsafe plans. We planned our model to known good waypoints (where the expert actually went) and known bad waypoints (off-road) on 1650 held-out test scenes. We used the planning criterion to classify these as good and bad plans and found that we can detect these bad plans with 97.5% recall and 90.2% precision. This result indicates imitative models could be effective in estimating the reliability of plans.

We determined a threshold on the planning criterion by single-goal planning to the expert’s final location on offline validation data and setting it to the criterion’s mean minus one stddev. Although a more intelligent calibration could be performed by analyzing the information retrieval statistics on the offline validation, we found this simple calibration to yield reasonably good performance.

We used 1650 test scenes to perform classification of plans to three different types of waypoints 1) where the expert actually arrived at time T (89.4% reliable), 2) waypoints 20m ahead along the waypointer-provided route, which are often near where the expert arrives (73.8% reliable) 3) the same waypoints from 2), shifted 2.5m off of the road (2.5% reliable). This shows that our learned model exhibits a strong preference for valid waypoints. Therefore, a waypointer that provides expert waypoints via 1) half of the time, and slightly out-of-distribution waypoints via 3) in the other half, an “unreliable” plan classifier achieves 97.5% recall and 90.2% precision. 

## F POTHOLE EXPERIMENT DETAILS

We simulated potholes in the environment by randomly inserting them in the cost map near each waypoint i with offsets distributed Ni(µ=[−15m, 0m], Σ = diag([1, 0.01])), (i.e. mean-centered on the right side of the lane 15m before each waypoint). We inserted pixels of root cost −1e3 in the cost map at a single sample of each Ni , binary-dilated the cost map by 1/3 of the lane-width (spreading the cost to neighboring pixels), and then blurred the cost map by convolving with a normalized truncated

Gaussian filter of σ = 1 and truncation width 1.

## G BASELINE VISUALIZATIONS

See Fig. 15 for a visualization of our baseline methods.

Figure 15: Baseline methods we compare against. The red crosses indicate the past 10 positions of the agent. Left: Imitation Learning baseline: the green cross indicates the provided goal, and the yellow plus indicates the predicted setpoint for the controller. Right: Model-based RL baseline: the green regions indicate the model’s predicted reachability, the red regions are post-processed LIDAR used to create its obstacle map.

## H HYPERPARAMETERS

In order to tune the  hyperparameter of the unconstrained likelihoods, we undertook the following binary-search procedure. When the prior frequently overwhelmed the posterior, we set  ← 0.2 , to yield tighter covariances, and thus more penalty for failing to satisfy the goals. When the posterior frequently overwhelmed the prior, we set  ← 5 , to yield looser covariances, and thus less penalty for failing to satisfy the goals. We executed this process three times: once for the “Gaussian FinalState Mixture” experiments (Section 4), once for the “Noise Robustness” Experiments (Section 4.1), and once for the pothole-planning experiments (Section 4.2). Note that for the Constrained-Goal

Likelihoods introduced no hyperparameters to tune.
