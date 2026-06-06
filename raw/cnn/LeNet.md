# GradientBased Learning Applied to Document Recognition
1998.11.11 https://ieeexplore.ieee.org/document/726791

## Abstract
Multilayer Neural Networks trained with the backpropa gation algorithm constitute the best example of a successful

GradientBased Learning technique Given an appropriate network architecture GradientBased Learning algorithms can be used to synthesize a complex decision surface that can classify highdimensional patterns such as handwritten char acters with minimal preprocessing This paper reviews var ious methods applied to handwritten character recognition and compares them on a standard handwritten digit recog nition task Convolutional Neural Networks that are specif ically designed to deal with the variability of D shapes are shown to outperform all other techniques

Reallife document recognition systems are composed of multiple modules including eld extraction segmenta tion recognition and language modeling A new learning paradigm called Graph Transformer Networks GTN al lows such multimodule systems to be trained globally using

GradientBased methods so as to minimize an overall per formance measure

Two systems for online handwriting recognition are de scribed Experiments demonstrate the advantage of global training and the 	exibility of Graph Transformer Networks

A Graph Transformer Network for reading bank check is also described It uses Convolutional Neural Network char acter recognizers combined with global training techniques to provides record accuracy on business and personal checks

It is deployed commercially and reads several million checks per day Keywords Neural Networks OCR Document Recogni tion Machine Learning GradientBased Learning Convo lutional Neural Networks Graph Transformer Networks Fi nite State Transducers

Nomenclature 

GT Graph transformer

GTN Graph transformer network

HMM Hidden Markov model

HOS Heuristic oversegmentation

KNN Knearest neighbor

NN Neural network

OCR Optical character recognition

PCA Principal component analysis

RBF Radial basis function

RSSVM Reducedset support vector method

SDNN Space displacement neural network

SVM Support vector method

TDNN Time delay neural network

VSVM Virtual support vector method

The authors are with the Speech and Image Pro cessing Services Research Laboratory ATT Labs

Research  Schulz Drive Red Bank NJ  Email fyannleonbyoshuaha nergresearchattcom Yoshua Bengio is also with the Departement d

Informatique et de Recherche

Operationelle Universite de Montreal CP  Succ CentreVille  Chemin de la Tour Montreal Quebec Canada HC J

## I Introduction
Over the last several years machine learning techniques particularly when applied to neural networks have played an increasingly important role in the design of pattern recognition systems In fact it could be argued that the availability of learning techniques has been a crucial fac tor in the recent success of pattern recognition applica tions such as continuous speech recognition and handwrit ing recognition

The main message of this paper is that better pattern recognition systems can be built by relying more on auto matic learning and less on handdesigned heuristics This is made possible by recent progress in machine learning and computer technology Using character recognition as a case study we show that handcrafted feature extrac tion can be advantageously replaced by carefully designed learning machines that operate directly on pixel images

Using document understanding as a case study we show that the traditional way of building recognition systems by manually integrating individually designed modules can be replaced by a unied and wellprincipled design paradigm called Graph Transformer Networks that allows training all the modules to optimize a global performance criterion

Since the early days of pattern recognition it has been known that the variability and richness of natural data be it speech glyphs or other types of patterns make it almost impossible to build an accurate recognition system entirely by hand Consequently most pattern recognition systems are built using a combination of automatic learn ing techniques and handcrafted algorithms The usual method of recognizing individual patterns consists in divid ing the system into two main modules shown in gure 

The rst module called the feature extractor transforms the input patterns so that they can be represented by low dimensional vectors or short strings of symbols that a can be easily matched or compared and b are relatively in variant with respect to transformations and distortions of the input patterns that do not change their nature The feature extractor contains most of the prior knowledge and is rather specic to the task It is also the focus of most of the design e	ort because it is often entirely handcrafted

The classier on the other hand is often generalpurpose and trainable One of the main problems with this ap proach is that the recognition accuracy is largely deter mined by the ability of the designer to come up with an appropriate set of features This turns out to be a daunt ing task which unfortunately must be redone for each new problem A large amount of the pattern recognition liter ature is devoted to describing and comparing the relative

PROC OF THE IEEE NOVEMBER  

TRAINABLE CLASSIFIER MODULE

FEATURE EXTRACTION MODULE

Class scores

Feature vector

Raw input

Fig  Traditional pattern recognition is performed with two mod ules	 a xed feature extractor and a trainable classier merits of di	erent feature sets for particular tasks

Historically the need for appropriate feature extractors was due to the fact that the learning techniques used by the classiers were limited to lowdimensional spaces with easily separable classes  A combination of three factors have changed this vision over the last decade First the availability of lowcost machines with fast arithmetic units allows to rely more on bruteforce numerical methods than on algorithmic renements Second the availability of large databases for problems with a large market and wide interest such as handwriting recognition has enabled designers to rely more on real data and less on handcrafted feature extraction to build recognition systems The third and very important factor is the availability of powerful ma chine learning techniques that can handle highdimensional inputs and can generate intricate decision functions when fed with these large data sets It can be argued that the recent progress in the accuracy of speech and handwriting recognition systems can be attributed in large part to an increased reliance on learning techniques and large training data sets As evidence to this fact a large proportion of modern commercial OCR systems use some form of multi layer Neural Network trained with backpropagation

In this study we consider the tasks of handwritten char acter recognition Sections I and II and compare the per formance of several learning techniques on a benchmark data set for handwritten digit recognition Section III

While more automatic learning is benecial no learning technique can succeed without a minimal amount of prior knowledge about the task In the case of multilayer neu ral networks a good way to incorporate knowledge is to tailor its architecture to the task Convolutional Neu ral Networks  introduced in Section II are an exam ple of specialized neural network architectures which in corporate knowledge about the invariances of D shapes by using local connection patterns and by imposing con straints on the weights A comparison of several methods for isolated handwritten digit recognition is presented in section III To go from the recognition of individual char acters to the recognition of words and sentences in docu ments the idea of combining multiple modules trained to reduce the overall error is introduced in Section IV Rec ognizing variablelength ob jects such as handwritten words using multimodule systems is best done if the modules manipulate directed graphs This leads to the concept of trainable Graph Transformer Network GTN also intro duced in Section IV Section V describes the now clas sical method of heuristic oversegmentation for recogniz ing words or other character strings Discriminative and nondiscriminative gradientbased techniques for training a recognizer at the word level without requiring manual segmentation and labeling are presented in Section VI Sec tion VII presents the promising SpaceDisplacement Neu ral Network approach that eliminates the need for seg mentation heuristics by scanning a recognizer at all pos sible locations on the input In section VIII it is shown that trainable Graph Transformer Networks can be for mulated as multiple generalized transductions based on a general graph composition algorithm The connections be tween GTNs and Hidden Markov Models commonly used in speech recognition is also treated Section IX describes a globally trained GTN system for recognizing handwrit ing entered in a pen computer This problem is known as online handwriting recognition since the machine must produce immediate feedback as the user writes The core of the system is a Convolutional Neural Network The results clearly demonstrate the advantages of training a recognizer at the word level rather than training it on presegmented handlabeled isolated characters Section X describes a complete GTNbased system for reading handwritten and machineprinted bank checks The core of the system is the

Convolutional Neural Network called LeNet described in

Section II This system is in commercial use in the NCR

Corporation line of check recognition systems for the bank ing industry It is reading millions of checks per month in several banks across the United States

## A Learning from Data

There are several approaches to automatic machine learning but one of the most successful approaches pop ularized in recent years by the neural network community can be called numerical or gradientbased learning The learning machine computes a function Y p  F Zp

W where Zp is the pth input pattern and W represents the collection of adjustable parameters in the system In a pattern recognition setting the output Y p may be inter preted as the recognized class label of pattern Zp or as scores or probabilities associated with each class A loss function Ep  DDp

F W Zp  measures the discrep ancy between Dp  the correct or desired output for pat tern Zp and the output produced by the system The average loss function EtrainW is the average of the er rors Ep over a set of labeled examples called the training set fZ

D ZP

DP g In the simplest setting the learning problem consists in nding the value of W that minimizes EtrainW In practice the performance of the system on a training set is of little interest The more rel evant measure is the error rate of the system in the eld where it would be used in practice This performance is estimated by measuring the accuracy on a set of samples disjoint from the training set called the test set Much theoretical and experimental work    has shown

PROC OF THE IEEE NOVEMBER   that the gap between the expected error rate on the test set Etest and the error rate on the training set Etrain de creases with the number of training samples approximately as

Etest  Etrain  khP   where P is the number of training samples h is a measure of e	ective capacity or complexity of the machine    is a number between  and  and k is a constant This gap always decreases when the number of training samples increases Furthermore as the capacity h increases Etrain decreases Therefore when increasing the capacity h there is a tradeo	 between the decrease of Etrain and the in crease of the gap with an optimal value of the capacity h that achieves the lowest generalization error Etest  Most learning algorithms attempt to minimize Etrain as well as some estimate of the gap A formal version of this is called structural risk minimization   and is based on den ing a sequence of learning machines of increasing capacity corresponding to a sequence of subsets of the parameter space such that each subset is a superset of the previous subset In practical terms Structural Risk Minimization is implemented by minimizing Etrain  HW where the function HW is called a regularization function and  is a constant HW is chosen such that it takes large val ues on parameters W that belong to highcapacity subsets of the parameter space Minimizing HW in e	ect lim its the capacity of the accessible subset of the parameter space thereby controlling the tradeo	 between minimiz ing the training error and minimizing the expected gap between the training error and test error

## B GradientBased Learning

The general problem of minimizing a function with re spect to a set of parameters is at the root of many issues in computer science GradientBased Learning draws on the fact that it is generally much easier to minimize a reason ably smooth continuous function than a discrete combi natorial function The loss function can be minimized by estimating the impact of small variations of the parame ter values on the loss function This is measured by the gradient of the loss function with respect to the param eters Ecient learning algorithms can be devised when the gradient vector can be computed analytically as op posed to numerically through perturbations This is the basis of numerous gradientbased learning algorithms with continuousvalued parameters In the procedures described in this article the set of parameters W is a realvalued vec tor with respect to which EW is continuous as well as di	erentiable almost everywhere The simplest minimiza tion procedure in such a setting is the gradient descent algorithm where W is iteratively adjusted as follows

Wk  Wk   EW W  

In the simplest case  is a scalar constant More sophisti cated procedures use variable  or substitute it for a diag onal matrix or substitute it for an estimate of the inverse

Hessian matrix as in Newton or QuasiNewton methods

The Conjugate Gradient method  can also be used

However Appendix B shows that despite many claims to the contrary in the literature the usefulness of these secondorder methods to large learning machines is very limited

A popular minimization procedure is the stochastic gra dient algorithm also called the online update It consists in updating the parameter vector using a noisy or approx imated version of the average gradient In the most com mon instance of it W is updated on the basis of a single sample

Wk  Wk   Epk W W 

With this procedure the parameter vector uctuates around an average tra jectory but usually converges consid erably faster than regular gradient descent and second or der methods on large training sets with redundant samples such as those encountered in speech or character recogni tion The reasons for this are explained in Appendix B

The properties of such algorithms applied to learning have been studied theoretically since the s    but practical successes for nontrivial tasks did not occur until the mid eighties

## C Gradient BackPropagation

GradientBased Learning procedures have been used since the late s but they were mostly limited to lin ear systems  The surprising usefulness of such sim ple gradient descent techniques for complex machine learn ing tasks was not widely realized until the following three events occurred The rst event was the realization that despite early warnings to the contrary  the presence of local minima in the loss function does not seem to be a ma jor problem in practice This became apparent when it was noticed that local minima did not seem to be a ma jor impediment to the success of early nonlinear gradientbased Learning techniques such as Boltzmann ma chines   The second event was the popularization by Rumelhart Hinton and Williams  and others of a simple and ecient procedure the backpropagation al gorithm to compute the gradient in a nonlinear system composed of several layers of processing The third event was the demonstration that the backpropagation proce dure applied to multilayer neural networks with sigmoidal units can solve complicated learning tasks The basic idea of backpropagation is that gradients can be computed e ciently by propagation from the output to the input This idea was described in the control theory literature of the early sixties  but its application to machine learning was not generally realized then Interestingly the early derivations of backpropagation in the context of neural network learning did not use gradients but virtual tar gets for units in intermediate layers   or minimal disturbance arguments  The Lagrange formalism used in the control theory literature provides perhaps the best rigorous method for deriving backpropagation  and for deriving generalizations of backpropagation to recurrent

PROC OF THE IEEE NOVEMBER  networks  and networks of heterogeneous modules 

A simple derivation for generic multilayer systems is given in Section IE

The fact that local minima do not seem to be a problem for multilayer neural networks is somewhat of a theoretical mystery It is conjectured that if the network is oversized for the task as is usually the case in practice the presence of extra dimensions in parameter space reduces the risk of unattainable regions Backpropagation is by far the most widely used neuralnetwork learning algorithm and probably the most widely used learning algorithm of any form

## D Learning in Real Handwriting Recognition Systems

Isolated handwritten character recognition has been ex tensively studied in the literature see   for reviews and was one of the early successful applications of neural networks  Comparative experiments on recognition of individual handwritten digits are reported in Section III

They show that neural networks trained with Gradient

Based Learning perform better than all other methods tested here on the same data The best neural networks called Convolutional Networks are designed to learn to extract relevant features directly from pixel images see

Section II

One of the most dicult problems in handwriting recog nition however is not only to recognize individual charac ters but also to separate out characters from their neigh bors within the word or sentence a process known as seg mentation The technique for doing this that has become the standard is called Heuristic OverSegmentation It consists in generating a large number of potential cuts between characters using heuristic image processing tech niques and subsequently selecting the best combination of cuts based on scores given for each candidate character by the recognizer In such a model the accuracy of the sys tem depends upon the quality of the cuts generated by the heuristics and on the ability of the recognizer to distin guish correctly segmented characters from pieces of char acters multiple characters or otherwise incorrectly seg mented characters Training a recognizer to perform this task poses a ma jor challenge because of the diculty in cre ating a labeled database of incorrectly segmented charac ters The simplest solution consists in running the images of character strings through the segmenter and then man ually labeling all the character hypotheses Unfortunately not only is this an extremely tedious and costly task it is also dicult to do the labeling consistently For example should the right half of a cut up  be labeled as a  or as a noncharacter should the right half of a cut up  be labeled as a 

The rst solution described in Section V consists in training the system at the level of whole strings of char acters rather than at the character level The notion of

GradientBased Learning can be used for this purpose The system is trained to minimize an overall loss function which measures the probability of an erroneous answer Section V explores various ways to ensure that the loss function is dif ferentiable and therefore lends itself to the use of Gradient

Based Learning methods Section V introduces the use of directed acyclic graphs whose arcs carry numerical infor mation as a way to represent the alternative hypotheses and introduces the idea of GTN

The second solution described in Section VII is to elim inate segmentation altogether The idea is to sweep the recognizer over every possible location on the input image and to rely on the character spotting property of the rec ognizer ie its ability to correctly recognize a wellcentered character in its input eld even in the presence of other characters besides it while rejecting images containing no centered characters   The sequence of recognizer outputs obtained by sweeping the recognizer over the in put is then fed to a Graph Transformer Network that takes linguistic constraints into account and nally extracts the most likely interpretation This GTN is somewhat similar to Hidden Markov Models HMM which makes the ap proach reminiscent of the classical speech recognition    While this technique would be quite expensive in the general case the use of Convolutional Neural Networks makes it particularly attractive because it allows signicant savings in computational cost

## E Globally Trainable Systems

As stated earlier most practical pattern recognition sys tems are composed of multiple modules For example a document recognition system is composed of a eld locator which extracts regions of interest a eld segmenter which cuts the input image into images of candidate characters a recognizer which classies and scores each candidate char acter and a contextual postprocessor generally based on a stochastic grammar which selects the best grammatically correct answer from the hypotheses generated by the recog nizer In most cases the information carried from module to module is best represented as graphs with numerical in formation attached to the arcs For example the output of the recognizer module can be represented as an acyclic graph where each arc contains the label and the score of a candidate character and where each path represent a alternative interpretation of the input string Typically each module is manually optimized or sometimes trained outside of its context For example the character recog nizer would be trained on labeled images of presegmented characters Then the complete system is assembled and a subset of the parameters of the modules is manually ad justed to maximize the overall performance This last step is extremely tedious timeconsuming and almost certainly suboptimal

A better alternative would be to somehow train the en tire system so as to minimize a global error measure such as the probability of character misclassications at the docu ment level Ideally we would want to nd a good minimum of this global loss function with respect to all the param eters in the system If the loss function E measuring the performance can be made di	erentiable with respect to the systems tunable parameters W we can nd a local min imum of E using GradientBased Learning However at

PROC OF THE IEEE NOVEMBER   rst glance it appears that the sheer size and complexity of the system would make this intractable

To ensure that the global loss function EpZp

W is dif ferentiable the overall system is built as a feedforward net work of di	erentiable modules The function implemented by each module must be continuous and di	erentiable al most everywhere with respect to the internal parameters of the module eg the weights of a Neural Net character rec ognizer in the case of a character recognition module and with respect to the modules inputs If this is the case a simple generalization of the wellknown backpropagation procedure can be used to eciently compute the gradients of the loss function with respect to all the parameters in the system  For example let us consider a system built as a cascade of modules each of which implements a function Xn  FnWn Xn where Xn is a vector rep resenting the output of the module Wn is the vector of tunable parameters in the module a subset of W and

Xn is the modules input vector as well as the previous modules output vector The input X to the rst module is the input pattern Zp If the partial derivative of Ep with respect to Xn is known then the partial derivatives of Ep with respect to Wn and Xn can be computed using the backward recurrence Ep Wn  F W Wn Xn Ep Xn Ep Xn  F X Wn Xn Ep Xn  where F

W Wn Xn is the Jacobian of F with respect to

W evaluated at the point Wn Xn and F

X Wn Xn is the Jacobian of F with respect to X The Jacobian of a vector function is a matrix containing the partial deriva tives of all the outputs with respect to all the inputs

The rst equation computes some terms of the gradient of EpW while the second equation generates a back ward recurrence as in the wellknown backpropagation procedure for neural networks We can average the gradi ents over the training patterns to obtain the full gradient

It is interesting to note that in many instances there is no need to explicitly compute the Jacobian matrix The above formula uses the product of the Jacobian with a vec tor of partial derivatives and it is often easier to compute this product directly without computing the Jacobian be forehand In By analogy with ordinary multilayer neural networks all but the last module are called hidden layers because their outputs are not observable from the outside more complex situations than the simple cascade of mod ules described above the partial derivative notation be comes somewhat ambiguous and awkward A completely rigorous derivation in more general cases can be done using

Lagrange functions   

Traditional multilayer neural networks are a special case of the above where the state information Xn is represented with xedsized vectors and where the modules are al ternated layers of matrix multiplications the weights and componentwise sigmoid functions the neurons However as stated earlier the state information in complex recogni tion system is best represented by graphs with numerical information attached to the arcs In this case each module called a Graph Transformer takes one or more graphs as input and produces a graph as output Networks of such modules are called Graph Transformer Networks GTN

Sections IV VI and VIII develop the concept of GTNs and show that GradientBased Learning can be used to train all the parameters in all the modules so as to mini mize a global loss function It may seem paradoxical that gradients can be computed when the state information is represented by essentially discrete ob jects such as graphs but that diculty can be circumvented as shown later

II Convolutional Neural Networks for

Isolated Character Recognition

The ability of multilayer networks trained with gradi ent descent to learn complex highdimensional nonlinear mappings from large collections of examples makes them obvious candidates for image recognition tasks In the tra ditional model of pattern recognition a handdesigned fea ture extractor gathers relevant information from the input and eliminates irrelevant variabilities A trainable classier then categorizes the resulting feature vectors into classes

In this scheme standard fullyconnected multilayer net works can be used as classiers A potentially more inter esting scheme is to rely on as much as possible on learning in the feature extractor itself In the case of character recognition a network could be fed with almost raw in puts eg sizenormalized images While this can be done with an ordinary fully connected feedforward network with some success for tasks such as character recognition there are problems

Firstly typical images are large often with several hun dred variables pixels A fullyconnected rst layer with say one hundred hidden units in the rst layer would al ready contain several tens of thousands of weights Such a large number of parameters increases the capacity of the system and therefore requires a larger training set In ad dition the memory requirement to store so many weights may rule out certain hardware implementations But the main deciency of unstructured nets for image or speech applications is that they have no builtin invariance with respect to translations or local distortions of the inputs

Before being sent to the xedsize input layer of a neural net character images or other D or D signals must be approximately sizenormalized and centered in the input eld Unfortunately no such preprocessing can be perfect handwriting is often normalized at the word level which can cause size slant and position variations for individual characters This combined with variability in writing style will cause variations in the position of distinctive features in input ob jects In principle a fullyconnected network of sucient size could learn to produce outputs that are in variant with respect to such variations However learning such a task would probably result in multiple units with similar weight patterns positioned at various locations in the input so as to detect distinctive features wherever they appear on the input Learning these weight congurations

PROC OF THE IEEE NOVEMBER  requires a very large number of training instances to cover the space of possible variations In convolutional networks described below shift invariance is automatically obtained by forcing the replication of weight congurations across space

Secondly a deciency of fullyconnected architectures is that the topology of the input is entirely ignored The in put variables can be presented in any xed order without a	ecting the outcome of the training On the contrary images or timefrequency representations of speech have a strong D local structure variables or pixels that are spatially or temporally nearby are highly correlated Local correlations are the reasons for the wellknown advantages of extracting and combining local features before recogniz ing spatial or temporal ob jects because congurations of neighboring variables can be classied into a small number of categories eg edges corners Convolutional Net works force the extraction of local features by restricting the receptive elds of hidden units to be local

A Convolutional Networks

Convolutional Networks combine three architectural ideas to ensure some degree of shift scale and distor tion invariance local receptive elds shared weights or weight replication and spatial or temporal subsampling A typical convolutional network for recognizing characters dubbed LeNet is shown in gure  The input plane receives images of characters that are approximately size normalized and centered Each unit in a layer receives in puts from a set of units located in a small neighborhood in the previous layer The idea of connecting units to local receptive elds on the input goes back to the Perceptron in the early s and was almost simultaneous with Hubel and

Wiesels discovery of locallysensitive orientationselective neurons in the cats visual system  Local connections have been used many times in neural models of visual learn ing       With local receptive elds neurons can extract elementary visual features such as oriented edges endpoints corners or similar features in other signals such as speech spectrograms These features are then combined by the subsequent layers in order to de tect higherorder features As stated earlier distortions or shifts of the input can cause the position of salient features to vary In addition elementary feature detectors that are useful on one part of the image are likely to be useful across the entire image This knowledge can be applied by forcing a set of units whose receptive elds are located at di	erent places on the image to have identical weight vectors     Units in a layer are organized in planes within which all the units share the same set of weights The set of outputs of the units in such a plane is called a feature map Units in a feature map are all constrained to per form the same operation on di	erent parts of the image

A complete convolutional layer is composed of several fea ture maps with di	erent weight vectors so that multiple features can be extracted at each location A concrete ex ample of this is the rst layer of LeNet shown in Figure 

Units in the rst hidden layer of LeNet are organized in  planes each of which is a feature map A unit in a feature map has  inputs connected to a  by  area in the input called the receptive eld of the unit Each unit has  in puts and therefore  trainable coecients plus a trainable bias The receptive elds of contiguous units in a feature map are centered on correspondingly contiguous units in the previous layer Therefore receptive elds of neighbor ing units overlap For example in the rst hidden layer of LeNet the receptive elds of horizontally contiguous units overlap by  columns and  rows As stated earlier all the units in a feature map share the same set of  weights and the same bias so they detect the same feature at all possible locations on the input The other feature maps in the layer use di	erent sets of weights and biases thereby extracting di	erent types of local features In the case of LeNet at each input location six di	erent types of features are extracted by six units in identical locations in the six feature maps A sequential implementation of a feature map would scan the input image with a single unit that has a local receptive eld and store the states of this unit at corresponding locations in the feature map

This operation is equivalent to a convolution followed by an additive bias and squashing function hence the name convolutional network The kernel of the convolution is the set of connection weights used by the units in the feature map An interesting property of convolutional layers is that if the input image is shifted the feature map output will be shifted by the same amount but will be left unchanged otherwise This property is at the basis of the robustness of convolutional networks to shifts and distortions of the input

Once a feature has been detected its exact location becomes less important Only its approximate position relative to other features is relevant For example once we know that the input image contains the endpoint of a roughly horizontal segment in the upper left area a corner in the upper right area and the endpoint of a roughly ver tical segment in the lower portion of the image we can tell the input image is a  Not only is the precise position of each of those features irrelevant for identifying the pattern it is potentially harmful because the positions are likely to vary for di	erent instances of the character A simple way to reduce the precision with which the position of distinc tive features are encoded in a feature map is to reduce the spatial resolution of the feature map This can be achieved with a socalled subsampling layers which performs a local averaging and a subsampling reducing the resolution of the feature map and reducing the sensitivity of the output to shifts and distortions The second hidden layer of LeNet  is a subsampling layer This layer comprises six feature maps one for each feature map in the previous layer The receptive eld of each unit is a  by  area in the previous layers corresponding feature map Each unit computes the average of its four inputs multiplies it by a trainable coef cient adds a trainable bias and passes the result through a sigmoid function Contiguous units have nonoverlapping contiguous receptive elds Consequently a subsampling layer feature map has half the number of rows and columns

PROC OF THE IEEE NOVEMBER  

INPUT 32x32

Convolutions Subsampling Convolutions

C1: feature maps 6@28x28

Subsampling

S2: f. maps 6@14x14

S4: f. maps 16@5x5

C5: layer 120

C3: f. maps 16@10x10

F6: layer 84

Full connection

Full connection

Gaussian connections

OUTPUT 10

Fig  Architecture of LeNet a Convolutional Neural Network here for digits recognition Each plane is a feature map ie a set of units whose weights are constrained to be identical as the feature maps in the previous layer The trainable coecient and bias control the e	ect of the sigmoid non linearity If the coecient is small then the unit operates in a quasilinear mode and the subsampling layer merely blurs the input If the coecient is large subsampling units can be seen as performing a noisy OR or a noisy

AND function depending on the value of the bias Succes sive layers of convolutions and subsampling are typically alternated resulting in a bipyramid  at each layer the number of feature maps is increased as the spatial resolu tion is decreased Each unit in the third hidden layer in g ure  may have input connections from several feature maps in the previous layer The convolutionsubsampling com bination inspired by Hubel and Wiesels notions of sim ple and complex cells was implemented in Fukushimas

Neocognitron  though no globally supervised learning procedure such as backpropagation was available then A large degree of invariance to geometric transformations of the input can be achieved with this progressive reduction of spatial resolution compensated by a progressive increase of the richness of the representation the number of feature maps

Since all the weights are learned with backpropagation convolutional networks can be seen as synthesizing their own feature extractor The weight sharing technique has the interesting side e	ect of reducing the number of free parameters thereby reducing the capacity of the ma chine and reducing the gap between test error and training error  The network in gure  contains  con nections but only  trainable free parameters because of the weight sharing

Fixedsize Convolutional Networks have been applied to many applications among other handwriting recogni tion   machineprinted character recognition  online handwriting recognition  and face recogni tion  Fixedsize convolutional networks that share weights along a single temporal dimension are known as

TimeDelay Neural Networks TDNNs TDNNs have been used in phoneme recognition without subsampling    spoken word recognition with subsampling    online recognition of isolated handwritten charac ters  and signature verication 

B LeNet

This section describes in more detail the architecture of

LeNet the Convolutional Neural Network used in the experiments LeNet comprises  layers not counting the input all of which contain trainable parameters weights

The input is a x pixel image This is signicantly larger than the largest character in the database at most x pixels centered in a x eld The reason is that it is desirable that potential distinctive features such as stroke endpoints or corner can appear in the center of the recep tive eld of the highestlevel feature detectors In LeNet the set of centers of the receptive elds of the last convolu tional layer C see below form a x area in the center of the x input The values of the input pixels are nor malized so that the background level white corresponds to a value of  and the foreground black corresponds to  This makes the mean input roughly  and the variance roughly  which accelerates learning 

In the following convolutional layers are labeled Cx sub sampling layers are labeled Sx and fullyconnected layers are labeled Fx where x is the layer index

Layer C is a convolutional layer with  feature maps

Each unit in each feature map is connected to a x neigh borhood in the input The size of the feature maps is x which prevents connection from the input from falling o the boundary C contains  trainable parameters and  connections

Layer S is a subsampling layer with  feature maps of size x Each unit in each feature map is connected to a x neighborhood in the corresponding feature map in C

The four inputs to a unit in S are added then multiplied by a trainable coecient and added to a trainable bias

The result is passed through a sigmoidal function The x receptive elds are nonoverlapping therefore feature maps in S have half the number of rows and column as feature maps in C Layer S has  trainable parameters and  connections

Layer C is a convolutional layer with  feature maps

Each unit in each feature map is connected to several x neighborhoods at identical locations in a subset of Ss feature maps Table I shows the set of S feature maps

PROC OF THE IEEE NOVEMBER                    X X X X X X X X X X  X X X X X X X X X X  X X X X X X X X X X  X X X X X X X X X X  X X X X X X X X X X  X X X X X X X X X X TABLE I

Each column indicates which feature map in S are combined by the units in a particular feature map of C combined by each C feature map Why not connect ev ery S feature map to every C feature map The rea son is twofold First a noncomplete connection scheme keeps the number of connections within reasonable bounds

More importantly it forces a break of symmetry in the net work Di	erent feature maps are forced to extract di	erent hopefully complementary features because they get dif ferent sets of inputs The rationale behind the connection scheme in table I is the following The rst six C feature maps take inputs from every contiguous subsets of three feature maps in S The next six take input from every contiguous subset of four The next three take input from some discontinuous subsets of four Finally the last one takes input from all S feature maps Layer C has  trainable parameters and  connections

Layer S is a subsampling layer with  feature maps of size x Each unit in each feature map is connected to a x neighborhood in the corresponding feature map in C in a similar way as C and S Layer S has  trainable parameters and  connections

Layer C is a convolutional layer with  feature maps

Each unit is connected to a x neighborhood on all  of Ss feature maps Here because the size of S is also x the size of Cs feature maps is x this amounts to a full connection between S and C C is labeled as a convolutional layer instead of a fullyconnected layer because if LeNet input were made bigger with everything else kept constant the feature map dimension would be larger than x This process of dynamically increasing the size of a convolutional network is described in the section

Section VII Layer C has  trainable connections

Layer F contains  units the reason for this number comes from the design of the output layer explained be low and is fully connected to C It has  trainable parameters

As in classical neural networks units in layers up to F compute a dot product between their input vector and their weight vector to which a bias is added This weighted sum denoted ai for unit i is then passed through a sigmoid squashing function to produce the state of unit i denoted by xi  xi  f ai 

The squashing function is a scaled hyperbolic tangent f a  A tanhSa  where A is the amplitude of the function and S determines its slope at the origin The function f is odd with horizon tal asymptotes at A and A The constant A is chosen to be  The rationale for this choice of a squashing function is given in Appendix A

Finally the output layer is composed of Euclidean Radial

Basis Function units RBF one for each class with  inputs each The outputs of each RBF unit yi is computed as follows yi  Xj xj  wij  

In other words each output RBF unit computes the Eu clidean distance between its input vector and its parameter vector The further away is the input from the parameter vector the larger is the RBF output The output of a particular RBF can be interpreted as a penalty term mea suring the t between the input pattern and a model of the class associated with the RBF In probabilistic terms the

RBF output can be interpreted as the unnormalized nega tive loglikelihood of a Gaussian distribution in the space of congurations of layer F Given an input pattern the loss function should be designed so as to get the congu ration of F as close as possible to the parameter vector of the RBF that corresponds to the patterns desired class

The parameter vectors of these units were chosen by hand and kept xed at least initially The components of those parameters vectors were set to  or  While they could have been chosen at random with equal probabilities for  and  or even chosen to form an error correcting code as suggested by  they were instead designed to repre sent a stylized image of the corresponding character class drawn on a x bitmap hence the number  Such a representation is not particularly useful for recognizing iso lated digits but it is quite useful for recognizing strings of characters taken from the full printable ASCII set The rationale is that characters that are similar and therefore confusable such as uppercase O lowercase O and zero or lowercase l digit  square brackets and uppercase I will have similar output codes This is particularly useful if the system is combined with a linguistic postprocessor that can correct such confusions Because the codes for confus able classes are similar the output of the corresponding

RBFs for an ambiguous character will be similar and the postprocessor will be able to pick the appropriate interpre tation Figure  gives the output codes for the full ASCII set

Another reason for using such distributed codes rather than the more common  of N code also called place code or grandmother cell code for the outputs is that non distributed codes tend to behave badly when the num ber of classes is larger than a few dozens The reason is that output units in a nondistributed code must be o most of the time This is quite dicult to achieve with sigmoid units Yet another reason is that the classiers are often used to not only recognize characters but also to re ject noncharacters RBFs with distributed codes are more appropriate for that purpose because unlike sigmoids they are activated within a well circumscribed region of their in

PROC OF THE IEEE NOVEMBER   ! " # $ % & ’ ( ) * + , − . / 0 1 2 3 4 5 6 7 8 9 : ; < = > ? @ A B C D E F G H I J K L M N O P Q R S T U V W X Y Z [ \ ] ^ _ ‘ a b c d e f g h i j k l m n o p q r s t u v w x y z { | } ~ Fig  Initial parameters of the output RBFs for recognizing the full ASCII set put space that nontypical patterns are more likely to fall outside of

The parameter vectors of the RBFs play the role of target vectors for layer F It is worth pointing out that the com ponents of those vectors are  or  which is well within the range of the sigmoid of F and therefore prevents those sigmoids from getting saturated In fact  and  are the points of maximum curvature of the sigmoids This forces the F units to operate in their maximally nonlinear range

Saturation of the sigmoids must be avoided because it is known to lead to slow convergence and illconditioning of the loss function

C Loss Function

The simplest output loss function that can be used with the above network is the Maximum Likelihood Estimation criterion MLE which in our case is equivalent to the Min imum Mean Squared Error MSE The criterion for a set of training samples is simply

EW  P PXp yDp Zp

W  where yDp is the output of the Dpth RBF unit ie the one that corresponds to the correct class of input pattern

Zp While this cost function is appropriate for most cases it lacks three important properties First if we allow the parameters of the RBF to adapt EW has a trivial but totally unacceptable solution In this solution all the RBF parameter vectors are equal and the state of F is constant and equal to that parameter vector In this case the net work happily ignores the input and all the RBF outputs are equal to zero This collapsing phenomenon does not occur if the RBF weights are not allowed to adapt The second problem is that there is no competition between the classes Such a competition can be obtained by us ing a more discriminative training criterion dubbed the

MAP maximum a posteriori criterion similar to Maxi mum Mutual Information criterion sometimes used to train

HMMs    It corresponds to maximizing the posterior probability of the correct class Dp or minimiz ing the logarithm of the probability of the correct class given that the input image can come from one of the classes or from a background rubbish class label In terms of penalties it means that in addition to pushing down the penalty of the correct class like the MSE criterion this criterion also pulls up the penalties of the incorrect classes

EW  P PXp yDp Zp

W  logej Xi eyiZp W 

The negative of the second term plays a competitive role

It is necessarily smaller than or equal to the rst term therefore this loss function is positive The constant j is positive and prevents the penalties of classes that are al ready very large from being pushed further up The pos terior probability of this rubbish class label would be the ratio of ej and ej  Pi eyiZp W This discrimina tive criterion prevents the previously mentioned collaps ing e	ect when the RBF parameters are learned because it keeps the RBF centers apart from each other In Sec tion VI we present a generalization of this criterion for systems that learn to classify multiple ob jects in the input eg characters in words or in documents

Computing the gradient of the loss function with respect to all the weights in all the layers of the convolutional network is done with backpropagation The standard al gorithm must be slightly modied to take account of the weight sharing An easy way to implement it is to rst com pute the partial derivatives of the loss function with respect to each connection as if the network were a conventional multilayer network without weight sharing Then the par tial derivatives of all the connections that share a same parameter are added to form the derivative with respect to that parameter

Such a large architecture can be trained very eciently but doing so requires the use of a few techniques that are described in the appendix Section A of the appendix describes details such as the particular sigmoid used and the weight initialization Section B and C describe the minimization procedure used which is a stochastic version of a diagonal approximation to the LevenbergMarquardt procedure

III Results and Comparison with Other

Methods

While recognizing individual digits is only one of many problems involved in designing a practical recognition sys tem it is an excellent benchmark for comparing shape recognition methods Though many existing method com bine a handcrafted feature extractor and a trainable clas sier this study concentrates on adaptive methods that operate directly on sizenormalized images

A Database the Modied NIST set

The database used to train and test the systems de scribed in this paper was constructed from the NISTs Spe cial Database  and Special Database  containing binary images of handwritten digits NIST originally designated

SD as their training set and SD as their test set How ever SD is much cleaner and easier to recognize than SD  The reason for this can be found on the fact that SD

PROC OF THE IEEE NOVEMBER   was collected among Census Bureau employees while SD was collected among highschool students Drawing sensi ble conclusions from learning experiments requires that the result be independent of the choice of training set and test among the complete set of samples Therefore it was nec essary to build a new database by mixing NISTs datasets

SD contains  digit images written by  dif ferent writers In contrast to SD where blocks of data from each writer appeared in sequence the data in SD is scrambled Writer identities for SD are available and we used this information to unscramble the writers We then split SD in two characters written by the rst  writers went into our new training set The remaining  writers were placed in our test set Thus we had two sets with nearly  examples each The new training set was completed with enough examples from SD starting at pattern   to make a full set of  training patterns

Similarly the new test set was completed with SD exam ples starting at pattern   to make a full set with  test patterns In the experiments described here we only used a subset of  test images  from SD and  from SD but we used the full  training samples The resulting database was called the Modied

NIST or MNIST dataset

The original black and white bilevel images were size normalized to t in a x pixel box while preserving their aspect ratio The resulting images contain grey lev els as result of the antialiasing image interpolation tech nique used by the normalization algorithm Three ver sions of the database were used In the rst version the images were centered in a x image by comput ing the center of mass of the pixels and translating the image so as to position this point at the center of the x eld In some instances this x eld was ex tended to x with background pixels This version of the database will be referred to as the regular database

In the second version of the database the character im ages were deslanted and cropped down to x pixels im ages The deslanting computes the second moments of in ertia of the pixels counting a foreground pixel as  and a background pixel as  and shears the image by horizon tally shifting the lines so that the principal axis is verti cal This version of the database will be referred to as the deslanted database In the third version of the database used in some early experiments the images were reduced to x pixels The regular database  training examples  test examples sizenormalized to x and centered by center of mass in x elds is avail able at httpwwwresearchattcom yannocrmnist Figure  shows examples randomly picked from the test set

B Results

Several versions of LeNet were trained on the regular

MNIST database  iterations through the entire train ing data were performed for each session The values of the global learning rate 	 see Equation  in Appendix C for a denition was decreased using the following sched ule  for the rst two passes  for the next

Fig  Sizenormalized examples from the MNIST database three  for the next three  for the next  and  thereafter Before each iteration the diagonal

Hessian approximation was reevaluated on  samples as described in Appendix C and kept xed during the entire iteration The parameter was set to  The resulting e	ective learning rates during the rst pass varied between approximately    and  over the set of parame ters The test error rate stabilizes after around  passes through the training set at ! The error rate on the training set reaches ! after  passes Many authors have reported observing the common phenomenon of over training when training neural networks or other adaptive algorithms on various tasks When overtraining occurs the training error keeps decreasing over time but the test error goes through a minimum and starts increasing after a certain number of iterations While this phenomenon is very common it was not observed in our case as the learn ing curves in gure  show A possible reason is that the learning rate was kept relatively large The e	ect of this is that the weights never settle down in the local minimum but keep oscillating randomly Because of those uctua tions the average cost will be lower in a broader minimum

Therefore stochastic gradient will have a similar e	ect as a regularization term that favors broader minima Broader minima correspond to solutions with large entropy of the parameter distribution which is benecial to the general ization error

The inuence of the training set size was measured by training the network with   and  exam ples The resulting training error and test error are shown in gure  It is clear that even with specialized architec tures such as LeNet more training data would improve the accuracy To verify this hypothesis we articially generated more training examples by randomly distorting the original training images The increased training set was composed of the  original patterns plus  instances of

PROC OF THE IEEE NOVEMBER   0 4 8 12 16 20 4% 2% 0%

Test

Training

Error Rate (%) 1% 3% 5%

Training set Iterations

Fig  Training and test error of LeNet as a function of the num ber of passes through the  pattern training set without distortions The average training error is measured onthey as training proceeds This explains why the training error appears to be larger than the test error Convergence is attained after  to  passes through the training set 0 0

## 0.2
## 0.4
## 0.6
## 0.8
 1

## 1.2
## 1.4
## 1.6
## 1.8
Training error (no distortions)

Test error (no distortions)

Test error (with distortions)

Training Set Size (x1000) 10 20 30 40 50 60 70 80 90 100

Error Rate (%)

Fig  Training and test errors of LeNet achieved using training sets of various sizes This graph suggests that a larger training set could improve the performance of LeNet The hollow square show the test error when more training patterns are articially generated using random distortions The test patterns are not distorted distorted patterns with randomly picked distortion param eters The distortions were combinations of the follow ing planar ane transformations horizontal and verti cal translations scaling squeezing simultaneous horizon tal compression and vertical elongation or the reverse and horizontal shearing Figure  shows examples of dis torted patterns used for training When distorted data was used for training the test error rate dropped to ! from ! without deformation The same training parame ters were used as without deformations The total length of the training session was left unchanged  passes of  patterns each It is interesting to note that the network e	ectively sees each individual sample only twice over the course of these  passes

Figure  shows all  misclassied test examples some of those examples are genuinely ambiguous but several are

Fig  Examples of distortions of ten training patterns 4−>6 3−>5 8−>2 2−>1 5−>3 4−>8 2−>8 3−>5 6−>5 7−>3 9−>4 8−>0 7−>8 5−>3 8−>7 0−>6 3−>7 2−>7 8−>3 9−>4 8−>2 5−>3 4−>8 3−>9 6−>0 9−>8 4−>9 6−>1 9−>4 9−>1 9−>4 2−>0 6−>1 3−>5 3−>2 9−>5 6−>0 6−>0 6−>0 6−>8 4−>6 7−>3 9−>4 4−>6 2−>7 9−>7 4−>3 9−>4 9−>4 9−>4 8−>7 4−>2 8−>4 3−>5 8−>4 6−>5 8−>5 3−>8 3−>8 9−>8 1−>5 9−>8 6−>3 0−>2 6−>5 9−>5 0−>7 1−>6 4−>9 2−>1 2−>8 8−>5 4−>9 7−>2 7−>2 6−>5 9−>7 6−>1 5−>6 5−>0 4−>9 2−>8 Fig  The  test patterns misclassied by LeNet Below each image is displayed the correct answers left and the network an swer right These errors are mostly caused either by genuinely ambiguous patterns or by digits written in a style that are under represented in the training set perfectly identiable by humans although they are writ ten in an underrepresented style This shows that further improvements are to be expected with more training data

C Comparison with Other Classiers

For the sake of comparison a variety of other trainable classiers was trained and tested on the same database An early subset of these results was presented in  The error rates on the test set for the various methods are shown in gure 

C Linear Classier and Pairwise Linear Classier

Possibly the simplest classier that one might consider is a linear classier Each input pixel value contributes to a weighted sum for each output unit The output unit with the highest sum including the contribution of a bias con

PROC OF THE IEEE NOVEMBER  

K−NN Euclidean [deslant] K−NN Euclidean 40 PCA + quadratic 1000 RBF + linear

SVM poly 4

RS−SVM poly 5 28x28−300−10 28x28−1000−10 28x28−300−100−10 28x28−500−150−10 

LeNet−4 / Local

LeNet−4 / K−NN

LeNet−5 −−−− 12.0 −−−−> −−−− 8.4 −−−−> −−−− 7.6 −−−−> 5

## 2.4
## 3.3
## 3.6
## 1.1
## 1.1
 1

## 0.8
## 4.7
## 3.6
## 1.6
## 4.5
## 3.8
## 3.05
## 2.5
## 2.95
## 2.45
 

## 1.7
## 1.1
## 1.1
## 1.1
## 0.95
## 0.8
## 0.7
 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 [dist] V−SVM poly 9 [dist] 28x28−300−10 [dist] 28x28−1000−10 [dist] 28x28−300−100−10 [dist] 28x28−500−150−10 [16x16] LeNet−1 [dist] LeNet−5 [dist] Boosted LeNet−4

LeNet−4 [16x16] Tangent Distance [deslant] 20x20−300−10

Linear [deslant] Linear

Pairwise

Fig  Error rate on the test set  for various classication methods deslant indicates that the classier was trained and tested on the deslanted version of the database dist indicates that the training set was augmented with articially distorted examples x indicates that the system used the x pixel images The uncertainty in the quoted error rates is about  stant indicates the class of the input character On the regular data the error rate is ! The network has  free parameters On the deslanted images the test error rate is ! The network has  free parameters The deciencies of the linear classier are well documented  and it is included here simply to form a basis of comparison for more sophisticated classiers Various combinations of sigmoid units linear units gradient descent learning and learning by directly solving linear systems gave similar re sults

A simple improvement of the basic linear classier was tested  The idea is to train each unit of a singlelayer network to separate each class from each other class In our case this layer comprises  units labeled    Unit ij is trained to produce  on patterns of class i  on patterns of class j and is not trained on other patterns The nal score for class i is the sum of the outputs all the units labeled ix minus the sum of the output of all the units labeled yi for all x and y The error rate on the regular test set was !

C Baseline Nearest Neighbor Classier

Another simple classier is a Knearest neighbor classi er with a Euclidean distance measure between input im ages This classier has the advantage that no training time and no brain on the part of the designer are required

However the memory requirement and recognition time are large the complete  twenty by twenty pixel training images about  Megabytes at one byte per pixel must be available at run time Much more compact representations could be devised with modest increase in error rate On the regular test set the error rate was ! On the deslanted data the error rate was ! with k   Naturally a realistic Euclidean distance nearestneighbor system would operate on feature vectors rather than directly on the pix els but since all of the other systems presented in this study operate directly on the pixels this result is useful for a baseline comparison

C Principal Component Analysis PCA and Polynomial

Classier

Following   a preprocessing stage was con structed which computes the pro jection of the input pat tern on the  principal components of the set of training vectors To compute the principal components the mean of each input component was rst computed and subtracted from the training vectors The covariance matrix of the re sulting vectors was then computed and diagonalized using

Singular Value Decomposition The dimensional feature vector was used as the input of a second degree polynomial classier This classier can be seen as a linear classier with  inputs preceded by a module that computes all

PROC OF THE IEEE NOVEMBER   products of pairs of input variables The error on the reg ular test set was !

C Radial Basis Function Network

Following  an RBF network was constructed The rst layer was composed of  Gaussian RBF units with x inputs and the second layer was a simple  inputs  outputs linear classier The RBF units were divided into  groups of  Each group of units was trained on all the training examples of one of the  classes using the adaptive Kmeans algorithm The second layer weights were computed using a regularized pseudoinverse method

The error rate on the regular test set was !

C OneHidden Layer Fully Connected Multilayer Neural

Network

Another classier that we tested was a fully connected multilayer neural network with two layers of weights one hidden layer trained with the version of backpropagation described in Appendix C Error on the regular test set was ! for a network with  hidden units and ! for a network with  hidden units Using articial distortions to generate more training data brought only marginal im provement ! for  hidden units and ! for  hidden units When deslanted images were used the test error jumped down to ! for a network with  hidden units

It remains somewhat of a mystery that networks with such a large number of free parameters manage to achieve reasonably low testing errors We conjecture that the dy namics of gradient descent learning in multilayer nets has a selfregularization e	ect Because the origin of weight space is a saddle point that is attractive in almost every direction the weights invariably shrink during the rst few epochs recent theoretical analysis seem to conrm this  Small weights cause the sigmoids to operate in the quasilinear region making the network essentially equivalent to a lowcapacity singlelayer network As the learning proceeds the weights grow which progressively increases the e	ective capacity of the network This seems to be an almost perfect if fortuitous implementation of

Vapniks Structural Risk Minimization principle  A better theoretical understanding of these phenomena and more empirical evidence are denitely needed

C TwoHidden Layer Fully Connected Multilayer Neural

Network

To see the e	ect of the architecture several twohidden layer multilayer neural networks were trained Theoreti cal results have shown that any function can be approxi mated by a onehidden layer neural network  However several authors have observed that twohidden layer archi tectures sometimes yield better performance in practical situations This phenomenon was also observed here The test error rate of a x network was ! a much better result than the onehidden layer network obtained using marginally more weights and connections

Increasing the network size to x yielded only marginally improved error rates ! Training with distorted patterns improved the performance some what ! error for the x network and ! for the x network

C A Small Convolutional Network LeNet

Convolutional Networks are an attempt to solve the dilemma between small networks that cannot learn the training set and large networks that seem over parameterized LeNet was an early embodiment of the

Convolutional Network architecture which is included here for comparison purposes The images were downsampled to x pixels and centered in the x input layer Al though about  multiplyadd steps are required to evaluate LeNet its convolutional nature keeps the num ber of free parameters to only about  The LeNet  architecture was developed using our own version of the USPS US Postal Service zip codes database and its size was tuned to match the available data  LeNet achieved ! test error The fact that a network with such a small number of parameters can attain such a good error rate is an indication that the architecture is appropriate for the task

C LeNet

Experiments with LeNet made it clear that a larger convolutional network was needed to make optimal use of the large size of the training set LeNet and later LeNet  were designed to address this problem LeNet is very similar to LeNet except for the details of the architec ture It contains  rstlevel feature maps followed by  subsampling maps connected in pairs to each rstlayer feature maps then  feature maps followed by  sub sampling map followed by a fully connected layer with  units followed by the output layer  units LeNet contains about  connections and has about  free parameters Test error was ! In a series of ex periments we replaced the last layer of LeNet with a

Euclidean Nearest Neighbor classier and with the local learning method of Bottou and Vapnik  in which a lo cal linear classier is retrained each time a new test pattern is shown Neither of those methods improved the raw error rate although they did improve the rejection performance

C Boosted LeNet

Following theoretical work by R Schapire  Drucker et al  developed the boosting method for combining multiple classiers Three LeNets are combined the rst one is trained the usual way the second one is trained on patterns that are ltered by the rst net so that the second machine sees a mix of patterns ! of which the rst net got right and ! of which it got wrong Finally the third net is trained on new patterns on which the rst and the second nets disagree During testing the outputs of the three nets are simply added Because the error rate of

LeNet is very low it was necessary to use the articially distorted images as with LeNet in order to get enough samples to train the second and third nets The test error

PROC OF THE IEEE NOVEMBER   rate was ! the best of any of our classiers At rst glance boosting appears to be three times more expensive as a single net In fact when the rst net produces a high condence answer the other nets are not called The average computational cost is about  times that of a single net

C Tangent Distance Classier TDC

The Tangent Distance classier TDC is a nearest neighbor method where the distance function is made in sensitive to small distortions and translations of the input image  If we consider an image as a point in a high dimensional pixel space where the dimensionality equals the number of pixels then an evolving distortion of a char acter traces out a curve in pixel space Taken together all these distortions dene a lowdimensional manifold in pixel space For small distortions in the vicinity of the original image this manifold can be approximated by a plane known as the tangent plane An excellent measure of closeness for character images is the distance between their tangent planes where the set of distortions used to generate the planes includes translations scaling skewing squeezing rotation and line thickness variations A test error rate of ! was achieved using x pixel images

Preltering techniques using simple Euclidean distance at multiple resolutions allowed to reduce the number of nec essary Tangent Distance calculations

C Support Vector Machine SVM

Polynomial classiers are wellstudied methods for gen erating complex decision surfaces Unfortunately they are impractical for highdimensional problems because the number of product terms is prohibitive The Support Vec tor technique is an extremely economical way of represent ing complex surfaces in highdimensional spaces including polynomials and many other types of surfaces 

A particularly interesting subset of decision surfaces is the ones that correspond to hyperplanes that are at a max imum distance from the convex hulls of the two classes in the highdimensional space of the product terms Boser

Guyon and Vapnik  realized that any polynomial of degree k in this maximum margin set can be computed by rst computing the dot product of the input image with a subset of the training samples called the support vec tors  elevating the result to the kth power and linearly combining the numbers thereby obtained Finding the sup port vectors and the coecients amounts to solving a high dimensional quadratic minimization problem with linear inequality constraints For the sake of comparison we in clude here the results obtained by Burges and Sch"olkopf reported in  With a regular SVM their error rate on the regular test set was ! Cortes and Vapnik had reported an error rate of ! with SVM on the same data using a slightly di	erent technique The computa tional cost of this technique is very high about  million multiplyadds per recognition Using Sch"olkopf s Virtual

Support Vectors technique VSVM ! error was at tained More recently Sch"olkopf personal communication

## 8.1
## 1.9
## 1.8
## 3.2
## 3.7
## 1.8
## 1.4
## 1.6
## 0.5
 [deslant] K−NN Euclidean [16x16] Tangent Distance

SVM poly 4

LeNet−4

LeNet−4 / Local

LeNet−4 / K−NN [dist] Boosted LeNet−4 0123456789 [deslant] 20x20−300−10 [16x16] LeNet−1

Fig  Rejection Performance	 percentage of test patterns that must be rejected to achieve  error for some of the systems 4 36 −−−− 24,000 −−−−> 39 794 −−−− 20,000 −−−−> −−−− 14,000 −−−−> 650 −−−− 28,000 −−−−> 123 795 267 469 100 260 −−−− 20,000 −−−−> −−−− 10,000 −−−−> 401 460 [deslant] K−NN Euclidean 1000 RBF [16x16] Tangent Distance

SVM poly 4

RS−SVM poly 5 [dist] V−SVM poly 9 [deslant] 20x20−300−10 28x28−1000−10 28x28−300−100−10 28x28−500−150−10 [16x16] LeNet−1

LeNet−4

LeNet−4 / Local

LeNet−4 / K−NN

LeNet−5

Boosted LeNet−4 0 300 600 900

Linear

Pairwise 40 PCA+quadratic

Fig  Number of multiplyaccumulate operations for the recogni tion of a single character starting with a sizenormalized image has reached ! using a modied version of the VSVM

Unfortunately VSVM is extremely expensive about twice as much as regular SVM To alleviate this problem Burges has proposed the Reduced Set Support Vector technique RSSVM which attained ! on the regular test set  with a computational cost of only  multiplyadds per recognition ie only about ! more expensive than

LeNet

D Discussion

A summary of the performance of the classiers is shown in Figures  to  Figure  shows the raw error rate of the classiers on the  example test set Boosted LeNet performed best achieving a score of ! closely followed by LeNet at !

Figure  shows the number of patterns in the test set that must be rejected to attain a ! error for some of the methods Patterns are rejected when the value of cor responding output is smaller than a predened threshold

In many applications rejection performance is more signif icant than raw error rate The score used to decide upon the rejection of a pattern was the di	erence between the scores of the top two classes Again Boosted LeNet has the best performance The enhanced versions of LeNet did better than the original LeNet even though the raw

PROC OF THE IEEE NOVEMBER    4 35 −−− 24,000 −−−> 40 794 −−− 25,000 −−−> −−−− 14,000 −−−−> 650 −−−− 28,000 −−−−> 123 795 267 469 3 17 −−− 24,000 −−−> −−− 24,000 −−−> 60 51 1000 RBF [16x16] Tangent Distance

SVM poly 4

RS−SVM poly 5 [dist] V−SVM poly 5 [deslant] 20x20−300−10 28x28−1000−10 28x28−300−100−10 28x28−500−150−10 [16x16] LeNet 1

LeNet 4

LeNet 4 / Local

LeNet 4 / K−NN

LeNet 5

Boosted LeNet 4 0 300 600 900

Linear

Pairwise 40 PCA+quadratic [deslant] K−NN Euclidean

Fig  Memory requirements measured in number of variables for each of the methods Most of the methods only require one byte per variable for adequate performance accuracies were identical

Figure  shows the number of multiplyaccumulate op erations necessary for the recognition of a single size normalized image for each method Expectedly neural networks are much less demanding than memorybased methods Convolutional Neural Networks are particu larly well suited to hardware implementations because of their regular structure and their low memory requirements for the weights Single chip mixed analogdigital imple mentations of LeNets predecessors have been shown to operate at speeds in excess of  characters per sec ond  However the rapid progress of mainstream com puter technology renders those exotic technologies quickly obsolete Coste	ective implementations of memorybased techniques are more elusive due to their enormous memory requirements and computational requirements

Training time was also measured Knearest neighbors and TDC have essentially zero training time While the singlelayer net the pairwise net and PCAquadratic net could be trained in less than an hour the multilayer net training times were expectedly much longer but only re quired  to  passes through the training set This amounts to  to  days of CPU to train LeNet on a Sil icon Graphics Origin  server using a single MHz

R processor It is important to note that while the training time is somewhat relevant to the designer it is of little interest to the nal user of the system Given the choice between an existing technique and a new technique that brings marginal accuracy improvements at the price of considerable training time any nal user would chose the latter

Figure  shows the memory requirements and therefore the number of free parameters of the various classiers measured in terms of the number of variables that need to be stored Most methods require only about one byte per variable for adequate performance However Nearest

Neighbor methods may get by with  bits per pixel for stor ing the template images Not surprisingly neural networks require much less memory than memorybased methods

The Overall performance depends on many factors in cluding accuracy running time and memory requirements

As computer technology improves largercapacity recog nizers become feasible Larger recognizers in turn require larger training sets LeNet was appropriate to the avail able technology in  just as LeNet is appropriate now In  a recognizer as complex as LeNet would have re quired several weeks training and more data than was available and was therefore not even considered For quite a long time LeNet was considered the state of the art

The local learning classier the optimal margin classier and the tangent distance classier were developed to im prove upon LeNet # and they succeeded at that How ever they in turn motivated a search for improved neural network architectures This search was guided in part by estimates of the capacity of various learning machines de rived from measurements of the training and test error as a function of the number of training examples We dis covered that more capacity was needed Through a series of experiments in architecture combined with an analy sis of the characteristics of recognition errors LeNet and

LeNet were crafted

We nd that boosting gives a substantial improvement in accuracy with a relatively modest penalty in memory and computing expense Also distortion models can be used to increase the e	ective size of a data set without actually requiring to collect more data

The Support Vector Machine has excellent accuracy which is most remarkable because unlike the other high performance classiers it does not include a priori knowl edge about the problem In fact this classier would do just as well if the image pixels were permuted with a xed mapping and lost their pictorial structure However reach ing levels of performance comparable to the Convolutional

Neural Networks can only be done at considerable expense in memory and computational requirements The reduced set SVM requirements are within a factor of two of the

Convolutional Networks and the error rate is very close

Improvements of those results are expected as the tech nique is relatively new

When plenty of data is available many methods can at tain respectable accuracy The neuralnet methods run much faster and require much less space than memory based techniques The neural nets advantage will become more striking as training databases continue to increase in size

E Invariance and Noise Resistance

Convolutional networks are particularly well suited for recognizing or rejecting shapes with widely varying size position and orientation such as the ones typically pro duced by heuristic segmenters in realworld string recogni tion systems

In an experiment like the one described above the im portance of noise resistance and distortion invariance is not obvious The situation in most real applications is

PROC OF THE IEEE NOVEMBER   quite di	erent Characters must generally be segmented out of their context prior to recognition Segmentation algorithms are rarely perfect and often leave extraneous marks in character images noise underlines neighboring characters or sometimes cut characters too much and pro duce incomplete characters Those images cannot be re liably sizenormalized and centered Normalizing incom plete characters can be very dangerous For example an enlarged stray mark can look like a genuine  Therefore many systems have resorted to normalizing the images at the level of elds or words In our case the upper and lower proles of entire elds amounts in a check are detected and used to normalize the image to a xed height While this guarantees that stray marks will not be blown up into characterlooking images this also creates wide variations of the size and vertical position of characters after segmen tation Therefore it is preferable to use a recognizer that is robust to such variations Figure  shows several exam ples of distorted characters that are correctly recognized by

LeNet It is estimated that accurate recognition occurs for scale variations up to about a factor of  vertical shift variations of plus or minus about half the height of the character and rotations up to plus or minus  degrees

While fully invariant recognition of complex shapes is still an elusive goal it seems that Convolutional Networks o	er a partial answer to the problem of invariance or robustness with respect to geometrical distortions

Figure  includes examples of the robustness of LeNet  under extremely noisy conditions Processing those images would pose unsurmountable problems of segmen tation and feature extraction to many methods but

LeNet seems able to robustly extract salient features from these cluttered images The training set used for the network shown here was the MNIST training set with salt and pepper noise added Each pixel was ran domly inverted with probability  More examples of LeNet in action are available on the Internet at httpwwwresearchattcomyannocr

IV MultiModule Systems and Graph

Transformer Networks

The classical backpropagation algorithm as described and used in the previous sections is a simple form of

GradientBased Learning However it is clear that the gradient backpropagation algorithm given by Equation  describes a more general situation than simple multilayer feedforward networks composed of alternated linear trans formations and sigmoidal functions In principle deriva tives can be backpropagated through any arrangement of functional modules as long as we can compute the prod uct of the Jacobians of those modules by any vector Why would we want to train systems composed of multiple het erogeneous modules The answer is that large and complex trainable systems need to be built out of simple specialized modules The simplest example is LeNet which mixes convolutional layers subsampling layers fullyconnected layers and RBF layers Another less trivial example de scribed in the next two sections is a system for recognizing

F0(X0)

E

W1

D

X1

F1(X0,X1,W1)

F2(X2,W2)

X2

X3

X4

X5 F3(X3,X4) 

Function

Zt Input

Desired Output

Loss

W2

Fig  A trainable system composed of heterogeneous modules words that can be trained to simultaneously segment and recognize words without ever being given the correct seg mentation

Figure  shows an example of a trainable multimodular system A multimodule system is dened by the function implemented by each of the modules and by the graph of interconnection of the modules to each other The graph implicitly denes a partial order according to which the modules must be updated in the forward pass For exam ple in Figure  module  is rst updated then modules  and  are updated possibly in parallel and nally mod ule  Modules may or may not have trainable parameters

Loss functions which measure the performance of the sys tem are implemented as module  In the simplest case the loss function module receives an external input that carries the desired output In this framework there is no qualitative di	erence between trainable parameters WW in the gure external inputs and outputs ZDE and intermediate state variablesXXX	X X

A An ObjectOriented Approach

Ob jectOriented programming o	ers a particularly con venient way of implementing multimodule systems Each module is an instance of a class Module classes have a for ward propagation method or member function called fprop whose arguments are the inputs and outputs of the module For example computing the output of module  in Figure  can be done by calling the method fprop on module  with the arguments X	X X Complex mod ules can be constructed from simpler modules by simply dening a new class whose slots will contain the member modules and the intermediate state variables between those modules The fprop method for the class simply calls the fprop methods of the member modules with the appro priate intermediate state variables or external input and outputs as arguments Although the algorithms are eas ily generalizable to any network of such modules including those whose inuence graph has cycles we will limit the dis cussion to the case of directed acyclic graphs feedforward networks

Computing derivatives in a multimodule system is just as simple A backward propagation method called bprop for each module class can be dened for that pur pose The bprop method of a module takes the same ar

PROC OF THE IEEE NOVEMBER    3 4 4 4 4 3 4 83

C1 S2 C3 S4 C5

F6

Output

Fig  Examples of unusual distorted and noisy characters correctly recognized by LeNet The greylevel of the output label represents the penalty lighter for higher penalties guments as the fprop method All the derivatives in the system can be computed by calling the bprop method on all the modules in reverse order compared to the forward prop agation phase The state variables are assumed to contain slots for storing the gradients computed during the back ward pass in addition to storage for the states computed in the forward pass The backward pass e	ectively computes the partial derivatives of the loss E with respect to all the state variables and all the parameters in the system There is an interesting duality property between the forward and backward functions of certain modules For example a sum of several variables in the forward direction is trans formed into a simple fanout replication in the backward direction Conversely a fanout in the forward direction is transformed into a sum in the backward direction The software environment used to obtain the results described in this paper called SN uses the above concepts It is based on a homegrown ob jectoriented dialect of Lisp with a compiler to C

The fact that derivatives can be computed by propaga tion in the reverse graph is easy to understand intuitively The best way to justify it theoretically is through the use of

Lagrange functions   The same formalism can be used to extend the procedures to networks with recurrent connections

B Special Modules

Neural networks and many other standard pattern recog nition techniques can be formulated in terms of multi modular systems trained with GradientBased Learning

Commonly used modules include matrix multiplications and sigmoidal modules the combination of which can be used to build conventional neural networks Other mod ules include convolutional layers subsampling layers RBF layers and softmax layers  Loss functions are also represented as modules whose single output produces the value of the loss Commonly used modules have simple bprop methods In general the bprop method of a func tion F is a multiplication by the Jacobian of F  Here are a few commonly used examples The bprop method of a fanout a Y connection is a sum and vice versa The bprop method of a multiplication by a coecient is a mul tiplication by the same coecient The bprop method of a multiplication by a matrix is a multiplication by the trans pose of that matrix The bprop method of an addition with a constant is the identity

PROC OF THE IEEE NOVEMBER  

Layer

Layer

Layer

Layer (a)

Graph

Transformer

Graph

Transformer (b)

Fig  Traditional neural networks and multimodule systems com municate xedsize vectors between layer MultiLayer Graph

Transformer Networks are composed of trainable modules that operate on and produce graphs whose arcs carry numerical in formation

Interestingly certain nondi	erentiable modules can be inserted in a multimodule system without adverse e	ect

An interesting example of that is the multiplexer module

It has two or more regular inputs one switching input and one output The module selects one of its inputs de pending upon the discrete value of the switching input and copies it on its output While this module is not dif ferentiable with respect to the switching input it is di	er entiable with respect to the regular inputs Therefore the overall function of a system that includes such modules will be di	erentiable with respect to its parameters as long as the switching input does not depend upon the parameters

For example the switching input can be an external input

Another interesting case is the min module This mod ule has two or more inputs and one output The output of the module is the minimum of the inputs The func tion of this module is di	erentiable everywhere except on the switching surface which is a set of measure zero In terestingly this function is continuous and reasonably reg ular and that is sucient to ensure the convergence of a

GradientBased Learning algorithm

The ob jectoriented implementation of the multimodule idea can easily be extended to include a bbprop method that propagates GaussNewton approximations of the sec ond derivatives This leads to a direct generalization for modular systems of the secondderivative backpropagation

Equation  given in the Appendix

The multiplexer module is a special case of a much more general situation described at length in Section VIII where the architecture of the system changes dynamically with the input data Multiplexer modules can be used to dynamically rewire or recongure the architecture of the system for each new input pattern

C Graph Transformer Networks

Multimodule systems are a very exible tool for build ing large trainable system However the descriptions in the previous sections implicitly assumed that the set of parameters and the state information communicated be tween the modules are all xedsize vectors The limited exibility of xedsize vectors for data representation is a serious deciency for many applications notably for tasks that deal with variable length inputs eg continuous speech recognition and handwritten word recognition or for tasks that require encoding relationships between ob jects or fea tures whose number and nature can vary invariant per ception scene analysis recognition of composite ob jects

An important special case is the recognition of strings of characters or words

More generally xedsize vectors lack exibility for tasks in which the state must encode probability distributions over sequences of vectors or symbols as is the case in lin guistic processing Such distributions over sequences are best represented by stochastic grammars or in the more general case directed graphs in which each arc contains a vector stochastic grammars are special cases in which the vector contains probabilities and symbolic information

Each path in the graph represents a di	erent sequence of vectors Distributions over sequences can be represented by interpreting elements of the data associated with each arc as parameters of a probability distribution or simply as a penalty Distributions over sequences are particularly handy for modeling linguistic knowledge in speech or hand writing recognition systems each sequence ie each path in the graph represents an alternative interpretation of the input Successive processing modules progressively rene the interpretation For example a speech recognition sys tem might start with a single sequence of acoustic vectors transform it into a lattice of phonemes distribution over phoneme sequences then into a lattice of words distribu tion over word sequences and then into a single sequence of words representing the best interpretation

In our work on building largescale handwriting recog nition systems we have found that these systems could much more easily and quickly be developed and designed by viewing the system as a networks of modules that take one or several graphs as input and produce graphs as out put Such modules are called Graph Transformers and the complete systems are called Graph Transformer Networks or GTN Modules in a GTN communicate their states and gradients in the form of directed graphs whose arcs carry numerical information scalars or vectors 

From the statistical point of view the xedsize state vectors of conventional networks can be seen as represent ing the means of distributions in state space In variable size networks such as the SpaceDisplacement Neural Net works described in section VII the states are variable length sequences of xed size vectors They can be seen as representing the mean of a probability distribution over variablelength sequences of xedsize vectors In GTNs the states are represented as graphs which can be seen as representing mixtures of probability distributions over structured collections possibly sequences of vectors Fig ure 

One of the main points of the next several sections is to show that GradientBased Learning procedures are not limited to networks of simple modules that communicate

PROC OF THE IEEE NOVEMBER   through xedsize vectors but can be generalized to GTNs

Gradient backpropagation through a Graph Transformer takes gradients with respect to the numerical informa tion in the output graph and computes gradients with re spect to the numerical information attached to the input graphs and with respect to the modules internal param eters GradientBased Learning can be applied as long as di	erentiable functions are used to produce the numerical data in the output graph from the numerical data in the input graph and the functions parameters

The second point of the next several sections is to show that the functions implemented by many of the modules used in typical document processing systems and other image recognition systems though commonly thought to be combinatorial in nature are indeed di	erentiable with respect to their internal parameters as well as with respect to their inputs and are therefore usable as part of a globally trainable system

In most of the following we will purposely avoid making references to probability theory All the quantities manip ulated are viewed as penalties or costs which if necessary can be transformed into probabilities by taking exponen tials and normalizing

V Multiple Object Recognition Heuristic

OverSegmentation

One of the most dicult problems of handwriting recog nition is to recognize not just isolated characters but strings of characters such as zip codes check amounts or words Since most recognizers can only deal with one character at a time we must rst segment the string into individual character images However it is almost impos sible to devise image analysis techniques that will infallibly segment naturally written sequences of characters into well formed characters

The recent history of automatic speech recognition    is here to remind us that training a recognizer by opti mizing a global criterion at the word or sentence level is much preferable to merely training it on handsegmented phonemes or other units Several recent works have shown that the same is true for handwriting recognition  op timizing a wordlevel criterion is preferable to solely train ing a recognizer on presegmented characters because the recognizer can learn not only to recognize individual char acters but also to reject missegmented characters thereby minimizing the overall word error

This section and the next describe in detail a simple ex ample of GTN to address the problem of reading strings of characters such as words or check amounts The method avoids the expensive and unreliable task of handtruthing the result of the segmentation often required in more tra ditional systems trained on individually labeled character images

A Segmentation Graph

A now classical method for word segmentation and recog nition is called Heuristic OverSegmentation   Its main advantages over other approaches to segmentation are

Fig  Building a segmentation graph with Heuristic Over Segmentation that it avoids making hard decisions about the segmenta tion by taking a large number of di	erent segmentations into consideration The idea is to use heuristic image pro cessing techniques to nd candidate cuts of the word or string and then to use the recognizer to score the alter native segmentations thereby generated The process is depicted in Figure  First a number of candidate cuts are generated Good candidate locations for cuts can be found by locating minima in the vertical pro jection prole or minima of the distance between the upper and lower contours of the word Better segmentation heuristics are described in section X The cut generation heuristic is de signed so as to generate more cuts than necessary in the hope that the correct set of cuts will be included Once the cuts have been generated alternative segmentations are best represented by a graph called the segmentation graph The segmentation graph is a Directed Acyclic Graph DAG with a start node and an end node Each internal node is associated with a candidate cut produced by the segmen tation algorithm Each arc between a source node and a destination node is associated with an image that contains all the ink between the cut associated with the source node and the cut associated with the destination node An arc is created between two nodes if the segmentor decided that the ink between the corresponding cuts could form a can didate character Typically each individual piece of ink would be associated with an arc Pairs of successive pieces of ink would also be included unless they are separated by a wide gap which is a clear indication that they belong to di	erent characters Each complete path through the graph contains each piece of ink once and only once Each path corresponds to a di	erent way of associating pieces of ink together so as to form characters

B Recognition Transformer and Viterbi Transformer

A simple GTN to recognize character strings is shown in Figure  It is composed of two graph transformers called the recognition transformer Trec and the Viterbi transformer Tvit  The goal of the recognition transformer is to generate a graph called the interpretation graph or recognition graph Gint  that contains all the possible inter pretations for all the possible segmentations of the input

Each path in Gint represents one possible interpretation of one particular segmentation of the input The role of the

Viterbi transformer is to extract the best interpretation from the interpretation graph

The recognition transformer Trec takes the segmentation graph Gseg as input and applies the recognizer for single characters to the images associated with each of the arcs

PROC OF THE IEEE NOVEMBER  

NN NN NN NN NN NN 32 34 14 234 34 1 4 3 2 4

Interpretation

Graph

Segmentation

Graph

Σ

Viterbi Penalty

Viterbi

Path

Gseg

T rec

T G vit int

Gvit

Viterbi

Transformer

Recognition

Transformer

Fig  Recognizing a character string with a GTN For readability only the arcs with low penalties are shown in the segmentation graph The interpretation graph Gint has almost the same structure as the segmentation graph except that each arc is replaced by a set of arcs from and to the same node In this set of arcs there is one arc for each possible class for the image associated with the cor responding arc in Gseg  As shown in Figure  to each arc is attached a class label and the penalty that the im age belongs to this class as produced by the recognizer If the segmentor has computed penalties for the candidate segments these penalties are combined with the penalties computed by the character recognizer to obtain the penal ties on the arcs of the interpretation graph Although com bining penalties of di	erent nature seems highly heuristic the GTN training procedure will tune the penalties and take advantage of this combination anyway Each path in the interpretation graph corresponds to a possible inter pretation of the input word The penalty of a particular interpretation for a particular segmentation is given by the sum of the arc penalties along the corresponding path in the interpretation graph Computing the penalty of an in terpretation independently of the segmentation requires to combine the penalties of all the paths with that interpre tation An appropriate rule for combining the penalties of parallel paths is given in section VIC

The Viterbi transformer produces a graph Gvit with a single path This path is the path of least cumulated penalty in the Interpretation graph The result of the recognition can be produced by reading o	 the labels of the arcs along the graph Gvit extracted by the Viterbi transformer The Viterbi transformer owes its name to the 3 0.1

## 0.5
 penalty given by the segmentor "0" "1"

## 6.7
## 10.3
## 0.3
## 12.5
 "0" "1" "2" "3"

## 7.9
## 11.2
## 6.8
## 0.2
## 13.5
## 8.4
W character recognizer penalty for each class class label

PIECE OF THE

SEGMENTATION

GRAPH candidate segment image

PIECE OF THE

INTERPRETATION

GRAPH

Character

Recognizer Character

Recognizer "8" "9" "8" "9" 8

Fig  The recognition transformer renes each arc of the segmen tation arc into a set of arcs in the interpretation graph one per character class with attached penalties and labels famous Viterbi algorithm  an application of the prin ciple of dynamic programming to nd the shortest path in a graph eciently Let ci be the penalty associated to arc i with source node si  and destination node di note that there can be multiple arcs between two nodes In the interpretation graph arcs also have a label li  The

Viterbi algorithm proceeds as follows Each node n is as sociated with a cumulated Viterbi penalty vn Those cu mulated penalties are computed in any order that satises the partial order dened by the interpretation graph which is directed and acyclic The start node is initialized with the cumulated penalty vstart   The other nodes cu mulated penalties vn are computed recursively from the v values of their parent nodes through the upstream arcs

Un  farc i with destination di  ng vn  min iUnci  vsi  

Furthermore the value of i for each node n which minimizes the right hand side is noted mn the minimizing entering arc When the end node is reached we obtain in vend the total penalty of the path with the smallest total penalty We call this penalty the Viterbi penalty and this sequence of arcs and nodes the Viterbi path To obtain the Viterbi path with nodes n nT and arcs i iT  we trace back these nodes and arcs as follows starting with nT  the end node and recursively using the minimizing entering arc it  mnt  and nt  sit until the start node is reached

The label sequence can then be read o	 the arcs of the

Viterbi path

PROC OF THE IEEE NOVEMBER  

VI Global Training for Graph Transformer

Networks

The previous section describes the process of recognizing a string using Heuristic OverSegmentation assuming that the recognizer is trained so as to give low penalties for the correct class label of correctly segmented characters high penalties for erroneous categories of correctly segmented characters and high penalties for all categories for badly formed characters This section explains how to train the system at the string level to do the above without requiring manual labeling of character segments This training will be performed with a GTN whose architecture is slightly di	erent from the recognition architecture described in the previous section

In many applications there is enough a priori knowl edge about what is expected from each of the modules in order to train them separately For example with Heuris tic OverSegmentation one could individually label single character images and train a character recognizer on them but it might be dicult to obtain an appropriate set of noncharacter images to train the model to reject wrongly segmented candidates Although separate training is sim ple it requires additional supervision information that is often lacking or incomplete the correct segmentation and the labels of incorrect candidate segments Furthermore it can be shown that separate training is suboptimal 

The following section describes three di	erent gradient based methods for training GTNbased handwriting recog nizers at the string level Viterbi training discriminative

Viterbi training forward training and discriminative for ward training The last one is a generalization to graph based systems of the MAP criterion introduced in Sec tion IIC Discriminative forward training is somewhat similar to the socalled Maximum Mutual Information cri terion used to train HMM in speech recognition However our rationale di	ers from the classical one We make no recourse to a probabilistic interpretation but show that within the GradientBased Learning approach discrimina tive training is a simple instance of the pervasive principle of error correcting learning

Training methods for graphbased sequence recognition systems such as HMMs have been extensively studied in the context of speech recognition  Those methods re quire that the system be based on probabilistic generative models of the data which provide normalized likelihoods over the space of possible input sequences Popular HMM learning methods such as the the BaumWelsh algorithm rely on this normalization The normalization cannot be preserved when nongenerative models such as neural net works are integrated into the system Other techniques such as discriminative training methods must be used in this case Several authors have proposed such methods to train neural networkHMM speech recognizers at the word or sentence level           

Other globally trainable sequence recognition systems avoid the diculties of statistical modeling by not resorting to graphbased techniques The best example is Recurrent

Recognition

Transformer

Interpretation Graph

Desired Sequence Path Selector

Best Constrained Path

Σ

Constrained Viterbi Penalty

Constrained

Interpretation Graph Gc Gcvit

Ccvit

Gint

Viterbi Transformer

Fig  Viterbi Training GTN Architecture for a character string recognizer based on Heuristic OverSegmentation

Neural Networks RNN Unfortunately despite early en thusiasm the training of RNNs with gradientbased tech niques has proved very dicult in practice 

The GTN techniques presented below simplify and gen eralize the global training methods developed for speech recognition

A Viterbi Training

During recognition we select the path in the Interpre tation Graph that has the lowest penalty with the Viterbi algorithm Ideally we would like this path of lowest penalty to be associated with the correct label sequence as often as possible An obvious loss function to minimize is therefore the average over the training set of the penalty of the path associated with the correct label sequence that has the low est penalty The goal of training will be to nd the set of recognizer parameters the weights if the recognizer is a neural network that minimize the average penalty of this correct lowest penalty path The gradient of this loss function can be computed by backpropagation through the GTN architecture shown in gure  This training architecture is almost identical to the recognition archi tecture described in the previous section except that an extra graph transformer called a path selector is inserted between the Interpretation Graph and the Viterbi Trans former This transformer takes the interpretation graph and the desired label sequence as input It extracts from the interpretation graph those paths that contain the cor rect desired label sequence Its output graph Gc is called the constrained interpretation graph also known as forced alignment in the HMM literature and contains all the paths that correspond to the correct label sequence The constrained interpretation graph is then sent to the Viterbi transformer which produces a graph Gcvit with a single path This path is the correct path with the lowest penalty Finally a path scorer transformer takes Gcvit and simply computes its cumulated penalty Ccvit by adding up the penalties along the path The output of this GTN is

PROC OF THE IEEE NOVEMBER   the loss function for the current pattern

Evit  Ccvit 

The only label information that is required by the above system is the sequence of desired character labels No knowledge of the correct segmentation is required on the part of the supervisor since it chooses among the segmen tations in the interpretation graph the one that yields the lowest penalty The process of backpropagating gradients through the

Viterbi training GTN is now described As explained in section IV the gradients must be propagated backwards through all modules of the GTN in order to compute gra dients in preceding modules and thereafter tune their pa rameters Backpropagating gradients through the path scorer is quite straightforward The partial derivatives of the loss function with respect to the individual penalties on the constrained Viterbi path Gcvit are equal to  since the loss function is simply the sum of those penalties Back propagating through the Viterbi Transformer is equally simple The partial derivatives of Evit with respect to the penalties on the arcs of the constrained graph Gc are  for those arcs that appear in the constrained Viterbi path

Gcvit and  for those that do not Why is it legitimate to backpropagate through an essentially discrete function such as the Viterbi Transformer The answer is that the

Viterbi Transformer is nothing more than a collection of min functions and adders put together It was shown in

Section IV that gradients can be backpropagated through min functions without adverse e	ects Backpropagation through the path selector transformer is similar to back propagation through the Viterbi transformer Arcs in Gint that appear in Gc have the same gradient as the corre sponding arc in Gc ie  or  depending on whether the arc appear in Gcvit The other arcs ie those that do not have an alter ego in Gc because they do not contain the right label have a gradient of  During the forward propagation through the recognition transformer one in stance of the recognizer for single character was created for each arc in the segmentation graph The state of rec ognizer instances was stored Since each arc penalty in

Gint is produced by an individual output of a recognizer instance we now have a gradient  or  for each out put of each instance of the recognizer Recognizer outputs that have a non zero gradient are part of the correct an swer and will therefore have their value pushed down The gradients present on the recognizer outputs can be back propagated through each recognizer instance For each rec ognizer instance we obtain a vector of partial derivatives of the loss function with respect to the recognizer instance parameters All the recognizer instances share the same pa rameter vector since they are merely clones of each other therefore the full gradient of the loss function with respect to the recognizers parameter vector is simply the sum of the gradient vectors produced by each recognizer instance

Viterbi training though formulated di	erently is often use in HMMbased speech recognition systems  Similar al gorithms have been applied to speech recognition systems that integrate neural networks with time alignment     or hybrid neuralnetworkHMM systems    

While it seems simple and satisfying this training ar chitecture has a aw that can potentially be fatal The problem was already mentioned in Section IIC If the recognizer is a simple neural network with sigmoid out put units the minimum of the loss function is attained not when the recognizer always gives the right answer but when it ignores the input and sets its output to a constant vector with small values for all the components This is known as the col lapse problem The collapse only occurs if the recognizer outputs can simultaneously take their min imum value If on the other hand the recognizers out put layer contains RBF units with xed parameters then there is no such trivial solution This is due to the fact that a set of RBF with xed distinct parameter vectors cannot simultaneously take their minimum value In this case the complete collapse described above does not occur

However this does not totally prevent the occurrence of a milder collapse because the loss function still has a at spot for a trivial solution with constant recognizer out put This at spot is a saddle point but it is attractive in almost all directions and is very dicult to get out of using gradientbased minimization procedures If the parameters of the RBFs are allowed to adapt then the collapse prob lems reappears because the RBF centers can all converge to a single vector and the underlying neural network can learn to produce that vector and ignore the input A dif ferent kind of collapse occurs if the width of the RBFs are also allowed to adapt The collapse only occurs if a train able module such as a neural network feeds the RBFs The collapse does not occur in HMMbased speech recognition systems because they are generative systems that produce normalized likelihoods for the input data more on this later Another way to avoid the collapse is to train the whole system with respect to a discriminative training cri terion such as maximizing the conditional probability of the correct interpretations correct sequence of class labels given the input image

Another problem with Viterbi training is that the penalty of the answer cannot be used reliably as a mea sure of condence because it does not take lowpenalty or highscoring competing answers into account

B Discriminative Viterbi Training

A modication of the training criterion can circumvent the collapse problem described above and at the same time produce more reliable condence values The idea is to not only minimize the cumulated penalty of the lowest penalty path with the correct interpretation but also to somehow increase the penalty of competing and possibly incorrect paths that have a dangerously low penalty This type of criterion is called discriminative because it plays the good answers against the bad ones Discriminative training pro cedures can be seen as attempting to build appropriate separating surfaces between classes rather than to model individual classes independently of each other For exam

PROC OF THE IEEE NOVEMBER  

Path Selector

Viterbi Tansformer

Gcvit

Gvit

Viterbi Transformer

Σ

Segmentation

Graph

Gseg

Recognition

Transfomer

T rec

Interpretation

Graph

Gint + + [0.6](−1) [0.7](+1) 1 [0.1](−1) 4 [2.4](0) 4 [0.4](−1) 2 [1.3](0) 3 [0.1](0) 5 [2.3](0) 3 [3.4](0) 4 [4.4](0) 4 [0.6](+1) 9 [1.2](0) [0.1](+1) + − "34"

Desired

Answer

W

Neural Net

Weights

NN NN NN NN NN 4 4 1 (−1) (+1) (−1) 4 [0.6](+1) 3 [0.1](+1) 4 [0.4](−1) 3 [0.1](−1) 1 [0.1](−1)

Gc 3 [3.4](0) 4 [0.6](+1) 4 [2.4](0) 3 [0.1](+1)

Loss Function

Segmenter

Fig  Discriminative Viterbi Training GTN Architecture for a character string recognizer based on Heuristic OverSegmentation Quantities in square brackets are penalties computed during the forward propagation Quantities in parentheses are partial derivatives computed during the backward propagation

PROC OF THE IEEE NOVEMBER   ple modeling the conditional distribution of the classes given the input image is more discriminative focussing more on the classication surface than having a separate generative model of the input data associated to each class which with class priors yields the whole joint distribu tion of classes and inputs This is because the conditional approach does not need to assume a particular form for the distribution of the input data

One example of discriminative criterion is the di	erence between the penalty of the Viterbi path in the constrained graph and the penalty of the Viterbi path in the uncon strained interpretation graph ie the di	erence between the penalty of the best correct path and the penalty of the best path correct or incorrect The corresponding

GTN training architecture is shown in gure  The left side of the diagram is identical to the GTN used for non discriminative Viterbi training This loss function reduces the risk of collapse because it forces the recognizer to in creases the penalty of wrongly recognized ob jects Dis criminative training can also be seen as another example of error correction procedure which tends to minimize the di	erence between the desired output computed in the left half of the GTN in gure  and the actual output com puted in the right half of gure 

Let the discriminative Viterbi loss function be denoted

Edvit and let us call Ccvit the penalty of the Viterbi path in the constrained graph and Cvit the penalty of the Viterbi path in the unconstrained interpretation graph

Edvit  Ccvit  Cvit 

Edvit is always positive since the constrained graph is a subset of the paths in the interpretation graph and the

Viterbi algorithm selects the path with the lowest total penalty In the ideal case the two paths Ccvit and Cvit coincide and Edvit is zero

Backpropagating gradients through the discriminative

Viterbi GTN adds some negative training to the pre viously described nondiscriminative training Figure  shows how the gradients are backpropagated The left half is identical to the nondiscriminative Viterbi training

GTN therefore the backpropagation is identical The gra dients backpropagated through the right half of the GTN are multiplied by  since Cvit contributes to the loss with a negative sign Otherwise the process is similar to the left half The gradients on arcs of Gint get positive contribu tions from the left half and negative contributions from the right half The two contributions must be added since the penalties on Gint arcs are sent to the two halves through a Y connection in the forward pass Arcs in Gint that appear neither in Gvit nor in Gcvit have a gradient of zero

They do not contribute to the cost Arcs that appear in both Gvit and Gcvit also have zero gradient The  contri bution from the right half cancels the the  contribution from the left half In other words when an arc is rightfully part of the answer there is no gradient If an arc appears in Gcvit but not in Gvit the gradient is  The arc should have had a lower penalty to make it to Gvit If an arc is in Gvit but not in Gcvit the gradient is  The arc had a low penalty but should have had a higher penalty since it is not part of the desired answer

Variations of this technique have been used for the speech recognition Driancourt and Bottou  used a version of it where the loss function is saturated to a xed value

This can be seen as a generalization of the Learning Vector

Quantization  LVQ loss function  Other variations of this method use not only the Viterbi path but the K best paths The Discriminative Viterbi algorithm does not have the aws of the nondiscriminative version but there are problems nonetheless The main problem is that the criterion does not build a margin between the classes The gradient is zero as soon as the penalty of the constrained

Viterbi path is equal to that of the Viterbi path It would be desirable to push up the penalties of the wrong paths when they are dangerously close to the good one The following section presents a solution to this problem

C Forward Scoring and Forward Training

While the penalty of the Viterbi path is perfectly appro priate for the purpose of recognition it gives only a partial picture of the situation Imagine the lowest penalty paths corresponding to several dierent segmentations produced the same answer the same label sequence Then it could be argued that the overall penalty for the interpretation should be smaller than the penalty obtained when only one path produced that interpretation because multiple paths with identical label sequences are more evidence that the label sequence is correct Several rules can be used com pute the penalty associated to a graph that contains several parallel paths We use a combination rule borrowed from a probabilistic interpretation of the penalties as negative log posteriors In a probabilistic framework the posterior probability for the interpretation should be the sum of the posteriors for all the paths that produce that interpreta tion Translated in terms of penalties the penalty of an interpretation should be the negative logarithm of the sum of the negative exponentials of the penalties of the individ ual paths The overall penalty will be smaller than all the penalties of the individual paths

Given an interpretation there is a well known method called the forward algorithm for computing the above quan tity eciently  The penalty computed with this pro cedure for a particular interpretation is called the forward penalty Consider again the concept of constrained graph the subgraph of the interpretation graph which contains only the paths that are consistent with a particular label sequence There is one constrained graph for each pos sible label sequence some may be empty graphs which have innite penalties Given an interpretation running the forward algorithm on the corresponding constrained graph gives the forward penalty for that interpretation

The forward algorithm proceeds in a way very similar to the Viterbi algorithm except that the operation used at each node to combine the incoming cumulated penalties instead of being the min function is the socalled logadd operation which can be seen as a soft version of the min

PROC OF THE IEEE NOVEMBER    function fn  logaddiUn ci  fsi   where fstart   Un is the set of upstream arcs of node n ci is the penalty on arc i and logaddx xxn   log nXi exi  

Note that because of numerical inaccuracies it is better to factorize the largest exi corresponding to the smallest penalty out of the logarithm

An interesting analogy can be drawn if we consider that a graph on which we apply the forward algorithm is equiv alent to a neural network on which we run a forward prop agation except that multiplications are replaced by addi tions the additions are replaced by logadds and there are no sigmoids

One way to understand the forward algorithm is to think about multiplicative scores eg probabilities instead of additive penalties on the arcs score  exp penalty  In that case the Viterbi algorithm selects the path with the largest cumulative score with scores multiplied along the path whereas the forward score is the sum of the cumula tive scores associated to each of the possible paths from the start to the end node The forward penalty is always lower than the cumulated penalty on any of the paths but if one path dominates with a much lower penalty its penalty is almost equal to the forward penalty The forward algo rithm gets its name from the forward pass of the wellknown

BaumWelsh algorithm for training Hidden Markov Mod els  Section VIIIE gives more details on the relation between this work and HMMs

The advantage of the forward penalty with respect to the Viterbi penalty is that it takes into account all the di	erent ways to produce an answer and not just the one with the lowest penalty This is important if there is some ambiguity in the segmentation since the combined forward penalty of two paths C and C associated with the same label sequence may be less than the penalty of a path C associated with another label sequence even though the penalty of C might be less than any one of C or C The Forward training GTN is only a slight modica tion of the previously introduced Viterbi training GTN It suces to turn the Viterbi transformers in Figure  into

Forward Scorers that take an interpretation graph as input an produce the forward penalty of that graph on output

Then the penalties of all the paths that contain the correct answer are lowered instead of just that of the best one

Backpropagating through the forward penalty computa tion the forward transformer is quite di	erent from back propagating through a Viterbi transformer All the penal ties of the input graph have an inuence on the forward penalty but penalties that belong to lowpenalty paths have a stronger inuence Computing derivatives with re spect to the forward penalties fn computed at each n node of a graph is done by backpropagation through the graph

Constrained

Interpretation Graph

Recognition

Transformer

Interpretation Graph

Path Selector

Forward Scorer

Forward Scorer

Edforw

Cforw

Cdforw + − Gc Gint

Desired

Sequence

Fig  Discriminative Forward Training GTN Architecture for a character string recognizer based on Heuristic Over Segmentation

Gc E fn  efn XiDn f E di efdi ci  where Dn  farc i with source si  ng is the set of down stream arcs from node n From the above derivatives the derivatives with respect to the arc penalties are obtained E ci  E fdi ecifsi fdi 

This can be seen as a soft version of the backpropagation through a Viterbi scorer and transformer All the arcs in

Gc have an inuence on the loss function The arcs that belong to low penalty paths have a larger inuence Back propagation through the path selector is the same as before

The derivative with respect to Gint arcs that have an alter ego in Gc are simply copied from the corresponding arc in

Gc The derivatives with respect to the other arcs are 

Several authors have applied the idea of back propagating gradients through a forward scorer to train speech recognition systems including Bridle and his net model  and Ha	ner and his TDNN model  but these authors recommended discriminative training as de scribed in the next section

D Discriminative Forward Training

The information contained in the forward penalty can be used in another discriminative training criterion which we will call the discriminative forward criterion This criterion corresponds to maximization of the posterior probability of choosing the paths associated with the correct interpreta tion This posterior probability is dened as the exponen tial of the minus the constrained forward penalty normal ized by the exponential of minus the unconstrained forward penalty Note that the forward penalty of the constrained graph is always larger or equal to the forward penalty of the unconstrained interpretation graph Ideally we would like the forward penalty of the constrained graph to be equal to

PROC OF THE IEEE NOVEMBER   the forward penalty of the complete interpretation graph

Equality between those two quantities is achieved when the combined penalties of the paths with the correct label se quence is negligibly small compared to the penalties of all the other paths or that the posterior probability associ ated to the paths with the correct interpretation is almost  which is precisely what we want The corresponding

GTN training architecture is shown in gure 

Let the di	erence be denoted Edforw and let us call

Ccforw the forward penalty of the constrained graph and

Cforw the forward penalty of the complete interpretation graph

Edforw  Ccforw  Cforw 

Edforw is always positive since the constrained graph is a subset of the paths in the interpretation graph and the forward penalty of a graph is always larger than the for ward penalty of a subgraph of this graph In the ideal case the penalties of incorrect paths are innitely large there fore the two penalties coincide and Edforw is zero Readers familiar with the Boltzmann machine connectionist model might recognize the constrained and unconstrained graphs as analogous to the clamped constrained by the ob served values of the output variable and free uncon strained phases of the Boltzmann machine algorithm 

Backpropagating derivatives through the discriminative Forward GTN distributes gradients more evenly than in the

Viterbi case Derivatives are backpropagated through the left half of the the GTN in Figure  down to the interpre tation graph Derivatives are negated and backpropagated through the righthalf and the result for each arc is added to the contribution from the left half Each arc in Gint now has a derivative Arcs that are part of a correct path have a positive derivative This derivative is very large if an incorrect path has a lower penalty than all the correct paths Similarly the derivatives with respect to arcs that are part of a lowpenalty incorrect path have a large nega tive derivative On the other hand if the penalty of a path associated with the correct interpretation is much smaller than all other paths the loss function is very close to  and almost no gradient is backpropagated The training therefore concentrates on examples of images which yield a classication error and furthermore it concentrates on the pieces of the image which cause that error Discriminative forward training is an elegant and ecient way of solving the infamous credit assignment problem for learning ma chines that manipulate dynamic data structures such as graphs More generally the same idea can be used in all situations where a learning machine must choose between discrete alternative interpretations

As previously the derivatives on the interpretation graph penalties can then be backpropagated into the character recognizer instances Backpropagation through the char acter recognizer gives derivatives on its parameters All the gradient contributions for the di	erent candidate segments are added up to obtain the total gradient associated to one pair input image correct label sequence that is one ex ample in the training set A step of stochastic gradient descent can then be applied to update the parameters

E Remarks on Discriminative Training

In the above discussion the global training criterion was given a probabilistic interpretation but the individ ual penalties on the arcs of the graphs were not There are good reasons for that For example if some penalties are associated to the di	erent class labels they would  have to sum to  class posteriors or  integrate to  over the input domain likelihoods

Let us rst discuss the rst case class posteriors normal ization This local normalization of penalties may elimi nate information that is important for locally rejecting all the classes  eg when a piece of image does not cor respond to a valid character class because some of the segmentation candidates may be wrong Although an ex plicit garbage class can be introduced in a probabilistic framework to address that question some problems remain because it is dicult to characterize such a class probabilis tically and to train a system in this way it would require a density model of unseen or unlabeled samples

The probabilistic interpretation of individual variables plays an important role in the BaumWelsh algorithm in combination with the ExpectationMaximization proce dure Unfortunately those methods cannot be applied to discriminative training criteria and one is reduced to us ing gradientbased methods Enforcing the normalization of the probabilistic quantities while performing gradient based learning is complex inecient time consuming and creates illconditioning of the lossfunction

Following  we therefore prefer to postpone normal ization as far as possible in fact until the nal decision stage of the system Without normalization the quanti ties manipulated in the system do not have a direct prob abilistic interpretation

Let us now discuss the second case using a generative model of the input Generative models build the boundary indirectly by rst building an independent density model for each class and then performing classication decisions on the basis of these models This is not a discriminative approach in that it does not focus on the ultimate goal of learning which in this case is to learn the classication de cision surface Theoretical arguments   suggest that estimating input densities when the real goal is to obtain a discriminant function for classication is a suboptimal strategy In theory the problem of estimating densities in highdimensional spaces is much more illposed than nd ing decision boundaries

Even though the internal variables of the system do not have a direct probabilistic interpretation the overall sys tem can still be viewed as producing posterior probabilities for the classes In fact assuming that a particular label se quence is given as the desired sequence to the GTN in gure  the exponential of minus Edforw can be inter preted as an estimate of the posterior probability of that label sequence given the input The sum of those posteriors for all the possible label sequences is  Another approach would consists of directly minimizing an approximation of the number of misclassications   We prefer to use the discriminative forward loss function because it causes

PROC OF THE IEEE NOVEMBER    "U"

Recognizer

Fig  Explicit segmentation can be avoided by sweeping a recog nizer at every possible location in the input eld less numerical problems during the optimization We will see in Section XC that this is a good way to obtain scores on which to base a rejection strategy The important point being made here is that one is free to choose any param eterization deemed appropriate for a classication model

The fact that a particular parameterization uses internal variables with no clear probabilistic interpretation does not make the model any less legitimate than models that ma nipulate normalized quantities

An important advantage of global and discriminative training is that learning focuses on the most important errors and the system learns to integrate the ambigui ties from the segmentation algorithm with the ambigui ties of the character recognizer In Section IX we present experimental results with an online handwriting recogni tion system that conrm the advantages of using global training versus separate training Experiments in speech recognition with hybrids of neural networks and HMMs also showed marked improvements brought by global train ing    

VII Multiple Object Recognition Space

Displacement Neural Network

There is a simple alternative to explicitly segmenting im ages of character strings using heuristics The idea is to sweep a recognizer at all possible locations across a nor malized image of the entire word or string as shown in

Figure  With this technique no segmentation heuris tics are required since the system essentially examines al l the possible segmentations of the input However there are problems with this approach First the method is in general quite expensive The recognizer must be applied at every possible location on the input or at least at a large enough subset of locations so that misalignments of characters in the eld of view of the recognizers are small enough to have no e	ect on the error rate Second when the recognizer is centered on a character to be recognized the neighbors of the center character will be present in the eld of view of the recognizer possibly touching the cen ter character Therefore the recognizer must be able to correctly recognize the character in the center of its input eld even if neighboring characters are very close to or touching the central character Third a word or charac ter string cannot be perfectly size normalized Individual $

Fig  A Space Displacement Neural Network is a convolutional network that has been replicated over a wide input eld characters within a string may have widely varying sizes and baseline positions Therefore the recognizer must be very robust to shifts and size variations

These three problems are elegantly circumvented if a convolutional network is replicated over the input eld

First of all as shown in section III convolutional neu ral networks are very robust to shifts and scale varia tions of the input image as well as to noise and extra neous marks in the input These properties take care of the latter two problems mentioned in the previous para graph Second convolutional networks provide a drastic saving in computational requirement when replicated over large input elds A replicated convolutional network also called a Space Displacement Neural Network or SDNN  is shown in Figure  While scanning a recognizer can be prohibitively expensive in general convolutional net works can be scanned or replicated very eciently over large variablesize input elds Consider one instance of a convolutional net and its alter ego at a nearby location

Because of the convolutional nature of the network units in the two instances that look at identical locations on the input have identical outputs therefore their states do not need to be computed twice Only a thin slice of new states that are not shared by the two network instances needs to be recomputed When all the slices are put to gether the result is simply a larger convolutional network whose structure is identical to the original network except that the feature maps are larger in the horizontal dimen sion In other words replicating a convolutional network can be done simply by increasing the size of the elds over which the convolutions are performed and by replicating the output layer accordingly The output layer e	ectively becomes a convolutional layer An output whose receptive eld is centered on an elementary ob ject will produce the class of this ob ject while an inbetween output may indi cate no character or contain rubbish The outputs can be interpreted as evidences for the presence of ob jects at all possible positions in the input eld

The SDNN architecture seems particularly attractive for

PROC OF THE IEEE NOVEMBER   recognizing cursive handwriting where no reliable segmen tation heuristic exists Although the idea of SDNN is quite old and very attractive by its simplicity it has not gener ated wide interest until recently because as stated above it puts enormous demands on the recognizer   In speech recognition where the recognizer is at least one order of magnitude smaller replicated convolutional net works are easier to implement for instance in Ha	ners

MultiState TDNN model  

A Interpreting the Output of an SDNN with a GTN

The output of an SDNN is a sequence of vectors which encode the likelihoods penalties or scores of nding char acter of a particular class label at the corresponding lo cation in the input A postprocessor is required to pull out the best possible label sequence from this vector se quence An example of SDNN output is shown in Fig ure  Very often individual characters are spotted by several neighboring instances of the recognizer a conse quence of the robustness of the recognizer to horizontal translations Also quite often characters are erroneously detected by recognizer instances that see only a piece of a character For example a recognizer instance that only sees the right third of a  might output the label  How can we eliminate those extraneous characters from the out put sequence and pullout the best interpretation This can be done using a new type of Graph Transformer with two input graphs as shown in Figure  The sequence of vectors produced by the SDNN is rst coded into a linear graph with multiple arcs between pairs of successive nodes

Each arc between a particular pair of nodes contains the label of one of the possible categories together with the penalty produced by the SDNN for that class label at that location This graph is called the SDNN Output Graph The second input graph to the transformer is a grammar transducer more specically a nitestate transducer  that encodes the relationship between input strings of class labels and corresponding output strings of recognized char actersThe transducer is a weighted nite state machine a graph where each arc contains a pair of labels and possibly a penalty Like a nitestate machine a transducer is in a state and follows an arc to a new state when an observed input symbol matches the rst symbol in the symbol pair attached to the arc At this point the transducer emits the second symbol in the pair together with a penalty that com bines the penalty of the input symbol and the penalty of the arc A transducer therefore transforms a weighted sym bol sequence into another weighted symbol sequence The graph transformer shown in gure  performs a composi tion between the recognition graph and the grammar trans ducer This operation takes every possible sequence corre sponding to every possible path in the recognition graph and matches them with the paths in the grammar trans ducer The composition produces the interpretation graph which contains a path for each corresponding output label sequence This composition operation may seem combina torially intractable but it turns out there exists an ecient algorithm for it described in more details in Section VIII

Viterbi Transformer

SDNN

Transformer

Compose

Viterbi Answer

Character

Model

Transducer

S....c.....r......i....p....t s....e.....n.....e.j...o.T

######################## 5......a...i...u......p.....f SDNN Output
Interpretation Graph

Viterbi Graph

Fig  A Graph Transformer pulls out the best interpretation from the output of the SDNN 2 3 3 4 5 2345

C1 C3 C5

F6

Input

SDNN

Output

Compose + Viterbi

Answer

Fig  An example of multiple character recognition with SDNN

With SDNN no explicit segmentation is performed

B Experiments with SDNN

In a series of experiments LeNet was trained with the goal of being replicated so as to recognize multiple char acters without segmentations The data was generated from the previously described Modied NIST set as fol lows Training images were composed of a central char acter anked by two side characters picked at random in the training set The separation between the bounding boxes of the characters were chosen at random between  and  pixels In other instances no central character was present in which case the desired output of the network was the blank space class In addition training images were degraded with ! salt and pepper noise random pixel inversions

Figures  and  show a few examples of success ful recognitions of multiple characters by the LeNet

SDNN Standard techniques based on Heuristic Over

Segmentation would fail miserably on many of those ex amples As can be seen on these examples the network exhibits striking invariance and noise resistance properties

While some authors have argued that invariance requires more sophisticated models than feedforward neural net works  LeNet exhibits these properties to a large ex tent

PROC OF THE IEEE NOVEMBER   6 7 7 7 8 8 678 3 5 5 1 1 4 3514 1 1 1 4 4 1 1114 5 5 4 0 540

Input

F6

SDNN output

Answer

Fig  An SDNN applied to a noisy image of digit string The digits shown in the SDNN output represent the winning class labels with a lighter grey level for highpenalty answers

Similarly it has been suggested that accurate recognition of multiple overlapping ob jects require explicit mechanisms that would solve the socalled feature binding problem 

As can be seen on Figures  and  the network is able to tell the characters apart even when they are closely inter twined a task that would be impossible to achieve with the more classical Heuristic OverSegmentation technique The

SDNN is also able to correctly group disconnected pieces of ink that form characters Good examples of that are shown in the upper half of gure  In the top left ex ample the  and the  are more connected to each other than they are connected with themselves yet the system correctly identies the  and the  as separate ob jects The top right example is interesting for several reasons First the system correctly identies the three individual ones

Second the left half and right half of disconnected  are correctly grouped even though no geometrical information could decide to associate the left half to the vertical bar on its left or on its right The right half of the  does cause the appearance of an erroneous  on the SDNN output but this one is removed by the character model transducer which prevents characters from appearing on contiguous outputs

Another important advantage of SDNN is the ease with which they can be implemented on parallel hardware Spe cialized analogdigital chips have been designed and used in character recognition and in image preprocessing appli cations  However the rapid progress of conventional processor technology with reducedprecision vector arith metic instructions such as Intels MMX make the success of specialized hardware hypothetical at best

Short video clips of the LeNet SDNN can be viewed at httpwwwresearchattcomyannocr C Global Training of SDNN

In the above experiments the string image were arti cially generated from individual character The advantage is that we know in advance the location and the label of the important character With real training data the cor rect sequence of labels for a string is generally available but the precise locations of each corresponding character in the input image are unknown

In the experiments described in the previous section the best interpretation was extracted from the SDNN output using a very simple graph transformer Global training of an SDNN can be performed by backpropagating gradients through such graph transformers arranged in architectures similar to the ones described in section VI

PROC OF THE IEEE NOVEMBER  

Constrained

Interpretation Graph

Interpretation Graph

Path Selector

Forward Scorer

Forward Scorer

Edforw

Cforw

Cdforw + − Gc Gint

Desired

Sequence

SDNN

Transformer

Compose

Character

Model

Transducer

S....c.....r......i....p....t s....e.....n.....e.j...o.T

######################## 5......a...i...u......p.....f SDNN Output
Fig  A globally trainable SDNNHMM hybrid system expressed as a GTN

This is somewhat equivalent to modeling the output of an SDNN with a Hidden Markov Model Globally trained variablesize TDNNHMM hybrids have been used for speech recognition and online handwriting recogni tion     Space Displacement Neural Net works have been used in combination with HMMs or other elastic matching methods for handwritten word recogni tion  

Figure  shows the graph transformer architecture for training an SDNNHMM hybrid with the Discriminative Forward Criterion The top part is comparable to the top part of gure  On the right side the composition of the recognition graph with the grammar gives the interpreta tion graph with all the possible legal interpretations On the left side the composition is performed with a grammar that only contains paths with the desired sequence of la bels This has a somewhat similar function to the path selector used in the previous section Like in Section VID the loss function is the di	erence between the forward score obtained from the left half and the forward score obtained from the right half To backpropagate through the com position transformer we need to keep a record of which arc in the recognition graph originated which arcs in the inter pretation graph The derivative with respect to an arc in the recognition graph is equal to the sum of the derivatives with respect to all the arcs in the interpretation graph that originated from it Derivative can also be computed for the penalties on the grammar graph allowing to learn them as well As in the previous example a discriminative criterion must be used because using a nondiscriminative criterion could result in a collapse e	ect if the networks output RBF are adaptive The above training procedure can be equiv alently formulated in term of HMM Early experiments in zip code recognition  and more recent experiments in online handwriting recognition  have demonstrated the idea of globallytrained SDNNHMM hybrids SDNN is an extremely promising and attractive technique for OCR but so far it has not yielded better results than Heuristic Over

Segmentation We hope that these results will improve as more experience is gained with these models

D Object Detection and Spotting with SDNN

An interesting application of SDNNs is ob ject detection and spotting The invariance properties of Convolutional

Networks combined with the eciency with which they can be replicated over large elds suggest that they can be used for brute force ob ject spotting and detection in large images The main idea is to train a single Convolu tional Network to distinguish images of the ob ject of inter est from images present in the background In utilization mode the network is replicated so as to cover the entire image to be analyzed thereby forming a twodimensional

Space Displacement Neural Network The output of the

SDNN is a twodimensional plane in which activated units indicate the presence of the ob ject of interest in the corre sponding receptive eld Since the sizes of the ob jects to be detected within the image are unknown the image can be presented to the network at multiple resolutions and the results at multiple resolutions combined The idea has been applied to face location  address block location on envelopes  and hand tracking in video 

To illustrate the method we will consider the case of face detection in images as described in  First images containing faces at various scales are collected Those im ages are ltered through a zeromean Laplacian lter so as to remove variations in global illumination and low spatial frequency illumination gradients Then training samples of faces and nonfaces are manually extracted from those images The face subimages are then size normalized so that the height of the entire face is approximately  pixels while keeping fairly large variations within a factor of two

The scale of background subimages are picked at random

A single convolutional network is trained on those samples to classify face subimages from nonface subimages

When a scene image is to be analyzed it is rst ltered through the Laplacian lter and subsampled at powers oftwo resolutions The network is replicated over each of multiple resolution images A simple voting technique is used to combine the results from multiple resolutions

A twodimensional version of the global training method described in the previous section can be used to allevi ate the need to manually locate faces when building the training sample  Each possible location is seen as an alternative interpretation ie one of several parallel arcs in a simple graph that only contains a start node and an end node

Other authors have used Neural Networks or other clas siers such as Support Vector Machines for face detection with great success   Their systems are very similar to the one described above including the idea of presenting the image to the network at multiple scales But since those

PROC OF THE IEEE NOVEMBER   systems do not use Convolutional Networks they cannot take advantage of the speedup described here and have to rely on other techniques such as preltering and realtime tracking to keep the computational requirement within reasonable limits In addition because those classiers are much less invariant to scale variations than Convolutional

Networks it is necessary to multiply the number of scales at which the images are presented to the classier

VIII Graph Transformer Networks and

Transducers

In Section IV Graph Transformer Networks GTN were introduced as a generalization of multilayer multi module networks where the state information is repre sented as graphs instead of xedsize vectors This section reinterprets the GTNs in the framework of Generalized Transduction and proposes a powerful Graph Composition algorithm

A Previous Work

Numerous authors in speech recognition have used

GradientBased Learning methods that integrate graph based statistical models notably HMM with acoustic recognition modules mainly Gaussian mixture models but also neural networks     Similar ideas have been applied to handwriting recognition see  for a re view However there has been no proposal for a system atic approach to multilayer graphbased trainable systems

The idea of transforming graphs into other graphs has re ceived considerable interest in computer science through the concept of weighted nitestate transducers  Trans ducers have been applied to speech recognition  and language translation  and proposals have been made for handwriting recognition  This line of work has been mainly focused on ecient search algorithms  and on the algebraic aspects of combining transducers and graphs called acceptors in this context but very little e	ort has been devoted to building globally trainable sys tems out of transducers What is proposed in the follow ing sections is a systematic approach to automatic training in graphmanipulating systems A di	erent approach to graphbased trainable systems called InputOutput HMM was proposed in  

B Standard Transduction

In the established framework of nitestate transduc ers  discrete symbols are attached to arcs in the graphs

Acceptor graphs have a single symbol attached to each arc whereas transducer graphs have two symbols an input symbol and an output symbol A special null symbol is absorbed by any other symbol when concatenating sym bols to build a symbol sequence Weighted transducers and acceptors also have a scalar quantity attached to each arc In this framework the composition operation takes as input an acceptor graph and a transducer graph and builds an output acceptor graph Each path in this output graph with symbol sequence Sout corresponds to one path with symbol sequence Sin in the input acceptor graph and one path and a corresponding pair of inputoutput sequences SoutSin in the transducer graph The weights on the arcs of the output graph are obtained by adding the weights from the matching arcs in the input acceptor and trans ducer graphs In the rest of the paper we will call this graph composition operation using transducers the 	stan dard transduction operation A simple example of transduction is shown in Figure 

In this simple example the input and output symbols on the transducer arcs are always identical This type of trans ducer graph is called a grammar graph To better under stand the transduction operation imagine two tokens sit ting each on the start nodes of the input acceptor graph and the transducer graph The tokens can freely follow any arc labeled with a null input symbol A token can follow an arc labeled with a nonnull input symbol if the other token also follows an arc labeled with the same in put symbol We have an acceptable trajectory when both tokens reach the end nodes of their graphs ie the tokens have reached the terminal conguration This tra jectory represents a sequence of input symbols that complies with both the acceptor and the transducer We can then collect the corresponding sequence of output symbols along the tra jectory of the transducer token The above procedure produces a tree but a simple technique described in Sec tion VIIIC can be used to avoid generating multiple copies of certain subgraphs by detecting when a particular output state has already been seen

The transduction operation can be performed very e ciently  but presents complex bookkeeping problems concerning the handling of all combinations of null and non null symbols If the weights are interpreted as probabilities normalized appropriately then an acceptor graph repre sents a probability distribution over the language dened by the set of label sequences associated to all possible paths from the start to the end node in the graph

An example of application of the transduction opera tion is the incorporation of linguistic constraints a lexicon or a grammar when recognizing words or other character strings The recognition transformer produces the recog nition graph an acceptor graph by applying the neural network recognizer to each candidate segment This ac ceptor graph is composed with a transducer graph for the grammar The grammar transducer contains a path for each legal sequence of symbol possibly augmented with penalties to indicate the relative likelihoods of the possi ble sequences The arcs contain identical input and output symbols Another example of transduction was mentioned in Section V the path selector used in the heuristic over segmentation training GTN is implementable by a compo sition The transducer graph is linear graph which con tains the correct label sequence The composition of the interpretation graph with this linear graph yields the con strained graph

C Generalized Transduction

If the data structures associated to each arc took only a nite number of values composing the input graph and

PROC OF THE IEEE NOVEMBER   an appropriate transducer would be a sound solution For our applications however the data structures attached to the arcs of the graphs may be vectors images or other highdimensional ob jects that are not readily enumerated

We present a new composition operation that solves this problem

Instead of only handling graphs with discrete symbols and penalties on the arcs we are interested in considering graphs whose arcs may carry complex data structures in cluding continuousvalued data structures such as vectors and images Composing such graphs requires additional information

When examining a pair of arcs one from each input graph we need a criterion to decide whether to create cor responding arcs and nodes in the output graph based on the information attached to the input arcs We can de cide to build an arc several arcs or an entire subgraph with several nodes and arcs

When that criterion is met we must build the corre sponding arcs and nodes in the output graph and com pute the information attached to the newly created arcs as a function the the information attached to the input arcs

These functions are encapsulated in an ob ject called a

Composition Transformer An instance of Composition

Transformer implements three methods checkarc arc  compares the data structures pointed to by arcs arc from the rst graph and arc from the second graph and re turns a boolean indicating whether corresponding arcs should be created in the output graph fpropngraph upnode downnode arc arc  is called when checkarc arc returns true This method creates new arcs and nodes between nodes upnode and downnode in the output graph ngraph and computes the information attached to these newly created arcs as a function of the attached information of the input arcs arc and arc  bpropngraph upnode downnode arc arc  is called during training in order to propagate gradient in formation from the output subgraph between upnode and downnode into the data structures on the arc and arc as well as with respect to the parameters that were used in the fprop call with the same arguments This assumes that the function used by fprop to compute the values attached to its output arcs is di	erentiable

The check method can be seen as constructing a dy namic architecture of functional dependencies while the fprop method performs a forward propagation through that architecture to compute the numerical information at tached to the arcs The bprop method performs a back ward propagation through the same architecture to com pute the partial derivatives of the loss function with respect to the information attached to the arcs This is illustrated in Figure 

Figure  shows a simplied generalized graph composi tion algorithm This simplied algorithm does not handle null transitions and does not check whether the tokens tra "o" "c" "d" "x" "a" "u" "p" "t"

## 0.4
## 1.0
## 1.8
## 0.1
## 0.2
## 0.8
## 0.2
## 0.8
Recognition

Graph "b" "c" "a" "u" "u" "a" "r" "n" "t" "t" "r" "e" "e" "p" "t" "r" "d" "c" "u" "a" "t" "p" "t"

### 0.4 0.2
## 0.8
## 0.8
## 0.2
## 0.8
 interpretation graph match & add match & add match & add interpretations: cut (2.0) cap (0.8) cat (1.4) grammar graph

Fig  Example of composition of the recognition graph with the grammar graph in order to build an interpretation that is consistent with both of them During the forward propagation dark arrows the methods check and fprop are used Gradients dashed arrows are backpropagated with the application of the method bprop jectory is acceptable ie both tokens simultaneously reach the end nodes of their graphs The management of null transitions is a straightforward modication of the token simulation function Before enumerating the possible non null joint token transitions we loop on the possible null transitions of each token recursively call the token sim ulation function and nally call the method fprop The safest way for identifying acceptable tra jectories consists in running a preliminary pass for identifying the token con gurations from which we can reach the terminal congu ration ie both tokens on the end nodes This is easily achieved by enumerating the tra jectories in the opposite direction We start on the end nodes and follow the arcs upstream During the main pass we only build the nodes that allow the tokens to reach the terminal conguration

Graph composition using transducers ie standard transduction is easily and eciently implemented as a gen eralized transduction The method check simply tests the equality of the input symbols on the two arcs and the method fprop creates a single arc whose symbol is the output symbol on the transducers arc

The composition between pairs of graphs is particularly useful for incorporating linguistic constraints in a hand writing recognizer Examples of its use are given in the online handwriting recognition system described in Sec tion IX and in the check reading system described in Sec tion X

In the rest of the paper the term Composition Trans former will denote a Graph Transformer based on the gen eralized transductions of multiple graphs The concept of generalized transduction is a very general one In fact many of the graph transformers described earlier in this paper such as the segmenter and the recognizer can be Graph Composition

PROC OF THE IEEE NOVEMBER  

Function generalizedcompositionPGRAPH graph

PGRAPH graph

PTRANS trans 

Returns PGRAPH 

Create new graph

PGRAPH ngraph  newgraph 

Create map between token positions and nodes of the new graph

PNODE mapPNODEPNODE  newemptymap  mapendnodegraph  endnodegraph   endnodenewgraph 

Recursive subroutine for simulating tokens

Function simtokensPNODE node PNODE node 

Returns PNODE 

PNODE currentnode  mapnode node

Check if already visited

If currentnode  nil 

Record new configuration currentnode  ngraphcreatenode  mapnode node  currentnode

Enumerate the possible nonnull joint token transitions

For ARC arc in downarcsnode 

For ARC arc in downarcsnode 

If transcheckarc arc  

PNODE newnode  simtokensdownnodearc  downnodearc   transfpropngraph currentnode newnode arc arc 

Return node in composed graph

Return currentnode 

Perform token simulation simtokensstartnodegraph  startnodegraph  

Delete map

Return ngraph 

Fig  Pseudocode for a simplied generalized composition algo rithm For simplifying the presentation we do not handle null transitions nor implement dead end avoidance The two main component of the composition appear clearly here	 a the re cursive function simtoken enumerating the token tra jectories and b the associative array map used for remembering which nodes of the composed graph have been visited formulated in terms of generalized transduction In this case the the generalized transduction does not take two in put graphs but a single input graph The method fprop of the transformer may create several arcs or even a complete subgraph for each arc of the initial graph In fact the pair check fprop itself can be seen as procedurally dening a transducer

In addition It can be shown that the generalized trans duction of a single graph is theoretically equivalent to the standard composition of this graph with a particular trans ducer graph However implementing the operation this way may be very inecient since the transducer can be very complicated

In practice the graph produced by a generalized trans duction is represented procedurally in order to avoid build ing the whole output graph which may be huge when for example the interpretation graph is composed with the grammar graph We only instantiate the nodes which are visited by the search algorithm during recognition eg

Viterbi This strategy propagates the benets of pruning algorithms eg Beam Search in all the Graph Transformer

Network

D Notes on the Graph Structures

Section VI has discussed the idea of global training by backpropagating gradient through simple graph trans formers The bprop method is the basis of the back propagation algorithm for generic graph transformers A generalized composition transformer can be seen as dynam ically establishing functional relationships between the nu merical quantities on the input and output arcs Once the check function has decided that a relationship should be es tablished the fprop function implements the numerical re lationship The check function establishes the structure of the ephemeral network inside the composition transformer

Since fprop is assumed to be di	erentiable gradients can be backpropagated through that structure Most param eters a	ect the scores stored on the arcs of the successive graphs of the system A few threshold parameters may de termine whether an arc appears or not in the graph Since non existing arcs are equivalent to arcs with very large penalties we only consider the case of parameters a	ect ing the penalties

In the kind of systems we have discussed until now and the application described in Section X much of the knowl edge about the structure of the graph that is produced by a Graph Transformer is determined by the nature of the

Graph Transformer but it may also depend on the value of the parameters and on the input It may also be interest ing to consider Graph Transformer modules which attempt to learn the structure of the output graph This might be considered a combinatorial problem and not amenable to GradientBased Learning but a solution to this prob lem is to generate a large graph that contains the graph candidates as subgraphs and then select the appropriate subgraph

PROC OF THE IEEE NOVEMBER  

E GTN and Hidden Markov Models

GTNs can be seen as a generalization and an extension of

HMMs On the one hand the probabilistic interpretation can be either kept with penalties being logprobabilities pushed to the nal decision stage with the di	erence of the constrained forward penalty and the unconstrained forward penalty being interpreted as negative logprobabilities of label sequences or dropped altogether the network just represents a decision surface for label sequences in input space On the other hand Graph Transformer Networks extend HMMs by allowing to combine in a wellprincipled framework multiple levels of processing or multiple mod els eg Pereira et al have been using the transducer framework for stacking HMMs representing di	erent levels of processing in automatic speech recognition 

Unfolding a HMM in time yields a graph that is very sim ilar to our interpretation graph at the nal stage of pro cessing of the Graph Transformer Network before Viterbi recognition It has nodes nt i associated to each time step t and state i in the model The penalty ci for an arc from nt   j to nt i then corresponds to the nega tive logprobability of emitting observed data ot at posi tion t and going from state j to state i in the time interval t   t With this probabilistic interpretation the for ward penalty is the negative logarithm of the likelihood of whole observed data sequence given the model

In Section VI we mentioned that the collapsing phe nomenon can occur when nondiscriminative loss functions are used to train neural networksHMM hybrid systems

With classical HMMs with xed preprocessing this prob lem does not occur because the parameters of the emission and transition probability models are forced to satisfy cer tain probabilistic constraints the sum or the integral of the probabilities of a random variable over its possible val ues must be  Therefore when the probability of certain events is increased the probability of other events must au tomatically be decreased On the other hand if the prob abilistic assumptions in an HMM or other probabilistic model are not realistic discriminative training discussed in Section VI can improve performance as this has been clearly shown for speech recognition systems      

The InputOutput HMM model IOHMM   is strongly related to graph transformers Viewed as a probabilistic model an IOHMM represents the conditional distribution of output sequences given input sequences of the same or a di	erent length It is parameterized from an emission probability module and a transition probabil ity module The emission probability module computes the conditional emission probability of an output variable given an input value and the value of discrete state vari able The transition probability module computes condi tional transition probabilities of a change in the value of the state variable given the an input value Viewed as a graph transformer it assigns an output graph representing a probability distribution over the sequences of the output variable to each path in the input graph All these output graphs have the same structure and the penalties on their arcs are simply added in order to obtain the complete out put graph The input values of the emission and transition modules are read o	 the data structure on the input arcs of the IOHMM Graph Transformer In practice the out put graph may be very large and needs not be completely instantiated ie it is pruned only the low penalty paths are created

IX An OnLine Handwriting Recognition System

Natural handwriting is often a mixture of di	erent styles  lower case printed upper case and cursive A reliable recognizer for such handwriting would greatly im prove interaction with penbased devices but its imple mentation presents new technical challenges Characters taken in isolation can be very ambiguous but consider able information is available from the context of the whole word We have built a word recognition system for pen based devices based on four main modules a preprocessor that normalizes a word or word group by tting a geomet rical model to the word structure$ a module that produces an annotated image from the normalized pen tra jectory$ a replicated convolutional neural network that spots and recognizes characters$ and a GTN that interprets the net works output by taking wordlevel constraints into account

The network and the GTN are jointly trained to minimize an error measure dened at the word level

In this work we have compared a system based on

SDNNs such as described in Section VII and a system based on Heuristic OverSegmentation such as described in Section V Because of the sequential nature of the infor mation in the pen tra jectory which reveals more informa tion than the purely optical input from in image Heuristic

OverSegmentation can be very ecient in proposing can didate character cuts especially for noncursive script

A Preprocessing

Input normalization reduces intracharacter variability simplifying character recognition We have used a word normalization scheme  based on tting a geometrical model of the word structure Our model has four exi ble lines representing respectively the ascenders line the core line the base line and the descenders line The lines are tted to local minima or maxima of the pen tra jectory The parameters of the lines are estimated with a modied version of the EM algorithm to maximize the joint prob ability of observed points and parameter values using a prior on parameters that prevents the lines from collapsing on each other

The recognition of handwritten characters from a pen tra jectory on a digitizing surface is often done in the time domain    Typically tra jectories are nor malized and local geometrical or dynamical features are extracted The recognition may then be performed us ing curve matching  or other classication techniques such as TDNNs   While these representations have several advantages their dependence on stroke order ing and individual writing styles makes them dicult to

PROC OF THE IEEE NOVEMBER    "Script"

Viterbi Graph

Segmentation Graph

Recognition Graph

Compose

Recognition

Transformer

Segmentation

Transformer

Word Normalization

Normalized Word

Interpretation Graph

Language

Model

AMAP Computation

AMAP Graph

Beam Search

Transformer

Fig  An online handwriting recognition GTN based on heuristic oversegmentation use in high accuracy writer independent systems that in tegrate the segmentation with the recognition

Since the intent of the writer is to produce a legible im age it seems natural to preserve as much of the pictorial nature of the signal as possible while at the same time ex ploit the sequential information in the tra jectory For this purpose we have designed a representation scheme called

AMAP  where pen tra jectories are represented by low resolution images in which each picture element contains information about the local properties of the tra jectory An

AMAP can be viewed as an annotated image in which each pixel is a element feature vector  features are as sociated to four orientations of the pen tra jectory in the area around the pixel and the fth one is associated to local curvature in the area around the pixel A particu larly useful feature of the AMAP representation is that it makes very few assumptions about the nature of the input tra jectory It does not depend on stroke ordering or writ ing speed and it can be used with all types of handwriting capital lower case cursive punctuation symbols Un like many other representations such as global features

AMAPs can be computed for complete words without re quiring segmentation

Recognition Graph

Compose

Word Normalization

Normalized Word

Interpretation Graph "Script"

AMAP Computation

SDNN

Transformer

AMAP

Compose Character

Model

Language

Model

Viterbi Graph

SDNN Output

Beam Search

Transformer

Fig  An online handwriting recognition GTN based on Space

Displacement Neural Network

B Network Architecture

One of the best networks we found for both online and o%ine character recognition is a layer convolutional net work somewhat similar to LeNet Figure  but with multiple input planes and di	erent numbers of units on the last two layers$ layer  convolution with  kernels of size x layer  x subsampling layer  convolution with  kernels of size x layer  convolution with  kernels of size x layer  x subsampling classication layer  RBF units one per class in the full printable ASCII set The distributed codes on the output are the same as for LeNet except they are adaptive unlike with LeNet

When used in the heuristic oversegmentation system the input to above network consisted of an AMAP with ve planes  rows and  columns It was determined that this resolution was sucient for representing handwritten characters In the SDNN version the number of columns was varied according to the width of the input word Once the number of subsampling layers and the sizes of the ker nels are chosen the sizes of all the layers including the input are determined unambiguously The only architec tural parameters that remain to be selected are the num ber of feature maps in each layer and the information as to what feature map is connected to what other feature map In our case the subsampling rates were chosen as small as possible x and the kernels as small as pos

PROC OF THE IEEE NOVEMBER   sible in the rst layer x to limit the total number of connections Kernel sizes in the upper layers are chosen to be as small as possible while satisfying the size constraints mentioned above Larger architectures did not necessarily perform better and required considerably more time to be trained A very small architecture with half the input eld also performed worse because of insucient input resolu tion Note that the input resolution is nonetheless much less than for optical character recognition because the an gle and curvature provide more information than would a single grey level at each pixel

C Network Training

Training proceeded in two phases First we kept the centers of the RBFs xed and trained the network weights so as to minimize the output distance of the RBF unit corresponding to the correct class This is equivalent to minimizing the meansquared error between the previous layer and the center of the correctclass RBF This boot strap phase was performed on isolated characters In the second phase all the parameters network weights and RBF centers were trained globally to minimize a discriminative criterion at the word level

With the Heuristic OverSegmentation approach the

GTN was composed of four main Graph Transformers  The Segmentation Transformer performs the

Heuristic OverSegmentation and outputs the segmenta tion graph An AMAP is then computed for each image attached to the arcs of this graph  The Character Recognition Transformer applies the the convolutional network character recognizer to each candidate segment and outputs the recognition graph with penalties and classes on each arc  The Composition Transformer composes the recog nition graph with a grammar graph representing a language model incorporating lexical constraints  The Beam Search Transformer extracts a good inter pretation from the interpretation graph This task could have been achieved with the usual Viterbi Transformer

The Beam Search algorithm however implements pruning strategies which are appropriate for large interpretation graphs

With the SDNN approach the main Graph Transformers are the following  The SDNN Transformer replicates the convolutional network over the a whole word image and outputs a recog nition graph that is a linear graph with class penalties for every window centered at regular intervals on the input image  The CharacterLevel Composition Transformer composes the recognition graph with a lefttoright HMM for each character class as in Figure   The WordLevel Composition Transformer com poses the output of the previous transformer with a lan guage model incorporating lexical constraints and outputs the interpretation graph  The Beam Search Transformer extracts a good in terpretation from the interpretation graph

In this application the language model simply constrains the nal output graph to represent sequences of character labels from a given dictionary Furthermore the interpre tation graph is not actually completely instantiated the only nodes created are those that are needed by the Beam

Search module The interpretation graph is therefore rep resented procedurally rather than explicitly A crucial contribution of this research was the joint train ing of all graph transformer modules within the network with respect to a single criterion as explained in Sec tions VI and VIII We used the Discriminative Forward loss function on the nal output graph minimize the forward penalty of the constrained interpretation ie along all the correct paths while maximizing the forward penalty of the whole interpretation graph ie along all the paths

During global training the loss function was optimized with the stochastic diagonal LevenbergMarquardt proce dure described in Appendix C that uses second derivatives to compute optimal learning rates This optimization op erates on al l the parameters in the system most notably the network weights and the RBF centers

D Experimental Results

In the rst set of experiments we evaluated the general ization ability of the neural network classier coupled with the word normalization preprocessing and AMAP input representation All results are in writer independent mode di	erent writers in training and testing Initial train ing on isolated characters was performed on a database of approximately  hand printed characters  classes of upper case lower case digits and punctuation Tests on a database of isolated characters were performed sepa rately on the four types of characters upper case ! error on  patterns lower case ! error on  patterns digits ! error on  patterns and punc tuation ! error on  patterns Experiments were performed with the network architecture described above

To enhance the robustness of the recognizer to variations in position size orientation and other distortions addi tional training data was generated by applying local ane transformations to the original characters

The second and third set of experiments concerned the recognition of lower case words writer independent The tests were performed on a database of  words First we evaluated the improvements brought by the word nor malization to the system For the SDNNHMM system we have to use wordlevel normalization since the net work sees one whole word at a time With the Heuris tic OverSegmentation system and before doing any word level training we obtained with characterlevel normaliza tion ! and ! word and character errors adding in sertions deletions and substitutions when the search was constrained within a word dictionary When using the word normalization preprocessing instead of a charac ter level normalization error rates dropped to ! and ! for word and character errors respectively ie a rel ative drop of ! and ! in word and character error respectively This suggests that normalizing the word in

PROC OF THE IEEE NOVEMBER    its entirety is better than rst segmenting it and then nor malizing and processing each of the segments

No Language Model

## 12.4
## 8.2
No Language Model

## 8.5
## 6.3
 25K Word Lexicon 2

## 1.4
 0 5 10 15

SDNN/HMM no global training with global training no global training with global training no global training with global training

HOS

HOS

Fig  Comparative results character error rates showing the improvement brought by global training on the SDNNHMM hybrid and on the Heuristic OverSegmentation system HOS without and with a  words dictionary

In the third set of experiments we measured the im provements obtained with the joint training of the neural network and the postprocessor with the wordlevel crite rion in comparison to training based only on the errors performed at the character level After initial training on individual characters as above global wordlevel discrim inative training was performed with a database of  lower case words For the SDNNHMM system without any dictionary constraints the error rates dropped from ! and ! word and character error to ! and ! respectively after wordlevel training ie a relative drop of ! and ! For the Heuristic OverSegmentation sys tem and a slightly improved architecture without any dic tionary constraints the error rates dropped from ! and ! word and character error to ! and ! re spectively ie a relative drop of ! and ! With a word dictionary errors dropped from ! and ! word and character errors to ! and ! respectively after wordlevel training ie a relative drop of ! and ! Even lower error rates can be obtained by dras tically reducing the size of the dictionary to  words yielding ! and ! word and character errors

These results clearly demonstrate the usefulness of glob ally trained NeuralNetHMM hybrids for handwriting recognition This conrms similar results obtained earlier in speech recognition 

X A Check Reading System

This section describes a GTN based Check Reading Sys tem intended for immediate industrial deployment It also shows how the use of Gradient BasedLearning and GTNs make this deployment fast and coste	ective while yielding an accurate and reliable solution

The verication of the amount on a check is a task that is extremely time and money consuming for banks As a consequence there is a very high interest in automating the process as much as possible see for example     Even a partial automation would result in consid erable cost reductions The threshold of economic viability for automatic check readers as set by the bank is when ! of the checks are read with less than ! error The other ! of the check being rejected and sent to human operators In such a case we describe the performance of the system as  correct  reject  error The system presented here was one of the rst to cross that threshold on representative mixtures of business and per sonal checks

Checks contain at least two versions of the amount The

Courtesy amount is written with numerals while the Legal amount is written with letters On business checks which are generally machineprinted these amounts are relatively easy to read but quite dicult to nd due to the lack of standard for business check layout On the other hand these amounts on personal checks are easy to nd but much harder to read

For simplicity and speed requirements our initial task is to read the Courtesy amount only This task consists of two main steps

The system has to nd among all the elds lines of text the candidates that are the most likely to contain the courtesy amount This is obvious for many personal checks where the position of the amount is standardized However as already noted nding the amount can be rather dicult in business checks even for the human eye There are many strings of digits such as the check number the date or even not to exceed amounts that can be confused with the actual amount In many cases it is very dicult to decide which candidate is the courtesy amount before performing a full recognition

In order to read and choose some Courtesy amount candidates the system has to segment the elds into char acters read and score the candidate characters and nally nd the best interpretation of the amount using contextual knowledge represented by a stochastic grammar for check amounts

The GTN methodology was used to build a check amount reading system that handles both personal checks and busi ness checks

A A GTN for Check Amount Recognition

We now describe the successive graph transformations that allow this network to read the check amount cf Fig ure  Each Graph Transformer produces a graph whose paths encode and score the current hypotheses considered at this stage of the system

The input to the system is a trivial graph with a single arc that carries the image of the whole check cf Figure 

The eld location transformer Tf ield rst performs classical image analysis including connected component analysis ink density histograms layout analysis etc and heuristically extracts rectangular zones that may con tain the check amount Tf ield produces an output graph called the eld graph cf Figure  such that each can didate zone is associated with one arc that links the start node to the end node Each arc contains the image of the zone and a penalty term computed from simple features extracted from the zone absolute position size aspect ra tio etc The penalty term is close to zero if the features suggest that the eld is a likely candidate and is large if the eld is deemed less likely to be an amount The penalty

PROC OF THE IEEE NOVEMBER  

Segmentation Graph

Interpretation Graph

Grammar

Recognition Graph

Field Graph

Check Graph

Best Amount Graph

Compose 2nd Nat. Bank $ *** 3.45 three dollars and 45/xx not to exceed $10,000.00 $ *** 3.45 $10,000.00 45/xx $ * 3 ** 45 "$" 0.2 "*" 0.4 "3" 0.1 "B" 23.6 ....... "$" 0.2 "*" 0.4 "3" 0.1 .......

Recognition

Transformer

Segmentation Transf.

Field Location Transf.

Viterbi Answer

Viterbi Transformer

Fig  A complete check amount reader implemented as a single cascade of Graph Transformer modules Successive graph trans formations progressively extract higher level information function is di	erentiable therefore its parameter are glob ally tunable

An arc may represent separate dollar and cent amounts as a sequence of elds In fact in handwritten checks the cent amount may be written over a fractional bar and not aligned at all with the dollar amount In the worst case one may nd several cent amount candidates above and below the fraction bar for the same dollar amount

The segmentation transformer Tseg  similar to the one described in Section VIII examines each zone contained in the eld graph and cuts each image into pieces of ink using heuristic image processing techniques Each piece of ink may be a whole character or a piece of character

Each arc in the eld graph is replaced by its correspond ing segmentation graph that represents all possible group ings of pieces of ink Each eld segmentation graph is ap pended to an arc that contains the penalty of the eld in the eld graph Each arc carries the segment image to gether with a penalty that provides a rst evaluation of the likelihood that the segment actually contains a charac ter This penalty is obtained with a di	erentiable function that combines a few simple features such as the space be tween the pieces of ink or the compliance of the segment image with a global baseline and a few tunable parame ters The segmentation graph represents al l the possible segmentations of al l the eld images We can compute the penalty for one segmented eld by adding the arc penalties along the corresponding path As before using a di	eren tiable function for computing the penalties will ensure that the parameters can be optimized globally

The segmenter uses a variety of heuristics to nd candi date cut One of the most important ones is called hit and deect   The idea is to cast lines downward from the top of the eld image When a line hits a black pixel it is deected so as to follow the contour of the ob ject When a line hits a local minimum of the upper prole ie when it cannot continue downward without crossing a black pixel it is just propagated vertically downward through the ink

When two such lines meet each other they are merged into a single cut The procedure can be repeated from the bot tom up This strategy allows the separation of touching characters such as double zeros

The recognition transformer Trec iterates over all segment arcs in the segmentation graph and runs a charac ter recognizer on the corresponding segment image In our case the recognizer is LeNet the Convolutional Neural

Network described in Section II whose weights constitute the largest and most important subset of tunable parame ters The recognizer classies segment images into one of  classes full printable ASCII set plus a rubbish class for unknown symbols or badlyformed characters Each arc in the input graph Trec is replaced by  arcs in the output graph Each of those  arcs contains the label of one of the classes and a penalty that is the sum of the penalty of the corresponding arc in the input segmentation graph and the penalty associated with classifying the image in the corresponding class as computed by the recognizer In other words the recognition graph represents a weighted trellis of scored character classes Each path in this graph represents a possible character string for the correspond ing eld We can compute a penalty for this interpretation by adding the penalties along the path This sequence of characters may or may not be a valid check amount

The composition transformer Tgram selects the paths of the recognition graph that represent valid char acter sequences for check amounts This transformer takes two graphs as input the recognition graph and the gram mar graph The grammar graph contains all possible se quences of symbols that constitute a wellformed amount

The output of the composition transformer called the in terpretation graph contains all the paths in the recognition graph that are compatible with the grammar The oper ation that combines the two input graphs to produce the output is a generalized transduction see Section VIIIA di	erentiable function is used to compute the data attached to the output arc from the data attached to the input arcs

In our case the output arc receives the class label of the two arcs and a penalty computed by simply summing the penalties of the two input arcs the recognizer penalty and the arc penalty in the grammar graph Each path in the interpretation graph represents one interpretation of one segmentation of one eld on the check The sum of the penalties along the path represents the badness of the corresponding interpretation and combines evidence from each of the modules along the process as well as from the grammar

The Viterbi transformer nally selects the path with the lowest accumulated penalty corresponding to the best

PROC OF THE IEEE NOVEMBER  

Interpretation Graph

Path Selector

Forward Scorer

Forward Scorer

Edforw

Cforw

Cdforw + −

Viterbi

Answer

Fig  Additional processing required to compute the condence grammatically correct interpretations

B GradientBased Learning

Each stage of this check reading system contains tun able parameters While some of these parameters could be manually adjusted for example the parameters of the eld locator and segmenter the vast ma jority of them must be learned particularly the weights of the neural net recog nizer

Prior to globally optimizing the system each module pa rameters are initialized with reasonable values The param eters of the eld locator and the segmenter are initialized by hand while the parameters of the neural net charac ter recognizer are initialized by training on a database of presegmented and labeled characters Then the entire system is trained globally from whole check images labeled with the correct amount No explicit segmentation of the amounts is needed to train the system it is trained at the check level

The loss function E minimized by our global train ing procedure is the Discriminative Forward criterion de scribed in Section VI the di	erence between a the for ward penalty of the constrained interpretation graph con strained by the correct label sequence and b the forward penalty of the unconstrained interpretation graph Deriva tives can be backpropagated through the entire structure although it only practical to do it down to the segmenter

C Rejecting Low Condence Checks

In order to be able to reject checks which are the most likely to carry erroneous Viterbi answers we must rate them with a condence and reject the check if this con dence is below a given threshold To compare the un normalized Viterbi Penalties of two di	erent checks would be meaningless when it comes to decide which answer we trust the most

The optimal measure of condence is the probability of the Viterbi answer given the input image As seen in Sec tion VIE given a target sequence which in this case would be the Viterbi answer the discriminative forward loss function is an estimate of the logarithm of this prob ability Therefore a simple solution to obtain a good esti mate of the condence is to reuse the interpretation graph see Figure  to compute the discriminative forward loss as described in Figure  using as our desired sequence the

Viterbi answer This is summarized in Figure  with condence  expEdforw

D Results

A version of the above system was fully implemented and tested on machineprint business checks This sys tem is basically a generic GTN engine with task specic heuristics encapsulated in the check and fprop method

As a consequence the amount of code to write was min imal mostly the adaptation of an earlier segmenter into the segmentation transformer The system that deals with handwritten or personal checks was based on earlier im plementations that used the GTN concept in a restricted way The neural network classier was initially trained on  images of character images from various origins spanning the entire printable ASCII set This contained both handwritten and machineprinted characters that had been previously size normalized at the string level Addi tional images were generated by randomly distorting the original images using simple ane transformations of the images The network was then further trained on character images that had been automatically segmented from check images and manually truthed The network was also ini tially trained to reject noncharacters that resulted from segmentation errors The recognizer was then inserted in the check reading system and a small subset of the parame ters were trained globally at the eld level on whole check images

On  business checks that were automatically catego rized as machine printed the performance was ! cor rectly recognized checks ! errors and ! rejects This can be compared to the performance of the previous sys tem on the same test set ! correct ! errors and ! rejects A check is categorized as machineprinted when characters that are near a standard position Dollar sign are detected as machine printed or when if nothing is found in the standard position at least one courtesy amount candidate is found somewhere else The improve ment is attributed to three main causes First the neural network recognizer was bigger and trained on more data

Second because of the GTN architecture the new system could take advantage of grammatical constraints in a much more ecient way than the previous system Third the

GTN architecture provided extreme exibility for testing heuristics adjusting parameters and tuning the system

This last point is more important than it seems The GTN framework separates the algorithmic part of the system from the knowledgebased part of the system allowing easy adjustments of the latter The importance of global training was only minor in this task because the global training only concerned a small subset of the parameters

An independent test performed by systems integrators in  showed the superiority of this system over other commercial Courtesy amount reading systems The system was integrated in NCRs line of check reading systems It

PROC OF THE IEEE NOVEMBER  has been elded in several banks across the US since June  and has been reading millions of checks per day since then

XI Conclusions

During the short history of automatic pattern recogni tion increasing the role of learning seems to have invari ably improved the overall performance of recognition sys tems The systems described in this paper are more ev idence to this fact Convolutional Neural Networks have been shown to eliminate the need for handcrafted fea ture extractors Graph Transformer Networks have been shown to reduce the need for handcrafted heuristics man ual labeling and manual parameter tuning in document recognition systems As training data becomes plentiful as computers get faster as our understanding of learning al gorithms improves recognition systems will rely more and more of learning and their performance will improve

Just as the backpropagation algorithm elegantly solved the credit assignment problem in multilayer neural net works the gradientbased learning procedure for Graph

Transformer Networks introduced in this paper solves the credit assignment problem in systems whose functional ar chitecture dynamically changes with each new input The learning algorithms presented here are in a sense nothing more than unusual forms of gradient descent in complex dynamic architectures with ecient backpropagation al gorithms to compute the gradient The results in this pa per help establish the usefulness and relevance of gradient based minimization methods as a general organizing prin ciple for learning in large systems

It was shown that all the steps of a document analysis system can be formulated as graph transformers through which gradients can be backpropagated Even in the nontrainable parts of the system the design philosophy in terms of graph transformation provides a clear separa tion between domainspecic heuristics eg segmentation heuristics and generic procedural knowledge the gener alized transduction algorithm

It is worth pointing out that data generating models such as HMMs and the Maximum Likelihood Principle were not called upon to justify most of the architectures and the training criteria described in this paper Gradient based learning applied to global discriminative loss func tions guarantees optimal classication and rejection with out the use of hard to justify principles that put strong constraints on the system architecture often at the expense of performances

More specically the methods and architectures pre sented in this paper o	er generic solutions to a large num ber of problems encountered in pattern recognition sys tems  Feature extraction is traditionally a xed transform generally derived from some expert prior knowledge about the task This relies on the probably incorrect assumption that the human designer is able to capture all the rele vant information in the input We have shown that the application of GradientBased Learning to Convolutional

Neural Networks allows to learn appropriate features from examples The success of this approach was demonstrated in extensive comparative digit recognition experiments on the NIST database  Segmentation and recognition of ob jects in images can not be completely decoupled Instead of taking hard seg mentation decisions too early we have used Heuristic Over

Segmentation to generate and evaluate a large number of hypotheses in parallel postponing any decision until the overall criterion is minimized  Hand truthing images to obtain segmented characters for training a character recognizer is expensive and does not take into account the way in which a whole document or sequence of characters will be recognized in particular the fact that some segmentation candidates may be wrong even though they may look like true characters Instead we train multimodule systems to optimize a global mea sure of performance which does not require time consum ing detailed handtruthing and yields signicantly better recognition performance because it allows to train these modules to cooperate towards a common goal  Ambiguities inherent in the segmentation character recognition and linguistic model should be integrated op timally Instead of using a sequence of taskdependent heuristics to combine these sources of information we have proposed a unied framework in which generalized transduction methods are applied to graphs representing a weighted set of hypotheses about the input The success of this approach was demonstrated with a commercially de ployed check reading system that reads millions of business and personal checks per day the generalized transduction engine resides in only a few hundred lines of code  Traditional recognition systems rely on many hand crafted heuristics to isolate individually recognizable ob jects The promising Space Displacement Neural Network approach draws on the robustness and eciency of Con volutional Neural Networks to avoid explicit segmentation altogether Simultaneous automatic learning of segmenta tion and recognition can be achieved with GradientBased

Learning methods

This paper presents a small number of examples of graph transformer modules but it is clear that the concept can be applied to many situations where the domain knowledge or the state information can be represented by graphs This is the case in many audio signal recognition tasks and visual scene analysis applications Future work will attempt to apply Graph Transformer Networks to such problems with the hope of allowing more reliance on automatic learning and less on detailed engineering

Appendices

A Preconditions for faster convergence

As seen before the squashing function used in our Con volutional Networks is f a  A tanhSa Symmetric functions are believed to yield faster convergence although the learning can become extremely slow if the weights are too small The cause of this problem is that in weight space the origin is a xed point of the learning dynamics and

PROC OF THE IEEE NOVEMBER  	 although it is a saddle point it is attractive in almost all directions  For our simulations we use A   and S   see   With this choice of parame ters the equalities f    and f    are satised

The rationale behind this is that the overall gain of the squashing transformation is around  in normal operat ing conditions and the interpretation of the state of the network is simplied Moreover the absolute value of the second derivative of f is a maximum at  and  which improves the convergence towards the end of the learning session This particular choice of parameters is merely a convenience and does not a	ect the result

Before training the weights are initialized with random values using a uniform distribution between Fi and Fi where Fi is the number of inputs fanin of the unit which the connection belongs to Since several connections share a weight this rule could be dicult to apply but in our case all connections sharing a same weight belong to units with identical fanins The reason for dividing by the fanin is that we would like the initial standard deviation of the weighted sums to be in the same range for each unit and to fall within the normal operating region of the sigmoid If the initial weights are too small the gradients are very small and the learning is slow If they are too large the sigmoids are saturated and the gradient is also very small The standard deviation of the weighted sum scales like the square root of the number of inputs when the inputs are independent and it scales linearly with the number of inputs if the inputs are highly correlated We chose to assume the second hypothesis since some units receive highly correlated signals

B Stochastic Gradient vs Batch Gradient

GradientBased Learning algorithms can use one of two classes of methods to update the parameters The rst method dubbed Batch Gradient  is the classical one the gradients are accumulated over the entire training set and the parameters are updated after the exact gradient has been so computed In the second method called Stochas tic Gradient  a partial or noisy gradient is evaluated on the basis of one single training sample or a small num ber of samples and the parameters are updated using this approximate gradient The training samples can be selected randomly or according to a properly randomized sequence In the stochastic version the gradient estimates are noisy but the parameters are updated much more often than with the batch version An empirical result of con siderable practical importance is that on tasks with large redundant data sets the stochastic version is considerably faster than the batch version sometimes by orders of mag nitude  Although the reasons for this are not totally understood theoretically an intuitive explanation can be found in the following extreme example Let us take an example where the training database is composed of two copies of the same subset Then accumulating the gradient over the whole set would cause redundant computations to be performed On the other hand running Stochas tic Gradient once on this training set would amount to performing two complete learning iterations over the small subset This idea can be generalized to training sets where there exist no precise repetition of the same pattern but where some redundancy is present In fact stochastic up date must be better when there is redundancy ie when a certain level of generalization is expected

Many authors have claimed that secondorder meth ods should be used in lieu of gradient descent for neu ral net training The literature abounds with recom mendations  for classical secondorder methods such as the GaussNewton or LevenbergMarquardt algorithms for QuasiNewton methods such as the BroydenFletcher

GoldfarbShanno method BFGS Limitedstorage BFGS or for various versions of the Conjugate Gradients CG method Unfortunately all of the above methods are un suitable for training large neural networks on large data sets The GaussNewton and LevenbergMarquardt meth ods require ON operations per update where N is the number of parameters which makes them impracti cal for even moderate size networks QuasiNewton meth ods require only

ON  operations per update but that still makes them impractical for large networks Limited

Storage BFGS and Conjugate Gradient require only ON operations per update so they would appear appropriate

Unfortunately their convergence speed relies on an accu rate evaluation of successive conjugate descent directions  which only makes sense in batch mode For large data sets the speedup brought by these methods over regular batch gradient descent cannot match the enormous speed up brought by the use of stochastic gradient Several au thors have attempted to use Conjugate Gradient with small batches or batches of increasing sizes   but those attempts have not yet been demonstrated to surpass a care fully tuned stochastic gradient Our experiments were per formed with a stochastic method that scales the parameter axes so as to minimize the eccentricity of the error surface

C Stochastic Diagonal LevenbergMarquardt

Owing to the reasons given in Appendix B we prefer to update the weights after each presentation of a single pat tern in accordance with stochastic update methods The patterns are presented in a constant random order and the training set is typically repeated  times

Our update algorithm is dubbed the Stochastic Diagonal

LevenbergMarquardt method where an individual learning rate step size is computed for each parameter weight before each pass through the training set   

These learning rates are computed using the diagonal terms of an estimate of the GaussNewton approximation to the

Hessian second derivative matrix This algorithm is not believed to bring a tremendous increase in learning speed but it converges reliably without requiring extensive ad justments of the learning parameters It corrects ma jor ill conditioning of the loss function that are due to the pecu liarities of the network architecture and the training data

The additional cost of using this procedure over standard stochastic gradient descent is negligible

At each learning iteration a particular parameter wk is

PROC OF THE IEEE NOVEMBER  	 updated according to the following stochastic update rule wk  wk  k Ep wk   where Ep is the instantaneous loss function for pattern p In Convolutional Neural Networks because of the weight sharing the partial derivative

Ep wk is the sum of the partial derivatives with respect to the connections that share the parameter wk  Ep wk  XijVk Ep uij  where uij is the connection weight from unit j to unit i Vk is the set of unit index pairs i j such that the connection between i and j share the parameter wk  ie uij  wk i j  Vk 

As stated previously the step sizes k are not constant but are function of the second derivative of the loss function along the axis wk  k   hkk  where is a handpicked constant and hkk is an estimate of the second derivative of the loss function E with re spect to wk  The larger hkk  the smaller the weight update

The parameter prevents the step size from becoming too large when the second derivative is small very much like the modeltrust methods and the LevenbergMarquardt methods in nonlinear optimization  The exact formula to compute hkk from the second derivatives with respect to the connection weights is hkk  XijVk XklVk E uij ukl 

However we make three approximations The rst approx imation is to drop the o	diagonal terms of the Hessian with respect to the connection weights in the above equa tion hkk  XijVk E u ij 

Naturally the terms E u ij are the average over the training set of the local second derivatives E u ij  P PXp Ep u ij 

Those local second derivatives with respect to connection weights can be computed from local second derivatives with respect to the total input of the downstream unit Ep u ij  Ep ai xj  where xj is the state of unit j and Ep ai is the second derivative of the instantaneous loss function with respect to the total input to unit i denoted ai Interestingly there is an ecient algorithm to compute those second derivatives which is very similar to the backpropagation procedure used to compute the rst derivatives   Ep ai  f aiXk u ki Ep ak  f ai Ep xi 

Unfortunately using those derivatives leads to wellknown problems associated with every Newtonlike algorithm these terms can be negative and can cause the gradient algorithm to move uphill instead of downhill Therefore our second approximation is a wellknown trick called the

GaussNewton approximation which guarantees that the second derivative estimates are nonnegative The Gauss

Newton approximation essentially ignores the nonlinearity of the estimated function the Neural Network in our case but not that of the loss function The backpropagation equation for GaussNewton approximations of the second derivatives is Ep ai  f aiXk u ki Ep ak 

This is very similar to the formula for backpropagating the rst derivatives except that the sigmoids derivative and the weight values are squared The righthand side is a sum of products of nonnegative terms therefore the lefthand side term is nonnegative

The third approximation we make is that we do not run the average in Equation  over the entire training set but run it on a small subset of the training set instead In addition the reestimation does not need to be done of ten since the second order properties of the error surface change rather slowly In the experiments described in this paper we reestimate the hkk on  patterns before each training pass through the training set Since the size of the training set is  the additional cost of reestimating the hkk is negligible The estimates are not particularly sensitive to the particular subset of the training set used in the averaging This seems to suggest that the secondorder properties of the error surface are mainly determined by the structure of the network rather than by the detailed statistics of the samples This algorithm is particularly use ful for sharedweight networks because the weight sharing creates illconditionning of the error surface Because of the sharing one single parameter in the rst few layers can have an enormous inuence on the output Consequently the second derivative of the error with respect to this pa rameter may be very large while it can be quite small for other parameters elsewhere in the network The above al gorithm compensates for that phenomenon

Unlike most other secondorder acceleration methods for backpropagation the above method works in stochastic mode It uses a diagonal approximation of the Hessian

Like the classical LevenbergMarquardt algorithm it uses a safety factor to prevent the step sizes from getting too large if the second derivative estimates are small Hence the method is called the Stochastic Diagonal Levenberg

Marquardt method

PROC OF THE IEEE NOVEMBER  	

Acknowledgments

Some of the systems described in this paper is the work of many researchers now at AT&T and Lucent Technolo gies In particular Christopher Burges Craig Nohl Troy

Cauble and Jane Bromley contributed much to the check reading system Experimental results described in sec tion III include contributions by Chris Burges Aymeric

Brunot Corinna Cortes Harris Drucker Larry Jackel Urs

M"uller Bernhard Sch"olkopf and Patrice Simard The au thors wish to thank Fernando Pereira Vladimir Vapnik

John Denker and Isabelle Guyon for helpful discussions

Charles Stenard and Ray Higgins for providing the appli cations that motivated some of this work and Lawrence R

Rabiner and Lawrence D Jackel for relentless support and encouragements

## References
  R O Duda and P E Hart Pattern Classication And Scene Analysis Wiley and Son   Y LeCun B Boser J S Denker D Henderson R E Howard W Hubbard and L D Jackel Backpropagation applied to handwritten zip code recognition Neural Computation vol  no  pp  Winter   S Seung H Sompolinsky and N Tishby Statistical mechan ics of learning from examples Physical Review A vol  pp    V N Vapnik E Levin and Y LeCun Measuring the vc dimension of a learning machine Neural Computation vol  no  pp    C Cortes L Jackel S Solla V N Vapnik and J Denker Learning curves	 asymptotic values and rate of convergence in Advances in Neural Information Processing Systems  J D Cowan G Tesauro and J Alspector Eds San Mateo CA  pp  Morgan Kaufmann  V N Vapnik The Nature of Statistical Learning Theory Springer NewYork   V N Vapnik Statistical Learning Theory John Wiley  Sons NewYork   W H Press B P Flannery S A Teukolsky and W T Vet terling Numerical Recipes The Art of Scientic Computing Cambridge University Press Cambridge   S I Amari A theory of adaptive pattern classiers IEEE Transactions on Electronic Computers vol EC pp     Ya Tsypkin Adaptation and Learning in automatic systems Academic Press   Ya Tsypkin Foundations of the theory of learning systems Academic Press   M Minsky and O Selfridge Learning in random nets in th London symposium on Information Theory London  pp   D H Ackley G E Hinton and T J Sejnowski A learning algorithm for boltzmann machines Cognitive Science vol  pp    G E Hinton and T J Sejnowski Learning and relearning in Boltzmann machines in Paral lel Distributed Processing Explorations in the Microstructure of Cognition Volume  Foundations D E Rumelhart and J L McClelland Eds MIT Press Cambridge MA   D E Rumelhart G E Hinton and R J Williams Learning internal representations by error propagation in Paral lel dis tributed processing Explorations in the microstructure of cog nition vol I pp  Bradford Books Cambridge MA   A E Jr Bryson and YuChi Ho Applied Optimal Control Blaisdell Publishing Co   Y LeCun A learning scheme for asymmetric threshold net works in Proceedings of Cognitiva Paris France  pp   Y LeCun Learning processes in an asymmetric threshold network in Disordered systems and biological organization E Bienenstock F FogelmanSoulie and G Weisbuch Eds Les Houches France  pp  SpringerVerlag  D B Parker Learninglogic Tech Rep TR Sloan School of Management MIT Cambridge Mass April   Y LeCun Modeles connexionnistes de lapprentissage con nectionist learning models PhD thesis Universite P et M Curie Paris  June   Y LeCun A theoretical framework for backpropagation in Proceedings of the 		 Connectionist Models Summer School D Touretzky G Hinton and T Sejnowski Eds CMU Pitts burgh Pa  pp  Morgan Kaufmann  L Bottou and P Gallinari A framework for the cooperation of learning algorithms in Advances in Neural Information Pro cessing Systems D Touretzky and R Lippmann Eds Denver  vol  Morgan Kaufmann  C Y Suen C Nadal R Legault T A Mai and L Lam Computer recognition of unconstrained handwritten numer als Proceedings of the IEEE Special issue on Optical Char acter Recognition vol  no  pp  July   S N Srihari Highperformance reading machines Proceed ings of the IEEE Special issue on Optical Character Recogni tion vol  no  pp  July   Y LeCun L D Jackel B Boser J S Denker H P Graf I Guyon D Henderson R E Howard and W Hubbard Handwritten digit recognition	 Applications of neural net chips and automatic learning IEEE Communication pp   November  invited paper  J Keeler D Rumelhart and W K Leow Integrated seg mentation and recognition of handprinted numerals in Neu ral Information Processing Systems R P Lippmann J M Moody and D S Touretzky Eds vol  pp  Morgan Kaufmann Publishers San Mateo CA   Ofer Matan Christopher J C Burges Yann LeCun and John S Denker Multidigit recognition using a space dis placement neural network in Neural Information Processing Systems J M Moody S J Hanson and R P Lippman Eds  vol  Morgan Kaufmann Publishers San Mateo CA  L R Rabiner A tutorial on hidden Markov models and se lected applications in speech recognition Proceedings of the IEEE vol  no  pp  February   H A Bourlard and N Morgan CONNECTIONIST SPEECH RECOGNITION A Hybrid Approach Kluwer Academic Pub lisher Boston   D H Hubel and T N Wiesel Receptive elds binocular interaction and functional architecture in the cat s visual cor tex Journal of Physiology London vol  pp    K Fukushima Cognitron	 A selforganizing multilayered neu ral network Biological Cybernetics vol  no  pp  November   K Fukushima and S Miyake Neocognitron	 A new algorithm for pattern recognition tolerant of deformations and shifts in position Pattern Recognition vol  pp    M C Mozer The perception of multiple objects A connec tionist approach MIT PressBradford Books Cambridge MA   Y LeCun Generalization and network design strategies in Connectionism in Perspective R Pfeifer Z Schreter F Fogel man and L Steels Eds Zurich Switzerland  Elsevier an extended version was published as a technical report of the University of Toronto  Y LeCun B Boser J S Denker D Henderson R E Howard W Hubbard and L D Jackel Handwritten digit recognition with a backpropagation network in Advances in Neural In formation Processing Systems  NIPS	 David Touretzky Ed Denver CO  Morgan Kaufmann  G L Martin Centeredob ject integrated segmentation and recognition of overlapping handprinted characters Neural Computation vol  no  pp    J Wang and J Jean Multiresolution neural networks for om nifont character recognition in Proceedings of International Conference on Neural Networks  vol III pp   Y Bengio Y LeCun C Nohl and C Burges Lerec	 A NNHMM hybrid for online handwriting recognition Neural Computation vol  no    S Lawrence C Lee Giles A C Tsoi and A D Back Face recognition	 A convolutional neural network approach IEEE PROC OF THE IEEE NOVEMBER  Transactions on Neural Networks vol  no  pp    K J Lang and G E Hinton A time delay neural network architecture for speech recognition Tech Rep CMUCS  CarnegieMellon University Pittsburgh PA   A H Waibel T Hanazawa G Hinton K Shikano and K Lang Phoneme recognition using timedelay neural net works IEEE Transactions on Acoustics Speech and Signal Processing vol  pp  March   L Bottou F Fogelman P Blanchet and J S Lienard Speaker independent isolated digit recognition	 Multilayer perceptron vs dynamic time warping Neural Networks vol  pp    P Ha ner and A H Waibel Timedelay neural networks embedding time alignment	 a performance analysis in EU ROSPEECH nd European Conference on Speech Commu nication and Technology Genova Italy Sept   I Guyon P Albrecht Y LeCun J S Denker and W Hub bard Design of a neural network character recognizer for a touch terminal Pattern Recognition vol  no  pp     J Bromley J W Bentz L Bottou I Guyon Y LeCun C Moore E S ackinger and R Shah Signature verica tion using a siamese time delay neural network International Journal of Pattern Recognition and Articial Intel ligence vol  no  pp  August   Y LeCun I Kanter and S Solla Eigenvalues of covariance matrices	 application to neuralnetwork learning Physical Review Letters vol  no  pp  May   T G Dietterich and G Bakiri Solving multiclass learning problems via errorcorrecting output codes Journal of Arti cial Intel ligence Research vol  pp    L R Bahl P F Brown P V de Souza and R L Mercer Maximum mutual information of hidden Markov model pa rameters for speech recognition in Proc Int Conf Acoust Speech Signal Processing  pp   L R Bahl P F Brown P V de Souza and R L Mercer Speech recognition with continuousparameter hidden Markov models Computer Speech and Language vol  pp    B H Juang and S Katagiri Discriminative learning for min imum error classication IEEE Trans on Acoustics Speech and Signal Processing vol  no  pp  December   Y LeCun L D Jackel L Bottou A Brunot C Cortes J S Denker H Drucker I Guyon U A Muller E S ackinger P Simard and V N Vapnik Comparison of learning al gorithms for handwritten digit recognition in International Conference on Articial Neural Networks F Fogelman and P Gallinari Eds Paris  pp  EC  Cie  I Guyon I Poujaud L Personnaz G Dreyfus J Denker and Y LeCun Comparing di erent neural net architectures for classifying handwritten digits in Proc of IJCNN Washing ton DC  vol II pp  IEEE  R Ott construction of quadratic polynomial classiers in Proc of International Conference on Pattern Recognition  pp  IEEE  J Sch urmann A multifont word recognition system for postal address reading IEEE Transactions on Computers vol C no  pp  August   Y Lee Handwritten digit recognition using knearest neigh bor radialbasis functions and backpropagation neural net works Neural Computation vol  no  pp    D Saad and S A Solla Dynamics of online gradient de scent learning for multilayer neural networks in Advances in Neural Information Processing Systems David S Touretzky Michael C Mozer and Michael E Hasselmo Eds  vol  pp  The MIT Press Cambridge  G Cybenko Approximation by superpositions of sigmoidal functions Mathematics of Control Signals and Systems vol  no  pp    L Bottou and V N Vapnik Local learning algorithms Neu ral Computation vol  no  pp    R E Schapire The strength of weak learnability Machine Learning vol  no  pp    H Drucker R Schapire and P Simard Improving perfor mance in neural networks using a boosting algorithm in Ad vances in Neural Information Processing Systems S J Han son J D Cowan and C L Giles Eds San Mateo CA  pp  Morgan Kaufmann  P Simard Y LeCun and Denker J E!cient pattern recog nition using a new transformation distance in Advances in Neural Information Processing Systems S Hanson J Cowan and L Giles Eds vol  Morgan Kaufmann   B Boser I Guyon and V Vapnik A training algorithm for optimal margin classiers in Proceedings of the Fifth Annual Workshop on Computational Learning Theory  vol  pp   C J C Burges and B Schoelkopf Improving the accuracy and speed of support vector machines in Advances in Neural Information Processing Systems  M Jordan M Mozer and T Petsche Eds  The MIT Press Cambridge  Eduard S ackinger Bernhard Boser Jane Bromley Yann Le Cun and Lawrence D Jackel Application of the ANNA neu ral network chip to highspeed character recognition IEEE Transaction on Neural Networks vol  no  pp  March   J S Bridle Probabilistic interpretation of feedforward classi cation networks outputs with relationship to statistical pattern recognition in Neurocomputing Algorithms Architectures and Applications F Fogelman J Herault and Y Burnod Eds Les Arcs France  Springer  Y LeCun L Bottou and Y Bengio Reading checks with graph transformer networks in International Conference on Acoustics Speech and Signal Processing Munich  vol  pp  IEEE  Y Bengio Neural Networks for Speech and Sequence Recogni tion International Thompson Computer Press London UK   C Burges O Matan Y LeCun J Denker L Jackel C Ste nard C Nohl and J Ben Shortest path segmentation	 A method for training a neural network to recognize character strings in International Joint Conference on Neural Net works Baltimore  vol  pp   T M Breuel A system for the o line recognition of hand written text in ICPR IEEE Ed Jerusalem   pp   A Viterbi Error bounds for convolutional codes and an asymptotically optimum decoding algorithm IEEE Trans actions on Information Theory pp  April   Lippmann R P and Gold B Neuralnet classiers useful for speech recognition in Proceedings of the IEEE First Interna tional Conference on Neural Networks San Diego June  pp   H Sakoe R Isotani K Yoshida K Iso and T Watan abe Speakerindependent word recognition using dynamic programming neural networks in International Conference on Acoustics Speech and Signal Processing Glasgow  pp   J S Bridle Alphanets	 a recurrent "neural network archi tecture with a hidden markov model interpretation Speech Communication vol  no  pp    M A Franzini K F Lee and A H Waibel Connectionist viterbi training	 a new hybrid method for continuous speech recognition in International Conference on Acoustics Speech and Signal Processing Albuquerque NM  pp   L T Niles and H F Silverman Combining hidden markov models and neural network classiers in International Con ference on Acoustics Speech and Signal Processing Albu querque NM  pp   X Driancourt and L Bottou MLP LVQ and DP	 Compari son  cooperation in Proceedings of the International Joint Conference on Neural Networks Seattle  vol  pp    Y Bengio R De Mori G Flammia and R Kompe Global optimization of a neural networkhidden Markov model hy brid IEEE Transactions on Neural Networks vol  no  pp    P Ha ner and A H Waibel Multistate timedelay neural networks for continuous speech recognition in Advances in Neural Information Processing Systems  vol  pp   Morgan Kaufmann San Mateo  Y Bengio  P Simard and P Frasconi Learning longterm dependencies with gradient descent is di!cult IEEE Trans actions on Neural Networks vol  no  pp  March  Special Issue on Recurrent Neural Network PROC OF THE IEEE NOVEMBER    T Kohonen G Barna and R Chrisley Statistical pattern recognition with neural network	 Benchmarking studies in Proceedings of the IEEE Second International Conference on Neural Networks San Diego  vol  pp   P Ha ner Connectionist speech recognition with a global MMI algorithm in EUROSPEECH rd European Confer ence on Speech Communication and Technology Berlin Sept   J S Denker and C J Burges Image segmentation and recog nition in The Mathematics of Induction  Addison Wes ley  L Bottou Une Approche theorique de lApprentissage Connex ionniste Applications a la Reconnaissance de la Parole PhD thesis Universite de Paris XI  Orsay cedex France   M Rahim Y Bengio and Y LeCun Discriminative feature and model design for automatic speech recognition in Proc of Eurospeech Rhodes Greece   U Bodenhausen S Manke and A Waibel Connectionist ar chitectural learning for high performance character and speech recognition in International Conference on Acoustics Speech and Signal Processing Minneapolis  vol  pp   F Pereira M Riley and R Sproat Weighted rational trans ductions and their application to human language processing in ARPA Natural Language Processing workshop   M Lades J C Vorbr uggen J Buhmann and C von der Mals burg Distortion invariant ob ject recognition in the dynamic link architecture IEEE Trans Comp vol  no  pp    B Boser E S ackinger J Bromley Y LeCun and L Jackel An analog neural network processor with programmable topol ogy IEEE Journal of SolidState Circuits vol  no  pp  December   M Schenkel H Weissman I Guyon C Nohl and D Hender son Recognitionbased segmentation of online handprinted words in Advances in Neural Information Processing Systems  S J Hanson J D Cowan and C L Giles Eds Denver CO  pp   C Dugast L Devillers and X Aubert Combining TDNN and HMM in a hybrid system for improved continuousspeech recognition IEEE Transactions on Speech and Audio Pro cessing vol  no  pp    Ofer Matan Henry S Baird Jane Bromley Christopher J C Burges John S Denker Lawrence D Jackel Yann Le Cun Ed win P D Pednault William D Sattereld Charles E Stenard and Timothy J Thompson Reading handwritten digits	 A ZIP code recognition system Computer vol  no  pp  July   Y Bengio and Y Le Cun Word normalization for online handwritten word recognition in Proc of the International Conference on Pattern Recognition IAPR Ed Jerusalem  IEEE  R Vaillant C Monrocq and Y LeCun Original approach for the localization of ob jects in images IEE Proc on Vision Image and Signal Processing vol  no  pp  August   R Wolf and J Platt Postal address block location using a convolutional locator network in Advances in Neural Infor mation Processing Systems  J D Cowan G Tesauro and J Alspector Eds  pp  Morgan Kaufmann Pub lishers San Mateo CA  S Nowlan and J Platt A convolutional neural network hand tracker in Advances in Neural Information Processing Sys tems  G Tesauro D Touretzky and T Leen Eds San Ma teo CA  pp  Morgan Kaufmann  H A Rowley S Baluja and T Kanade Neural network based face detection in Proceedings of CVPR  pp  IEEE Computer Society Press  E Osuna R Freund and F Girosi Training support vector machines	 an application to face detection in Proceedings of CVPR  pp  IEEE Computer Society Press  H Bourlard and C J Wellekens Links between Markov mod els and multilayer perceptrons in Advances in Neural Infor mation Processing Systems D Touretzky Ed Denver  vol  pp  MorganKaufmann  Y Bengio R De Mori G Flammia and R Kompe Neu ral network  gaussian mixture hybrid for speech recognition or density estimation in Advances in Neural Information Processing Systems  J E Moody S J Hanson and R P Lippmann Eds Denver CO  pp  Morgan Kauf mann  F C N Pereira and M Riley Speech recognition by compo sition of weighted nite automata in FiniteState Devices for Natural Langue Processing Cambridge Massachusetts  MIT Press  M Mohri Finitestate transducers in language and speech processing Computational Linguistics vol  no  pp     I Guyon M Schenkel and J Denker Overview and syn thesis of online cursive handwriting recognition techniques in Handbook on Optical Character Recognition and Document Image Analysis P S P Wang and Bunke H Eds  World Scientic  M Mohri and M Riley Weighted determinization and min imization for large vocabulary recognition in Proceedings of Eurospeech  Rhodes Greece September  pp   Y Bengio and P Frasconi An inputoutput HMM architec ture in Advances in Neural Information Processing Systems G Tesauro D Touretzky and T Leen Eds  vol  pp  MIT Press Cambridge MA  Y Bengio and P Frasconi InputOutput HMMs for sequence processing IEEE Transactions on Neural Networks vol  no  pp    M Mohri F C N Pereira and M Riley A rational design for a weighted nitestate transducer library Lecture Notes in Computer Science Springer Verlag   M Rahim C H Lee and B H Juang Discriminative ut terance verication for connected digits recognition IEEE Trans on Speech  Audio Proc vol  pp    M Rahim Y Bengio and Y LeCun Discriminative feature and model design for automatic speech recognition in Eu rospeech  Rhodes Greece  pp   S Bengio and Y Bengio An EM algorithm for asynchronous inputoutput hidden Markov models in International Con ference On Neural Information Processing L Xu Ed Hong Kong  pp   C Tappert C Suen and T Wakahara The state of the art in online handwriting recognition IEEE Transactions on Pattern Analysis and Machine Intel ligence vol  no  pp    S Manke and U Bodenhausen A connectionist recognizer for online cursive handwriting recognition in International Con ference on Acoustics Speech and Signal Processing Adelaide  vol  pp   M Gilloux and M Leroux Recognition of cursive script amounts on postal checks in European Conference dedicated to Postal Technologies Nantes France June  pp    D Guillevic and C Y Suen Cursive script recognition applied to the processing of bank checks in Int Conf on Document Analysis and Recognition Montreal Canada August  pp   L Lam C Y Suen D Guillevic N W Strathy M Cheriet K Liu and J N Said Automatic processing of information on checks in Int Conf on Systems Man  Cybernetics Vancouver Canada October  pp   C J C Burges J I Ben J S Denker Y LeCun and C R Nohl O line recognition of handwritten postal words using neural networks Int Journal of Pattern Recognition and Ar ticial Intel ligence vol  no  pp   Special Issue on Applications of Neural Networks to Pattern Recognition I Guyon Ed  Y LeCun Y Bengio D Henderson A Weisbuch H Weiss man and Jackel L Online handwriting recognition with neural networks	 spatial representation versus temporal repre sentation in Proc International Conference on handwriting and drawing  Ecole Nationale Superieure des Telecommu nications  U M uller A Gunzinger and W Guggenb uhl Fast neural net simulation with a DSP processor array IEEE Trans on Neural Networks vol  no  pp    R Battiti First and secondorder methods for learning	 Be tween steepest descent and newton s method Neural Com putation vol  no  pp    A H Kramer and A SangiovanniVincentelli E!cient par allel learning algorithms for neural networks in Advances in Neural Information Processing Systems DS Touretzky Ed PROC OF THE IEEE NOVEMBER  Denver   vol  pp  Morgan Kaufmann San Mateo  M Moller Ecient Training of FeedForward Neural Net works PhD thesis Aarhus University Aarhus Denmark   S Becker and Y LeCun Improving the convergence of back propagation learning with secondorder methods Tech Rep CRGTR University of Toronto Connectionist Research Group September  Yann LeCun Yann LeCun received a Dipl#ome d Ingenieur from the Ecole Superieure d Ingenieur en Electrotechnique et Electron ique Paris in  and a PhD in Computer Science from the Universite Pierre et Marie Curie Paris in  during which he proposed an early version of the backpropagation learn ing algorithm for neural networks He then joined the Department of Computer Science at the University of Toronto as a research asso ciate In  he joined the Adaptive Systems Research Department at ATT Bell Laboratories in Holmdel NJ where he worked among other thing on neural networks machine learning and handwriting recognition Following ATT s second breakup in  he became head of the Image Processing Services Research Department at ATT LabsResearch He is serving on the board of the Machine Learning Journal and has served as associate editor of the IEEE Trans on Neural Networks He is general chair of the Machines that Learn workshop held every year since  in Snowbird Utah He has served as program cochair of IJCNN  INNC  NIPS  and  He is a member of the IEEE Neural Network for Signal Processing Technical Committee He has published over  technical papers and book chapters on neural networks machine learning pattern recognition handwriting recognition document understanding image processing VLSI design and information theory In addition to the above topics his current interests include videobased user interfaces image compression and contentbased indexing of multimedia material L eon Bottou Leon Bottou received a Dipl#ome from Ecole Polytechnique Paris in  a Magist$ere en Mathematiques Fondamentales et Appliquees et Informatiques from Ecole Nor male Superieure Paris in  and a PhD in Computer Science from Universite de Paris Sud in  during which he worked on speech recognition and proposed a framework for stochastic gradient learning and global train ing He then joined the Adaptive Systems Re search Department at ATT Bell Laboratories where he worked on neural network statistical learning theory and local learning algorithms He returned to France in  as a research engineer at ONERA He then became chairman of Neuristique SA a company making neural network simulators and tra!c forecast ing software He eventually came back to ATT Bell Laboratories in  where he worked on graph transformer networks for optical character recognition He is now a member of the Image Process ing Services Research Department at ATT LabsResearch Besides learning algorithms his current interests include arithmetic coding image compression and indexing Yoshua Bengio Yoshua Bengio received his BEng in electrical engineering in  from McGill University He also received a MSc and a PhD in computer science from McGill University in  and  respectively In  he was a postdoctoral fellow at the Massachusetts Institute of Technology In  he joined ATT Bell Laboratories which later became ATT LabsResearch In  he joined the faculty of the computer science de partment of the Universite de Montreal where he is now an associate professor Since his rst work on neural net works in  his research interests have been centered around learn ing algorithms especially for data with a sequential or spatial nature such as speech handwriting and timeseries Patrick Haner Patrick Ha ner graduated from Ecole Polytechnique Paris France in  and from Ecole Nationale Superieure des Telecommunications ENST Paris France in  He received his PhD in speech and sig nal processing from ENST in  In  and  he worked with Alex Waibel on the design of the TDNN and the MSTDNN ar chitectures at ATR Japan and Carnegie Mel lon University From  to  as a re search scientist for CNETFranceTelecom in Lannion France he developed connectionist learning algorithms for telephone speech recognition In  he joined ATT Bell Labora tories and worked on the application of Optical Character Recognition and transducers to the processing of nancial documents In  he joined the Image Processing Services Research Department at ATT LabsResearch His research interests include statistical and connec tionist models for sequence recognition machine learning speech and image recognition and information theory
