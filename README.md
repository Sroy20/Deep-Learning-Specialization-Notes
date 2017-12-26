This repository helps me in remembering the materials presented in Coursera's Deep Learning specialization. I have tried to frame the key concepts presented in each lecture as a set of questions.

# Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization

## Week 1 - Practical Aspects of Deep Learning

### Setting up your Machine Learning Application

#### Train/Dev/Test sets
1. What are train/dev/test sets?
1. How do you choose train, test, and dev sets? (Previous era vs. this era; smaller dataset vs. larger dataset)
1. What happens when the data distribution of train, test, and dev sets are different? (test and dev. data distribution have to be same)

#### Bias/Variance 
1. Describe bias and variance intuitively. What is the bias variance trade-off? (Trick question since people don't talk about the bias variance trade-off nowadays. All 4 scenarios can be seen.)
1. Give examples for each scenario.
1. How do you decide which 4 of the possible scenarios you are in? (Check train error, compare with Bayes error - decide high bias/low bias; then check dev error, compare with train error)

#### Basic recipe for Machine Learning
1. How to approach in reducing bias and variance? (Talk about the basic ML recipe)
1. Why is the bias-variance tradeoff not important anymore in the era of DL? (Because in the earlier era of ML, almost all techniques that increased one decreased the other. Now we have techniques that can almost solely affect one of them. For example, bigger network solely reduces bias and more data solely reduces variance)

### Regularizing your neural network

#### Regularization

1. What is L2 regularization? 
1. Do we have to regularize both weights are biases? (For each neuron, most parameters are in the weight vector since it's high dimensional. So regularizing bias is not required. Can be omitted.)
1. What is the L1 regularization? What are the consequences? (sparse network - most weights zero) When is it typically used? (compressing neural networks, but doesn't work much in pratice)
1. What is the regularization parameter? How do you set it? (another hyperparameter - so use dev set)
1. What is the formula for L2 regularization for a multi-layer neural network?
1. What is Frobenius norm? (L2 norm of a matrix; basically sum of squares of all elements)
1. How does the weight update rule get changed when L2 regularization is added?
1. Why is L2 regularization called the weight decay?

### Why regularization reduces overfitting?

1. Intuitively why does regularization prevent overfitting? (automatic pruning of network; classicial reason of operating in the linear region of an activation function)
1. Difference between cost function and loss. (loss is a part of CF)

### Dropout regularization

1. What is dropout? Why does it help in regularization?
1. How do you implement dropout? (Inverted dropout implementation)
1. How does backprop happen in a dropout layer?
1. Why is the inversion with keep_prob required during training? (Reducing the computation at test time- just no dropout network is fine)

### Understanding dropout

1. Why does dropout work? (From the network perspective - smaller network at each iteration; from a neuron's perspective - spreading out its weights since inputs can randomly go away -> effect is shrinking the L2 norm of weights i.e. adaptive form of L2 regularization
1. How do you choose the dropout factor for layers? (keep prob smaller for layers with more parameters i.e. layers that have more chance of overfitting)
1. What is a downside of dropout? (cost function of J is not well-defined)

### Other regularization methods

1. What are other regularization tricks? (data augmentation, early stopping, 
1. What is data augmentation and why is it used? (flip horizontal, random zooming., random rotations, distortions depending on the problem; Doesn't add much informarion but might regularize)
1. What is early stopping and why is it used? Why does it work? (weights start from close to zero and then increases; early stopping chooses weights at the mid range)
1. What is the advantage and disadvantage of early stopping? (adv. = unlike L2 norm which requires multiple passes of SGD to find the hyperparameter lamda early stopping requires only 1 pass. disadv. = combines the optimization and regularization part)
1. What is orthogonalization in the context of ML?

## Setting up your optimization problem

### Normalizaing inputs

1. How to normalize input? (for each dimension subtract mean and dividie by std. dev., remember that test set has to see the same transformation)
1. Why do we normalize the input data? (otherwise elongatated cost function so magnitude of weights for differnt dimensions are very dissimilar -> slower to optimize )
1. When do we normalize the input data? (When i/p features come from very different ranges, although doing it doesn't do any harm)

### Vanishing/Exploding gradients

1. Explain the phenomenon in case of vanishing/exploding gradients. (explain the phenomenon and talk about why it is not very important anymore in case of feed-forward networks. But still important for RNNs.) (Very interesting read - https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b for the case of sigmoids and tanh)(It is better to talk about the classical problems first and then come to modern probelms and talk about the random block initialization paper and why it is less of an issue.)(Networks facing this problem in order of severity -> RNN, FFN w sigmoid/tanh, Mod. ReLU).


### Weight initialization for deep networks

1. How do you intialize the weight of any layer of a deep network? Why? Discuss the changes for different activation functions (ReLU and tanh).
1. What is He intialization? Xavier intialization? Glorot and Bengio initialization? (Also an interesting paper: https://arxiv.org/pdf/1704.08863.pdf)

### Numerical approximation of gradients

1. Should you take a two sided difference or one sided difference while calculating gradients? (Accuracy of calculation: two-sided error is in O(ephsilon^2) whereas one-sided is in O(ephsilon). Although, two-sided is much more accurate, one-sided is faster.)

### Gradient checking

1. Why is gradient checking necessary?
1. How to perform gradient checking for a neural network? (Give formula) 
1. What is a good ballpark value of epshilon that can be used for calculating gradients?

### Gradient checking implementation notes

1. What to do if your algorithm fails gradient checking?
1. Should you using grad check throughout training? If not, why not?
1. How do you handle regularization during grad check? (IF L2 or additive regularzation just add that; but doesn't work with dropout)
1. How do you do grad check for a neural network that has dropout? (Just turn off dropout, check and it the algo passes grad check, turn on dropout)
1. Do we do grad check before training? (Yes, and also after some training epochs have passed.)


## Week 2 - Optimization Algorithms

### Mini-batch gradient descent

1. What is mini-batch gradient descent? Compare it with batch gradient descent (stable but too long per iteration)? Also compare with stochastic gradient descent (very noise, doesn't converge, and loose speed up from vectorization) (Also remember this paper - https://arxiv.org/pdf/1609.04836.pdf)
1. What are the advantages of mini-batch GD? (Can exploit vectorized implementations, more gradient descent steps than batch GD, less noisy than Stochastic GD)
1. How to choose the size of the mini-batch? (In between 1 and m; Guidelines: If small training set (<2000) just use m i.e. batch GD, for bigger training set typical mini-batch sizes: 64, 128, 256, 512 (power of 2 since makes use of computer architecture), Make sure each minibatch fits GPU/CPU memory, mini-batch can also be a hyperparameter - explore powers of 2).

### Exponentially weighted averages

1. Explain the general exponential moving average formula. (V_t = beta * (V_t-1) + (1-beta) * theta_t))
1. Approximately, how many previous steps does it average upon? (1/(1-beta))

### Understanding exponentially weighted averages

1. Why the name "exponentially"?
1. What is the advantage of this technique? (Computationally efficient)

### Bias correction in exponentially weighted averages

1. Why is bias correction required? (Starts with zero, the estimate for initial phases are inaccurate)
1. How is bias correction done? 

### Gradient descent with momentum

1. What is the core idea of this algorithm? (calculate exponentially weighted moving averages of gradients and use this to update weights)
1. Why can't larger learning rates might sometime create problem in GD? (If oscillations happen in some directions, they would get intensified)
1. What is the formula for this algorithm?
1. Intuitively, why does momentum work? (oscillations are averaged out to some extent; so larger learning rates can be used -> bigger steps towards the minimum -> speeding up the optimization process.)
1. What is a common value of beta that is used?

### RMSprop

1. What is the RMSprop formula? Explain it intuitively.
1. How to add numerical stability to the RMSprop algorithm?
1. What is the advantage of RMSprop? (Averaging out oscillations -> allows us to use a larger learning rate -> speeding up the optimization process)

### Adam optimization algorithm

1. What is the core idea of Adam? (Combining gradient descent with momentum and RMSprop)
1. Write the formula.
1. How are the hyperparameters selected for Adam? (alpha = needs to be tuned, beta_1 = 0.9, beta_2 = 0.999, ephsilon = 1e-8; typically only alpha is tuned)

### Learning rate decay

1. What is the intuition behind slowly reducing the learning rate decay?
1. How to implement the learning rate decay? (staircase or formulae as a function of epoch - some of them introduce new hyperparameters)

### The problem of local optima

1. How has the concept of local minima evolved in modern deep learning? (lots of intuitions on lower dimensional spaces doesn't translate to higher dimensions; For example, local optima rarely occur in high dimensional spaces, saddle points are more likely . Derivatives are also zero in a saddle point.)
1. What is the challenge faced by optimization in deep learning? (Problem of plateau - makes learning pretty slow - Adam, RMSprop can really help speed up learning here.)

## Week 3 - Hyperparameter tuning, Batch Normalization, and Programming Frameworks

### Hyperparameter tuning

#### Tuning process

1. What are some of the hyperparameters used in deep learning? (learning rate alpha, momentum beta, Adam optimization hparameters, # layers, # units in layers, learning rate decay, mini-batch size; Acc. to Andrew Ng the most important parameter to tune is alpha, followed by beta, mini-batch-size, and # hidden units; next comes # layers and learning rate decay. Adam hparameters beta_1, beta_2 and ephsilon are almost never tuned)
1. How to select values of hyperparameters to explore? 
1. Why is grid search better than random search? (Random search -> Consider a two hyperparameters situation -> If one of them doesn't affect the performance much, then ideally in a nxn grid search only n values of the other one has been explored and this means effectively only n points in total have been explored)
1. What is coarse-to-fine sampling scheme?

#### Using an appropriate scale to pick hyperparameters

1. How do you choose the range of hyperparameter exploration? (Depends on the sensitivty of performance on the hyperparameter at hand; # units in hidden layers (can be searched in a linear scale), learning rate (searched in a logarithmic scale))
1. What can be a range to search the # hidden units or # hidden layers and how to search in that range?
1. What can be a range to search the learning rate alpha and how to search in that range?
1. What can be a range to search the momentum term beta and how to search in that range? (Change to 1 - beta and repeat process for above)

#### Hyperparameters tuning in practice: Pandas vs Caviar

1. In general, what are two ways to choose hyperparameters? (babysit one model or try multiple models in parallel)

### Batch Normalization

#### Normalizing activations in a network

1. What are the advantages of BNORM? (makes hyperparameter search much easier, makes neural network much robust to the choice of hyperparameters -> networks tend to perform well for a wide range of hyperparameters, makes training of deeper networks easier)
1. What is the high level idea of BNORM? (We know that normalizing inputs makes learning faster because it changes the contours to a more round-ish shape by enforcing input dimensions to be in similar range. BNORM extends this idea of all layers.)
1. Should you normalize the output of activation or normalize before the activation layer? (Debate - but before activation is much more often)
1. Write the BNORM formula. (First convert to zero mean and std. dev. of 1, then add hyperparameters gamma and beta to learn the mean and std. dev. -> this makes the mean and std. dev. learnable) (Automatically learns the range in which the data to put into)

#### Fitting batch norm into a neural network

1. Draw the signal flow graph of a NN with BNORM along with proper hyperparameters.
1. Can we remove the biases for layers that we are using BNORM? (Yes. the first step of BNORM is to subtract the mean, that automatically cancels any constant (bias) added. So while using BNORM, biases can be permanently set to zeros.)
1. How do you implement gradient descent with BNORM?

#### Why does batch normalization work?

1. Why batch norm works? (Reason 1: Scaling of different dimensions to the same range thereby making the optimization contours more rounded. Reason 2: makes later layers more robust to weight changes in the earlier layers. The later layers face covriate shift. The values of their input changes during training. BNORM at least keept their mean and std. dev. constant. Reason 3. Regularization effect. Since mean and std. dev. is calculated on a mini-batch, it is noisy. This adds some noise on the hidden layer output. BNORM offers both multiplicative and additive noise. Note that in contrast, dropout only adds multiplicative noise but it is much more powerful. Increasing the mini-batch size decreases the regularizing effect.)
1. What is covariate shift? (If we learn X-> y, then if the input or output changes then the classifier has to be retrained.)

#### Batch norm at test time

1. Why can't we employ the same method mini-batch method for testing?
1. How do you handle batch norm during test time? (Get an estimate of mean and variance by keeping an exponentially weighted average of the means and variance calculated from all the mini-batches) 

### Multi-class classification



# Convolutional Neural Networks

## Week 1 - Foundation of CNNs

### Computer Vision

1. What are different types of Computer Vision problems? (Image classification, object detetion, neural art transfer, etc.)
1. What is the primary advantage of a Convolution layer? (for bigger real-world images -> lesser parameters -> reduce chance of overfitting and computationally feasible)

### Edge detection example

1. Intuitively explain what features do progressive layers of a neural network extract? (gradually higher level features: edges, parts of objecte, etc.)
1. Describe the forward convolution operation.
1. Can you explain for a convolution layer works as a edge detector? (vertical edge detector)

### More edge detection

1. How to detect dark-to-bright and bright-to-dark vertical edges?
1. Construct a simple vertical edge detector. Now construct a simple horizontal edge detector.
1. What is a Sobel filter? (more weight on the central element on first and third column)
1. What is a Scharr filter?
1. What do we not use hand-coded filters anymore? (train them using back-propagation, better chance of capturing the statistics of the data; also can learn slanted edge detectors)

### Padding

1. If image size is nxn and filter size is fxf, what is the dimension of the output image?
1. What are the problems of the shrinking output size? (pixels around the edges of the input don't get convolved much i.e. some information is thrown away; for deeper layers it is difficult to keep track)
1. How to solve the above problem? (Padding)
1. How to calculate how many pixels to pad on each side of the input?
1. What is the meaning of "valid" and "same" convolution in padding?
1. Why is the filter dimension _f_ typically odd in Computer Vision? (1. the padding _p = (f-1)/2_ is symmetric; 2. filter has a central position)

### Strided convolutions

1. When input size is nxn, filter size fxf, padding of p is done on all sides, and stride is s, what is the size of the output image?
1. Cross correlation vs. convolution in math vs. deep learning? (No horizontal and vertical flipping before point-wise multiplication)

### Convolutions over volume

1. How does convolution over volume work? If your input is 6x6x3 and filter is 3x3x3, what is the output size? (4x4)
1. What does each filter learn? (a particular type of edge detector feature)

### One layer of a convolutional neural network

1. Where and how is bias added to a CNN?
1. Why is a convolution layer less prone to overfitting?
1. For a convolutional layer _l_ write all the relevant equations for forward pass.

### Simple convolutional network example

1. What are the different types of layers in a convolutional neural network?

### Pooling layers

1. What are the advantages of pooling layers? (1. reduces size of representation to speed up computation 2. makes some of the features it detects a bit more robust)
1. What is max-pooling? What is average-pooling? Which one is more used? In what context is average-pooling still used?
1. What is the size of the output of a max-pooling layer?
1. How does backprop in the pooling layers?

### CNN Example

1. How to count the number of layers in a CNN? (two school of thoughts: number of trainable layers or all)

### Why convolutions?

1. What are the advantages of convolutional layers? (1. parameter sharing - a feature detector that is useful in one part of the image might also be useful in other parts 2. sparsity of connections - each output value depends on only a small number of inputs, less prone to overfitting 3. Translation invariance)
1. 










