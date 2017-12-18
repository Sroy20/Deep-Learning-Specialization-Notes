# Coursera-Deep-Learning-Specialization
Things about the DL specialization from Coursera.

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

1. 







