# Net2Net
End-to-End implementation of Net2Net Model as described in https://arxiv.org/abs/1511.05641

# What is this?

The no. of hidden layers and no. of units in each of these hidden layers are some of the hyper parameters in designing a neural network. A good network will neither overfit or underfit the training set. Various empirical methods exits to find an optimal architecture design that depends on the variability of data in training set. 

## Hyper-parameter search
One of them is random search in the hyper parameter search space. In this method a new network is trained with different set of hyper-parameters at random points in the hyper-parameter space. The function that approximates training loss or test accuracy to these hyper-parameters may be obtained.  As you have already guessed, it's resource hungry and requires lot of time to find the optimal network design.

## Why?
This optimal design is essential for latency and power consumption requirements in mobile devices and remote sensing systems. However, over time as new training data is added, this optimal network design becomes small for the dataset. In that case, typically, designers train a bigger network from zero with the larger dataset. 

The paper mentioned in the description provides an approach where smaller network can grow gradually both in depth and width, with the training set without the need to relearn the weights. So a network can grow with training data and arriving at an optimal network design will no longer require relearning at each step. The larger network will acquire the knowledge from a smaller network and begin its training from that phase. It allows the network to scale with data.


## Typical Use-cases

- Grow network overtime as the training data grows
- A form of regularization; start from a simple network and grow until the network is about to overfit. This way you end up with optimal design everytime.
- Knowledge transfer, transfer learned weights to another larger network. Output of this larger network is same as the smaller one.

## Watch out

- This repo only conmtains implementation of net2net model for dense layers, one could extend the same logic to convolution and pooling layers.
