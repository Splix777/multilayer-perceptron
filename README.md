Project in progress, check back soon for updates...

<h1 align="center">Multilayer-Perceptron</h1>


<div align="center">
<img src="images/logo.png" alt="Harry Potter" width="25%">
</div>

## A Machine Learning Project

This project is an introduction to artificial neural networks, with the
implementation of a multilayer perceptron (MLP)

### What is a Multilayer Perceptron?

The `multilayer perceptron` is a feedforward network (meaning that the data
flows from the input layer to the output layer)
defined by the presence of one or more hidden layers as well as an
interconnection of all the neurons of one layer to the next.

<div align="center">
<img src="images/mlp_vis.png" alt="Harry Potter" width="75%">
</div>

The diagram above represents a network containing 4 dense layers (also called fully connected layers). Its inputs consist of 4 neurons and its output of 2 (perfect for binary classification). The weights of one layer to the next are represented by two-dimensional matrices noted `W_l0_l1`. The matrix `W_l0_l1` is of size (3, 4), for example, as it contains the weights of the connections between the layer l0 and the layer l1.

The bias is often represented as a special neuron which has no inputs and with an output always equal to 1. Like a perceptron, it is connected to all the neurons of the following layer (`b_lj` on the diagram above). The bias is generally useful as it allows to “control the behavior” of a layer.
