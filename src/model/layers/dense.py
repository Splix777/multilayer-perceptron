import random

import numpy as np
# from .layer import Layer
# from ..activations.relu import ReLU
# from ..activations.sigmoid import Sigmoid
# from ..activations.tanh import Tanh
# from ..activations.softmax import Softmax

from src.model.layers.layer import Layer
from src.model.activations.relu import ReLU
from src.model.activations.sigmoid import Sigmoid
from src.model.activations.tanh import Tanh
from src.model.activations.softmax import Softmax
from src.model.optimizers.adam import AdamOptimizer


class Dense(Layer):
    def __init__(self,
                 units: int,
                 activation: str = None,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 **kwargs):
        """
        Initialize a Dense layer with the given number
        of units and activations function.

        Args:
            units (int): Number of neurons in the layer.
            activation (str, optional): Activation function
                to use. Defaults to None.
            kernel_initializer (str, optional): Initialization
                strategy for kernel weights. Defaults to
                'glorot_uniform'.
            bias_initializer (str, optional): Initialization
                strategy for bias weights. Defaults to 'zeros'.
            kernel_regularizer (str, optional): Regularization
                strategy for kernel weights. Defaults to None.
            bias_regularizer (str, optional): Regularization
                strategy for bias weights. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.bias_gradient = None
        self.weights_gradient = None
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.optimizer = None
        self.activation_output = None
        self.input = None
        self.weights = None
        self.bias = None
        self.activation_function = {
            'relu': ReLU(),
            'tanh': Tanh(),
            'sigmoid': Sigmoid(),
            'softmax': Softmax()}.get(activation)

    def initialize_weights(self, input_shape: tuple[int, ...]) -> None:
        # Number of Features
        fan_in = input_shape[-1]
        # Number of Neurons
        fan_out = self.units

        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
        elif self.kernel_initializer == 'glorot_normal':
            std_dev = np.sqrt(2 / (fan_in + fan_out))
            self.weights = np.random.normal(0, std_dev, size=(fan_in, fan_out))
        elif self.kernel_initializer == 'he_uniform':
            limit = np.sqrt(6 / fan_in)
            self.weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
        elif self.kernel_initializer == 'he_normal':
            std_dev = np.sqrt(2 / fan_in)
            self.weights = np.random.normal(0, std_dev, size=(fan_in, fan_out))
        else:
            raise ValueError(f"Unknown kernel initializer: "
                             f"{self.kernel_initializer}")

    def initialize_bias(self) -> None:
        if self.bias_initializer == 'zeros':
            self.bias = np.zeros((1, self.units))
        elif self.bias_initializer == 'ones':
            self.bias = np.ones((1, self.units))
        elif self.bias_initializer == 'random_normal':
            self.bias = np.random.randn(1, self.units)
        elif self.bias_initializer == 'random_uniform':
            self.bias = np.random.rand(1, self.units)
        else:
            raise ValueError(f"Unknown bias initializer: "
                             f"{self.bias_initializer}")

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Build the Dense layer with the given input
        shape and initialize weights.

        Args:
            input_shape (tuple): Shape of the input
                tensor (batch_size, input_dim).

        Returns:
            tuple: Shape of the output tensor (batch_size, units).
        """
        self.initialize_weights(input_shape)
        self.initialize_bias()
        self._output_shape = (input_shape[0], self.units)
        self.built = True

        return self._output_shape

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through the Dense layer.

        Args:
            inputs (np.ndarray): Input data or features
                of shape (batch_size, input_dim).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, units).
        """
        self.input = inputs
        # print(f'Input Shape: {inputs.shape}')

        if inputs.shape[-1] != self.weights.shape[0]:
            print(f"Inputs shape: {inputs.shape}")
            print(f"Weights shape: {self.weights.shape}")
            raise ValueError("Dimensions mismatch: "
                             "inputs and weights are not compatible.")

        z = np.dot(inputs, self.weights) + self.bias

        if self.activation_function:
            self.activation_output = self.activation_function(z)
        else:
            raise ValueError("Activation function not found.")

        return self.activation_output

    def backward(self, loss_gradients: np.ndarray) -> np.ndarray:
        if not self.activation_function:
            raise ValueError("Activation function not found.")

        activation_gradients = self.activation_function.gradient(self.activation_output)
        loss_gradients = loss_gradients * activation_gradients

        self.weights_gradient = np.dot(self.input.T, loss_gradients)
        self.bias_gradient = np.sum(loss_gradients, axis=0, keepdims=True)

        return np.dot(loss_gradients, self.weights.T)




    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the output from the Dense layer.

        Args:
            input_shape (tuple): Shape of the input tensor (batch_size, input_dim).

        Returns:
            tuple: Shape of the output tensor (batch_size, units).
        """
        return self._output_shape

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the Dense layer.

        Returns:
            int: Total number of parameters (weights and biases) in the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.bias.shape)

    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the weights and biases of the Dense layer.

        Returns:
            tuple: Weights and biases of the layer.
        """
        return self.weights, self.bias

    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """
        Set the weights and biases of the Dense layer.

        Args:
            weights (np.ndarray): Weights of the layer.
            bias (np.ndarray): Biases of the layer.
        """
        self.weights = weights
        self.bias = bias


if __name__ == "__main__":
    dense = Dense(units=2, activation='softmax')
    print(dense.build(input_shape=(30, 64)))
    weights, bias = dense.get_weights()
    print(dense.output_shape)

    sample = [random.uniform(-1, 1) for _ in range(64)]
    previous_output = np.array([sample])

    output = dense.call(previous_output)

    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
