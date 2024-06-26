import numpy as np

from src.model.layers.layer import Layer
from src.model.activations.leaky_relu import LeakyReLU
from src.model.activations.parametric_relu import ParametricReLU
from src.model.activations.relu import ReLU
from src.model.activations.sigmoid import Sigmoid
from src.model.activations.tanh import Tanh
from src.model.activations.softmax import Softmax
from src.model.regulizers.l1_regulizer import L1Regularizer
from src.model.regulizers.l2_regulizer import L2Regularizer
from src.utils.logger import Logger


class Dense(Layer):
    def __init__(self,
                 units: int,
                 activation: str = None,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 kernel_regularizer: str = None,
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
        self.logger = Logger('Dense')()
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_initializer = bias_initializer
        self.weights = None
        self.bias = None
        self.optimizer = None
        self.input = None
        self.z = None
        self.activation_output = None
        self.bias_gradients = None
        self.weights_gradients = None
        self.activation_function = {
            'relu': ReLU(),
            'lrelu': LeakyReLU(),
            'prelu': ParametricReLU(),
            'tanh': Tanh(),
            'sigmoid': Sigmoid(),
            'softmax': Softmax()}.get(activation)

    def _initialize_weights(self, input_shape: tuple[int, ...]) -> None:
        """
        Initialize the weights of the Dense layer using
        the given input shape.

        Args:
            input_shape (tuple): Shape of the input tensor
                (batch_size, input_dim).

        Returns:
            None
        """
        # Number of Features
        fan_in = input_shape[-1]
        # Number of Neurons
        fan_out = self.units

        if self.kernel_initializer == 'glorot_uniform':
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = np.random.uniform(-limit, limit,
                                             size=(fan_in, fan_out))
        elif self.kernel_initializer == 'glorot_normal':
            std_dev = np.sqrt(2 / (fan_in + fan_out))
            self.weights = np.random.normal(0, std_dev, size=(fan_in, fan_out))
        elif self.kernel_initializer == 'he_uniform':
            limit = np.sqrt(6 / fan_in)
            self.weights = np.random.uniform(-limit, limit,
                                             size=(fan_in, fan_out))
        elif self.kernel_initializer == 'he_normal':
            std_dev = np.sqrt(2 / fan_in)
            self.weights = np.random.normal(0, std_dev, size=(fan_in, fan_out))
        else:
            raise ValueError(f"Unknown kernel initializer: "
                             f"{self.kernel_initializer}")

    def _initialize_bias(self) -> None:
        """
        Initialize the bias of the Dense layer.

        Returns:
            None
        """
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

    def _initialize_regularizer(self) -> None:
        """
        Initialize the kernel regularizer of the Dense layer.

        Returns:
            None
        """
        if self.kernel_regularizer == 'l1':
            self.kernel_regularizer = L1Regularizer(lambda_param=0.01)
        elif self.kernel_regularizer == 'l2':
            self.kernel_regularizer = L2Regularizer(lambda_param=0.01)
        else:
            raise ValueError(f"Unknown kernel regularizer: "
                             f"{self.kernel_regularizer}")

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
        self._initialize_weights(input_shape)
        self._initialize_bias()
        if self.kernel_regularizer:
            self._initialize_regularizer()
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

        if inputs.shape[-1] != self.weights.shape[0]:
            raise ValueError("Dimensions mismatch: "
                             "inputs and weights are not compatible.")

        self.z = np.dot(inputs, self.weights) + self.bias

        if self.activation_function:
            self.activation_output = self.activation_function(self.z)

        return self.activation_output

    def backward(self, loss_gradients: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass through the Dense layer.

        Args:
            loss_gradients (np.ndarray): Gradients of the loss
                with respect to the output of the Dense layer.

        Returns:
            np.ndarray: Gradients of the loss with respect to
                the input of the Dense layer.
        """
        if self.activation_function:
            activation_gradients = self.activation_function.gradient(
                x=self.activation_output)

            loss_gradients = loss_gradients * activation_gradients

        if isinstance(self.activation_function, ParametricReLU):
            self.activation_function.update_alpha(
                loss_gradients=loss_gradients,
                input_data=self.z,
                learning_rate=self.optimizer.learning_rate)

        self.weights_gradients = np.dot(self.input.T, loss_gradients)
        self.bias_gradients = np.sum(loss_gradients, axis=0, keepdims=True)

        return np.dot(loss_gradients, self.weights.T)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the output from the Dense layer.

        Args:
            input_shape (tuple): Shape of the input tensor
                (batch_size, input_dim).

        Returns:
            tuple: Shape of the output tensor (batch_size, units).
        """
        return self._output_shape

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the Dense layer.

        Returns:
            int: Total number of parameters (weights and biases)
                in the layer.
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

    @property
    def learning_rate(self):
        """
        Get the learning rate of the optimizer.
        """
        if not self.optimizer:
            raise ValueError("Optimizer not found.")
        return self.optimizer.learning_rate
