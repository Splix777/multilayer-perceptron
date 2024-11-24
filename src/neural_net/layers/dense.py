import numpy as np
from typing import Optional

from src.neural_net.activations.leaky_relu import LeakyReLU
from src.neural_net.activations.parametric_relu import ParametricReLU
from src.neural_net.activations.relu import ReLU
from src.neural_net.activations.sigmoid import Sigmoid
from src.neural_net.activations.tanh import Tanh
from src.neural_net.activations.softmax import Softmax
from src.neural_net.regulizers.l1_regulizer import L1Regularizer
from src.neural_net.regulizers.l2_regulizer import L2Regularizer
from src.neural_net.regulizers.regulizer import Regularizer
from src.neural_net.optimizers.optimizer import Optimizer
from src.utils.logger import Logger


class Dense:
    def __init__(
        self,
        units: int,
        activation: str,
        kernel_regularizer: Optional[str],
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "ones",
        **kwargs,
    ) -> None:
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
        # Protocol Attributes
        self.trainable = True
        self.built = False
        self.input_shape: tuple[int, ...] = (0,0)
        self.output_shape: tuple[int, ...] = (0,0)
        # Specific Attributes
        self.logger = Logger("Dense")
        self.units: int = units
        self.kernel_initializer: str = kernel_initializer
        self.bias_initializer: str = bias_initializer
        self.kernel_regularizer: Optional[str | Regularizer] = kernel_regularizer
        self.optimizer: Optional[Optimizer] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray]  = None
        self.input = None
        self.z = None
        self.activation_output = None
        self.bias_gradients: Optional[np.ndarray] = None
        self.weights_gradients: Optional[np.ndarray] = None
        self.activation_function = {
            "relu": ReLU(),
            "lrelu": LeakyReLU(),
            "prelu": ParametricReLU(),
            "tanh": Tanh(),
            "sigmoid": Sigmoid(),
            "softmax": Softmax(),
        }.get(activation)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass through the Dense layer.

        Args:
            inputs (np.ndarray): Input data or features
                of shape (batch_size, input_dim).

        Returns:
            np.ndarray: Output tensor of shape (batch_size, units).
        """
        return self.call(inputs)

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
        self.logger.info(f"Building Dense layer with input shape: {input_shape}")
        if not isinstance(input_shape, tuple) or not input_shape:
            raise ValueError("Input shape must be a non-empty tuple.")
        if any(dim <= 0 for dim in input_shape):
            raise ValueError("All dimensions must be positive integers.")
        self._initialize_weights(input_shape)
        self._initialize_bias()
        if self.kernel_regularizer:
            self._initialize_regularizer()
        self.output_shape = (input_shape[0], self.units)
        self.built = True

        return self.output_shape

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
            raise ValueError(
                "Dimensions mismatch: "
                "inputs and weights are not compatible."
            )

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
                x=self.activation_output
            )

            loss_gradients = loss_gradients * activation_gradients

        if isinstance(self.activation_function, ParametricReLU):
            self.activation_function.update_alpha(
                loss_gradients=loss_gradients,
                input_data=self.z,
                learning_rate=self.optimizer.learning_rate,
            )

        self.weights_gradients = np.dot(self.input.T, loss_gradients)
        self.bias_gradients = np.sum(loss_gradients, axis=0, keepdims=True)

        return np.dot(loss_gradients, self.weights.T)

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
        fan_in: int = input_shape[-1]
        # Number of Neurons
        fan_out: int = self.units

        if self.kernel_initializer == "glorot_uniform":
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights = np.random.uniform(
                -limit, limit, size=(fan_in, fan_out)
            )
        elif self.kernel_initializer == "glorot_normal":
            std_dev = np.sqrt(2 / (fan_in + fan_out))
            self.weights = np.random.normal(0, std_dev, size=(fan_in, fan_out))
        elif self.kernel_initializer == "he_uniform":
            limit = np.sqrt(6 / fan_in)
            self.weights = np.random.uniform(
                -limit, limit, size=(fan_in, fan_out)
            )
        elif self.kernel_initializer == "he_normal":
            std_dev = np.sqrt(2 / fan_in)
            self.weights = np.random.normal(0, std_dev, size=(fan_in, fan_out))
        else:
            raise ValueError(
                f"Unknown kernel initializer: " f"{self.kernel_initializer}"
            )

    def _initialize_bias(self) -> None:
        """
        Initialize the bias of the Dense layer.

        Returns:
            None
        """
        if self.bias_initializer == "zeros":
            self.bias = np.zeros((1, self.units))
        elif self.bias_initializer == "ones":
            self.bias = np.ones((1, self.units))
        elif self.bias_initializer == "random_normal":
            self.bias = np.random.randn(1, self.units)
        elif self.bias_initializer == "random_uniform":
            self.bias = np.random.rand(1, self.units)
        else:
            raise ValueError(
                f"Unknown bias initializer: " f"{self.bias_initializer}"
            )

    def _initialize_regularizer(self) -> None:
        """
        Initialize the kernel regularizer of the Dense layer.

        Returns:
            None
        """
        if self.kernel_regularizer == "l1":
            self.kernel_regularizer = L1Regularizer(lambda_param=0.01)
        elif self.kernel_regularizer == "l2":
            self.kernel_regularizer = L2Regularizer(lambda_param=0.01)
        else:
            raise ValueError(
                f"Unknown kernel regularizer: " f"{self.kernel_regularizer}"
            )

    @property
    def learning_rate(self) -> float:
        """
        Get the learning rate of the optimizer.
        """
        if not self.optimizer:
            raise ValueError("Optimizer not found.")

        return self.optimizer.learning_rate
