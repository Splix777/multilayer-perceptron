import numpy as np
from .layer import Layer


class Dense(Layer):
    def __init__(self,
                 units: int,
                 activation: str = None,
                 use_bias: bool = True,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',

                 **kwargs):
        """
        Initialize a Dense layer with the given number
        of units and activation function.

        Args:
            units (int): Number of neurons in the layer.
            activation (str, optional): Activation function
                to use. Defaults to None.
            use_bias (bool, optional): Whether to use bias or not.
                Defaults to True.
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
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation_output = None
        self.input = None
        self.weights = None
        self.bias = None

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
        fan_in = input_shape[-1]
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

        if self.use_bias:
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
            print(f"Inputs shape: {inputs.shape}")
            print(f"Weights shape: {self.weights.shape}")
            raise ValueError("Dimensions mismatch: "
                             "inputs and weights are not compatible.")

        z = np.dot(inputs, self.weights) + self.bias

        # Efficient computation due to its simple form (max(0, z)).
        if self.activation == 'relu':
            self.activation_output = np.maximum(0, z)
        # Outputs are in the range [0, 1].
        elif self.activation == 'sigmoid':
            self.activation_output = 1 / (1 + np.exp(-z))
        # Outputs are zero-centered, making optimization potentially easier.
        elif self.activation == 'tanh':
            self.activation_output = np.tanh(z)
        # Outputs in the range [0, 1] and sum to 1. (multiclass classification)
        elif self.activation == 'softmax':
            exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
            self.activation_output = exps / np.sum(exps, axis=-1,
                                                   keepdims=True)
        else:
            self.activation_output = z

        return self.activation_output

    def backward(self, output_gradient: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """
        Perform backpropagation through the Dense layer.

        Args:
            output_gradient (np.ndarray): Gradient of the loss
                with respect to the output of this layer.
            learning_rate (float): Learning rate for
                gradient descent optimization.

        Returns:
            np.ndarray: Gradient of the loss with
                respect to the input of this layer.
        """
        if self.activation == 'relu':
            activation_gradient = output_gradient * (
                        self.activation_output > 0)
        elif self.activation == 'sigmoid':
            activation_gradient = output_gradient * self.activation_output * (
                        1 - self.activation_output)
        elif self.activation == 'tanh':
            activation_gradient = output_gradient * (
                        1 - np.square(self.activation_output))
        else:
            activation_gradient = output_gradient

        input_gradient = np.dot(activation_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, activation_gradient)
        bias_gradient = np.sum(activation_gradient, axis=0)

        # Update weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

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
