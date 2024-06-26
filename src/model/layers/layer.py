import numpy as np

from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, input_shape: tuple[int, ...] = None,
                 trainable: bool = True):
        """
        Initialize the Layer with common properties.

        Args:
            trainable (bool): Whether the layer's weights
                are trainable. Defaults to True.
        """
        self.trainable = trainable

        if input_shape is not None:
            self.input_shape = input_shape

        self.built = False
        self.weights = None
        self.bias = None
        self._output_shape = None
        self.weights_gradients = None
        self.bias_gradients = None
        self.kernel_initializer = None
        self.kernel_regularizer = None
        self.bias_initializer = None
        self.optimizer = None

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(trainable={self.trainable}, "
                f"output_shape={self.output_shape})")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(trainable={self.trainable}, "
                f"built={self.built})")

    def __len__(self) -> int:
        return self.count_parameters()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.

        Args:
            inputs (np.ndarray): Input data or features.

        Returns:
            np.ndarray: Output of the layer.
        """
        if not self.built:
            self.build(input_shape=inputs.shape)
            self.built = True

        return self.call(inputs)

    @abstractmethod
    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Initialize the layer with the given input shape.

        Args:
            input_shape (tuple[int, ...]): Shape of the input to the layer.

        Returns:
            tuple[int, ...]: Shape of the output from the layer.
        """
        pass

    @abstractmethod
    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.

        Args:
            inputs (np.ndarray): Input data or features.

        Returns:
            np.ndarray: Output of the layer.
        """
        pass

    @abstractmethod
    def backward(self, loss_gradients: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass.

        Args:
            loss_gradients (np.ndarray): Gradients of the loss.

        Returns:
            np.ndarray: Gradients of the loss with respect
                to the inputs.
        """
        pass

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the output from the layer.

        Returns:
            tuple[int, ...]: Shape of the output from the layer.
        """
        pass

    @abstractmethod
    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the layer.

        Returns:
            int: Total number of parameters in the layer.
        """
        return 0

    @abstractmethod
    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the weights and biases of the layer.

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights and biases of the layer.
        """
        return self.weights, self.bias

    @abstractmethod
    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """
        Set the weights and biases of the layer.

        Args:
            weights (np.ndarray): Weights of the layer.
            bias (np.ndarray): Bias of the layer.

        """
        self.weights = weights
        self.bias = bias

    @property
    def built(self) -> bool:
        return self._built

    @built.setter
    def built(self, value: bool):
        self._built = value
