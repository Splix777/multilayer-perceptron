from .layer import Layer
import numpy as np


class Dropout(Layer):
    def __init__(self, rate: float):
        """
        Initialize a Dropout layer with the given rate.

        Args:
            rate (float): Fraction of the input units to drop.
        """
        super().__init__(trainable=False)
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in the range [0, 1).")
        self.rate = rate
        self.mask = None

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Build the Dropout layer with the given input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor.
        """
        self._output_shape = input_shape
        return self._output_shape

    def call(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform the forward pass.

        Args:
            inputs (np.ndarray): Input data or features.
            training (bool): Whether the model is in training mode.

        Returns:
            np.ndarray: Output tensor.
        """
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
            return inputs * self.mask / (1 - self.rate)
        return inputs

    def backward(self, loss_gradients: np.ndarray) -> np.ndarray:
        return loss_gradients * self.mask / (1 - self.rate)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the output tensor.

        Returns:
            tuple: Shape of the output tensor.
        """
        return self._output_shape

    def count_parameters(self) -> int:
        """
        Count and return the number of parameters.

        Returns:
            int: Number of parameters.
        """
        return 0

    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the weights of the layer.
        InputLayer has no weights, so it returns an empty tuple.

        Returns:
            tuple: Empty tuple.
        """
        return np.array([]), np.array([])

    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """
        Set the weights and biases of the layer.
        InputLayer has no weights, so it does nothing.

        Args:
            weights (np.ndarray): Weights of the layer.
            bias (np.ndarray): Bias of the layer.
        """
        pass
