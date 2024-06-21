import numpy as np
from .layer import Layer


class InputLayer(Layer):
    def __init__(self, input_shape: tuple[int, ...], **kwargs):
        """
        Initialize the InputLayer with the given input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.
                                 Example: (batch_size, input_dim)
        """
        super().__init__(input_shape=input_shape, trainable=False)
        self.input_shape = input_shape
        self._output_shape = input_shape

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Initialize the layer with the given input shape.
        For InputLayer, it verifies and sets the input_shape.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor (same as input_shape).

        Raises:
            ValueError: If input_shape is not a tuple or is empty.
        """
        if not isinstance(input_shape, tuple) or len(input_shape) == 0:
            raise ValueError("Input shape must be a non-empty tuple.")

        self._output_shape = input_shape
        self.built = True
        return self._output_shape

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.
        For InputLayer, it simply returns the input tensor.

        Args:
            inputs (np.ndarray): Input data or features.

        Returns:
            np.ndarray: Output tensor (same as input).

        Raises:
            ValueError: If the layer is not built.
        """
        if not self.built:
            raise ValueError("InputLayer not built. Call build() first.")

        return inputs

    def backward(self, output_gradient: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """
        Perform the backward pass.
        For InputLayer, it simply returns the output_gradient.

        Args:
            output_gradient (np.ndarray): Gradient of the
                losses with respect to the output.
            learning_rate (float): Learning rate used for
                gradient descent optimization.

        Returns:
            np.ndarray: Gradient of the losses with respect
                to the input (same as output_gradient).
        Raises:
            ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("The layer has not been built yet.")

        return output_gradient

    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the output from the layer.
        For InputLayer, it returns the input_shape.

        Returns:
            tuple: Shape of the output from the layer.

        Raises:
            ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("The layer has not been built yet.")

        return self._output_shape

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the layer.
        InputLayer has no parameters, so it returns 0.

        Returns:
            int: Total number of parameters in the layer.
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
