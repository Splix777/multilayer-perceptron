from typing import Tuple
import numpy as np

class InputLayer:
    def __init__(self, input_shape: tuple[int, ...], **kwargs):
        """
        Initialize the InputLayer with the given input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.
                                 Example: (batch_size, input_dim)
        """
        self.trainable = False
        self.built = False
        self.input_shape: Tuple[int, ...] = input_shape
        self.output_shape: Tuple[int, ...] = input_shape

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
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
        if not isinstance(input_shape, tuple) or not input_shape:
            raise ValueError("Input shape must be a non-empty tuple.")
        if any(dim <= 0 for dim in input_shape):
            raise ValueError("All dimensions must be positive integers.")

        self.output_shape = input_shape
        self.built = True
        return self.output_shape

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

    def backward(self, loss_gradients: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass.
        For InputLayer, it simply returns the loss_gradients.

        Args:
            loss_gradients (np.ndarray): Gradients of the loss
                with respect to the output of the layer.

        Returns:
            np.ndarray: Gradients of the loss
                with respect to the input.
        """
        if not self.built:
            raise ValueError("The layer has not been built yet.")

        return loss_gradients

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
        InputLayer has no weight, so it returns an empty tuple.

        Returns:
            tuple: Empty tuple.
        """
        return np.array([]), np.array([])

    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """
        Set the weights and biases of the layer.
        InputLayer has no weight, so it does nothing.

        Args:
            weights (np.ndarray): Weights of the layer.
            bias (np.ndarray): Bias of the layer.
        """
        pass
