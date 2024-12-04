from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from mlp.neural_net.optimizers.optimizer import Optimizer
from mlp.neural_net.regulizers.regulizer import Regularizer


class InputLayer:
    def __init__(self, input_shape: tuple[int, ...], **kwargs) -> None:
        """
        Initialize the InputLayer with the given input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.
                                 Example: (batch_size, input_dim)
        """
        # Protocol attributes
        self.trainable = False
        self.built = False
        self.input_shape: Tuple[int, ...] = input_shape
        self.output_shape: Tuple[int, ...] = input_shape
        self.weights: NDArray[np.float64] = np.empty(0)
        self.bias: NDArray[np.float64] = np.empty(0)
        self.optimizer: Optional[Optimizer] = None
        self.kernel_regularizer: Optional[str | Regularizer] = None
        self.weight_gradients: NDArray[np.float64] = np.empty(0)
        self.bias_gradients: NDArray[np.float64] = np.empty(0)

    def __call__(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
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
        return self.call(inputs)

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

    def call(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def get_weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the weights of the layer.
        InputLayer has no weight, so it returns an empty tuple.

        Returns:
            tuple: Empty tuple.
        """
        return self.weights, self.bias

    def set_weights(
        self, weights: NDArray[np.float64], bias: NDArray[np.float64]
    ):
        """
        Set the weights and biases of the layer.
        InputLayer has no weight, so it does nothing.

        Args:
            weights (np.ndarray): Weights of the layer.
            bias (np.ndarray): Bias of the layer.
        """
        pass
