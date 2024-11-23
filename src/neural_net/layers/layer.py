from typing import Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class Layer(Protocol):
    trainable: bool
    built: bool
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.
        """
        ...

    def build(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Initialize the layer with the given input shape.
        """
        ...

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.
        """
        ...

    def backward(self, loss_gradients: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass.
        """
        ...

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the layer.
        """
        ...

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the weights and biases of the layer.
        """
        ...

    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """
        Set the weights and biases of the layer.
        """
        ...
