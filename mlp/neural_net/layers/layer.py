from typing import Protocol, Tuple, runtime_checkable, Optional

import numpy as np
from numpy.typing import NDArray

from mlp.neural_net.optimizers.optimizer import Optimizer
from mlp.neural_net.regulizers.regulizer import Regularizer

@runtime_checkable
class Layer(Protocol):
    trainable: bool
    built: bool
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    weights: NDArray[np.float64]
    bias: NDArray[np.float64]
    optimizer: Optional[Optimizer] = None
    kernel_regularizer: Optional[str | Regularizer] = None
    weight_gradients: NDArray[np.float64] 
    bias_gradients: NDArray[np.float64]


    def __call__(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform the forward pass.
        """
        ...

    def build(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Initialize the layer with the given input shape.
        """
        ...

    def call(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
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

    def get_weights(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the weights and biases of the layer.
        """
        ...

    def set_weights(self, weights: NDArray[np.float64], bias: NDArray[np.float64]) -> None:
        """
        Set the weights and biases of the layer.
        """
        ...
