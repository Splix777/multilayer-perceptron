from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Optimizer(Protocol):
    learning_rate: float

    def update(
        self,
        weights: NDArray[np.float64],
        bias: NDArray[np.float64],
        weights_gradient: NDArray[np.float64],
        bias_gradients: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Update the weights and biases of the model.

        Args:
            weights (np.ndarray): Weights of the model.
            biases (np.ndarray): Biases of the model.
            weights_gradient (np.ndarray): Gradients of the weights.
            biases_gradient (np.ndarray): Gradients of the biases.

        Returns:
            tuple: Updated weights and biases.
        """
        ...

    def get_config(self) -> dict:
        ...
