from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Regularizer(Protocol):
    """
    Abstract base class for regularizers.
    """
    def __call__(self, weights: NDArray[np.float64]) -> float:
        """
        Calculate the regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            float: Regularization penalty.
        """
        ...

    def gradient(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the gradient of the regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            np.ndarray: Gradient of the regularization penalty.
        """
        ...
