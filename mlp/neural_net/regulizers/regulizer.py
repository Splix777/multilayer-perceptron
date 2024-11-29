import numpy as np
from numpy.typing import NDArray

from abc import ABC, abstractmethod


class Regularizer(ABC):
    """
    Abstract base class for regularizers.
    """

    @abstractmethod
    def __call__(self, weights: NDArray[np.float64]) -> float:
        """
        Calculate the regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            float: Regularization penalty.
        """
        pass

    @abstractmethod
    def gradient(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the gradient of the regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            np.ndarray: Gradient of the regularization penalty.
        """
        pass
