import numpy as np

from abc import ABC, abstractmethod


class Regularizer(ABC):
    """
    Abstract base class for regularizers.
    """

    @abstractmethod
    def __call__(self, weights: np.ndarray) -> float:
        """
        Calculate the regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            float: Regularization penalty.
        """
        pass

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            np.ndarray: Gradient of the regularization penalty.
        """
        pass
