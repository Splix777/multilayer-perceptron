import numpy as np
from numpy.typing import NDArray

from src.neural_net.regulizers.regulizer import Regularizer


class L1Regularizer(Regularizer):
    """
    L1 regularization penalizes the absolute magnitude of weights.

    L1 regularization adds a penalty equivalent to the sum of the
    absolute values of the weights to the loss function. This
    encourages sparsity in the weights, leading to some weights
    becoming exactly zero. This property makes L1 regularization useful
    for feature selection in models with high-dimensional input spaces.

    Advantages:
    - Encourages sparsity: L1 regularization promotes models where
        many weights are zero, which can improve interpretability
        and reduce overfitting.
    - Feature selection: The sparsity induced by L1 regularization
        can help in selecting important features from noisy or
        redundant input data.

    Disadvantages:
    - Non-smooth penalty: The absolute value function used in
        L1 regularization is not differentiable at zero, which
        can make optimization more challenging compared to L2
        regularization.
    """
    def __init__(self, lambda_param: float):
        """
        Initialize the L1 regularizer.

        Args:
            lambda_param (float): Regularization parameter
                controlling the strength of the penalty.
        """
        self.lambda_param: float = lambda_param

    def __call__(self, weights: NDArray[np.float64]) -> float:
        """
        Calculate the L1 regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            float: L1 regularization penalty.
        """
        return self.lambda_param * np.sum(np.abs(weights))

    def gradient(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the gradient of the L1 regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            np.ndarray: Gradient of the L1 regularization penalty.
        """
        return self.lambda_param * np.sign(weights)
