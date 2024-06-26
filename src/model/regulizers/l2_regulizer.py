import numpy as np

from src.model.regulizers.regulizer import Regularizer


class L2Regularizer(Regularizer):
    """
    L2 regularization penalizes the squared magnitude of weights.

    L2 regularization adds a penalty equivalent to the sum of
    the squares of the weights to the loss function. This encourages
    smaller weights and helps prevent overfitting by reducing the
    model's reliance on any single weight. It is one of the most
    commonly used regularization techniques in machine learning.

    Advantages:
    - Smoother penalty: L2 regularization is differentiable everywhere,
        which simplifies optimization compared to L1 regularization.
    - Generalization: Encourages the model to generalize better by
        penalizing large weights, reducing the risk of overfitting.

    Disadvantages:
    - May not induce sparsity: Unlike L1 regularization,
        L2 regularization typically results in weights that
        are small but non-zero.
    """
    def __init__(self, lambda_param: float):
        """
        Initialize the L2 regularizer.

        Args:
            lambda_param (float): Regularization parameter
                controlling the strength of the penalty.
        """
        self.lambda_param = lambda_param

    def __call__(self, weights: np.ndarray) -> float:
        """
        Calculate the L2 regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            float: L2 regularization penalty.
        """
        return 0.5 * self.lambda_param * np.sum(weights ** 2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the L2 regularization penalty.

        Args:
            weights (np.ndarray): The weights of the layer.

        Returns:
            np.ndarray: Gradient of the L2 regularization penalty.
        """
        return self.lambda_param * weights
