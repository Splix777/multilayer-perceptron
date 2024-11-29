import numpy as np

from mlp.neural_net.activations.activation import Activation


class LeakyReLU(Activation):
    """
    Leaky ReLU (Leaky Rectified Linear Unit) is an activation
    function commonly used in neural networks to address the
    "dying ReLU" problem where neurons can sometimes get stuck
    during training and always output zero. Unlike the standard
    ReLU, which outputs zero for all negative inputs, Leaky ReLU
    allows a small, non-zero gradient
    (defined by a small constant multiplier) when the input is
    negative. This helps maintain the flow of gradients through
    the network, preventing neurons from becoming inactive and
    improving model performance and convergence during training.
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function with
            an optional alpha parameter.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with ReLU applied element-wise.
        """
        return np.maximum(self.alpha * x, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the ReLU function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient array where elements
                are 1 if the input is greater than 0, else alpha.
        """
        return np.where(x > 0, 1, self.alpha)

    def get_config(self) -> dict:
        """
        Get the configuration of the ReLU activation.

        Returns:
            dict: Configuration dictionary.
        """
        return {"name": self.__class__.__name__, "alpha": self.alpha}
