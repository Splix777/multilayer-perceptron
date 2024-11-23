import numpy as np

from src.neural_net.activations.activation import Activation


class Tanh(Activation):
    """
    Tanh (Hyperbolic Tangent) is an activation function used
    in neural networks. It squashes input values to the range
    [-1, 1], mapping large negative values to -1 and large
    positive values to 1, with values near zero mapped
    closely around zero. Tanh is advantageous because it
    maintains the non-linearity of the data while preserving
    negative values unlike ReLU. However, it is susceptible
    to vanish gradient issues for very large inputs,
    similar to sigmoid activation. Despite this, Tanh is
    useful in scenarios where the output needs to be
    normalized to a range that centers around zero, making
    it effective for certain types of data normalization tasks.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Tanh activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with Tanh applied element-wise.
        """
        return np.tanh(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Tanh function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient array where elements
                are the derivatives of Tanh.
        """
        return 1 - np.power(x, 2)

    def get_config(self) -> dict:
        """
        Get the configuration of the Tanh activation.

        Returns:
            dict: Configuration dictionary.
        """
        return {"name": self.__class__.__name__}
