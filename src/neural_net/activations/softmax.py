import numpy as np

from src.neural_net.activations.activation import Activation


class Softmax(Activation):
    """
    Softmax activation function computes the probability distribution
    over multiple classes. It takes in a vector of values
    (usually logits)and exponentiates each element to make them
    positive. Then, it divides each exponentiated value by the
    sum of all exponentiated values, ensuring the sum of all
    probabilities equals 1. This function is commonly used as the
    output layer for multi-class classification problems, where
    it outputs a probability distribution over mutually
    exclusive classes. The output is interpretable as probabilities,
    allowing us to choose the class with the highest probability.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Softmax activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with Softmax-applied element-wise.
        """
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Softmax function.

        Args:
            x (np.ndarray): Input array with Softmax applied.

        Returns:
            np.ndarray: Gradient array where elements
                are the derivatives of Softmax.
        """
        return x * (1 - x)

    def get_config(self) -> dict:
        """
        Get the configuration of the Softmax activation.

        Returns:
            dict: Configuration dictionary.
        """
        return {"name": self.__class__.__name__}
