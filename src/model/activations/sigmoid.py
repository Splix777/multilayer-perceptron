from ..activations.activation import Activation
import numpy as np

"""
Sigmoid is an activation function commonly used in neural
networks to map input values to a range between 0 and 1.
It has a characteristic S-shaped curve that smoothly
transforms any real-valued input to a probability-like output.
Sigmoid is advantageous because it squashes the input to a
manageable range, making it useful for binary classification
tasks where the output represents probabilities. However,
sigmoid suffers from the vanishing gradient problem,
especially for very large or very small input values,
which can slow down the learning process during training.
Despite this limitation, sigmoid remains widely used in
the output layer of binary classification models.
"""

# S-shaped Logistic Function


class Sigmoid(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Sigmoid activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with Sigmoid applied element-wise.
        """
        return 1 / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Sigmoid function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient array where elements are the derivatives of Sigmoid.
        """
        return x * (1 - x)

    def get_config(self) -> dict:
        """
        Get the configuration of the Sigmoid activation.

        Returns:
            dict: Configuration dictionary.
        """
        return {'name': self.__class__.__name__}
