import numpy as np

from src.model.activations.activation import Activation


class ReLU(Activation):
    """
    ReLU (Rectified Linear Unit) is an activation function
    commonly used in neural networks. It operates by passing
    through all positive input values unchanged, effectively
    setting negative values to zero. This non-linear
    transformation introduces sparsity in the network,
    enhancing its ability to model complex patterns in data.
    ReLU is preferred over other activation functions like sigmoid
    and tanh due to its simplicity and effectiveness in mitigating
    issues like vanishing gradients during training, thereby
    accelerating convergence and improving the efficiency
    of deep learning models.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with ReLU applied element-wise.
        """
        return np.maximum(0, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the ReLU function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient array where elements are 1
                if the input is greater than 0, else 0.
        """
        return np.where(x > 0, 1, 0)

    def get_config(self) -> dict:
        """
        Get the configuration of the ReLU activation.

        Returns:
            dict: Configuration dictionary.
        """
        return {'name': self.__class__.__name__}
