import numpy as np
from ..activations.activation import Activation


# Parametric Rectified Linear Unit (PReLU)


class ParametricReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Parametric ReLU activation function with an optional alpha parameter.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with Parametric ReLU applied element-wise.
        """
        return np.maximum(self.alpha * x, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Parametric ReLU function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient array where elements are 1 if the input is greater than 0, else alpha.
        """
        return np.where(x > 0, 1, self.alpha)

    def update_alpha(self, loss_gradients: np.ndarray, input_data: np.ndarray,
                     learning_rate: float) -> None:
        """
        Update the alpha parameter of the Parametric ReLU activation.

        Args:
            loss_gradients (np.ndarray): Gradients of the
                loss with respect to layer output.
            input_data (np.ndarray): Input data to the layer.
            learning_rate (float): Learning rate for the update.
        """
        # Mask the input data to only consider negative values
        masked_input = np.where(input_data < 0, input_data, 0)
        # Update the alpha parameter using the gradients and masked input
        self.alpha -= learning_rate * np.mean(loss_gradients * masked_input)

    def get_config(self) -> dict:
        """
        Get the configuration of the Parametric ReLU activation.

        Returns:
            dict: Configuration dictionary.
        """
        return {'name': self.__class__.__name__, 'alpha': self.alpha}
