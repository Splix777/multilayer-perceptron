import numpy as np

from src.model.activations.activation import Activation


class ParametricReLU(Activation):
    """
    Parametric ReLU (PReLU) is an activation function
    used in neural networks that extends the concept
    of Leaky ReLU by allowing the negative slope to be
    learned during training. Unlike Leaky ReLU, where
    the negative slope is a fixed small constant,
    PReLU introduces a learnable parameter that adjusts
    the slope for negative inputs. This flexibility
    enables the model to adapt the activation function more
    effectively to the data, potentially improving model
    performance and convergence. By learning the appropriate
    negative slope, PReLU can help mitigate the "dying ReLU"
    problem and enhance the overall expressiveness of the
    neural network.
    """
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Parametric ReLU activation function
        with an optional alpha parameter.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array with Parametric
                ReLU applied element-wise.
        """
        return np.maximum(self.alpha * x, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Parametric ReLU function.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient array where elements are 1
                if the input is greater than 0, else alpha.
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
        masked_input = np.where(input_data < 0, input_data, 0)
        self.alpha -= learning_rate * np.mean(loss_gradients * masked_input)

    def get_config(self) -> dict:
        """
        Get the configuration of the Parametric ReLU activation.

        Returns:
            dict: Configuration dictionary.
        """
        return {'name': self.__class__.__name__, 'alpha': self.alpha}
