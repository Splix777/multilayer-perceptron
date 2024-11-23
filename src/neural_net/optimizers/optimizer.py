import numpy as np

from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate=0.001, **kwargs):
        """
        Initialize the optimizers.

        Args:
            learning_rate (float, optional): Learning rate
                for the optimizers. Default is 0.001.
            **kwargs: Additional keyword arguments for
                the optimizers initialization.
        """
        self.learning_rate = learning_rate
        self._init_kwargs = kwargs

    @abstractmethod
    def update(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        weights_gradient: np.ndarray,
        biases_gradient: np.ndarray,
    ):
        """
        Update the weights and biases of the model.

        Args:
            weights (np.ndarray): Weights of the model.
            biases (np.ndarray): Biases of the model.
            weights_gradient (np.ndarray): Gradients of the weights.
            biases_gradient (np.ndarray): Gradients of the biases.

        Returns:
            tuple: Updated weights and biases.
        """
        pass

    @abstractmethod
    def get_config(self):
        """
        Get the configuration of the optimizers.

        Returns:
            dict: Configuration of the optimizers.
        """
        pass
