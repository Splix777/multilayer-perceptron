from abc import ABC, abstractmethod
import numpy as np


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
    def update(self, weights, biases, weights_gradient, bias_gradient):
        pass

    @abstractmethod
    def get_config(self):
        """
        Get the configuration of the optimizers.

        Returns:
            dict: Configuration of the optimizers.
        """
        pass

    def set_learning_rate(self, learning_rate):
        """
        Set a new learning rate for the optimizers.

        Args:
            learning_rate (float): New learning rate to be set.
        """
        self.learning_rate = learning_rate

    def get_learning_rate(self):
        """
        Get the current learning rate of the optimizers.

        Returns:
            float: Current learning rate.
        """
        return self.learning_rate

    def set_parameters(self, **kwargs):
        """
        Set additional parameters specific to the optimizers.

        Args:
            **kwargs: Additional keyword arguments specific to the optimizers.
        """
        self._init_kwargs.update(kwargs)

    def get_parameters(self):
        """
        Get the current parameters of the optimizers.

        Returns:
            dict: Current parameters of the optimizers.
        """
        return self._init_kwargs

    @staticmethod
    def clip_gradients(grads, threshold):
        """
        Clip gradients to prevent exploding gradients.

        Args:
            grads (list): List of gradients.
            threshold (float): Maximum allowed gradient value.

        Returns:
            list: Clipped gradients.
        """
        return [np.clip(g, -threshold, threshold) for g in grads]
