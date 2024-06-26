import numpy as np

from abc import ABC, abstractmethod


class Activation(ABC):
    """
    Abstract base class for defining activation functions.

    Methods:
        __call__(self, x: np.ndarray) -> np.ndarray:
            Compute the activation function given input tensor.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the activation function given input tensor.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Output tensor after applying the activation function.
        """
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function
        with respect to the input tensor.

        Args:
            x (np.ndarray): Input tensor.

        Returns:
            np.ndarray: Gradient of the activation function with respect to x.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """
        Get the configuration of the activation function.

        Returns:
            dict: Configuration of the activation function.
        """
        pass