import numpy as np

from abc import ABC, abstractmethod


class Loss(ABC):
    """
    Abstract base class for defining losses functions.

    Methods:
        __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Compute the losses value given true labels and predicted outputs.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the losses value given true labels and predicted outputs.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            float: Value of the losses function.
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the losses function
        with respect to the predicted output.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            np.ndarray: Gradient of the losses function
            with respect to y_pred.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """
        Get the configuration of the losses function.

        Returns:
            dict: Configuration of the losses function.
        """
        pass
