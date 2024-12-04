from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Loss(Protocol):
    def __call__(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        """
        Compute the losses value given true labels and predicted outputs.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            float: Value of the losses function.
        """
        ...

    def call(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        """
        Compute the losses value given true labels and predicted outputs.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            float: Value of the losses function.
        """
        ...

    def gradient(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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
        ...

    def get_config(self) -> dict:
        """
        Get the configuration of the losses function.

        Returns:
            dict: Configuration of the losses function.
        """
        ...
