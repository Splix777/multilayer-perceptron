from .loss import Loss
import numpy as np


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy (Log Loss) losses function.

    Methods:
        __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Compute the binary cross-entropy losses given
            true labels and predicted outputs.

        gradient (self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            Compute the gradient of binary cross-entropy losses
            with respect to predicted outputs.

        get_config(self) -> dict:
            Get the configuration of the
            Binary Cross-Entropy losses function.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the binary cross-entropy losses.

        Args:
            y_true (np.ndarray): True labels or target
                values (binary: 0 or 1).
            y_pred (np.ndarray): Predicted output from
                the model (probability between 0 and 1).

        Returns:
            float: Binary cross-entropy losses value.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must match.")

        # Clip values to avoid log(0) and log(1) issues
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calculate binary cross-entropy losses
        loss = - np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true)
            * np.log(1 - y_pred)
        )

        return float(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of binary cross-entropy losses.

        Args:
            y_true (np.ndarray): True labels or target
                values (binary: 0 or 1).
            y_pred (np.ndarray): Predicted output
                from the model (probability between 0 and 1).

        Returns:
            np.ndarray: Gradient of binary cross-entropy
                losses with respect to y_pred.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("Shapes of y_true and y_pred must match.")

        # Clip values to avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # Calculate gradient of binary cross-entropy losses
        gradient = (y_pred - y_true) / (y_pred * (1 - y_pred))

        return gradient

    def get_config(self) -> dict:
        """
        Get the configuration of the Binary Cross-Entropy losses function.

        Returns:
            dict: Configuration of the Binary Cross-Entropy losses function.
        """
        return {
            'name': self.__class__.__name__
        }
