from .loss import Loss
import numpy as np


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy losses function.

    Methods:
        __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Compute the categorical cross-entropy losses given
            true labels and predicted outputs.

        gradient (self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
            Compute the gradient of categorical cross-entropy losses
            with respect to predicted outputs.

        get_config(self) -> dict:
            Get the configuration of the Categorical Cross-Entropy losses function.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy losses.

        Args:
            y_true (np.ndarray): True labels (one-hot encoded) or target
                values (integer indices).
            y_pred (np.ndarray): Predicted output from
                the model (probability distribution over classes).

        Returns:
            float: Categorical cross-entropy losses value.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of y_true and y_pred must match.")

        # Clip values to avoid log(0) issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Calculate categorical cross-entropy loss
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        loss = - np.mean(y_true * np.log(y_pred))

        return float(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of categorical cross-entropy losses.

        Args:
            y_true (np.ndarray): True labels (one-hot encoded) or target
                values (integer indices).
            y_pred (np.ndarray): Predicted output from
                the model (probability distribution over classes).

        Returns:
            np.ndarray: Gradient of categorical cross-entropy
                losses with respect to y_pred.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of y_true and y_pred must match.")

        # Clip values to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Calculate gradient of categorical cross-entropy loss
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        return - y_true / y_pred

    def get_config(self) -> dict:
        """
        Get the configuration of the Categorical Cross-Entropy losses function.

        Returns:
            dict: Configuration of the Categorical Cross-Entropy losses function.
        """
        return {
            'name': self.__class__.__name__
        }
