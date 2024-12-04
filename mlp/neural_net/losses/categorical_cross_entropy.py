import numpy as np
from numpy.typing import NDArray


class CategoricalCrossEntropy:
    def __call__(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        """
        Compute the categorical cross-entropy loss.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            float: Value of the categorical cross-entropy loss.
        """
        return self.call(y_true, y_pred)

    def call(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        """
        Compute the categorical cross-entropy loss.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            float: Value of the categorical cross-entropy loss.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of y_true and y_pred must match.")

        # Clip values to avoid log(0) issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        # One-hot encode y_true if it contains class indices
        if y_true.ndim <= 1:
            num_classes: int = y_pred.shape[1]
            y_true = np.eye(num_classes)[y_true.astype(int)]

        loss = -np.mean(y_true * np.log(y_pred))
        return float(loss)

    def gradient(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of y_true and y_pred must match.")

        # Clip values to avoid log(0) issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        # One-hot encode y_true if it contains class indices
        if y_true.ndim <= 1:
            num_classes: int = y_pred.shape[1]
            y_true = np.eye(num_classes)[y_true.astype(int)]

        gradient = -y_true / y_pred
        return gradient / len(y_true)

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}
