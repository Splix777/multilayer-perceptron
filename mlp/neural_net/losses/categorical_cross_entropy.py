from .loss import Loss
import numpy as np
from numpy.typing import NDArray


class CategoricalCrossEntropy(Loss):
    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.call(y_true, y_pred)

    def call(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of y_true and y_pred must match.")

        # Clip values to avoid log(0) issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        if y_true.ndim <= 1:
            num_classes = y_pred.shape[1]
            y_true = np.eye(num_classes)[y_true]

        loss = -np.mean(y_true * np.log(y_pred))
        return float(loss)

    def gradient(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(y_true) != len(y_pred):
            raise ValueError("Lengths of y_true and y_pred must match.")

        # Clip values to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        if y_true.ndim <= 1:
            num_classes = y_pred.shape[1]
            y_true = np.eye(num_classes)[y_true]

        gradient = -y_true / y_pred
        return gradient / len(y_true)

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}
