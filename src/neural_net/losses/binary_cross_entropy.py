from .loss import Loss
import numpy as np
from numpy.typing import NDArray


class BinaryCrossEntropy(Loss):
    def __call__(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return self.call(y_true, y_pred)

    def call(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        if y_true.shape != y_pred.shape:
            if y_true.shape[0] == y_pred.shape[0]:
                y_true = y_true.reshape(-1, 1)
            else:
                raise ValueError("Shapes of y_true and y_pred must match.")

        # Clip values to avoid log(0) and log(1) issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate binary cross-entropy losses
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

        return float(loss)

    def gradient(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> NDArray[np.float64]:
        if y_true.shape != y_pred.shape:
            if y_true.shape[0] == y_pred.shape[0]:
                y_true = y_true.reshape(-1, 1)
            else:
                raise ValueError("Shapes of y_true and y_pred must match.")

        # Clip values to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__}
