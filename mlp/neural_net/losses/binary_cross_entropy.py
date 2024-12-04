import numpy as np
from numpy.typing import NDArray


class BinaryCrossEntropy:
    def __call__(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        """
        Compute the binary cross-entropy loss value given true labels
        and predicted outputs.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            float: Value of the binary cross-entropy loss function.
        """
        return self.call(y_true, y_pred)

    def call(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        """
        Compute the binary cross-entropy loss value given true labels
        and predicted outputs.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            float: Value of the binary cross-entropy loss function.
        """
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

    def gradient(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute the gradient of the binary cross-entropy loss function
        with respect to the predicted output.

        Args:
            y_true (np.ndarray): True labels or target values.
            y_pred (np.ndarray): Predicted output from the model.

        Returns:
            np.ndarray: Gradient of the binary cross-entropy loss function
        """
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


if __name__ == "__main__":
    # Initialize the loss class
    bce_loss = BinaryCrossEntropy()

    # Define small test inputs
    y_true = np.array([1, 0, 1, 0], dtype=np.float64)
    y_pred = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float64)

    # Test loss calculation
    computed_loss = bce_loss.call(y_true, y_pred)
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    expected_loss = -np.mean(
        y_true * np.log(y_pred_clipped)
        + (1 - y_true) * np.log(1 - y_pred_clipped)
    )
    print(f"Computed Loss: {computed_loss}")
    print(f"Expected Loss: {expected_loss}")
    assert np.isclose(
        computed_loss, expected_loss, atol=1e-6
    ), "Loss calculation mismatch!"

    # Test gradient calculation
    computed_gradient = bce_loss.gradient(y_true, y_pred)
    expected_gradient = (y_pred_clipped - y_true) / (
        y_pred_clipped * (1 - y_pred_clipped)
    )
    print(f"Computed Gradient: {computed_gradient}")
    print(f"Expected Gradient: {expected_gradient}")
    assert np.allclose(
        computed_gradient, expected_gradient, atol=1e-6
    ), "Gradient calculation mismatch!"
