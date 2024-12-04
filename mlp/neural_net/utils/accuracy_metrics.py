import numpy as np
from numpy.typing import NDArray


def binary_accuracy(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    """
    Compute binary accuracy.

    Args:
        y_true (NDArray[np.float64]): Ground truth binary labels.
        y_pred (NDArray[np.float64]): Predicted probabilities
            or binary predictions.

    Returns:
        float: Binary accuracy score.
    """
    y_true = y_true.flatten() if y_true.ndim > 1 else y_true
    y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
    return float(np.mean((y_pred >= 0.5).astype(int) == y_true))


def categorical_accuracy(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> float:
    """
    Compute categorical accuracy.

    Args:
        y_true (NDArray[np.float64]): Ground truth one-hot labels
            or class indices.
        y_pred (NDArray[np.float64]): Predicted probabilities
            or class scores.

    Returns:
        float: Categorical accuracy score.

    Raises:
        ValueError: If `y_true` and `y_pred` do not have
            the same number of samples.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            """Shape mismatch: y_true and y_pred must have the
            same number of samples."""
        )
    true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
    pred_classes = np.argmax(y_pred, axis=1)
    return float(np.mean(pred_classes == true_classes))


if __name__ == "__main__":
    # Test binary accuracy
    y_true_binary = np.array([1, 0, 1, 1])
    y_pred_binary = np.array([0.9, 0.2, 0.8, 0.4])
    print("Binary Accuracy:", binary_accuracy(y_true_binary, y_pred_binary))

    # Test categorical accuracy
    y_true_categorical = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_pred_categorical = np.array(
        [[0.2, 0.7, 0.1], [0.8, 0.1, 0.1], [0.1, 0.2, 0.7]]
    )
    print(
        "Categorical Accuracy:",
        categorical_accuracy(y_true_categorical, y_pred_categorical),
    )

    # Additional tests for edge cases
    y_true_binary_2 = np.array([[1], [0], [1], [1]])
    y_pred_binary_2 = np.array([[0.9], [0.2], [0.8], [0.4]])
    print(
        "Binary Accuracy (2D input):",
        binary_accuracy(y_true_binary_2, y_pred_binary_2),
    )

    y_true_categorical_2 = np.array([0, 1, 2])
    y_pred_categorical_2 = np.array(
        [[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]]
    )
    print(
        "Categorical Accuracy (class indices):",
        categorical_accuracy(y_true_categorical_2, y_pred_categorical_2),
    )
