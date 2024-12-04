from typing import Generator

import numpy as np
from numpy.typing import NDArray


def shuffle_data(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Shuffle the input data.

    Args:
        x (NDArray[np.float64]): Features data.
        y (NDArray[np.float64]): Target data.

    Returns:
            tuple: Shuffled features and target data.
    """
    # Ensure the number of samples in features and target are equal
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "The number of samples in features and target not equal."
        )

    # Generate random indices
    random_indices: NDArray[np.int64] = np.random.permutation(x.shape[0])

    # Shuffle the features and target data
    x_shuffled: NDArray[np.float64] = x[random_indices]
    y_shuffled: NDArray[np.float64] | NDArray[np.intp] = y[random_indices]

    return x_shuffled, y_shuffled


def iter_batches(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    batch_size: int = 32,
    shuffle: bool = True,
) -> Generator[
    tuple[NDArray[np.float64], NDArray[np.float64]],
    None,
    None
]:
    """
    Generate mini-batches of data for training or evaluation.

    Args:
        X (NDArray[np.float64]): Features data.
        y (NDArray[np.float64] | NDArray[np.intp]): Target data.
        batch_size (int): Size of mini-batches (default 32).
        shuffle (bool): Whether to shuffle the data (default True

    Yields:
        Generator: Mini-batches of features and target data.
    """
    if shuffle:
        X, y = shuffle_data(X, y)

    for i in range(0, X.shape[0], batch_size):
        X_batch: NDArray[np.float64] = X[i: i + batch_size]
        y_batch: NDArray[np.float64] | NDArray[np.intp] = y[i: i + batch_size]
        yield X_batch, y_batch


if __name__ == "__main__":
    X: NDArray[np.float64] = np.random.rand(100, 10)
    y: NDArray[np.float64] = np.random.rand(100, 1)

    for X_batch, y_batch in iter_batches(X, y, batch_size=32, shuffle=True):
        print(X_batch.shape, y_batch.shape)
        break
