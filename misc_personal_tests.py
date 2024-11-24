import numpy as np


def shuffle_data(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Shuffle the input data.

    Args:
            x (ndarray): Input features data.
            y (ndarray): Target data.

    Returns:
            tuple: Shuffled features and target data.
    """
    # Ensure the number of samples in features and target are equal
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "The number of samples in features and target not equal."
        )

    # Generate random indices
    random_indices = np.random.permutation(x.shape[0])

    # Shuffle the features and target data
    x_shuffled = x[random_indices]
    y_shuffled = y[random_indices]

    return x_shuffled, y_shuffled


if __name__ == "__main__":
    # Create some dummy data
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])

    # Shuffle the data
    x_shuffled, y_shuffled = shuffle_data(x, y)

    # Print the shuffled data
    print("Shuffled features:")
    print(x_shuffled)
    print("Shuffled target:")
    print(y_shuffled)
