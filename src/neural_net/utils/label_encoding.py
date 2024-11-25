import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def one_hot_encoding(
    X: pd.DataFrame, target_column: str = "diagnosis"
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Prepare the data for training and one-hot encode the target column.

    Args:
        X (DataFrame): Input features data.
        target_column (str): Name of the target column in the dataset (default 'diagnosis').

    Returns:
        tuple: Numpy array of features and one-hot encoded target.
    """
    # Ensure the target column exists in the DataFrame
    if target_column not in X.columns:
        raise ValueError(f"The target column '{target_column}' is missing.")

    # Ensure no missing values in the target column and features
    if X[target_column].isnull().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values."
        )

    if X.drop(columns=[target_column]).isnull().any().any():
        raise ValueError("Features contain missing values.")

    # Separate features and target
    X_feature: NDArray[np.float64] = X.drop(columns=[target_column]).to_numpy()
    y_true: NDArray[np.int_] = X[target_column].values.astype(int)

    # Check for invalid values in the target column
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError(
            f"Target column '{target_column}' must contain integers."
        )

    if np.min(y_true) < 0:
        raise ValueError(
            f"Target column '{target_column}' contains negative values."
        )

    # Determine the number of classes based on unique values in the target column
    num_classes: int = len(np.unique(y_true))

    # Ensure num_classes is greater than 1
    if num_classes <= 1:
        raise ValueError(
            f"num_classes should be greater than 1, but got {num_classes}."
        )

    # One-hot encode the labels
    y_one_hot: NDArray[np.float64] = np.eye(num_classes, dtype=np.float64)[y_true]

    return X_feature, y_one_hot


def label_encoding(
    X: pd.DataFrame, target_column: str = "diagnosis",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Label encode the target column and prepare the data for training.

    Args:
        X (DataFrame): Input features data.
        target_column (str): Name of the target column
            in the dataset (default 'diagnosis').

    Returns:
        tuple: Numpy array of features and label encoded target.
    """
    # Ensure the target column exists in the DataFrame
    if target_column not in X.columns:
        raise ValueError(f"The target column '{target_column}' is missing.")

    # Ensure no missing values in the target column and features
    if X[target_column].isnull().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values."
        )

    if X.drop(columns=[target_column]).isnull().any().any():
        raise ValueError("Features contain missing values.")

    # Separate features and target
    X_feature: NDArray[np.float64] = X.drop(columns=[target_column]).to_numpy()
    y_true: NDArray[np.int_] = X[target_column].values.astype(int)

    # Check for invalid values in the target column
    if not np.issubdtype(y_true.dtype, np.integer):
        raise ValueError(
            f"Target column '{target_column}' must contain integers."
        )

    if np.min(y_true) < 0:
        raise ValueError(
            f"Target column '{target_column}' contains negative values."
        )

    # Check if there is more than one class
    num_classes = len(np.unique(y_true))
    if num_classes <= 1:
        raise ValueError(
            f"Label encoding requires more than one class, but got {num_classes}."
        )

    # Label encode the labels (map each class to an integer)
    _, y_encoded = np.unique(y_true, return_inverse=True)

    # Convert to float64
    y_label: NDArray[np.float64] = y_encoded.astype(np.float64)

    return X_feature, y_label


if __name__ == "__main__":
    # Sample DataFrame
    sample_df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [6, 7, 8, 9, 10],
            "diagnosis": [0, 1, 0, 1, 0],
        }
    )

    # Run the function with a valid DataFrame
    try:
        X, y = one_hot_encoding(sample_df)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X:\n{X}\n\ny:\n{y}")

        X, y = label_encoding(sample_df)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X:\n{X}\n\ny:\n{y}")

    except ValueError as e:
        print(f"Error: {e}")
