import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler

from ..utils.logger import Logger
from ..utils.decorators import error_handler


class DataPreprocessor:
    def __init__(self):
        """
        Initialize the DataLoader with the given scaler.
        """
        self.label_encoder = LabelEncoder()
        self.logger = Logger("data_preprocessor")()

    @error_handler(handle_exceptions=(FileNotFoundError, ValueError, KeyError))
    def load_from_df(self, df: pd.DataFrame,
                     label_col: str,
                     shuffle: bool = True,
                     seed: int = 42,
                     drop_columns: list[str] = None,
                     scaler: StandardScaler = None,
                     val_split: float = 0.2) -> tuple:
        """
        Load a dataset from a CSV file, preprocess labels,
        and split it into training and validation sets.

        Args:
            df (pd.DataFrame): DataFrame to load.
            label_col (str): Column name of the labels (target column).
            shuffle (bool): Whether to shuffle the data
                before splitting. Default is True.
            seed (int): Random seed for shuffling. Default is 42.
            drop_columns (list): List of column names
                to drop from the DataFrame. Default is None.
            scaler (StandardScaler): Scaler to standardize
            val_split (float): Proportion of the data
                to use for validation. Default is 0.2.

        Returns:
            tuple: (train_df, val_df) if subset is 'both',
                   train_df if subset is 'train',
                   val_df if subset is 'validation'.

        Raises:
            FileNotFoundError: If the CSV file is not found.
            ValueError: If subset is not one
                of 'both', 'train', or 'validation'.
            KeyError: If the label column is not found.
        """
        if drop_columns:
            df = df.drop(columns=drop_columns)

        df[label_col] = self.label_encoder.fit_transform(df[label_col])
        original_labels = self.label_encoder.classes_
        encoded_labels = self.label_encoder.transform(original_labels)
        labels = dict(zip(original_labels, encoded_labels))

        if scaler is None:
            scaler = StandardScaler()
            df = self.fit_transform(df, label_col, scaler)
        else:
            df = self.transform(df, label_col, scaler)

        if val_split is None or val_split == 0:
            return df, None, scaler, labels

        train_df, val_df = train_test_split(
            df,
            test_size=val_split,
            shuffle=shuffle,
            random_state=seed)

        self.logger.info(f"Created training and validation sets with "
                         f"{train_df.shape[0]} and {val_df.shape[0]} samples.")

        return train_df, val_df, scaler, labels

    @staticmethod
    def fit_transform(df: pd.DataFrame, label_col: str,
                      scaler: StandardScaler) -> pd.DataFrame:
        """
        Fit and transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit and transform.
            label_col (str): Column name of the labels (target column).
            scaler (StandardScaler): Scaler to standardize the features.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        feature_df = df.drop(columns=[label_col])
        df[feature_df.columns] = scaler.fit_transform(feature_df)
        return df

    @staticmethod
    def transform(df: pd.DataFrame, label_col: str,
                  scaler: StandardScaler) -> pd.DataFrame:
        """
        Transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to transform.
            label_col (str): Column name of the labels (target column).
            scaler (StandardScaler): Scaler to standardize the features.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        feature_df = df.drop(columns=[label_col])
        df[feature_df.columns] = scaler.transform(feature_df)
        return df

    @staticmethod
    def inv_transform(df: pd.DataFrame, label_col: str,
                      scaler: StandardScaler) -> pd.DataFrame:
        """
        Inverse transforms the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to inverse transform.
            label_col (str): Column name of the labels (target column).
            scaler (StandardScaler): Scaler to standardize the features.

        Returns:
            pd.DataFrame: Inverse transformed DataFrame.
        """
        feature_df = df.drop(columns=[label_col])
        df[feature_df.columns] = scaler.inverse_transform(feature_df)
        return df
