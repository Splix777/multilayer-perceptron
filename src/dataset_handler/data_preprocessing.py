import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from ..utils.logger import Logger
from ..utils.decorators import error_handler


class DataPreprocessor:
    def __init__(self, scaler: StandardScaler):
        """
        Initialize the DataLoader with the given scaler.

        Args:
            scaler (StandardScaler): Scaler to standardize
                the features.
        """
        self.label_mapping = None
        self.label_encoder = LabelEncoder()
        self.scaler = scaler
        self.logger = Logger("data_preprocessor")()

    @error_handler(handle_exceptions=(FileNotFoundError, ValueError, KeyError))
    def dataset_from_csv(self, csv: str, label_col: str,
                         label_mode: str = 'int', shuffle: bool = True,
                         seed: int = 42, subset: str = 'both',
                         drop_columns: list[str] = None,
                         val_split: float = 0.2) -> tuple[pd.DataFrame, ...]:
        """
        Load a dataset from a CSV file, preprocess labels,
        and split it into training and validation sets.

        Args:
            csv (str): Path to the CSV file.
            label_col (str): Column name of the labels (target column).
            label_mode (str): Mode of the labels
                ('int' or 'categorical'). Default is 'int'.
            shuffle (bool): Whether to shuffle the data
                before splitting. Default is True.
            seed (int): Random seed for shuffling. Default is 42.
            subset (str): Which subsets to return
                ('both', 'train', 'validation'). Default is 'both'.
            drop_columns (list): List of column names
                to drop from the DataFrame. Default is None.
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
        if not 0 < val_split < 1:
            raise ValueError("val_split must be between 0 and 1.")

        df = pd.read_csv(csv)
        if df.empty:
            raise ValueError("The CSV file is empty.")

        if label_col not in df.columns:
            raise KeyError(
                f"The label column '{label_col}' "
                f"is not found in the DataFrame.")

        if drop_columns:
            if missing_cols := [
                col for col in drop_columns if col not in df.columns
            ]:
                raise KeyError(
                    f"The following columns to drop are "
                    f"not in the DataFrame: {missing_cols}")
            df = df.drop(columns=drop_columns)

        if df[label_col].isnull().any():
            raise ValueError(
                f"The label column '{label_col}' contains NaN values.")

        if label_mode not in ['int', 'categorical']:
            raise ValueError(
                "label_mode must be either 'int' or 'categorical'.")

        encoded_labels, original_labels = None, None

        if label_mode == 'categorical':
            df[label_col] = pd.Categorical(df[label_col])
            df[label_col] = df[label_col].cat.codes
            original_labels = df[label_col].astype('category').cat.categories
            encoded_labels = df[label_col].astype(
                'category').cat.codes.unique()

        elif label_mode == 'int':
            df[label_col] = self.label_encoder.fit_transform(df[label_col])
            original_labels = self.label_encoder.classes_
            encoded_labels = self.label_encoder.transform(original_labels)

        self.label_mapping = dict(zip(original_labels, encoded_labels))

        df = self.fit_transform(df, label_col)

        if subset == 'both':
            train_df, val_df = train_test_split(
                df,
                test_size=val_split,
                shuffle=shuffle,
                random_state=seed)
            return train_df, val_df

        elif subset == 'train':
            train_df, _ = train_test_split(
                df,
                test_size=val_split,
                shuffle=shuffle,
                random_state=seed)
            return train_df

        elif subset == 'validation':
            _, val_df = train_test_split(
                df,
                test_size=val_split,
                shuffle=shuffle,
                random_state=seed)
            return val_df

        else:
            raise ValueError("subset must be one of "
                             "'both', 'train', or 'validation'")

    def fit_transform(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """
        Fit and transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit and transform.
            label_col (str): Column name of the labels (target column).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        feature_df = df.drop(columns=[label_col])
        df[feature_df.columns] = self.scaler.fit_transform(feature_df)
        return df

    def transform(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """
        Transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to transform.
            label_col (str): Column name of the labels (target column).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        feature_df = df.drop(columns=[label_col])
        df[feature_df.columns] = self.scaler.transform(feature_df)
        return df

    def inv_transform(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """
        Inverse transforms the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to inverse transform.
            label_col (str): Column name of the labels (target column).

        Returns:
            pd.DataFrame: Inverse transformed DataFrame.
        """
        feature_df = df.drop(columns=[label_col])
        df[feature_df.columns] = self.scaler.inverse_transform(feature_df)
        return df
