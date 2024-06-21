import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    def __init__(self, scaler: StandardScaler = None):
        """
        Initialize the DataLoader with the given batch
        size and validation split ratio.

        Args:
            batch_size (int): Number of samples per batch.
            val_split (float): Proportion of the data
                to use for validation.
        """
        self.labels = None
        self.val_split = None
        self.label_encoder = LabelEncoder()
        self.scaler = scaler

    def dataset_from_csv(self, csv: str, label_col: str, label_mode: str = 'int',
                         shuffle: bool = True, seed: int = 42, subset: str = 'both',
                         drop_columns: list[str] = None, val_split: float = 0.2):
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
        """
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv)
        self.labels = label_col

        # Drop specified columns if provided
        if drop_columns:
            df = df.drop(columns=drop_columns)

        # Encode labels if label_mode is 'int' or 'categorical'
        if label_mode == 'int':
            df[label_col] = self.label_encoder.fit_transform(df[label_col])
        elif label_mode == 'categorical':
            df[label_col] = pd.Categorical(df[label_col])
            df[label_col] = df[label_col].cat.codes

        # Standardize the features
        df = self.fit_transform(df)

        # Split the data into training and validation sets
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

    def fit_transform(self, df: pd.DataFrame):
        """
        Fit and transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit and transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        feature_df = df.drop(columns=[self.labels])
        df[feature_df.columns] = self.scaler.fit_transform(feature_df)
        return df

    def transform(self, df: pd.DataFrame):
        """
        Transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        feature_df = df.drop(columns=[self.labels])
        df[feature_df.columns] = self.scaler.transform(feature_df)
        return df

    def inverse_transform(self, df: pd.DataFrame):
        """
        Inverse transforms the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to inverse transform.

        Returns:
            pd.DataFrame: Inverse transformed DataFrame.
        """
        feature_df = df.drop(columns=[self.labels])
        df[feature_df.columns] = self.scaler.inverse_transform(feature_df)
        return df
