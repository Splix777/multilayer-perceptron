from typing import Optional
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mlp.schemas.processed_data import ProcessedData


class DataPreprocessor:
    """DataPreprocessor class to preprocess the data."""

    def __init__(self, **kwargs) -> None:
        """
        Initializes the DataPreprocessor class with
        an optional LabelEncoder.

        This constructor allows for the initialization
        of a LabelEncoder instance, which is used to
        transform categorical labels into numerical values.
        If a custom label encoder is provided through keyword
        arguments, it will be used; otherwise, a new instance
        of LabelEncoder will be created.

        Args:
            **kwargs: Optional keyword arguments that can
                include a custom label_encoder.

        Returns:
            None
        """
        self.label_encoder: LabelEncoder = kwargs.get(
            "label_encoder", LabelEncoder()
        )

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        scaler: StandardScaler,
        shuffle: bool = True,
        seed: int = 42,
        drop_columns: Optional[list[str]] = None,
        val_split: Optional[float] = None,
        fit: bool = True,
    ) -> ProcessedData:
        """
        Process the dataset for training, evaluation, or prediction.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column.
            scaler (Optional[StandardScaler]): Scaler for standardization.
                If None, a new scaler will be created.
            shuffle (bool): Whether to shuffle the data before splitting.
            seed (int): Random seed for reproducibility.
            drop_columns (Optional[list[str]]): Columns to drop
                from the dataset.
            val_split (Optional[float]): Fraction of data to use
                for validation.
            fit (bool): Whether to fit the scaler (True for training,
                False for evaluation/prediction).

        Returns:
            ProcessedData: Object containing train, validation,
                and scaler details.
        """
        if drop_columns:
            df = df.drop(columns=drop_columns)

        df, target_encoding_map = self._encode_target_column(df, target_col)

        if fit:
            if val_split is None or val_split >= 1:
                raise ValueError("Invalid validation split fraction.")
            if val_split > 0:
                train_df, val_df = train_test_split(
                    df, test_size=val_split, shuffle=shuffle, random_state=seed
                )
                train_df = self.scale_dataframe(
                    train_df, target_col, scaler, fit=True
                )
                val_df = self.scale_dataframe(
                    val_df, target_col, scaler, fit=False
                )
            else:
                train_df = self.scale_dataframe(
                    df, target_col, scaler, fit=True
                )
                val_df = pd.DataFrame()

        else:
            train_df = pd.DataFrame()
            val_df = self.scale_dataframe(df, target_col, scaler, fit=False)

        return ProcessedData(
            train_df=train_df,
            val_df=val_df,
            scaler=scaler,
            target_encoding_map=target_encoding_map,
        )

    def _encode_target_column(
        self, df: pd.DataFrame, target_col: str
    ) -> tuple[pd.DataFrame, dict]:
        """
        Encode the target column into numeric values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column.

        Returns:
            tuple: Updated DataFrame and label mapping dictionary.
        """
        df[target_col] = self.label_encoder.fit_transform(df[target_col])
        labels: dict[str, int] = {
            label: idx for idx, label in enumerate(self.label_encoder.classes_)
        }
        return df, labels

    def scale_dataframe(
        self,
        df: pd.DataFrame,
        target_col: str,
        scaler: StandardScaler,
        fit: bool,
    ) -> pd.DataFrame:
        """
        Scale the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_col (str): Name of the target column.
            scaler (StandardScaler): Scaler for standardization.
            fit (bool): Whether to fit the scaler on the data.

        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        features: pd.DataFrame = df.drop(columns=[target_col])
        if fit:
            scaled_features = scaler.fit_transform(features)
        else:
            scaled_features = scaler.transform(features)

        scaled_df = pd.DataFrame(
            scaled_features, columns=features.columns, index=df.index
        )
        scaled_df[target_col] = df[target_col]
        return scaled_df

    def standardize_data(
        self, df: pd.DataFrame, label_col: str, scaler: StandardScaler
    ):
        """
        Standardize the data.

        Args:
            df (pd.DataFrame): DataFrame to standardize.
            label_col (str): Column name of the labels (target column).
            scaler (StandardScaler): Scaler to standardize the features.

        Returns:
            pd.DataFrame: Standardized DataFrame.
        """
        if scaler is None:
            scaler = StandardScaler()
            df = self.fit_transform(df, label_col, scaler)
        else:
            df = self.transform(df, label_col, scaler)
        return df, scaler

    @staticmethod
    def fit_transform(
        df: pd.DataFrame, label_col: str, scaler: StandardScaler
    ) -> pd.DataFrame:
        """
        Fit and transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to fit and transform.
            label_col (str): Column name of the labels (target column).
            scaler (StandardScaler): Scaler to standardize the features.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        features = df.drop(columns=[label_col]).values
        transformed_features = scaler.fit_transform(features)
        df.loc[:, df.columns != label_col] = transformed_features
        return df

    @staticmethod
    def transform(
        df: pd.DataFrame, label_col: str, scaler: StandardScaler
    ) -> pd.DataFrame:
        """
        Transform the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to transform.
            label_col (str): Column name of the labels (target column).
            scaler (StandardScaler): Scaler to standardize the features.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        features = df.drop(columns=[label_col]).values
        transformed_features = scaler.transform(features)
        df.loc[:, df.columns != label_col] = transformed_features
        return df

    @staticmethod
    def inv_transform(
        df: pd.DataFrame, label_col: str, scaler: StandardScaler
    ) -> pd.DataFrame:
        """
        Inverse transforms the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to inverse transform.
            label_col (str): Column name of the labels (target column).
            scaler (StandardScaler): Scaler to standardize the features.

        Returns:
            pd.DataFrame: Inverse transformed DataFrame.
        """
        features = df.drop(columns=[label_col]).values
        inverse_transformed_features = scaler.inverse_transform(features)
        df.loc[:, df.columns != label_col] = inverse_transformed_features
        return df
