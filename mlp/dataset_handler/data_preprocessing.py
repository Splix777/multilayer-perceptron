import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mlp.utils.decorators import error_handler
from mlp.schemas.processed_data import ProcessedData

class DataPreprocessor:
    """
    DataPreprocessor class to preprocess the data
    before training the model.

    Attributes:
        label_encoder (LabelEncoder): Encoder for the labels.
        logger (Logger): Logger to log messages.

    Methods:
        load_from_df: Load a dataset from a CSV file,
            preprocess labels, and split it into training
            and validation sets.
        fit_transform: Fit and transform the DataFrame.
        transform: Transform the DataFrame.
        inv_transform: Inverse transforms the DataFrame.
    """
    def __init__(self) -> None:
        """
        Initialize the DataLoader with the given scaler.
        """
        self.label_encoder: LabelEncoder = LabelEncoder()

    @error_handler(exceptions_to_handle=(ValueError, KeyError))
    def load_from_df(
        self,
        df: pd.DataFrame,
        target_col: str,
        scaler: StandardScaler,
        shuffle: bool = True,
        seed: int = 42,
        drop_columns: list[str] = [],
        val_split: float = 0.2,
    ) -> ProcessedData:
        """
        Load a dataset from a DataFrame, preprocess labels,
        and split it into training and validation sets.

        Args:
            df (pd.DataFrame): DataFrame to load.
            target_col (str): Column name of the target labels.
            scaler (StandardScaler): Scaler for standardizing data.
            shuffle (bool): Whether to shuffle the data. Default is True.
            seed (int): Random seed for reproducibility. Default is 42.
            drop_columns (list): Columns to drop from the DataFrame.
            val_split (float): Proportion of the data for validation. Default is 0.2.

        Returns:
            ProcessedData: Object containing train/validation data, scaler, and labels.
        """
        if drop_columns:
            df = df.drop(columns=drop_columns)

        # Encode target column
        df, binary_target_map = self._encode_target_column(df, target_col)

        # Standardize entire dataset if no validation split
        if val_split == 0:
            df = self._scale_dataframe(df, target_col, scaler, fit=False)
            return ProcessedData(
                train_df=df,
                val_df=pd.DataFrame(),
                scaler=scaler,
                binary_target_map=binary_target_map,
            )

        train_df, val_df = train_test_split(
            df, test_size=val_split, shuffle=shuffle, random_state=seed
        )

        # Standardize train and validation sets. Scale them separately to avoid data leakage.
        train_df = self._scale_dataframe(train_df, target_col, scaler, fit=True)
        val_df = self._scale_dataframe(val_df, target_col, scaler, fit=False)

        return ProcessedData(
            train_df=train_df,
            val_df=val_df,
            scaler=scaler,
            binary_target_map=binary_target_map,
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
        labels = {
            label: idx for idx, label in enumerate(self.label_encoder.classes_)
        }
        return df, labels

    def _scale_dataframe(
        self, df: pd.DataFrame, target_col: str, scaler: StandardScaler, fit: bool
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
