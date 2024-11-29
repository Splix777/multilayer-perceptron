from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from mlp.utils.config import Config
from mlp.utils.file_handlers import (
    csv_to_dataframe,
    json_to_dict,
    pickle_to_file,
    file_to_pickle,
)

from mlp.schemas.csv_labels import CSVColNames
from mlp.schemas.processed_data import ProcessedData
from mlp.schemas.sequential_config import SequentialModelConfig
from mlp.schemas.packaged_model import PackagedModel

from mlp.dataset_handler.data_preprocessing import DataPreprocessor
from mlp.data_plotter.plotter import Plotter

from mlp.neural_net.core.sequential import Sequential
from mlp.neural_net.layers.input import InputLayer
from mlp.neural_net.layers.dense import Dense
from mlp.neural_net.layers.dropout import Dropout
from mlp.neural_net.callbacks.early_stopping import EarlyStopping


class MultiLayerPerceptron:
    def __init__(self, **kwargs) -> None:
        """
        Initialize the MultiLayerPerceptron object.

        Attributes:
            logger (Logger): Logger object.
            config (Config): Configuration object.
            plotter (Plotter): Plotter object.
            data_processor (DataPreprocessor): DataPreprocessor object.

        Returns:
            None
        """
        self.config: Config = kwargs.get("config", Config())
        self.plotter: Plotter = kwargs.get(
            "plotter", Plotter(config=self.config)
        )
        self.data_processor: DataPreprocessor = kwargs.get(
            "data_processor", DataPreprocessor()
        )

    # <--- CLI Methods --->
    def evaluate_model(
        self, model_path: Path, data_path: Path
    ) -> tuple[float, float]:
        """
        Evaluate a trained model using the given data.

        Args:
            model_path (str): Path to the trained model pickle file.
            data_path (str): Path to the data CSV file.

        Returns:
            str: Evaluation results.
        """
        model_package: PackagedModel = file_to_pickle(file_path=model_path)

        data: pd.DataFrame = csv_to_dataframe(file_path=data_path)
        labeled_df, _ = self._create_df_with_labels(data=data)
        processed_data: ProcessedData = self._preprocess_data(
            data=labeled_df,
            scaler=model_package.scaler,
            df_col_names=model_package.df_col_names,
            drop_columns=[model_package.df_col_names.id],
            val_split=0.0,
        )

        loss, acc = model_package.model.evaluate(X=processed_data.train_df)

        return loss, acc

    def predict(self, model_path: Path, data_path: Path) -> list[str]:
        """
        Predict the target labels using the given data.

        Args:
            model_path (str): Path to the trained model pickle file.
            data_path (str): Path to the data CSV file.

        Returns:
            list: Predicted labels.
        """
        model_package: PackagedModel = file_to_pickle(file_path=model_path)

        data: pd.DataFrame = csv_to_dataframe(file_path=data_path)
        labeled_df, _ = self._create_df_with_labels(data=data)
        processed_data: ProcessedData = self._preprocess_data(
            data=labeled_df,
            scaler=model_package.scaler,
            df_col_names=model_package.df_col_names,
            drop_columns=[model_package.df_col_names.id],
            val_split=0.0,
        )

        predictions = model_package.model.predict(X=processed_data.train_df)

        labeled_predictions = self._predictions_labels(
            predictions=predictions,
            binary_target_map=model_package.binary_target_map,
        )

        return labeled_predictions

    def train_model(
        self, config_path: Path, dataset_path: Path, plot: bool
    ) -> Path:
        """
        Train a new model using the given dataset
        and model configuration.

        Args:
            dataset_path (Path): Path to the dataset CSV file.
            config_path (Path): Path to the model
                configuration JSON file.

        Raises:
            FileNotFoundError: If the dataset path is invalid.
            ValueError: If the dataset path is empty, or
                if the dataset is missing the required columns.
            KeyError: If the model configuration is
                missing required keys.

        Returns:
            str: Success message with the name of the trained model.
        """
        data: pd.DataFrame = csv_to_dataframe(file_path=dataset_path)
        labeled_df, df_col_names = self._create_df_with_labels(data=data)

        proccessed_data: ProcessedData = self._preprocess_data(
            data=labeled_df,
            df_col_names=df_col_names,
            drop_columns=[df_col_names.id],
        )

        model_config: dict = json_to_dict(file_path=config_path)
        validated_config = SequentialModelConfig(**model_config)
        model: Sequential = self._build_model(
            validated_config=validated_config
        )
        model.summary()
        trained_model: Sequential = self._train_new_model(
            model=model,
            proccessed_data=proccessed_data,
            validated_config=validated_config,
        )
        if plot:
            self._plot_data(data=labeled_df, labels=df_col_names)
            self.plotter.plot_model_history(
                model_name=validated_config.name,
                history=trained_model.history,
            )

        saved_model_path: Path = self._save_model_to_pkl(
            model=trained_model,
            scaler=proccessed_data.scaler,
            df_col_names=df_col_names,
            binary_target_map=proccessed_data.binary_target_map,
            config=validated_config,
        )

        return saved_model_path

    # <--- Private Methods --->
    def _create_df_with_labels(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, CSVColNames]:
        """
        Create a new CSV file with labeled columns.

        Args:
            dataset_path (str): Path to the dataset CSV file.

        Raises:
            FileNotFoundError: If the dataset path is invalid.
            ValueError: If the dataset path is empty, or
                if the dataset is missing the required columns.

        Returns:
            DataFrame: Data with labeled columns.
        """
        base_features: list[str] = self.config.config.wdbc_labels.base_features
        patient_id: str = self.config.config.wdbc_labels.id
        diagnosis: str = self.config.config.wdbc_labels.diagnosis

        col_names: list[str] = (
            [patient_id, diagnosis]
            + ["mean_" + feature for feature in base_features]
            + [feature + "_se" for feature in base_features]
            + ["worst_" + feature for feature in base_features]
        )

        if len(data.columns) != len(col_names):
            raise ValueError("Dataset is missing required columns.")

        data.columns = col_names

        labels = CSVColNames(
            id=patient_id, target=diagnosis, features=col_names[2:]
        )

        return data, labels

    def _plot_data(self, data: pd.DataFrame, labels: CSVColNames) -> None:
        """
        Plot the data distribution, correlation heatmap,
        pairplot, and boxplots.

        Args:
            data (pd.DataFrame): Data to plot.

        Returns:
            None
        """
        self.plotter.target_distribution(column=labels.target, data=data)

        self.plotter.correlation_heatmap(columns=labels.features, data=data)

        self.plotter.pairplot(
            columns=[labels.target] + labels.features,
            hue=labels.target,
            data=data,
        )

        self.plotter.boxplots(
            columns=labels.features, hue=labels.target, data=data
        )

    def _preprocess_data(
        self,
        data: pd.DataFrame,
        df_col_names: CSVColNames,
        scaler: StandardScaler = StandardScaler(),
        drop_columns: list[str] = [],
        val_split: float = 0.2,
    ) -> ProcessedData:
        """
        Preprocess the data by loading, shuffling, and scaling it.

        Args:
            data (pd.DataFrame): Data to preprocess.
            scaler (StandardScaler): Scaler to use for data normalization.
            label_col (str): Column name for the target labels.
            drop_columns (list): Columns to drop from the data.
            val_split (float): Validation split ratio.

        Returns:
            pd.DataFrame: Preprocessed training data.
            pd.DataFrame: Preprocessed validation data.
            StandardScaler: Scaler used for data normalization.
            dict: Target
        """
        return self.data_processor.load_from_df(
            df=data,
            target_col=df_col_names.target,
            scaler=scaler,
            drop_columns=drop_columns,
            val_split=val_split,
        )

    def _build_model(
        self, validated_config: SequentialModelConfig
    ) -> Sequential:
        """
        Build a new model using the given configuration.

        Args:
            model_config (dict): Model configuration dictionary.

        Raises:
            KeyError: If a required key is missing.
            ValueError: If a value is invalid.

        Returns:
            Sequential: New model.
        """
        model = Sequential()
        for layer in validated_config.layers:
            if layer.type == "input":
                model.add(InputLayer(input_shape=(layer.input_shape,)))
            elif layer.type == "dense":
                model.add(
                    Dense(
                        units=layer.units,
                        activation=layer.activation,
                        kernel_initializer=layer.kernel_initializer,
                        kernel_regularizer=layer.kernel_regularizer,
                    )
                )
            elif layer.type == "dropout":
                model.add(Dropout(rate=layer.rate))

        model.compile(
            loss=validated_config.loss,
            optimizer=validated_config.optimizer.type,
            learning_rate=validated_config.optimizer.learning_rate,
        )

        return model

    def _train_new_model(
        self,
        model: Sequential,
        proccessed_data: ProcessedData,
        validated_config: SequentialModelConfig,
    ) -> Sequential:
        """
        Train a new model using the given data and configuration.

        Args:
            model (Sequential): Model to train.
            train_data (pd.DataFrame): Training data.
            val_data (pd.DataFrame): Validation data.
            config (dict): Model configuration.

        Returns:
            Sequential: Trained model.
        """
        model.fit(
            X=proccessed_data.train_df,
            epochs=validated_config.epochs,
            val_data=proccessed_data.val_df,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=200, verbose=True)
            ],
            batch_size=validated_config.batch_size,
            verbose=True,
        )

        return model

    def _save_model_to_pkl(
        self,
        model: Sequential,
        scaler: StandardScaler,
        df_col_names: CSVColNames,
        binary_target_map: dict,
        config: SequentialModelConfig,
    ) -> Path:
        """
        Save the trained model to a pickle file.

        Args:
            model (Sequential): Trained model.
            scaler (StandardScaler): Scaler used for
                data normalization.
            labels (dict): Target labels.
            config (dict): Model configuration.

        Returns:
            str: Name of the saved model.
        """
        pkl_model = PackagedModel(
            model=model,
            scaler=scaler,
            df_col_names=df_col_names,
            binary_target_map=binary_target_map,
            config=config,
        )

        model_name: str = config.name
        model_path: Path = Path(
            f"{self.config.trained_models_dir}/{model_name}.pkl"
        )
        pickle_to_file(obj=pkl_model, file_path=model_path)

        return model_path

    @staticmethod
    def _predictions_labels(
        predictions: np.ndarray, binary_target_map: dict
    ) -> list:
        """
        Convert the model predictions to target labels. Depending
        on the model output shape, the predictions are either
        converted to binary labels or to the corresponding labels.

        Args:
            predictions (np.ndarray): Model predictions.
            labels (dict): Target labels.

        Returns:
            list: Labeled predictions.
        """
        if predictions.ndim == 2:
            if predictions.shape[1] == 1:
                predictions = (predictions >= 0.5).astype(int).flatten()
            else:
                predictions = np.argmax(predictions, axis=1)

        labeled_predictions: list[str] = []
        for pred in predictions:
            pred: str = next(
                key
                for key, value in binary_target_map.items()
                if value == pred
            )
            labeled_predictions.append(pred)

        return labeled_predictions


if __name__ == "__main__":
    try:
        # CSV Data Paths
        train_path: Path = Path(__file__).parent / "data/csv/data_training.csv"
        test_path: Path = Path(__file__).parent / "data/csv/data_test.csv"

        # Softmax Model Paths
        mpath: Path = Path(__file__).parent / "data/models/softmax_model.pkl"
        conf_path: Path = (
            Path(__file__).parent / "data/models/softmax_model.json"
        )
        # Sigmoid Model Paths
        # mpath: Path = Path(__file__).parent / "data/models/sigmoid_model.pkl"
        # conf_path: Path = (
        #     Path(__file__).parent / "data/models/sigmoid_model.json"
        # )

        mlp = MultiLayerPerceptron()
        mlp.train_model(config_path=conf_path, dataset_path=train_path)
        print(mlp.predict(model_path=mpath, data_path=test_path))
        print(mlp.evaluate_model(model_path=mpath, data_path=test_path))

    except Exception as e:
        print(f"Error: {e}")
        raise e
