import json
import os
import pickle

from typing import List
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.decorators import timeit
from src.utils.file_handlers import csv_to_dataframe, json_to_dict

from src.schemas.csv_labels import CSVLabels
from src.schemas.processed_data import ProcessedData

from src.dataset_handler.data_preprocessing import DataPreprocessor
from src.neural_net.callbacks.early_stopping import EarlyStopping
from src.data_plotter.plotter import Plotter
from src.neural_net.core.sequential import Sequential
from src.schemas.sequential_config import SequentialModelConfig
from src.neural_net.layers.input import InputLayer
from src.neural_net.layers.dense import Dense
from src.neural_net.layers.dropout import Dropout


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
        self.logger: Logger = kwargs.get("logger", Logger("mlp"))
        self.config: Config = kwargs.get("config", Config())
        self.plotter: Plotter = kwargs.get(
            "plotter", Plotter(config=self.config)
        )
        self.data_processor: DataPreprocessor = kwargs.get(
            "data_processor", DataPreprocessor()
        )

    @timeit
    def train_model(self, dataset_path: Path, config_path: Path) -> None:
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
        labeled_df, labels = self._create_df_with_labels(data=data)

        # self._plot_data(data=labeled_df, labels=labels)

        proccessed_data: ProcessedData = self._preprocess_data(
            data=labeled_df,
            labels=labels,
            drop_columns=[labels.id]
        )

        model_config: dict = json_to_dict(file_path=config_path)
        validated_config = SequentialModelConfig(**model_config)
        model: Sequential = self._build_model(
            validated_config=validated_config
        )
        trained_model: Sequential = self._train_new_model(
            model=model,
            proccessed_data=proccessed_data,
            validated_config=validated_config,
        )
        # self._plot_model_history(model=trained_model, config=validated_config)

        # named_model: str = self.__save_model(
        #     model=trained_model,
        #     scaler=scaler,
        #     labels=labels,
        #     config=model_config
        # )

        # return f"Successfully trained model: {named_model}"

    def _create_df_with_labels(self, data: pd.DataFrame) -> tuple[pd.DataFrame, CSVLabels]:
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
        base_features: List[str] = self.config.config.wdbc_labels.base_features
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

        labels = CSVLabels(
            id=patient_id,
            target=diagnosis,
            features=col_names[2:]
        )

        return data, labels

    def _plot_data(self, data: pd.DataFrame, labels: CSVLabels) -> None:
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
        labels: CSVLabels,
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
            target_col=labels.target,
            scaler=scaler,
            drop_columns=drop_columns,
            val_split=val_split,
        )

    def _build_model(self, validated_config: SequentialModelConfig) -> Sequential:
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
                        kernel_regularizer=layer.kernel_regularizer
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
        self.logger.info(f"\n{model.summary()}")
        model.fit(
            X=proccessed_data.train_df,
            epochs=validated_config.epochs,
            val_data=proccessed_data.val_df,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=200,
                    verbose=True)
            ],
            batch_size=validated_config.batch_size,
            verbose=True,
        )

        return model

    def _plot_model_history(self, model: Sequential, config: SequentialModelConfig) -> None:
        """
        Plot the model training history.

        Args:
            model (Sequential): Trained model.

        Raises:
            ValueError: If the model history is empty.

        Returns:
            None
        """
        history = model.history
        model_name = config.name
        if history is None:
            raise ValueError("Model history is empty.")

        self.logger.info(f"Model history: {json.dumps(history, indent=4)}")

        df = pd.DataFrame(
            {
                "train_loss": history["loss"]["training"],
                "val_loss": history["loss"]["validation"],
                "train_accuracy": history["accuracy"]["training"],
                "val_accuracy": history["accuracy"]["validation"],
            }
        )

        plotter = Plotter(data=df, save_dir=self.config.plot_dir)

        plotter.plot_loss(model_name=model_name)
        plotter.plot_accuracy(model_name=model_name)

        self.logger.info(f"Plotted model history for: {model_name}.")

    def __save_model(
        self,
        model: Sequential,
        scaler: StandardScaler,
        labels: dict,
        config: dict,
    ) -> str:
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
        pkl_model = {
            "model": model,
            "scaler": scaler,
            "labels": labels,
            "config": config,
        }

        model_name = config.get("model_name", "model")
        model_path = f"{self.config.model_dir}/{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pkl_model, f)

        self.logger.info(f"Saved model to:\n{model_path}")

        return model_name

    def _load_model(self, model_path: str):
        """
        Load a trained model from a pickle file.

        Args:
            model_path (str): Path to the trained model pickle file.

        Returns:
            Sequential: Model.
            StandardScaler: Scaler.
            dict: Labels.
            dict: Model configuration.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        with open(model_path, "rb") as f:
            pkl_model = pickle.load(f)

        model = pkl_model["model"]
        scaler = pkl_model["scaler"]
        labels = pkl_model["labels"]
        model_config = pkl_model["config"]

        self.logger.info(
            f"Loaded model from:\n{model_path}\n"
            f"Model: {model.summary()}\n"
            f"Scaler: {scaler}\n"
            f"Labels: {labels}\n"
            f"Config: {json.dumps(model_config, indent=4)}"
        )

        return model, scaler, labels, model_config

    def evaluate_model(self, model_path: str, data_path: str) -> str:
        """
        Evaluate a trained model using the given data.

        Args:
            model_path (str): Path to the trained model pickle file.
            data_path (str): Path to the data CSV file.

        Returns:
            str: Evaluation results.
        """
        model, scaler, labels, model_config = self._load_model(
            model_path=model_path
        )
        data = self._create_df_with_labels(data_path)
        processed_data, _, _, _ = self._preprocess_data(
            data=data,
            scaler=scaler,
            target="diagnosis",
            drop_columns=["id"],
            val_split=0.0,
        )

        loss, accuracy = model.evaluate(X=processed_data)

        message = f"Model evaluation: Loss: {loss}, Accuracy: {accuracy}"

        self.logger.info(message)

        return message

    def predict(self, model_path: str, data_path: str) -> list:
        """
        Predict the target labels using the given data.

        Args:
            model_path (str): Path to the trained model pickle file.
            data_path (str): Path to the data CSV file.

        Returns:
            list: Predicted labels.
        """
        model, scaler, labels, model_config = self._load_model(
            model_path=model_path
        )
        data = self._create_df_with_labels(data_path)
        processed_data, _, _, _ = self._preprocess_data(
            data=data,
            scaler=scaler,
            target="diagnosis",
            drop_columns=["id"],
            val_split=0.0,
        )

        predictions = model.predict(X=processed_data)

        labeled_predictions = self._predictions_labels(
            predictions=predictions, labels=labels
        )

        self.logger.info(f"Predictions: {labeled_predictions}")
        loss, accuracy = model.evaluate(X=processed_data)
        # print(f"Loss: {loss}, Accuracy: {accuracy}")

        return labeled_predictions

    @staticmethod
    def _predictions_labels(predictions: np.ndarray, labels: dict) -> list:
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

        labeled_predictions = []
        for pred in predictions:
            pred = next(key for key, value in labels.items() if value == pred)
            labeled_predictions.append(pred)

        return labeled_predictions


if __name__ == "__main__":
    try:
        # "data/csv/data_train.csv"
        train_path = Path(__file__).parent / "data/csv/data_training.csv"
        test_path: Path = Path(__file__).parent / "data/csv/data_test.csv"
        # mpath: Path = Path(__file__).parent / "data/models/softmax_model.pkl"
        conf_path: Path = Path(__file__).parent / "data/models/softmax_model.json"
        # mpath = "data/models/sigmoid_model.pkl"
        # conf_path = Path(__file__).parent /  "data/models/sigmoid_model.json"

        mlp = MultiLayerPerceptron()
        mlp.train_model(dataset_path=train_path, config_path=conf_path)
        # mlp.evaluate_model(model_path=mpath, data_path=dpath)
        # print(mlp.predict(model_path=mpath, data_path=test_path))
        # print(mlp.evaluate_model(model_path=mpath, data_path=test_path))
    except Exception as e:
        print(f"Error: {e}")
        raise e
