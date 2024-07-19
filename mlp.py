import json
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.utils.config import Config
from src.utils.logger import Logger
from src.utils.decorators import error_handler
from src.dataset_handler.data_preprocessing import DataPreprocessor
from src.model.callbacks.early_stopping import EarlyStopping
from src.data_plotter.plotter import Plotter
from src.model.model.sequential import Sequential
from src.model.layers.input import InputLayer
from src.model.layers.dense import Dense
from src.model.layers.dropout import Dropout


class MultiLayerPerceptron:
    """
    MultiLayerPerceptron class to train, evaluate,
    and predict using a Sequential model.

    Attributes:
        config (Config): Configuration object.
        logger (Logger): Logger object.
        data_processor (DataPreprocessor): DataPreprocessor object.

    Methods:
        train_model: Train a new model using the given dataset
            and model configuration.
        _create_labeled_data: Create a new df with labeled columns.
        _plot_data: Plot the data distribution, correlation heatmap,
            pairplot, and boxplots.
        _preprocess_data: Preprocess the data by
            loading, shuffling, and scaling it.
        _load_model_config: Load the model configuration
            from the given path.
        _check_model_config: Check the model configuration
            for required keys and values.
        _build_model: Build a new model using the given configuration.
        _train_new_model: Train a new model using the given
            data and configuration.
        _plot_model_history: Plot the model training history.
        _save_model: Save the trained model to a pickle file.
        _load_model: Load a trained model from a pickle file.
        evaluate_model: Evaluate a trained model using the given data.
        predict: Predict the target labels using the given data.
        _predictions_labels: Convert the model predictions
            to target labels.
    """
    def __init__(self):
        self.config = Config()
        self.logger = Logger("mlp")()
        self.data_processor = DataPreprocessor()

    @error_handler(handle_exceptions=(FileNotFoundError, ValueError, KeyError))
    def train_model(self, dataset_path: str, config_path: str) -> str:
        """
        Train a new model using the given dataset
        and model configuration.

        Args:
            dataset_path (str): Path to the dataset CSV file.
            config_path (str): Path to the model
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
        self.logger.info("Training model...")
        data = self._create_labeled_data(dataset_path=dataset_path)
        # self._plot_data(data=data)
        train_df, val_df, scaler, labels = self._preprocess_data(data=data)
        model_config = self._load_model_config(config_path=config_path)
        model = self._build_model(model_config=model_config)
        trained_model = self._train_new_model(
            model=model,
            train_data=train_df,
            val_data=val_df,
            config=model_config
        )
        self._plot_model_history(model=trained_model, config=model_config)
        named_model = self._save_model(
            model=trained_model,
            scaler=scaler,
            labels=labels,
            config=model_config
        )

        return f"Successfully trained model: {named_model}"

    def _create_labeled_data(self, dataset_path: str) -> pd.DataFrame:
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
        data = pd.read_csv(dataset_path)

        self.logger.info(
            f"Loaded dataset from: {dataset_path}: "
            f"Shape {data.shape}.",
        )

        base_features = self.config.wdbc_labels['base_features']
        patient_id = self.config.wdbc_labels['id']
        diagnosis = self.config.wdbc_labels['diagnosis']

        # Define new column names
        mean_radius = ['mean_' + feature for feature in base_features]
        radius_se = [feature + '_se' for feature in base_features]
        worst_radius = ['worst_' + feature for feature in base_features]

        # Add the new column names to the dataframe
        data.columns = (
                [patient_id, diagnosis]
                + mean_radius
                + radius_se
                + worst_radius
        )

        self.logger.info(f"Updated column names: {list(data.columns)}")

        return data

    def _plot_data(self, data: pd.DataFrame) -> None:
        """
        Plot the data distribution, correlation heatmap,
        pairplot, and boxplots.

        Args:
            data (pd.DataFrame): Data to plot.

        Returns:
            None
        """
        plotter = Plotter(
            data=data,
            save_dir=self.config.plot_dir
        )

        plotter.data_distribution(column=self.config.wdbc_labels['diagnosis'])

        plotter.correlation_heatmap(exclude_columns=[
            self.config.wdbc_labels['id'],
            self.config.wdbc_labels['diagnosis']
        ])

        plotter.pairplot(
            columns=(
                    [self.config.wdbc_labels['diagnosis']]
                    + ['mean_' + feature
                       for feature in self.config.wdbc_labels['base_features']]
            ),
            hue=self.config.wdbc_labels['diagnosis']
        )

        plotter.boxplots(
            columns=(
                ['mean_' + feature
                 for feature in self.config.wdbc_labels['base_features']]
            ),
            hue=self.config.wdbc_labels['diagnosis']
        )

        self.logger.info("Plotted Data Successfully.")

    def _preprocess_data(self, data: pd.DataFrame,
                         scaler: StandardScaler = None,
                         label_col: str = None,
                         drop_columns: list = None,
                         val_split: float = 0.2
                         ) -> tuple:
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
        label_col = label_col or self.config.wdbc_labels['diagnosis']
        drop_columns = drop_columns or [self.config.wdbc_labels['id']]

        train_df, val_df, scaler, labels = self.data_processor.load_from_df(
            df=data,
            label_col=label_col,
            shuffle=True,
            seed=69,
            drop_columns=drop_columns,
            scaler=scaler,
            val_split=val_split
        )

        return train_df, val_df, scaler, labels

    def _load_model_config(self, config_path: str) -> dict:
        """
        Load the model configuration from the given path.

        Args:
            config_path (str): Path to the model
                configuration JSON file.

        Returns:
            dict: Model configuration.
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.logger.info(f"Loaded model config from: {config_path}.")
        self.logger.info(f"Model config: {json.dumps(config, indent=4)}")

        return config

    @staticmethod
    def _check_model_config(model_config: dict) -> tuple:
        """
        Check the model configuration for required keys and values.

        Args:
            model_config (dict): Model configuration dictionary.

        Raises:
            KeyError: If a required key is missing.
            ValueError: If a value is invalid.

        Returns:
            list: Model layers.
            str: Optimizer type.
            str: Loss function.
        """
        required_keys = ['model_name', 'layers', 'optimizer', 'loss', 'epochs']
        for key in required_keys:
            if key not in model_config:
                raise KeyError(f"Missing required key: {key}")

        layers = model_config.get('layers', [])
        if len(layers) < 2:
            raise ValueError(
                "Model must have at least an input and output layer.")

        for layer in layers:
            layer_type = layer.get('type')
            if layer_type not in ['input', 'dense', 'dropout']:
                raise ValueError("Invalid layer type.")

            if layer_type == 'input' and 'input_shape' not in layer:
                raise KeyError("Input layer must have an input shape.")

            if layer_type == 'dense':
                if 'units' not in layer:
                    raise KeyError("Dense layer must have units.")
                if 'activation' not in layer:
                    layer['activation'] = 'relu'
                if 'kernel_initializer' not in layer:
                    layer['kernel_initializer'] = 'glorot_uniform'

            if layer_type == 'dropout' and 'rate' not in layer:
                raise KeyError("Dropout layer must have a rate.")

        optimizer = model_config.get('optimizer')
        if 'type' not in optimizer or 'learning_rate' not in optimizer:
            raise KeyError("Optimizer must have a type.")
        optimizer_type = optimizer.get('type')
        if optimizer_type not in ['adam', 'rmsprop']:
            raise ValueError("Invalid optimizer.")

        loss = model_config.get('loss')
        if loss not in ['binary_crossentropy', 'categorical_crossentropy']:
            raise ValueError("Invalid loss function.")

        return layers, optimizer_type, loss

    def _build_model(self, model_config: dict) -> Sequential:
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
        layers, optimizer, loss = self._check_model_config(
            model_config=model_config
        )

        model = Sequential()
        for layer in layers:
            layer_type = layer.get('type')
            if layer_type == 'input':
                model.add(InputLayer(
                    input_shape=(layer.get('input_shape', 30),)
                ))
            elif layer_type == 'dense':
                model.add(Dense(
                    units=layer.get('units'),
                    activation=layer.get('activation'),
                    kernel_initializer=layer.get('kernel_initializer'),
                    kernel_regularizer=layer.get('kernel_regularizer')
                ))
            elif layer_type == 'dropout':
                model.add(Dropout(
                    rate=layer.get('rate')
                ))

        model.compile(
            optimizer=optimizer,
            learning_rate=model_config.get('optimizer').get('learning_rate'),
            loss=loss
        )

        self.logger.info(f"Model Summary:\n{model.summary()}")

        return model

    @staticmethod
    def _train_new_model(model: Sequential,
                         train_data: pd.DataFrame,
                         val_data: pd.DataFrame,
                         config: dict) -> Sequential:
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
            X=train_data,
            epochs=config.get('epochs', 1_000),
            val_data=val_data,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=1000,
                verbose=True
            )],
            batch_size=config.get('batch_size', 32),
            verbose=True
        )

        return model

    def _plot_model_history(self, model: Sequential, config: dict) -> None:
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
        model_name = config.get('model_name', 'model')
        if history is None:
            raise ValueError("Model history is empty.")

        self.logger.info(f"Model history: {json.dumps(history, indent=4)}")

        df = pd.DataFrame({
            'train_loss': history['loss']['training'],
            'val_loss': history['loss']['validation'],
            'train_accuracy': history['accuracy']['training'],
            'val_accuracy': history['accuracy']['validation']
        })

        plotter = Plotter(
            data=df,
            save_dir=self.config.plot_dir
        )

        plotter.plot_loss(model_name=model_name)
        plotter.plot_accuracy(model_name=model_name)

        self.logger.info(f"Plotted model history for: {model_name}.")

    def _save_model(self, model: Sequential,
                    scaler: StandardScaler,
                    labels: dict,
                    config: dict) -> str:
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

        model_name = config.get('model_name', 'model')
        model_path = f"{self.config.model_dir}/{model_name}.pkl"
        with open(model_path, 'wb') as f:
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

        with open(model_path, 'rb') as f:
            pkl_model = pickle.load(f)

        model = pkl_model["model"]
        scaler = pkl_model["scaler"]
        labels = pkl_model["labels"]
        model_config = pkl_model["config"]

        self.logger.info(f"Loaded model from:\n{model_path}\n"
                         f"Model: {model.summary()}\n"
                         f"Scaler: {scaler}\n"
                         f"Labels: {labels}\n"
                         f"Config: {json.dumps(model_config, indent=4)}")

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
        data = self._create_labeled_data(data_path)
        processed_data, _, _, _ = self._preprocess_data(
            data=data,
            scaler=scaler,
            label_col='diagnosis',
            drop_columns=['id'],
            val_split=0.0
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
        data = self._create_labeled_data(data_path)
        processed_data, _, _, _ = self._preprocess_data(
            data=data,
            scaler=scaler,
            label_col='diagnosis',
            drop_columns=['id'],
            val_split=0.0
        )

        predictions = model.predict(X=processed_data)

        labeled_predictions = self._predictions_labels(
            predictions=predictions,
            labels=labels
        )

        self.logger.info(f"Predictions: {labeled_predictions}")
        loss, accuracy = model.evaluate(X=processed_data)
        print(f"Loss: {loss}, Accuracy: {accuracy}")

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
    train_path = "data/csv/data_training.csv"
    test_path = "data/csv/data_test.csv"
    # mpath = "data/models/softmax_model.pkl"
    # conf_path = "data/models/softmax_model.json"
    mpath = "data/models/sigmoid_model.pkl"
    conf_path = "data/models/sigmoid_model.json"

    mlp = MultiLayerPerceptron()
    mlp.train_model(dataset_path=train_path, config_path=conf_path)
    # mlp.evaluate_model(model_path=mpath, data_path=dpath)
    print(mlp.predict(model_path=mpath, data_path=test_path))
