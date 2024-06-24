import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler

from src.utils.logger import Logger
from src.utils.config import Config
from src.utils.decorators import error_handler
from src.dataset_handler.data_preprocessing import DataPreprocessor
from src.model.callbacks.early_stopping import EarlyStopping
from src.data_plotter.plotter import Plotter
from src.model.model.sequential import Sequential
from src.model.layers.input import InputLayer
from src.model.layers.dense import Dense
from src.model.layers.dropout import Dropout
from src.model.losses.binary_cross_entropy import BinaryCrossEntropy
from src.model.losses.categorical_cross_entropy import CategoricalCrossEntropy
from src.model.optimizers.adam import AdamOptimizer


class MultiLayerPerceptron:
    def __init__(self):
        self.config = Config()
        self.logger = Logger("mlp")()
        self.data_processor = DataPreprocessor()

    @error_handler(handle_exceptions=(FileNotFoundError, ValueError, KeyError))
    def train_model(self, dataset_path: str, config_path: str = None):
        data = self._create_labeled_csv(dataset_path=dataset_path)
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
        named_model = self._save_model(
            model=trained_model,
            scaler=scaler,
            labels=labels,
            config=model_config,
            val_data=val_df
        )

        return f"Successfully trained model: {named_model}"

    def _create_labeled_csv(self, dataset_path: str) -> DataFrame:
        """
        Create a new CSV file with labeled columns.

        Raises:
            FileNotFoundError: If the dataset path is invalid.
            ValueError: If the dataset path is empty, or
                if the dataset is missing the required columns.
        """
        data = pd.read_csv(dataset_path)

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

        return data

    def _plot_data(self, data: pd.DataFrame) -> None:
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

    def _preprocess_data(self, data: pd.DataFrame,
                         scaler: StandardScaler = None,
                         label_col: str = None,
                         drop_columns: list = None,
                         val_split: float = 0.2
                         ):
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

    def _load_model_config(self, config_path: str = None):
        path = config_path or f"{self.config.model_dir}/softmax_model.json"
        # path = config_path or f"{self.config.model_dir}/sigmoid_model.json"

        with open(path, 'r') as f:
            config = json.load(f)

        return config

    @staticmethod
    def _check_model_config(model_config: dict):
        required_keys = ['optimizer', 'loss', 'layers']
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
        if optimizer_type not in ['adam']:
            raise ValueError("Invalid optimizer.")

        loss = model_config.get('loss')
        if loss not in ['binary_crossentropy', 'categorical_crossentropy']:
            raise ValueError("Invalid loss function.")

        return layers, optimizer_type, loss

    def _build_model(self, model_config: dict):
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
                    kernel_initializer=layer.get('kernel_initializer')
                ))
            elif layer_type == 'dropout':
                model.add(Dropout(
                    rate=layer.get('rate')
                ))
        model.compile(
            optimizer=optimizer,
            loss=loss
        )

        print(model.summary())
        return model

    @staticmethod
    def _train_new_model(model: Sequential,
                         train_data: pd.DataFrame,
                         val_data: pd.DataFrame,
                         config: dict):
        model.fit(
            X=train_data,
            epochs=config.get('epochs', 1_000),
            val_data=val_data,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=15,
                verbose=True
            )],
            batch_size=config.get('batch_size', 32),
            verbose=True
        )

        return model

    def _save_model(self, model: Sequential,
                    scaler: StandardScaler,
                    labels: dict,
                    config: dict,
                    val_data: pd.DataFrame):
        model_json = {
            "model": model,
            "scaler": scaler,
            "labels": labels,
            "config": config,
            "val_data": val_data
        }

        model_name = config.get('model_name', 'model')
        with open(f"{self.config.model_dir}/{model_name}.pkl", 'wb') as f:
            pickle.dump(model_json, f)

        return model_name



    def evaluate_model(self):
        self.model.evaluate(self.val_data)

    def predict(self, data_path: str):
        """
        Make predictions using the trained model.

        Args:
            data_path (pd.DataFrame | pd.Series): Data to make predictions on.

        Returns:
            None
        """
        if not isinstance(data_path, str) and data_path.endswith('.csv'):
            raise ValueError("Data must be a CSV file.")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        data = self.preprocess_predict_data(data_path)

        predictions = []
        for _, row in data.iterrows():
            prediction = self.model.predict(row)
            prediction = 1 if prediction >= 0.5 else 0
            prediction_label = next(
                (
                    key
                    for key, value in self.target_binary_labels.items()
                    if prediction == value
                ),
                None,
            )
            predictions.append(prediction_label)

        original_data = pd.read_csv(data_path)
        original_data = original_data.iloc[:, 1].values

        total_correct = sum(
            original == prediction
            for original, prediction in zip(original_data, predictions)
        )
        print(f"Predictions: {predictions}")
        print(f"Accuracy: {total_correct / len(predictions) * 100:.2f}%")

    def load_model(self):
        with open(f"{self.config.model_dir}/model.pkl", 'rb') as f:
            model_json = pickle.load(f)
        self.model = model_json["model"]
        self.scaler = model_json["scaler"]
        self.target_binary_labels = model_json["labels"]

    def run(self):
        self._create_labeled_csv(self.dataset_path)
        # self.plot_data()
        self._preprocess_data()
        self._build_model()
        self._train_model()
        self.evaluate_model()
        self._save_model()

        self.load_model()
        self.evaluate_model()
        self.predict('data/csv/data.csv')

    def load_dataset(self, dataset_path: str):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File not found: {dataset_path}")

        self.dataset_path = dataset_path


if __name__ == '__main__':
    dataset = 'data/csv/data.csv'
    mlp = MultiLayerPerceptron()
    mlp.train_model(dataset)
