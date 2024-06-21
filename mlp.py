import argparse
import os
import pickle

import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import StandardScaler

from src.utils.logger import Logger
from src.utils.config import Config
from src.dataset_handler.data_preprocessing import DataPreprocessor
from src.model.callbacks.early_stopping import EarlyStopping
from src.data_plotter.plotter import Plotter
from src.model.model.sequential import Sequential
from src.model.layers.input import InputLayer
from src.model.layers.dense import Dense
from src.model.layers.dropout import Dropout
from src.model.losses.binary_cross_entropy import BinaryCrossEntropy
from src.model.optimizers.adam import AdamOptimizer


class MultiLayerPerceptron:
    def __init__(self, dataset_path: str):
        self.config = Config()
        self.logger = Logger("mlp")()
        self.scaler = StandardScaler()
        self.data_processor = DataPreprocessor(scaler=self.scaler)
        # CSV Paths
        self.dataset_path = dataset_path
        self.labeled_csv_path = None
        self.train_data = None
        self.val_data = None
        # Model
        self.model = None

    def create_labeled_csv(self) -> None:
        base_features = self.config.wdbc_labels['base_features']
        patient_id = self.config.wdbc_labels['id']
        diagnosis = self.config.wdbc_labels['diagnosis']

        # Define new column names
        mean_radius = ['mean_' + feature for feature in base_features]
        radius_se = [feature + '_se' for feature in base_features]
        worst_radius = ['worst_' + feature for feature in base_features]

        # Read the CSV file
        data = pd.read_csv(self.dataset_path)

        # Add the new column names to the dataframe
        data.columns = (
                [patient_id, diagnosis]
                + mean_radius
                + radius_se
                + worst_radius
        )

        # Save the updated dataframe to a new CSV file
        output_filename = os.path.join(
            os.path.dirname(self.dataset_path),
            os.path.basename(self.dataset_path).replace(
                '.csv',
                '_with_labels.csv'))
        data.to_csv(output_filename, index=False)

        self.labeled_csv_path = output_filename

    def plot_data(self) -> None:
        plotter = Plotter(
            data=pd.read_csv(self.labeled_csv_path),
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

    def preprocess_data(self):
        train_df, val_df = self.data_processor.dataset_from_csv(
            csv=self.labeled_csv_path,
            label_col=self.config.wdbc_labels['diagnosis'],
            seed=69,
            subset='both',
            drop_columns=[self.config.wdbc_labels['id']],
            val_split=0.2
        )

        train_df.to_csv(os.path.join(self.config.csv_directory,
                                     'train.csv'), index=False)
        val_df.to_csv(os.path.join(self.config.csv_directory,
                                   'val.csv'), index=False)

        self.train_data = train_df
        self.val_data = val_df

    def build_model(self):
        model = Sequential()

        # Input layer
        model.add(InputLayer(input_shape=self.train_data
                             .values[:, 1:].shape[1:]))

        # Hidden layers
        model.add(Dense(units=32, activation='relu',
                        kernel_initializer='glorot_uniform'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=64, activation='relu',
                        kernel_initializer='glorot_uniform'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=32, activation='relu',
                        kernel_initializer='glorot_uniform'))
        model.add(Dropout(rate=0.5))

        # Output layer
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(
            optimizer=AdamOptimizer(),
            losses=BinaryCrossEntropy(),
        )

        self.model = model

        print(model.summary())

    def train_model(self):
        self.model.fit(
            X=self.train_data,
            epochs=100,
            val_data=self.val_data,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=False
            )],
            batch_size=32,
            verbose=True
        )

    def evaluate_model(self):
        self.model.evaluate(self.val_data)

    def save_model(self):
        model_json = {"model": self.model, "scaler": self.scaler}
        with open(f"{self.config.model_dir}/model.pkl", 'wb') as f:
            pickle.dump(model_json, f)

    def load_model(self):
        with open(f"{self.config.model_dir}/model.pkl", 'rb') as f:
            model_json = pickle.load(f)
        self.model = model_json["model"]
        self.scaler = model_json["scaler"]

    def run(self):
        self.create_labeled_csv()
        # self.plot_data()
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        self.load_model()
        self.evaluate_model()


if __name__ == '__main__':
    dataset = 'data/csv/data.csv'
    mlp = MultiLayerPerceptron(dataset_path=dataset)
    mlp.run()

