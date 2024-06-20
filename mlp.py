import argparse
import os

import pandas as pd

from src.utils.logger import Logger
from src.utils.config import Config
from src.utils.decorators import error_handler

from src.data_plotter.plotter import Plotter
from src.dataset_handler.data_splitter import DataSplitter
from src.model.model.sequential import Sequential
from src.model.layers.input import InputLayer
from src.model.layers.dense import Dense
from src.model.losses.binary_cross_entropy import BinaryCrossEntropy
from src.model.optimizers.adam import AdamOptimizer

logger = Logger("mlp")()
config = Config()


@error_handler(handle_exceptions=(FileNotFoundError, Exception))
def get_csv_file_with_labels(filename: str) -> str:
    """
    Add column labels to the CSV file. Specifically, to csv
    files from Wisconsin Breast Cancer dataset.

    Args:
        filename: str: Path to the CSV file

    Returns:
        str: Path to the new CSV file with column labels

    Raises:
        FileNotFoundError: If the file is not found
        Exception: If any other error occurs
    """
    # Define base column names
    base_features = config.wdbc_labels['base_features']
    patient_id = config.wdbc_labels['id']
    diagnosis = config.wdbc_labels['diagnosis']

    # Define new column names
    mean_radius = ['mean_' + feature for feature in base_features]
    radius_se = [feature + '_se' for feature in base_features]
    worst_radius = ['worst_' + feature for feature in base_features]

    # Read the CSV file
    data = pd.read_csv(filename)

    # Add the new column names to the dataframe
    data.columns = (
            [patient_id, diagnosis]
            + mean_radius
            + radius_se
            + worst_radius
    )

    # Save the updated dataframe to a new CSV file
    output_filename = os.path.join(
        os.path.dirname(filename),
        os.path.basename(filename).replace('.csv', '_with_labels.csv'))
    data.to_csv(output_filename, index=False)

    return output_filename


@error_handler(handle_exceptions=(FileNotFoundError, ValueError))
def plot_data(plotter: Plotter) -> None:
    """
    Plot the data from the CSV file with column labels.

    Args:
        data_file: str: Path to the CSV file with column labels
        plotter: Plotter: An instance of the Plotter class
    """
    plotter.data_distribution(column=config.wdbc_labels['diagnosis'])

    plotter.correlation_heatmap(exclude_columns=[
        config.wdbc_labels['id'],
        config.wdbc_labels['diagnosis']
    ])

    plotter.pairplot(
        columns=(
                [config.wdbc_labels['diagnosis']]
                + ['mean_' + feature
                   for feature in config.wdbc_labels['base_features']]
        ),
        hue=config.wdbc_labels['diagnosis']
    )

    plotter.boxplots(
        columns=(
            ['mean_' + feature
             for feature in config.wdbc_labels['base_features']]
        ),
        hue=config.wdbc_labels['diagnosis']
    )


@error_handler(handle_exceptions=(FileNotFoundError, Exception))
def split_data(data_splitter: DataSplitter) -> tuple:
    """
    Split the data into training and validation sets.

    Args:
        data_splitter: DataSplitter: An instance of
            the DataSplitter class

    Returns:
        tuple: Training and validation dataframes
    """
    data_splitter.split()

    train_path = os.path.join(config.csv_directory, 'train.csv')
    val_path = os.path.join(config.csv_directory, 'val.csv')

    data_splitter.train_df.to_csv(train_path, index=False)
    data_splitter.test_df.to_csv(val_path, index=False)

    return data_splitter.train_df, data_splitter.test_df


def create_model(model: Sequential, input_shape: tuple = (500, 30)):
    model.add(InputLayer(input_shape=input_shape))
    # Hidden layers
    model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(units=64, activation='tanh', kernel_initializer='glorot_uniform'))
    model.add(Dense(units=32, activation='relu', kernel_initializer='glorot_uniform'))

    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=AdamOptimizer(), loss=BinaryCrossEntropy())


def main(data_file: str = None):
    if data_file is None:
        args = argparse.ArgumentParser()
        args.add_argument('--dataset', type=str, required=True,
                          help="Path to the dataset CSV file")
        parsed_args = args.parse_args()
        data_file = parsed_args.dataset

    if os.path.exists(data_file) and data_file.endswith('.csv'):
        df = get_csv_file_with_labels(data_file)

        # Create Plotter instance
        plotter = Plotter(
            data=pd.read_csv(df),
            save_dir=config.plot_dir
        )
        # plot_data(plotter)

        # Create DataSplitter instance
        data_splitter = DataSplitter(
            dataset=pd.read_csv(df),
            split_ratio=0.2,
            seed=42
        )
        train_df, val_df = split_data(data_splitter=data_splitter)

        model = Sequential()
        create_model(model=model, input_shape=train_df.shape)
        print(model.summary())

        model.fit(X=data_splitter.train_df, epochs=100, val_df=data_splitter.test_df)

        #TODO: Implement the following:
        # Train Model
        # Evaluate Model
        # Save Model
        # Predict Test Data

    else:
        logger.error(f"Invalid Data File: {data_file}")


if __name__ == '__main__':
    dataset = 'data/csv/data.csv'
    main(dataset)

