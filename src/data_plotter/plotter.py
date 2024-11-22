from typing import Any, Union, Generator, Tuple

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import seaborn as sns

from src.utils.config import Config
from src.utils.logger import Logger


class Plotter:
    """
    A class to generate and save plots for data visualization.

    Attributes:
        config (Config): Config object to load configuration settings.
        logger (Logger): Logger object to log messages.
        save_dir (Path): Path to the directory to save the plots.

    Methods:
        save_or_show: Save or show the plot based
            on the save_dir attribute.
        data_distribution: Plot the distribution
            of a specified column in the data.
        correlation_heatmap: Generate and display
            a correlation heatmap for the data.
        pairplot: Generate a pairplot for
            selected columns in the data.
        boxplots: Generate boxplots for selected
            columns in the data.
        plot_loss: Plot the training and validation loss
            over epochs for a specified model.
        plot_accuracy: Plot the training and validation
            accuracy over epochs for a specified model.
    """
    def __init__(self, **kwargs) -> None:
        self.config: Config = kwargs.get('config', Config())
        self.logger: Logger = kwargs.get('logger', Logger('plotter'))
        self.save_dir: Path = kwargs.get('save_dir', self.config.plot_dir)

    def __fig_generator(
            self,
            figsize: tuple = (8, 6)
            ) -> Generator[Tuple[Figure, Axes], None, None]:
        """
        Generate Matplotlib Figure and Axes objects.

        Args:
            figsize (tuple): Tuple of the figure size (width, height).

        Yields:
            Generator[Tuple[Figure, Axes], None, None]: A generator yielding Figure and Axes objects.
        """
        while True:
            fig: Figure
            ax: Axes

            fig, ax = plt.subplots(figsize=figsize)
            try:
                yield fig, ax
            except Exception as e:
                self.logger.info(f"Error generating Figure and Axes: {e}")
            finally:
                plt.close(fig)

    def save_or_show(self, fig: Figure, output_path: str = ""):
        """
        Save the plot to the output path if a valid save
        directory is provided, otherwise display the plot.

        Args:
            fig (Figure): The Matplotlib Figure object.
            output_path (str): The filename to save the plot. =
                If not provided, the plot will be displayed instead.

        Returns:
            None
        """
        if output_path and self.save_dir.is_dir():
            save_path: Path = self.save_dir / output_path

            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)

            except Exception as e:
                self.logger.info(f"Error ensuring save directory exists: {e}")
                plt.close(fig)
                return

            try:
                fig.savefig(save_path)
                self.logger.info(f"Figure saved to {save_path}")
            except Exception as e:
                self.logger.info(f"Error saving figure: {e}")
            finally:
                plt.close(fig)

        else:
            try:
                plt.show()
            except Exception as e:
                self.logger.info(f"Error displaying figure: {e}")
            finally:
                plt.close(fig)
        
    def target_distribution(self, column: str, data: pd.DataFrame):
        """
        Plot the distribution of a specified column in the data.

        Args:
            column (str): The column name for which the
                distribution is to be plotted.
            data (pd.DataFrame): The data to be visualized.

        Returns:
            None
        """
        fig, ax = next(self.__fig_generator(figsize=(8, 6)))

        sns.countplot(
            x=column,
            data=data,
            hue=column,
            palette='Set2',
            legend=True,
            ax=ax
            )
        ax.set_title('Distribution of Diagnosis')
        ax.set_xlabel('Diagnosis')
        ax.set_ylabel('Count')

        self.save_or_show(fig=fig, output_path='data_distribution.png')

    def correlation_heatmap(self, columns: list[str], data: pd.DataFrame):
        """
        Generate and display a correlation heatmap for
        the data excluding specified columns.

        Args:
            exclude_columns (list): List of column names
                to exclude from the correlation calculation.

        Returns:
            None
        """
        try:
            corr_matrix: pd.DataFrame = data[columns].corr()
        except Exception as e:
            self.logger.info(f"Error generating correlation matrix: {e}")
            return

        fig, ax = next(self.__fig_generator(figsize=(16, 18)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            robust=True,
            fmt='.2f',
            cmap='coolwarm',
            cbar=False,
            ax=ax
            )
        ax.set_title(
            label='Correlation Heatmap',
            fontdict={'fontsize': 24},
            pad=20,
            )
        ax.set_xticklabels(
            labels=corr_matrix.columns,
            rotation=45,
        )
        self.save_or_show(fig=fig, output_path='correlation_heatmap.png')

    def pairplot(self, columns: list[str], hue: str, data: pd.DataFrame):
        """
        Generate a pairplot for selected columns
        in the data with a specified hue.

        Args:
            columns (list): List of column names
                to include in the pairplot.
            hue (str): Column name to use for coloring the plot.

        Returns:
            None
        """
        grid: sns.PairGrid = sns.pairplot(
            data[columns],
            hue=hue,
            palette='Set2',
            )
        grid.figure.suptitle('Pairplot of Selected Features', y=1.02)
        self.save_or_show(fig=grid.figure, output_path='pairplot.png')  

    def boxplots(self, columns: list[str], hue: str, data: pd.DataFrame):
        """
        Generate a pairplot for selected columns
        in the data with a specified hue.

        Args:
            columns (list): List of column names
                to include in the pairplot.
            hue (str): Column name to use for coloring the plot.

        Returns:
            None
        """
        fig, ax = next(self.__fig_generator(figsize=(24, 16)))
        
        features: list[str] = columns
        data_melted: pd.DataFrame = pd.melt(
            data[[hue] + features],
            id_vars=[hue],
            var_name='feature',
            value_name='value'
        )

        sns.boxplot(
            x='feature',
            y='value',
            data=data_melted,
            hue=hue,
            palette='Set2',
            ax=ax
        )
        ax.set_title('Boxplots of Selected Features')
        ax.set_xticklabels(
            labels=features,
            rotation=45
        )

        self.save_or_show(fig=fig, output_path='boxplots.png')

    # def plot_loss(self, model_name: str) -> None:
    #     """
    #     Plot the training and validation loss over epochs.

    #     Returns:
    #         None
    #     """
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.data['train_loss'], label='Training Loss')
    #     plt.plot(self.data['val_loss'], label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     min_val_loss = min(self.data['val_loss'])
    #     plt.title(
    #         f'Training and Validation Loss for {model_name} - '
    #         f'Min Validation Loss: {min_val_loss:.4f}'
    #     )
    #     plt.legend()
    #     plt.grid(True)
    #     plt.legend()

    #     self.save_or_show(plot=plt, output_path=f'{model_name}_loss.png')

    # def plot_accuracy(self, model_name: str) -> None:
    #     """
    #     Plot the training and validation accuracy over epochs.

    #     Returns:
    #         None
    #     """
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.data['train_accuracy'], label='Training Accuracy')
    #     plt.plot(self.data['val_accuracy'], label='Validation Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     max_val_acc_percent = max(self.data['val_accuracy']) * 100
    #     plt.title(f'Training and Validation Accuracy for {model_name} - '
    #               f'Max Validation Accuracy: {max_val_acc_percent:.0f}%')
    #     plt.legend()
    #     plt.grid(True)

    #     self.save_or_show(plot=plt, output_path=f'{model_name}_accuracy.png')
