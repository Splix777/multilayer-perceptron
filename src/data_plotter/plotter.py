from typing import Generator, Tuple

import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import seaborn as sns

from src.utils.config import Config
from src.neural_net.history.model_history import History
from src.utils.logger import error_logger


class Plotter:
    """
    Plotter class to generate and save visualizations.

    Attributes:
        config (Config): Configuration object.
        logger (Logger): Logger object.
        save_dir (Path): Directory to save the plots.

    Methods:    
        __init__(**kwargs): Initialize the Plotter with the
            given configuration.
        __fig_generator(figsize: tuple): Generate Matplotlib Figure
            and Axes objects.
        save_or_show(fig: Figure, output_path: str): Save the plot
            to the output path if a valid save directory is provided,
            otherwise display the plot.
        target_distribution(column: str, data: pd.DataFrame): Plot the
            distribution of a specified column in the data.
        correlation_heatmap(columns: list[str], data: pd.DataFrame):
            Generate and display a correlation heatmap for
            the data excluding specified columns.
        pairplot(columns: list[str], hue: str, data: pd.DataFrame):
            Generate a pairplot for selected columns
            in the data with a specified hue.
        boxplots(columns: list[str], hue: str, data: pd.DataFrame):
            Generate a pairplot for selected columns
            in the data with a specified hue.
    """
    def __init__(self, **kwargs) -> None:
        """
        Initialize the Plotter with the given configuration.

        Args:
            config (Config): Configuration object.
            logger (Logger): Logger object.
            save_dir (Path): Directory to save the plots.

        Returns:
            None
        """
        self.config: Config = kwargs.get("config", Config())
        self.save_dir: Path = kwargs.get("save_dir", self.config.plot_dir)

    def __fig_generator(
        self, figsize: tuple = (8, 6)
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
                error_logger.error(f"Error generating Figure and Axes: {e}")
            finally:
                plt.close(fig)

    def save_or_show(self, fig: Figure, output_path: str = ""):
        """
        Save the plot to the output path if a valid save
        directory is provided, otherwise display the plot.

        Args:
            fig (Figure): The Matplotlib Figure object.
            output_path (str): The filename to save the plot.
                If not provided, the plot will be displayed instead.

        Returns:
            None
        """
        if output_path and self.save_dir.is_dir():
            save_path: Path = self.save_dir / output_path

            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)

            except Exception as e:
                error_logger.error(f"Error ensuring save directory exists: {e}")
                plt.close(fig)
                return

            try:
                fig.savefig(save_path)
            except Exception as e:
                error_logger.error(f"Error saving figure: {e}")
            finally:
                plt.close(fig)

        else:
            try:
                plt.show()
            except Exception as e:
                error_logger.error(f"Error displaying figure: {e}")
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
            x=column, data=data, hue=column, palette="Set2", legend=True, ax=ax
        )
        ax.set_title("Distribution of Diagnosis")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Count")

        self.save_or_show(fig=fig, output_path="data_distribution.png")

    def correlation_heatmap(self, columns: list[str], data: pd.DataFrame):
        """
        Generate and display a correlation heatmap for
        the data excluding specified columns.

        Args:
            columns (list): List of column names to correlate.
            data (pd.DataFrame): The data to be visualized.

        Returns:
            None
        """
        try:
            corr_matrix: pd.DataFrame = data[columns].corr()
        except Exception as e:
            error_logger.error(f"Error generating correlation matrix: {e}")
            return

        fig, ax = next(self.__fig_generator(figsize=(16, 18)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            robust=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=False,
            ax=ax,
        )
        ax.set_title(
            label="Correlation Heatmap",
            fontdict={"fontsize": 24},
            pad=20,
        )
        ax.set_xticklabels(
            labels=corr_matrix.columns,
            rotation=45,
        )
        self.save_or_show(fig=fig, output_path="correlation_heatmap.png")

    def pairplot(self, columns: list[str], hue: str, data: pd.DataFrame):
        """
        Generate a pairplot for selected columns
        in the data with a specified hue.

        Args:
            columns (list): List of column names
                to include in the pairplot.
            hue (str): Column name to use for coloring the plot.
            data (pd.DataFrame): The data to be visualized.

        Returns:
            None
        """
        grid: sns.PairGrid = sns.pairplot(
            data[columns],
            hue=hue,
            palette="Set2",
        )
        grid.figure.suptitle("Pairplot of Selected Features", y=1.02)
        self.save_or_show(fig=grid.figure, output_path="pairplot.png")

    def boxplots(self, columns: list[str], hue: str, data: pd.DataFrame):
        """
        Generate a pairplot for selected columns
        in the data with a specified hue.

        Args:
            columns (list): List of column names
                to include in the pairplot.
            hue (str): Column name to use for coloring the plot.
            data (pd.DataFrame): The data to be visualized.

        Returns:
            None
        """
        fig, ax = next(self.__fig_generator(figsize=(24, 16)))

        features: list[str] = columns
        data_melted: pd.DataFrame = pd.melt(
            data[[hue] + features],
            id_vars=[hue],
            var_name="feature",
            value_name="value",
        )

        sns.boxplot(
            x="feature",
            y="value",
            data=data_melted,
            hue=hue,
            palette="Set2",
            ax=ax,
        )
        tick_positions = range(len(features))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(labels=features, rotation=45)
        ax.set_title("Boxplots of Selected Features")

        self.save_or_show(fig=fig, output_path="boxplots.png")

    def plot_model_history(self, model_name: str, history: History) -> None:
        """
        Plot the training and validation metrics over epochs.

        Args:
            model_name (str): Name of the model being visualized.
            history (History): Object containing training and validation metrics.
            output_path (str): Path to save the plot. If not provided, the plot is displayed.

        Returns:
            None
        """
        data: pd.DataFrame = history.to_dataframe()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot training and validation loss
        axes[0].plot(data.index, data["train_loss"], label="Train Loss")
        axes[0].plot(data.index, data["val_loss"], label="Validation Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"Loss Over Epochs ({model_name})")
        axes[0].legend()

        # Plot training and validation accuracy
        axes[1].plot(data.index, data["train_accuracy"], label="Train Accuracy")
        axes[1].plot(data.index, data["val_accuracy"], label="Validation Accuracy")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"Accuracy Over Epochs ({model_name})")
        axes[1].legend()

        plt.tight_layout()
        self.save_or_show(fig=fig, output_path=f"{model_name}_history.png")
