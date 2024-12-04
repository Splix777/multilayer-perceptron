from typing import Any, Optional

import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

import seaborn as sns

from mlp.utils.config import Config
from mlp.neural_net.history.model_history import History
from mlp.utils.logger import error_logger


class Plotter:
    """Plotter class for visualizing data distributions."""

    def __init__(self, **kwargs) -> None:
        """
        Plotter class for visualizing model training history
        and data distributions.

        This class provides methods to create various plots,
        including model training history, distribution plots,
        correlation heatmaps, pair plots, and box plots.
        It also handles saving or displaying the generated
        figures based on the provided configuration.

        Attributes:
            config (Config): Configuration object for the plotter.
            save_dir (Path): Directory path where plots will be saved.

        Methods:
            save_or_show(fig: Figure, output_path: Optional[str]):
                Saves the figure to the specified output path
                or displays it if no path is provided.

            target_distribution(column: str, data: pd.DataFrame):
                Creates and saves a count plot showing the
                distribution of a specified column in the data.

            correlation_heatmap(columns: list[str], data: pd.DataFrame):
                Generates and saves a heatmap representing
                the correlation matrix of specified columns in the data.

            pairplot(columns: list[str], hue: str, data: pd.DataFrame):
                Creates and saves a pair plot for the specified
                columns, colored by the specified hue.

            boxplots(columns: list[str], hue: str, data: pd.DataFrame):
                Generates and saves box plots for the specified
                columns, grouped by the specified hue.

            plot_model_history(
                model_name: str, history: History, save_path: Path):
                Plots and saves the training and validation loss and
                accuracy over epochs for a given model history.
        """
        self.config: Config = kwargs.get("config", Config())
        self.save_dir: Path = kwargs.get("save_dir", self.config.plot_dir)

    def save_or_show(self, fig: Figure, output_path: Optional[str]):
        """
        Saves a figure to a specified path or displays it
        if no path is provided.

        This method checks if a valid output path is given and if the
        save directory exists. If both conditions are met, it saves
        the figure to the specified path; otherwise, it displays the
        figure. In both cases, the figure is closed after the operation
        to free up resources.

        Args:
            fig (Figure): The figure object to be saved or displayed.
            output_path (Optional[str]): The path where the figure
                should be saved. If None, the figure will be
                displayed instead.

        Returns:
            None
        """
        if output_path and self.save_dir.is_dir():
            save_path: Path = self.save_dir / output_path
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
        Generates and saves a count plot showing the
        distribution of a specified column in the data.

        This method creates a count plot for the given column,
        visualizing the frequency of each unique value. The
        resulting figure is saved as "data_distribution.png"
        in the specified directory.

        Args:
            column (str): The name of the column for which the
                distribution is to be plotted.
            data (pd.DataFrame): The DataFrame containing the
                data to be visualized.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(
            data=data, x=column, hue=column, palette="Set2", legend=True, ax=ax
        )
        ax.set_title("Distribution of Diagnosis")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Count")
        self.save_or_show(fig=fig, output_path="data_distribution.png")

    def correlation_heatmap(self, columns: list[str], data: pd.DataFrame):
        """
        Generates and saves a heatmap representing the
        correlation matrix of specified columns in the data.

        This method calculates the correlation matrix for the
        given columns and visualizes it as a heatmap. The resulting
        figure is saved as "correlation_heatmap.png" in the
        specified directory.

        Args:
            columns (list[str]): A list of column names for which
                the correlation matrix is to be computed.
            data (pd.DataFrame): The DataFrame containing the
                data to be analyzed.

        Returns:
            None
        """
        try:
            corr_matrix: pd.DataFrame = data[columns].corr()

        except Exception as e:
            error_logger.error(f"Error generating correlation matrix: {e}")
            return

        fig, ax = plt.subplots(figsize=(16, 18))
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
            label="Correlation Heatmap", fontdict={"fontsize": 24}, pad=20
        )
        ax.set_xticklabels(labels=corr_matrix.columns, rotation=45)
        self.save_or_show(fig=fig, output_path="correlation_heatmap.png")

    def pairplot(self, columns: list[str], hue: str, data: pd.DataFrame):
        grid: sns.PairGrid = sns.pairplot(
            data[columns], hue=hue, palette="Set2"
        )
        grid.figure.suptitle("Pairplot of Selected Features", y=1.02)
        self.save_or_show(fig=grid.figure, output_path="pairplot.png")

    def boxplots(self, columns: list[str], hue: str, data: pd.DataFrame):
        """
        Generates and saves box plots for specified features,
        grouped by a specified hue.

        This method creates box plots to visualize the distribution
        of values for the given features, with the data segmented
        by the specified hue. The resulting figure is saved as
        "boxplots.png" in the specified directory.

        Args:
            columns (list[str]): A list of feature names to be plotted.
            hue (str): The name of the column used to group the
                data in the box plots.
            data (pd.DataFrame): The DataFrame containing
                the data to be visualized.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(24, 16))

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

    def plot_model_history(
        self, model_name: str, history: History, save_path: Path
    ):
        """
        Plots and saves the training and validation loss
        and accuracy over epochs for a given model history.

        This method converts the model training history into a
        DataFrame and generates animated plots for both loss
        and accuracy metrics. The resulting animation is saved
        to the specified path.

        Args:
            model_name (str): The name of the model being
                evaluated, used in the plot titles.
            history (History): An object containing
                the training history data.
            save_path (Path): The path where the animation
                will be saved.

        Returns:
            None
        """
        # Convert history to a DataFrame
        data: pd.DataFrame = history.to_dataframe()
        epochs = range(1, len(data) + 1)
        train_loss = data["train_loss"].values
        val_loss = data["val_loss"].values
        train_accuracy = data["train_accuracy"].values
        val_accuracy = data["val_accuracy"].values

        # Create figure and axes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Precompute plot limits
        max_loss = max(max(train_loss), max(val_loss)) * 1.1

        # Loss Plot
        (train_loss_line,) = axes[0].plot(
            [], [], label="Train Loss: N/A", color="blue"
        )
        (val_loss_line,) = axes[0].plot(
            [], [], label="Validation Loss: N/A", color="orange"
        )
        axes[0].set_xlim(0, len(epochs))
        axes[0].set_ylim(0, max_loss)
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"Loss Over Epochs ({model_name})")
        loss_legend = axes[0].legend()

        # Accuracy Plot
        (train_acc_line,) = axes[1].plot(
            [], [], label="Train Accuracy: N/A", color="blue"
        )
        (val_acc_line,) = axes[1].plot(
            [], [], label="Validation Accuracy: N/A", color="orange"
        )
        axes[1].set_xlim(0, len(epochs))
        axes[1].set_ylim(0, 1.1)
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title(f"Accuracy Over Epochs ({model_name})")
        acc_legend = axes[1].legend()

        # Update function for animation
        def update(frame) -> Any:
            if frame < len(epochs):
                train_loss_line.set_data(epochs[:frame], train_loss[:frame])
                val_loss_line.set_data(epochs[:frame], val_loss[:frame])
                train_acc_line.set_data(epochs[:frame], train_accuracy[:frame])
                val_acc_line.set_data(epochs[:frame], val_accuracy[:frame])

                # Update loss legend only if values change significantly
                if frame > 0:
                    loss_legend.texts[0].set_text(
                        f"Train Loss: {train_loss[frame-1]:.4f}"
                    )
                    loss_legend.texts[1].set_text(
                        f"Validation Loss: {val_loss[frame-1]:.4f}"
                    )

                    acc_legend.texts[0].set_text(
                        f"Train Accuracy: {train_accuracy[frame-1]:.4f}"
                    )
                    acc_legend.texts[1].set_text(
                        f"Validation Accuracy: {val_accuracy[frame-1]:.4f}"
                    )

            return train_loss_line, val_loss_line, train_acc_line, val_acc_line

        # Add pause frames to the animation to 'pause' at the end
        pause_frames = 240
        total_frames = len(epochs) + pause_frames

        # Create animation
        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=total_frames, interval=200, blit=True
        )

        # Save or show animation
        ani.save(save_path, writer=PillowWriter(fps=24))
