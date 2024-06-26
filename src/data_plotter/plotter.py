import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    """
    A class to generate and save plots for data visualization.

    Attributes:
        data (pd.DataFrame): The data to be visualized.
        save_dir (str): The directory to save the plots.

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
    def __init__(self, data: pd.DataFrame, save_dir: str = None):
        self.data = data
        self.save_dir = save_dir

    def save_or_show(self, plot: plt, output_path: str) -> None:
        """
        Save the plot to the output path if save_dir is provided,
        otherwise show the plot.

        Args:
            plot (plt): Matplotlib plot object.
            output_path (str): Output path to save the plot

        Returns:
            None
        """
        save_path = f'{self.save_dir}/{output_path}' if self.save_dir else None
        plot.savefig(save_path) if self.save_dir else plot.show()
        plt.close()

    def data_distribution(self, column: str) -> None:
        """
        Plot the distribution of a specified column in the data.

        Args:
            column (str): The column name for which the
                distribution is to be plotted.

        Returns:
            None
        """
        plt.figure(figsize=(8, 6))
        sns.countplot(
            x=column,
            data=self.data,
            hue=column,
            palette='Set2',
            legend=False)
        plt.title('Distribution of Diagnosis')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')

        self.save_or_show(plot=plt, output_path='data_distribution.png')

    def correlation_heatmap(self, exclude_columns: list) -> None:
        """
        Generate and display a correlation heatmap for
        the data excluding specified columns.

        Args:
            exclude_columns (list): List of column names
                to exclude from the correlation calculation.

        Returns:
            None
        """
        plt.figure(figsize=(14, 12))
        corr_matrix = self.data.drop(columns=exclude_columns).corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')

        self.save_or_show(plot=plt, output_path='correlation_heatmap.png')

    def pairplot(self, columns: list, hue: str) -> None:
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
        sns.pairplot(self.data[columns], hue=hue, palette='Set2')
        plt.suptitle('Pairplot of Selected Features', y=1.02)

        self.save_or_show(plot=plt, output_path='pairplot.png')

    def boxplots(self, columns: list, hue: str) -> None:
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
        features = columns
        data_melted = pd.melt(
            self.data[[hue] + features],
            id_vars=hue,
            var_name='feature',
            value_name='value'
        )

        plt.figure(figsize=(14, 8))
        sns.boxplot(
            x='feature',
            y='value',
            data=data_melted,
            hue=hue,
            palette='Set2'
        )
        plt.title('Boxplots of Selected Features')

        self.save_or_show(plot=plt, output_path='boxplots.png')

    def plot_loss(self, model_name: str) -> None:
        """
        Plot the training and validation loss over epochs.

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.data['train_loss'], label='Training Loss')
        plt.plot(self.data['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        min_val_loss = min(self.data['val_loss'])
        plt.title(
            f'Training and Validation Loss for {model_name} - '
            f'Min Validation Loss: {min_val_loss:.4f}'
        )
        plt.legend()
        plt.grid(True)
        plt.legend()

        self.save_or_show(plot=plt, output_path=f'{model_name}_loss.png')

    def plot_accuracy(self, model_name: str) -> None:
        """
        Plot the training and validation accuracy over epochs.

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.data['train_accuracy'], label='Training Accuracy')
        plt.plot(self.data['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        max_val_acc_percent = max(self.data['val_accuracy']) * 100
        plt.title(f'Training and Validation Accuracy for {model_name} - '
                  f'Max Validation Accuracy: {max_val_acc_percent:.0f}%')
        plt.legend()
        plt.grid(True)

        self.save_or_show(plot=plt, output_path=f'{model_name}_accuracy.png')
