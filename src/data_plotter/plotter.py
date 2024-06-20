import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def __init__(self, data: pd.DataFrame, save_dir: str = None):
        self.data = data
        self.save_dir = save_dir

    def save_or_show(self, plot, output_path):
        save_path = f'{self.save_dir}/{output_path}' if self.save_dir else None
        plot.savefig(save_path) if self.save_dir else plot.show()
        plt.close()

    def data_distribution(self, column: str):
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

    def correlation_heatmap(self, exclude_columns: list):
        plt.figure(figsize=(14, 12))
        corr_matrix = self.data.drop(columns=exclude_columns).corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')

        self.save_or_show(plot=plt, output_path='correlation_heatmap.png')

    def pairplot(self, columns: list, hue: str):
        sns.pairplot(self.data[columns], hue=hue, palette='Set2')
        plt.suptitle('Pairplot of Selected Features', y=1.02)

        self.save_or_show(plot=plt, output_path='pairplot.png')

    def boxplots(self, columns: list, hue: str):
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
