import pandas as pd


class DataSplitter:
    def __init__(self, dataset: pd.DataFrame, split_ratio: float = 0.2, seed: int = 42):
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.seed = seed
        self.train_dataset = None
        self.test_dataset = None

    def __str__(self):
        return (f"Train dataset shape: "
                f"{self.train_dataset.shape}"
                f"\nTest dataset shape: {self.test_dataset.shape}")

    def split(self):
        self.dataset['diagnosis'] = self.dataset['diagnosis'].apply(
            lambda x: 1 if x == 'M' else 0)

        self.train_dataset = self.dataset.sample(
            frac=1 - self.split_ratio,
            random_state=self.seed).drop('id', axis=1)

        self.test_dataset = self.dataset.drop(self.train_dataset.index).drop('id', axis=1)

    @property
    def train_df(self):
        return self.train_dataset

    @property
    def test_df(self):
        return self.test_dataset
