from abc import ABC, abstractmethod
from pandas import DataFrame

class Model(ABC):
    def __init__(self):
        self._layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        """
        Add a layer to the model.
        """
        self._layers.append(layer)

    def pop(self):
        """
        Remove the last layer from the model.
        """
        self._layers.pop()

    @abstractmethod
    def compile(self, loss, optimizer):
        """
        Configure the model for training.
        """
        pass

    @abstractmethod
    def call(self, inputs):
        """
        Perform the forward pass through all layers.
        """
        pass

    @abstractmethod
    def backward(self, loss_gradient, learning_rate):
        """
        Perform the backward pass through all layers.
        """
        pass

    @abstractmethod
    def fit(self, X: DataFrame, epochs: int, val_df: DataFrame = None,
            callbacks: list[object] = None) -> None:
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions.
        """
        pass

    @abstractmethod
    def summary(self):
        """
        Print a summary of the model.
        """
        pass
