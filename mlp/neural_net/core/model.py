from abc import ABC, abstractmethod

from numpy import ndarray
from pandas import DataFrame


class Model(ABC):
    @abstractmethod
    def add(self, layer) -> None:
        """
        Add a layer to the model.
        """
        pass

    @abstractmethod
    def compile(self, loss, optimizer) -> None:
        """
        Configure the model for training.
        """
        pass

    @abstractmethod
    def call(self, inputs) -> ndarray:
        """
        Perform the forward pass through all layers.
        """
        pass

    @abstractmethod
    def backward(self, loss_gradients: ndarray) -> None:
        """
        Perform the backward pass through all layers.
        """
        pass

    @abstractmethod
    def fit(
        self, X, epochs, val_data, callbacks, batch_size, verbose, val_split
    ) -> None:
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, X: DataFrame) -> ndarray:
        """
        Make predictions.
        """
        pass

    @abstractmethod
    def summary(self) -> str:
        """
        Print a summary of the model.
        """
        pass
