from typing import Protocol

from numpy import ndarray
from pandas import DataFrame


class Model(Protocol):
    def add(self, layer) -> None:
        """
        Add a layer to the model.
        """
        ...

    def compile(self, loss, optimizer, learning_rate) -> None:
        """
        Configure the model for training.
        """
        ...

    def call(self, inputs) -> ndarray:
        """
        Perform the forward pass through all layers.
        """
        ...

    def backward(self, loss_gradients: ndarray) -> None:
        """
        Perform the backward pass through all layers.
        """
        ...

    def fit(
        self, X, epochs, val_data, callbacks, batch_size, verbose, val_split
    ) -> None:
        """
        Train the model.
        """
        ...

    def predict(self, X: DataFrame) -> ndarray:
        """
        Make predictions.
        """
        ...

    def summary(self) -> str:
        """
        Print a summary of the model.
        """
        ...
