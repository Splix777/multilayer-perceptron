from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame


class Model(ABC):
    @abstractmethod
    def add(self, layer):
        """
        Add a layer to the model.
        """
        self._layers.append(layer)

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
    def backward(self, loss_gradients: DataFrame):
        """
        Perform the backward pass through all layers.
        """
        pass

    @abstractmethod
    def fit(self, X, epochs, val_df=None, callbacks=None) -> None:
        """
        Train the model.
        """
        pass

    @abstractmethod
    def predict(self, X: DataFrame) -> np.ndarray:
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
