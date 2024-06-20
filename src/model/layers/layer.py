import json
from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    def __init__(self, input_shape: tuple[int, ...] = None,
                 trainable: bool = True):
        """
        Initialize the Layer with common properties.

        Args:
            trainable (bool): Whether the layer's weights
                are trainable. Defaults to True.
        """
        self.trainable = trainable

        if input_shape is not None:
            self.input_shape = input_shape

        self.built = False
        self.weights = None
        self._output_shape = None

    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(trainable={self.trainable}, "
                f"output_shape={self.output_shape})")

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(trainable={self.trainable}, "
                f"built={self.built})")

    def __len__(self) -> int:
        return self.count_parameters()

    @abstractmethod
    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Initialize the layer with the given input shape.

        Args:
            input_shape (tuple[int, ...]): Shape of the input to the layer.

        Returns:
            tuple[int, ...]: Shape of the output from the layer.
        """
        pass

    @abstractmethod
    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.

        Args:
            inputs (np.ndarray): Input data or features.

        Returns:
            np.ndarray: Output of the layer.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Perform the backward pass.

        Args:
            output_gradient (np.ndarray): Gradient of the losses with respect to the output.
            learning_rate (float): Learning rate used for gradient descent optimization.

        Returns:
            np.ndarray: Gradient of the losses with respect to the input of the layer.
        """
        pass

    @property
    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the output from the layer.

        Returns:
            tuple[int, ...]: Shape of the output from the layer.
        """
        pass

    @property
    @abstractmethod
    def config(self) -> dict[str, any]:
        """
        Get the configuration of the layer.

        Returns:
            dict[str, any]: Configuration dictionary of the layer.
        """
        pass

    @abstractmethod
    def set_params(self, **params):
        """
        Set the parameters of the layer.

        Args:
            **params: Keyword arguments representing layer parameters.
        """
        pass

    @abstractmethod
    def save_weights(self, filepath: str):
        """
        Save the weights of the layer to a file.

        Args:
            filepath (str): Filepath where the weights will be saved.
        """
        pass

    @abstractmethod
    def load_weights(self, filepath: str):
        """
        Load the weights of the layer from a file.

        Args:
            filepath (str): Filepath from which the weights will be loaded.
        """
        pass

    @abstractmethod
    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the layer.

        Returns:
            int: Total number of parameters in the layer.
        """
        return 0

    @abstractmethod
    def update_weights(self, dW: np.ndarray, db: np.ndarray):
        """
        Update the weights of the layer.

        Args:
            dW (np.ndarray): Gradient of the loss with respect to the weights.
            db (np.ndarray): Gradient of the loss with respect to the bias.
        """
        pass

    @abstractmethod
    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the weights of the layer.

        Returns:
            tuple[np.ndarray, np.ndarray]: Weights and bias of the layer.
        """
        pass

    @abstractmethod
    def set_weights(self, weights: tuple[np.ndarray, np.ndarray]):
        """
        Set the weights of the layer.

        Args:
            weights (tuple[np.ndarray, np.ndarray]):
                Weights and bias of the layer.
        """
        pass

    @abstractmethod
    def initialize_parameters(self):
        """
        Initialize the parameters of the layer. Should be called in build().
        """
        pass

    def summary(self) -> str:
        """
        Return a summary of the layer, including name,
        output shape, and number of parameters.
        """
        return (f"Layer(name={self.__class__.__name__},"
                f"output_shape={self.output_shape},"
                f"parameters={self.count_parameters()})")

    def serialize(self) -> str:
        """
        Serialize the layer configuration and
        weights to a JSON string.
        """
        config = self.config
        weights, biases = self.get_weights()
        serialized_data = {
            "config": config,
            "weights": [w.tolist() for w in weights],
            "biases": [b.tolist() for b in biases]
        }
        return json.dumps(serialized_data)

    def deserialize(self, data: str):
        """
        Deserialize the layer configuration and
        weights from a JSON string.
        """
        serialized_data = json.loads(data)
        self.set_params(**serialized_data['config'])
        weights = np.array(serialized_data['weights'])
        biases = np.array(serialized_data['biases'])
        self.set_weights((weights, biases))

    @property
    def built(self) -> bool:
        return self._built

    @built.setter
    def built(self, value: bool):
        self._built = value


