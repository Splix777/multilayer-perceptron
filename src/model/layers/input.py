import numpy as np
from .layer import Layer


class InputLayer(Layer):
    def __init__(self, input_shape: tuple[int, ...], **kwargs):
        """
        Initialize the InputLayer with the given input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.
                                 Example: (batch_size, input_dim)
        """
        super().__init__(input_shape=input_shape, trainable=False)
        self.input_shape = input_shape
        self._output_shape = input_shape

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Initialize the layer with the given input shape.
        For InputLayer, it verifies and sets the input_shape.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor (same as input_shape).

        Raises:
            ValueError: If input_shape is not a tuple or is empty.
        """
        if not isinstance(input_shape, tuple) or len(input_shape) == 0:
            raise ValueError("Input shape must be a non-empty tuple.")

        self._output_shape = input_shape
        self.built = True
        return self._output_shape

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.
        For InputLayer, it simply returns the input tensor.

        Args:
            inputs (np.ndarray): Input data or features.

        Returns:
            np.ndarray: Output tensor (same as input).

        Raises:
            ValueError: If the layer is not built.
        """
        if not self.built:
            raise ValueError("InputLayer not built. Call build() first.")

        return inputs

    def backward(self, output_gradient: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """
        Perform the backward pass.
        For InputLayer, it simply returns the output_gradient.

        Args:
            output_gradient (np.ndarray): Gradient of the
                losses with respect to the output.
            learning_rate (float): Learning rate used for
                gradient descent optimization.

        Returns:
            np.ndarray: Gradient of the losses with respect
                to the input (same as output_gradient).
        Raises:
            ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("The layer has not been built yet.")

        return output_gradient

    @property
    def output_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the output from the layer.
        For InputLayer, it returns the input_shape.

        Returns:
            tuple: Shape of the output from the layer.

        Raises:
            ValueError: If the layer has not been built.
        """
        if not self.built:
            raise ValueError("The layer has not been built yet.")

        return self._output_shape

    @property
    def config(self) -> dict[str, tuple[int, ...]]:
        """
        Get the configuration of the layer.

        Returns:
            dict: Configuration dictionary of the layer.
        """
        return {'input_shape': self.input_shape, 'trainable': self.trainable}

    def set_params(self, **params: dict[str, object]) -> None:
        """
        Set the parameters of the layer.
        InputLayer has no trainable parameters, so it does nothing.

        Args:
            **params: Keyword arguments representing layer parameters.
        """
        if 'input_shape' in params:
            self.input_shape = params['input_shape']

    def save_weights(self, filepath: str) -> None:
        """
        Save the weights of the layer to a file.
        InputLayer has no weight to save, so it does nothing.

        Args:
            filepath (str): Filepath where the weights will be saved.
        """
        pass

    def load_weights(self, filepath: str) -> None:
        """
        Load the weights of the layer from a file.
        InputLayer has no weight to load, so it does nothing.

        Args:
            filepath (str): Filepath from which the weights will be loaded.
        """
        pass

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the layer.
        InputLayer has no parameters, so it returns 0.

        Returns:
            int: Total number of parameters in the layer.
        """
        return 0

    def update_weights(self, dW: np.ndarray, db: np.ndarray):
        # The Input layer has no weights to update
        pass

    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        # The Input layer has no weights
        return np.array([]), np.array([])

    def set_weights(self, weights: tuple[np.ndarray, np.ndarray]):
        # The Input layer has no weights to set
        pass

    def initialize_parameters(self):
        # No parameters to initialize for input layer
        pass
