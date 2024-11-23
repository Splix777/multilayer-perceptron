import numpy as np


class Dropout:
    def __init__(self, rate: float):
        """
        Initialize a Dropout layer with the given rate.

        Args:
            rate (float): Fraction of the input units to drop.
        """
        self.trainable = False
        self.built = False
        self.input_shape: tuple[int, ...] = (0, 0)
        self.output_shape: tuple[int, ...] = (0, 0)
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in the range [0, 1).")
        self.rate: float = rate
        self.train_mode = True
        self.mask = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.

        Args:
            inputs (np.ndarray): Input data or features.

        Returns:
            np.ndarray: Output tensor.
        """
        return self.call(inputs)

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Build the Dropout layer with the given input shape.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: Shape of the output tensor.
        """
        self.output_shape: tuple[int, ...] = input_shape
        return self.output_shape

    def call(self, inputs: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass.

        Args:
            inputs (np.ndarray): Input data or features.

        Returns:
            np.ndarray: Output tensor.
        """
        if self.train_mode:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
            return inputs * self.mask / (1 - self.rate)
        return inputs

    def backward(self, loss_gradients: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass.

        Args:
            loss_gradients (np.ndarray): Gradients of the loss
                with respect to the output of the layer.

        Returns:
            np.ndarray: Gradients of the loss
            with respect to the input.

        """
        if self.train_mode:
            return loss_gradients * self.mask / (1 - self.rate)
        return loss_gradients

    def count_parameters(self) -> int:
        """
        Count and return the number of parameters.

        Returns:
            int: Number of parameters.
        """
        return 0

    def get_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the weights of the layer.

        Returns:
            tuple: Empty tuple.
        """
        return np.array([]), np.array([])

    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """
        Set the weights and biases of the layer.

        Args:
            weights (np.ndarray): Weights of the layer.
            bias (np.ndarray): Bias of the layer.
        """
        pass
