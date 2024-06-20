import numpy as np
import h5py
import json

from pandas import DataFrame

from .model import Model
from ..layers.layer import Layer
from ..layers.input import InputLayer
from ..optimizers.optimizer import Optimizer
from ..losses.loss import Loss


class Sequential(Model):
    def __init__(self):
        super().__init__()
        self._layers = None
        self.built = False

    def add(self, layer: Layer) -> None:
        """
        Add a layer to the model.

        Args:
            layer (Layer): A layer instance.

        Returns:
            None

        Raises:
            ValueError: If the layer is not an instance of Layer.
        """
        if not isinstance(layer, Layer):
            raise ValueError('You can only add a Layer instance to the model.')

        self.built = False

        if not self._layers:
            if not isinstance(layer, InputLayer):
                if layer.input_shape is None:
                    raise ValueError(
                        "The first layer should specify the input shape.")

            self._layers = []

        if self._layers:
            layer.input_shape = self._layers[-1].output_shape

        self._layers.append(layer)
        self._layers[-1].build(layer.input_shape)

        self.built = True

    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        """
        Configure the model for training.

        Args:
        losses (Callable): A losses function that measures
            the model's performance.
        optimizers (Optimizer): An optimizers that updates the
            model's weights to minimize the losses.

        Raises:
        TypeError: If losses is not callable or optimizers
            is not an instance of a valid optimizers class.
        """
        if not callable(loss):
            raise TypeError("The losses function must be callable.")
        if not hasattr(optimizer, 'update'):
            raise TypeError("The optimizers must have an 'update' method.")
        self.loss = loss
        self.optimizer = optimizer

    def call(self, inputs: np.ndarray | list) -> np.ndarray | list:
        """
        Perform the forward pass through all layers.

        Args:
        inputs (Union[np.ndarray, list]): The input data
            or features to propagate through the layers.

        Returns:
        np.ndarray | list: The output of the model.

        Raises:
        TypeError: If inputs are not of type np.ndarray or list.
        """
        if not isinstance(inputs, (np.ndarray, list)):
            raise TypeError("Input data must be of type np.ndarray or list.")

        for layer in self._layers:
            inputs = layer.call(inputs)

        return inputs

    def backward(self, loss_gradient: np.ndarray,
                 learning_rate: float) -> np.ndarray:
        """
        Perform the backward pass through all layers.

        Args:
        loss_gradient (np.ndarray): Gradient of the losses
            with respect to the output of the model.
        learning_rate (float): Learning rate used
            for gradient descent optimization.

        Returns:
        np.ndarray: Gradient of the losses with respect
            to the input of the model (typically not used).

        Raises:
        TypeError: If loss_gradient is not a numpy array
            or learning_rate is not a float.
        """
        if not isinstance(loss_gradient, np.ndarray):
            raise TypeError("loss_gradient must be a numpy ndarray.")
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be a float.")

        for layer in reversed(self._layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

        return loss_gradient

    def fit(self, X: DataFrame, epochs: int, val_df: DataFrame = None,
            callbacks: list[object] = None) -> None:
        """
        Train the model using the provided training data
        and optionally validate using validation data.

        Args:
            X (DataFrame): Training features data.
            epochs (int): Number of epochs to train the model.
            val_df (DataFrame, optional): Validation features data
                to evaluate the model during training.
            callbacks (list of objects, optional): List of callback
                objects to monitor and possibly interrupt training.

        Raises:
            ValueError: If input data types are incorrect.
        """
        if not isinstance(X, DataFrame):
            raise ValueError("X should be a pandas DataFrame.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs should be a positive integer.")
        if val_df is not None and not isinstance(val_df, DataFrame):
            raise ValueError(
                "val_df should be a pandas DataFrame if provided.")
        if callbacks is not None and not isinstance(callbacks, list):
            raise ValueError(
                "callbacks should be a list of objects if provided.")

        # Convert DataFrame to numpy arrays
        X_train = X.values
        if 'diagnosis' not in X.columns:
            raise ValueError("The target column is missing.")
        y_train = X['diagnosis'].values

        for epoch in range(epochs):
            # Forward pass
            output = self.call(X_train)

            # Compute training losses (assuming self.losses is defined)
            loss_value = self.loss(y_train, output)

            # Compute gradient of the losses
            loss_gradient = self.loss_gradient(y_train, output)

            # Backward pass and parameter update using optimizers
            params = [layer.get_params() for layer in self._layers]
            grads = self.backward(loss_gradient,
                                  learning_rate=self.optimizer.learning_rate)

            # Convert grads to list if it is a numpy array
            if isinstance(grads, np.ndarray):
                grads = [grad.tolist() for grad in grads]
            params = self.optimizer.update(params, grads)

            # Update model parameters
            for layer, updated_params in zip(self._layers, params):
                layer.set_params(updated_params)

            # Validation if validation data is provided
            if val_df is not None:
                X_val = val_df.values
                if 'diagnosis' not in val_df.columns:
                    raise ValueError("The target column is missing.")
                y_val = val_df['diagnosis'].values
                val_output = self.call(X_val)
                val_loss = self.loss(y_val, val_output)
                print(
                    f'Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}')

            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {loss_value}')

            #TODO apply callbacks
            # if callbacks and val_df is not None:
            #     for callback in callbacks:
            #         callback.on_epoch_end(
            #             epoch,
            #             logs={
            #                 'losses': loss_value,
            #                 'val_loss': (val_loss
            #                              if val_df is not None else None)
            #             })

    def predict(self, X: DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (DataFrame): Input features for prediction.

        Returns:
            np.ndarray: Predicted output from the model.

        Raises:
            ValueError: If input data type is incorrect.
        """
        if not isinstance(X, DataFrame):
            raise ValueError("X should be a pandas DataFrame.")

        # Convert DataFrame to a numpy array for compatibility with the model
        X_pred = X.values

        # Perform forward pass through the model
        predictions = self.call(X_pred)

        return predictions

    def loss_gradient(self, y: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the losses function
        with respect to the output.

        Args:
            y (np.ndarray): True labels or target values.
            output (np.ndarray): Predicted output from the model.

        Returns:
            np.ndarray: Gradient of the losses function
                with respect to the output.

        Raises:
            ValueError: If input data shapes do not match.
        """
        return self.loss.gradient(y, output)

    def summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        for i, layer in enumerate(self._layers):
            print(
                f"Layer {i + 1}: {layer.__class__.__name__}, "
                f"Output Shape: {layer.output_shape}")

        print(f"Total Parameters: {self.count_parameters()}")

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the model.

        Returns:
            int: Total number of parameters.
        """
        total_params = 0
        for layer in self._layers:
            total_params += layer.count_parameters()

        return total_params

    # def save_model(self, filepath: str) -> None:
    #     """
    #     Save the model architecture and trained weights to HDF5 format.
    #
    #     Args:
    #         filepath (str): Filepath where the model will be saved.
    #     """
    #     with h5py.File(filepath, 'w') as f:
    #         # Save model configuration
    #         f.attrs['model_config'] = json.dumps(self.model_config)
    #
    #         # Save model weights
    #         for i, layer in enumerate(self.layers):
    #             g = f.create_group(f'layer_{i}')
    #             layer.save_weights(g)

    # @classmethod
    # def load_model(cls, filepath: str) -> 'Sequential':
    #     """
    #     Load a model from HDF5 format.
    #
    #     Args:
    #         filepath (str): Filepath from which to load the model.
    #
    #     Returns:
    #         Sequential: Loaded Sequential model instance.
    #     """
    #     with h5py.File(filepath, 'r') as f:
    #         model_config = json.loads(f.attrs['model_config'])
    #
    #         # Initialize an empty Sequential model
    #         model = Sequential()
    #
    #         # Add layers based on model configuration
    #         for layer_config in model_config['layers']:
    #             layer_class = getattr(layers_module, layer_config['class_name'])
    #             layer = layer_class.from_config(layer_config)
    #             model.add(layer)
    #
    #         # Restore model losses and optimizers
    #         loss_class = getattr(losses_module, model_config['losses'])
    #         optimizer_class = getattr(optimizers_module, model_config['optimizers'])
    #         model.compile(losses=loss_class(), optimizers=optimizer_class())
    #
    #         # Load weights for each layer
    #         for i, layer in enumerate(model.layers):
    #             if hasattr(layer, 'load_weights'):
    #                 layer.load_weights(f[f'layer_{i}'])
    #
    #     return model
