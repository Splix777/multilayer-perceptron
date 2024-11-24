from typing import Generator
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.model_selection import train_test_split
from pydantic import ValidationError

from src.schemas.fit_params import FitParams
from src.schemas.split_data import SplitData

from src.neural_net.core.model import Model
from src.neural_net.layers.layer import Layer
from src.neural_net.layers.dropout import Dropout
from src.neural_net.layers.input import InputLayer
from src.neural_net.optimizers.optimizer import Optimizer
from src.neural_net.optimizers.adam import AdamOptimizer
from src.neural_net.optimizers.rms_prop import RMSpropOptimizer
from src.neural_net.losses.loss import Loss
from src.neural_net.losses.binary_cross_entropy import BinaryCrossEntropy
from src.neural_net.losses.categorical_cross_entropy import (
    CategoricalCrossEntropy,
)
from src.neural_net.callbacks.callback import Callback

from src.utils.logger import Logger
from src.neural_net.utils.label_encoding import (
    one_hot_encoding,
    label_encoding,
)
from src.neural_net.utils.data_batch_utils import shuffle_data, iter_batches


class Sequential(Model):
    def __init__(self) -> None:
        self.logger: Logger = Logger("Sequential")
        self.losses = {"training": {}, "validation": {}}
        self.accuracy = {"training": {}, "validation": {}}

        self.layers: list[Layer] = []
        self.callbacks: list[Callback] = []
        self.loss: Loss = None
        self.built = False
        self.stop_condition = False
        self.dropout_active = True
        self.training_mode = True

    # <-- Add Layers -->
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
            self.logger.error(
                f"Layer: {layer.__class__.__name__} is not an instance of Layer."
            )
            raise ValueError("You can only add a Layer instance to the model.")

        # The First layer should specify the input shape
        if len(self.layers) == 0:
            if not isinstance(layer, InputLayer) or layer.input_shape is None:
                raise ValueError("First layer should specify the input shape.")

        if len(self.layers) > 0:
            layer.input_shape = self.layers[-1].output_shape

        layer.build(layer.input_shape)
        self.layers.append(layer)

        if len(self.layers) > 1:
            self.built = True

    # <-- Compile Model -->
    def compile(self, loss: str, optimizer: str, learning_rate: float = 0.001):
        """
        Configure the model for training.

        Args:
        loss (str): Name of the loss function to use.
        optimizer (Optimizer | str): Optimizer instance
            or name of the optimizer to use.
        learning_rate (float): Learning rate for the optimizer.

        Raises:
        ValueError: If the model is not built or the loss
            is not recognized.
        TypeError: If the optimizer type is unsupported.
        """
        if not self.built:
            raise ValueError(
                "You must add layers to the model before compiling."
            )

        self.loss: Loss = (
            CategoricalCrossEntropy()
            if loss == "categorical_crossentropy"
            else BinaryCrossEntropy()
        )

        for layer in self.layers:
            if layer.trainable:
                layer.optimizer = (
                    AdamOptimizer(learning_rate)
                    if optimizer == "adam"
                    else RMSpropOptimizer(learning_rate)
                )

    # <-- Forward and Backward Pass -->
    def call(self, inputs: np.ndarray | list) -> np.ndarray:
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

        for layer in self.layers:
            inputs = layer.call(inputs)
            if layer.trainable and self.training_mode:
                inputs = self._apply_regularization(layer, inputs)

        return inputs

    def backward(self, loss_gradients: np.ndarray) -> None:
        """
        Perform the backward pass through all layers.

        Args:
        loss_gradient (np.ndarray): Gradient of the losses
            with respect to the output of the model.

        Returns:
        np.ndarray: Gradient of the losses with respect
            to the input of the model (typically not used).

        Raises:
        TypeError: If loss_gradient is not a numpy array
            or learning_rate is not a float.
        """
        for layer in reversed(self.layers):
            loss_gradients = layer.backward(loss_gradients)
            if layer.trainable:
                self._update_epoch_weights(layer)

    # <-- Training Methods -->
    def fit(
        self,
        X: pd.DataFrame,
        epochs: int,
        val_data: pd.DataFrame,
        callbacks: list[Callback] = [],
        batch_size: int = 32,
        verbose: bool = False,
        val_split: float = 0.2,
        batch_size_mode: str = "fixed",
        min_batch_size: int = 16,
        max_batch_size: int = 128,
        batch_size_factor: float = 1.1,
    ) -> None:
        """
        Train the model using the provided training data
        and optionally validate using validation data.

        Args:
            X (DataFrame): Training features data.
            epochs (int): Number of epochs to train the model.
            val_split (float, optional): Fraction of the training
                data to be used for validation. Defaults to None.
            val_data (DataFrame, optional): Validation features data.
                Defaults to None.
            callbacks (list of objects, optional): List of callback
                objects to monitor and possibly interrupt training.
            batch_size (int, optional): Number of samples per batch.
                Defaults to 32.
            verbose (bool, optional): Whether to print training
                progress. Defaults to False.
            batch_size_mode (str, optional): Mode for batch size
                adjustment. Must be 'auto' or 'fixed'. Defaults to 'fixed'.
            min_batch_size (int, optional): Minimum batch size allowed
                in 'auto' mode. Defaults to 16.
            max_batch_size (int, optional): Maximum batch size allowed
                in 'auto' mode. Defaults to 128.
            batch_size_factor (float, optional): Scaling factor for
                batch size. Defaults to 1.1.

        Raises:
            ValueError: If input data types are incorrect.
        """
        params: FitParams = self._prepare_fit(
            X=X,
            epochs=epochs,
            val_split=val_split,
            val_data=val_data,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=verbose,
            batch_size_mode=batch_size_mode,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            batch_size_factor=batch_size_factor,
        )
        split_data: SplitData = self._split_data(
            params.X, params.val_data, params.val_split
        )

        previous_train_loss = None

        for epoch in range(params.epochs):
            if self.stop_condition:
                break

            train_loss = []
            train_accuracy = []

            for X_batch, y_batch in iter_batches(split_data.X_train, split_data.y_train, batch_size):
                loss_value, accuracy = self._train_batch(X_batch, y_batch)
                train_loss.append(loss_value)
                train_accuracy.append(accuracy)

            val_loss, val_accuracy = self._eval_validation_data(X_val, y_val)

            log = self._update_metrics(
                train_loss, train_accuracy, val_loss, val_accuracy, epoch
            )

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Accuracy: {log['accuracy']:.3f}, "
                    f"Loss: {log['loss']:.3f} -- "
                    f"Val Accuracy: {log['val_accuracy']:.3f}, "
                    f"Val Loss: {log['val_loss']:.3f}"
                )

            # Adjust batch size if necessary
            if batch_size_mode == "auto":
                if previous_train_loss is not None:
                    # If loss decreased significantly, increase the batch size
                    if train_loss[-1] < previous_train_loss:
                        new_batch_size = int(batch_size * batch_size_factor)
                        batch_size = min(new_batch_size, max_batch_size)
                    # If loss increased, decrease the batch size
                    elif train_loss[-1] > previous_train_loss:
                        new_batch_size = int(batch_size / batch_size_factor)
                        batch_size = max(new_batch_size, min_batch_size)

                # Store the current epoch's loss for comparison in the next epoch
                previous_train_loss = train_loss[-1]

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=log)

        for callback in callbacks:
            callback.on_train_end()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (DataFrame): Input features for prediction.

        Returns:
            np.ndarray: Predicted output from the model.

        Raises:
            ValueError: If an input data type is incorrect.
        """
        self._deactivate_train_mode()
        X_eval, _ = self._one_hot_encoding(X)
        predictions = self.call(X_eval)
        self.dropout_active = True
        return predictions

    def evaluate(self, X: pd.DataFrame) -> tuple[float, float]:
        """
        Evaluate the model using the provided data.

        Args:
            X (DataFrame): Input features data.

        Returns:
            tuple: Tuple of losses and accuracy.
        """
        self._deactivate_train_mode()
        if self.model_output_units > 1:
            X_eval, y_eval = self._one_hot_encoding(X)
        else:
            X_eval, y_eval = self._label_encoding(X)
        predictions = self.call(X_eval)
        loss = self.loss(y_eval, predictions)
        if isinstance(self.loss, CategoricalCrossEntropy):
            accuracy = self._categorical_accuracy(y_eval, predictions)
        else:
            accuracy = self._binary_accuracy(y_eval, predictions)
        self.dropout_active = True
        return loss, accuracy

    # <-- Epoch Methods -->
    def _train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Train the model on a single batch of data.

        Args:
            X_batch (np.ndarray): Input features data.
            y_batch (np.ndarray): Target labels data.
        """
        output = self.call(X_batch)

        loss_value = self.loss(y_batch, output)
        loss_gradients = self.loss.gradient(y_batch, output)

        self.backward(loss_gradients)

        if isinstance(self.loss, CategoricalCrossEntropy):
            accuracy = self._categorical_accuracy(y_batch, output)
        else:
            accuracy = self._binary_accuracy(y_batch, output)

        return loss_value, accuracy

    @staticmethod
    def _update_epoch_weights(layer: Layer):
        """
        Update the weights of the layer for the current epoch.

        Args:
            layer (Layer): The layer to update the weights for.
        """
        if layer.kernel_regularizer:
            layer.weights_gradients += layer.kernel_regularizer.gradient(
                layer.weights
            )
            layer.bias_gradients += layer.kernel_regularizer.gradient(
                layer.bias
            )
        layer.weights, layer.bias = layer.optimizer.update(
            weights=layer.weights,
            bias=layer.bias,
            weights_gradients=layer.weights_gradients,
            bias_gradients=layer.bias_gradients,
        )

    @staticmethod
    def _apply_regularization(layer: Layer, inputs: np.ndarray) -> np.ndarray:
        """
        Apply regularization to the layer weights.

        Args:
            layer (Layer): The layer to apply regularization to.
            inputs (np.ndarray): Input data or features.
        """
        if layer.kernel_regularizer:
            regularization_penalty = layer.kernel_regularizer(layer.weights)
            inputs += regularization_penalty

        return inputs

    def _deactivate_train_mode(self):
        """
        Deactivate the training mode.
        """
        self.dropout_active = False
        self.training_mode = False

    def _activate_train_mode(self):
        """
        Activate the training mode.
        """
        self.dropout_active = True
        self.training_mode = True

    def _eval_validation_data(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Test the model on the validation data.

        Args:
            X_val (np.ndarray): Validation input features data.
            y_val (np.ndarray): Validation target labels data.

        Returns:
            Tuple[float, float]: Validation loss and accuracy.
        """
        self._deactivate_train_mode()
        val_pred = self.call(X_val)
        val_loss = self.loss(y_val, val_pred)

        if self.loss == CategoricalCrossEntropy:
            val_accuracy = self._categorical_accuracy(y_val, val_pred)
        else:
            val_accuracy = self._binary_accuracy(y_val, val_pred)

        self._activate_train_mode()
        return val_loss, val_accuracy

    def _binary_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the accuracy of the model.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy of the model.
        """
        if self.model_output_units > 1:
            return np.mean((y_pred >= 0.5).astype(int) == y_true)
        return np.mean((y_pred >= 0.5).astype(int).flatten() == y_true)

    @staticmethod
    def _categorical_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the accuracy of the model.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy of the model.
        """
        true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return float(accuracy)

    # <-- Getters and Setters -->
    def get_weights(self):
        """
        Get the weights of the model.

        Returns:
            list: List of weights for each layer.
        """
        return [layer.get_weights() for layer in self._layers]

    def set_weights(self, weights: list):
        """
        Set the weights of the model.

        Args:
            weights (list): List of weights for each layer.
        """
        for layer, (w, b) in zip(self.layers, weights):
            layer.set_weights(weights=w, bias=b)

    # <-- Validation and Initializer -->
    def _prepare_fit(
        self,
        X: pd.DataFrame,
        epochs: int,
        val_split: float,
        val_data: pd.DataFrame,
        callbacks: list[Callback],
        batch_size: int,
        verbose: bool,
        batch_size_mode: str,
        min_batch_size: int,
        max_batch_size: int,
        batch_size_factor: float,
    ) -> FitParams:
        params = FitParams(
            X=X,
            epochs=epochs,
            val_split=val_split,
            val_data=val_data,
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=verbose,
            batch_size_mode=batch_size_mode,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            batch_size_factor=batch_size_factor,
        )
        # Validation
        self._validate_inputs(**params.model_dump())
        # Callbacks initialization
        self._init_callbacks(params.callbacks or [])
        return params

    def _validate_inputs(
        self,
        X: pd.DataFrame,
        epochs: int,
        val_split: float,
        val_data: pd.DataFrame,
        callbacks: list[Callback],
        batch_size: int,
        verbose: bool,
        batch_size_mode: str,
        min_batch_size: int,
        max_batch_size: int,
        batch_size_factor: float,
    ) -> None:
        """
        Validate inputs for the fit method.

        Raises:
            ValueError: If input data types or values are incorrect.
        """
        try:
            params = FitParams(
                X=X,
                epochs=epochs,
                val_split=val_split,
                val_data=val_data,
                callbacks=callbacks,
                batch_size=batch_size,
                verbose=verbose,
                batch_size_mode=batch_size_mode,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                batch_size_factor=batch_size_factor,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid inputs to fit method: {e}")

        self.epochs: int = params.epochs
        self.val_split: float | None = params.val_split
        self.batch_size: int = params.batch_size
        self.verbose: bool = params.verbose

    def _init_callbacks(self, callbacks: list[Callback]) -> None:
        """
        Initialize the callbacks.

        Args:
            callbacks (list of objects): List of callback objects.
        """
        if len(callbacks) == 0:
            return
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()

        self.callbacks: list[Callback] = callbacks

    def _split_data(
        self, X: pd.DataFrame, val_data: pd.DataFrame, val_split: float
    ) -> SplitData:
        """
        Split the data into training and validation sets.

        Args:
            X (DataFrame): Input features data.
            val_split (float): Fraction of the training data
                to be used for validation.
            val_data (DataFrame): Validation features data.

        Returns:
            tuple: Tuple of training and validation data.
        """
        if val_data.empty:
            X, val_data = train_test_split(X, test_size=val_split)

        # For Softmax output layer (multi-class classification)
        if self.model_output_units > 1:
            X_train, y_train = one_hot_encoding(X)
            X_val, y_val = one_hot_encoding(val_data)
        # For Sigmoid output layer (binary classification)
        else:
            X_train, y_train = label_encoding(X)
            X_val, y_val = label_encoding(val_data)

        return SplitData(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
        )

    # <-- Metrics and Logging -->
    def _update_metrics(
        self,
        losses: list[float],
        metrics: list[float],
        val_loss: float,
        val_accuracy: float,
        epoch: int,
    ):
        """
        Update the metrics for the model.

        Args:
            losses (list): List of losses.
            metrics (list): List of metrics.
            val_loss (float): Validation loss.
            val_accuracy (float): Validation accuracy.

        Returns:
            dict: Dictionary of updated metrics.
        """
        self.losses["training"][epoch] = np.mean(losses)
        self.accuracy["training"][epoch] = np.mean(metrics)
        self.losses["validation"][epoch] = val_loss
        self.accuracy["validation"][epoch] = val_accuracy

        return {
            "loss": np.mean(losses),
            "accuracy": np.mean(metrics),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

    @property
    def history(self) -> dict:
        """
        Return the training history of the model.

        Returns:
            dict: Dictionary of training history.
        """
        return {"loss": self.losses, "accuracy": self.accuracy}

    @property
    def model_output_units(self) -> int:
        """
        Return the number of output units of the model.

        Returns:
            int: Number of output units.
        """
        return self.layers[-1].output_shape[1]

    # <-- Dropout Inference Mode -->
    @property
    def dropout_active(self):
        """
        Return the dropout active mode.
        """
        return self._dropout_active

    @dropout_active.setter
    def dropout_active(self, value):
        """
        Set the dropout active mode.

        Args:
            value (bool): Whether to activate dropout (True)
                or deactivate dropout (False).
        """
        self._dropout_active = bool(value)
        self._set_dropout_inference_mode(train_mode=self._dropout_active)

    def _set_dropout_inference_mode(self, train_mode=True):
        """
        Set all Dropout layers to the specified mode.

        Args:
            train_mode (bool): Whether to activate dropout (True)
                or deactivate dropout (False).
        """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train_mode = train_mode

    # <-- Summary and Layer Parameters -->
    def summary(self) -> str:
        """
        Print a summary of the model architecture.
        """
        summary = ""
        for i, layer in enumerate(self.layers):
            output_shape = layer.output_shape
            parameters = layer.count_parameters()
            summary += (
                f"Layer {i + 1} {layer.__class__.__name__}: "
                f"Trainable: {layer.trainable}, "
                f"Output Shape: {output_shape}, "
                f"Parameters: {parameters}\n"
            )

        summary += f"Total Parameters: {self.count_parameters()}"
        return summary

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the model.

        Returns:
            int: Total number of parameters.
        """
        return sum(layer.count_parameters() for layer in self.layers)
