import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from .model import Model
from ..layers.layer import Layer
from ..layers.dropout import Dropout
from ..layers.input import InputLayer
from ..optimizers.optimizer import Optimizer
from ..optimizers.adam import AdamOptimizer
from ..optimizers.rms_prop import RMSpropOptimizer
from ..losses.loss import Loss
from ..losses.binary_cross_entropy import BinaryCrossEntropy
from ..losses.categorical_cross_entropy import CategoricalCrossEntropy
from ..callbacks.callback import Callback
from src.utils.logger import Logger


class Sequential(Model):
    def __init__(self):
        super().__init__()
        self.logger = Logger("Sequential")()
        self.losses = {'training': {}, 'validation': {}}
        self.accuracy = {'training': {}, 'validation': {}}
        self.loss = None
        self._layers = None
        self.built = False
        self.callbacks = None
        self.stop_training = False
        self._dropout_active = True
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
            raise ValueError('You can only add a Layer instance to the model.')

        # The First layer should specify the input shape
        if not self._layers:
            if not isinstance(layer, InputLayer) or layer.input_shape is None:
                raise ValueError("First layer should specify the input shape.")
            self._layers = []

        # The input shape of the current layer should
        # match the output shape of the previous layer
        if self._layers:
            layer.input_shape = self._layers[-1].output_shape

        self._layers.append(layer)
        layer.build(layer.input_shape)

        if len(self._layers) > 1:
            self.built = True

        self.logger.info(f"Added {layer.__class__.__name__}, "
                         f"Output Shape: {layer.output_shape}")

    # <-- Compile Model -->
    def compile(self, loss: str, optimizer: str, learning_rate: int = 0.001):
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
                "You must add layers to the model before compiling.")

        self.loss = self._set_loss_function(loss=loss)

        for layer in self._layers:
            if layer.trainable:
                layer.optimizer = self._create_optimizer(optimizer,
                                                         learning_rate)

        self.logger.info(f"Loss Function: {loss} -- "
                         f"Optimizer: {optimizer} - LR {learning_rate}")

    @staticmethod
    def _set_loss_function(loss: str) -> Loss:
        """
        Set the loss function for the model.

        Args:
            loss (str): Name of the loss function.

        Returns:
            Loss: Loss function instance.

        Raises:
            ValueError: If the loss function is not recognized.
        """
        loss_functions = {
            'binary_crossentropy': BinaryCrossEntropy,
            'categorical_crossentropy': CategoricalCrossEntropy,
        }

        if loss not in loss_functions:
            raise ValueError(f"Unknown loss: {loss}")

        return loss_functions[loss]()

    @staticmethod
    def _create_optimizer(optimizer: str, learning_rate: float) -> Optimizer:
        """
        Create an optimizer instance for the model.

        Args:
            optimizer (Optimizer | str): Optimizer instance
                or name of the optimizer to use.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            Optimizer: Optimizer instance.

        Raises:
            TypeError: If the optimizer type is unsupported.
        """
        optimizer_classes = {
            'adam': AdamOptimizer,
            'rmsprop': RMSpropOptimizer
        }

        if optimizer.lower() not in optimizer_classes:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        return optimizer_classes[optimizer.lower()](
            learning_rate=learning_rate)

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

        for layer in self._layers:
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
        for layer in reversed(self._layers):
            loss_gradients = layer.backward(loss_gradients)
            if layer.trainable:
                self._update_epoch_weights(layer)

    # <-- Training Methods -->
    def fit(self, X: pd.DataFrame, epochs: int, val_split: float = None,
            val_data: pd.DataFrame = None, callbacks: list[Callback] = None,
            batch_size: int = 32, verbose: bool = False) -> None:
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

        Raises:
            ValueError: If input data types are incorrect.
        """
        self._validate_inputs(X, epochs, val_split, val_data,
                              callbacks, batch_size, verbose)

        self._init_callbacks(callbacks)

        self.training_mode = True

        X_trn, y_trn, X_val, y_val = self._split_data(X, val_split, val_data)

        for epoch in range(epochs):
            if self.stop_training:
                break

            X_trn, y_trn = self._shuffle_data(X_trn, y_trn)

            trn_loss = []
            trn_accuracy = []
            for X, y in self._iter_batches(X_trn, y_trn, batch_size):
                loss_value, accuracy = self._train_batch(X, y)
                trn_loss.append(loss_value)
                trn_accuracy.append(accuracy)

            val_loss, val_accuracy = self._eval_validation_data(X_val, y_val)

            log = self._update_metrics(trn_loss, trn_accuracy, val_loss,
                                       val_accuracy, epoch)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Accuracy: {log['accuracy']:.3f}, "
                      f"Loss: {log['loss']:.3f} -- "
                      f"Val Accuracy: {log['val_accuracy']:.3f}, "
                      f"Val Loss: {log['val_loss']:.3f}")

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
    @staticmethod
    def _iter_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
        """
        Iterate over mini-batches of the dataset.

        Args:
            X (np.ndarray): Input features data.
            y (np.ndarray): Target labels data.
            batch_size (int): Number of samples per batch.

        Yields:
            Tuple[np.ndarray, np.ndarray]: Mini-batches of
                input features and target labels.
        """
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            yield X_batch, y_batch

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
                layer.weights)
            layer.bias_gradients += layer.kernel_regularizer.gradient(
                layer.bias)
        layer.weights, layer.bias = layer.optimizer.update(
            weights=layer.weights,
            bias=layer.bias,
            weights_gradients=layer.weights_gradients,
            bias_gradients=layer.bias_gradients
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
            regularization_penalty = layer.kernel_regularizer(
                layer.weights)
            inputs += regularization_penalty

        return inputs

    def _deactivate_train_mode(self):
        self.dropout_active = False
        self.training_mode = False

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

        self.dropout_active = True
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
        for layer, (w, b) in zip(self._layers, weights):
            layer.set_weights(weights=w, bias=b)

    # <-- Validation and Initializer -->
    def _init_callbacks(self, callbacks: list[Callback]):
        """
        Initialize the callbacks.

        Args:
            callbacks (list of objects): List of callback objects.
        """
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            callback.set_model(self)
            callback.on_train_begin()
            self.logger.info(f"Init Callback: {callback.__class__.__name__}")

        self.callbacks = callbacks

    def _validate_inputs(self, X: pd.DataFrame, epochs: int, val_split: float,
                         val_data: pd.DataFrame, callbacks: list[Callback],
                         batch_size: int, verbose: bool):
        """
        Validate inputs for the fit method.

        Raises:
            ValueError: If input data types or values are incorrect.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame.")
        if val_data is not None and not isinstance(val_data, pd.DataFrame):
            raise ValueError("val_data should be a pandas DataFrame.")
        if val_data is None and val_split is None:
            raise ValueError("You must provide either val_split or val_data.")
        if val_data is None and not (0 < val_split < 1):
            raise ValueError("val_split should be a float between 0 and 1.")
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("Number of epochs should be a positive integer.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size should be a positive integer.")
        if callbacks is not None:
            if not isinstance(callbacks, list):
                raise ValueError("Callbacks must be a list callback objects.")
            for callback in callbacks:
                if not isinstance(callback, Callback):
                    raise ValueError("Callbacks must be instance of Callback.")
        if not isinstance(verbose, bool):
            raise ValueError("Verbose should be a boolean value.")

        self.epochs = epochs
        self.val_split = val_split
        self.batch_size = batch_size
        self.verbose = verbose

    # <-- Data Preparation -->
    def _split_data(self, X: pd.DataFrame, val_split: float,
                    val_data: pd.DataFrame):
        """
        Split the data into training and validation sets.

        Args:
            X (DataFrame): Input features data.
            val_split (float): Fraction of the training data to be used for validation.
            val_data (DataFrame): Validation features data.

        Returns:
            tuple: Tuple of training and validation data.
        """
        if val_data is None:
            X_train, val_data = train_test_split(X, test_size=val_split)
        else:
            X_train = X

        if self.model_output_units > 1:
            X_train, y_train = self._one_hot_encoding(X_train)
            X_val, y_val = self._one_hot_encoding(val_data)
            self.logger.info(f"Output Units: {self.model_output_units}: "
                             f"Using One-Hot Encoding")
        else:
            X_train, y_train = self._label_encoding(X_train)
            X_val, y_val = self._label_encoding(val_data)
            self.logger.info(f"Output Units: {self.model_output_units}: "
                             f"Using Label Encoding")

        return X_train, y_train, X_val, y_val

    @staticmethod
    def _shuffle_data(X_trn: np.ndarray, y_trn: np.ndarray):
        """
        Shuffle the training data and corresponding labels.

        Args:
            X_trn (np.ndarray): Training features data.
            y_trn (np.ndarray): Training labels data.

        Returns:
            np.ndarray: Shuffled training features data.
            np.ndarray: Shuffled training labels data.
        """
        # Shuffle X_trn and y_trn using the same random permutation
        indices = np.random.permutation(len(X_trn))
        return X_trn[indices], y_trn[indices]

    # <-- Hot-Encode / Label Encoding -->
    @staticmethod
    def _one_hot_encoding(X: pd.DataFrame, num_classes: int = None) -> tuple:
        """
        Prepare the data for training.

        Args:
            X (DataFrame): Input features data.
            num_classes (int): Number of classes in the dataset.

        Returns:
            np.ndarray: Numpy array of input features.
        """
        if 'diagnosis' not in X.columns:
            raise ValueError("The target column is missing.")

        # Separate features and target
        X_feature = X.drop(columns=['diagnosis']).values
        y_true = X['diagnosis'].values.astype(int)

        # Get the number of samples (Rows)
        n_samples = y_true.shape[0]

        # Flatten y_true if necessary (1D array)
        y_true = y_true.ravel()

        # Determine the number of classes if not provided
        if num_classes is None:
            num_classes = np.max(y_true) + 1

        # One-hot encode the labels
        y_one_hot = np.zeros((n_samples, num_classes))
        y_one_hot[np.arange(n_samples), y_true] = 1

        return X_feature, y_one_hot

    @staticmethod
    def _label_encoding(data: pd.DataFrame) -> tuple:
        """
        Get the labels from the DataFrame.

        Args:
            data (DataFrame): Input features data.

        Returns:
            np.ndarray: Numpy array of labels.
        """
        X = data.drop(columns=['diagnosis']).values
        y = data['diagnosis'].values.astype(int)

        return X, y

    # <-- Metrics and Logging -->
    def _update_metrics(self, losses: list[float], metrics: list[float],
                        val_loss: float, val_accuracy: float, epoch: int):
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
        self.losses['training'][epoch] = np.mean(losses)
        self.accuracy['training'][epoch] = np.mean(metrics)
        self.losses['validation'][epoch] = val_loss
        self.accuracy['validation'][epoch] = val_accuracy

        return {
            'loss': np.mean(losses),
            'accuracy': np.mean(metrics),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }

    @property
    def history(self) -> dict:
        """
        Return the training history of the model.

        Returns:
            dict: Dictionary of training history.
        """
        return {'loss': self.losses, 'accuracy': self.accuracy}

    @property
    def model_output_units(self) -> int:
        """
        Return the number of output units of the model.

        Returns:
            int: Number of output units.
        """
        return self._layers[-1].output_shape[1]

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
        for layer in self._layers:
            if isinstance(layer, Dropout):
                layer.train_mode = train_mode

    # <-- Summary and Layer Parameters -->
    def summary(self) -> str:
        """
        Print a summary of the model architecture.
        """
        summary = ""
        for i, layer in enumerate(self._layers):
            output_shape = layer.output_shape
            parameters = layer.count_parameters()
            summary += (f"Layer {i + 1} {layer.__class__.__name__}: "
                        f"Trainable: {layer.trainable}, "
                        f"Output Shape: {output_shape}, "
                        f"Parameters: {parameters}\n")

        summary += f"Total Parameters: {self.count_parameters()}"
        return summary

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the model.

        Returns:
            int: Total number of parameters.
        """
        return sum(layer.count_parameters() for layer in self._layers)
