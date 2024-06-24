import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from .model import Model
from ..layers.layer import Layer
from ..layers.input import InputLayer
from ..optimizers.adam import AdamOptimizer
from ..optimizers.optimizer import Optimizer
from ..losses.binary_cross_entropy import BinaryCrossEntropy
from ..losses.categorical_cross_entropy import CategoricalCrossEntropy
from ..callbacks.callback import Callback


class Sequential(Model):
    def __init__(self):
        super().__init__()
        self.losses = {'training': [], 'validation': []}
        self.metrics = {'training': [], 'validation': []}
        self.optimizer = None
        self.loss = None
        self._layers = None
        self.built = False
        self.callbacks = None
        self.stop_training = False

    # Core Methods
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
            if not isinstance(layer, InputLayer) and layer.input_shape is None:
                raise ValueError(
                    "The first layer should specify the input shape.")

            self._layers = []

        if self._layers:
            layer.input_shape = self._layers[-1].output_shape

        self._layers.append(layer)
        self._layers[-1].build(layer.input_shape)

        if len(self._layers) > 1:
            self.built = True

    def compile(self, loss: str, optimizer: Optimizer | str) -> None:
        """
        Configure the model for training.

        Args:
        loss (str): Name of the loss function to use.
        optimizer (Optimizer | str): Optimizer instance or name
            of the optimizer to use.

        Raises:
        ValueError: If the model is not built or the loss is
            not recognized.
        """
        if not self.built:
            raise ValueError(
                "You must add layers to the model before compiling.")

        if loss == 'binary_crossentropy':
            self.loss = BinaryCrossEntropy()
        elif loss == 'categorical_crossentropy':
            self.loss = CategoricalCrossEntropy()
        else:
            raise ValueError("Unknown loss.")

        def create_optimizer_instance(optimizer_instance: Optimizer | str):
            if isinstance(optimizer_instance, str):
                if optimizer_instance == 'adam':
                    return AdamOptimizer()
                else:
                    raise ValueError("Unknown optimizer.")
            if isinstance(optimizer_instance, AdamOptimizer):
                return AdamOptimizer(
                    learning_rate=optimizer_instance.learning_rate
                )
            else:
                raise TypeError("Unsupported optimizer type.")

        for layer in self._layers:
            if layer.trainable:
                layer.optimizer = create_optimizer_instance(optimizer)

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
                layer.weights, layer.bias = layer.optimizer.update(
                    layer.weights, layer.bias,
                    layer.weights_gradient, layer.bias_gradient)

    def fit(self, X: DataFrame, epochs: int, val_split: float = None,
            val_data: DataFrame = None, callbacks: list[Callback] = None,
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

        X_trn, y_trn, X_val, y_val = self._split_data(X, val_split, val_data)

        for epoch in range(epochs):
            if self.stop_training:
                break

            losses = []
            metrics = []
            for X, y in self._iter_batches(X_trn, y_trn, batch_size):
                loss_value, accuracy = self._train_batch(X, y)
                losses.append(loss_value)
                metrics.append(accuracy)

            val_loss, val_accuracy = self._eval_validation_data(X_val, y_val)

            log = self._update_metrics(losses, metrics, val_loss, val_accuracy)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, "
                      f"Accuracy: {log['accuracy']:.3f}, "
                      f"Loss: {log['loss']:.3f}, "
                      f"Val Accuracy: {log['val_accuracy']:.3f}, "
                      f"Val Loss: {log['val_loss']:.3f}")

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=log)

        for callback in callbacks:
            callback.on_train_end()

    def predict(self, X: DataFrame | pd.Series) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (DataFrame): Input features for prediction.

        Returns:
            np.ndarray: Predicted output from the model.

        Raises:
            ValueError: If an input data type is incorrect.
        """
        if not isinstance(X, DataFrame) and not isinstance(X, pd.Series):
            raise ValueError("X should be a pandas DataFrame or Series.")

        # Convert DataFrame to a numpy array for compatibility with the model
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X).T

        X_pred, _ = self._one_hot_encoding(X)

        return self.call(X_pred)

    def evaluate(self, X: DataFrame) -> tuple[float, float]:
        """
        Evaluate the model using the provided data.

        Args:
            X (DataFrame): Input features data.

        Returns:
            tuple: Tuple of losses and accuracy.
        """
        X_eval, y_eval = self._one_hot_encoding(X)
        output = self.call(X_eval)
        loss_value = self.loss(y_eval, output)

        predictions = (output >= 0.5).astype(int)
        accuracy = np.mean(predictions == y_eval)

        print(f"Accuracy: {accuracy:.3f}, Loss: {loss_value:.3f}")

        return loss_value, float(accuracy)

    def summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        for i, layer in enumerate(self._layers):
            output_shape = layer.output_shape
            parameters = layer.count_parameters()
            print(f"Layer {i + 1} {layer.__class__.__name__}: "
                  f"Trainable: {layer.trainable}, "
                  f"Output Shape: {output_shape}, "
                  f"Parameters: {parameters}",
                  sep=' ', end='\n'
                  )

        print(f"Total Parameters: {self.count_parameters()}")

    def count_parameters(self) -> int:
        """
        Count the total number of parameters in the model.

        Returns:
            int: Total number of parameters.
        """
        return sum(layer.count_parameters() for layer in self._layers)

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

        if self.loss == CategoricalCrossEntropy:
            accuracy = self._categorical_accuracy(y_batch, output)
        else:
            accuracy = self._binary_accuracy(y_batch, output)

        return loss_value, accuracy

    @staticmethod
    def _binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the accuracy of the model.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy of the model.
        """
        return float(np.mean((y_pred >= 0.5).astype(int) == y_true))

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
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_classes == y_true)
        return float(accuracy)

    def _eval_validation_data(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Test the model on the validation data.

        Args:
            X_val (np.ndarray): Validation input features data.
            y_val (np.ndarray): Validation target labels data.

        Returns:
            Tuple[float, float]: Validation loss and accuracy.
        """
        val_pred = self.call(X_val)
        val_loss = self.loss(y_val, val_pred)

        if self.loss == CategoricalCrossEntropy:
            val_accuracy = self._categorical_accuracy(y_val, val_pred)
        else:
            val_accuracy = self._binary_accuracy(y_val, val_pred)

        return val_loss, val_accuracy

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

    # <-- Callbacks -->
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

        self.callbacks = callbacks

    # <-- DataFrame to Numpy Array -->
    @staticmethod
    def _one_hot_encoding(X: DataFrame, num_classes: int = None):
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

    def _validate_inputs(self, X: DataFrame, epochs: int, val_split: float,
                         val_data: DataFrame, callbacks: list[Callback],
                         batch_size: int, verbose: bool):
        """
        Validate inputs for the fit method.

        Raises:
            ValueError: If input data types or values are incorrect.
        """
        if not isinstance(X, DataFrame):
            raise ValueError("X should be a pandas DataFrame.")
        if val_data is not None and not isinstance(val_data, DataFrame):
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

    def _split_data(self, X: DataFrame, val_split: float, val_data: DataFrame):
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

        X_train, y_train = self._one_hot_encoding(X_train)
        X_val, y_val = self._one_hot_encoding(val_data)

        return X_train, y_train, X_val, y_val

    def _update_metrics(self, losses: list[float], metrics: list[float],
                        val_loss: float, val_accuracy: float):
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
        self.losses['training'].append(np.mean(losses))
        self.metrics['training'].append(np.mean(metrics))
        self.losses['validation'].append(val_loss)
        self.metrics['validation'].append(val_accuracy)

        return {
            'loss': np.mean(losses),
            'accuracy': np.mean(metrics),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }