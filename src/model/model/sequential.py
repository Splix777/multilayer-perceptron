
import numpy as np

from pandas import DataFrame

from .model import Model
from ..layers.layer import Layer
from ..layers.input import InputLayer
from ..optimizers.optimizer import Optimizer
from ..callbacks.callback import Callback

from ..losses.loss import Loss


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

    def compile(self, losses: Loss, optimizer: Optimizer) -> None:
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
        if not callable(losses):
            raise TypeError("The losses function must be callable.")
        if not hasattr(optimizer, 'update'):
            raise TypeError("The optimizers must have an 'update' method.")
        self.loss = losses
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
        X_train, y_train = self.df_to_numpy(X)

        self.init_callbacks(callbacks=callbacks)

        for epoch in range(epochs):
            if self.stop_training:
                break

            losses = []
            metrics = []
            for X, y in self.iter_batches(X_train, y_train, batch_size):
                loss_value, accuracy = self.train_batch(X, y)
                losses.append(loss_value)
                metrics.append(accuracy)

            self.losses['training'].append(np.mean(losses))
            self.metrics['training'].append(np.mean(metrics))

            logs = {
                'loss': np.mean(losses),
                'accuracy': np.mean(metrics),
            }

            print(f"Epoch {epoch + 1}/{epochs}, "
                  f"Training Accuracy: {np.mean(metrics):.3f}, "
                  f"Training Loss: {np.mean(losses):.3f} -- ", end='')

            if val_data is not None:
                val_loss, val_accuracy = self.test_validation_data(val_data)
                logs['val_loss'] = val_loss
                logs['val_accuracy'] = val_accuracy

                print(f"Validation Accuracy: {val_accuracy:.3f}, "
                      f"Validation Loss: {val_loss:.3f}")

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs=logs)

        for callback in callbacks:
            callback.on_train_end()

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
        X_pred, _ = self.df_to_numpy(X)

        # Perform forward pass through the model
        predictions = self.call(X_pred)

        return predictions

    def evaluate(self, X: DataFrame) -> tuple[float, float]:
        """
        Evaluate the model using the provided data.

        Args:
            X (DataFrame): Input features data.
            y (np.ndarray): Target labels data.

        Returns:
            tuple: Tuple of losses and accuracy.
        """
        X_eval, y_eval = self.df_to_numpy(X)
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
        total_params = 0
        for layer in self._layers:
            total_params += layer.count_parameters()

        return total_params

    # <-- Epoch Methods -->
    @staticmethod
    def iter_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
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

    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray):
        """
        Train the model on a single batch of data.

        Args:
            X_batch (np.ndarray): Input features data.
            y_batch (np.ndarray): Target labels data.
        """
        output = self.call(X_batch)
        loss_value = self.loss(y_batch, output)
        loss_gradients = self.loss.gradient(y_batch, output)

        for layer in reversed(self._layers):
            loss_gradients = layer.backward(loss_gradients,
                                            self.optimizer.learning_rate)

        predictions = (output >= 0.5).astype(int)
        accuracy = np.mean(predictions == y_batch)

        return loss_value, accuracy

    def test_validation_data(self, val_data: DataFrame):
        """
        Test the model on the validation data.
        """
        X_val, y_val = self.df_to_numpy(val_data)
        val_output = self.call(X_val)
        val_loss = self.loss(y_val, val_output)

        val_predictions = (val_output >= 0.5).astype(int)
        val_accuracy = np.mean(val_predictions == y_val)
        self.losses['validation'].append(val_loss)
        self.metrics['validation'].append(val_accuracy)

        return val_loss, val_accuracy

    # <-- Getters and Setters -->
    def get_weights(self) -> list:
        """
        Get the weights of the model.

        Returns:
            list: List of weights for each layer.
        """
        weights = []
        for layer in self._layers:
            weights.append(layer.get_weights())

        return weights

    def set_weights(self, weights: list) -> None:
        """
        Set the weights of the model.

        Args:
            weights (list): List of weights for each layer.
        """
        for layer, (w, b) in zip(self._layers, weights):
            layer.set_weights(weights=w, bias=b)

    # <-- Callbacks -->
    def init_callbacks(self, callbacks: list[Callback]) -> None:
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
    def df_to_numpy(X: DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare the data for training.

        Args:
            X (DataFrame): Input features data.

        Returns:
            np.ndarray: Numpy array of input features.
        """
        X_feature = X.values[:, 1:]
        if 'diagnosis' not in X.columns:
            raise ValueError("The target column is missing.")
        y_target = X['diagnosis'].values.reshape(-1, 1)
        return X_feature, y_target
