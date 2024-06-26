import numpy as np

from src.model.callbacks.callback import Callback


class EarlyStopping(Callback):
    def __init__(self, monitor: str = 'val_loss', patience: int = 5,
                 verbose: bool = False):
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best = np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = None
        self.best_weights = None
        self.model = None

    def set_model(self, model) -> None:
        """
        Set the model for the callback.

        Args:
            model (Model): Model instance

        Returns:
            None
        """
        self.model = model

    def on_epoch_start(self, epoch: int, logs: dict = None) -> None:
        """
        Called at the start of an epoch.

        Args:
            epoch (int): Current epoch
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        pass

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Called at the end of an epoch.

        Args:
            epoch (int): Current epoch
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        current = logs.get(self.monitor)
        if current is None:
            raise ValueError(f"Early stopping requires "
                             f"{self.monitor} available in logs.")

        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if current < self.best:
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            if self.verbose:
                print(f"Found new best weights at epoch {epoch + 1}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")

    def on_train_begin(self, logs: dict = None) -> None:
        """
        Called at the start of training.

        Args:
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        self.best = np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_train_end(self, logs: dict = None) -> None:
        """
        Called at the end of training.

        Args:
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        if self.verbose:
            print(f"Restoring model weights from the end of the best epoch"
                  f"({self.best_epoch + 1}).")

        self.model.set_weights(weights=self.best_weights)
