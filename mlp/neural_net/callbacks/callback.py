from typing import Protocol, runtime_checkable


@runtime_checkable
class Callback(Protocol):
    def set_model(self, model) -> None:
        """
        Set the model for the callback.

        Args:
            model (Model): Model instance

        Returns:
            None
        """
        ...

    def on_epoch_start(self, epoch: int, logs: dict) -> None:
        """
        Called at the start of an epoch.

        Args:
            epoch (int): Current epoch
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        ...

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """
        Called at the end of an epoch.

        Args:
            epoch (int): Current epoch
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        ...

    def on_train_begin(self) -> None:
        """
        Called at the start of training.

        Args:
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        ...

    def on_train_end(self) -> None:
        """
        Called at the end of training.

        Args:
            logs (dict): Dictionary of metrics

        Returns:
            None
        """
        ...
