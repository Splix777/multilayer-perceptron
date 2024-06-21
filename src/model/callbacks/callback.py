from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def set_model(self, model) -> None:
        pass

    @abstractmethod
    def on_epoch_start(self, epoch: int, logs: dict = None) -> None:
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        pass

    @abstractmethod
    def on_train_begin(self, logs: dict = None) -> None:
        pass

    @abstractmethod
    def on_train_end(self, logs: dict = None) -> None:
        pass
