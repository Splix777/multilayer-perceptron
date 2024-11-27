from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


class History:
    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def log_epoch(
        self,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
    ) -> None:
        """Logs metrics for a single epoch."""
        self.metrics["train_loss"].append(train_loss)
        self.metrics["train_accuracy"].append(train_accuracy)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["val_accuracy"].append(val_accuracy)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts metrics to a Pandas DataFrame for analysis."""
        return pd.DataFrame(self.metrics)

    def plot(self, model_name: Optional[str] = None) -> None:
        """Plots training and validation metrics."""
        df: pd.DataFrame = self.to_dataframe()

        # Plot loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(df.index, df["train_loss"], label="Train Loss")
        plt.plot(df.index, df["val_loss"], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Over Epochs ({model_name})")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(df.index, df["train_accuracy"], label="Train Accuracy")
        plt.plot(df.index, df["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Over Epochs ({model_name})")
        plt.legend()

        plt.tight_layout()
        plt.show()
