from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from pathlib import Path

from mlp import MultiLayerPerceptron
from src.utils.config import Config
from src.utils.file_handlers import save_json_to_file

app = typer.Typer()
console = Console()

config = Config()


def list_models(model_dir: Path) -> list[Path]:
    """Helper function to list available models."""
    models: list[Path] = [
        file for file in model_dir.iterdir() if file.suffix == ".pkl"
    ]
    return models


@app.command("train")
def train_model() -> None:
    """Train a new MultiLayerPerceptron model."""
    try:
        data_path: str = Prompt.ask(
            "[blue]Enter the path to the data file[/blue]"
        )
        if not Path(data_path).exists() or not data_path.endswith(".csv"):
            console.print("[red]Invalid file path or file is not a CSV.[/red]")
            raise typer.Exit()

        config_option: str = Prompt.ask(
            "[blue]Do you want to load a custom configuration? (y/n)[/blue]"
        ).lower()

        conf_path: Optional[str] = None
        if config_option == "y":
            conf_path = Prompt.ask(
                "[blue]Enter the path to the configuration file[/blue]"
            )
            if not Path(conf_path).exists() or not conf_path.endswith(".json"):
                console.print("[red]Invalid configuration file path.[/red]")
                raise typer.Exit()
        else:
            create_new: str = Prompt.ask(
                "[blue]Do you want to configure a new model? (y/n)[/blue]"
            ).lower()
            if create_new == "y":
                conf_path = configure_new_model()

        if not conf_path:
            console.print("[red]No configuration provided. Exiting...[/red]")
            raise typer.Exit()

        mlp = MultiLayerPerceptron()
        results: Path = mlp.train_model(
            config_path=Path(conf_path), dataset_path=Path(data_path)
        )
        console.print(
            f"[green]Training completed![/green]\nModel saved to: {results}"
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command("predict")
def predict_model() -> None:
    """Load a trained model and make predictions."""
    model_path: Path = select_model(config.trained_models_dir)
    if not model_path:
        console.print("[red]No model selected. Exiting...[/red]")
        raise typer.Exit()

    data_path: str = Prompt.ask("[blue]Enter the path to the data file[/blue]")
    if not Path(data_path).exists():
        console.print("[red]Invalid data file path.[/red]")
        raise typer.Exit()

    mlp = MultiLayerPerceptron()
    predictions: list[str] = mlp.predict(model_path, Path(data_path))
    console.print(f"[green]Predictions:[/green]\n{predictions}")


@app.command("evaluate")
def evaluate_model() -> None:
    """Load a trained model and evaluate its performance."""
    model_path: Path = select_model(config.trained_models_dir)
    if not model_path:
        console.print("[red]No model selected. Exiting...[/red]")
        raise typer.Exit()

    data_path: str = Prompt.ask("[blue]Enter the path to the data file[/blue]")
    if not Path(data_path).exists():
        console.print("[red]Invalid data file path.[/red]")
        raise typer.Exit()

    mlp = MultiLayerPerceptron()
    loss, acc = mlp.evaluate_model(model_path, Path(data_path))

    console.print(
        f"[green]Evaluation results:[/green]\n"
        f"{acc * 100:.2f}% Accuracy\n"
        f"{loss:.4f} Loss"
        )


def configure_new_model() -> str:
    """Create and save a new model configuration."""
    model_name: str = Prompt.ask("[blue]Enter the model name[/blue]")
    num_layers: str = Prompt.ask(
        "[blue]Enter the number of hidden layers[/blue]", default="1"
    )

    layers: list[dict] = []
    layers.append({"type": "input", "input_shape": 30})
    for i in range(int(num_layers)):
        console.print(f"[cyan]Configuring layer {i + 1}[/cyan]")
        layer_type: str = Prompt.ask(
            "[blue]Select layer type (dense/dropout)[/blue]",
            choices=["dense", "dropout"],
        )

        if layer_type == "dense":
            units: str = Prompt.ask("[blue]Enter the number of units[/blue]")
            activation: str = Prompt.ask(
                "[blue]Select activation function (lrelu, relu, sigmoid, tanh, softmax)[/blue]",
                choices=["lrelu", "relu", "sigmoid", "tanh", "softmax"],
            )
            kernel_initializer: str = Prompt.ask(
                "[blue]Select kernel initializer (glorot_uniform, he_normal)[/blue]",
                choices=["glorot_uniform", "he_normal"],
            )
            kernel_regularizer: str = Prompt.ask(
                "[blue]Select kernel regularizer (l1, l2, None)[/blue]",
                choices=["l1", "l2", "None"],
            )
            if kernel_regularizer == "None":
                dense_layer: dict[str, str | int] = {
                    "type": "dense",
                    "units": int(units),
                    "activation": activation,
                    "kernel_initializer": kernel_initializer,
                }
            else:
                dense_layer = {
                    "type": "dense",
                    "units": int(units),
                    "activation": activation,
                    "kernel_initializer": kernel_initializer,
                    "kernel_regularizer": kernel_regularizer,
                }

            layers.append(dense_layer)


        elif layer_type == "dropout":
            rate: str = Prompt.ask(
                "[blue]Enter dropout rate (0-1)[/blue]", default="0.5"
            )
            layers.append({"type": "dropout", "rate": float(rate)})

    optimizer_type: str = Prompt.ask(
        "[blue]Select optimizer (adam, rmsprop)[/blue]",
        choices=["adam", "rmsprop"],
    )
    learning_rate: str = Prompt.ask(
        "[blue]Enter learning rate[/blue]", default="0.001"
    )
    optimizer: dict[str, str | float] = {
        "type": optimizer_type,
        "learning_rate": float(learning_rate)
    }

    loss: str = Prompt.ask(
        "[blue]Select loss function (categorical_crossentropy, binary_crossentropy)[/blue]",
        choices=["categorical_crossentropy", "binary_crossentropy"],
    )

    batch_size: str = Prompt.ask("[blue]Enter batch size[/blue]", default="32")
    epochs: str = Prompt.ask("[blue]Enter number of epochs[/blue]", default="10")

    config_path: Path = config.trained_models_dir / f"{model_name}.json"
    model_config = {
        "name": model_name,
        "layers": layers,
        "optimizer": optimizer,
        "loss": loss,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
    }

    save_json_to_file(file_path=config_path, data=model_config)

    console.print(f"[green]Configuration saved at {config_path}[/green]")
    return str(config_path)


def select_model(model_dir: Path) -> Path:
    """Select a model from the directory."""
    models: list[Path] = list_models(model_dir)
    if not models:
        console.print("[red]No models found in the directory.[/red]")
        raise typer.Exit()

    table = Table(title="Available Models")
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("Model Name", style="magenta")

    for idx, model in enumerate(models, 1):
        table.add_row(str(idx), model.name)

    console.print(table)
    choice = Prompt.ask("[blue]Enter the model index[/blue]")
    if not choice.isdigit() or not (1 <= int(choice) <= len(models)):
        console.print("[red]Invalid choice.[/red]")
        raise typer.Exit()
    return models[int(choice) - 1]


if __name__ == "__main__":
    app()
