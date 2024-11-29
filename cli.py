from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from pathlib import Path

from mlp.mlp_service import MultiLayerPerceptron
from mlp.utils.config import Config
from mlp.utils.file_handlers import save_json_to_file

from src.rag.main import run

app = typer.Typer()
console = Console()

config = Config()


@app.command("train")
def train_model() -> None:
    """Train a new MultiLayerPerceptron model."""
    try:
        console.print("[cyan]Select a CSV to train the model with.[/cyan]")
        data_path = select_csv_file(config.csv_dir)
        config_option = Prompt.ask(
            "[blue]Do you want to load a custom configuration? (y/n)[/blue]"
        ).lower()

        conf_path: Optional[Path] = None
        if config_option == "y":
            conf_path = select_json_file(config.trained_models_dir)
        else:
            create_new = Prompt.ask(
                "[blue]Do you want to configure a new model? (y/n)[/blue]"
            ).lower()
            if create_new == "y":
                conf_path = Path(configure_new_model())

        if not conf_path:
            console.print("[red]No configuration provided. Exiting...[/red]")
            raise typer.Exit()

        plot_option: str = Prompt.ask(
            "[blue]Do you want to plot the training history? (y/n)[/blue]"
        )
        plot: bool = plot_option.lower() == "y"

        mlp = MultiLayerPerceptron()
        results: Path = mlp.train_model(
            config_path=conf_path, dataset_path=data_path, plot=plot
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

    data_path: Path = select_csv_file(config.csv_dir)
    mlp = MultiLayerPerceptron()
    predictions: list[str] = mlp.predict(model_path, data_path)
    console.print(f"[green]Predictions:[/green]\n{predictions}")


@app.command("evaluate")
def evaluate_model() -> None:
    """Load a trained model and evaluate its performance."""
    model_path: Path = select_model(config.trained_models_dir)
    if not model_path:
        console.print("[red]No model selected. Exiting...[/red]")
        raise typer.Exit()

    data_path: Path = select_csv_file(config.csv_dir)
    mlp = MultiLayerPerceptron()
    loss, acc = mlp.evaluate_model(model_path, data_path)

    console.print(
        f"[green]Evaluation results:[/green]\n"
        f"{acc * 100:.2f}% Accuracy\n"
        f"{loss:.4f} Loss"
    )

@app.command("rag")
def ask_ollama() -> None:
    while True:
        prompt: str = Prompt.ask("[blue]Ask Ollama a question[/blue]")
        user_input: dict[str, str] = {"topic": f"{prompt}"}
        if prompt == "exit":
            break
        run(user_input)


def configure_new_model() -> str:
    """Create and save a new model configuration."""
    model_name = Prompt.ask("[blue]Enter the model name[/blue]")
    num_layers = Prompt.ask(
        "[blue]Enter the number of hidden layers[/blue]", default="1"
    )

    layers: list[dict] = []

    # Add Input Layer Automatically
    layers = [{"type": "input", "input_shape": 30}]
    for i in range(int(num_layers)):
        console.print(f"[cyan]Configuring layer {i + 1}[/cyan]")
        layer_type = Prompt.ask(
            "[blue]Select layer type (dense/dropout)[/blue]",
            choices=["dense", "dropout"],
        )

        if layer_type == "dense":
            units = Prompt.ask("[blue]Enter the number of units[/blue]")
            activation = Prompt.ask(
                "[blue]Select activation function (lrelu, relu, sigmoid, tanh, softmax)[/blue]",
                choices=["lrelu", "relu", "sigmoid", "tanh", "softmax"],
            )
            kernel_initializer = Prompt.ask(
                "[blue]Select kernel initializer (glorot_uniform, he_normal)[/blue]",
                choices=["glorot_uniform", "he_normal"],
            )
            kernel_regularizer = Prompt.ask(
                "[blue]Select kernel regularizer (l1, l2, None)[/blue]",
                choices=["l1", "l2", "None"],
            )
            dense_layer = {
                "type": "dense",
                "units": int(units),
                "activation": activation,
                "kernel_initializer": kernel_initializer,
            }
            if kernel_regularizer != "None":
                dense_layer["kernel_regularizer"] = kernel_regularizer
            layers.append(dense_layer)

        elif layer_type == "dropout":
            rate: str = Prompt.ask(
                "[blue]Enter dropout rate (0-1)[/blue]", default="0.5"
            )
            layers.append({"type": "dropout", "rate": float(rate)})

    optimizer = {
        "type": Prompt.ask(
            "[blue]Select optimizer (adam, rmsprop)[/blue]",
            choices=["adam", "rmsprop"],
        ),
        "learning_rate": float(
            Prompt.ask("[blue]Enter learning rate[/blue]", default="0.001")
        ),
    }

    loss = Prompt.ask(
        "[blue]Select loss function (categorical_crossentropy, binary_crossentropy)[/blue]",
        choices=["categorical_crossentropy", "binary_crossentropy"],
    )

    config_path = config.trained_models_dir / f"{model_name}.json"
    model_config = {
        "name": model_name,
        "layers": layers,
        "optimizer": optimizer,
        "loss": loss,
        "batch_size": int(
            Prompt.ask("[blue]Enter batch size[/blue]", default="32")
        ),
        "epochs": int(
            Prompt.ask("[blue]Enter number of epochs[/blue]", default="10")
        ),
    }

    save_json_to_file(file_path=config_path, data=model_config)
    console.print(f"[green]Configuration saved at {config_path}[/green]")
    return str(config_path)


def select_model(model_dir: Path) -> Path:
    """Select a model from the directory."""
    return select_file(model_dir, ".pkl", "Available Models")


def select_csv_file(csv_dir: Path) -> Path:
    """Select a CSV file from the directory."""
    return select_file(csv_dir, ".csv", "Available CSV Files")


def select_json_file(json_dir: Path) -> Path:
    """Select a JSON file from the directory."""
    return select_file(json_dir, ".json", "Available JSON Files")


def select_file(directory: Path, extension: str, title: str) -> Path:
    """Helper function to select a file with a specific extension."""
    files: list[Path] = [
        file for file in directory.iterdir() if file.suffix == extension
    ]
    if not files:
        console.print(
            f"[red]No {extension} files found in the directory.[/red]"
        )
        raise typer.Exit()

    table = Table(title=title)
    table.add_column("Index", justify="center", style="cyan")
    table.add_column("File Name", style="magenta")

    for idx, file in enumerate(files, 1):
        table.add_row(str(idx), file.name)

    console.print(table)
    choice: str = Prompt.ask("[blue]Enter the file index[/blue]")
    if not choice.isdigit() or not (1 <= int(choice) <= len(files)):
        console.print("[red]Invalid choice.[/red]")
        raise typer.Exit()
    return files[int(choice) - 1]


if __name__ == "__main__":
    app()
