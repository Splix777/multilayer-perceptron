import json
import os
import sys
from pathlib import Path
from mlp import MultiLayerPerceptron
from src.utils.config import Config

config = Config()


def load_model_and_predict(mlp: MultiLayerPerceptron, model_path: str):
    """
    Load a model and make predictions using a MultiLayerPerceptron.

    This function prompts the user to enter the path to a
    data file for prediction, checks if the file exists,
    and then uses the provided MultiLayerPerceptron model
    to make predictions on the data.

    Args:
        mlp (MultiLayerPerceptron): The MultiLayerPerceptron
            model for prediction.
        model_path (str): The path to the model file to load.

    Returns:
        None
    """
    data_path = input("Enter the path to the data file to predict: ")
    if not os.path.exists(data_path):
        print("File does not exist. Exiting...")
        sys.exit(0)

    print(mlp.predict(model_path.rstrip(), data_path.rstrip()))


def load_model_and_evaluate(mlp: MultiLayerPerceptron, model_path: str):
    """
    Load a model and evaluate it using a MultiLayerPerceptron.

    This function prompts the user to enter the path to a data
    file for evaluation, checks if the file exists, and then
    uses the provided MultiLayerPerceptron model to evaluate
    the model's performance on the data.

    Args:
        mlp (MultiLayerPerceptron): The MultiLayerPerceptron
            model for evaluation.
        model_path (str): The path to the model file to load.

    Returns:
        None
    """
    data_path = input("Enter the path to the data file to predict: ")
    if not os.path.exists(data_path):
        print("File does not exist. Exiting...")
        sys.exit(0)

    print(mlp.evaluate_model(model_path.rstrip(), data_path.rstrip()))


def train_model(mlp: MultiLayerPerceptron):
    """
    Train a model using a MultiLayerPerceptron.

    This function guides the user through providing a path
    to a data file, optionally loading a custom model
    configuration, and configuring a new model before
    training the MultiLayerPerceptron model on the dataset.

    Args:
        mlp: The MultiLayerPerceptron model to train.

    Returns:
        None
    """
    data_path = input("Please provide a path to the data file: ").lower()
    if not os.path.exists(data_path) or not data_path.endswith('.csv'):
        print("File does not exist or is not a CSV file. Exiting...")
        sys.exit(0)

    model_config = input(
        "Do you want to load a custom model configuration? (y/n): ").lower()

    conf_path = None
    if model_config == 'y':
        conf_path = input(
            "Please provide a path to the model configuration file: ")
        if not os.path.exists(conf_path) or not conf_path.endswith('.json'):
            print("File does not exist. Exiting...")
            sys.exit(0)

    configure_new = input(
        "Do you want to configure a new model? (y/n): ").lower()
    if configure_new == 'y':
        conf_path = configure_new_model()

    results = mlp.train_model(dataset_path=data_path, config_path=conf_path)
    print(results)


def get_user_choice(prompt: str, options_dict: dict) -> str:
    """
    Helper function to display options and get user choice.

    This function displays a prompt and a list of options
    for the user to choose from, then returns the key
    corresponding to the selected option.

    Args:
        prompt (str): The prompt to display to the user.
        options_dict (dict): A dictionary of options to display.

    Returns:
        str: The key corresponding to the selected option.
    """
    while True:
        print(prompt)
        for num, (key, value) in enumerate(options_dict.items(), start=1):
            print(f"{num}. {value}")
        choice = input("Enter your choice: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options_dict):
            return list(options_dict.keys())[int(choice) - 1]
        else:
            print("Invalid choice...")


def configure_new_model() -> Path:
    """
    Configure a new model based on user input.

    This function guides the user through creating
    a new model configuration by selecting various
    options such as model name, layer types, activations,
    optimizers, losses, batch size, and epochs,
    and then saves the configuration to a JSON file.

    Returns:
        Path: The path to the saved model configuration file.
    """
    possible_layers = {'dense': 'Dense Layer', 'dropout': 'Dropout Layer'}
    possible_activations = {'relu': 'ReLU', 'lrelu': 'Leaky ReLU',
                            'prelu': 'Parametric ReLU',
                            'sigmoid': 'Sigmoid',
                            'tanh': 'Hyperbolic Tangent (Tanh)',
                            'softmax': 'Softmax'}
    possible_kernel_initializers = {'glorot_uniform': 'Glorot Uniform',
                                    'glorot_normal': 'Glorot Normal',
                                    'he_normal': 'He Normal',
                                    'he_uniform': 'He Uniform'}
    possible_optimizers = {'adam': 'Adam', 'rmsprop': 'RMSprop'}
    possible_losses = {'categorical_crossentropy': 'Categorical Crossentropy',
                       'binary_crossentropy': 'Binary Crossentropy'}

    model_name = input("Enter the name of the model: ").strip()
    if not model_name:
        print("Invalid model name. Exiting...")
        sys.exit(0)

    num_layers = input("Enter the number of hidden layers: ").strip()
    if not num_layers.isdigit():
        print("Invalid number of layers. Exiting...")
        sys.exit(0)

    layers = [{
        'type': 'input',
        'input_shape': 30
    }]
    for i in range(int(num_layers)):
        print(f"\nConfiguring layer {i + 1}")
        layer_type = get_user_choice("Select layer type:", possible_layers)
        if layer_type == 'dense':
            units = input(f"Enter the size of layer {i + 1}: ").strip()
            if not units.isdigit():
                print("Invalid size. Exiting...")
                sys.exit(0)
            activation = get_user_choice("Select activation function:",
                                         possible_activations)
            kernel_initializer = get_user_choice("Select kernel initializer:",
                                                 possible_kernel_initializers)
            layers.append({
                'type': 'dense',
                'units': int(units),
                'activation': activation,
                'kernel_initializer': kernel_initializer
            })
        elif layer_type == 'dropout':
            drop_rate = input(
                f"Enter the dropout rate for layer {i + 1} (0-1): ").strip()
            try:
                drop_rate = float(drop_rate)
                if not (0 <= drop_rate <= 1):
                    raise ValueError
            except ValueError:
                print("Invalid dropout rate. Exiting...")
                sys.exit(0)
            layers.append({
                'type': 'dropout',
                'rate': drop_rate
            })
    layers.append({
        'type': 'dense',
        'units': 2,
        'activation': 'softmax'
    })

    optimizer = get_user_choice("Select optimizer:", possible_optimizers)
    learning_rate = input("Enter the learning rate: ").strip()
    try:
        learning_rate = float(learning_rate)
        if learning_rate <= 0:
            raise ValueError
    except ValueError:
        print("Invalid learning rate. Exiting...")
        sys.exit(0)
    final_optimizer = {
        'type': optimizer,
        'learning_rate': learning_rate
    }

    loss = get_user_choice("Select loss function:", possible_losses)

    batch_size = input("Enter the batch size: ").strip()
    if not batch_size.isdigit() or int(batch_size) <= 0:
        print("Invalid batch size. Exiting...")
        sys.exit(0)

    epochs = input("Enter the number of epochs: ").strip()
    if not epochs.isdigit() or int(epochs) <= 0:
        print("Invalid number of epochs. Exiting...")
        sys.exit(0)

    final_config = {
        'model_name': model_name,
        'layers': layers,
        'optimizer': final_optimizer,
        'loss': loss,
        'batch_size': int(batch_size),
        'epochs': int(epochs)
    }

    save_path = config.model_dir / f"{model_name}.json"
    with open(save_path, 'w') as f:
        json.dump(final_config, f, indent=4)

    print(f"Configuration saved to {save_path}")
    return save_path


def select_model(model_dir: Path) -> str:
    """
    Select a model from the provided directory.

    Args:
        model_dir (Path): The directory containing the models.

    Returns:
        str: The path to the selected model file,
            or an empty string if no model is selected.
    """
    if not os.path.exists(model_dir):
        print("Model directory does not exist. Exiting...")
        sys.exit(0)

    models = []
    for root, dirs, files in os.walk(model_dir):
        models.extend(file for file in files if file.endswith(".pkl"))

    if not models:
        print("No models found in the directory...")
        return ""
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")

    response = input("Select a Model or press any other key to continue: ")
    try:
        response = int(response)
        if response < 1 or response > len(models):
            raise ValueError
    except ValueError:
        print("No model selected.")
        return ""

    return os.path.join(model_dir, models[response - 1])


def main():
    """
    Execute the main functionality of the program.

    This function orchestrates the main flow of the program,
    including loading, predicting, evaluating,
    and training a model based on user input.

    Returns:
        None
    """
    model_path = select_model(config.model_dir)
    mlp = MultiLayerPerceptron()

    if os.path.exists(model_path):
        response = input(
            "Model exists. Do you want to load it? (y/n): ").lower()
        if response == 'y':
            action = input(
                "Do you want to predict | evaluate the model? (p/e): ").lower()
            if action == 'p':
                load_model_and_predict(mlp, model_path)
            elif action == 'e':
                load_model_and_evaluate(mlp, model_path)
            else:
                print("Invalid option. Exiting...")
        else:
            response = input(
                "Do you want to re-train the model? (y/n): ").lower()
            if response == 'y':
                train_model(mlp)
            else:
                print("Exiting...")
    else:
        response = input(
            "Do you want to train a model? (y/n): ").lower()
        if response == 'y':
            train_model(mlp)
        else:
            print("Exiting...")


if __name__ == "__main__":
    main()
