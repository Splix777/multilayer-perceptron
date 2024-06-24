import os
from mlp import MultiLayerPerceptron
from src.utils.config import Config


def load_model_and_predict(mlp):
    data_path = input("Enter the path to the data file to predict: ")
    if not os.path.exists(data_path):
        print("File does not exist. Exiting...")
        return
    results = mlp.predict(data_path.rstrip())
    print(results)


def load_model_and_evaluate(mlp):
    results = mlp.evaluate_model()
    print(results)


def train_model(mlp):
    path = input("Please provide a path to the data file: ").lower()
    if not os.path.exists(path) or not path.endswith('.csv'):
        print("File does not exist or is not a CSV file. Exiting...")
        return
    model_config = input(
        "Do you want to use a custom model configuration? (y/n): ").lower()

    conf_path = None
    if model_config == 'y':
        conf_path = input(
            "Please provide a path to the model configuration file: ")
        if not os.path.exists(conf_path) or not conf_path.endswith('.json'):
            print("File does not exist. Exiting...")
            return

    results = mlp.train_model(dataset_path=path, config_path=conf_path)
    print(results)


def main():
    config = Config()
    model_path = f"{config.model_dir}/model.pkl"
    mlp = MultiLayerPerceptron()

    if os.path.exists(model_path):
        response = input(
            "Model exists. Do you want to load it? (y/n): ").lower()
        if response == 'y':
            action = input(
                "Do you want to predict or evaluate the model? (p/e): ").lower()
            if action == 'p':
                load_model_and_predict(mlp)
            elif action == 'e':
                load_model_and_evaluate(mlp)
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
            "Model does not exist. Do you want to train it? (y/n): ").lower()
        if response == 'y':
            train_model(mlp)
        else:
            print("Exiting...")


if __name__ == "__main__":
    main()
