import json
import pickle
from typing import IO, Any
from json.decoder import JSONDecodeError
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

from src.utils.logger import error_logger


def open_file(
    file_path: Path, mode: str, create_if_missing: bool = False
) -> IO[Any]:
    """
    Open a file with the given mode, with robust error handling.

    Args:
        file_path (Path): Path to the file to open.
        mode (str): Mode to open the file in.
        create_if_missing (bool): If True, creates the file if it
            doesn't exist (only for write modes).

    Raises:
        FileNotFoundError: If the file does not exist
            and `create_if_missing` is False.
        ValueError: If the mode is invalid or incompatible
            with the file state.
        PermissionError: If the program does not have the
            necessary permissions.
        OSError: If an unexpected error occurs.

    Returns:
        file: Opened file object.
    """
    valid_modes: set[str] = {
        "r",
        "w",
        "x",
        "a",
        "b",
        "t",
        "r+",
        "w+",
        "x+",
        "a+",
    }

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one: {valid_modes}")

    if not file_path.exists() and "r" in mode:
        raise FileNotFoundError(f"File {file_path} does not exist.")

    try:
        if (
            not file_path.exists()
            and create_if_missing
            and ("w" in mode or "x" in mode or "a" in mode)
        ):
            file_path.touch(exist_ok=True)

        file: IO[Any] = open(file_path, mode=mode)
        return file

    except FileNotFoundError as e:
        error_logger.error(f"File not found: {e}")
        raise
    except PermissionError as e:
        error_logger.error(f"Permission error: {e}")
        raise
    except Exception as e:
        error_logger.error(f"Unexpected error: {e}")
        raise OSError(f"Error opening file {file_path} in mode {mode}: {e}")


def save_json_to_file(data: dict, file_path: Path, **kwargs: Any) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): Dictionary to save to the JSON file.
        file_path (Path): Path to the JSON file.
        **kwargs (Any): Additional keyword arguments
            to pass to json.dump.

    Raises:
        ValueError: If saving the JSON file fails.
    """
    try:
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, **kwargs)
    except Exception as e:
        error_logger.error(f"Error saving JSON to file: {e}")
        raise ValueError(f"Error saving JSON to file {file_path}: {e}")


def csv_to_dataframe(file_path: Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        file_path (Path): Path to the CSV file to read.
        **kwargs (Any): Additional keyword arguments
            to pass to pandas.read_csv.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    try:
        df = pd.read_csv(file_path, **kwargs)
        if df.empty:
            raise ValueError(f"CSV file {file_path} is empty.")
    except (EmptyDataError, ValueError) as e:
        error_logger.error(f"Error reading CSV: {e}")
        raise
    except Exception as e:
        error_logger.error(f"Error reading CSV file {file_path}: {e}")
        raise
    return df


def dataframe_to_csv(df: pd.DataFrame, file_path: Path, **kwargs: Any):
    """
    Write a pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to write to the CSV file.
        file_path (Path): Path to the CSV file to write.
        **kwargs (Any): Additional keyword arguments to pass
            to pandas.DataFrame.to_csv.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame cannot be None or empty.")

    try:
        df.to_csv(file_path, **kwargs)
    except Exception as e:
        error_logger.error(f"Error writing DataFrame to CSV: {e}")
        raise


def json_to_dict(file_path: Path, **kwargs: Any) -> dict:
    """
    Read a JSON file into a dictionary.

    Args:
        file_path (Path): Path to the JSON file to read.
        **kwargs (Any): Additional keyword arguments
            to pass to json.load.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    try:
        with open(file_path, "r") as json_file:
            return json.load(json_file, **kwargs)
    except JSONDecodeError as e:
        error_logger.error(f"JSON Decode error: {e}")
        raise ValueError(f"Error reading JSON file {file_path}: {e}")
    except Exception as e:
        error_logger.error(f"Unexpected error: {e}")
        raise ValueError(f"Error: {file_path}: {e}")


def pickle_to_file(obj: Any, file_path: Path) -> None:
    """
    Pickle an object to a file.

    Args:
        obj (Any): Object to pickle.
        file_path (Path): Path to the pickle file.

    Raises:
        ValueError: If pickling fails.
    """
    try:
        with open(file_path, "wb") as pkl_file:
            pickle.dump(obj, pkl_file)
    except pickle.PickleError as e:
        error_logger.error(f"Pickle error: {e}")
        raise ValueError(f"Error pickling object to file {file_path}: {e}")
    except Exception as e:
        error_logger.error(f"Unexpected error: {e}")
        raise ValueError(f"Error pickling object: {e}")


def file_to_pickle(file_path: Path) -> Any:
    """
    Load a pickled object from a file.

    Args:
        file_path (Path): Path to the pickle file.

    Returns:
        Any: The object loaded from the pickle file.

    Raises:
        ValueError: If unpickling fails.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    try:
        with open(file_path, "rb") as pkl_file:
            return pickle.load(pkl_file)
    except pickle.PickleError as e:
        error_logger.error(f"Pickle error: {e}")
        raise ValueError(f"Error unpickling object from file {file_path}:{e}")
    except Exception as e:
        error_logger.error(f"Unexpected error: {e}")
        raise ValueError(f"Error unpickling object: {e}")


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd

    test_dir = Path("./test_files")
    test_dir.mkdir(exist_ok=True)

    # Test file paths
    test_csv_path = test_dir / "test.csv"
    test_json_path = test_dir / "test.json"
    test_pickle_path = test_dir / "test.pkl"

    # Test DataFrame
    test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Test JSON dictionary
    test_dict = {"key1": "value1", "key2": 2}

    try:
        # Test open_file
        print("Testing open_file...")
        with open_file(test_csv_path, "w", create_if_missing=True) as f:
            f.write("col1,col2\n1,a\n2,b\n3,c\n")
        print("open_file: Success")

        # Test csv_to_dataframe
        print("Testing csv_to_dataframe...")
        df = csv_to_dataframe(test_csv_path)
        print("DataFrame loaded:")
        print(df)

        # Test dataframe_to_csv
        print("Testing dataframe_to_csv...")
        dataframe_to_csv(test_data, test_csv_path, index=False)
        print(f"Data written to {test_csv_path}")

        # Test json_to_dict
        print("Testing json_to_dict...")
        with open(test_json_path, "w") as f:
            json.dump(test_dict, f)
        loaded_dict = json_to_dict(test_json_path)
        print("JSON loaded:", loaded_dict)

        # Test pickle_to_file
        print("Testing pickle_to_file...")
        pickle_to_file(test_dict, test_pickle_path)
        print(f"Object pickled to {test_pickle_path}")

        # Test file_to_pickle
        print("Testing file_to_pickle...")
        unpickled_obj = file_to_pickle(test_pickle_path)
        print("Unpickled object:", unpickled_obj)

    except Exception as e:
        print(f"An error occurred: {e}")
