import json
import pandas as pd
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import IO, Any, Optional


def open_file(
	file_path: Path,
	mode: str,
	create_if_missing: bool = False) -> IO[Any]:
	"""
	Open a file with the given mode, with robust error handling.

	Args:
		file_path (Path): Path to the file to open.
		mode (str): Mode to open the file in.
		create_if_missing (bool): If True, creates the file if
			it doesn't exist (only for write modes).

	Raises:
		FileNotFoundError: If the file does not exist
			and `create_if_missing` is False.
		ValueError: If the mode is invalid or incompatible
			with the file state.
		PermissionError: If the program does not have
			the necessary permissions.
		OSError: If an unexpected error occurs.

	Returns:
		file: Opened file object, or None if an error occurs.
	"""
	valid_modes: set[str] = {
		'r', 'w', 'x', 'a', 'b', 't', 'r+', 'w+', 'x+', 'a+'
		}
	if not any(m in mode for m in valid_modes):
		raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

	if not file_path.exists():
		if 'r' in mode:
			raise FileNotFoundError(f"File {file_path} does not exist.")

		elif create_if_missing and ('w' in mode or 'x' in mode or 'a' in mode):
			file_path.touch(exist_ok=True)

	with open(file=file_path, mode=mode) as file:
		if file is None:
			raise ValueError(f"Error opening file {file_path} in mode {mode}")
		
		return file

def csv_to_dataframe(
	file_path: Path,
	**kwargs: Any) -> pd.DataFrame:
	"""
	Read a CSV file into a pandas DataFrame.

	Args:
		file_path (Path): Path to the CSV file to read.
		**kwargs (Any): Additional keyword arguments to pass to
			pandas.read_csv.

	Returns:
		pd.DataFrame: DataFrame containing the CSV data, or None if
			an error occurs.
	"""
	if not file_path.exists():
		raise FileNotFoundError(f"File {file_path} does not exist.")
	
	df: Optional[pd.DataFrame] = pd.read_csv(file_path, **kwargs)
	if df is None:
		raise ValueError(f"Error reading CSV file {file_path}")
	
	return df

def dataframe_to_csv(
	df: pd.DataFrame,
	file_path: Path,
	**kwargs: Any) -> None:
	"""
	Write a pandas DataFrame to a CSV file.

	Args:
		df (pd.DataFrame): DataFrame to write to the CSV file.
		file_path (Path): Path to the CSV file to write.
		**kwargs (Any): Additional keyword arguments to pass to
			pandas.DataFrame.to_csv.

	Returns:
		None
	"""
	if df is None:
		raise ValueError("DataFrame cannot be None.")

	df.to_csv(file_path, **kwargs)

def json_to_dict(
	file_path: Path,
	**kwargs: Any) -> dict:
	"""
	Read a JSON file into a dictionary.

	Args:
		file_path (Path): Path to the JSON file to read.
		**kwargs (Any): Additional keyword arguments to pass to
			pandas.read_json.

	Returns:
		dict: Dictionary containing the JSON data, or None if
			an error occurs.
	"""
	if not file_path.exists():
		raise FileNotFoundError(f"File {file_path} does not exist.")
	
	with open(file=file_path, mode='r') as json_file:
		return json.load(json_file, **kwargs)