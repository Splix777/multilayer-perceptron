import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, ValidationError, field_validator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ProjectConfig(BaseModel):
    """Represents the project configuration."""

    name: str
    version: str
    description: str


class WdbcLabelsConfig(BaseModel):
    """Configuration for WDBC labels."""

    id: str
    diagnosis: str
    radius_types: List[str]
    base_features: List[str]


class PathsConfig(BaseModel):
    """Configuration for various paths in the project."""

    output_directory: str
    source_directory: str
    docs_directory: str
    logs_directory: str
    temp_directory: str
    csv_directory: str
    trained_models_directory: str
    plot_directory: str


class ConfigSchema(BaseModel):
    """Main configuration schema combining all sub-configurations."""

    project: ProjectConfig
    settings: Dict[str, Any]
    paths: PathsConfig
    wdbc_labels: WdbcLabelsConfig

    @field_validator("settings")
    def validate_settings(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validates the 'settings' dictionary."""
        if "log_level" not in v or "max_retries" not in v:
            raise ValueError(
                "Settings must include 'log_level' and 'max_retries'"
            )
        return v


@dataclass
class Config:
    """Manages the loading and validation of the configuration file.

    Attributes:
        config_path (Optional[Path]): Path to the configuration file.
        config (ConfigSchema): Parsed and validated config schema.
        base_dir (Path): Base directory of the configuration file.
        output_dir (Path): Path to the output directory.
        source_dir (Path): Path to the source directory.
        logs_dir (Path): Path to the logs directory.
        temp_dir (Path): Path to the temporary files directory.
        csv_dir (Path): Path to the CSV directory.
        trained_models_dir (Path): Path to trained models directory.
        plot_dir (Path): Path to the plot directory.
    """

    config_path: Optional[Path] = None
    config: ConfigSchema = field(init=False)
    base_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    source_dir: Path = field(init=False)
    docs_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    temp_dir: Path = field(init=False)
    csv_dir: Path = field(init=False)
    trained_models_dir: Path = field(init=False)
    plot_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        """Initializes configuration and directory setup."""
        if self.config_path is None:
            self.config_path = Path(__file__).parent / "../../config.json"
        else:
            self.config_path = Path(self.config_path)

        try:
            self._load_config_file()
            self._load_directories()

        except FileNotFoundError as e:
            logging.error(f"Configuration file not found: {e}")
            raise
        except ValidationError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error parsing JSON file: {e}")
            raise

    def _load_config_file(self) -> None:
        """Loads the configuration file and validates it."""
        if self.config_path is None:
            raise ValueError("Config path cannot be None")

        with open(self.config_path, "r", encoding="utf-8") as config_file:
            config_data: Dict[str, Any] = json.load(config_file)
            self.config = ConfigSchema(**config_data)

    def _load_directories(self) -> None:
        """Initializes and ensures the existence of directories."""
        if self.config_path is None:
            raise ValueError("Config path cannot be None")

        self.base_dir = self.config_path.parent
        paths: PathsConfig = self.config.paths

        path_mapping: Dict[str, str] = {
            "output_directory": "output_dir",
            "source_directory": "source_dir",
            "docs_directory": "docs_dir",
            "logs_directory": "logs_dir",
            "temp_directory": "temp_dir",
            "csv_directory": "csv_dir",
            "trained_models_directory": "trained_models_dir",
            "plot_directory": "plot_dir",
        }

        for config_key, attribute_name in path_mapping.items():
            setattr(
                self,
                attribute_name,
                self.base_dir / getattr(paths, config_key),
            )

        # Ensure directories exist
        for dir_path in [
            self.output_dir,
            self.logs_dir,
            self.csv_dir,
            self.trained_models_dir,
            self.plot_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    try:
        config = Config(
            config_path=Path(__file__).parent / "../../config.json"
        )

        pretty_config: str = json.dumps(config.config.model_dump(), indent=4)
        logging.info(f"Loaded configuration:\n{pretty_config}")

    except Exception as e:
        logging.critical(f"Failed to initialize config: {e}")
