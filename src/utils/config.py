import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class Config:
    config_path: Optional[Path] = None
    config: Optional[Dict] = field(init=False, default=None)
    project: Optional[str] = field(init=False, default=None)
    settings: Optional[Dict] = field(init=False, default=None)
    paths: Optional[Dict] = field(init=False, default=None)
    base_dir: Optional[Path] = field(init=False, default=None)
    output_dir: Optional[Path] = field(init=False, default=None)
    source_dir: Optional[Path] = field(init=False, default=None)
    logs_dir: Optional[Path] = field(init=False, default=None)
    temp_dir: Optional[Path] = field(init=False, default=None)
    csv_directory: Optional[Path] = field(init=False, default=None)
    model_dir: Optional[Path] = field(init=False, default=None)
    wdbc_labels: Optional[Dict] = field(init=False, default=None)
    plot_dir: Optional[Path] = field(init=False, default=None)

    def __post_init__(self):
        """
        Initialize the Config object by loading the configuration file.
        """
        if self.config_path is None:
            self.config_path = Path(__file__).parent / '../../config.json'
        else:
            self.config_path = Path(self.config_path)

        with open(self.config_path) as config_file:
            self.config = json.load(config_file)

        self.load_config()

    def load_config(self) -> None:
        """
        Set the attributes.
        """
        self.project = self.config.get('project')
        self.settings = self.config.get('settings')
        self.paths = self.config.get('paths')
        self.wdbc_labels = self.config.get('wdbc_labels')

        # Define base directory and paths
        self.base_dir = self.config_path.parent
        self.output_dir = self.base_dir / self.paths['output_directory']
        self.source_dir = self.base_dir / self.paths['source_directory']
        self.logs_dir = self.base_dir / self.paths['logs_directory']
        self.temp_dir = self.base_dir / self.paths['temp_directory']
        self.csv_directory = self.base_dir / self.paths['csv_directory']
        self.model_dir = self.base_dir / self.paths['model_directory']
        self.plot_dir = self.base_dir / self.paths['plot_directory']

        # Ensure the necessary directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_directory.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    config = Config()
    print(config.config)
    print(config.project)
    print(config.settings)
    print(config.paths)
    print(config.base_dir)
    print(config.output_dir)
    print(config.source_dir)
    print(config.logs_dir)
    print(config.temp_dir)
    print(config.csv_directory)
