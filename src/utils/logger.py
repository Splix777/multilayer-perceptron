import json
import logging
import coloredlogs

from pathlib import Path
from .config import Config
from typing import Optional

class Logger(logging.Logger):
    def __init__(self, name=__name__, config: Config = Config(), **kwargs):
        """Initializes the logger object."""
        super().__init__(name)
        self.config: Config = config
        self.color: str = kwargs.get('color', 'green')
        self.setLevel(self.config.config.settings['log_level'])

        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s - ' \
                     'File: %(filename)s'

        log_file: Path = self.config.logs_dir / f"{name}.log"
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(self.config.config.settings['log_level'])

        coloredlogs.install(level=self.config.config.settings['log_level'], 
                            logger=self, 
                            fmt=format,
                            level_styles={'info': {'color': self.color}, 
                                          'debug': {'color': self.color}, 
                                          'error': {'color': self.color},
                                          'warning': {'color': self.color}})

        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def __call__(self) -> logging.Logger:
        """Returns the logger object."""
        return self

    def log_json_object(self, obj: dict) -> None:
        """Logs a JSON object."""
        pretty_obj: str = json.dumps(obj, indent=4)
        self.info(pretty_obj)

    def log_config(self) -> None:
        """Logs the configuration object."""
        self.log_json_object(self.config.config.model_dump())

    def log_with_context(
            self, 
            level: str,
            message: str,
            context: Optional[dict] = None):
        """
        Logs a message with optional context information in JSON format.
        
        Args:
            level (str): The logging level (e.g., 'info', 'error', 'warning').
            message (str): The log message.
            context (Optional[dict]): A dictionary of additional context to log.
        """
        context_message = json.dumps(context, indent=4) if context else ''
        log_message = f"{message} | Context: {context_message}"

        if level.lower() == 'debug':
            self.debug(log_message)
        elif level.lower() == 'info':
            self.info(log_message)
        elif level.lower() == 'warning':
            self.warning(log_message)
        elif level.lower() == 'error':
            self.error(log_message)
        elif level.lower() == 'critical':
            self.critical(log_message)
        else:
            self.info(log_message)  # Default to info if invalid level is provided

    def log_to_file(self, message: str, file_name: str) -> None:
        """Logs a message to a specific file."""
        log_file = Path(self.config.logs_dir) / file_name
        with open(log_file, 'a', encoding='utf-8') as log_file_obj:
            log_file_obj.write(f"{message}\n")

# Example usage
if __name__ == "__main__":
    logger: Logger = Logger("example_logger")
    logger.info("This is an info message.")
    logger.error("This is an error message.")
    logger.warning("This is a warning message.")

    logger.log_config()

    # Example of using log_with_context
    context_data: dict[str, str] = {"user": "john_doe", "action": "login"}
    logger.log_with_context("info", "User performed an action", context_data)

    # Example of logging to a specific file
    logger.log_to_file("This is a custom file log message.", "custom_log.txt")
