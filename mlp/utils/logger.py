from typing import Optional
import json
import logging
import coloredlogs

from pathlib import Path
from mlp.utils.config import Config


class Logger(logging.Logger):
    def __init__(
        self, name=__name__, config: Config = Config(), **kwargs
    ) -> None:
        """
        Initializes the logger object.

        Args:
            name (str): The name of the logger.
            config (Config): The configuration object.
            **kwargs: Additional keyword arguments.
                color (str): The color

        Returns:
            None
        """
        super().__init__(name)
        self.config: Config = config
        self.color: str = kwargs.get("color", "green")
        self.setLevel(self.config.config.settings["log_level"])

        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - "
            "File: %(filename)s"
        )

        log_file: Path = self.config.logs_dir / f"{name}.log"
        handler = logging.FileHandler(log_file, mode="w")
        handler.setLevel(self.config.config.settings["log_level"])

        coloredlogs.install(
            level=self.config.config.settings["log_level"],
            logger=self,
            fmt=log_format,
            level_styles={
                "info": {"color": self.color},
                "debug": {"color": self.color},
                "error": {"color": self.color},
                "warning": {"color": self.color},
            },
        )

        formatter = logging.Formatter(log_format)
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
        self, level: str, message: str, context: Optional[dict] = None
    ) -> None:
        """
        Logs a message with optional context information in JSON format.

        Args:
            level (str): The logging level (e.g., 'info', 'error', 'warning').
            message (str): The log message.
            context (Optional[dict]): Dictionary of additional context to log.
        """
        context_message = json.dumps(context, indent=4) if context else ""
        log_message = f"{message} | Context: {context_message}"

        # Dictionary for method lookup
        log_methods = {
            "debug": self.debug,
            "info": self.info,
            "warning": self.warning,
            "error": self.error,
            "critical": self.critical,
        }

        # Use the dictionary to get the method, default to self.info
        log_method = log_methods.get(level.lower(), self.info)
        log_method(log_message)

    def log_to_file(self, message: str, file_name: str) -> None:
        """Logs a message to a specific file."""
        log_file = Path(self.config.logs_dir) / file_name
        with open(log_file, "a", encoding="utf-8") as log_file_obj:
            log_file_obj.write(f"{message}\n")


# Singletons
error_logger: Logger = Logger("error_logger", color="red")
info_logger: Logger = Logger("info_logger", color="green")
warning_logger: Logger = Logger("warning_logger", color="yellow")


# Example usage
if __name__ == "__main__":
    error_logger: Logger = Logger("example_logger")
    error_logger.info("This is an info message.")
    error_logger.error("This is an error message.")
    error_logger.warning("This is a warning message.")

    error_logger.log_config()

    # Example of using log_with_context
    context_data: dict[str, str] = {"user": "john_doe", "action": "login"}
    error_logger.log_with_context(
        "info", "User performed an action", context_data
    )

    # Example of logging to a specific file
    error_logger.log_to_file(
        "This is a custom file log message.", "custom_log.txt"
    )
