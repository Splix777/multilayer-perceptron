from typing import Literal

import logging
import coloredlogs
import json
from pathlib import Path
from typing import Optional
from mlp.utils.config import Config


class Logger(logging.Logger):
    def __init__(
        self,
        name: str = __name__,
        config: Config = Config(),
        color: Literal["red", "green", "yellow", "cyan", "white"] = "green",
    ):
        """
        Initializes the logger object.

        Args:
            name (str): The name of the logger.
            config (Config): The configuration object.
            color (Literal["red", "green", "yellow", "cyan", "white"]):
                The color of the logger.
        """
        super().__init__(name)
        self.config: Config = config
        self.color: str = color
        self.setLevel(self.config.config.settings["log_level"])

        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - "
            "File: %(filename)s"
        )

        # File handler
        log_file: Path = self.config.logs_dir / f"{name}.log"
        file_handler = logging.FileHandler(filename=log_file, mode="w")
        file_handler.setLevel(self.config.config.settings["log_level"])
        file_formatter = logging.Formatter(fmt=log_format)
        file_handler.setFormatter(fmt=file_formatter)
        self.addHandler(hdlr=file_handler)

        # Stream handler with colors
        stream_handler = logging.StreamHandler()
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
            stream=stream_handler.stream,
        )
        self.addHandler(hdlr=stream_handler)

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
            level (str): The logging level.
            message (str): The log message.
            context (Optional[dict]): Dictionary of
                additional context to log.
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
    info_logger.info("This is an info message.")
    error_logger.error("This is an error message.")
    warning_logger.warning("This is a warning message.")

    # Example of using log_with_context
    context_data: dict[str, str] = {"user": "john_doe", "action": "login"}
    error_logger.log_with_context(
        "info", "User performed an action", context_data
    )

    # Example of logging to a specific file
    error_logger.log_to_file(
        "This is a custom file log message.", "custom_log.txt"
    )
