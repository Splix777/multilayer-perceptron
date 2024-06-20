import logging
from .config import Config


config = Config()


class Logger:
    def __init__(self, name=__name__):
        self.config = Config()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.config.settings['log_level'])

        log_file = self.config.logs_dir / f"{name}.log"
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(self.config.settings['log_level'])

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def __call__(self):
        return self.logger


# Example usage
if __name__ == "__main__":
    logger = Logger('Logger')()
    logger.info("This is an info message.")
    logger.error("This is an error message.")
