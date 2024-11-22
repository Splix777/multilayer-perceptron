import time
from typing import Callable, Tuple, Type, Any
from json import JSONDecodeError
from functools import wraps
from .logger import Logger

# Logger setup
logger: Logger = Logger("decorators")

# List of exceptions to handle
KNOWN_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ValueError,
    KeyError,
    TypeError,
    PermissionError,
    FileNotFoundError,
    RuntimeError,
    AttributeError,
    JSONDecodeError,
    OSError
)

def error_handler(
        exceptions_to_handle: Tuple[Type[Exception], ...] = (),
        suppress: bool = False) -> Callable:
    """
    Decorator to handle exceptions in the decorated method.
    """
    if not exceptions_to_handle:
        exceptions_to_handle = KNOWN_EXCEPTIONS

    def decorator(method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(method)
        def wrapper(*args: Tuple, **kwargs: dict) -> Any:
            try:
                return method(*args, **kwargs)

            except exceptions_to_handle as e:
                msg: str = (
                    f"{type(e).__name__} in {method.__qualname__} "
                    f"({method.__module__}): {e}"
                )
                logger.error(msg, exc_info=True)
                if not suppress:
                    raise e
            
            except Exception as e:
                msg: str = (
                    f"Unexpected error in {method.__qualname__} "
                    f"({method.__module__}): {e}"
                )                
                logger.error(msg, exc_info=True)
                raise e

        return wrapper

    return decorator


def timeit(method: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to measure execution time of a function.
    """
    @wraps(method)
    def wrapper(*args: Tuple, **kwargs: dict) -> Any:
        start_time: float = time.time()
        result = method(*args, **kwargs)
        end_time: float = time.time()
        duration: float = end_time - start_time
        log_message: str = f"{method.__qualname__} took {duration:.4f} seconds"
        logger.info(log_message)
        return result

    return wrapper


if __name__ == '__main__':
    @error_handler(suppress=True)
    def test_error_handler() -> None:
        """Test function for the error_handler decorator."""
        with open('non_existent_file.txt', 'r') as f:
            f.read
            
    
    @timeit
    def test_timeit() -> None:
        """Test function for the timeit decorator."""
        time.sleep(1)

    test_timeit()
    test_error_handler()

