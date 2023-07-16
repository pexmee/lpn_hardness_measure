import logging
import os
from typing import Any, Callable

import psutil
from utilities.config import LOG_FILE


def set_logging_defaults(level: int):
    """
    Sets the default logging level, format and output file.

    Args:
        level (int): Logging level to set. Use constants from the logging module,
                    like logging.INFO, logging.DEBUG, etc.
    """
    logging.basicConfig(
        filename=LOG_FILE,
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_callback(
    dim: int,
    n_sample_amount: int,
    error_rate: float,
) -> Callable[[Any], None]:
    """
    Generates a callback function in order to utilize closures.

    Args:
        dim (int): Length of the secret.
        n_sample_amount (int): Number of samples.
        error_rate (float): The error rate.

    Returns:
        Callable[[Any], None]: A callback function to be used for logging.
    """

    def log_callback(future: Any) -> None:
        try:
            logging.info(
                f"Done with dimension={dim}, n_sample_amount={n_sample_amount}, error_rate={error_rate}"
            )
        except Exception as e:
            logging.warning(
                f"Task for dimension={dim}, n_sample_amount={n_sample_amount}, error_rate={error_rate} raised an exception: {e}"
            )

    return log_callback


def log_memory_usage(process_name):
    """
    Logs the current memory usage of the process.

    Args:
        process_name (str): Name of the process.
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024**2  # Convert to MB
    logging.debug(f"{process_name} memory usage: {memory_usage:.2f} MB")
