import logging
import logging.handlers
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Tuple

def setup_logging(log_dir: str = "logs", base_filename: str = "processing") -> Tuple[multiprocessing.Queue, logging.handlers.QueueListener]:
    """
    Configures the logging system to log info and error messages to separate files
    and prints tracebacks to the console. Each run generates unique log filenames based on the current timestamp.

    Args:
        log_dir (str): Directory to store log files.
        base_filename (str): Base name for log files.

    Returns:
        Tuple[multiprocessing.Queue, logging.handlers.QueueListener]: Logging queue and listener.
    """
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique log filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define log file paths
    info_log_file = log_path / f"{base_filename}_{timestamp}.info.log"
    
    # Create a multiprocessing queue for logs
    log_queue = multiprocessing.Queue()
    
    # Define logging format
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(processName)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File Handler for INFO and below (INFO and DEBUG)
    info_handler = logging.FileHandler(info_log_file)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(log_format)
    info_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    
    # Console Handler for ERROR and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(log_format)
    console_handler.addFilter(lambda record: record.levelno == logging.ERROR)

    # Create a QueueListener to listen for logs from all processes
    listener = logging.handlers.QueueListener(log_queue, info_handler, console_handler)
    listener.start()
    
    return log_queue, listener

def get_logger(log_queue: multiprocessing.Queue) -> logging.Logger:
    """
    Creates a logger that sends logs to the multiprocessing queue.

    Args:
        log_queue (multiprocessing.Queue): The queue to send log messages to.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers will filter appropriately

    # Prevent adding multiple QueueHandlers
    if not any(isinstance(handler, logging.handlers.QueueHandler) for handler in logger.handlers):
        queue_handler = logging.handlers.QueueHandler(log_queue)
        logger.addHandler(queue_handler)
    
    return logger
