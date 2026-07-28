import logging
import os
import sys

def setup_logging(log_file=None, level=logging.INFO):
    """Configures the logging system for training or evaluation.
    
    Sets up a root logger that prints to the console and optionally writes to a file.
    
    Args:
        log_file (str, optional): Path to a log file where logs should be appended.
        level (int): The logging level to use (e.g. logging.INFO, logging.DEBUG).
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to avoid duplicate logs when setup is called multiple times
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(level)
    
    # Define log format
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_format)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
    # Suppress noise from third-party libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("desmume").setLevel(logging.WARNING)
    logging.getLogger("stable_baselines3").setLevel(logging.INFO)
    
    return root_logger


from stable_baselines3.common.logger import KVWriter

class LoggerOutputFormat(KVWriter):
    """Custom KVWriter that outputs training metrics using python standard logging instead of ASCII tables."""
    def __init__(self) -> None:
        self.logger = logging.getLogger("sb3")

    def write(self, key_values: dict, key_excluded: dict, step: int) -> None:
        # Group metrics to make the line extremely readable
        rollout_parts = []
        train_parts = []
        time_parts = []
        other_parts = []
        
        for k, v in sorted(key_values.items()):
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            if k.startswith("rollout/"):
                rollout_parts.append(f"{k.split('/')[-1]}={val_str}")
            elif k.startswith("train/"):
                train_parts.append(f"{k.split('/')[-1]}={val_str}")
            elif k.startswith("time/"):
                time_parts.append(f"{k.split('/')[-1]}={val_str}")
            else:
                other_parts.append(f"{k}={val_str}")
                
        log_msg = f"Step {step}"
        if rollout_parts:
            log_msg += " | Rollout: " + ", ".join(rollout_parts)
        if train_parts:
            log_msg += " | Train: " + ", ".join(train_parts)
        if time_parts:
            log_msg += " | Time: " + ", ".join(time_parts)
        if other_parts:
            log_msg += " | Other: " + ", ".join(other_parts)
            
        self.logger.info(log_msg)

    def close(self) -> None:
        pass


class StdoutLogger:
    """Wrapper to redirect stdout to python standard logging."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.rstrip("\r\n")
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        pass


def redirect_stdout_to_logger():
    """Redirects sys.stdout to python standard logging and captures python warnings."""
    import warnings
    sys.stdout = StdoutLogger(logging.getLogger("stdout"))
    logging.captureWarnings(True)

