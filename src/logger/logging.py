import logging
import logging.config
from pathlib import Path

import yaml


def setup_logging(
    config_path: str | Path | None = None,
    default_level: int = logging.INFO,
    capture_warnings: bool = True,
) -> None:
    """
    Setup logging configuration from YAML file.

    Args:
        config_path: Path to logging config YAML file
        default_level: Default logging level if config file not found
        capture_warnings: If True, redirect warnings module to logging
    """
    if config_path is None:
        # Default to config file in package
        config_path = Path.cwd() / "config" / "logging_config.yaml"

    path = Path(config_path)

    if path.exists():
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config not found at {path}, using basic config")

    # Redirect warnings module to logging
    if capture_warnings:
        logging.captureWarnings(True)
        # Get the warnings logger and configure it
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified module.

    Args:
        name: Logger name (use __name__ from calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
