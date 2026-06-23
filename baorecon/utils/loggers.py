import logging
import os

def setup_logger(name: str , debug: bool = None) -> logging.Logger:
    """
    Create and return a logger for a class.

    Parameters
    ----------
    name : str
        Name of the logger.
    debug : bool, optional
        If True, set logger to DEBUG level. Otherwise INFO. If None, the level is determined by the DEBUG_LOGGER environment variable.

    Returns
    -------
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    
    # Set level
    if debug is None:
        debug_env = os.environ.get("DEBUG_LOGGER", "False")
        debug = debug_env.lower() in ("1", "true", "yes")

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    logger.propagate = False  # avoid messages being duplicated by the root logger



    # If it has no handler yet, add a console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


