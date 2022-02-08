# Standard Libraries
from pathlib import Path

# Third party libraries

# Local imports
from ..logs import logging
logger = logging.getLogger("tool box")

def create_directory(directory_path:str) -> None:
    """Create new folder in path"""
    Path(directory_path).mkdir(parents=True, exist_ok=True)