"""
This module provides tools for reconstructing quantum states using maximum-likelihood
methods. It includes utilities for creating projector arrays, selecting computational
backends, and performing quantum state and process reconstruction.
Exports:
    - Reconstructer: A class for performing quantum state reconstruction.
    - make_projector_array: A utility function for creating projector arrays.
Logging:
    The module is configured with a logger named "dvml" for debugging and tracking
    operations.
"""
__version__ = "1.0.0"

import logging

from dvml.face import Reconstructer
from dvml.backend import get_backend
from dvml.utils import make_projector_array

# Configure the module-wide logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dvml")

__all__ = ["Reconstructer", "make_projector_array"]
