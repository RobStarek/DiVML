"""
dvml
====

A Python package for discrete-variable quantum maximum-likelihood reconstruction.

"""
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
