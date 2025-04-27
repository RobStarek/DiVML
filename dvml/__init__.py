"""
dvml
====

A Python package for discrete-variable quantum maximum-likelihood reconstruction.

"""
import logging

from dvml.face import Reconstructer
from dvml.backend import get_backend
from dvml.utils import make_projector_array

logger = logging.getLogger(__name__)
logging.info("Init of dvml module.")
logging.info(str(get_backend()))

__all__ = ["Reconstructer", "make_projector_array"]
print("Init of main dvml.")


# import importlib
# from .backend
# _backend = None

# # try:
# #     import torch
# #     if torch.cuda.is_available() or torch.version.cuda is not None:
# #         from .torch_backend import Backend
# #         _backend = Backend
# #     else:
# #         raise ImportError
# # except ImportError:
# #     try:
# #         import numba
# #         from .numba_backend import Backend
# #         _backend = Backend
# #     except ImportError:
# #         from .numpy_backend import Backend
# #         _backend = Backend
# _backend = DummyBackend

# def get_backend():
#     return _backend