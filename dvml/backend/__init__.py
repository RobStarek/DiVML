import logging
from dvml.backend.numpy_backend import NumpyBackend
# Configure the logger for the backend module
logger = logging.getLogger("dvml.backend")
logger.setLevel(logging.INFO)

_backend = NumpyBackend

try:
    import torch
    if torch.cuda.is_available() or torch.version.cuda is not None:
        from dvml.backend.pytorch_backend import TorchBackend
        _backend = TorchBackend
        logger.info("Torch backend selected.")
    else:
        logger.warning("Torch not applicable! Falling back to numpy.")
        raise ImportError
except ImportError:
    logger.info("Falling back to numpy backend.")

def get_backend():
    return _backend
