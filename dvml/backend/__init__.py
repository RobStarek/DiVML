import logging
# from dvml.backend.dummy_backend import DummyBackend

logger = logging.getLogger(__name__)
logging.info("Init of backend dvml.module.")

_backend = None
FORCE_NPY = True

try:
    import torch
    if torch.cuda.is_available() or torch.version.cuda is not None:
        from dvml.backend.pytorch_backend import TorchBackend
        _backend = TorchBackend
        logging.info("Torch backend selected.")
    else:
        logging.warning("Torch not applicable! Falling back to numpy.")
        raise ImportError
except ImportError:
    from dvml.backend.numpy_backend import NumpyBackend
    _backend = NumpyBackend
    logging.info("Numpy backend selected.")
# _backend = DummyBackend

def get_backend():
    return _backend
