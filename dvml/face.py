"""
Reconstructer Class
====================

The `Reconstructer` class provides a unified interface for reconstructing density matrices 
from tomographic data using a specified backend (e.g., NumPy or PyTorch). It initializes 
the backend, sets parameters, and performs the reconstruction process.

Usage:
------
```python
from dvml.face import Reconstructer

# Initialize with measurement description and parameters
meas_description = np.array([...])  # Replace with actual measurement description
reconstructer = Reconstructer(meas_description, renorm=True, max_iters=200, thres=1e-8)

# Perform reconstruction
data = np.array([...])  # Replace with actual tomographic data
density_matrices = reconstructer.reconstruct(data)

# Update parameters
reconstructer.set_parameters(max_iters=300, thres=1e-7)
"""

import logging
import numpy as np
from .backend import get_backend
logger = logging.getLogger(__name__)


class Reconstructer(object):
    """
    The `Reconstructer` class provides a unified interface for reconstructing density matrices 
    from tomographic data using a specified backend (e.g., NumPy or PyTorch). It initializes 
    the backend, sets parameters, and performs the reconstruction process.

    Methods:
    --------
    - `__init__(meas_description, **kwargs)`: Initializes the `Reconstructer` with a measurement 
    description and optional parameters.
    - `reconstruct(data)`: Reconstructs density matrices from the provided tomographic data.
    - `set_parameters(*args, **kwargs)`: Updates backend parameters.
    """    
    
    def __init__(self, meas_description : np.ndarray, **kwargs):
        """
        Initializes the face object with a measurement description and additional parameters.

        Args:
            meas_description (ndarray): A description of the measurement to be used for initialization.
            **kwargs: Additional keyword arguments to be passed to the backend initialization.
        Kwargs
        `renorm` (bool, default=False): Whether to apply renormalization using the inverse sum operator.
        `md_has_ops` (bool, default=False): Indicates whether the measurement description includes operators.
        `max_iters` (int, default=100): Maximum number of iterations for the reconstruction loop.
        `thres` (float, default=1e-6): Convergence threshold for the reconstruction loop.
        `batch_size` (int, default=None):
            - For NumPy: Defaults to the number of available CPU cores minus one.
            - For PyTorch: Defaults to an estimated batch size based on available GPU memory.
        `paralelize` (bool, default=True): Whether to enable parallelization for batch processing.

        """
        backend_class = get_backend()
        self._backend = backend_class() #instantiate
        self._backend.initialize(meas_description, **kwargs)

    def reconstruct(self, data : np.ndarray):
        """
        Reconstructs the given data using the backend's reconstruction method.

        Args:
            data (ndarray n x m): The input tomogram batch (n tomograms) to be reconstructed, each with m elements.

        Returns:
            The reconstructed data as ndarray processed by the backend.

        Raises:
            Any exceptions raised by the backend's `reconstruct_data` method.
        """
        logger.debug("Reconstructing...")
        return self._backend.reconstruct_data(data)

    def set_parameters(self, *args, **kwargs):
        self._backend(*args, **kwargs)


