import logging
import numpy as np
from .backend import get_backend
logger = logging.getLogger(__name__)

class Reconstructer(object):    
    
    def __init__(self, meas_description, **kwargs):
        backend_class = get_backend()
        self._backend = backend_class() #instantiate
        self._backend.initialize(meas_description, **kwargs)

    def reconstruct(self, data):
        logger.debug("Reconstructing...")
        return self._backend.reconstruct_data(data)



