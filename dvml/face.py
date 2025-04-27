import numpy as np
from .backend import get_backend

class Reconstructer(object):
    
    
    def __init__(self, meas_description):
        self._backend = get_backend()()
        self._backend.initialize(meas_description)
        pass

    def reconstruct(self, data):
        print("Reconstructing...")
        print()
        return self._backend.reconstruct_data(data)



