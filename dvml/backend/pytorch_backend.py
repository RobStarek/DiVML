import numpy as np
from dvml.backend_template import BackendTemplate


class TorchBackend(BackendTemplate):

    def __init__(self):
        super().__init__()

    def initialize(self, measurement_description, *args, **kwargs):
        """
        Initialize the backend with measurement description.
        measurement_description : nparray describing the measurements
        """
        self.projectors = measurement_description

    def set_parameters(self, *args, **kwargs):
        pass

    def reconstruct_data(self, tomograms, *args, **kwargs):
        n = len(tomograms)
        return np.random.random((n,4,4))
    