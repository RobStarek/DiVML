import abc
import typing
import numpy as np

# EXPECTED_INP_ARR = np.typing.NDArray[np.float32 |
#                                      np.float64 | np.uint32 | int | np.uint64]
# EXPECTED_PROJ_ARR = np.typing.NDArray[np.complex128 | np.complex64]
# EXPECTED_RHO_ARR = np.typing.NDArray[np.complex128 | np.complex64]


class BackendTemplate(metaclass=abc.ABCMeta):
    """Template for backend.
    It defines mandatory function which are required by the core class.
    No matter, how the backend treats the data, it should expose methods for
    initialization and reconstruction.

    Inputs and outputs are given as numpy arrays.

    Args:
        metaclass (_type_, optional): _description_. Defaults to abc.ABCMeta.
    """
    @abc.abstractmethod
    def initialize(self, measurement_description, *args, **kwargs):
        """
        Initialize the backend with measurement description.
        measurement_description : nparray describing the measurements
        """
        pass

    @abc.abstractmethod
    def set_parameters(self, *args, **kwargs):
        """Set reconstruction parameters, which are set during initialization."""
        pass

    @abc.abstractmethod
    def reconstruct_data(self, tomograms, *args, **kwargs):
        """
        Process the given tomograms.
        tomograms : real nparray or iterable of its that contains measurement outcomes
        
        Returns:
            iterable of reconstructed density matrices
        """
        pass
