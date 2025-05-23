"""
This module provides a Numpy-based backend for density matrix reconstruction using 
iterative algorithms. It includes a vectorized Gram-Schmidt orthogonalization function 
optimized for Numba, as well as a `NumpyBackend` class that implements the reconstruction 
process.
Classes:
    NumpyBackend: Implements the backend for reconstructing density matrices from 
                  tomographic data using Numpy and optional parallelization.
Functions:
    _gram_schmidt(input_col_vectors): Performs vectorized Gram-Schmidt orthogonalization 
                                      to create an orthonormal system of vectors.
    _nb_reconstution_loop_vect(data_in, max_iters, tres, aux_rpv_columns, rho, rho_old, 
                               k_operator, probs, scaler_vect, renorm, proj_sum_inv): 
                               Numba-optimized iterative reconstruction loop.
Usage:
    This module is designed to be used as part of a larger framework for quantum state 
    tomography. The `NumpyBackend` class can be initialized with measurement descriptions 
    and used to reconstruct density matrices from tomographic data.

Returns:
    _type_: _description_
"""
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import functools
import numpy as np
from dvml.backend.backend_template import BackendTemplate
logger = logging.getLogger(__name__)

try:
    import numba as nb
    njit = nb.njit
except ImportError:
    logger.warning("Numba could not be imported.")
    def njit(f):
        return f

DEFAULT_MAX_ITERS = 100
DEFAULT_THRES = 1e-6
HERM_TOL = 1e-12
ORTH_TOL = 1e-12

@njit
def _gram_schmidt(input_col_vectors):
    """
    Vectorized Gramm-Schmidt orthogonalization.
    Creates an orthonormal system of vectors spanning the same vector space
    which is spanned by vectors in matrix X.

    Args:
        X: matrix of col vectors
    Returns:
        Y: matrix of orthogonalized col vectors
    Original source: https://gist.github.com/iizukak/1287876#gistcomment-1348649
    Modification is that arrays are allocated beforehand to make it run smoothly in numba.
    """
    # Allocate arrays (for numba)
    input_col_vectors = input_col_vectors.T
    h, w = input_col_vectors.shape
    output_col_vectors = np.zeros((h, w), dtype=np.complex64)
    output_col_vectors[0, :] = input_col_vectors[0, :]
    row = np.zeros((1, w), np.complex64)

    # Run Gram-Schmidt process
    for i in range(1, input_col_vectors.shape[0]):
        Ynorm2 = np.sum(output_col_vectors*output_col_vectors.conj(), axis=1)
        Ynorm2[i:] = 1
        row[0] = input_col_vectors[i]
        proj = (row.dot(output_col_vectors.T.conj()) /
                Ynorm2).reshape((-1, 1)) * output_col_vectors
        proj[i:] = 0
        output_col_vectors[i] = (
            input_col_vectors[i, ::1] - proj.sum(0))  # .ravel()
    out_norm = np.sqrt(
        (np.sum(output_col_vectors*output_col_vectors.conj(), axis=1)).reshape((-1, 1)))
    output_col_vectors = output_col_vectors/out_norm
    return output_col_vectors.T

# Original version
# def _gram_schmidt(X, row_vecs=False, norm=True):
#     """
#     Vectorized Gramm-Schmidt orthogonalization.
#     Creates an orthonormal system of vectors spanning the same vector space
#     which is spanned by vectors in matrix X.

#     Args:
#         X: matrix of vectors
#         row_vecs: are vectors store as line vectors? (if not, then use column vectors)
#         norm: normalize vector to unit size
#     Returns:
#         Y: matrix of orthogonalized vectors
#     Source: https://gist.github.com/iizukak/1287876#gistcomment-1348649
#     """
#     if not row_vecs:
#         X = X.T
#     Y = X[0:1, :].copy()
#     for i in range(1, X.shape[0]):
#         proj = np.diag(
#             (X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
#         Y = np.vstack((Y, X[i, :] - proj.sum(0)))
#     if norm:
#         Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
#     if row_vecs:
#         return Y
#     else:
#         return Y.T


class NumpyBackend(BackendTemplate):

    def __init__(self):
        # super().__init__()
        self.renorm = None
        self.aux_rpv_cols = None
        self.dim1 = None
        self.dim2 = None
        self.last_counter = None
        self.last_distance = None
        self.proj_sum_inv = None
        self.max_iters = None
        self.thres = None
        self.paralelize = None
        self.batch_size = None
        self.n_proj = None

    def initialize(self, measurement_description, *args, **kwargs):
        """
        Initialize the backend with measurement description.
        measurement_description : nparray describing the measurements
        """
        renorm = kwargs.get('renorm', False)
        md_has_ops = kwargs.get('md_has_ops', False)
        max_iters = kwargs.get('max_iters', DEFAULT_MAX_ITERS)
        thres = kwargs.get('thres', DEFAULT_THRES)
        self.paralelize = kwargs.get('paralelize', True)
        self.batch_size = kwargs.get('batch_size', cpu_count() - 1)

        dim = measurement_description.shape[1]
        self.renorm = renorm
        if not md_has_ops:
            # construct projector operators (using outer product) from kets
            meas_ops = np.array(
                [np.einsum("i,j", ket.ravel(), ket.conj().ravel())
                 for ket in measurement_description]
            )
        else:
            meas_ops = measurement_description
        self.n_proj, self.dim1, self.dim2 = meas_ops.shape
        # this is equivalent to np.out plus sum raveling and transpoistion
        self.aux_rpv_cols = np.einsum("ijk->kji", meas_ops).reshape(
            (self.dim1 * self.dim2, self.n_proj)
        )
        self.max_iters = max_iters
        self.thres = thres
        self.last_distance = -1
        self.last_counter = -1

        if self.renorm:
            projector_sum_col = np.sum(self.aux_rpv_cols, axis=1)
            proj_sum_eval, proj_sum_evec = np.linalg.eigh(
                projector_sum_col.reshape((dim, dim)).T
            )
            if np.sum(np.abs(proj_sum_eval - proj_sum_eval[0])) < dim * HERM_TOL:
                proj_sum_evec = np.eye(dim, dtype=complex)
            elif (
                np.abs(
                    np.sum(proj_sum_evec @
                           proj_sum_evec.T.conjugate() - np.eye(dim))
                )
                > dim * ORTH_TOL
            ):
                proj_sum_evec = _gram_schmidt(proj_sum_evec)
            proj_sum_diag_inv = np.diag(proj_sum_eval**-1)
            self.proj_sum_inv = (
                proj_sum_evec @ proj_sum_diag_inv @ proj_sum_evec.T.conjugate()
            )
        else:
            self.proj_sum_inv = np.eye(1, dtype=complex)
        # pro-forma pass to compile numba
        self.max_iters = 1
        _ = self.reconstruct(np.ones(self.n_proj))
        self.max_iters = max_iters

    def set_parameters(self, *args, **kwargs):
        self.max_iters = kwargs.get('max_iters', self.max_iters)
        self.thres = kwargs.get('thres', self.thres)
        self.renorm = kwargs.get('renorm', self.renorm)
        self.paralelize = kwargs.get('renorm', self.paralelize)

    def reconstruct_data(self, tomograms, *args, **kwargs):
        if self.paralelize:
            logger.info('running in paralel')
            entries = len(tomograms)
            chunksize = int(entries/self.batch_size)+1
            foo = functools.partial(self.reconstruct)

            with ProcessPoolExecutor(self.batch_size) as executor:
                results = executor.map(foo, tomograms, chunksize=chunksize)
                return np.array(list(results))
        else:
            return np.array([self.reconstruct(t) for t in tomograms])

    def reconstruct(self, data):
        """Reconstruct data into density matrix using parameters of the Reconstructer object.

        Args:
            data (ndarray[int] or ndarray[float]): input tomogram. It should have as many entries as there is projectors.

        Returns:
            ndarray[complex]: reconstructed and trace-normed density matrix
        """
        sel = np.asarray(data) > 0
        aux_rpv_columns = self.aux_rpv_cols[:, sel]
        data_in = np.asarray(data[sel], dtype=np.float64)
        # these will mutate
        rho = np.eye(self.dim1, dtype=np.complex128) / self.dim1
        rho_old = np.copy(rho)
        k_operator = np.empty_like(rho)
        scaler_vect = np.zeros((data_in.shape[0],), dtype=np.float64)
        probs = np.empty_like(scaler_vect)

        counter, distance = self._nb_reconstution_loop_vect(
            data_in,
            self.max_iters,
            self.thres,
            aux_rpv_columns,
            rho,
            rho_old,
            k_operator,
            probs,
            scaler_vect,
            self.renorm,
            self.proj_sum_inv,
        )
        self.last_counter = counter
        self.last_distance = distance
        return rho

    # @nb.jit(nopython=True)
    @staticmethod
    @njit
    def _nb_reconstution_loop_vect(
        data_in,
        max_iters,
        tres,
        aux_rpv_columns,
        rho,
        rho_old,
        k_operator,
        probs,
        scaler_vect,
        renorm,
        proj_sum_inv,
    ):
        """Numba implementation of iterative reconstruction.
        Careful typing is required by numba.

        Args:
            data_in (ndarray[np.float64]): tomogram array
            max_iters (int): see __init__()
            tres (float): see __init__()
            aux_rpv_columns (ndarray[np.complex128]): transposed measurment operators unraveled and column-wise stacked
            rho (ndarray[np.complex128]): density matrix, this function mutate it
            rho_old (ndarray[np.complex128], like rho): old density matrix, this function mutate it
            k_operator (ndarray[np.complex128], like rho): array with mapping operator, this function mutate it
            probs (ndarray[np.float64]): array with detection probabilities
            scaler_vect (ndarray[np.float64]): auxiliary array used to weight operators contributing to the mapping operator
            renorm (bool): if True, mapping operator get renormalized. Use it with non-complete measurement operators. When on, the performance is lower.
            proj_sum_inv (_type_): auxilary array containing inverted sum measurement operators

        Returns:
            counter (int): iteration at which the reconstruction stopped
            distance (float): Frobenius distance of matrices in the last iteration step
        """
        counter = 0
        distance = 10 * tres
        dim_proj, n_proj = aux_rpv_columns.shape
        dim1, dim2 = rho.shape
        while (counter < max_iters) and (distance > tres):
            rho_old[:, :] = rho[:, :]
            # create K operator
            probs = (rho.ravel().reshape((1, dim_proj))
                     @ aux_rpv_columns).ravel().real
            scaler_vect = data_in / probs
            k_operator = (
                (
                    aux_rpv_columns
                    @ scaler_vect.reshape((n_proj, 1)).astype(np.complex128)
                )
                .reshape((dim1, dim2))
                .T
            )
            # apply contractive mapping
            if renorm:
                # this line is not needed, since we tracenorm anywat.
                # renorm_scale = rho.ravel() @ projector_sum_col.ravel()
                rho[:, :] = proj_sum_inv @ k_operator @ rho @ k_operator @ proj_sum_inv
            else:
                # use with slice to forco array mutations
                rho[:, :] = k_operator @ rho @ k_operator
            rho[:, :] = rho / np.trace(rho)  # use tracenorm
            # prepare for next round
            # consider dot product impl. here
            distance = np.linalg.norm(rho - rho_old).real
            scaler_vect[:] = 0  # reset scaler_vect
            counter += 1
        return counter, distance
