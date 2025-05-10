"""
This module provides a PyTorch-based backend for iterative maximum-likelihood 
reconstruction of density matrices from tomographic data. It includes utilities 
for GPU memory management, Gram-Schmidt orthogonalization, and reconstruction 
algorithms optimized for parallel processing.
Classes:
    TorchBackend: Implements the backend for reconstructing density matrices 
    using PyTorch. It supports GPU acceleration and provides methods for 
    initialization, parameter setting, and reconstruction.
Functions:
    _npy_chunk_iter(array, batch_size):
        Generator function to iterate over chunks of a NumPy array.
    _get_available_gpu_memory(device='cuda'):
        Returns the available GPU memory in bytes for the specified device.
    _guess_needed_memory(batch, dim, projs):
        Estimates the memory required for a reconstruction process.
    _gram_schmidt(input_col_vectors):
        Performs vectorized Gram-Schmidt orthogonalization on input column vectors.
TorchBackend Methods:
    __init__():
        Initializes the TorchBackend object with default parameters.
    initialize(measurement_description, *args, **kwargs):
        Initializes the reconstruction process by setting up projectors, 
        iteration limits, and renormalization.
    set_parameters(*args, **kwargs):
        Placeholder for setting additional parameters (not implemented).
    reconstruct_data(tomograms, *args, **kwargs):
        Reconstructs density matrices from tomographic data in batches.
    reconstruct(datas):
        Reconstructs a single batch of tomographic data into density matrices.
    _reconstruct_loop_torch(data_in, max_iters, thres, aux_rpv_columns, renorm=False, proj_sum_inv=None, check_every=10):
        Implements the iterative maximum-likelihood reconstruction loop using PyTorch. (Not intended to direct use.)
Constants:
    DEVICE: Specifies the device to use ('cuda' if available, otherwise 'cpu').
    DTYPE: Data type for complex tensors (torch.complex64).
    FDTYPE: Data type for float tensors (torch.float32).
Dependencies:
    - PyTorch
    - NumPy
    - Optional: Numba (for accelerated Gram-Schmidt orthogonalization)
"""
import logging
import itertools
import numpy as np
import torch
from dvml.backend.backend_template import BackendTemplate
logger = logging.getLogger(__name__)

try:
    import numba as nb
    njit = nb.njit
except ImportError:
    logger.warning("Numba could not be imported.")

    def njit(f):
        return f


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    logger.warning("Warning: CPU backend used.")

# fast variant
DTYPE = torch.complex64
FDTYPE = torch.float32
# # precise variant
# DTYPE = torch.complex128
# FDTYPE = torch.float64

DEFAULT_MAX_ITERS = 100
DEFAULT_THRES = 1e-6
HERM_TOL = 1e-8
ORTH_TOL = 1e-7



def _npy_chunk_iter(array, batch_size):
    n = array.shape[0]
    i = 0
    logger.debug('Batch size: %d', batch_size)
    while ((i*batch_size + batch_size) < (n+batch_size-1)) or (i == 0):
        logger.debug('Batch: %d', i)
        yield array[batch_size*i: batch_size*(i+1)]
        i += 1


def _get_available_gpu_memory(device='cuda'):
    """
    Get the available GPU memory in bytes.
    Args:
        device (str): The device to query (e.g., 'cuda:0').
    Returns:
        int: Available memory in bytes.
    """
    if torch.cuda.is_available():
        # gpu_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = reserved_memory - allocated_memory
        return free_memory
    else:
        raise RuntimeError("CUDA is not available.")


def _guess_needed_memory(batch, dim, projs):
    """Estimate how much memory is consumed by a reconstruction."""
    meas_op_size = dim*dim*projs*DTYPE.itemsize
    rhos_size = batch*dim*dim*DTYPE.itemsize
    k_size = dim*dim*DTYPE.itemsize
    proj_sum_size = dim*dim*DTYPE.itemsize
    data_size = projs*FDTYPE.itemsize
    total = meas_op_size+batch*rhos_size+k_size+proj_sum_size
    return meas_op_size, rhos_size, k_size, proj_sum_size, data_size, total

# I have rewritten that in torch, but for some reason
# it still seems quicker on CPU with numba


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


class TorchBackend(BackendTemplate):

    def __init__(self):
        # super().__init__()
        self.renorm = None
        self.aux_meas_ops_cols = None
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
        """Initialize the reconstructer object by specifying projectors, defining iteration 
        limits, and setting renormalization.

        Args:
            measurement_description (ndarray): Array of projector kets or operators.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments.

        Keyword Args:
            renorm (bool): If True, renormalization of operators is performed. 
            Needed for measurements that do not sum to the identity matrix. Defaults to False.
            md_has_ops (bool): If True, indicates that `measurement_description` contains operators. 
            Defaults to False.
            max_iters (int): Maximum number of iterations. Defaults to 100.
            thres (float): Threshold for the Frobenius norm of the reconstruction step. 
            If the norm is less than this, the iteration stops. Defaults to 1e-6.
            paralelize (bool): If True, enables parallelization. Defaults to True.
            batch_size (int): Batch size for processing. If not provided, it is estimated 
            based on available GPU memory.
        """
        # rpv_gpu = torch.from_numpy(rpv).to(dtype).cuda(device=device)
        # rp_rho = (rpv_gpu.reshape((nproj, dim, 1)) * rpv_gpu.conj().reshape((nproj, 1, dim)))
        self.renorm = kwargs.get('renorm', False)
        md_has_ops = kwargs.get('md_has_ops', False)
        self.max_iters = kwargs.get('max_iters', DEFAULT_MAX_ITERS)
        self.thres = kwargs.get('thres', DEFAULT_THRES)
        self.paralelize = kwargs.get('paralelize', True)
        dim = measurement_description.shape[1]

        # estimate recommented batch size
        torch.cuda.empty_cache()
        est_mem_size = torch.cuda.get_device_properties(DEVICE).total_memory

        meas_op_size, rhos_size, k_size, proj_sum_size, data_size, total_size = _guess_needed_memory(
            1, dim, measurement_description.shape[0])
        est_batch_size = int(
            (est_mem_size - (meas_op_size + proj_sum_size))/(k_size + rhos_size + data_size)) - 1
        est_batch_size = max(1, est_batch_size)
        logger.debug('recommended batch: %d', est_batch_size)

        self.batch_size = kwargs.get('batch_size', est_batch_size)
        if not self.paralelize:
            self.batch_size = 1

        nproj = measurement_description.shape[0]
        if not md_has_ops:
            meas_op_gpu = torch.from_numpy(
                measurement_description).to(DTYPE).cuda(device=DEVICE)
            rp_rho_gpu = (meas_op_gpu.reshape((nproj, dim, 1)) *
                          meas_op_gpu.conj().reshape((nproj, 1, dim))).detach()
        else:
            rp_rho_gpu = torch.from_numpy(
                measurement_description).to(DTYPE).cuda(device=DEVICE)
        self.n_proj, self.dim1, self.dim2 = rp_rho_gpu.shape
        self.aux_meas_ops_cols = torch.transpose(rp_rho_gpu, dim0=0, dim1=2).reshape(
            self.dim1*self.dim2, self.n_proj).detach()

        self.last_distance = -1
        self.last_counter = -1

        # numpy-calculation, it is done just once
        if self.renorm:
            projector_sum_col = np.sum(
                self.aux_meas_ops_cols.clone().cpu().numpy(), axis=1)
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
        self.proj_sum_inv = torch.from_numpy(self.proj_sum_inv).to(
            DTYPE).cuda(device=DEVICE).detach()

    def set_parameters(self, *args, **kwargs):
        self.max_iters = kwargs.get('max_iters', self.max_iters)
        self.thres = kwargs.get('thres', self.thres)
        self.renorm = kwargs.get('renorm', self.renorm)
        self.paralelize = kwargs.get('renorm', self.paralelize)

    def reconstruct_data(self, tomograms, *args, **kwargs):
        dim = self.dim1
        if isinstance(tomograms, np.ndarray):
            chunk_iterator = _npy_chunk_iter(tomograms, self.batch_size)
        else:
            chunk_iterator = itertools.batched(tomograms, self.batch_size)
        reconst_iterator = (self.reconstruct(tomogram)
                            for tomogram in chunk_iterator)
        output_arr = np.array(list(reconst_iterator), dtype=np.complex64)
        torch.cuda.empty_cache()
        return output_arr.reshape((-1, dim, dim))

    # ------------
    def reconstruct(self, datas):
        """Reconstruct data into density matrix using parameters of the Reconstructer object.

        Args:
            data (ndarray[int] or ndarray[float]): input tomogram. Shape (m, n), where is number of tomograms, and n is number of measured projectors.

        Returns:
            ndarray[complex]: reconstructed and trace-normed density matrix, (m,d,d).
        """
        torch.cuda.empty_cache()
        _data_in = np.asarray(datas, dtype=np.float32)
        data_in = torch.from_numpy(_data_in).to(FDTYPE).cuda(device=DEVICE)
        rho, iteration, distance = self._reconstruct_loop_torch(
            data_in,
            self.max_iters,
            self.thres,
            self.aux_meas_ops_cols,
            self.renorm,
            self.proj_sum_inv
        )
        self.last_counter = iteration
        self.last_distance = distance
        return rho.cpu().numpy()

    @staticmethod
    def _reconstruct_loop_torch(
        data_in,
        max_iters,
        thres,
        aux_rpv_columns,
        renorm=False,
        proj_sum_inv=None,
        check_every=10
    ):
        """
        PyTorch implementation of iterative maximum-likelihood reconstruction.
        All tensors should be on the same device (CPU or GPU).

        Args:
            data_in (torch.Tensor): shape (n_proj,) with float values (measurement data)
            max_iters (int)
            thres (float)
            aux_rpv_columns (torch.Tensor): shape (d², n_proj), complex dtype
            rho (torch.Tensor): shape (d, d), complex dtype
            renorm (bool): apply inverse sum operator to normalize (optional)
            proj_sum_inv (torch.Tensor): shape (d, d), complex dtype (optional)

        Returns:
            torch.Tensor: reconstructed density matrix (complex)
            int: iteration count
            float: final distance
        """
        device = aux_rpv_columns.device
        dtype = aux_rpv_columns.dtype

        dim_proj, n_proj = aux_rpv_columns.shape
        dim1 = int(np.sqrt(dim_proj))
        dim2 = dim1
        batch_size, n_proj_data = data_in.shape
        assert n_proj_data == n_proj
        # allocate tensors, keep the detached (no backpropagation is needed)
        with torch.no_grad():
            probs = torch.zeros((batch_size, n_proj), dtype=FDTYPE)
            scaler_vect = torch.zeros_like(probs)
            weighted = torch.zeros((dim1*dim2, batch_size), dtype=DTYPE)
            k_operator = torch.zeros((batch_size, dim1, dim2), dtype=DTYPE)
            traces = torch.zeros((batch_size, 1, 1), dtype=FDTYPE)
            rho_old = torch.eye(dim1, dtype=dtype, device=device).unsqueeze(
                0).repeat(batch_size, 1, 1)
            rho = rho_old.clone()
            final_distances = torch.zeros(
                batch_size, dtype=torch.float32, device=device)
            if renorm:
                proj_sum_inv_w = proj_sum_inv.view(1, dim1, dim2)
            for counter in range(max_iters):
                rho_old.copy_(rho)

                # Calculate probabilities (batched)
                probs = torch.matmul(
                    rho.view(batch_size, 1, -1), aux_rpv_columns).real.view(batch_size, -1)
                scaler_vect = data_in / probs
                # Build K operator (batched)
                weighted = torch.matmul(
                    aux_rpv_columns, scaler_vect.T.to(dtype=dtype))
                k_operator = weighted.T.view(
                    batch_size, dim1, dim2).transpose(-2, -1)
                # Contractive map update (batched)
                if renorm:  # and proj_sum_inv is not None
                    rho = (torch.matmul(
                        proj_sum_inv @ k_operator, torch.matmul(
                            rho, k_operator)
                    ) @ proj_sum_inv)
                else:
                    rho = torch.matmul(
                        k_operator, torch.matmul(rho, k_operator))

                # Normalize (batched)
                traces = torch.einsum('bii->b', rho).view(batch_size, 1, 1)
                rho /= traces

                # Convergence check (batched Frobenius distance)
                if (counter % check_every) == 0:
                    distances = torch.linalg.norm(
                        rho - rho_old, dim=(-2, -1)).real
                    converged = distances <= thres
                    if converged.all():
                        final_distances[converged] = distances[converged]
                        break
        return rho, counter, final_distances
