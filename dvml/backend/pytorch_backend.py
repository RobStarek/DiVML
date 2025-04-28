import logging
import itertools
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
import torch
import numpy as np
from dvml.backend.backend_template import BackendTemplate


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    logging.warning("Warning: CPU backend used.")

# fast variant
DTYPE = torch.complex64
FDTYPE = torch.float32


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

# to be torch-rewritten


def _gram_schmidt(X, row_vecs=False, norm=True):
    """
    Vectorized Gramm-Schmidt orthogonalization.
    Creates an orthonormal system of vectors spanning the same vector space
    which is spanned by vectors in matrix X.

    Args:
        X: matrix of vectors
        row_vecs: are vectors store as line vectors? (if not, then use column vectors)
        norm: normalize vector to unit size
    Returns:
        Y: matrix of orthogonalized vectors
    Source: https://gist.github.com/iizukak/1287876#gistcomment-1348649
    """
    if not row_vecs:
        X = X.T
    Y = X[0:1, :].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag(
            (X[i, :].dot(Y.T) / np.linalg.norm(Y, axis=1) ** 2).flat).dot(Y)
        Y = np.vstack((Y, X[i, :] - proj.sum(0)))
    if norm:
        Y = np.diag(1 / np.linalg.norm(Y, axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


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
        """Initialize Reconstructer. Specifiy used projectors, define iteration limits, and
        set renormalization.
        Args:
            rpv (iterable or ndarray(complex)): iterable/array of projector kets or operators
            max_iters (int): maximum number of iterations
            thres (float): if frobenius norm of reconstruction step is less than this, stop iteration
            renorm (bool, optional): If true, renormalization of operators is performed.
                Needed for measurements that do not sum to identity matrix. Defaults to False.
            rpv_has_ops (bool, optional): If rp_rho contains operators, set it to True and rpv is assigned to rp_rho. Defaults to False.
        """
        # rpv_gpu = torch.from_numpy(rpv).to(dtype).cuda(device=device)
        # rp_rho = (rpv_gpu.reshape((nproj, dim, 1)) * rpv_gpu.conj().reshape((nproj, 1, dim)))
        self.renorm = kwargs.get('renorm', False)
        md_has_ops = kwargs.get('md_has_ops', False)
        self.max_iters = kwargs.get('max_iters', 100)
        self.thres = kwargs.get('thres', 1e-6)
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
        if self.renorm:
            projector_sum_col = np.sum(self.aux_meas_ops_cols, axis=1)
            proj_sum_eval, proj_sum_evec = np.linalg.eigh(
                projector_sum_col.reshape((dim, dim)).T
            )
            if np.sum(np.abs(proj_sum_eval - proj_sum_eval[0])) < dim * 1e-14:
                proj_sum_evec = np.eye(dim, dtype=complex)
            elif (
                np.abs(
                    np.sum(proj_sum_evec @
                           proj_sum_evec.T.conjugate() - np.eye(dim))
                )
                > dim * 1e-14
            ):
                # to be torch-rewritten
                proj_sum_evec = _gram_schmidt(proj_sum_evec, False, True)
            proj_sum_diag_inv = np.diag(proj_sum_eval**-1)
            self.proj_sum_inv = (
                proj_sum_evec @ proj_sum_diag_inv @ proj_sum_evec.T.conjugate()
            )
        else:
            self.proj_sum_inv = np.eye(1, dtype=complex)
        self.proj_sum_inv = torch.from_numpy(self.proj_sum_inv).to(
            DTYPE).cuda(device=DEVICE).detach()        

    def set_parameters(self, *args, **kwargs):
        pass

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
                if renorm and proj_sum_inv is not None:
                    rho = (torch.matmul(
                        proj_sum_inv @ k_operator, torch.matmul(
                            rho, k_operator)
                    ) @ proj_sum_inv)
                else:
                    rho = torch.matmul(
                        k_operator, torch.matmul(rho, k_operator))

                # Normalize (batched)
                traces = torch.einsum('bii->b', rho).view(batch_size, 1, 1)
                # print(traces.shape)
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
