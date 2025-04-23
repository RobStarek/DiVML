# -*- coding: utf-8 -*-
"""MaxLik2.py
Discrete-variable quantum maximum-likelihood reconstruction.

This module provides a simple numpy-implementation of Maximum likelihood reconstruction
method [1,2] for reconstructing low-dimensional quantum states and processes (<=6 qubits in total).

This package is created to work with pure preparation and projections.
However, Reconstructer() accepts any POVM, we just lose the convenience 
of make_projector_array() helper function.

This version avoid unnecessary repeated calculations by memoizing some arrays
in the stage of reconstructed initialization.

The reconstructor object act like function which returns reconstructed density matrix.

Example:
    Minimal single-qubit reconstruction
        import numpy as np
        from MaxLikCore import MakeRPV, Reconstruct
        #Definition of projection vector
        LO = np.array([[1],[0]])
        HI = np.array([[0],[1]])
        PLUS = (LO+HI)*(2**-.5)
        MINUS = (LO-HI)*(2**-.5)
        RPLUS = (LO+1j*HI)*(2**-.5)
        RMINUS = (LO-1j*HI)*(2**-.5)
        #Definion of measurement order, matching data order
        order = [[LO,HI,PLUS,MINUS,RPLUS,RMINUS]]
        #Measured counts
        testdata = np.array([500,500,500,500,1000,1])
        #Prepare vector with measurement definition
        proj_kets = make_projector_array(order, False)
        
        #initialize Reconstructer object
        reconstruct = Reconstructer(proj_kets, max_iters=1000, thres=1e-12)
        #reconstruct data
        rho = reconstruct(testdata)

References:
    1. Fiurasek, Hradil, Maximum-likelihood estimation of quantum processes, 
       Phys. Rev. A 63, 020101(R) (2001) 
       https://journals.aps.org/pra/abstract/10.1103/PhysRevA.63.020101
    2. Paris (ed.), Rehacek, Quantum State Estimation - 2004,
       Lecture Notes in Physics, ISBN: 978-3-540-44481-7,
       https://doi.org/10.1007/b98673

Todo:
    * unit tests
    * benchmarks (agains the first version, MaxLik.py)

"""

# version for experimenting with class based ml
from functools import reduce
import itertools
import numpy as np
import numba as nb
import torch
from concurrent.futures import ThreadPoolExecutor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.complex64
fdtype = torch.float32


# Auxiliary string, used to generate einsum expressions
ALPHABET = ",".join([chr(97 + i) for i in range(10)])


def veckron(*vecs):
    """Kronecker product of multiple vectors.
    If there is only 1 vector, result is trivial.
    For up to 9 vectors, it is implemented with einsum.
    For more vector, it falls back to reduce on np.kron
    Args:
        *vecs (ndarray) : column vectors to be multiplied
    Returns:
        (complex ndarray) : new column vector
    """
    n = len(vecs)
    if n == 1:
        return vecs[0]
    if n > 9:
        return reduce(np.kron, vecs, np.eye(1, dtype=complex))
    # EINSUM_EXPRS_VEC[len(vecs)-2],
    return np.einsum(ALPHABET[0: 2 * n - 1], *(v.ravel() for v in vecs)).reshape(-1, 1)


def make_projector_array(projection_order, process_tomo=False):
    """
    Create list preparation-projection kets.
    Kets are stored as line-vectors or column-vectors in n x d ndarray.
    This function is here to avoid writing explicit nested loops for all 
    combinations of measured projection.
    Args:
        projection_order: list of measured/prepared states on each qubit, first axis denotes qubit,
            second measured states, elements are kets stored column-vectors.
        process_tomo: if True, first half of Order list is regarded as input states and therefore
            conjugated prior building RPV ket.
    Returns:
        RPVvectors (complex ndarray): complex ndarray of preparation-projection kets, 
        order should match the measurement
    """
    n_half = len(projection_order) / 2
    views = (
        map(np.conjugate, proj_kets) if (
            (i < n_half) and process_tomo) else proj_kets
        for i, proj_kets in enumerate(projection_order)
    )
    return np.array(
        [veckron(*kets) for kets in itertools.product(*views)], dtype=complex
    )


def gram_schmidt(X, row_vecs=False, norm=True):
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


class Reconstructer:
    """Reconstructe object.
    Upon initialization, it performs some aux. calculation which are required in the iteration 
    loop, but remains the same throughout the loop. Then it saves some auxiliary arrays as 
    its attributes. These arrays are then mutated by numba which actually performs the 
    reconstruction.
    """

    def __init__(self, rpv, max_iters, thres, renorm=False, rpv_has_ops=False):
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
        dim = rpv.shape[1]
        self.renorm = renorm
        if not rpv_has_ops:
            # construct projector operators (using outer product) from kets
            rp_rho = np.array(
                [np.einsum("i,j", ket.ravel(), ket.conj().ravel())
                 for ket in rpv]
            )
        else:
            rp_rho = rpv

        
        self.n_proj, self.dim1, self.dim2 = rp_rho.shape
        # this is equivalent to np.out plus sum raveling and transpoistion
        _aux_rpv_cols = np.einsum("ijk->kji", rp_rho).reshape(
            (self.dim1 * self.dim2, self.n_proj)
        )
        self.aux_rpv_cols = torch.from_numpy(_aux_rpv_cols).to(dtype).cuda(device=device)

        self.max_iters = max_iters
        self.thres = thres
        self.last_distance = -1
        self.last_counter = -1
        if self.renorm:
            projector_sum_col = np.sum(self.aux_rpv_cols, axis=1)
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
                proj_sum_evec = gram_schmidt(proj_sum_evec, False, True)
            proj_sum_diag_inv = np.diag(proj_sum_eval**-1)
            self.proj_sum_inv = (
                proj_sum_evec @ proj_sum_diag_inv @ proj_sum_evec.T.conjugate()
            )
        else:
            self.proj_sum_inv = np.eye(1, dtype=complex)
        self.proj_sum_inv = torch.from_numpy(self.proj_sum_inv).to(dtype).cuda(device=device)

    def __call__(self, data):
        """
        See Reconstructer.reconstruct()
        """
        return self.reconstruct(data)

    def reconstruct(self, data):
        """Reconstruct data into density matrix using parameters of the Reconstructer object.

        Args:
            data (ndarray[int] or ndarray[float]): input tomogram. It should have as many entries as there is projectors.

        Returns:
            ndarray[complex]: reconstructed and trace-normed density matrix
        """
        sel = np.asarray(data) > 0
        aux_rpv_columns = self.aux_rpv_cols[:, sel]
        _data_in = np.asarray(data[sel], dtype=np.float32)
        data_in = torch.from_numpy(_data_in).to(fdtype).cuda(device= device)        
        # these will mutate
        _rho = np.eye(self.dim1, dtype=np.complex128) / self.dim1
        rho = torch.from_numpy(_rho).to(dtype).cuda(device= device)        
        _rho_old = np.copy(_rho)
        rho = torch.from_numpy(_rho_old).to(dtype).cuda(device= device)

        rho, iteration, distance = self.reconstruct_loop_torch(
            data_in,
            self.max_iters,
            self.thres,
            aux_rpv_columns,
            rho,
            self.proj_sum_inv,
        )
        print('stopped at', iteration, distance)
        self.last_counter = iteration
        self.last_distance = distance
        return rho.cpu().numpy()

    @staticmethod
    def reconstruct_loop_torch(
        data_in,
        max_iters,
        thres,
        aux_rpv_columns,
        rho,
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
        device = rho.device
        dtype = rho.dtype

        # data_in = data_in.to(dtype=fdtype, device=device)
        # aux_rpv_columns = aux_rpv_columns.to(dtype=dtype, device=device)
        # rho = rho.clone()
        rho_old = rho.clone()

        dim1, dim2 = rho.shape
        dim_proj, n_proj = aux_rpv_columns.shape

        for counter in range(max_iters):
            rho_old.copy_(rho)

            # Calculate probabilities
            probs = torch.matmul(rho.view(1, -1), aux_rpv_columns).real.view(-1)
            scaler_vect = data_in / probs

            # Build K operator
            weighted = aux_rpv_columns @ scaler_vect.to(dtype=dtype).view(-1, 1)
            k_operator = weighted.view(dim1, dim2).T

            # Contractive map update
            if renorm and proj_sum_inv is not None:
                rho = proj_sum_inv @ k_operator @ rho @ k_operator @ proj_sum_inv
            else:
                rho = k_operator @ rho @ k_operator

            rho /= torch.trace(rho)

            # Convergence check (Frobenius distance)
            if (counter % check_every) == 0:
                distance = torch.linalg.norm(rho - rho_old).real.item()
                if distance <= thres:
                    break

        return rho, counter + 1, distance

if __name__ == '__main__':
    from time import time
    LO = np.array([[1],[0]])
    HI = np.array([[0],[1]])
    Plus = (LO+HI)*(2**-.5)
    Minus = (LO-HI)*(2**-.5)
    RPlu = (LO+1j*HI)*(2**-.5)
    RMin = (LO-1j*HI)*(2**-.5)
    
    print('Defining measurements...')
    Order = [[LO,HI,Plus,Minus,RPlu,RMin]]*6 #Definion of measurement order, matching data order
    #|01+-RL> state
    ket_gt = reduce(np.kron, [LO, HI, Plus, Minus, RPlu, RMin])
    rho_gt = ket_gt @ ket_gt.T.conj()
    pis = make_projector_array(Order, False) #Prepare (Rho)-Pi vect  

    print('Generating data...')
    probs = 1e-6 + np.array([np.abs(pi.T @ ket_gt)**2 for pi in pis]).ravel()

    print('Initialization of Reconstructer...')
    R = Reconstructer(pis, 100, 1e-6, False, False)
    print('Reconstruction pro forma (1 cycle)')
    R.max_iters = 5
    rho = R.reconstruct(probs)



    print('Reconstruction actually, trying in paralel')
    R.max_iters = 100
    my_data = [probs, np.copy(probs)+1e-4, np.copy(probs)+1e-3]
    t0 = time()
    with ThreadPoolExecutor(3) as executor:
        rhos = list(executor.map(R.reconstruct, my_data))
    t1 = time()
    print('Paralel')
    print(t1-t0, 'sec')
    print((t1-t0)/3, 'sec per data')
    print("Purity")
    for r in rhos:
        print(np.trace(r @ r))

    t0 = time()
    rhos = []
    for _d in my_data:
        rhos.append(R.reconstruct(_d))
    t1 = time()
    print('Sequential')
    print(t1-t0, 'sec')
    print((t1-t0)/3, 'sec per data')
    print("Purity")
    for r in rhos:
        print(np.trace(r @ r))
