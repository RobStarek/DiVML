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
        self.aux_rpv_cols = np.einsum("ijk->kji", rp_rho).reshape(
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

    @staticmethod
    @nb.jit(nopython=True)
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
            #consider dot product impl. here
            distance = np.linalg.norm(rho - rho_old).real
            scaler_vect[:] = 0  # reset scaler_vect
            counter += 1
        return counter, distance


if __name__ == '__main__':
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

    print('Reconstruction actual')
    R.max_iters = 100
    %timeit rho = R.reconstruct(probs)
    print("Purity")
    print(np.trace(rho @ rho))