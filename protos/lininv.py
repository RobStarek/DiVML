
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

def braket(x : np.ndarray, y : np.ndarray):
    """
    Inner product of two ket-vectors -> C-number
    """
    return np.dot(x.T.conjugate(), y)[0,0]

def ketbra(x : np.ndarray, y : np.ndarray):
    """
    Outer product of two ket-vectors -> C-matrix
    """
    return np.dot(x, y.T.conjugate())

def gen_base10_to_base_m(m):
    """
    Get a function that maps base10 integer to 
    list of base m representation (most significant first)
    """
    def _f(i, digits):
        powers = (m**np.arange(digits))[::-1]
        iact = i
        indices = []
        for j in range(digits):
            idx = iact // powers[j]
            indices.append(idx)
            iact -= (indices[-1]*powers[j])
        return indices
    return _f

def gen_base_m_to_base10(m):
    """
    Get a function that maps list of base m digits (most significant first)
    to base10 integer.
    """    
    def _f(args, n):
        powers = (m**np.arange(n))[::-1]
        return np.array(args) @ powers
    return _f

#auxiliary conversion functions
base10_to_base4 = gen_base10_to_base_m(4)
base10_to_base6 = gen_base10_to_base_m(6)
base6_to_base10 = gen_base_m_to_base10(6)

def spawn_generalized_pauli_operators(*basis_mats):
    """
    Generate list of generalized Pauli operators
    constructed from given basis vectors.
    Args:
        *basis_mats ... n 2x2 array of basis column vectors
    Returns:
        4**n operators
        indexing:        
        j=0 identity
        j=1 sigma z
        j=2 sigma x
        j=3 sigma y
        I = sum_j i_j*4**(n-j-1)
    """
    isqrt = 2**(-.5)
    pauli_ops_base = []
    for basis in basis_mats:
        low = basis[:,0].reshape((2,1))
        high = basis[:,1].reshape((2,1))
        kz0 = low
        kz1 = high
        kx0 = (low+high)*isqrt
        kx1 = (low-high)*isqrt
        ky0 = (low+1j*high)*isqrt
        ky1 = (low-1j*high)*isqrt
        pauli_ops_base.append([np.eye(2)] + [ketbra(ket1, ket1) - ketbra(ket2, ket2) for ket1, ket2 in [(kz0, kz1), (kx0, kx1), (ky0, ky1)]])
    
    n = len(basis_mats)
    gammas = []
    for i in range(4**n):
        indices = base10_to_base4(i, n)
        operators = [pauli_ops_base[j][idx] for j, idx in enumerate(indices)]                    
        gamma = reduce(np.kron, operators)
        gammas.append(gamma)
    return gammas

def make_meas_matrix(basis_ops, meas_ket_array):
    arr = np.zeros((meas_ket_array.shape[0], basis_ops.shape[0]))
    for i, ket in enumerate(meas_ket_array):
        for j, op in enumerate(basis_ops):
            arr[i,j] = (ket.T.conjugate() @ op @ ket)[0,0].real
    inv_arr = np.linalg.pinv(arr)
    return arr, inv_arr

def make_meas_matrix_ops(basis_ops, meas_op_array):
    n, d, d = basis_ops.shape
    left = basis_ops.conj().reshape((n,d*d))
    right = meas_op_array.reshape((-1, d*d)).T
    print(left.shape)
    print(right.shape)
    return left @ right

def psd_proj(rho):
    rho_s = (rho + rho.T.conj())/2
    vals, vecs = np.linalg.eigh(rho_s)
    vals_clip = np.clip(vals, 0, 1)
    rho_proj = vecs @ np.diag(vals_clip) @ vecs.T.conj()
    return rho_proj/np.trace(rho_proj)

def reconstruct(datas, meas_mat_inv, op_basis, project_to_psd=True):
    """
    Args:
        data ... s x n array, n in number of measurements, s is number of tomograms
        meas_mat_inv... n x 4^d complex array with pseudoinverse of meas mat
    """
    ws = (meas_mat_inv @ datas.T).T #array of 4^d x 
    ws = ws.reshape((*ws.shape, 1, 1))
    rp = op_basis.reshape((1, *op_basis.shape))
    rhos = np.sum(rp * ws, axis=1)
    if not project_to_psd:
        return rhos
    return np.array([psd_proj(r) for r in rhos])
    


## Example
# import KetSugar as ks
# import itertools

# eigenbase_dict = {
#     'I' : ((ks.LO, ks.HI), (1,1)),
#     #'X' : ((ks.HHI, ks.HLO), (1,-1)), #sign flip for X measurement
#     'X' : ((ks.HLO, ks.HHI), (1,-1)), #sign flip for X measurement
#     'Y' : ((ks.CLO, ks.CHI),(1,-1)),
#     'Z' : ((ks.LO, ks.HI), (1,-1)),
# }

# lintrap_order = 'ZYX'

# def base_string_to_proj(string : str) -> list:
#     """
#     Input measurement string and get projection operators corresponding to that string, 
#     all combinations that can happen.    
#     """
#     eigenvects = [eigenbase_dict.get(b)[0] for b in string]
#     proj_kets = [ks.kron(*vecs) for vecs in enumerate(itertools.product(*eigenvects))]
#     return proj_kets

# qubits = 1
# tomo_str_gen = (''.join(mstr) for mstr in itertools.product(lintrap_order, repeat=qubits))
# meas3 = np.array([base_string_to_proj(mstr) for mstr in tomo_str_gen])
# meas3 = meas3.reshape((6**qubits, 2**qubits, 1))
# # meas3.shape

# paulis = np.array(spawn_generalized_pauli_operators(*[np.eye(2)]*qubits))
# mm, mmi = make_meas_matrix(paulis, meas3)

# rho1 = ketbra(ks.LO, ks.LO)
# rho2 = ketbra(ks.CLO, ks.CLO)
# data1 = np.array([(ket.T.conj() @ rho1 @ ket)[0,0].real for ket in meas3])
# data2 = np.array([(ket.T.conj() @ rho2 @ ket)[0,0].real for ket in meas3])
# data = np.array([data1, data2])

# rhos = reconstruct(data, mmi, paulis)

