import numpy as np
import itertools
import functools

# Auxiliary string, used to generate einsum expressions
_ALPHABET = ",".join([chr(97 + i) for i in range(10)])

def veckron(*vecs) -> np.ndarray:
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
        return functools.reduce(np.kron, vecs, np.eye(1, dtype=complex))
    return np.einsum(_ALPHABET[0: 2 * n - 1], *(v.ravel() for v in vecs)).reshape(-1, 1)


def make_projector_array(projection_order : list , process_tomo : bool = False) -> np.ndarray:
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

