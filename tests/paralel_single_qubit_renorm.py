"""
Test of renormalization.
"""

from functools import reduce
from multiprocessing import Process, freeze_support
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib
import dvml
import dvml.utils

importlib.reload(dvml)
importlib.reload(dvml.utils)

def block_ket(theta, phi):
    """
    Return a qubit column-vector with Bloch sphere coordinates theta, phi.
    theta - lattitude measured from |0> on Bloch sphere
    phi - longitude measured from |+> on Bloch sphere (phase)
    """
    return np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j*phi)]])

def expectation_value(Ket, M):
    """
    Expectation value <bra|M|ket>.
    """
    return (Ket.T.conjugate() @ M @ Ket)[0,0]

def purity(M):
    """
    Purity of the density matrix M.
    For n qubits, minimum is (2^n).
    """
    norm = np.trace(M).real
    #equivalent to np.trace(M @ M)/(norm**2), but off-diagonal elements are not computed
    return (M.T.ravel() @ M.ravel()).real/(norm**2)

if __name__ == '__main__':
    freeze_support()
    ps = [
        block_ket(0.1, 0),
        block_ket(np.pi-0.1, 0),
        block_ket(np.pi/2, 0+0.1),
        block_ket(np.pi/2, np.pi-0.1),
        block_ket(np.pi/2, np.pi/2),
        block_ket(np.pi/2+0.05, -np.pi/2)
    ]
    ops = np.array([
        (ket @ ket.T.conj()) for ket in ps
    ])

    state = block_ket(1*np.pi/4, -np.pi/2)
    rho = state @ state.T.conj()

    print('Defining measurements...')
    pis = np.array(ps)

    print('Generating data...')
    data = np.array([expectation_value(ket, rho).real for ket in ps])

    my_data = [data]*64

    print("Reconstruction...")
    print("With normalization")
    R = dvml.Reconstructer(pis, max_iters=1000, thres=1e-9, renorm=True, paralelize=True)
    print(R)
    out = R.reconstruct(my_data)
    print("P=", purity(out[0]).real)
    
    print("And without normalization")
    R = dvml.Reconstructer(pis, max_iters=1000, thres=1e-9, renorm=False, paralelize=True)
    print(R)
    out = R.reconstruct(my_data)
    print("P=", purity(out[0]).real)
    