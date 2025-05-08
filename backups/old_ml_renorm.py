"""
Test reconstruction of with measurement operators that do not add up to 
identity operator.
Without renormalization, the reconstruction purity is systematically lower.
"""

import numpy as np
import MaxLik as ml

#Constants
LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
HLO = (LO+HI)*(2**-.5)
HHI = (LO-HI)*(2**-.5)
CLO = (LO+1j*HI)*(2**-.5)
CHI = (LO-1j*HI)*(2**-.5)

#Short-hand notation
def dagger(x : np.ndarray):
    """
    Hermite conjugation of x.
    """
    return x.T.conjugate()

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

def kron(*arrays):
    """
    Multiple Kronecker (tensor) product.
    Multiplication is performed from left.    
    """
    E = np.eye(1, dtype=complex)
    for M in arrays:
        E = np.kron(E,M)
    return E

def BinKet(i=0,imx=1):
    """
    Computational base states i in imx+1-dimensional vectors.
    """
    ket = np.zeros((imx+1,1), dtype=complex)
    ket[i] = 1
    return ket

#Ket constructors
def BlochKet(theta, phi):
    """
    Return a qubit column-vector with Bloch sphere coordinates theta, phi.
    theta - lattitude measured from |0> on Bloch sphere
    phi - longitude measured from |+> on Bloch sphere (phase)
    """
    return np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j*phi)]])


def ExpectationValue(Ket, M):
    """
    Expectation value <bra|M|ket>.
    """
    return (Ket.T.conjugate() @ M @ Ket)[0,0]

def Purity(M):
    """
    Purity of the density matrix M.
    For n qubits, minimum is (2^n).
    """
    norm = np.trace(M)
    #equivalent to np.trace(M @ M)/(norm**2), but off-diagonal elements are not computed
    return (M.T.ravel() @ M.ravel())/(norm**2)


state = BlochKet(1*np.pi/4, -np.pi/2)
rho = ketbra(state, state)

ps = [
    BlochKet(0.1, 0),
    BlochKet(np.pi-0.1, 0),
    BlochKet(np.pi/2, 0+0.1),
    BlochKet(np.pi/2, np.pi-0.1),
    BlochKet(np.pi/2, np.pi/2),
    BlochKet(np.pi/2+0.05, -np.pi/2)
]
ops = np.array([
    ketbra(ket, ket) for ket in ps
])

data = np.array([ExpectationValue(ket, rho).real for ket in ps])
rhorec = ml.Reconstruct(data, ops, 10000, 1e-9, RhoPiVect=True, Renorm=False)

P = Purity(rhorec)
print("Without normalization")
print(P)


data = np.array([ExpectationValue(ket, rho).real for ket in ps])
rhorec = ml.Reconstruct(data, ops, 10000, 1e-9, RhoPiVect=True, Renorm=True)
P = Purity(rhorec)
print("With normalization")
print(P)
