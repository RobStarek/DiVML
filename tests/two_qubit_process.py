"""
Test of two-qubit quantum process reconstruction (four-qubit Choi matrix).
"""

from functools import reduce
import numpy as np
#Minimal example
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import importlib
import dvml
import dvml.utils

importlib.reload(dvml)
importlib.reload(dvml.utils)
from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    kron = lambda *ops: reduce(np.kron, ops)
    # freeze_support()
    LO = np.array([[1],[0]])
    HI = np.array([[0],[1]])
    Plus = (LO+HI)*(2**-.5)
    Minus = (LO-HI)*(2**-.5)
    RPlu = (LO+1j*HI)*(2**-.5)
    RMin = (LO-1j*HI)*(2**-.5)

    print('Defining measurements...')
    Order = [[LO,HI,Plus,Minus,RPlu,RMin]]*4 #Definion of measurement order, matching data order
    #|01+-RL> state
    ket_gt = reduce(np.kron, [LO, HI])
    rho_gt = ket_gt @ ket_gt.T.conj()
    pis = dvml.utils.make_projector_array(Order, True) #Prepare (Rho)-Pi vect  

    #CNOT CP map
    chiket0 = (kron(LO,LO, LO, LO) + kron(LO, HI, LO, HI) + kron(HI, LO, HI, HI) + kron(HI, HI, HI, LO))*0.5
    chi0 = chiket0 @ chiket0.T.conj()
    chi0 = chi0/np.trace(chi0)


    print('Generating data...')
    probs = 1e-6 + np.array([np.abs(pi.T.conj() @ chiket0)**2 for pi in pis]).ravel()
    probs = np.random.poisson(probs*1000)
    print(probs)
    my_data = np.array([probs])

    
    R = dvml.Reconstructer(pis, paralelize=True)
    print(R)
    
    outs = R.reconstruct(my_data)
    outs = np.array(outs)
    print(outs.shape)

    print("Testing fidelity...")
    print(np.trace(chi0 @ outs[0]))