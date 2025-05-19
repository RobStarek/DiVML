"""
Test of two-qubit quantum state reconstruction.
* Two-qubit state reconstruction.
* Reconstruction with renormalization.
* Process reconstruction.
"""
from multiprocessing import Process, freeze_support
import unittest
from functools import reduce
from time import time_ns
import numpy as np
# import sys
# import os
# sys.path.insert(0, os.path.abspath(
#     os.path.join(os.path.dirname(__file__), '..')))
import dvml.utils
import dvml

import logging
logger = logging.getLogger(__name__)


def block_ket(theta, phi):
    """
    Return a qubit column-vector with Bloch sphere coordinates theta, phi.
    theta - lattitude measured from |0> on Bloch sphere
    phi - longitude measured from |+> on Bloch sphere (phase)
    """
    return np.array([[np.cos(theta/2)], [np.sin(theta/2)*np.exp(1j*phi)]])


class TestSixQubitStareReconstruction(unittest.TestCase):

    def test_paralel_random(self):
        SAMPLES = 16
        LO = np.array([[1], [0]])
        HI = np.array([[0], [1]])
        Plus = (LO+HI)*(2**-.5)
        Minus = (LO-HI)*(2**-.5)
        RPlu = (LO+1j*HI)*(2**-.5)
        RMin = (LO-1j*HI)*(2**-.5)

        qubits = 6
        print('Defining measurements...')
        # Definion of measurement order, matching data order
        Order = [[LO, HI, Plus, Minus, RPlu, RMin]]*qubits
        pis = dvml.utils.make_projector_array(
            Order, False)  # Prepare (Rho)-Pi vect

        print('Generating data...')
        my_data = []
        true_rhos = []

        for i in range(SAMPLES):
            subkets = []
            for j in range(qubits):
                theta = np.random.random()*np.pi
                phi = np.random.random()*np.pi*2
                subkets.append(np.cos(theta)*LO + np.sin(theta)*HI*np.exp(1j*phi))            
            ket_gt = reduce(np.kron, subkets)
            rho_gt = ket_gt @ ket_gt.T.conj()
            true_rhos.append(rho_gt)
            my_data.append(
                1e-6 + np.array([np.abs(pi.T.conj() @ ket_gt)**2 for pi in pis]).ravel())

        my_data = np.array(my_data)

        print("Preparing reconstructer.")
        R = dvml.Reconstructer(
            pis, max_iters=150, threshold=1e-7, paralelize=True)
        print(R)

        print("Running reconstructions.")
        t0 = time_ns()
        outs = R.reconstruct(my_data)
        t1 = time_ns()
        print(f"Duration : {(t1-t0)*1e-9} sec per {SAMPLES} = {(t1-t0)*1e-9/SAMPLES:.3e} sec per sample.")

        outs = np.array(outs)

        print("Testing fidelity...")
        fidelities = np.array([np.trace(rho_ref @ rho_rec).real
                              for rho_rec, rho_ref in zip(outs, true_rhos)])
        print(np.min(fidelities))
        print(np.mean(fidelities))
        print(np.max(fidelities))
        self.assertGreater(np.min(fidelities), 0.99,
                           "low reconstruction fidelity")

if __name__ == '__main__':
    unittest.main()
