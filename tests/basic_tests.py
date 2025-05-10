"""
Test of two-qubit quantum state reconstruction.
* Two-qubit state reconstruction.
* Reconstruction with renormalization.
* Process reconstruction.
"""
from multiprocessing import Process, freeze_support
import unittest
from functools import reduce
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


class TestTwoQubitStareReconstruction(unittest.TestCase):

    def test_paralel_fixed(self):
        SAMPLES = 64
        LO = np.array([[1], [0]])
        HI = np.array([[0], [1]])
        Plus = (LO+HI)*(2**-.5)
        Minus = (LO-HI)*(2**-.5)
        RPlu = (LO+1j*HI)*(2**-.5)
        RMin = (LO-1j*HI)*(2**-.5)

        print('Defining measurements...')
        # Definion of measurement order, matching data order
        Order = [[LO, HI, Plus, Minus, RPlu, RMin]]*2
        # |01+-RL> state
        ket_gt = reduce(np.kron, [LO, HI])
        rho_gt = ket_gt @ ket_gt.T.conj()
        pis = dvml.utils.make_projector_array(
            Order, False)  # Prepare (Rho)-Pi vect

        print('Generating data...')
        probs = 1e-6 + \
            np.array([np.abs(pi.T.conj() @ ket_gt)**2 for pi in pis]).ravel()
        print(probs)
        my_data = np.array([probs]*SAMPLES)

        R = dvml.Reconstructer(pis, paralelize=True)
        print(R)

        outs = R.reconstruct(my_data)
        outs = np.array(outs)
        print(outs.shape)

        print("Testing fidelity...")
        fidelities = np.array([np.trace(rho_gt @ r) for r in outs])
        self.assertGreater(np.min(fidelities), 0.999,
                           "low reconstruction fidelity")

    def test_paralel_random(self):
        SAMPLES = 64
        LO = np.array([[1], [0]])
        HI = np.array([[0], [1]])
        Plus = (LO+HI)*(2**-.5)
        Minus = (LO-HI)*(2**-.5)
        RPlu = (LO+1j*HI)*(2**-.5)
        RMin = (LO-1j*HI)*(2**-.5)

        print('Defining measurements...')
        # Definion of measurement order, matching data order
        Order = [[LO, HI, Plus, Minus, RPlu, RMin]]*2
        pis = dvml.utils.make_projector_array(
            Order, False)  # Prepare (Rho)-Pi vect

        print('Generating data...')
        my_data = []
        true_rhos = []
        for i in range(SAMPLES):
            theta = np.random.random()*np.pi
            phi = np.random.random()*np.pi*2
            q1 = np.cos(theta)*LO + np.sin(theta)*HI*np.exp(1j*phi)
            theta = np.random.random()*np.pi
            phi = np.random.random()*np.pi*2
            q2 = np.cos(theta)*LO + np.sin(theta)*HI*np.exp(1j*phi)

            ket_gt = reduce(np.kron, [q1, q2])
            rho_gt = ket_gt @ ket_gt.T.conj()
            true_rhos.append(rho_gt)

            my_data.append(
                1e-6 + np.array([np.abs(pi.T.conj() @ ket_gt)**2 for pi in pis]).ravel())

        my_data = np.array(my_data)

        R = dvml.Reconstructer(
            pis, max_iters=100, threshold=1e-8, paralelize=True)
        print(R)

        outs = R.reconstruct(my_data)
        outs = np.array(outs)

        print("Testing fidelity...")
        fidelities = np.array([np.trace(rho_ref @ rho_rec)
                              for rho_rec, rho_ref in zip(outs, true_rhos)])
        self.assertGreater(np.min(fidelities), 0.99,
                           "low reconstruction fidelity")

    def test_process_cnot(self):
        kron = lambda *ops: reduce(np.kron, ops)
        # freeze_support()
        LO = np.array([[1], [0]])
        HI = np.array([[0], [1]])
        Plus = (LO+HI)*(2**-.5)
        Minus = (LO-HI)*(2**-.5)
        RPlu = (LO+1j*HI)*(2**-.5)
        RMin = (LO-1j*HI)*(2**-.5)

        print('Defining measurements...')
        # Definion of measurement order, matching data order
        Order = [[LO, HI, Plus, Minus, RPlu, RMin]]*4
        # |01+-RL> state
        ket_gt = reduce(np.kron, [LO, HI])
        rho_gt = ket_gt @ ket_gt.T.conj()
        pis = dvml.utils.make_projector_array(
            Order, True)  # Prepare (Rho)-Pi vect

        # CNOT CP map
        chiket0 = (kron(LO, LO, LO, LO) + kron(LO, HI, LO, HI) +
                   kron(HI, LO, HI, HI) + kron(HI, HI, HI, LO))*0.5
        chi0 = chiket0 @ chiket0.T.conj()
        chi0 = chi0/np.trace(chi0)

        print('Generating data...')
        probs = 1e-6 + \
            np.array([np.abs(pi.T.conj() @ chiket0)**2 for pi in pis]).ravel()
        probs = np.random.poisson(probs*10000)
        print(probs)
        my_data = np.array([probs])

        R = dvml.Reconstructer(pis, paralelize=True)
        print(R)

        outs = R.reconstruct(my_data)
        outs = np.array(outs)

        print("Testing fidelity...")
        fid = np.trace(chi0 @ outs[0])
        print(fid)
        self.assertGreater(fid, 0.99, "low fidelity")

    def test_renormalized(self):
        ps = [
            block_ket(0.1, 0),
            block_ket(np.pi-0.1, 0),
            block_ket(np.pi/2, 0+0.1),
            block_ket(np.pi/2, np.pi-0.1),
            block_ket(np.pi/2, np.pi/2),
            block_ket(np.pi/2+0.05, -np.pi/2)
        ]

        state = block_ket(1*np.pi/4, -np.pi/2)
        rho = state @ state.T.conj()

        print('Defining measurements...')
        pis = np.array(ps)

        print('Generating data...')
        data = np.array([(ket.T.conj() @ rho @ ket)[0, 0].real for ket in ps])

        my_data = [data]*32

        print("Reconstruction...")
        print("Without normalization")
        R = dvml.Reconstructer(pis, max_iters=1000,
                               thres=1e-9, renorm=False, paralelize=True)
        out = R.reconstruct(my_data)
        pur = np.trace(out[0] @ out[0]).real
        fid = np.trace(out[0] @ rho).real
        print(f'P={pur}, F={fid}')

        print("With normalization")
        R = dvml.Reconstructer(pis, max_iters=1000,
                               thres=1e-9, renorm=True, paralelize=True)

        print(R)
        out = R.reconstruct(my_data)
        pur = np.trace(out[0] @ out[0]).real
        fid = np.trace(out[0] @ rho).real
        print(f'P={pur}, F={fid}')

        self.assertGreater(fid, 0.99, 'low fidelity')
        self.assertGreater(pur, 0.99, 'low purity')


if __name__ == '__main__':
    unittest.main()
