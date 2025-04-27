#Minimal example
from functools import reduce
import numpy as np
import dvml
import dvml.utils


LO = np.array([[1],[0]])
HI = np.array([[0],[1]])
Plus = (LO+HI)*(2**-.5)
Minus = (LO-HI)*(2**-.5)
RPlu = (LO+1j*HI)*(2**-.5)
RMin = (LO-1j*HI)*(2**-.5)

print('Defining measurements...')
Order = [[LO,HI,Plus,Minus,RPlu,RMin]]*2 #Definion of measurement order, matching data order
#|01+-RL> state
ket_gt = reduce(np.kron, [LO, HI])
rho_gt = ket_gt @ ket_gt.T.conj()
pis = dvml.utils.make_projector_array(Order, False) #Prepare (Rho)-Pi vect  

print('Generating data...')
probs = 1e-6 + np.array([np.abs(pi.T @ ket_gt)**2 for pi in pis]).ravel()
print(probs)

R = dvml.Reconstructer(pis)
outs = R.reconstruct([probs, probs])
print(outs)