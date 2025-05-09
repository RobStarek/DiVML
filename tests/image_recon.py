from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
#Minimal example
import sys
import os
from tqdm import tqdm
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dvml.utils import make_projector_array
from dvml import Reconstructer

if __name__ == '__main__':
    print('Defining measurements...')
    LO = np.array([[1],[0]])
    HI = np.array([[0],[1]])
    Plus = (LO+HI)*(2**-.5)
    Minus = (LO-HI)*(2**-.5)
    RPlu = (LO+1j*HI)*(2**-.5)
    RMin = (LO-1j*HI)*(2**-.5)
    Order = [[LO,HI,Plus,Minus,RPlu,RMin]]*1 #Definion of measurement order, matching data order
    pis = make_projector_array(Order, False) #Prepare (Rho)-Pi vect  



    # -------- Generate data----------
    print("Generating data...")
    dragon = plt.imread('tests/dragon.png')
    h, w, _ = dragon.shape

    thetas = 0.5*np.pi*dragon[:,:,0]
    phis = np.pi*2*(dragon[:,:,1]-0.5)
    dop = 1 - dragon[:,:,2]


    ## Slower, but more readable generation of 
    ## density matrices
    # # def make_rho(theta, phi, dop):
    # #     ket = np.array((np.cos(theta), np.sin(theta)*np.exp(1j*phi))).reshape((2,1))
    # #     rho_pure = ket @ ket.T.conj()
    # #     rho = rho_pure*dop + (1-dop)*np.eye(2)/2
    # #     return rho
    # # rhos = np.array([make_rho(*args) for args in tqdm(dragon.reshape(-1,3))])
    # # rhos = rhos.reshape(h, w, 2, 2)

    # #Faster equivalent, vectorized approach.
    ket = np.vstack((
        np.cos(thetas.ravel()),
        np.sin(thetas.ravel())*np.exp(1j*phis.ravel())
        )).T.reshape((-1,2,1))
    rhos_pure = ket * ket.conj().reshape((-1,1,2))
    rhos_pure_rav = rhos_pure.reshape((-1,4))

    eye_vec = np.eye(2).reshape((1,4))*0.5
    dop = dop.reshape((-1,1))

    rhos_mixed = dop * rhos_pure_rav + eye_vec*(1-dop)

    pis_ravel = np.transpose(
        (pis @ pis.conj().reshape(6,1,2)), 
        axes=(0,2,1)
    ).reshape(6,4)
    data = (rhos_mixed @ pis_ravel.T).real.reshape((h,w,6)).astype(np.float32) + 1e-6
    data = np.clip(np.random.normal(data, 1e-2), 0, None)
    np.save('dragon_tomo.npy', data)

    # -------- Load generated data----------
    print("Reconstructing...")
    data = np.load('dragon_tomo.npy').reshape((-1,6))
    ITERS = 50
    t0 = time.time()
    rec = Reconstructer(pis, paralelize=True, iters=ITERS)
    t1 = time.time()
    rhos = rec.reconstruct(data)
    t2 = time.time()
    print(f'reconstructor initialization: {t1- t0:.3f} sec')
    print(f'reconstruction in total: {t2- t1:.1f} sec')
    print(f'reconstruction per pixel: {(t2- t1)/(1024*1024):.3e} sec/pixel')

    # rhos = rhos.reshape((1024,1024,2,2))

    rthetas = (np.arccos(rhos[:,0,0].real)/2).reshape((1024,1024))
    rphis = np.angle(rhos[:,1,0]).reshape((1024,1024))
    purs = np.sum((rhos.reshape((-1,4)) * np.transpose(rhos, axes=(0,2,1)).reshape((-1,4))), axis=1).real
    rdops = np.sqrt(2*purs - 1).reshape((1024,1024))

    red = 2*rthetas/np.pi
    green = (rphis+np.pi)/(np.pi*2)
    blue = 1 - rdops

    composed = np.zeros((1024,1024,3))
    composed[:,:,0] = red
    composed[:,:,1] = green
    composed[:,:,2] = blue
    composed = np.clip(composed, 0, 1)
    plt.imshow(composed)
    plt.show()
    plt.imshow(dragon)
    plt.show()
    print("Done")