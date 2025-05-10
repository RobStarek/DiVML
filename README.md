# DiVML

Discrete-variable quantum maximum-likelihood reconstruction.

## Features
- Maximum-likelihood reconstruction using PyTorch and numba/numpy.
- Reconstruction of quantum processes and states.
- Support for renormalization.
- GPU-batched processing.


## Installation

Here is how it can be installed locally.

First, build the package:
```bash
python setup.py sdist bdist_wheel   
```
and then, use pip to install it:
```bash
pip install dist/divml-0.1.0.tar.gz  
```

## Example:

```python
#define measurements
LO = np.array([[1], [0]])
HI = np.array([[0], [1]])
Plus = (LO+HI)*(2**-.5)
Minus = (LO-HI)*(2**-.5)
RPlu = (LO+1j*HI)*(2**-.5)
RMin = (LO-1j*HI)*(2**-.5)

Order = [[LO, HI, Plus, Minus, RPlu, RMin]]*1
ket_gt = Plus
rho_gt = ket_gt @ ket_gt.T.conj()
pis = dvml.utils.make_projector_array(
    Order, False)  # Prepare (Rho)-Pi vect

#generate data
probs = 1e-9 + \
    np.array([np.abs(pi.T.conj() @ ket_gt)**2 for pi in pis]).ravel()
my_data = np.array([probs])

#instantiate reconstructer
reconstructer = dvml.Reconstructer(pis, paralelize=True)
print(rhos[0])

#run reconstruction
rhos = reconstructer.reconstruct(my_data)
print(rhos[0])
```
        
## References:
1. Jezek, Fiurasek, Hradil, Quantum inference of states and processes, Phys. Rev. A 68, 012305 (2003) (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.68.012305)
2. [Fiurasek, Hradil, Maximum-likelihood estimation of quantum processes, Phys. Rev. A 63, 020101(R) (2001)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.63.020101)
3. [Quantum State Estimation. Lecture Notes in Physics (Springer Berlin Heidelberg, 2004), ISBN: 978-3-540-44481-7](https://doi.org/10.1007/b98673)

