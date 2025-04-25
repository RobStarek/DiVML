import numpy as np

N = 10
D = 4
a = np.arange((D*N)).reshape((N,D))
aa = a.reshape((N,D,1)) * a.reshape((N,1,D)).conj()