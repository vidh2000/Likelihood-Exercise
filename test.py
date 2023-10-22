from itertools import repeat
from statistics import mean
from multiprocessing import Pool, Manager
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# n=1000
# xarr = np.linspace(0,10,n)
# yarr = np.linspace(0,10,n)

# def gauss(x,mu,sig):
#     return 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mu)**2 / (2*sig**2))


# def f(x,y):
#     return x*y

# Z = []
# for y in yarr:
#     row = []
#     for x in xarr:
#         z = gauss(x,5,1)*gauss(y,5,3)
#         row.append(z)
#     Z.append(row)

# plt.figure()
# h = plt.contourf(xarr, yarr, Z, alpha=1.0,cmap="nipy_spectral",levels=200)
# plt.xlabel(r"x")
# plt.ylabel(r"y")
# clb = plt.colorbar()
# clb.set_label(r'Function')
# plt.tight_layout()
# plt.show()


