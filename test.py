from itertools import repeat
from statistics import mean
from multiprocessing import Pool, Manager
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

n=1000
mu_arr = np.random.normal(5,0.1,n)
sig_arr = np.random.normal(5,1,n)


def f(x,y):
    return x*y

Z = []
for x in mu_arr:
    row = []
    for y in (sig_arr):
        z = f(x,y)
        row.append(z)
    Z.append(row)

plt.figure()
h = plt.contourf(np.linspace(0,10,n), np.linspace(0,10,n), Z, alpha=1.0,cmap="nipy_spectral",levels=200)
plt.xlabel(r"$\mu $")
plt.ylabel(r"$\sigma $")
clb = plt.colorbar()
clb.set_label(r'NLL $(\mu, \sigma)$')
plt.tight_layout()
plt.show()