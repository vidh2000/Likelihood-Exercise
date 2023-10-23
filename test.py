from itertools import repeat
from statistics import mean
from multiprocessing import Pool, Manager
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from math import factorial

print(factorial(3), type(factorial(3)))
# VARIABLES
SIGMA = 1
MU = 6
N_BACKGROUND = 1000
N_SIGNAL = 15000
N_BINS = 25

bg = 10*np.random.rand(N_BACKGROUND)
signal = np.random.normal(MU,SIGMA,N_SIGNAL)
data = np.concatenate((bg,signal))



# Get histogram distribution of data
hist, bin_edges = np.histogram(data, bins=N_BINS)

# Get bin heights and centers
bin_heights = hist
bin_width = bin_edges[1]-bin_edges[0]
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

print(len(bin_centers),len(bin_edges))