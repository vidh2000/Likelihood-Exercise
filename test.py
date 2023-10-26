from itertools import repeat
from statistics import mean
from multiprocessing import Pool, Manager
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from scipy.stats import truncnorm
import scipy.stats as stats


N_signals=1500


print(len(signals_asimov_dataset))


plt.hist(signals_asimov_dataset)
plt.show()