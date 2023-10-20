from itertools import repeat
from statistics import mean
from multiprocessing import Pool, Manager
from functools import partial


SIGMA = 1
MU = 6
N_BACKGROUND = 1
N_SIGNAL = 15000

def data_generation_task(i,n_background,mu,sig,n_signal):
    
    return 1

if __name__=="__main__":

    fun = partial(data_generation_task,
                n_background=N_BACKGROUND,mu = MU,sig = SIGMA,n_signal = N_SIGNAL)

    with Pool() as pool:
        # Parallelised dataset production - [(mu1,sig1),..] returned
        params_arr = pool.map(fun, range(10))
    
    print(params_arr)