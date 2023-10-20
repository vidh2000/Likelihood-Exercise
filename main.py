import numpy as np
import random as rand
import matplotlib.pyplot as plt
from functions import *
from scipy import optimize
from multiprocessing import Pool, freeze_support
from functools import partial


# Functions
def gauss(x,mu,sig):
    return 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mu)**2 / (2*sig**2))

def nll(params, data):
    """
    Negative Log Likehood.
    Parameters:
    - params: [mu,sigma]. Can be floats or arrays
    - data: dataset of x values
    We're dealing with Gaussian signal here hence
    params are as defined below (hardcoded).
    """
    mu = params[0]
    sig = params[1]
    #if type(mu) == float and type(sig) == float:
    nll = len(data)*np.log(sig)
    for x in data:
        nll += (x-mu)**2 / (2*sig*sig)
    #else:

    return nll

# def nllMP(sig_arr,data,mu):
#     """
#     Form of NLL function suitable for the use in parallelised framework 
#     (due to parsing of arguments..)
#     """
#     print(r"mu=",round(mu,3))
#     row = []
#     for sig in sig_arr:
#         param_vec = [mu,sig]
#         NLL = nll(param_vec,data)
#         row.append(NLL)
#     return row

def task(mu,sig_arr,data):
    """
    Task for parallelised NLL mesh generation
    """
    #print(r"mu=",round(mu,3))
    row = []
    for sig in sig_arr:
        param_vec = [mu,sig]
        #if round(mu,3) == 11.388:
        #    print(sig)
        NLL = nll(param_vec,data)
        # NLL = len(data)*np.log(sig)
        # for x in data:
        #     NLL += (x-mu)**2 / (2*sig)
        row.append(NLL)
    return row

def nll_mesh(mu_arr, sig_arr, data):
    """
    Constructs a mesh of NLL values across the mu and sig parameter domain.
    Useful for plotting contours etc.
    Parallelised on the outer loop for speed.
    """
    #partial_nll_func = partial(nllMP,sig_arr,data)
    args = list(zip(mu_arr,
                    [sig_arr for _ in mu_arr],
                    [data for _ in mu_arr]))
    with Pool() as pool:
        mesh = list(pool.starmap(task, args))
    return mesh


if __name__=="__main__":
   
    # VARIABLES
    SIGMA = 1
    MU = 6
    N_meshpoints = 50

    # Create Signal and Background data 
    bg = 10*np.random.rand(1)
    sig = np.random.normal(MU,SIGMA,15000)
    data = np.concatenate((bg,sig))

    # Data plotting

    plt.figure("Data")
    plt.hist(data,bins=30)
    plt.xlabel("x")
    plt.ylabel("N")

    
    mu_arr = np.linspace(2,12,N_meshpoints)
    sig_arr = np.linspace(0.9,1.1,N_meshpoints)

    # NLL plotting 2D
    NLL = nll_mesh(mu_arr,sig_arr,data)
    vec_arr = [[4,0.93], [10,1.8], [5,1.5], [3,1],[6,1],[9,1],]
    for vec in vec_arr:
        print(vec,nll(vec, data))
    title="NLL_2D_mesh"
    plt.figure(title)
    h = plt.contourf(mu_arr, sig_arr, NLL, alpha=1.0,cmap="nipy_spectral",levels=200)
    plt.xlabel(r"$\mu $")
    plt.ylabel(r"$\sigma $")
    clb = plt.colorbar()
    clb.set_label(r'NLL $(\mu, \sigma)$')
    plt.tight_layout()

    # NLL plotting 1D per parameter variations
    title = "NLL(mu) and NLL(sigma)"
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(mu_arr,nll([mu_arr,SIGMA], data),label=rf"$\sigma$ = {SIGMA}")
    ax1.grid(0.1)
    ax1.legend()
    ax1.set_ylabel(r"NLL($\mu$)")
    ax1.set_xlabel(r"$\mu$")

    ax2.plot(sig_arr,nll([MU,sig_arr], data),label=rf"$\mu$ = {MU}")
    ax2.grid(0.1)
    ax2.legend()
    ax2.set_ylabel(r"NLL($\sigma$)")
    ax2.set_xlabel(r"$\sigma$")
    plt.tight_layout()
    #plt.savefig("plots/4/"+title+".pdf",
            #dpi=1200, 
            #bbox_inches="tight")


    # Minimising the NLL to obtain optimal parameters
    init_guess = [5,2]
    results = optimize.minimize(nll,init_guess,(data), method = 'Nelder-Mead')
    print(results)



    plt.show()