import numpy as np
import random as rand
import matplotlib.pyplot as plt
from functions import *
from scipy import optimize
from multiprocessing import Pool, freeze_support
from functools import partial
from functions import *




if __name__=="__main__":
   
    # VARIABLES
    SIGMA = 1
    MU = 6
    N_meshpoints = 50

    # Create Signal and Background data 
    bg = 10*np.random.rand(1)
    signal = np.random.normal(MU,SIGMA,15000)
    data = np.concatenate((bg,signal))

    # Data plotting

    plt.figure("Data")
    plt.hist(data,bins=30)
    plt.xlabel("x")
    plt.ylabel("N")

    
    mu_arr = np.linspace(4,12,N_meshpoints)
    sig_arr = np.linspace(0.7,5,N_meshpoints)

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