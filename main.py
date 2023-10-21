import numpy as np
import random as rand
import matplotlib.pyplot as plt
from functions import *
from scipy import optimize
from multiprocessing import Pool, Manager
from functools import partial
from functions import *
from statistics import mean 
from tqdm import tqdm
from progressbar import ProgressBar, SimpleProgress



# VARIABLES
SIGMA = 1
MU = 6
N_BACKGROUND = 0
N_SIGNAL = 1500


if __name__=="__main__":

    
    

    # Create Signal and Background data 
    print("\n################################################\
        \n 1. Generate datasets\n")
    bg = 10*np.random.rand(N_BACKGROUND)
    signal = np.random.normal(MU,SIGMA,N_SIGNAL)
    data = np.concatenate((bg,signal))

    # Data plotting

    plt.figure("Data")
    plt.hist(data,bins=30,label="Signal + Background")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("N")


    N_meshpoints = 100
    mu_arr = np.linspace(1,9,N_meshpoints)
    sig_arr = np.linspace(0.5,6, N_meshpoints)


    print("\n################################################\
        \n 2a. NLL 2D minimisation and plotting\n")

    # NLL plotting 2D
    NLL = nll_mesh(mu_arr,sig_arr,data)

    # vec_arr = [[4,0.93], [10,1.8], [5,1.5], [3,1],[6,1],[9,1],]
    # for vec in vec_arr:
    #       print(vec,nll(vec, data))

    title="NLL_2D_mesh"
    plt.figure(title)
    h = plt.contourf(mu_arr, sig_arr,NLL, alpha=1.0,cmap="nipy_spectral",levels=200)
    plt.xlabel(r"$\mu $")
    plt.ylabel(r"$\sigma $")
    clb = plt.colorbar()
    clb.set_label(r'NLL $(\mu, \sigma)$')
    plt.tight_layout()

    # Minimising the NLL to obtain optimal parameters
    init_guess = [5,2]
    results = optimize.minimize(nll,init_guess,(data), method = 'Nelder-Mead')
    #print(results)
    best_params = results.x
    print("The best fit parameters are: (mu,sigma) = ", best_params)

    print("\n################################################\
        \n 2b. Profile NLL for each parameter individually\n")

    # NLL plotting 1D per parameter variations
    title = "NLL(mu) and NLL(sigma)"

    mu_arr = np.linspace(1,9,10000)
    sig_arr = np.linspace(0.5,6, 10000)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(mu_arr,nll([mu_arr,SIGMA], data),label=rf"$\sigma$ = {SIGMA}")
    ax1.grid(alpha=0.1)
    ax1.legend()
    ax1.set_ylabel(r"NLL($\mu$)")
    ax1.set_xlabel(r"$\mu$")

    ax2.plot(sig_arr,nll([MU,sig_arr], data),label=rf"$\mu$ = {MU}")
    ax2.grid(alpha=0.1)
    ax2.legend()
    ax2.set_ylabel(r"NLL($\sigma$)")
    ax2.set_xlabel(r"$\sigma$")
    plt.tight_layout()
    #plt.savefig("plots/4/"+title+".pdf",
        #dpi=1200, 
        #bbox_inches="tight")

    print("\n################################################\
        \n 3. Creating multiple fluctuated datasets for comparison\n")

    print("\n################################################\
        \n 4a. Using Wilk's theorem to estimate uncertainties \n")
    uncs_mu, uncs_sigma = pm_error_finder(nll,data,best_params,[1,0.2])
    print(f"mu = {best_params[0]} +- {uncs_mu}")
    print(f"sigma = {best_params[1]} +- {uncs_sigma}")

    std_mu = mean([abs(x) for x in uncs_mu])
    std_sigma = mean([abs(x) for x in uncs_sigma])
    print(std_mu,std_sigma)

    print("\n################################################\
        \n 4a. Verify Wilk's theorem with many datasets \n")
    
    N_datasets = 1000
    
    # Generate fluctuating datasets
    partial_data_gen_task = partial(data_generation_task,
            n_background=N_BACKGROUND,mu = MU,sig = SIGMA,n_signal = N_SIGNAL,
            n_datasets = N_datasets)
    with Pool() as pool:
        # Parallelised dataset production - [(mu1,sig1),..] returned
        params_arr = pool.map(partial_data_gen_task, range(N_datasets))
    
    mus_arr = [params[0] for params in params_arr]
    sigs_arr = [params[1] for params in params_arr]
    
     # NLL plotting 1D per parameter variations
    title = "Fluctuated 1000 datasets parameters distributions"

    ymax = 1
    N_bins=20
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(mus_arr, #bins= N_bins, 
             density = True, stacked=True,
                    label=rf"Distribution over {N_datasets} datasets")
    ax1.vlines(MU,ymin=0,ymax=ymax,linestyles="solid",colors="red", 
              label="True value")
    ax1.vlines(MU+std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="black", 
              label=r"$1\sigma$")
    ax1.vlines(MU-std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="black")
    ax1.vlines(MU+2*std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="blue", 
              label=r"$2\sigma$")
    ax1.vlines(MU-2*std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="blue")
    ax1.grid(alpha=0.1)
    ax1.legend()
    #ax1.set_ylim(top=1)
    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel("Density")

    ax2.hist(sigs_arr,#bins=N_bins, 
             density = True, stacked=True,
                    label=rf"Distribution over {N_datasets} datasets")
    ax2.vlines(SIGMA,ymin=0,ymax=ymax,linestyles="solid",colors="red", 
              label="True value")
    ax2.vlines(SIGMA+std_sigma,ymin=0,ymax=ymax,linestyles="dashed",colors="black", 
              label=r"$1\sigma$")
    ax2.vlines(SIGMA-std_sigma,ymin=0,ymax=ymax,linestyles="dashed",colors="black")
    ax2.vlines(SIGMA+2*std_sigma,ymin=0,ymax=ymax,linestyles="dashed",colors="blue", 
              label=r"$2\sigma$")
    ax2.vlines(SIGMA-2*std_sigma,ymin=0,ymax=ymax,linestyles="dashed",colors="blue")

    ax2.grid(alpha=0.1)
    ax2.legend()
    #ax2.set_ylim(top=1)
    ax2.set_xlabel(r"$\sigma$")
    plt.tight_layout()
    #plt.savefig("plots/4/"+title+".pdf",
        #dpi=1200, 
        #bbox_inches="tight")
    
    
    
    
    
    
    
    
    
    
    print("FINISHED")
    plt.show() 