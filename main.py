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

    pass # yet to be implemented

    print("\n################################################\
        \n 4a. Using Wilk's theorem to estimate uncertainties \n")
    
    std_guesses = [1,0.2]
    uncs_mu, uncs_sigma = pm_error_finder(nll,data,best_params,std_guesses)
    print(f"mu = {best_params[0]} +- {uncs_mu}")
    print(f"sigma = {best_params[1]} +- {uncs_sigma}")

    std_mu = mean([abs(x) for x in uncs_mu])
    std_sigma = mean([abs(x) for x in uncs_sigma])
    print(f"std_mu = {round(std_mu,4)}, std_sigma = {round(std_sigma,4)}")

    print("\n################################################\
        \n 4. Verify Wilk's theorem with many datasets \n")
    
    N_datasets = 1000
    
    # Generate fluctuating datasets
    partial_data_gen_task = partial(data_generation_task,
            n_background=N_BACKGROUND,mu = MU,sig = SIGMA,n_signal = N_SIGNAL,
            n_datasets = N_datasets)
    with Pool() as pool:
        # Parallelised dataset production - [(mu1,sig1),..] returned
        #params_arr = pool.map(partial_data_gen_task, range(N_datasets))
        output = pool.map(partial_data_gen_task, range(N_datasets))
    
    params_arr = [out[0] for out in output]
    mus_arr = [params[0] for params in params_arr]
    sigs_arr = [params[1] for params in params_arr]
    
    data_arr = [out[1] for out in output]
    fulldata = [item for sublist in data_arr for item in sublist]
    

    # Checking what fraction of data lies in 1sigma/2sigma intervals
    muOneStdFrac = fracOfDataInRange(mus_arr,MU,std_mu)
    muTwoStdFrac = fracOfDataInRange(mus_arr,MU,2*std_mu)

    SigmaOneStdFrac = fracOfDataInRange(sigs_arr,SIGMA,std_sigma)
    SigmaTwoStdFrac = fracOfDataInRange(sigs_arr,SIGMA,2*std_sigma)

    print("\nFraction of data contained within [1 std, 2 std] for 1D = [68.27%, 95.45%]")
    print(f"MU fraction of data in [1std, 2std] = [{muOneStdFrac}, {muTwoStdFrac}]")
    print(f"SIGMA fraction of data in [1std, 2std] = [{SigmaOneStdFrac}, {SigmaTwoStdFrac}] \n")


    # 1D per parameter variations plots (see if Wilk's theorem works)
    title = "Fluctuated 1000 datasets parameters distributions"
    ymax = 1
    N_bins=20
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(mus_arr, #bins= N_bins, 
             #density = True, 
             weights=np.ones(len(mus_arr)) / float(len(mus_arr)),
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
             #density = True,
             weights = np.ones(len(sigs_arr)) / float(len(sigs_arr)),
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
    
    print("\n################################################\
        \n 5. 2D phase space Wilk's theorem check \n")
    
    # 2D parameter scatter plots (see if Wilk's theorem works)
    df = 2  # For a two-parameter problem

    # Calculate χ² values for 1 and 2 sigma confidence intervals
    chi2_1sigma = chi2.ppf(0.6827, df)
    chi2_2sigma = chi2.ppf(0.9545, df)


    print(f"1 Sigma (68%) Confidence Interval χ² Value for DoF={df}: {chi2_1sigma:.2f}")
    print(f"2 Sigma (95%) Confidence Interval χ² Value for DoF={df}: {chi2_2sigma:.2f}")


    title = "2D phasespace (Wilk's Theorem Contours)"
    plt.figure(title)
    
    xAxis_arr = np.linspace(min(mus_arr)-std_mu,max(mus_arr) +std_mu,100)#len(mus_arr))
    yAxis_arr = np.linspace(min(sigs_arr)-std_sigma,max(sigs_arr)+std_sigma,100)#len(sigs_arr))

    # For NLL plot and contour drawing take dataset which has param values
    # closest to the true MU and SIGMA values
    dataBest = data_arr[find_optimal_data_distrib(mus_arr,sigs_arr,MU,SIGMA)]
    nllOverNdatasets = nll_mesh(xAxis_arr,yAxis_arr,dataBest)
    nll_min = nll([MU,SIGMA],dataBest)
    h = plt.contour(xAxis_arr, yAxis_arr,nllOverNdatasets,
                    levels=[nll_min+chi2_1sigma, nll_min+chi2_2sigma],
                    colors=["darkorange","forestgreen"],linewidths = 2)
                    #alpha=1.0,cmap="nipy_spectral",levels=200)
    # Label contour lines
    fmt = {}
    contourNames = [r"$1\sigma$", r"$2\sigma$"]
    for l, s in zip(h.levels, contourNames):
        fmt[l] = s
    plt.clabel(h, h.levels, inline=True, fmt=fmt, fontsize=14)

    # Plot scatter points from different dataset fits
    plt.scatter(mus_arr,sigs_arr,color="black", marker=".", 
                                        label=r'Data $(\mu, \sigma)$')
    plt.scatter(MU,SIGMA, color="red", marker="o", label=r"True $(\mu,\sigma)$")
    
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\sigma$")
    plt.legend()
    plt.tight_layout()
    
    # Fraction of data lying in contours 1std/2std
    frac1std = fracOfDataInStd(nllOverNdatasets,nll_min,1)
    frac2std = fracOfDataInStd(nllOverNdatasets,nll_min,2)
    print("\nFraction of data contained within [1 std, 2 std] for 2D = [68.27%, 95.45%]")
    print(f"Data gives: [{frac1std}, {frac2std}]")




    print("FINISHED")
    plt.show() 