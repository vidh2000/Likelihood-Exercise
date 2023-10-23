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
N_BACKGROUND = 100
N_SIGNAL = 150
N_BINS = 20

if __name__=="__main__":   

    # Create Signal and Background data 
    print("\n################################################\
        \n 1. Generate datasets\n")
    bg = 10*np.random.rand(N_BACKGROUND)
    signal = np.random.normal(MU,SIGMA,N_SIGNAL)
    data = np.concatenate((bg,signal))

    # Get histogram distribution of data
    hist, bin_edges = np.histogram(data, bins=N_BINS)

    # Get bin heights and centers
    bin_heights = hist
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Data plotting
    plt.figure("Data")
    plt.hist(data,bins=N_BINS,label="Signal + Background")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("N")


    N_meshpoints = 100
    mu_arr = np.linspace(0,10,N_meshpoints)
    Nsignal_arr = np.linspace(0,N_SIGNAL*5, N_meshpoints)


    print("\n################################################\
        \n 2a. NLL 2D minimisation and plotting\n")

    # NLL plotting 2D
    NLL = nll_mesh(mu_arr,Nsignal_arr,
                   binData=[bin_heights,bin_edges,N_BACKGROUND])

    title="NLL_2D_mesh"
    plt.figure(title)
    h = plt.contourf(mu_arr, Nsignal_arr, NLL, 
                        alpha=1.0,cmap="nipy_spectral",levels=200)
    plt.xlabel(r"$\mu $")
    plt.ylabel(r"$N_{signal}$")
    clb = plt.colorbar()
    clb.set_label(r'NLL $(\mu,N_{signal})$')
    plt.tight_layout()

    # Minimising the NLL to obtain optimal parameters
    init_guess = [5,N_SIGNAL-1] #[mu, n_signal]
    results = optimize.minimize(nll,init_guess,
                                ([bin_heights,bin_edges,N_BACKGROUND]), 
                        method = "L-BFGS-B", bounds = ((None,None),(0,None)))
    print(results)
    best_params = results.x
    print("The best fit parameters are: (mu,N) = ", best_params)


    print("\n################################################\
        \n 2b. Profile NLL for each parameter individually\n")

    # NLL plotting 1D per parameter variations
    title = "NLL(mu) and NLL(sigma)"

    mu_arr = np.linspace(1,9,1000)
    Nsignal_arr = np.linspace(0,N_SIGNAL*5, 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    label = r"$N_{signal}$ = " + f"{N_SIGNAL}"
    ax1.plot(mu_arr,
        [nll([mu,N_SIGNAL],binData=[bin_heights, bin_edges,N_BACKGROUND]) for mu in mu_arr],
                        label=label)
    ax1.grid(alpha=0.1)
    ax1.legend()
    ax1.set_ylabel(r"NLL($\mu$)")
    ax1.set_xlabel(r"$\mu$")

    ax2.plot(Nsignal_arr,
             [nll([MU,nsignal],binData=[bin_heights, bin_edges,N_BACKGROUND]) for nsignal in Nsignal_arr],
                        label=rf"$\mu$ = {MU}")
    ax2.grid(alpha=0.1)
    ax2.legend()
    ax2.set_ylabel(r"NLL($N_{signal}$)")
    ax2.set_xlabel(r"$N_{signal}$")
    plt.tight_layout()
    #plt.savefig("plots/4/"+title+".pdf",
        #dpi=1200, 
        #bbox_inches="tight")

    print("\n################################################\
        \n 3. Creating multiple fluctuated datasets for comparison\n")

    pass # yet to be implemented

    print("\n################################################\
        \n 4a. Using Wilk's theorem to estimate uncertainties \n")
    
    std_guesses = [1,N_SIGNAL/5] # params = [mu, N_signal]
    uncs_mu, uncs_Nsignal = pm_error_finder(
        nll,[bin_heights,bin_edges,N_BACKGROUND],best_params,std_guesses)
    
    print(f"mu = {best_params[0]} +- {uncs_mu}")
    print(f"N_signal = {best_params[1]} +- {uncs_Nsignal}")

    std_mu = mean([abs(x) for x in uncs_mu])
    std_Nsignal = mean([abs(x) for x in uncs_Nsignal])
    print(f"std_mu = {round(std_mu,4)}, std_Nsignal = {round(std_Nsignal,4)}")

    print("\n################################################\
        \n 4. Verify Wilk's theorem with many datasets \n")
    
    N_datasets = 1000
    
    # Generate fluctuating datasets
    partial_data_gen_task = partial(data_generation_task,
            n_background=N_BACKGROUND,mu = MU,sig = SIGMA, n_signal = N_SIGNAL,
            n_datasets = N_datasets, N_bins = N_BINS)
    with Pool() as pool:
        # Parallelised dataset production - [(mu1,sig1),..] returned
        #params_arr = pool.map(partial_data_gen_task, range(N_datasets))
        output = pool.map(partial_data_gen_task, range(N_datasets))
    
    params_arr = [out[0] for out in output]
    mus_arr = [params[0] for params in params_arr]
    nsignals_arr = [params[1] for params in params_arr]
    
    data_arr = [out[1] for out in output]
    fulldata = [item for sublist in data_arr for item in sublist]
    

    # Checking what fraction of data lies in 1sigma/2sigma intervals
    muOneStdFrac = fracOfDataInRange(mus_arr,MU,std_mu)
    muTwoStdFrac = fracOfDataInRange(mus_arr,MU,2*std_mu)

    NsignalOneStdFrac = fracOfDataInRange(nsignals_arr,N_SIGNAL,std_Nsignal)
    NsignalTwoStdFrac = fracOfDataInRange(nsignals_arr,N_SIGNAL,2*std_Nsignal)

    print("\nFraction of data contained within [1 std, 2 std] for 1D = [68.27%, 95.45%]")
    print(f"MU fraction of data in [1std, 2std] = [{muOneStdFrac}, {muTwoStdFrac}]")
    print(f"N_signal fraction of data in [1std, 2std] = [{NsignalOneStdFrac}, {NsignalTwoStdFrac}] \n")


    # 1D per parameter variations plots (see if Wilk's theorem works)
    title = "Fluctuated 1000 datasets parameters distributions"
    ymax = 1
    N_bins=N_BINS
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(mus_arr, #bins= N_bins, 
             #density = True, 
             weights=np.ones(len(mus_arr)) / float(len(mus_arr)),
                    label=rf"Distribution over {N_datasets} datasets")
    ax1.vlines(MU,ymin=0,ymax=ymax,linestyles="solid",colors="red", 
              label="True value")
    ax1.vlines(MU+std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="black", 
              label=r"$1n_{signal}$")
    ax1.vlines(MU-std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="black")
    ax1.vlines(MU+2*std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="blue", 
              label=r"$2n_{signal}$")
    ax1.vlines(MU-2*std_mu,ymin=0,ymax=ymax,linestyles="dashed",colors="blue")
    ax1.grid(alpha=0.1)
    ax1.legend()
    #ax1.set_ylim(top=1)
    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel("Density")

    ax2.hist(nsignals_arr,#bins=N_bins, 
             #density = True,
             weights = np.ones(len(nsignals_arr)) / float(len(nsignals_arr)),
                    label=rf"Distribution over {N_datasets} datasets")
    ax2.vlines(N_SIGNAL,ymin=0,ymax=ymax,linestyles="solid",colors="red", 
              label="True value")
    ax2.vlines(N_SIGNAL+std_Nsignal,ymin=0,ymax=ymax,linestyles="dashed",colors="black", 
              label=r"$1n_{signal}$")
    ax2.vlines(N_SIGNAL-std_Nsignal,ymin=0,ymax=ymax,linestyles="dashed",colors="black")
    ax2.vlines(N_SIGNAL+2*std_Nsignal,ymin=0,ymax=ymax,linestyles="dashed",colors="blue", 
              label=r"$2n_{signal}$")
    ax2.vlines(N_SIGNAL-2*std_Nsignal,ymin=0,ymax=ymax,linestyles="dashed",colors="blue")

    ax2.grid(alpha=0.1)
    ax2.legend()
    #ax2.set_ylim(top=1)
    ax2.set_xlabel(r"$N_{signal}$")
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
    yAxis_arr = np.linspace(min(nsignals_arr)-std_Nsignal,max(nsignals_arr)+std_Nsignal,100)#len(nsignals_arr))

    # For NLL plot and contour drawing take dataset which has param values
    # closest to the true MU and Nsignal values
    dataBest = data_arr[find_optimal_data_distrib(mus_arr,nsignals_arr,MU,N_SIGNAL)]

    # Find histogram bins for the "best" dataset
    hist, bin_edges = np.histogram(dataBest, bins=N_BINS)
    bin_heights = hist
    binDataBest = [bin_heights, bin_edges,N_BACKGROUND]
    nllOverNdatasets = nll_mesh(xAxis_arr,yAxis_arr,binDataBest)
    
    # Rescale NLL so NLL_min = 0
    nll_min = nll([MU,N_SIGNAL],binDataBest)
    nllOverNdatasets=nllOverNdatasets-nll_min
    
    h = plt.contour(xAxis_arr, yAxis_arr,nllOverNdatasets,
                    levels=[chi2_1sigma, chi2_2sigma],
                    colors=["darkorange","forestgreen"],linewidths = 2)
                    #alpha=1.0,cmap="nipy_spectral",levels=200)
    # Label contour lines
    fmt = {}
    contourNames = [r"$1n_{signal}$", r"$2n_{signal}$"]
    for l, s in zip(h.levels, contourNames):
        fmt[l] = s
    plt.clabel(h, h.levels, inline=True, fmt=fmt, fontsize=15)

    #colourbar full nll
    h = plt.contourf(xAxis_arr, yAxis_arr,nllOverNdatasets, 
                     alpha=1.0,cmap="nipy_spectral",levels=200)
    clb = plt.colorbar()
    clb.set_label(r'NLL $(\mu, N_{signal})$')
    

    # Plot scatter points from different dataset fits
    plt.scatter(mus_arr,nsignals_arr,color="black", marker=".", 
                                        label=r'Data $(\mu, N_{signal})$')
    plt.scatter(MU,N_SIGNAL, color="red", marker="o", label=r"True $(\mu,N_{signal})$")
    
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$N_{signal}$")
    plt.legend()
    plt.tight_layout()
    


    # Fraction of data lying in contours 1std/2std
    nlls_from_datasets = [nll([mu,n_signal],binDataBest) for 
                                        (mu,n_signal) in zip(mus_arr,nsignals_arr)]
    
    frac1std = fracOfDataInStd(nlls_from_datasets,nll_min,1)
    frac2std = fracOfDataInStd(nlls_from_datasets,nll_min,2)
    print("\nFraction of data contained within [1 std, 2 std] for 2D = [68.27%, 95.45%]")
    print(f"Data gives: [{frac1std}, {frac2std}]")


    print("\n################################################\
        \n 5. Bayesian contours on 2D phase space for flat priors \n")
    
    # NLL is proportional to e^nll and since \int p(x) = 1
    # you can get the area of integration for the contour by
    # making space into grids and getting area that way...
    pass 

    print("FINISHED")
    plt.show() 