import numpy as np
import random as rand
import matplotlib.pyplot as plt
from functions import *
from scipy import optimize
from multiprocessing import Pool, freeze_support
from functools import partial
from copy import deepcopy
from tqdm import tqdm


def gauss(x,mu,sig):
    #if x>0 and x<10:
    return 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mu)**2 / (2*sig**2))
    #else:
    #    return 0


def nll(params, data):
    """
    Negative Log Likehood (up to a constant)
    Parameters:
    - params: [mu,sigma]. Can be floats or arrays.
    - data: dataset of x values
    We're dealing with Gaussian signal here hence
    params are as defined below (hardcoded).
    """
    mu = params[0]
    sig = params[1]
    nll = len(data)*np.log(sig)
    for x in data:
        nll += (x-mu)**2 / (2*sig*sig)
    #else:

    return nll


def task(sig,mu_arr,data,N_data,i):
    """
    Task for parallelised NLL mesh generation
    """
    row = []
    #print(f"Mesh generation Progress: {round(float(i)/N_data*100,3)}%")
    for mu in mu_arr:
        param_vec = [mu,sig]
        NLL = nll(param_vec,data)
        row.append(NLL)
    return row

def nll_mesh(mu_arr, sig_arr, data):
    """
    Constructs a mesh of NLL values across the mu and sig parameter domain.
    Useful for plotting contours etc.
    Parallelised on the outer loop for speed.
    """
    
    N = len(sig_arr)
    args = list(zip(sig_arr,
                    [mu_arr for _ in sig_arr],
                    [data for _ in sig_arr],
                    [N for _ in sig_arr],
                    list(range(N))))
    with Pool() as pool:
        mesh = list(pool.starmap(task, args))
		
    #Non parallelised version for debugging
    # mesh = []
    # for sig in tqdm(sig_arr):
    #     row = []
    #     for mu in mu_arr:
    #         param_vec = [mu,sig]
    #         NLL = nll(param_vec,data)
    #         row.append(NLL)
    #     mesh.append(row)

    return mesh


def pm_error_finder(f,params,vec,sig,pos=None,eps=1e-12):
    """
    Bisection method to find +- errors.
    Finds points where NLL changes by 0.5
    from the value at minimum.

    Parameters:
        - f: nll function
        - params: arguments of f that don't contribute to phase space (list)
        - vec: vector of the minimum position of the nll i.e "optimal params"
        - sig: vector of standard deviation guesses where ith index 
                corresponds to ith parameter in vec
        - pos: array of indices you want results for. Default: all

    Returns an array of uncertainties for each parameter dimension.
    uncs = [[+err_1,-err_1],...,[+err_n,-err_n]]
    """
    vec = np.array(vec)
    n = len(vec)
    if pos is None:
        pos = list(range(len(vec)))
    param_uncs = []
    func = lambda vec_i: f(vec_i,params)-f(vec,params)-0.5

    # find errors for each component - dimension
    for i in range(n):
        print(f"\nFinding errors for Param No. {i+1}")
        uncs = []
        
        #+ point
        xl = vec[i]
        xr = vec[i]+3*sig[i]
        x = (xl+xr)/2
        err = xr-xl
        #print("l,r",xl,xr)
        #print("x",x)
        err_old = deepcopy(err)*1e6
        while abs(err-err_old)>eps:
            err_old = deepcopy(err)
            # Find on which side estimation lies
            vec_i = deepcopy(vec)
            vec_i[i] = x
            y = func(vec_i)
            print(f"+err, x={x}, f={y}")
            #print("y",y)
            if y>0:
                xr = deepcopy(x)
                #print("right")
            else:
                xl = deepcopy(x)
                #print("left")
            #print("l,r",xl,xr)
            # Update 
            x = (xl+xr)/2
            err = xr-xl 
        #print("+err=",x)
        # input +error
        uncs.append(deepcopy(x))
        #- point 
        xl = vec[i]-3*sig[i]
        xr = vec[i]
        x = (xl+xr)/2
        err = xr-xl
        err_old = deepcopy(err)*1e6
        while abs(err-err_old)>eps:
            err_old = deepcopy(err)
            # Find on which side estimation lies
            vec_i = deepcopy(vec)
            vec_i[i] = x 
            y = func(vec_i)
            print(f"-err, x={x}, f={y}")
            if y>0:
                xl = deepcopy(x)
            else:
                xr = deepcopy(x)
            # Update 
            x = (xl+xr)/2
            err = xr-xl 
        #print("-err=",x)
        # Input -error
        uncs.append(deepcopy(x))
        param_uncs.append(uncs)


    res_all = [] 
    for i in range(n):
        component = vec[i]
        unc_arr = np.array(param_uncs[i])
        errs = unc_arr-component
        # print("vec",vec)
        # print("comp",component)
        # print("parmunc",unc_arr)
        # print("errrs",errs)
        res_all.append(unc_arr-component)
    #print("resal",res_all)
    results = [res_all[i] for i in pos]
    return results


def data_generation_task(i, n_background,mu,sig,n_signal,n_datasets):
    
    #print(f"DataGen: {round(float(i)/n_datasets*100,3)} %")
    bg = 10*np.random.rand(n_background)
    signal = np.random.normal(mu,sig,n_signal)
    data = np.concatenate((bg,signal))

    # Minimising the NLL to obtain optimal parameters
    init_guess = [5,2]
    results = optimize.minimize(nll,init_guess,(data),method = 'Nelder-Mead')
    return results.x