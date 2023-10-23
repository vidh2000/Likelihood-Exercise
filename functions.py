import numpy as np
import random as rand
import matplotlib.pyplot as plt
from functions import *
from scipy import optimize
from multiprocessing import Pool, freeze_support
from functools import partial
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import chi2
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def gauss(x,mu,sig):
    #if x>0 and x<10:
    return 1/(np.sqrt(2*np.pi)*sig) * np.exp(-(x-mu)**2 / (2*sig**2))
    #else:
    #    return 0


def nll(params, data):
    """
    2 * Negative Log Likehood (up to a constant)
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
    return 2*nll


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
    Finds points where 2*NLL changes by amount appropriate to 1std deviation
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

  
    ChiDiffEqNSigma = 1 #for looking at only 1D phase space errors

    vec = np.array(vec)
    n = len(vec)
    if pos is None:
        pos = list(range(len(vec)))
    param_uncs = []
    func = lambda vec_i: f(vec_i,params)-f(vec,params) - ChiDiffEqNSigma

    # find errors for each component - dimension
    for i in range(n):
        #print(f"\nFinding errors for Param No. {i+1}")
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
            #print(f"+err, x={x}, f={y}")
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
            #print(f"-err, x={x}, f={y}")
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
    return results.x, data


def fracOfDataInRange(data, center, maxDist):
    """
    Finds fraction of data that lies within some 
    specified distance from a specific value.
    
    Works for 2D and 1D phase space, depending on input vectors of 
    data and center.
    """
    N = float(len(data))
    count = 0.0
    if type(center) == float:
        for x in data:
            dist = abs(x-center)
            if dist < maxDist:
                count+=1.0
    else:
        data = np.array(data)
        center = np.array(center)
        for x in data:
            dist = np.linalg.norm(x-center)
            if dist < maxDist:
                    count+=1.0
    frac = count/N
    return frac

def fracOfDataInStd(nllOverNDatasets, NLL_min, NbStd):
    """
    Finds fraction of datasets that can be parametrised with
    values that lie within specified number of 
    standard deviations from the parameters that minimise NLL.
    NLL values for all datasets are stored in nllOverNDatasets.
    Compares values of -2NLL since -2NLL=chi^2
    Intended for 2D parametrised data.
    NbStd only works for = 1 and 2.
    """
    
    
    N=float(len(nllOverNDatasets)*len(nllOverNDatasets[0]))
    count=0.0
    # 2D parameter scatter plots (see if Wilk's theorem works)
    df = 2  # For a two-parameter problem
    # Calculate χ² values for 1 and 2 sigma confidence intervals
    chi2_1sigma = chi2.ppf(0.6827, df)
    chi2_2sigma = chi2.ppf(0.9545, df)
    
    if NbStd==1:
        val = chi2_1sigma
    elif NbStd==2:
        val = chi2_2sigma
    else:
        raise ValueError("NbStd takes values 1 or 2.")

    for row in nllOverNDatasets:
        for nll in row:
            nllDif = nll-NLL_min
            if nllDif <= val:
                print(nllDif)
                count +=1
    frac = count/N
    return frac
    



def find_optimal_data_distrib(muArr,sigmaArr,trueMu,trueSigma):
    """
    Takes arrays of mu and sigma values, each i corresponding to the same
    data distribution, and compares them with true values.
    The distribution - index which has closest mu and sigma to true values
    is returned as the output of the function.
    """
    iBest = 0
    minDist = 1e8
    trueVec = np.array([trueMu,trueSigma])
    for i,(mu,sigma) in enumerate(zip(muArr,sigmaArr)):
        paramVec = np.array([mu,sigma])
        dist = np.linalg.norm(paramVec-trueVec)
        if dist<minDist:
            #print(f"i={i}, newMinDist={dist}")
            iBest = i
            minDist = dist

    print("Dataset with distribution closest to true values:")
    print(f"Index={iBest} with mu={muArr[iBest]}, sigma={sigmaArr[iBest]}")
    return iBest




def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    FROM ONLINE
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
