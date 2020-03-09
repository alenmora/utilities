import pandas as pd
impor numpy as np
import seaborn as sns

def getCumHisto(x: ArrayLike, nperc: int) -> Tuple(np.array, np.array, int):
    """
    Takes a series of data, and divides it into nperc quantiles. Counts the number
    of elements in each quantile (which should be constant across the different quantiles) and returns such 
    number for each quantile, togeter with the quantile range and the number of quantiles used
    x: array-like object with the data to bin
    nperc: number of quantiles to group x into
    """
    size = len(x)
    percentiles=np.linspace(0,1,nperc+1)
    xper = np.quantile(x,percentiles) # Get the quantiles of the data
    histx,binsx=np.histogram(x,bins=xper) # Get the number of elements in each quantile
    
    # Though in principle the number of elements in each quantile
    # should be a constant given by len(x)/nperc, there can be very oddly
    # distributed data which cannot be separated in nperc quantiles (maybe
    # the variable can only take a discrete number of values), so this fixes it.
    #
    # First, assume that the number of elements in each bin is distributed like a 
    # Poisson with a mean of len(x)/nperc. Then, if the difference between the largest
    # bin and the smallest bin is higher than 5 standard deviations from the mean,
    # reduce the number of bins by one and try again
    
    while (np.amax(histx)-np.amin(histx) > 5*np.sqrt(size/nperc)):
        histx, bins, nperc = getCumHisto(x, nperc-1)
        
    return histx,binsx,nperc
    
def selectBinNumber(df: pandas.DataFrame, var1: str, var2: str , nperc: int = None)->Tuple(np.array, np.array, np.array):
    """
    Picks the adequate number of quantiles to divide the CDF of var1 and var2. Then returns a 2D histogram of the 2
    variables, where the bins of each axis correspond to a certain quantile of the variable in said axis. 
    df: pandas DataFrame with the data to select the binning from
    var1: name of the column representing the first variable
    var2: name of the column representing the second variable
    nperc: initial number of bins to try to divide the data into
    """
    nel = len(df.index)
    
    npercx = 20 if nperc is None else nperc 
    npercy = 20 if nperc is None else nperc
    
    x = df[var1]
    y = df[var2]
    
    _,binsx,npercx=getCumHisto(x,npercx)
    _,binsy,npercy=getCumHisto(y,npercy)
    
    nperc = max(npercx,npercy)
    
    # Reduce the number of percentiles of one of the two variables until
    # each bin has at least 25 elements, or until we can't reduce it further
    while (np.amin(np.histogram2d(x,y,bins=[binsx,binsy])[0]) < 25) and (nperc > 1):
        if nperc == npercx:
            npercx = npercx-1
            _,binsx,npercx = getCumHisto(x,npercx)
        else:
            npercy = npercy-1
            _,binsy,npercy = getCumHisto(y,npercy)
            
        nperc = max(npercx,npercy)
    
    # Get the 2D histogram of var1 and var2, with the bins in each axis given by the
    # calculated quantiles. 
    hist2d,binsx,binsy = np.histogram2d(x,y,bins=[binsx,binsy])
    
    return hist2d,binsx,binsy
    
def measureCorrelation(df: pandas.DataFrame, var1: str, var2: str, nperc: int = None) -> Tuple(np.array, np.array, np.array, float, float):
    """
    Measures the correlation between var1 and var2 in the data stored in df, by 
    performing a chi2 test on the number of elements of each bin of the joint cummulative distribution.
    Returns the array of errors for each bin, the bins for the first and the second variable, the chi2 value, and
    the pvalue for said chi2. 
    df: pandas DataFrame with the variables to measure
    var1: column name of the first variable
    var2: column name of the second variable
    nperc: initial number of quantiles to try
    """
    hist2d,binsx,binsy=selectBinNumber(df,var1,var2,nperc)
    
    #binsx = (binsx[1:]+binsx[:-1])/2
    #binsy = (binsy[1:]+binsy[:-1])/2
        
    mean = np.sum(hist2d)/((len(binsx)-1)*(len(binsy)-1))
    
    norm = (hist2d-mean)/np.sqrt(mean)
    
    stat = np.sum(norm**2)
        
    dof = (len(binsx)-2)*(len(binsy)-2)
    
    pval = 1-chi2.cdf(stat,dof)
    
    return norm,binsx,binsy,stat,pval
    
def plotCorrelation(df: pandas.DataFrame, var1: str, var2: str, ax: plt.axis = None, showcbar: bool = True, nperc: int = None) -> None:
    """
    Plots the 2D histogram representing the distribution of var1 and var2 over their quantiles respective.
    df: pandas DataFrame with the variables to measure
    var1: column name of the first variable
    var2: column name of the second variable
    ax: matplolib.pyplot axis to plot on
    showcbar: wether to show the cbar of the heatmap or not
    nperc: initial number of quantiles to try
    """
    norm,bins1,bins2,chi,pval = measureCorrelation(df,var1,var2,nperc=nperc)
    
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    sns.heatmap(norm,linewidths=.5, ax=ax, cmap='RdBu_r',vmin=-5,vmax=5,
                xticklabels=["{:.4f}".format(x) for x in (bins2[:-1]+bins2[1:])/2], 
                yticklabels=["{:2.2E}".format(y) for y in (bins1[:-1]+bins1[1:])/2],
                cbar_kws={'label': '$\sigma$'},cbar=showcbar)
    
    ax.set_xlabel(var2+' ($\mathcal{Q}$)',fontsize=15)
    ax.set_ylabel(var1+' ($\mathcal{Q}$)',fontsize=15)
    ax.set_title('$\chi^2 = {:4.3E}$ $(p = {:2.2E})$'.format(chi,pval),fontsize=15)
