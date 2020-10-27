# Title: util.py
# Description: Various utilities useful for online CCA tests
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)

##############################
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

##############################

def synthetic_data(s_dim, x_dim, samples):
    """
    Parameters:
    ====================
    s_dim   -- The dimension of sources
    x_dim   -- The dimension of mixtures
    samples -- The number of samples

    Output:
    ====================
    S       -- The source data matrix
    X       -- The mixture data matrix
    """
    
    print(f'Generating {s_dim}-dimensional sparse uniform source data...')

    # Generate sparse random samples

    U = np.random.uniform(0,np.sqrt(48/5),(s_dim,samples)) # independent non-negative uniform source RVs with variance 1
    B = np.random.binomial(1, .5, (s_dim,samples)) # binomial RVs to sparsify the source
    S = U*B # sources

    print(f'Generating {x_dim}-dimensional mixtures...')
    
    A = np.random.randn(x_dim,s_dim) # random mixing matrix

    # Generate mixtures
    
    X = A@S
    
    np.save(f'datasets/{s_dim}-dim_synthetic/sources.npy', S)
    np.save(f'datasets/{s_dim}-dim_synthetic/mixtures.npy', X)
    
def mediamill(s_dim, x_dim):
    """
    Parameters:
    ====================
    s_dim   -- The dimension of sources
    x_dim   -- The dimension of mixtures

    Output:
    ====================
    S       -- The source data matrix
    X       -- The mixture data matrix
    """
    
    print(f'Generating {s_dim}-dimensional image source data...')

    # Generate image sources
    
    image_numbers = [5, 6, 11]; winsize=252 # 3 pre-specified image patches
    posx = [220, 250, 200]
    posy = [1, 1, 1]

    S = np.zeros((s_dim, winsize**2))

    plt.figure(figsize=(15,10))

    for i in range(s_dim):
        image = imageio.imread(f"images/{image_numbers[i]}.tiff")
        window = image[posy[i]:posy[i] + winsize, posx[i]:posx[i] + winsize]
        plt.subplot(s_dim, 1, i+1)
        plt.imshow(window, cmap="gray")
        window = window.reshape(1,-1)
        window = window - window.min(axis=1)
        window_var = np.cov(window)
        window = window*(window_var**-.5)
        S[i,:] = window

    plt.show()

    S = np.array(S)

    print(f'Generating {x_dim}-dimensional mixtures...')
    
    A = np.random.randn(x_dim,s_dim) # random mixing matrix

    # Generate mixtures
    
    X = A@S
    
    np.save(f'datasets/image/sources.npy', S)
    np.save(f'datasets/image/mixtures.npy', X)

def permutation_error(S, Y):
    """
    Parameters:
    ====================
    S   -- The data matrix of sources
    Y   -- The data matrix of recovered sources
    
    Output:
    ====================
    err -- the (relative) Frobenius norm error
    """
    
    assert S.shape==Y.shape, "The shape of the sources S must equal the shape of the recovered sources Y"

    s_dim = S.shape[0]
    iters = S.shape[1]
    
    err = np.zeros(iters)
    
    # Determine the optimal permutation at the final time point.
    # We solve the linear assignment problem using the linear_sum_assignment package
    
    # Calculate cost matrix:
    
    C = np.zeros((s_dim,s_dim))
    
    for i in range(s_dim):
        for j in range(s_dim):
            C[i,j] = ((S[i] - Y[j])**2).sum()
    
    # Find the optimal assignment for the cost matrix C
    
    row_ind, col_ind = linear_sum_assignment(C)
        
    for t in range(1,iters):

        diff_t = (S[row_ind[:],t] - Y[col_ind[:],t])**2
        error_t = diff_t.sum()/s_dim
        err[t] = err[t-1] + (error_t - err[t-1])/t
    
    return err

def add_fill_lines(axis, t, err, plot_kwargs=None, ci_kwargs=None):
    """
    Parameters:
    ====================
    axis        -- Axis variable
    t           -- Array of time points
    err         -- The data matrix of errors over multiple trials
    plot_kwargs -- Arguments for axis.plot()
    ci_kwargs   -- Arguments for axis.fill_between()
    
    Output:
    ====================
    plot        -- Function axis.plot()
    fill        -- Function axis.fill_between() with standard deviation computed on a log scale
    """
        
    log_err = np.log(err+10**-5) # add 10**-5 to ensure the logarithm is well defined
    log_mu = log_err.mean(axis=0)
    sigma = np.std(log_err,axis=0)
    ci_lo, ci_hi = log_mu - sigma, log_mu + sigma
    plot_kwargs = plot_kwargs or {}
    ci_kwargs = ci_kwargs or {}
    plot = axis.plot(t, np.exp(log_mu), **plot_kwargs)
    fill = axis.fill_between(t, np.exp(ci_lo), np.exp(ci_hi), alpha=.1, **ci_kwargs)
    
    return plot, fill