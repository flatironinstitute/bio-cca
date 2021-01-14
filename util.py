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
    
    
def correlation_matrix(Cxx, Cyy, Cxy):
    """
    Parameters:
    ====================
    Cxx, Cyy, Cxy -- Covariance matrices
    
    Output:
    ====================
    Rxy -- correlation matrix
    """
    
    sig_x, Ux = np.linalg.eig(Cxx)
    sig_y, Uy = np.linalg.eig(Cyy)

    Rxy = Ux@np.diag(1./np.sqrt(sig_x))@Ux.T@Cxy@Uy@np.diag(1./np.sqrt(sig_y))@Uy.T

    return Rxy

def error(Vx, Vy, Cxx, Cyy, Cxy, max_obj):
    """
    Parameters:
    ====================
    Vx, Vy        -- The data matrix of sources
    Cxx, Cyy, Cxy -- Covariance matrices
    
    Output:
    ====================
    err           -- The (relative) Frobenius norm error
    """
    
    z_dim = Vx.shape[1]
    
    sig, U = np.linalg.eig(Vx.T@Cxx@Vx + Vy.T@Cyy@Vy)
    
    norm_matrix = U@np.diag(1./np.sqrt(sig))@U.T
    
    Vx_normalized = Vx@norm_matrix
    Vy_normalized = Vy@norm_matrix

    err = (max_obj - np.trace(Vx_normalized.T@Cxy@Vy_normalized))/max_obj

    return err


def constraint_error(Vx, Vy, Cxx, Cyy):
    """
    Parameters:
    ====================
    Vx, Vy    -- The data matrix of sources
    Cxx, Cyy  -- Covariance matrices
    
    Output:
    ====================
    const_err -- The (relative) Frobenius norm constraint error
    """
    
    z_dim = Vx.shape[1]

    const_err = np.linalg.norm(Vx.T@Cxx@Vx+Vy.T@Cyy@Vy-np.eye(z_dim))**2/z_dim
    
    return const_err


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