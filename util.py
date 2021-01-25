# Title: util.py
# Description: Various utilities useful for online CCA tests
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)

##############################
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

##############################

# def synthetic_data(z_dim, x_dim, y_dim, samples):

# def mediamill():    
    
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

def msg_error(M, Rxy, max_obj):

    error = (max_obj - np.trace(Rxy@M.T)/2)/max_obj

    return error

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