# Title: util.py
# Description: Various utilities useful for online CCA tests
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)

##############################
# Imports
import numpy as np
import matplotlib.pyplot as plt

##############################

def synthetic_data(s_dim, x_dim, y_dim, samples):

    # Transformation matrices

    Tx = np.random.randn(x_dim,s_dim)
    Ty = np.random.randn(y_dim,s_dim)

    # Positive definite covariance matrices

    Psixx = np.random.randn(x_dim,x_dim); Psixx = Psixx@Psixx
    Psiyy = np.random.randn(y_dim,y_dim); Psiyy = Psiyy@Psiyy
    
    # Matrices of iid standard normals

    S = np.random.randn(s_dim,samples)
    Nx = np.random.randn(x_dim,samples)
    Ny = np.random.randn(y_dim,samples)

    # Views

    X = Tx@S + Nx
    Y = Ty@S + Ny
    
    return X, Y
    
def adaptive_data(s_dim, x_dim, y_dim, samples):
    
    epochs = len(s_dim) # s_dim is a vector input with the dimensions of the latent signal
    
    X = np.zeros((x_dim, epochs*samples))
    Y = np.zeros((y_dim, epochs*samples))
        
    for epoch in range(epochs):
        X[:,epoch*samples:(epoch+1)*samples], Y[:,epoch*samples:(epoch+1)*samples] = synthetic_data(s_dim[epoch], x_dim, y_dim, samples)
    
    return X, Y
    
def mediamill_data():
    
    # Load data
    
    data, meta = scipy.io.arff.loadarff("data/mediamill/mediamill.arff")
    
    X = np.array(data.tolist(),dtype=np.float64)[:,0:120].T
    Y = np.array(data.tolist(),dtype=np.float64)[:,120:221].T
    
    samples = X.shape[1]

    # Center data
    
    X = X - np.outer(X.mean(axis=1),np.ones(samples))
    Y = Y - np.outer(Y.mean(axis=1),np.ones(samples))

    # Condition data
    
    X = X + np.sqrt(.1)*np.random.randn(X.shape[0],X.shape[1])
    Y = Y + np.sqrt(.1)*np.random.randn(Y.shape[0],Y.shape[1])
    
    return X, Y

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

def obj_error(Vx, Vy, Cxx, Cyy, Cxy, max_obj):
    """
    Parameters:
    ====================
    Vx, Vy        -- The data matrix of sources
    Cxx, Cyy, Cxy -- Covariance matrices
    
    Output:
    ====================
    err           -- The (relative) Frobenius norm error
    """
        
    sig, U = np.linalg.eig(Vx.T@Cxx@Vx + Vy.T@Cyy@Vy)
    
    norm_matrix = U@np.diag(1./np.sqrt(sig))@U.T
    
    Vx_normalized = Vx@norm_matrix
    Vy_normalized = Vy@norm_matrix

    err = (max_obj - np.trace(Vx_normalized.T@Cxy@Vy_normalized))/max_obj

    return err

def msg_error(M, Rxy, max_obj):

    err = (max_obj - np.trace(Rxy@M.T)/2)/max_obj

    return err

def subspace_error(V, P_opt):
    
    z_dim = V.shape[1]
    
    P = V@np.linalg.inv(V.T@V+1e-5*np.eye(z_dim))@V.T
    
    err = np.linalg.norm(P-P_opt)**2
    
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
#     log_err = np.log(err)
    log_mu = log_err.mean(axis=0)
    sigma = np.std(log_err,axis=0)
    ci_lo, ci_hi = log_mu - sigma, log_mu + sigma
    plot_kwargs = plot_kwargs or {}
    ci_kwargs = ci_kwargs or {}
    plot = axis.plot(t, np.exp(log_mu), **plot_kwargs)
    fill = axis.fill_between(t, np.exp(ci_lo), np.exp(ci_hi), alpha=.1, **ci_kwargs)
    
    return plot, fill