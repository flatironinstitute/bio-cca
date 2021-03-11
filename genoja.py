# Title: genoja.py
# Description: Implementation of Gen-Oja.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# References: K. Bhattia, A. Pacchiano and N. Flammarion, P. Bartlett and M.I. Jordan "Gen-Oja: A Simple and Efficient Algorithm for Streaming Generalized Eigenvector Computation" (2018)

##############################
# Imports
import numpy as np

##############################
            
class gen_oja:
    """
    Parameters:
    ====================
    x_dim, y_dim  -- Dimensions of inputs
    dataset.      -- Input dataset
    wx0, wy0      -- Initializations for wx and wy, must be of size z_dim by x_dim and z_dim by y_dim
    vx0, vy0      -- Initializations for vx and vy, must be of size z_dim by x_dim and z_dim by y_dim
    alpha         -- Maximization step size
    beta          -- Minimization learning parameters: beta = beta0/(1+beta_decay*t)
    
    Methods:
    ========
    fit_next()
    """
    
    def __init__(self, x_dim, y_dim, dataset=None, wx0=None, wy0=None, vx0=None, vy0=None, alpha=1e-3, beta=None):

        if wx0 is not None:
            assert wx0.shape==(x_dim,)
            wx = wx0
        else:
            wx = np.random.randn(x_dim)
            wx = wx/np.linalg.norm(wx)
            
        if wy0 is not None:
            assert wy0.shape==(y_dim,)
            wy = wy0
        else:
            wy = np.random.randn(y_dim)
            wy = wy/np.linalg.norm(wy)

        if vx0 is not None:
            assert vx0.shape==(x_dim,)
        else:
            vx = np.random.randn(x_dim)
            vx = vx/np.linalg.norm(vx)
            
        if vy0 is not None:
            assert vy0.shape==(y_dim,)
        else:
            vy = np.random.randn(y_dim)
            vy = vy/np.linalg.norm(vy)

        # optimal hyperparameters for test datasets
        
        if dataset is not None:
            if dataset=='synthetic':
                alpha = 1.4e-3
                beta0 = 1
                beta_decay = 1e-2
            elif dataset=='mediamill':
                alpha = 3.9e-2
                beta0 = 1e-2
                beta_decay = 1e-4
            else:
                print('The optimal learning rates for this dataset are not stored')
                
            def beta(t):
                return beta0/(1+beta_decay*t)
        
        # default learning rate:

        if beta is None:
            def beta(t):
                return 10**-3/(1+1e-4*t)
                
        self.t = 0
        self.alpha = alpha
        self.beta = beta
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.wx = wx
        self.wy = wy
        self.vx = vx
        self.vy = vy

    def fit_next(self, x, y):

        t, wx, wy, vx, vy, alpha = self.t, self.wx, self.wy, self.vx, self.vy, self.alpha
        
        beta = self.beta(t)

        wx -= alpha*(np.inner(wx,x) - np.inner(vy,y))*x
        wy -= alpha*(np.inner(wy,y) - np.inner(vx,x))*y
        vx += beta*wx
        vy += beta*wy
        v = np.hstack((vx,vy))
        vx /= np.linalg.norm(v); vy /= np.linalg.norm(v)
                        
        self.wx = wx
        self.wy = wy
        self.vx = vx
        self.vy = vy
        
        self.t += 1
                
        return vx.reshape(self.x_dim,1), vy.reshape(self.y_dim,1)