# Title: bio-cca.py
# Description: Implementation of Bio-CCA
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# Reference: D. Lipshutz, Y. Bahroun, S. Golkar, A.M. Sengupta and D.B. Chklovskii "A biologically plausible neural network for multi-channel Canonical Correlation Analysis" (2020)

##############################
# Imports
import numpy as np

##############################

class bio_cca:
    """
    Parameters:
    ====================
    z_dim         -- Dimension of output
    x_dim, y_dim  -- Dimensions of inputs
    dataset       -- Input dataset to use the optimal learning rates that were found using a grid search
    M0            -- Initialization for the lateral weight matrix M, must be of size z_dim by z_dim
    Wx0, Wy0      -- Initialization for the feedforward weight matrices Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    eta           -- Learning rate
    tau           -- Ratio of Wx, Wy learning rate and M learning rate
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, z_dim, x_dim, y_dim, dataset_name=None, M0=None, Wx0=None, Wy0=None, eta=None, tau=0.1):
        
        # synaptic weight initializations

        if M0 is not None:
            assert M0.shape==(z_dim,z_dim)
            M = M0
        else:
            M = np.eye(z_dim)

        if Wx0 is not None:
            assert Wx0.shape==(z_dim,x_dim)
            Wx = Wx0
        else:
            Wx = np.random.randn(z_dim,x_dim)/np.sqrt(x_dim)
            
        if Wy0 is not None:
            assert Wy0.shape==(z_dim,y_dim)
            Wy = Wy0
        else:
            Wy = np.random.randn(z_dim,y_dim)/np.sqrt(y_dim)

        # optimal hyperparameters for test datasets
            
        if dataset_name is not None:
            if dataset_name=='synthetic':
                def eta(t): return 1e-3/(1+1e-4*t)
                tau = 0.1
            elif dataset_name=='mediamill':
                def eta(t): return 1e-2/(1+1e-4*t)
                tau = 0.1
            else:
                print('The optimal learning rates for this dataset are not stored')
        
        # default learning rate:
        
        if eta is None:
            def eta(t): return 1e-3/(1+1e-4*t)
        
        self.t = 0
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.M = M
        self.Minv = np.linalg.inv(M)
        self.Wx = Wx
        self.Wy = Wy
        self.eta = eta
        self.tau = tau

    def fit_next(self, x, y):

        t, tau, Wx, Wy, M, Minv  = self.t, self.tau, self.Wx, self.Wy, self.M, self.Minv
        
        # project inputs
        
        a = Wx@x
        b = Wy@y
        
        # neural dynamics
        
        z = Minv@(a+b)

        # synaptic updates
        
        eta = self.eta(t)

        Wx += 2*eta*np.outer(z-a,x)
        Wy += 2*eta*np.outer(z-b,y)
                
        M += (eta/tau)*(np.outer(z,z)-M)
        Minv = np.linalg.inv(M)
        
        self.Wx = Wx
        self.Wy = Wy
        self.M = M
        self.Minv = Minv
        
        self.t += 1
        
        Vx = Wx.T@Minv
        Vy = Wy.T@Minv
        
        return Vx, Vy