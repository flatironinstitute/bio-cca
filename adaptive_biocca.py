# Title: adaptive_biocca.py
# Description: Implementation of Adaptive Bio-CCA with output whitening.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# References: D. Lipshutz, Y. Bahroun, S. Golkar, A.M. Sengupta and D.B. Chklovskii "A biologically plausible neural network for multi-channel Canonical Correlation Analysis" (2020)

##############################
# Imports
import numpy as np

##############################
    
class adaptive_bio_cca:
    """
    Parameters:
    ====================
    z_dim         -- Dimension of output
    x_dim, y_dim  -- Dimensions of inputs
    dataset       -- Input dataset to use the optimal learning rates that were found using a grid search
    M0            -- Initialization for the lateral weight matrix M, must be of size z_dim by z_dim
    Wx0, Wy0      -- Initialization for the feedforward weight matrices Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    eta           -- Learning rate
    tau           -- Ratio of Wx/Wy learning rate and M learning rate
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, z_dim, x_dim, y_dim, alpha=1, dataset=None, P0=None, Wx0=None, Wy0=None, eta=None, tau=0.1):
        
        # synaptic weight initializations

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
            
        if P0 is not None:
            assert P0.shape==(z_dim,z_dim)
            P = P0
        else:
            P = np.eye(z_dim)

        # optimal hyperparameters for test datasets
            
        if dataset is not None:
            if dataset=='synthetic':
                eta0 = 1e-3
                eta_decay = 1e-4
                tau = 0.01
            elif dataset=='mediamill':
                eta0 = 1e-2
                eta_decay = 1e-3
                tau = 0.5
            elif dataset=='adaptive':
                eta0 = 1e-4
                eta_decay = 0
                tau = 0.01
            else:
                print('The optimal learning rates for this dataset are not stored')
                
            def eta(t):
                return eta0/(1+eta_decay*t)
        
        # default learning rate:
        
        if eta is None:
            def eta(t):
                return 1e-4
        
        self.t = 0
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.P = P
        self.Wx = Wx
        self.Wy = Wy
        self.alpha = 1
        self.eta = eta
        self.tau = tau

    def fit_next(self, x, y):

        t, tau, alpha, Wx, Wy, P  = self.t, self.tau, self.alpha, self.Wx, self.Wy, self.P
        
        # project inputs
        
        a = Wx@x
        b = Wy@y
        
        # neural dynamics
        
        PP_inv = np.linalg.inv(P@P.T+alpha*np.eye(self.z_dim))
        
        z = PP_inv@(a+b)
        n = P.T@z

        # synaptic updates
        
        eta = self.eta(t)

        Wx += 2*eta*np.outer(z-a,x)
        Wy += 2*eta*np.outer(z-b,y)
        P += (eta/tau)*(np.outer(z,n)-P)
        
        self.Wx = Wx
        self.Wy = Wy
        self.P = P
        
        self.t += 1
        
        Vx = Wx.T@PP_inv
        Vy = Wy.T@PP_inv
        
        return Vx, Vy