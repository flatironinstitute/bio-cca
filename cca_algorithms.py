# Title: cca_algorithms.py
# Description: Implementation of online algorithms for Canonical Correlation Analysis, including Bio-CCA, Gen-Oja and MSG-CCA.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# References: D. Lipshutz, Y. Bahroun, S. Golkar, A.M. Sengupta and D.B. Chklovskii "A biologically plausible neural network for multi-channel Canonical Correlation Analysis" (2020)
#             K. Bhattia, A. Pacchiano and N. Flammarion, P. Bartlett and M.I. Jordan "Gen-Oja: A Simple and Efficient Algorithm for Streaming Generalized Eigenvector Computation" (2018)
#             R. Arora, T.V. Marinov, P. Mianjy and N. Sbrero "Stochastic Approximation for Canonical Correlation Analysis" (2017)

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
    M0            -- Initial guess for the lateral weight matrix M, must be of size z_dim by z_dim
    Wx0, Wy0      -- Initial guesses for the forward weight matrices Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    learning_rate -- Learning rate as a function of t
    tau           -- Learning rate factor for M (multiplier of the W learning rate)
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, z_dim, x_dim, y_dim, dataset=None, M0=None, Wx0=None, Wy0=None, eta0=None, decay=None, tau=None):

        if M0 is not None:
            Minv = np.lingalg.inv(M)
        else:
            Minv = np.eye(z_dim)

        if Wx0 is None:
            Wx = np.random.randn(z_dim,x_dim)
            for i in range(s_dim):
                Wx[i,:] = Wx[i,:]/np.linalg.norm(Wx[i,:])
            
        if Wy0 is None:
            Wy = np.random.randn(z_dim,y_dim)
            for i in range(s_dim):
                Wy[i,:] = Wy[i,:]/np.linalg.norm(Wy[i,:])

        # optimal hyperparameters for test datasets
            
        if dataset=='synthetic':
            if eta0 is None:
                eta0 = 0.1
            if decay is None:
                decay = 0.01
            if tau is None:
                tau = 0.8
        elif dataset=='mediamill':
            if eta0 is None:
                eta0 = 0.001
            if decay is None:
                decay = 0.0001
            if tau is None:
                tau = .03
        else:
            if eta0 is None:
                eta0 = 0.1
            if decay is None:
                decay = 0.001
            if tau is None:
                tau = 0.5

        self.t = 0
        self.eta0 = eta0
        self.decay = decay
        self.tau = tau
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.M = M
        self.Wx = Wx
        self.Wy = Wy

    def fit_next(self, x, y):

        t, tau, Wx, Wy, Minv  = self.t, self.tau, self.x_bar, self.Wx, self.Wy, self.Minv
        
        # project inputs
        
        a = Wx@x
        b = Wy@y
        
        # neural dynamics

        z = Minv@(a+b)

        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)

        Wx += 2*step*np.outer(z-a,x)
        Wy += 2*step*np.outer(z-b,y)
        
        Minv_z = Minv@z
        step_tau = step/tau
        Minv -= (step_tau/(1-step_tau+step_tau*z.T@Minv_z))@np.outer(Minv_z,Minv_z)
        Minv /= 1-step
        
        self.Wx = Wx
        self.Wy = Wy
        self.Minv = Minv
        
        self.t += 1
        
        return z