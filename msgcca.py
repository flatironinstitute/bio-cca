# Title: msgcca.py
# Description: Implementation of MSG-CCA
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# References: R. Arora, T.V. Marinov, P. Mianjy and N. Sbrero "Stochastic Approximation for Canonical Correlation Analysis" (2017)

##############################
# Imports
import numpy as np
from scipy.stats import ortho_group

##############################

class msg_cca:
    """
    Parameters:
    ====================
    z_dim            -- Dimension of output
    x_dim, y_dim     -- Dimensions of inputs
    training_samples -- Size of training set
    
    Methods:
    ========
    cov_estimation()
    projection()
    fit_next()
    """

    def __init__(self, z_dim, x_dim, y_dim, dataset=None, training_samples=1000):

        U = ortho_group.rvs(dim=x_dim)
        Vh = ortho_group.rvs(dim=y_dim)
        S = np.ones(z_dim)
        
        Smat = np.zeros((x_dim,y_dim))
        Smat[:len(S),:len(S)] = np.diag(S)

        M = U@Smat@Vh

        self.U = U
        self.Vh = Vh
        self.S = S
        self.M = M
        self.t = 0
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.training_samples = training_samples

    def cov_estimation(self, X, Y):
        
        training_samples = self.training_samples
        
        Cxx = X[:,:training_samples]@X[:,:training_samples].T/training_samples
        Cyy = Y[:,:training_samples]@Y[:,:training_samples].T/training_samples
        
        Sx, Ux = np.linalg.eig(Cxx)
        Sy, Uy = np.linalg.eig(Cyy)
                
        self.Sx = Sx
        self.Sy = Sy
        self.Ux = Ux
        self.Uy = Uy
        
    def projection(self, S):
        
        z_dim = self.z_dim
        
        capS = np.minimum(np.maximum(S,0),1)

        if capS.sum()<=z_dim:
            return capS

        l = len(capS)

        for i in range(l):
            for j in range(i,l):

                SS = capS[::-1]

                if i>0:
                    SS[0:i] = 0
                if j<l:
                    SS[j:l] = 1

                shift = (z_dim-SS.sum())/(j-i+1)
                SS[j:l] = SS[j:l] + shift

                if SS[i]>=0 and (i==0 or SS[i-1]+shift<=0) and SS[j]<=1 and (j==l-1 or SS[j+1]+shift>=1):
                    S = SS[::-1]
                    return S

        print('error')
    
    def fit_next(self, x, y):
        
        t, z_dim, x_dim, y_dim, training_samples, Sx, Sy, Ux, Uy, S, U, Vh, M = self.t, self.z_dim, self.x_dim, self.y_dim, self.training_samples, self.Sx, self.Sy, self.Ux, self.Uy, self.S, self.U, self.Vh, self.M
        
        if t>training_samples:
            
            # set step size
        
            step = .1/np.sqrt(t-training_samples)

            # update covariance svd

            Sx, Ux_update = np.linalg.eig(t*np.diag(Sx) + np.outer(Ux.T@x,Ux.T@x))
            Sy, Uy_update = np.linalg.eig(t*np.diag(Sy) + np.outer(Uy.T@y,Uy.T@y))

            Sx = Sx/(t+1)
            Sy = Sy/(t+1)

            Ux = Ux@Ux_update
            Uy = Uy@Uy_update

            # whiten x and y

            wx = Ux@np.diag(np.power(Sx,-1/2))@Ux.T@x
            wy = Uy@np.diag(np.power(Sy,-1/2))@Uy.T@y

            # update M via SVD

            Smat = np.zeros((x_dim,y_dim))
            Smat[:len(S),:len(S)] = np.diag(S)

            U_update, S, Vh_update = np.linalg.svd(Smat + step*np.outer(U.T@wx,Vh@wy), full_matrices=True)

            U = U@U_update
            Vh = Vh_update@Vh

            S = self.projection(S)
            Smat = np.zeros((x_dim,y_dim))
            Smat[:len(S),:len(S)] = np.diag(S)

            M = U@Smat@Vh
            
            self.Sx = Sx
            self.Sy = Sy
            self.Ux = Ux
            self.Uy = Uy
            self.S = S
            self.U = U
            self.Vh = Vh
            self.M = M

        self.t += 1
        
        return M