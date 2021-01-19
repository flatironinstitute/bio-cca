# Title: cca_algorithms.py
# Description: Implementation of online algorithms for Canonical Correlation Analysis, including Bio-CCA, Gen-Oja and MSG-CCA.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# References: D. Lipshutz, Y. Bahroun, S. Golkar, A.M. Sengupta and D.B. Chklovskii "A biologically plausible neural network for multi-channel Canonical Correlation Analysis" (2020)
#             R. Arora, T.V. Marinov, P. Mianjy and N. Sbrero "Stochastic Approximation for Canonical Correlation Analysis" (2017)
#             K. Bhattia, A. Pacchiano and N. Flammarion, P. Bartlett and M.I. Jordan "Gen-Oja: A Simple and Efficient Algorithm for Streaming Generalized Eigenvector Computation" (2018)
#             C. Pehlevan, X. Zhao, A.M. Sengupta, D.B. Chklovskii "Neurons as Canonical Correlation Analyzers" (2020)
#             S. Golkar, D. Lipshutz, Y. Bahroun, A.M. Sengupta and D.B. Chklovskii "A simple normative network approximates local non-Hebbian learning in the cortex" (2020)

##############################
# Imports
import numpy as np
from scipy.stats import ortho_group

##############################

class bio_cca:
    """
    Parameters:
    ====================
    dataset         -- Dataset
    z_dim           -- Dimension of output
    x_dim, y_dim    -- Dimensions of inputs
    M0              -- Initialization for the lateral weight matrix M, must be of size z_dim by z_dim
    Wx0, Wy0        -- Initialization for the feedforward weight matrices Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    eta0, eta_decay -- Learning rate parameters: eta = eta0/(1+eta_decay*t)
    tau             -- Ratio of Wx/Wy learning rate and M learning rate
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, z_dim, x_dim, y_dim, dataset=None, M0=None, Wx0=None, Wy0=None, eta0=None, eta_decay=None, tau=None):

        if M0 is not None:
            M = M0
        else:
            M = np.eye(z_dim)

        if Wx0 is None:
            Wx = np.random.randn(z_dim,x_dim)
            for i in range(z_dim):
                Wx[i,:] = Wx[i,:]/np.linalg.norm(Wx[i,:])
            
        if Wy0 is None:
            Wy = np.random.randn(z_dim,y_dim)
            for i in range(z_dim):
                Wy[i,:] = Wy[i,:]/np.linalg.norm(Wy[i,:])

        # optimal hyperparameters for test datasets
            
        if dataset=='synthetic':
            if eta0 is None:
                eta0 = 0.001
            if decay is None:
                decay = 0.0001
            if tau is None:
                tau = 0.1
        elif dataset=='mediamill':
            if eta0 is None:
                eta0 = 0.001
            if decay is None:
                decay = 0.0001
            if tau is None:
                tau = .03
        else:
            if eta0 is None:
                eta0 = 0.01
            if decay is None:
                decay = 0.001
            if tau is None:
                tau = 0.5

        self.t = 0
        self.eta0 = eta0
        self.decay = decay
        self.tau = tau
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.M = M
        self.Minv = np.linalg.inv(M)
        self.Wx = Wx
        self.Wy = Wy

    def fit_next(self, x, y):

        t, tau, Wx, Wy, M, Minv  = self.t, self.tau, self.Wx, self.Wy, self.M, self.Minv
        
        # project inputs
        
        a = Wx@x
        b = Wy@y
        
        # neural dynamics
        
        z = Minv@(a+b)

        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)

        Wx += 2*step*np.outer(z-a,x)
        Wy += 2*step*np.outer(z-b,y)
                
        M += (step/tau)*(np.outer(z,z)-M)
        Minv = np.linalg.inv(M)
        
        self.Wx = Wx
        self.Wy = Wy
        self.M = M
        self.Minv = Minv
        
        self.t += 1
        
        Vx = Wx.T@Minv
        Vy = Wy.T@Minv
        
        return Vx, Vy
    
class msg_cca:
    """
    Parameters:
    ====================
    dataset          -- Dataset
    z_dim            -- Dimension of output
    x_dim, y_dim     -- Dimensions of inputs
    training_samples -- Size of training set
    
    Methods:
    ========
    cov_estimation()
    projection()
    fit_next()
    """

    def __init__(self, z_dim, x_dim, y_dim, dataset=None, training_samples=100):

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
        
        t, z_dim, x_dim, y_dim, training_samples, Sx, Sy, Ux, Uy, S, U, Vh = self.t, self.z_dim, self.x_dim, self.y_dim, self.training_samples, self.Sx, self.Sy, self.Ux, self.Uy, self.S, self.U, self.Vh
        
        if t>training_samples:
            
            # set step size
        
            step = .1/np.sqrt(t-training_samples+1)

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
    
    def error(self, M, Rxy, max_obj):
        
        error = (max_obj - np.trace(Rxy@M.T)/2)/max_obj
        
        return error
        
class gen_oja:
    """
    Parameters:
    ====================
    x_dim, y_dim      -- Dimensions of inputs
    wx0, wy0          -- Initializations for wx and wy, must be of size z_dim by x_dim and z_dim by y_dim
    vx0, vy0          -- Initializations for vx and vy, must be of size z_dim by x_dim and z_dim by y_dim
    alpha             -- Maximization step size
    beta0, beta_decay -- Minimization learning parameters: beta = beta0/(1+beta_decay*t)
    
    Methods:
    ========
    fit_next()
    """
    
    def __init__(self, x_dim, y_dim, dataset=None, wx0=None, wy0=None, vx0=None, vy0=None, alpha=None, beta0=None, beta_decay=None):

        if wx0 is None:
            wx = np.random.randn(x_dim)
            wx = wx/np.linalg.norm(wx)
            
        if wy0 is None:
            wy = np.random.randn(y_dim)
            wy = wy/np.linalg.norm(wy)

        if vx0 is None:
            vx = np.random.randn(x_dim)
            vx = vx/np.linalg.norm(vx)
            
        if vy0 is None:
            vy = np.random.randn(y_dim)
            vy = vy/np.linalg.norm(vy)

        # optimal hyperparameters for test datasets
            
        if dataset=='synthetic':
            if alpha is None:
                alpha = 0.001
            if beta0 is None:
                beta0 = 1
            if beta_decay is None:
                beta_decay = 0.01
        elif dataset=='mediamill':
            if alpha is None:
                alpha = 0.001
            if beta0 is None:
                beta0 = 0.001
            if beta_decay is None:
                beta_decay = 0.0001
        else:
            if alpha is None:
                alpha = 0.01
            if beta0 is None:
                beta0 = 0.001
            if beta_decay is None:
                beta_decay = 0.001

        self.t = 0
        self.alpha = alpha
        self.beta0 = beta0
        self.beta_decay = beta_decay
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.wx = wx
        self.wy = wy
        self.vx = vx
        self.vy = vy

    def fit_next(self, x, y):

        t, wx, wy, vx, vy, alpha = self.t, self.wx, self.wy, self.vx, self.vy, self.alpha
        
        beta_t = self.beta0/(1+self.beta_decay*t)

        wx -= alpha*(np.inner(wx,x) - np.inner(vy,y))*x
        wy -= alpha*(np.inner(wy,y) - np.inner(vx,x))*y
        vx += beta_t*wx
        vy += beta_t*wy
        v = np.hstack((vx,vy))
        vx /= np.linalg.norm(v); vy /= np.linalg.norm(v)
                        
        self.wx = wx
        self.wy = wy
        self.vx = vx
        self.vy = vy
        
        self.t += 1
                
        return vx.reshape(self.x_dim,1), vy.reshape(self.y_dim,1)
    
class asymmetric:
    """
    Parameters:
    ====================
    z_dim         -- Dimension of output
    x_dim, y_dim  -- Dimensions of inputs
    Wx0, Wy0      -- Initialization for the forward weights Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    M0            -- Initialization for the asymmetric lateral weights, must be lower triangular 
    learning_rate -- Learning rate as a function of t
    
    Methods:
    ========
    fit_next()
    """
    
    def __init__(self, z_dim, x_dim, y_dim, dataset=None, M0=None, Wx0=None, Wy0=None, eta0=None, eta_decay=None):

        if M0 is not None:
            M = M0
        else:
            M = np.zeros(z_dim,z_dim)

        if Wx0 is None:
            Wx = np.random.randn(z_dim,x_dim)
            for i in range(z_dim):
                Wx[i,:] = Wx[i,:]/np.linalg.norm(Wx[i,:])
            
        if Wy0 is None:
            Wy = np.random.randn(z_dim,y_dim)
            for i in range(z_dim):
                Wy[i,:] = Wy[i,:]/np.linalg.norm(Wy[i,:])

        # optimal hyperparameters for test datasets
            
        if dataset=='synthetic':
            if eta0 is None:
                eta0 = 0.001
            if decay is None:
                decay = 0.0001
                
        elif dataset=='mediamill':
            if eta0 is None:
                eta0 = 0.001
            if decay is None:
                decay = 0.0001
                
        else:
            if eta0 is None:
                eta0 = 0.01
            if decay is None:
                decay = 0.001

        self.t = 0
        self.eta0 = eta0
        self.eta_decay = decay
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.M = M
        self.Wx = Wx
        self.Wy = Wy

    def fit_next(self, x, y):

        t, Wx, Wy, M  = self.t, self.Wx, self.Wy, self.M
        
        # project inputs
        
        a = Wx@x
        b = Wy@y
        
        # neural dynamics
        
        for i in range(z_dim):
            z[i] = a+b-M[i,:i]@z[:i]

        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)

        Wx += 2*step*np.outer(z-alpha*a,x)
        Wy += 2*step*np.outer(z-beta*b,y)
                
        step_tau = step/tau
        
        for i in range(z_dim):
            for j in :
                M[i,j] += step_tau*z[i]*z[j]
                
        alpha += (step/2)*(a**2-1)
        beta += (step/2)*(b**2-1)
        
        self.Wx = Wx
        self.Wy = Wy
        self.M = M
        
        self.t += 1
        
        Vx = Wx.T@Minv
        Vy = Wy.T@Minv
        
        return Vx, Vy

class bio_rrr:
    """
    Parameters:
    ====================
    z_dim         -- Dimension of output
    x_dim, y_dim  -- Dimensions of inputs
    Wx0, Wy0      -- Initial guesses for the forward weight matrices Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    P0
    learning_rate -- Learning rate as a function of t
    
    Methods:
    ========
    fit_next()
    """
    
    def __init__(self, z_dim, x_dim, y_dim, dataset=None, Vx0=None, Vy0=None, P0=None, eta0=None, decay=None, tau=None):

        if Vx0 is None:
            Vx = np.random.randn(x_dim,z_dim)
            for i in range(z_dim):
                Vx[i,:] = Vx[i,:]/np.linalg.norm(Vx[i,:])
            
        if Vy0 is None:
            Vy = np.random.randn(y_dim,z_dim)
            for i in range(z_dim):
                Vy[i,:] = Vy[i,:]/np.linalg.norm(Vy[i,:])
                
        if P0 is None:
            P = np.eye(z_dim)

        # optimal hyperparameters for test datasets
            
        if dataset=='synthetic':
            if eta0 is None:
                eta0 = 1.5
            if decay is None:
                decay = 0.002
            if tau is None:
                tau = 500
        elif dataset=='mediamill':
            if eta0 is None:
                eta0 = 0.001
            if decay is None:
                decay = 0.0001
            if tau is None:
                tau = .03
        else:
            if eta0 is None:
                eta0 = 0.01
            if decay is None:
                decay = 0.001
            if tau is None:
                tau = 0.5

        self.t = 0
        self.eta0 = eta0
        self.decay = decay
        self.tau = tau
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.P = P
        self.Vx = Vx
        self.Vy = Vy

    def fit_next(self, x, y):

        t, tau, Vx, Vy, P  = self.t, self.tau, self.Vx, self.Vy, self.P
        
        # project inputs
        
        a = Vx.T@x
        z = Vy.T@y
        n = P.T@z
        
        # synaptic updates
        
        step = self.eta0/(1+self.decay*t)
        step_tau = step/tau

        Vx += 2*step_tau*np.outer(x,z-a)
        Vy += 2*step*np.outer(y,a-P@n)
        P += step_tau*(np.outer(z,n)-P)
        
        self.Vx = Vx
        self.Vy = Vy
        self.P = P
        
        self.t += 1
        
        return Vx, Vy