# Title: cca_algorithms.py
# Description: Implementation of online algorithms for Canonical Correlation Analysis, including Bio-CCA, MSG-CCA, Gen-Oja, Asymmetric CCA, Bio-RRR.
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

    def __init__(self, z_dim, x_dim, y_dim, dataset=None, M0=None, Wx0=None, Wy0=None, eta=None, tau=0.1):
        
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
            Wx = np.random.randn(z_dim,x_dim)
            for i in range(z_dim):
                Wx[i,:] = Wx[i,:]/np.linalg.norm(Wx[i,:])
            
        if Wy0 is not None:
            assert Wy0.shape==(z_dim,y_dim)
            Wy = Wy0
        else:
            Wy = np.random.randn(z_dim,y_dim)
            for i in range(z_dim):
                Wy[i,:] = Wy[i,:]/np.linalg.norm(Wy[i,:])

        # optimal hyperparameters for test datasets
            
        if dataset is not None:
            if dataset=='synthetic':
                eta0 = 1e-3
                eta_decay = 1e-4
                tau = 0.1
            elif dataset=='mediamill':
                eta0 = 1e-2
                eta_decay = 1e-4
                tau = 0.1
            else:
                print('The optimal learning rates for this dataset are not stored')
                
            def eta(t):
                return eta0/(1+eta_decay*t)
        
        # default learning rate:
        
        if eta is None:
            def eta(t):
                return 1e-3/(1+1e-4*t)
        
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

class asy_cca:
    """
    Parameters:
    ====================
    z_dim         -- Dimension of output
    x_dim, y_dim  -- Dimensions of inputs
    dataset       -- Input dataset to use the optimal learning rates that were found using a grid search
    Wx0, Wy0      -- Initialization for the forward weights Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    M0            -- Initialization for the asymmetric lateral weights, must be lower triangular
    alpha0, beta0 -- Initialization for alpha, beta
    learning_rate -- Learning rate as a function of t
    
    Methods:
    ========
    fit_next()
    """

    def __init__(self, z_dim, x_dim, y_dim, dataset=None, M0=None, Vx0=None, Vy0=None, eta=None):

        if M0 is not None:
            assert M.shape==(z_dim,z_dim)
            
            # verify M0 is lower triangular
            
            for i in range(z_dim):
                for j in range(i+1):
                    assert M[i,j]==0
                    
            M = M0
        else:
            M = np.random.randn(z_dim,z_dim)
            M = np.tril(M+M.T,-1)

        if Vx0 is not None:
            assert Vx0.shape==(x_dim,z_dim)
            Vx = Vx0
        else:
            Vx = np.random.randn(x_dim,z_dim)
            for i in range(z_dim):
                Vx[:,i] = Vx[:,i]/np.linalg.norm(Vx[:,i])
        
        if Vy0 is not None:
            assert Vy0.shape==(y_dim,z_dim)
            Vy = Vy0
        else:
            Vy = np.random.randn(y_dim,z_dim)
            for i in range(z_dim):
                Vy[:,i] = Vy[:,i]/np.linalg.norm(Vy[:,i])
                
        if dataset=='synthetic':
            def eta(t):
                return .0001
        elif dataset=='mediamill':
            def eta(t):
                return .02*max(1-5e-4*t,.1)
        else:
            if eta is None:
                def eta(t):
                    return .001

        self.t = 0
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.M = M
        self.Vx = Vx
        self.Vy = Vy
        self.Lambda = np.diag(np.random.randn(z_dim))
        self.Gamma = np.diag(np.random.randn(z_dim))
        self.eta = eta

    def fit_next(self, x, y):
        
        t, z_dim, Vx, Vy, M, eta, Lambda, Gamma = self.t, self.z_dim, self.Vx, self.Vy, self.M, self.eta, self.Lambda, self.Gamma
        
        # project inputs
        
        a = Vx.T@x
        b = Vy.T@y
        
        # neural dynamics
        
        z = np.linalg.inv(np.eye(z_dim) + M)@(a+b)
        
        # synaptic weight updates
        
        step = eta(t)

        Vx += step*np.outer(x,z-Lambda@a)
        Vy += step*np.outer(y,z-Gamma@b)
        Lambda += (step/2)*(a@a.T-1)*np.eye(z_dim)
        Gamma += (step/2)*(b@b.T-1)*np.eye(z_dim)
        M += step*z@z.T
        M = np.tril(M,-1)
                
        self.Vx = Vx
        self.Vy = Vy
        self.M = M
        
        self.t += 1
        
        return Vx, Vy

class bio_rrr:
    """
    Parameters:
    ====================
    z_dim         -- Dimension of output
    x_dim, y_dim  -- Dimensions of inputs
    dataset       -- Input dataset
    Wx0, Wy0      -- Initialization for the forward weigts Wx and Wy, must be of size z_dim by x_dim and z_dim by y_dim
    P0            -- Initialization for the pyramidal neuron-to-interneuron weights
    learning_rate -- Learning rate as a function of t
    
    Methods:
    ========
    fit_next()
    """
    
    def __init__(self, z_dim, x_dim, y_dim, dataset=None, Vx0=None, Vy0=None, Q0=None, eta_x=None, eta_y=None, eta_q=None):

        if Vx0 is not None:
            assert Vx0.shape==(x_dim,z_dim)
            Vx = Vx0
        else:
            Vx = np.random.randn(x_dim,z_dim)
            for i in range(z_dim):
                Vx[i,:] = Vx[i,:]/np.linalg.norm(Vx[i,:])
            
        if Vy0 is not None:
            assert Vy0.shape==(y_dim,z_dim)
            Vy = Vy0
        else:
            Vy = np.random.randn(y_dim,z_dim)
            for i in range(z_dim):
                Vy[i,:] = Vy[i,:]/np.linalg.norm(Vy[i,:])
                
        if Q0 is not None:
            assert Q0.shape==(z_dim,z_dim)
            Q = Q0
        else:
            Q = np.eye(z_dim)

        # optimal hyperparameters for test datasets
                    
        if dataset is not None:
            if dataset=='synthetic':
                eta_x0 = 0.001
                eta_x_decay = 0.0001
                eta_y0 = 0.001
                eta_y_decay = 0.0001
                eta_q0 = 0.001
                eta_q_decay = 0.0001
            elif dataset=='mediamill' and z_dim==1:
                eta_x0 = .03
                eta_x_decay = 1e-2
                eta_y0 = .06e-2
                eta_y_decay = 1e-2
                eta_q0 = .06e-2
                eta_q_decay = 1e-2
            elif dataset=='mediamill' and z_dim==2:
                eta_x0 = 2.5
                eta_x_decay = 1e-2
                eta_y0 = 6e-2
                eta_y_decay = 1e-2
                eta_q0 = 6e-2
                eta_q_decay = 1e-2
            elif dataset=='mediamill' and z_dim==4:
                eta_x0 = 1.2
                eta_x_decay = 1e-3
                eta_y0 = 2.4e-2
                eta_y_decay = 1e-3
                eta_q0 = 2.4e-2
                eta_q_decay = 1e-3
            else:
                print('The optimal learning rates for this dataset are not stored')
                
            def eta_x(t):
                return eta_x0/(1+eta_x_decay*t)
            
            def eta_y(t):
                return eta_y0/(1+eta_x_decay*t)
                
            def eta_q(t):
                return eta_q0/(1+eta_q_decay*t)

        self.t = 0
        self.eta_x = eta_x
        self.eta_y = eta_y
        self.eta_q = eta_q
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.Q = Q
        self.Vx = Vx
        self.Vy = Vy

    def fit_next(self, x, y):

        t, Vx, Vy, Q  = self.t, self.Vx, self.Vy, self.Q
        
        # project inputs
        
        z = Vx.T@x
        a = Vy.T@y
        n = Q.T@z
        
        # synaptic updates
        
        Vx += 2*self.eta_x(t)*np.outer(x,a-Q@n)
        Vy += 2*self.eta_y(t)*np.outer(y,z-a)
        Q += self.eta_q(t)*(np.outer(z,n)-Q)
        
        self.Vx = Vx
        self.Vy = Vy
        self.Q = Q
        
        self.t += 1
                
        return Vx, Vy