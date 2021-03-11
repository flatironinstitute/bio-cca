# Title: biorrr.py
# Description: Implementation of Bio-RRR.
# Author: David Lipshutz (dlipshutz@flatironinstitute.org)
# References: S. Golkar, D. Lipshutz, Y. Bahroun, A.M. Sengupta and D.B. Chklovskii "A simple normative network approximates local non-Hebbian learning in the cortex" (2020)

##############################
# Imports
import numpy as np

##############################

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
            Vx = np.random.randn(x_dim,z_dim)/np.sqrt(x_dim)
            
        if Vy0 is not None:
            assert Vy0.shape==(y_dim,z_dim)
            Vy = Vy0
        else:
            Vy = np.random.randn(y_dim,z_dim)/np.sqrt(y_dim)
                
        if Q0 is not None:
            assert Q0.shape==(z_dim,z_dim)
            Q = Q0
        else:
            Q = np.eye(z_dim)

        # optimal hyperparameters for test datasets
                    
        if dataset is not None:
            if dataset=='synthetic' and z_dim==1:
                eta_x0 = 1e-3
                eta_x_decay = 1e-5
                eta_y0 = 1e-4
                eta_x_decay = 1e-5
                eta_q0 = 1e-4
                eta_q_decay = 1e-5
            elif dataset=='synthetic' and z_dim==2:
                eta_x0 = 1e-3
                eta_x_decay = 1e-4
                eta_y0 = 1e-4
                eta_x_decay = 1e-4
                eta_q0 = 1e-4
                eta_q_decay = 1e-4
            elif dataset=='synthetic' and z_dim==4:
                eta_x0 = 1e-3
                eta_x_decay = 1e-4
                eta_y0 = 1e-4
                eta_x_decay = 1e-4
                eta_q0 = 1e-4
                eta_q_decay = 1e-4
            elif dataset=='synthetic':
                eta_x0 = 1e-4
                eta_x_decay = 1e-4
                eta_y0 = 1e-4/500
                eta_x_decay = 1e-4
                eta_q0 = 1e-4/500
                eta_q_decay = 1e-4
            elif dataset=='mediamill':
                eta_x0 = 1e-2
                eta_x_decay = 1e-5
                eta_y0 = 1e-3
                eta_y_decay = 1e-5
                eta_q0 = 1e-3
                eta_q_decay = 1e-5
            elif dataset=='adaptive':
                eta_x0 = 1e-3
                eta_x_decay = 0
                eta_y0 = 1e-4
                eta_x_decay = 0
                eta_q0 = 1e-4
                eta_q_decay = 0
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
        
        Vx = np.minimum(np.maximum(Vx,-1e2),1e2)
        Vy = np.minimum(np.maximum(Vy,-1e2),1e2)
        Q = np.minimum(np.maximum(Q,-1e2),1e2)
        
        self.Vx = Vx
        self.Vy = Vy
        self.Q = Q
        
        self.t += 1
                
        return Vx, Vy