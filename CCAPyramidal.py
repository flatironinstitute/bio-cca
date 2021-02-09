import numpy as np
import scipy.linalg


def get_nprandom(seed):
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    elif seed is None:
        return np.random
    else:
        raise ValueError


def hierarchy_mask(M):
    return np.tril(M, -1)


def symmetric_mask(M):
    return M * (1 - np.eye(M.shape[0]))


class CCAPyramidal:
    def __init__(self, X: np.ndarray, Y: np.ndarray, k, eta, steps, alpha, mode, seed=None):
        """Initialize CCA neural network.

        :param X: m by T dataset.
        :param Y: n by T dataset.
        :param k: number of canonical components to extract, which is also the number of neurons.
        :param eta: dictionary of learning rates, should has entries "W", "V", "M", "Lambda", "Gamma".
        :param alpha: learning rate decay parameter in 1 / (alpha * t + 1)
        :param mode: "hierarchy" or "symmetric".
        :param seed: seed for generating random initial condition.
        """
        self.X = X
        self.Y = Y
        self.k = k
        self.eta = eta
        self.steps = steps
        self.alpha = alpha
        self.obj = np.zeros([self.steps])
        self.err = np.zeros([self.steps])
        self.M_mask = {"hierarchy": hierarchy_mask, "symmetric": symmetric_mask}[mode]

        self.m = X.shape[0]
        self.n = Y.shape[0]
        self.T = X.shape[1]
        assert Y.shape[1] == self.T, "X.shape={}, Y.shape={}".format(X.shape, Y.shape)

        self.C_XX = self.X @ self.X.T / self.T
        self.C_YY = self.Y @ self.Y.T / self.T
        self.C_XY = self.X @ self.Y.T / self.T
        self.C_YX = self.C_XY.T

        self.W_opt, self.V_opt, self.obj_opt = self.opt_WVobj()
        # print("obj_opt: {}".format(self.obj_opt))

        nprandom = get_nprandom(seed)
        self.W = nprandom.randn(self.m, self.k)
        self.V = nprandom.randn(self.n, self.k)
        self.M = nprandom.rand(self.k, self.k)
        self.M = self.M_mask(self.M + self.M.T)
        self.Lambda = np.diag(nprandom.randn(self.k))
        self.Gamma = np.diag(nprandom.randn(self.k))

    def reset(self, seed=None):
        """Reinitialize.

        :param seed: seed for generating random initial condition.
        """
        nprandom = get_nprandom(seed)
        self.W = nprandom.randn(self.m, self.k)
        self.V = nprandom.randn(self.n, self.k)
        self.M = nprandom.rand(self.k, self.k)
        self.M = self.M_mask(self.M + self.M.T)
        self.Lambda = np.diag(nprandom.randn(self.k))
        self.Gamma = np.diag(nprandom.randn(self.k))

    def opt_WVobj(self):
        """Compute optimal W, V and objective.

        :return optimal W, V and objective.
        """
        C_XX_invsqrt = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_XX))
        C_YY_invsqrt = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_YY))
        K = C_XX_invsqrt @ self.C_XY @ C_YY_invsqrt
        W_white, self.Sigma, Vh_white = np.linalg.svd(K)
        # print("Sigma: {}".format(self.Sigma))
        W_opt = C_XX_invsqrt @ W_white[:, 0: self.k]
        V_opt = C_YY_invsqrt @ Vh_white.T[:, 0: self.k]

        cov_WX_VY = np.trace(W_opt.T @ self.C_XY @ V_opt)
        var_WX = np.trace(W_opt.T @ self.C_XX @ W_opt)
        var_VY = np.trace(V_opt.T @ self.C_YY @ V_opt)
        obj_opt = cov_WX_VY / np.sqrt(var_WX * var_VY)

        return W_opt, V_opt, obj_opt

    def set_opt(self):
        """Set the network to optimal."""
        self.W = self.W_opt.copy()
        self.V = self.V_opt.copy()
        self.M = np.zeros([self.k, self.k])
        self.Lambda = self.W.T @ (self.C_XX @ self.W + self.C_XY @ self.V) * np.eye(self.k)
        self.Gamma = self.V.T @ (self.C_YX @ self.W + self.C_YY @ self.V) * np.eye(self.k)

        I_X = self.W.T @ self.X
        I_Y = self.V.T @ self.Y
        Z = I_X + I_Y
        assert np.allclose(self.X @ Z.T, self.X @ I_X.T @ self.Lambda)
        assert np.allclose(self.Y @ Z.T, self.Y @ I_Y.T @ self.Gamma)

    def perturb(self, noise, seed=None):
        """Perturb the network parameters.

        :param noise: noise level.
        :param seed: seed to generate random noise.
        """
        nprandom = get_nprandom(seed)
        self.W += nprandom.randn(self.m, self.k) * noise
        self.V += nprandom.randn(self.n, self.k) * noise
        self.Lambda += np.diag(nprandom.randn(self.k)) * noise
        self.Gamma += np.diag(nprandom.randn(self.k)) * noise

        M_noise = nprandom.randn(self.k, self.k) * noise / 2
        M_noise = self.M_mask(M_noise + M_noise.T)
        self.M += M_noise

    def objective(self):
        """Compute the objective function.

        :return: objective.
        """
        cov_WX_VY = np.trace(self.W.T @ self.C_XY @ self.V)
        var_WX = np.trace(self.W.T @ self.C_XX @ self.W)
        var_VY = np.trace(self.V.T @ self.C_YY @ self.V)
        return cov_WX_VY / np.sqrt(var_WX * var_VY)

    def error_angle(self):
        """Compute the angle between current W, V and W_opt, V_opt under diag(C_XX, C_YY) norm.

        :return: error angle.
        """
        inner_WV_WVopt = np.trace(abs(self.W.T @ self.C_XX @ self.W_opt) + abs(self.V.T @ self.C_YY @ self.V_opt))
        inner_WV_WV = np.trace(self.W.T @ self.C_XX @ self.W + self.V.T @ self.C_YY @ self.V)
        inner_WVopt_WVopt = np.trace(self.W_opt.T @ self.C_XX @ self.W_opt + self.V_opt.T @ self.C_YY @ self.V_opt)
        cos_theta = inner_WV_WVopt / np.sqrt(inner_WV_WV * inner_WVopt_WVopt)
        theta = np.arccos(abs(cos_theta))
        return theta * 180 / np.pi

    def if_converged(self):
        """Calculate absolute and relative gradient to determine whether the algorithm has converged."""
        IMinv = np.linalg.inv(np.eye(self.k) + self.M)
        dW = (self.C_XX @ self.W + self.C_XY @ self.V) @ IMinv.T - self.C_XX @ self.W @ self.Lambda
        dV = (self.C_YX @ self.W + self.C_YY @ self.V) @ IMinv.T - self.C_YY @ self.V @ self.Gamma
        dLambda = (self.W.T @ self.C_XX @ self.W - 1) * np.eye(self.k)
        dGamma = (self.V.T @ self.C_YY @ self.V - 1) * np.eye(self.k)
        dM = self.M_mask(IMinv @ (self.W.T @ self.C_XX @ self.W + self.V.T @ self.C_YX @ self.W
                                  + self.W.T @ self.C_XY @ self.V + self.V.T @ self.C_YY @ self.V) @ IMinv.T)

        print("absolute gradient: {}".format(
            max(abs(dW).max(), abs(dV).max(), abs(dLambda).max(), abs(dGamma).max(), abs(dM).max())))
        M_maskones = np.ones([self.k, self.k]) - self.M_mask(np.ones([self.k, self.k])) + self.M
        assert (self.M_mask(M_maskones) == self.M).all()
        print("relative gradient: {}".format(
            max(abs(dW / self.W).max(), abs(dV / self.V).max(), abs(dM / M_maskones).max(),
                abs(np.diag(dLambda) / np.diag(self.Lambda)).max(), abs(np.diag(dGamma) / np.diag(self.Gamma)).max())))

    def decay(self, t):
        """Return learning rate decay factor."""
        return max(1 - self.alpha * t, 0.1)

    def offline_train(self):
        """Offline training according to CCA algorithm.

        :param steps: number of training iterations.
        """
        for t in range(self.steps):
            self.obj[t], self.err[t] = self.objective(), self.error_angle()
            self.offline_step(self.decay(t))

    def offline_step(self, decay_t):
        """Perform an offline step."""
        IMinv = np.linalg.inv(np.eye(self.k) + self.M)
        dW = (self.C_XX @ self.W + self.C_XY @ self.V) @ IMinv.T - self.C_XX @ self.W @ self.Lambda
        dV = (self.C_YX @ self.W + self.C_YY @ self.V) @ IMinv.T - self.C_YY @ self.V @ self.Gamma
        dLambda = 1/2 *(self.W.T @ self.C_XX @ self.W - 1) * np.eye(self.k)
        dGamma = 1/2 * (self.V.T @ self.C_YY @ self.V - 1) * np.eye(self.k)
        dM = self.M_mask(IMinv @ (self.W.T @ self.C_XX @ self.W + self.V.T @ self.C_YX @ self.W
                                  + self.W.T @ self.C_XY @ self.V + self.V.T @ self.C_YY @ self.V) @ IMinv.T)

        self.W += decay_t * self.eta["W"] * dW
        self.V += decay_t * self.eta["V"] * dV
        self.Lambda += decay_t * self.eta["Lambda"] * dLambda
        self.Gamma += decay_t * self.eta["Gamma"] * dGamma
        self.M += decay_t * self.eta["M"] * dM

    def online_train(self, seed=None):
        """Online training according to CCA algorithm.

        :param steps: number of training iterations.
        """
        nprandom = get_nprandom(seed)
        for t in range(self.steps):
            self.obj[t], self.err[t] = self.objective(), self.error_angle()
            self.online_step(nprandom.randint(self.T), self.decay(t))

    def online_step(self, index, decay_t):
        """Perform an online step.

        :param index: X[:, index] and Y[:, index] will be used for this step.
        """
        I_X = self.W.T @ self.X[:, [index]]
        I_Y = self.V.T @ self.Y[:, [index]]
        Z = np.linalg.inv(np.eye(self.k) + self.M) @ (I_X + I_Y)
        self.W += decay_t * self.eta["W"] * self.X[:, [index]] @ (Z.T - I_X.T @ self.Lambda)
        self.V += decay_t * self.eta["V"] * self.Y[:, [index]] @ (Z.T - I_Y.T @ self.Gamma)
        self.W = np.maximum(np.minimum(self.W,1e5),-1e5) # this line was added to ensure stability of the algorithm
        self.V = np.maximum(np.minimum(self.V,1e5),-1e5)
        self.Lambda += decay_t * self.eta["Lambda"] * 1/2 * (I_X @ I_X.T - 1) * np.eye(self.k)
        self.Gamma += decay_t * self.eta["Gamma"] * 1/2 * (I_Y @ I_Y.T - 1) * np.eye(self.k)
        self.M += decay_t * self.eta["M"] * (Z @ Z.T)
        self.M = self.M_mask(self.M)