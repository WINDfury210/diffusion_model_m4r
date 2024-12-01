import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class Diffuser:
    def __init__(self, x0=None, beta=0.05, sigma=0.1, dt=0.01, T=1, model=None):
        """
        Initialize the Diffuser class with the given parameters.
        
        Parameters:
        - x0: Initial data for fitting the GMM (optional for initialization).
        - beta: Drift coefficient for the simulation.
        - sigma: Diffusion coefficient (for noise term).
        - dt: Time step for the simulation.
        - T: Final time for the simulation.
        """
        self.x0 = x0
        self.beta = beta if callable(beta) else lambda x, t: beta
        self.sigma = sigma if callable(sigma) else lambda x, t: sigma
        self.dt = dt
        self.T = T
        self.num_steps = int(T / dt)
        self.model = model


    def fit_gmm(self, data, n_components):
        """
        Fit a Gaussian Mixture Model (GMM) to the provided data.
        
        Parameters:
        - data: The data to fit the GMM to (any dimensionality).
        - n_components: Number of components for the GMM. If 'Auto', it determines based on clusters in data.
        """
        
        flat_data = data.reshape(data.shape[0], -1)
        self.model = GaussianMixture(n_components=n_components, covariance_type='full')
        self.model.fit(flat_data)

    
    def gmm_log_gradient(self, x, covariance_type='full'):
        """
        Compute the gradient of the log of the GMM PDF at a given point.

        Parameters:
        - x: A (..., d) array of points where the gradient should be evaluated.
        - covariance_type: The type of covariance matrix ('full' or 'diag').

        Returns:
        - grad_log_pdf: The gradient of the log PDF at each point, reshaped to match x.
        """
        epsilon = 1e-6
        original_shape = x.shape
        flat_x = x.reshape(x.shape[0], -1)

        means = self.model.means_
        weights = self.model.weights_

        if covariance_type == 'full':
            covariances = self.model.covariances_
            inv_cov = np.linalg.inv(covariances + epsilon * np.eye(covariances.shape[-1]))
            diff = flat_x[np.newaxis, :, :] - means[:, np.newaxis, :]
            grad = np.matmul(diff, inv_cov)
        elif covariance_type == 'diag':
            diag_covariances = self.model.covariances_
            inv_diag_cov = 1 / (diag_covariances + epsilon)
            diff = flat_x[np.newaxis, :, :] - means[:, np.newaxis, :]
            grad = diff * inv_diag_cov[:, np.newaxis, :]
        else:
            raise ValueError("Invalid covariance_type: {}".format(covariance_type))

        pdf_values = np.zeros((means.shape[0], flat_x.shape[0]))
        for k in range(means.shape[0]):
            if covariance_type == 'full':
                pdf_values[k, :] = multivariate_normal.pdf(flat_x, mean=means[k], cov=covariances[k])
            elif covariance_type == 'diag':
                var_diag = diag_covariances[k]
                norm_factor = np.prod(np.sqrt(2 * np.pi * var_diag))
                norm_factor = np.maximum(norm_factor, epsilon)
                exp_term = -0.5 * np.sum((diff[k] ** 2) * inv_diag_cov[k], axis=1)
                exp_term = np.clip(exp_term, -1e6, 1e6)
                pdf_values[k, :] = np.exp(exp_term) / norm_factor

        weighted_grad = np.sum(-weights[:, np.newaxis, np.newaxis] * grad * pdf_values[:, :, np.newaxis], axis=0)

        p = np.sum(weights[:, np.newaxis] * pdf_values, axis=0)
        p = np.maximum(p, epsilon)

        grad_log_pdf_flat = weighted_grad / p[:, np.newaxis]
        grad_log_pdf = grad_log_pdf_flat.reshape(original_shape)

        return grad_log_pdf



    def simulate(self, x0=None, **kwargs):
        """
        Simulate the forward diffusion process using the Euler-Maruyama method.
        
        Parameters:
        - x0: The initial state (starting point of the simulation).
        
        Returns:
        - trajectory: The simulated trajectory of the diffusion process.
        """
        
        num_steps = self.num_steps
        if x0 is None:
            x0 = self.x0.copy()
        trajectory = np.zeros((num_steps, *x0.shape))
        x = x0.copy()
        beta = self.beta
        sigma = self.sigma
        
        for t in range(num_steps):
            drift = beta(x, t) * self.dt
            diffusion = sigma(x, t) * np.sqrt(self.dt) * np.random.normal(size=x.shape)
            x += (drift + diffusion)
            trajectory[t] = x
        
        return trajectory

    def backward_simulate(self, x0, dt, T, **kwargs):
        """
        Simulate the backward diffusion process using the reverse of the forward process.
        
        Parameters:
        - x0: The starting point for the simulation.
        
        Returns:
        - trajectory: The simulated trajectory of the backward diffusion process.
        """
        num_steps = int(T / dt)
        
        trajectory = np.zeros((num_steps, *x0.shape))
        x = x0.copy()
        beta = self.beta
        sigma = self.sigma
        
        for t in range(num_steps):
            grad_log_pdf = self.gmm_log_gradient(x, covariance_type=kwargs.get('covariance_type', 'full'))
            dx = (beta(x, t) - sigma(x, t) ** 2 - grad_log_pdf) * dt
            dw = sigma(x, t) * np.random.randn(*x.shape) * np.sqrt(dt)
            x -= (dx + dw)
            
            trajectory[t] = x
        
        return trajectory


