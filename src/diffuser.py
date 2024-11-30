import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


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

    def fit_gmm(self, data, n_components='Auto'):
        """
        Fit a Gaussian Mixture Model (GMM) to the provided data.
        
        Parameters:
        - data: The data to fit the GMM to (any dimensionality).
        - n_components: Number of components for the GMM. If 'Auto', it determines based on clusters in data.
        """
        original_shape = data.shape
        flat_data = data.reshape(-1, data.shape[-1])  # Flatten the data if it's multi-dimensional

        if n_components == 'Auto':
            kmeans = KMeans(n_clusters='auto', random_state=42)  # Adjust to your preferred method for auto-detection
            kmeans.fit(flat_data)
            n_components = len(np.unique(kmeans.labels_))

        self.model = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.model.fit(flat_data)

    
    def gmm_log_gradient(self, x):
        """
        Compute the gradient of the log of the GMM PDF at a given point.

        Parameters:
        - x: A (..., d) array of points where the gradient should be evaluated.

        Returns:
        - grad_log_pdf: The gradient of the log PDF at each point, reshaped to match x.
        """
        epsilon = 1e-6
        original_shape = x.shape
        flat_x = x.flatten()  # Flatten to (n_samples, d)

        means = self.model.means_
        covariances = self.model.covariances_
        weights = self.model.weights_

        # Compute the difference between the data points and the means
        diff = flat_x[np.newaxis, :] - means  # Shape: (n_samples, n_components, d)
        
        # Inverse covariance matrix
        inv_cov = np.linalg.inv(covariances + epsilon * np.eye(covariances.shape[-1]))

        # Compute the gradient
        grad = np.matmul(diff, inv_cov)  # Shape: (n_samples, n_components, d)

        # Compute the PDF values for each component and data point
        pdf_values = np.zeros((means.shape[0], flat_x.shape[0]))
        for k in range(means.shape[0]):
            pdf_values[k, :] = multivariate_normal.pdf(flat_x, mean=means[k], cov=covariances[k])

        # Weighted gradient
        weighted_grad = np.sum(-weights[:, np.newaxis] * grad * pdf_values[:, :, np.newaxis], axis=1)  # Shape: (n_samples, d)

        # Compute the overall probability for each sample
        p = np.sum(weights[:, np.newaxis] * pdf_values, axis=0)
        p = np.maximum(p, epsilon)  # Avoid division by zero

        # Compute the final gradient of the log PDF
        grad_log_pdf_flat = weighted_grad / p[:, np.newaxis]

        # Reshape the gradient to the original shape
        grad_log_pdf = grad_log_pdf_flat.reshape(original_shape)

        return grad_log_pdf



    def simulate(self, x0=None):
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

    def backward_simulate(self, x0, dt, T):
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
            grad_log_pdf = self.gmm_log_gradient(x)
            dx = (beta(x, t) - sigma(x, t) ** 2 - grad_log_pdf) * dt
            dw = sigma(x, t) * np.random.randn(*x.shape) * np.sqrt(dt)
            x -= (dx + dw)
            
            trajectory[t] = x
        
        return trajectory


