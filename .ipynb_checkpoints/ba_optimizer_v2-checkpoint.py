import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import qmc
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """
    Complete Bayesian Optimization pipeline for noisy ML model optimization
    """
    
    def __init__(self, X_init: np.ndarray, y_init: np.ndarray, bounds: Tuple[float, float] = (0, 1)):
        """
        Initialize the Bayesian Optimizer
        
        Args:
            X_init: Initial input data (n_samples, n_features)
            y_init: Initial output data (n_samples,)
            bounds: Bounds for the optimization space
        """
        self.X_init = X_init
        self.y_init = y_init
        self.bounds = bounds
        self.dim = X_init.shape[1]
        
        # Storage for optimization history
        self.X_all = X_init.copy()
        self.y_all = y_init.copy()
        
        # Model will be initialized in fit_surrogate
        self.model = None
        self.metrics = {}
        
        # Compute initial metrics
        self.compute_function_metrics()
        
    def compute_function_metrics(self) -> Dict[str, float]:
        """
        Compute various metrics from the current data
        
        Returns:
            Dictionary of computed metrics
        """
        X, y = self.X_all, self.y_all
        
        # Basic statistics
        self.metrics = {
            'mean': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y),
            'range': np.ptp(y),
            'n_samples': len(y)
        }
        
        # Correlation between features
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            self.metrics['feature_correlation'] = corr_matrix[0, 1] if X.shape[1] == 2 else np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        
        # Distance metrics
        distances = pdist(X)
        self.metrics['min_distance'] = np.min(distances)
        self.metrics['mean_distance'] = np.mean(distances)
        
        # Feature importance (correlation with output)
        for i in range(X.shape[1]):
            self.metrics[f'feature_{i}_importance'] = np.corrcoef(X[:, i], y)[0, 1]
        
        # Noise estimation (from duplicate or near-duplicate points)
        noise_est = self._estimate_noise()
        self.metrics['estimated_noise'] = noise_est
        
        return self.metrics
    
    def _estimate_noise(self) -> float:
        """
        Estimate noise level from near-duplicate points
        """
        X, y = self.X_all, self.y_all
        distances = squareform(pdist(X))
        
        # Find pairs of points that are very close
        close_pairs = []
        threshold = 0.01  # Consider points within 0.01 distance as "duplicates"
        
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if distances[i, j] < threshold:
                    close_pairs.append(abs(y[i] - y[j]))
        
        if close_pairs:
            return np.std(close_pairs)
        else:
            # Fallback: estimate from residuals if we had a simple model
            return np.std(y) * 0.1  # Conservative estimate
    
    def update_surrogate(self, kernel):
        """
        Fit/update the Gaussian Process surrogate model
        """
        # Create and fit the GP model
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            n_restarts_optimizer=15,
            random_state=42,
            normalize_y=True  # Important for numerical stability
        )
        
        self.model.fit(self.X_all, self.y_all)
        
        print(f"GP fitted with kernel: {self.model.kernel_}")
        print(f"Log-marginal likelihood: {self.model.log_marginal_likelihood():.4f}")
    
    def compute_samples(self, num_samples: int = 1000) -> np.ndarray:
        """
        Generate candidate samples using Latin Hypercube Sampling
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        lower, upper = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        samples = sampler.random(n=num_samples)
        
        # Scale to bounds
        samples = qmc.scale(samples, [lower] * self.dim, [upper] * self.dim)
        
        return samples
    
    def compute_acquisitions(self, X_candidates: np.ndarray, 
                           acquisition_func: str = 'ucb', 
                           kappa: float = 2.0) -> np.ndarray:
        """
        Compute acquisition function values for candidate points
        
        Args:
            X_candidates: Candidate points to evaluate
            acquisition_func: Type of acquisition function ('ucb', 'ei', 'pi')
            kappa: Exploration parameter for UCB
            
        Returns:
            Acquisition function values
        """
        if self.model is None:
            raise ValueError("Surrogate model not fitted. Call update_surrogate() first.")
        
        # Get predictions from GP
        mean, std = self.model.predict(X_candidates, return_std=True)
        
        if acquisition_func.lower() == 'ucb':
            return self.upper_confidence_bound(mean, std, kappa)
        elif acquisition_func.lower() == 'ei':
            return self.expected_improvement_v2(X_candidates)
        elif acquisition_func.lower() == 'pi':
            return self.probability_of_improvement(mean, std)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_func}")
    
    def upper_confidence_bound(self, mean: np.ndarray, 
                             std: np.ndarray, 
                             kappa: float = 2.0) -> np.ndarray:
        """
        Compute Upper Confidence Bound acquisition function
        """
        return mean + kappa * std

        
    def expected_improvement_v2(self, X):
        from scipy.stats import norm
    
        mu, sigma = self.model.predict(X, return_std=True)
        f_best = np.max(self.y_all)
        sigma = sigma + 1e-9  # Avoid divide-by-zero
        z = (mu - f_best) / sigma
        ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    
    def probability_of_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Compute Probability of Improvement acquisition function
        """
        from scipy.stats import norm
        
        f_best = np.max(self.y_all)
        z = (mean - f_best) / (std + 1e-9)
        
        return norm.cdf(z)
    
    def save_submission_values(self, X_candidates: np.ndarray, 
                             acquisition_values: np.ndarray, 
                             top_k: int = 5) -> Dict[str, Any]:
        """
        Select and save the best points for submission
        
        Args:
            X_candidates: Candidate points
            acquisition_values: Acquisition function values
            top_k: Number of top points to return
            
        Returns:
            Dictionary with submission information
        """
        # Get top k points
        top_indices = np.argsort(acquisition_values)[-top_k:][::-1]
        
        submission_data = {
            'best_point': X_candidates[top_indices[0]],
            'best_acquisition': acquisition_values[top_indices[0]],
            'top_k_points': X_candidates[top_indices],
            'top_k_acquisitions': acquisition_values[top_indices],
            'gp_predictions': self.model.predict(X_candidates[top_indices], return_std=True),
            'current_best_observed': np.max(self.y_all),
            'current_best_point': self.X_all[np.argmax(self.y_all)]
        }
        
        return submission_data
    
    def optimize_step(self, kernel, num_candidates: int = 1000, 
                     acquisition_func: str = 'ucb', 
                     kappa: float = 2.0) -> Dict[str, Any]:
        """
        Perform one complete optimization step
        
        Args:
            num_candidates: Number of candidate points to consider
            acquisition_func: Acquisition function to use
            kappa: Exploration parameter
            
        Returns:
            Submission data with recommended next point
        """
        print("=" * 60)
        print("BAYESIAN OPTIMIZATION STEP")
        print("=" * 60)
        
        # 1. Compute function metrics
        print("1. Computing function metrics...")
        self.compute_function_metrics()
        print(f"   Current best: {self.metrics['max']:.4f}")
        print(f"   Dataset size: {self.metrics['n_samples']}")
        print(f"   Estimated noise: {self.metrics['estimated_noise']:.4f}")
        
        # 2. Update surrogate model
        print("\n2. Updating surrogate model...")
        self.update_surrogate(kernel)
        
        # 3. Generate candidate samples
        print(f"\n3. Generating {num_candidates} candidate samples...")
        X_candidates = self.compute_samples(num_candidates)
        
        # 4. Compute acquisition function
        print(f"4. Computing {acquisition_func.upper()} acquisition function...")
        acquisition_values = self.compute_acquisitions(
            X_candidates, acquisition_func, kappa
        )
        
        # 5. Save submission values
        print("5. Selecting best points for submission...")
        submission_data = self.save_submission_values(X_candidates, acquisition_values)
        
        print(f"\nRECOMMENDED NEXT POINT:")
        print(f"Location: {submission_data['best_point']}")
        print(f"Acquisition: {submission_data['best_acquisition']:.4f}")
        
        mean, std = submission_data['gp_predictions']
        print(f"GP Prediction: {mean[0]:.4f} ± {std[0]:.4f}")
        
        return submission_data
    
    def add_observation(self, x_new: np.ndarray, y_new: float):
        """
        Add a new observation to the dataset
        
        Args:
            x_new: New input point
            y_new: New output value
        """
        self.X_all = np.vstack([self.X_all, x_new.reshape(1, -1)])
        self.y_all = np.append(self.y_all, y_new)
        print(f"Added observation: {x_new} -> {y_new:.4f}")
