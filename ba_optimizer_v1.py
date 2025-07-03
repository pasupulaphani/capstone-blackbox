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
    
    
    def __init__(self, X_init: np.ndarray, y_init: np.ndarray, bounds: Tuple[float, float] = (0, 1)):

        self.X_init = X_init
        self.y_init = y_init
        self.bounds = bounds
        self.dim = X_init.shape[1]
        
        self.X_all = X_init.copy()
        self.y_all = y_init.copy()
        
        self.model = None
        self.metrics = {}
        
        self.compute_function_metrics()
        
    def compute_function_metrics(self) -> Dict[str, float]:

        X, y = self.X_all, self.y_all
        
        self.metrics = {
            'mean': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y),
            'range': np.ptp(y),
            'n_samples': len(y)
        }
        
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            self.metrics['feature_correlation'] = corr_matrix[0, 1] if X.shape[1] == 2 else np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        
        distances = pdist(X)
        self.metrics['min_distance'] = np.min(distances)
        self.metrics['mean_distance'] = np.mean(distances)
        
        for i in range(X.shape[1]):
            self.metrics[f'feature_{i}_importance'] = np.corrcoef(X[:, i], y)[0, 1]
        
        noise_est = self._estimate_noise()
        self.metrics['estimated_noise'] = noise_est
        
        return self.metrics
    
    def _estimate_noise(self) -> float:
        """
        Estimate noise level from near-duplicate points
        """
        X, y = self.X_all, self.y_all
        distances = squareform(pdist(X))
        
        close_pairs = []
        threshold = 0.01
        
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if distances[i, j] < threshold:
                    close_pairs.append(abs(y[i] - y[j]))
        
        if close_pairs:
            return np.std(close_pairs)
        else:
            return np.std(y) * 0.1 
    
    def update_surrogate(self):
        noise_level = max(self.metrics['estimated_noise'], 1e-6)
        
        k1 = ConstantKernel(
            constant_value=self.metrics['std']**2, 
            constant_value_bounds=(0.1, 10.0)
        )
        k2 = RBF(
            length_scale=self.metrics['mean_distance'], 
            length_scale_bounds=(0.05, 5.0)
        )
        k3 = WhiteKernel(
            noise_level=noise_level, 
            noise_level_bounds=(1e-10, 0.5)
        )
        
        kernel = k1 * k2 + k3
        
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            n_restarts_optimizer=15,
            random_state=42,
            normalize_y=True
        )
        
        self.model.fit(self.X_all, self.y_all)
        
        print(f"GP fitted with kernel: {self.model.kernel_}")
        print(f"Log-marginal likelihood: {self.model.log_marginal_likelihood():.4f}")
    
    def compute_samples(self, num_samples: int = 1000) -> np.ndarray:
  
        lower, upper = self.bounds
        sampler = qmc.LatinHypercube(d=self.dim, seed=42)
        samples = sampler.random(n=num_samples)
        
        samples = qmc.scale(samples, [lower] * self.dim, [upper] * self.dim)
        
        return samples
    
    def compute_acquisitions(self, X_candidates: np.ndarray, 
                           acquisition_func: str = 'ucb', 
                           kappa: float = 2.0) -> np.ndarray:

        if self.model is None:
            raise ValueError("Surrogate model not fitted. Call update_surrogate() first.")
        
        # Get predictions from GP
        mean, std = self.model.predict(X_candidates, return_std=True)
        
        if acquisition_func.lower() == 'ucb':
            return self.upper_confidence_bound(mean, std, kappa)
        elif acquisition_func.lower() == 'ei':
            return self.expected_improvement(mean, std)
        elif acquisition_func.lower() == 'pi':
            return self.probability_of_improvement(mean, std)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_func}")
    
    def upper_confidence_bound(self, mean: np.ndarray, 
                             std: np.ndarray, 
                             kappa: float = 2.0) -> np.ndarray:
        return mean + kappa * std
    
    def expected_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        from scipy.stats import norm
        
        f_best = np.max(self.y_all)
        z = (mean - f_best) / (std + 1e-9)
        
        return (mean - f_best) * norm.cdf(z) + std * norm.pdf(z)
    
    def probability_of_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        from scipy.stats import norm
        
        f_best = np.max(self.y_all)
        z = (mean - f_best) / (std + 1e-9)
        
        return norm.cdf(z)
    
    def save_submission_values(self, X_candidates: np.ndarray, 
                             acquisition_values: np.ndarray, 
                             top_k: int = 5) -> Dict[str, Any]:
      
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
    
    def optimize_step(self, num_candidates: int = 1000, 
                     acquisition_func: str = 'ucb', 
                     kappa: float = 2.0) -> Dict[str, Any]:

        print("BAYESIAN OPTIMIZATION STEP")

        
        self.compute_function_metrics()
        print(f"   Current best: {self.metrics['max']:.4f}")
        print(f"   Dataset size: {self.metrics['n_samples']}")
        print(f"   Estimated noise: {self.metrics['estimated_noise']:.4f}")
        
        print("\n2. Updating surrogate model...")
        self.update_surrogate()
        
        print(f"\n3. Generating {num_candidates} candidate samples...")
        X_candidates = self.compute_samples(num_candidates)
        
        print(f"4. Computing {acquisition_func.upper()} acquisition function...")
        acquisition_values = self.compute_acquisitions(
            X_candidates, acquisition_func, kappa
        )
        
        print("5. Selecting best points for submission...")
        submission_data = self.save_submission_values(X_candidates, acquisition_values)
        
        print(f"\nRECOMMENDED NEXT POINT:")
        print(f"Location: {submission_data['best_point']}")
        print(f"Acquisition: {submission_data['best_acquisition']:.4f}")
        
        mean, std = submission_data['gp_predictions']
        print(f"GP Prediction: {mean[0]:.4f} ± {std[0]:.4f}")
        
        return submission_data
    
    def add_observation(self, x_new: np.ndarray, y_new: float):
        self.X_all = np.vstack([self.X_all, x_new.reshape(1, -1)])
        self.y_all = np.append(self.y_all, y_new)
        print(f"Added observation: {x_new} -> {y_new:.4f}")
