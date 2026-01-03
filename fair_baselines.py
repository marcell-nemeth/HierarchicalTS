
import numpy as np
from scipy.linalg import inv, pinv

class BaselineReconciler:
    def __init__(self, method, S_total):
        self.method = method
        self.S = S_total
        self.P = None  # Projection Matrix
        
    def fit(self, residuals=None):
        """
        Compute the Projection Matrix P once.
        residuals: (n_samples, n_features) - flattened blocks
        """
        S = self.S
        
        if self.method == 'OLS':
            # P = S (S' S)^-1 S'
            # We can compute this via lstsq: P = S * pinv(S)
            S_pinv = pinv(S)
            self.P = S @ S_pinv
            
        elif self.method == 'MinT_Sample':
            n_samples = residuals.shape[0]
            # W = E[e e']
            W = (residuals.T @ residuals) / n_samples
            W += 1e-8 * np.eye(W.shape[0]) # Regularize
            W_inv = inv(W)
            
            # P = S (S' W^-1 S)^-1 S' W^-1
            STS_inv = inv(S.T @ W_inv @ S)
            self.P = S @ STS_inv @ S.T @ W_inv
            
        elif self.method == 'MinT_Shrink':
            n_samples = residuals.shape[0]
            # W_diag = diag(var(e_i))
            emp_cov = (residuals.T @ residuals) / n_samples
            W_diag = np.diag(np.diag(emp_cov))
            W_inv = np.diag(1.0 / (np.diag(emp_cov) + 1e-8))
            
            STS_inv = inv(S.T @ W_inv @ S)
            self.P = S @ STS_inv @ S.T @ W_inv
            
    def reconcile(self, y_hat_flat):
        # Inference is just Matrix-Vector multiplication
        return self.P @ y_hat_flat
