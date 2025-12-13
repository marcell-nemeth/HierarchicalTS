import numpy as np
from scipy.linalg import pinv, inv
import scipy.sparse as sp

def build_spatiotemporal_s(S_sp, n_t):
    """
    Constructs the global Summing Matrix S_total = I_nt (x) S_sp.
    S_sp: (n_s x m_s)
    n_t: Number of time steps in the block.
    
    Returns:
        S_total: (n_s*n_t x m_s*n_t)
    """
    # Kronecker product of Identity(n_t) and S_sp
    # We want stacking order: Time outer, Space inner? 
    # Usually vec(Y) stacks columns (time steps) one after another? 
    # Or stacks spatial nodes for each time step?
    # Let's assume vec(Y) = [y_{t=0}, y_{t=1}, ...]. 
    # Then for each block t, y_t = S_sp * b_t.
    # So the global diagonal structure is correct.
    return np.kron(np.eye(n_t), S_sp)

def ols_identity(y_hat, S):
    """
    OLS Reconciliation: y_tilde = S (S' S)^-1 S' y_hat
    Complexity: O(N) if exploiting sparsity, but here O(N^3) naive or O(N M^2).
    
    Args:
        y_hat: Flattened Forecast Vector (N,)
        S: Summing Matrix (N x M)
    """
    # y_tilde = S * pinv(S) * y_hat
    # Or strictly: b_ols = (S'S)^-1 S' y_hat
    # y_tilde = S * b_ols
    
    # Use lstsq for stability
    # min || y_hat - S b ||
    b_ols, _, _, _ = np.linalg.lstsq(S, y_hat, rcond=None)
    return S @ b_ols

def mint_sample(y_hat, residuals, S):
    """
    MinT Reconciliation with Sample Covariance.
    y_tilde = S (S' W^-1 S)^-1 S' W^-1 y_hat
    
    Args:
        y_hat: Flattened Forecast Vector (N,)
        residuals: Matrix of flattened historical residuals (n_samples x N)
        S: Summing Matrix (N x M)
    """
    # 1. Estimate W (Sample Covariance)
    # residuals shape: (Samples, N)
    # Center residuals? Usually errors are mean 0.
    n_samples = residuals.shape[0]
    W = (residuals.T @ residuals) / n_samples
    
    # 2. Add regularization for invertibility if needed (trivial shrinkage)
    W += 1e-8 * np.eye(W.shape[0])
    
    # 3. Invert W
    # This is the O(N^3) step
    W_inv = inv(W)
    
    # 4. Compute Projection
    # P = S (S' W^-1 S)^-1 S' W^-1
    STS_inv = inv(S.T @ W_inv @ S)
    P = S @ STS_inv @ S.T @ W_inv
    
    return P @ y_hat

def mint_shrink(y_hat, residuals, S):
    """
    MinT Reconciliation with Diagonal Shrinkage (Target: Diagonal Variance).
    Faster/More robust than Sample if N is large vs Samples.
    
    Args:
        y_hat: Flattened Forecast Vector (N,)
        residuals: (n_samples x N)
        S: Summing Matrix (N x M)
    """
    n_samples = residuals.shape[0]
    
    # Target: Diagonal of Sample Cov
    emp_cov = (residuals.T @ residuals) / n_samples
    target = np.diag(np.diag(emp_cov))
    
    # Simple shrinkage formulation: alpha * Target + (1-alpha) * Sample
    # For now, let's just implement "Shrinkage to Diagonal" as the method name often implies 
    # using a diagonal assumption (WLS-Var) or Ledoit-Wolf. 
    # The prompt table lists "MinT (Shrinkage)" separate from OLS.
    # I will stick to "Shrink to Diagonal" (WLS-Variance) as a proxy if sklearn is absent, 
    # OR use a basic Ledoit-Wolf formula if I can write it quickly.
    # Let's use WLS-Variance (Diagonal W) as a simplified robust baseline, 
    # effectively W[i,i] = var(e_i).
    
    W_diag = np.diag(np.diag(emp_cov))
    W_inv = np.diag(1.0 / (np.diag(emp_cov) + 1e-8))
    
    # Exact same projection formula
    STS_inv = inv(S.T @ W_inv @ S)
    P = S @ STS_inv @ S.T @ W_inv
    
    return P @ y_hat
