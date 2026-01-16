"""
MinT Reconciliation Module

Implements Minimum Trace (MinT) reconciliation for hierarchical time series.
"""

import numpy as np
from scipy import linalg
from typing import Optional, Literal
from hierarchy_builder import HierarchyBuilder


class MinTReconciler:
    """
    Minimum Trace (MinT) reconciler for hierarchical forecasts.
    
    MinT finds the reconciled forecasts that:
    1. Satisfy hierarchical constraints (coherent)
    2. Minimize the trace of the forecast error covariance matrix
    
    The reconciliation formula is:
    ŷ_reconciled = S(S'W⁻¹S)⁻¹S'W⁻¹ŷ_base
    
    where:
    - S is the summing matrix
    - W is the forecast error covariance matrix
    - ŷ_base are the base (incoherent) forecasts
    """
    
    def __init__(
        self,
        hierarchy: HierarchyBuilder,
        method: Literal['sample', 'shrinkage', 'ols', 'wls'] = 'sample'
    ):
        """
        Initialize MinT reconciler.
        
        Parameters
        ----------
        hierarchy : HierarchyBuilder
            Hierarchy structure.
        method : str
            Covariance estimation method:
            - 'sample': Sample covariance matrix (MinT-Sample)
            - 'shrinkage': Shrinkage estimator (MinT-Shrinkage)
            - 'ols': Ordinary least squares (W = I)
            - 'wls': Weighted least squares (W = diagonal)
        """
        self.hierarchy = hierarchy
        self.method = method
        self.S = hierarchy.S
        self.n_nodes = hierarchy.n_nodes
        self.n_bottom = hierarchy.n_bottom
        
        # Covariance matrix and reconciliation matrix (to be estimated)
        self.W = None
        self.P = None
        self.is_fitted = False
        
    def fit(self, forecast_errors: np.ndarray):
        """
        Fit the reconciler by estimating covariance matrix from errors.
        
        Parameters
        ----------
        forecast_errors : np.ndarray
            Historical forecast errors with shape (n_samples, n_nodes).
        """
        if self.method == 'sample':
            self.W = self._estimate_sample_covariance(forecast_errors)
        elif self.method == 'shrinkage':
            self.W = self._estimate_shrinkage_covariance(forecast_errors)
        elif self.method == 'ols':
            self.W = np.eye(self.n_nodes)
        elif self.method == 'wls':
            self.W = self._estimate_diagonal_covariance(forecast_errors)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute reconciliation matrix
        self.P = self._compute_projection_matrix()
        self.is_fitted = True
        
    def _estimate_sample_covariance(
        self,
        errors: np.ndarray,
        regularization: float = 1e-8
    ) -> np.ndarray:
        """
        Estimate sample covariance matrix.
        
        Parameters
        ----------
        errors : np.ndarray
            Forecast errors (n_samples, n_nodes).
        regularization : float
            Small value added to diagonal for numerical stability.
            
        Returns
        -------
        np.ndarray
            Sample covariance matrix.
        """
        n_samples = errors.shape[0]
        
        # Center the errors
        errors_centered = errors - errors.mean(axis=0)
        
        # Compute sample covariance
        W = (errors_centered.T @ errors_centered) / (n_samples - 1)
        
        # Add regularization for numerical stability
        W += regularization * np.eye(self.n_nodes)
        
        return W
    
    def _estimate_shrinkage_covariance(
        self,
        errors: np.ndarray,
        regularization: float = 1e-8
    ) -> np.ndarray:
        """
        Estimate shrinkage covariance matrix (Ledoit-Wolf).
        
        Shrinkage covariance is a convex combination of sample covariance
        and a structured estimator (diagonal matrix).
        
        Parameters
        ----------
        errors : np.ndarray
            Forecast errors (n_samples, n_nodes).
        regularization : float
            Small value added to diagonal for numerical stability.
            
        Returns
        -------
        np.ndarray
            Shrinkage covariance matrix.
        """
        n_samples, n_nodes = errors.shape
        
        # Sample covariance
        errors_centered = errors - errors.mean(axis=0)
        S_sample = (errors_centered.T @ errors_centered) / (n_samples - 1)
        
        # Target: diagonal matrix (variances only)
        S_target = np.diag(np.diag(S_sample))
        
        # Shrinkage intensity (simple estimator)
        # In practice, use more sophisticated methods like Ledoit-Wolf
        shrinkage = min(0.5, 1.0 / np.sqrt(n_samples))
        
        # Shrinkage estimator
        W = shrinkage * S_target + (1 - shrinkage) * S_sample
        
        # Add regularization
        W += regularization * np.eye(n_nodes)
        
        return W
    
    def _estimate_diagonal_covariance(
        self,
        errors: np.ndarray,
        regularization: float = 1e-8
    ) -> np.ndarray:
        """
        Estimate diagonal covariance matrix (for WLS).
        
        Parameters
        ----------
        errors : np.ndarray
            Forecast errors (n_samples, n_nodes).
        regularization : float
            Small value added to diagonal for numerical stability.
            
        Returns
        -------
        np.ndarray
            Diagonal covariance matrix.
        """
        variances = np.var(errors, axis=0, ddof=1)
        W = np.diag(variances + regularization)
        return W
    
    def _compute_projection_matrix(self) -> np.ndarray:
        """
        Compute reconciliation projection matrix P.
        
        P = S(S'W⁻¹S)⁻¹S'W⁻¹
        
        Returns
        -------
        np.ndarray
            Projection matrix of shape (n_nodes, n_nodes).
        """
        # Use pseudo-inverse for better numerical stability
        # Add small regularization to ensure invertibility
        reg = 1e-10 * np.eye(self.n_nodes)
        W_reg = self.W + reg
        W_inv = np.linalg.pinv(W_reg)
        
        # Compute S'W^{-1}S
        SW = self.S.T @ W_inv @ self.S
        
        # Add regularization to SW for numerical stability
        reg_bottom = 1e-10 * np.eye(self.n_bottom)
        SW_reg = SW + reg_bottom
        SW_inv = np.linalg.pinv(SW_reg)
        
        # Compute projection matrix
        P = self.S @ SW_inv @ self.S.T @ W_inv
        
        return P
    
    def reconcile(self, base_forecasts: np.ndarray) -> np.ndarray:
        """
        Reconcile base forecasts.
        
        Parameters
        ----------
        base_forecasts : np.ndarray
            Base forecasts with shape (n_periods, n_nodes) or (n_nodes,).
            
        Returns
        -------
        np.ndarray
            Reconciled forecasts with same shape as input.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Reconciler must be fitted before reconciliation. "
                "Call fit() first."
            )
        
        # Handle both 1D and 2D inputs
        input_1d = (base_forecasts.ndim == 1)
        if input_1d:
            base_forecasts = base_forecasts.reshape(1, -1)
        
        # Apply reconciliation: ŷ_rec = P @ ŷ_base
        reconciled = (self.P @ base_forecasts.T).T
        
        # Return in same format as input
        if input_1d:
            reconciled = reconciled.ravel()
        
        return reconciled
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the reconciliation projection matrix.
        
        Returns
        -------
        np.ndarray
            Projection matrix P.
        """
        if not self.is_fitted:
            raise RuntimeError("Reconciler must be fitted first.")
        return self.P.copy()
    
    def verify_projection_properties(self) -> dict:
        """
        Verify theoretical properties of projection matrix.
        
        The projection matrix P should satisfy:
        1. PS = S (preserves aggregation structure)
        2. P is idempotent for certain cases
        
        Returns
        -------
        dict
            Dictionary with verification results.
        """
        if not self.is_fitted:
            raise RuntimeError("Reconciler must be fitted first.")
        
        results = {}
        
        # Check PS = S
        PS = self.P @ self.S
        results['PS_equals_S'] = np.allclose(PS, self.S)
        results['PS_S_max_diff'] = np.max(np.abs(PS - self.S))
        
        return results
