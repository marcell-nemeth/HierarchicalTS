import numpy as np
import scipy.linalg
from scipy.special import kv, gamma

class HierarchicalDataGenerator:
    """
    Generates synthetic hierarchical time series with controlled spatiotemporal dependencies.
    """
    def __init__(self, hierarchy, n_timesteps):
        self.hierarchy = hierarchy
        self.n_timesteps = n_timesteps
        self.n_nodes = hierarchy.n_nodes
        self.nodes = hierarchy.nodes

    def matern_kernel(self, distance_matrix, rho, nu=0.5):
        """
        Computes the Matern covariance kernel.
        K(d) = (2^(1-nu) / Gamma(nu)) * (sqrt(2*nu)*d/rho)^nu * K_nu(sqrt(2*nu)*d/rho)
        
        For nu=0.5, this simplifies to exp(-d/rho).
        """
        d = distance_matrix
        if nu == 0.5:
            return np.exp(-d / rho)
        
        # General case (handling d=0 to avoid division by zero)
        # K_nu goes to infinity at 0, but x^nu * K_nu(x) -> 2^(nu-1) * Gamma(nu)
        
        scaled_d = (np.sqrt(2 * nu) * d) / rho
        # Mask zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            val = (scaled_d)**nu * kv(nu, scaled_d)
        
        # Fix diagonal (d=0) limits
        val[d == 0] = 2**(nu - 1) * gamma(nu)
        
        final = (2**(1 - nu) / gamma(nu)) * val
        return final

    def generate_spatiotemporal_noise(self, spatial_rho, temporal_ar_coefs, noise_scale=1.0, 
                                      bias_scale=0.0, heteroscedastic=False):
        """
        Generates noise E (n_nodes x n_timesteps).
        Covariance is separable: Sigma_tm (x) Sigma_sp.
        
        Args:
            bias_scale: If > 0, adds a coherent bias component (e.g. consistently overestimating trend).
            heteroscedastic: If True, scales noise variance over time (increasing variance).
        """
        # 1. Spatial Covariance
        dist_matrix = self.hierarchy.get_geodesic_distance_matrix()
        Sigma_sp = self.matern_kernel(dist_matrix, rho=spatial_rho, nu=0.5)
        # Cholesky decomp for spatial coloring
        L_sp = np.linalg.cholesky(Sigma_sp + 1e-6 * np.eye(self.n_nodes))
        
        # 2. Temporal Covariance (via AR process)
        # Generate white noise first
        Z = np.random.normal(0, 1, (self.n_nodes, self.n_timesteps))
        
        # Apply Spatial Correlation 
        Z_sp = L_sp @ Z
        
        # Apply Temporal Correlation (AR filter on each row)
        E = np.zeros_like(Z_sp)
        p = len(temporal_ar_coefs)
        
        # Warmup for stationarity
        warmup = 100
        Z_warmup = np.random.normal(0, 1, (self.n_nodes, warmup))
        Z_warmup = L_sp @ Z_warmup
        current_hist = list(Z_warmup.T) 
        
        history = [current_hist[-(i+1)] for i in range(p)]
        
        for t in range(self.n_timesteps):
            # Innovation
            innovation = Z_sp[:, t] * noise_scale
            
            # Heteroscedasticity: Increase variance linearly over time
            if heteroscedastic:
                scale_factor = 1.0 + 3.0 * (t / self.n_timesteps) # 1x to 4x noise
                innovation *= scale_factor
                
            val = innovation.copy()
            for i, coef in enumerate(temporal_ar_coefs):
                if i < len(history):
                    val += coef * history[i]
            
            E[:, t] = val
            history.insert(0, val)
            history.pop()
            
        # 3. Add Bias (Simulate Model Misspecification)
        # Bias is usually structural spread across the hierarchy
        if bias_scale > 0:
            # Create a bias vector B_bias ~ N(0, Sigma_sp)
            # This bias is constant over time, making residuals non-zero mean!
            B_bias = L_sp @ np.random.normal(0, 1, (self.n_nodes, 1))
            E += bias_scale * B_bias
            
        return E

    def generate_ground_truth(self, trend_slope=0.1, seasonal_amp=5.0, period=12):
        """
        Generates coherent ground truth Y (n_nodes x n_timesteps).
        Logic: 
        1. Generate smooth trends/seasonality for BOTTOM nodes.
        2. Aggregate to getting full coherent Y.
        """
        m = self.hierarchy.m_bottom
        T = self.n_timesteps
        
        B_truth = np.zeros((m, T))
        
        time = np.arange(T)
        
        for i in range(m):
            # Random variations in trend and phase
            local_slope = trend_slope * (1 + np.random.uniform(-0.5, 0.5))
            phase = np.random.uniform(0, 2*np.pi)
            
            trend = local_slope * time
            seasonal = seasonal_amp * np.sin(2 * np.pi * time / period + phase)
            base_level = 100 + np.random.uniform(0, 50)
            
            B_truth[i, :] = base_level + trend + seasonal
            
        # Aggregate
        S = self.hierarchy.get_summing_matrix() # n_nodes x m
        Y_truth = S @ B_truth
        
        return Y_truth
