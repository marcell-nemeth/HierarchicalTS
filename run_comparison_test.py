
import numpy as np
import pandas as pd
import time
from hierarchy import Hierarchy
from data_generator import HierarchicalDataGenerator
from gs_gls import GSGLS
from baselines import mint_sample, mint_shrink, ols_identity, build_spatiotemporal_s

def test_pipeline():
    print("Initializing Hierarchy...")
    # Create a small random hierarchy
    # 1 -> [2, 3] -> [4, 5, 6, 7]
    structure = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6]
    }
    # Nodes: 0..6. Bottom: 3,4,5,6 (m=4). Total=7.
    h = Hierarchy(structure)
    
    n_t = 50
    dg = HierarchicalDataGenerator(h, n_timesteps=n_t)
    
    # 1. Stationary Case
    print("\n--- Testing Stationary Case ---")
    Y_truth = dg.generate_ground_truth()
    E = dg.generate_spatiotemporal_noise(spatial_rho=2.0, temporal_ar_coefs=[0.5], bias_scale=0.0)
    Y_hat = Y_truth + E
    
    print(f"Y_hat shape: {Y_hat.shape}") # (7, 50)
    
    # Split Train/Test (Reconciliation usually uses residuals from history to reconcile future?)
    # Here we fit on the SAME residuals for now (In-Sample Reconcil) to test mechanism.
    # In practice: Fit on E_train, Reconcile Y_hat_test.
    # We will use E as "residuals".
    
    # GS-GLS Stationary
    print("Running GS-GLS (Stationary)...")
    model_stat = GSGLS(h, mode='stationary')
    model_stat.fit(E) # Fit on known errors
    Y_tilde_stat = model_stat.reconcile(Y_hat)
    print(f"Reconciled Shape: {Y_tilde_stat.shape}")
    
    # Check coherence
    S = h.get_summing_matrix()
    # Coherence: Y_agg = S_agg * Y_bottom
    # Y_tilde should be in Range(S).
    # i.e. Y_tilde - S @ (pinv(S) @ Y_tilde) approx 0
    # Or simply: Top node = Sum of children.
    # Node 0 = Node 1 + Node 2?
    err = np.abs(Y_tilde_stat[0] - (Y_tilde_stat[1] + Y_tilde_stat[2]))
    print(f"Max Coherence Error (Node 0): {np.max(err):.6e}")
    
    # Baselines
    # Need flattened S_total?
    # Baselines expect flat vectors?
    # baselines.py: y_hat (N,), S (N x M).
    # If we reconcile one time step at a time:
    S_sp = h.get_summing_matrix() # (n_s x m_s)
    
    print("Running MinT (Sample)...")
    # For MinT, we need covariance of residuals.
    # Residuals shape (n_s x n_t). Transpose to (Samples, Variables) -> (n_t, n_s).
    res_T = E.T
    # Reconcile first time step
    y_hat_t0 = Y_hat[:, 0]
    
    # MinT Sample
    y_mint = mint_sample(y_hat_t0, res_T, S_sp)
    print(f"MinT Result Shape: {y_mint.shape}")
    
    # 2. Non-Stationary Case
    print("\n--- Testing Non-Stationary Case ---")
    E_ns = dg.generate_spatiotemporal_noise(spatial_rho=2.0, temporal_ar_coefs=[0.5], 
                                            heteroscedastic=True)
    Y_hat_ns = Y_truth + E_ns
    
    print("Running GS-GLS (Non-Stationary)...")
    model_ns = GSGLS(h, mode='non_stationary', wavelet_family='db2')
    model_ns.fit(E_ns)
    Y_tilde_ns = model_ns.reconcile(Y_hat_ns)
    
    err_ns = np.abs(Y_tilde_ns[0] - (Y_tilde_ns[1] + Y_tilde_ns[2]))
    print(f"Max Coherence Error (Non-Stationary): {np.max(err_ns):.6e}")

if __name__ == "__main__":
    try:
        test_pipeline()
        print("TEST PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
