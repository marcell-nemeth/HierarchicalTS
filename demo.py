import numpy as np
from hierarchy import Hierarchy
from data_generator import HierarchicalDataGenerator
from gs_gls import GSGLS

def run_demo():
    print("Initializing Hierarchy...")
    # 1. Hierarchy: Root -> 2 Regions -> 2 Stores each
    structure = {
        'Total': ['Region_A', 'Region_B'],
        'Region_A': ['Store_A1', 'Store_A2'],
        'Region_B': ['Store_B1', 'Store_B2']
    }
    h = Hierarchy(structure)
    
    print("Generating Synthetic Data...")
    # 2. Data
    gen = HierarchicalDataGenerator(h, n_timesteps=200)
    Y_true = gen.generate_ground_truth()
    noise = gen.generate_spatiotemporal_noise(spatial_rho=1.5, temporal_ar_coefs=[0.8], noise_scale=2.0)
    Y_hat = Y_true + noise
    
    # Check Incoherence
    S = h.get_summing_matrix()
    bottom_idx = [h.node_to_idx[n] for n in h.bottom_nodes]
    top_idx = h.node_to_idx['Total']
    
    incoherence = np.abs(Y_hat[top_idx] - np.sum(Y_hat[bottom_idx], axis=0)).mean()
    print(f"Base Forecast Incoherence (MAE): {incoherence:.4f}")
    
    print("Running GS-GLS Estimation...")
    # 3. Estimation
    estimator = GSGLS(h)
    estimator.fit(noise) # Using noise as residuals
    
    print(f"Optimized Spatial Rho: {estimator.rho_opt:.4f}")
    print(f"Optimized GMRF Kappa: {estimator.kappa_opt:.4f}")
    
    print("Reconciling...")
    # 4. Reconciliation
    Y_tilde = estimator.reconcile(Y_hat)
    
    # Verify Coherence
    reconciled_incoherence = np.abs(Y_tilde[top_idx] - np.sum(Y_tilde[bottom_idx], axis=0)).max()
    print(f"Reconciled Incoherence (Max Error): {reconciled_incoherence:.2e}")
    
    # Accuracy
    mse_base = np.mean((Y_hat - Y_true)**2)
    mse_gsgls = np.mean((Y_tilde - Y_true)**2)
    
    print("-" * 30)
    print(f"MSE Base:   {mse_base:.4f}")
    print(f"MSE GS-GLS: {mse_gsgls:.4f}")
    improvement = (1 - mse_gsgls/mse_base)*100
    print(f"Improvement: {improvement:.2f}%")
    print("-" * 30)
    
    if reconciled_incoherence < 1e-5:
        print("SUCCESS: Forecasts are coherent.")
    else:
        print("WARNING: Forecasts might not be fully coherent.")

if __name__ == "__main__":
    run_demo()
