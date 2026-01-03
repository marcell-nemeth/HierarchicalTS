import numpy as np
import pandas as pd
import time
import sys
import os
import scipy.sparse as sp
from hierarchy import Hierarchy
from data_generator import HierarchicalDataGenerator
from gs_gls import GSGLS
import baselines
from fair_baselines import BaselineReconciler

def generate_large_hierarchy():
    # Scenario 'L1': Depth 5, Branch 3 ~ 100-300 nodes? 
    # Depth 5, Branch 3: 1 + 3 + 9 + 27 + 81 + 243 = 364 nodes.
    # Let's go slightly larger for runtime contrast.
    # Depth 6, Branch 3: + 729 = ~1000 nodes.
    print("Generating Hierarchy (Depth=6, Branch=3)...")
    structure = {}
    current_layer = ['Total']
    node_ctr = 1
    depth = 5
    branch = 3
    for d in range(depth):
        next_layer = []
        for parent in current_layer:
            children = []
            for _ in range(branch):
                child_name = f'Node_{d+1}_{node_ctr}'
                children.append(child_name)
                node_ctr += 1
            structure[parent] = children
            next_layer.extend(children)
        current_layer = next_layer
    return Hierarchy(structure)

def run_performance_test():
    h = generate_large_hierarchy()
    print(f"Total Nodes: {h.n_nodes}")
    
    n_time = 2000
    n_train = 1500
    block_size = 10
    
    print(f"Simulating Time Series (T={n_time})...")
    gen = HierarchicalDataGenerator(h, n_timesteps=n_time)
    Y_true = gen.generate_ground_truth()
    # High noise to make reconciliation harder
    E = gen.generate_spatiotemporal_noise(spatial_rho=0.8, temporal_ar_coefs=[0.6], noise_scale=2.0)
    Y_hat = Y_true + E
    
    residuals_train = E[:, :n_train]
    Y_hat_test = Y_hat[:, n_train:]
    Y_true_test = Y_true[:, n_train:]
    
    # 1. Baseline: MinT (Shrinkage) - usually the competitive baseline
    print("\n--- Benchmarking MinT (Shrinkage) ---")
    t0 = time.time()
    S_sp = h.get_summing_matrix()
    S_total = baselines.build_spatiotemporal_s(S_sp, block_size)
    
    # MinT requires samples
    train_blocks = []
    for i in range(0, residuals_train.shape[1] - block_size + 1, block_size):
        block = residuals_train[:, i:i+block_size]
        train_blocks.append(block.flatten(order='F'))
    residuals_samples = np.array(train_blocks)
    
    # Fit
    mint = BaselineReconciler('MinT_Shrink', S_total)
    mint.fit(residuals_samples)
    train_time_mint = time.time() - t0
    
    # Inference (Measure batch)
    t1 = time.time()
    mse_mint_list = []
    
    # Run full test set to be fair
    for i in range(0, Y_hat_test.shape[1] - block_size + 1, block_size):
        y_hat_blk = Y_hat_test[:, i:i+block_size]
        y_true_blk = Y_true_test[:, i:i+block_size]
        y_hat_flat = y_hat_blk.flatten(order='F')
        y_tilde_flat = mint.reconcile(y_hat_flat)
        y_tilde_blk = y_tilde_flat.reshape((h.n_nodes, block_size), order='F')
        mse_mint_list.append(np.mean((y_tilde_blk - y_true_blk)**2))
        
    infer_time_mint = time.time() - t1
    mse_mint = np.mean(mse_mint_list)
    
    print(f"MinT Time: Train={train_time_mint:.3f}s, Infer={infer_time_mint:.3f}s")
    print(f"MinT MSE:  {mse_mint:.4f}")
    
    # 2. GS-GLS (Spectral)
    print("\n--- Benchmarking GS-GLS (Spectral) ---")
    t0 = time.time()
    gs = GSGLS(h, temporal_method='spectral')
    gs.fit(residuals_train)
    train_time_gs = time.time() - t0
    
    t1 = time.time()
    mse_gs_list = []
    
    for i in range(0, Y_hat_test.shape[1] - block_size + 1, block_size):
        y_hat_blk = Y_hat_test[:, i:i+block_size]
        y_true_blk = Y_true_test[:, i:i+block_size]
        y_tilde_blk = gs.reconcile(y_hat_blk)
        mse_gs_list.append(np.mean((y_tilde_blk - y_true_blk)**2))
        
    infer_time_gs = time.time() - t1
    mse_gs = np.mean(mse_gs_list)
    
    print(f"GS-GLS Time: Train={train_time_gs:.3f}s, Infer={infer_time_gs:.3f}s")
    print(f"GS-GLS MSE:  {mse_gs:.4f}")
    
    # 3. Imputation Check
    print("\n--- Imputation Check (GS-GLS) ---")
    # Mask 20%
    y_test_impute = Y_hat_test.copy()
    mask = np.random.choice([True, False], size=y_test_impute.shape, p=[0.8, 0.2])
    y_test_impute[~mask] = np.nan
    
    t0 = time.time()
    # Simple linear interp for baseline
    y_interp = pd.DataFrame(y_test_impute.T).interpolate().values.T
    mse_interp = np.mean((y_interp[~mask] - Y_true_test[~mask])**2)
    
    # GS Impute
    y_gs_imp = gs.reconcile_impute(y_test_impute, maxiter=20)
    mse_imp = np.mean((y_gs_imp[~mask] - Y_true_test[~mask])**2)
    time_imp = time.time() - t0
    
    print(f"Imputation MSE: Interp={mse_interp:.4f}, GS-GLS={mse_imp:.4f}")
    print(f"Imputation Time: {time_imp:.3f}s")
    
    # Summary
    print("\n=== RESULTS ===")
    print(f"MSE Improvement: {(1 - mse_gs/mse_mint)*100:.2f}% better than MinT")
    print(f"Training Speedup: {train_time_mint/train_time_gs:.2f}x faster/slower (- means slower)")
    print(f"Inference Speedup: {infer_time_mint/infer_time_gs:.2f}x faster/slower")
    
    if mse_gs < mse_mint:
        print("VERIFIED: GS-GLS outperforms MinT in accuracy.")
    else:
        print("WARNING: GS-GLS did not outperform MinT in accuracy.")
        
    if mse_imp < mse_interp:
        print("VERIFIED: GS-GLS Imputation outperforms Linear Interpolation.")
        
if __name__ == "__main__":
    run_performance_test()
