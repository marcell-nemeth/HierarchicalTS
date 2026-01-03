
import numpy as np
import pandas as pd
import time
import random
import scipy.linalg
from scipy.linalg import inv, pinv
import sys
import os

# Ensure local imports work
sys.path.append(os.getcwd())

from hierarchy import Hierarchy
from data_generator import HierarchicalDataGenerator
from gs_gls import GSGLS
import baselines

np.random.seed(42)

def generate_random_hierarchy(depth=4, branching_factor=5):
    structure = {}
    current_layer = ['Total']
    node_ctr = 1
    for d in range(depth):
        next_layer = []
        for parent in current_layer:
            n_children = random.randint(2, branching_factor)
            children = []
            for _ in range(n_children):
                child_name = f'Node_{d+1}_{node_ctr}'
                children.append(child_name)
                node_ctr += 1
            structure[parent] = children
            next_layer.extend(children)
        current_layer = next_layer
    return structure

structure = generate_random_hierarchy()
h = Hierarchy(structure)
print(f"Hierarchy: {h.n_nodes} nodes, {h.m_bottom} bottom nodes.")

n_train = 500
n_test = 100
n_total = n_train + n_test
n_t_block = 10

gen = HierarchicalDataGenerator(h, n_timesteps=n_total)

S_sp = h.get_summing_matrix()
S_total = baselines.build_spatiotemporal_s(S_sp, n_t_block)

def get_dataset(heteroscedastic=False):
    Y_true = gen.generate_ground_truth()
    E = gen.generate_spatiotemporal_noise(
        spatial_rho=1.5,
        temporal_ar_coefs=[0.6, 0.2],
        noise_scale=2.0,
        heteroscedastic=heteroscedastic
    )
    Y_hat = Y_true + E
    residuals_train = E[:, :n_train]
    Y_hat_test = Y_hat[:, n_train:]
    Y_true_test = Y_true[:, n_train:]
    return residuals_train, Y_hat_test, Y_true_test

class BaselineReconciler:
    def __init__(self, method, S_total):
        self.method = method
        self.S = S_total
        self.P = None
        
    def fit(self, residuals=None):
        S = self.S
        if self.method == 'OLS':
            S_pinv = pinv(S)
            self.P = S @ S_pinv
        elif self.method == 'MinT_Sample':
            n_samples = residuals.shape[0]
            W = (residuals.T @ residuals) / n_samples
            W += 1e-8 * np.eye(W.shape[0])
            W_inv = inv(W)
            STS_inv = inv(S.T @ W_inv @ S)
            self.P = S @ STS_inv @ S.T @ W_inv
        elif self.method == 'MinT_Shrink':
            n_samples = residuals.shape[0]
            emp_cov = (residuals.T @ residuals) / n_samples
            W_inv = np.diag(1.0 / (np.diag(emp_cov) + 1e-8))
            STS_inv = inv(S.T @ W_inv @ S)
            self.P = S @ STS_inv @ S.T @ W_inv
            
    def reconcile(self, y_hat_flat):
        return self.P @ y_hat_flat

def run_comparison(scenario_name, residuals_train, Y_hat_test, Y_true_test):
    results = []
    
    # Flatten residuals for baselines
    train_blocks = []
    # Take non-overlapping blocks? The baseline usually takes samples.
    # original code: for i in range(0, residuals_train.shape[1] - n_t_block + 1, n_t_block)
    for i in range(0, residuals_train.shape[1] - n_t_block + 1, n_t_block):
        block = residuals_train[:, i:i+n_t_block]
        train_blocks.append(block.flatten(order='F'))
    residuals_samples = np.array(train_blocks)
    
    baseline_methods = ['OLS', 'MinT_Sample', 'MinT_Shrink']
    
    for method_name in baseline_methods:
        print(f"Running {method_name}...")
        t0 = time.time()
        model = BaselineReconciler(method_name, S_total)
        model.fit(residuals_samples)
        train_time = time.time() - t0
        
        t0 = time.time()
        mse_list = []
        for i in range(0, Y_hat_test.shape[1] - n_t_block + 1, n_t_block):
            y_hat_blk = Y_hat_test[:, i:i+n_t_block]
            y_true_blk = Y_true_test[:, i:i+n_t_block]
            y_hat_flat = y_hat_blk.flatten(order='F')
            y_tilde_flat = model.reconcile(y_hat_flat)
            y_tilde_blk = y_tilde_flat.reshape((h.n_nodes, n_t_block), order='F')
            mse_list.append(np.mean((y_tilde_blk - y_true_blk)**2))
        infer_time = time.time() - t0
        
        results.append({
            'Method': method_name,
            'Train (s)': train_time,
            'Infer (s)': infer_time,
            'MSE': np.mean(mse_list)
        })
        
    gs_methods = ['spectral', 'wavelet']
    for tm in gs_methods:
        print(f"Running GS-GLS ({tm})...")
        t0 = time.time()
        gs = GSGLS(h, temporal_method=tm)
        gs.fit(residuals_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        mse_list = []
        for i in range(0, Y_hat_test.shape[1] - n_t_block + 1, n_t_block):
            y_hat_blk = Y_hat_test[:, i:i+n_t_block]
            y_true_blk = Y_true_test[:, i:i+n_t_block]
            y_tilde_blk = gs.reconcile(y_hat_blk)
            mse_list.append(np.mean((y_tilde_blk - y_true_blk)**2))
        infer_time = time.time() - t0
        
        results.append({
            'Method': f"GS-GLS ({tm})",
            'Train (s)': train_time,
            'Infer (s)': infer_time,
            'MSE': np.mean(mse_list)
        })
        
    return pd.DataFrame(results)

print("\n--- Scenario 1: Stationary ---")
res_train, Y_hat, Y_true = get_dataset(heteroscedastic=False)
df = run_comparison("Stationary", res_train, Y_hat, Y_true)
print(df)
