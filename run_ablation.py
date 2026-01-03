
import numpy as np
import pandas as pd
import time
import random
import sys
import os

sys.path.append(os.getcwd())

from hierarchy import Hierarchy
from data_generator import HierarchicalDataGenerator
from gs_gls import GSGLS
import baselines
from fair_baselines import BaselineReconciler

def generate_hierarchy(depth, branching_factor):
    np.random.seed(42)
    random.seed(42)
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
    return Hierarchy(structure)

def run_experiment(config):
    name = config['name']
    depth = config['depth']
    branch = config['branch']
    n_total = config['n_total']
    n_train = int(n_total * 0.8)
    n_t_block = 10
    
    print(f"\nrunning Scenario: {name} (Depth={depth}, Branch={branch}, T={n_total})")
    
    h = generate_hierarchy(depth, branch)
    print(f"  Hierarchy Nodes: {h.n_nodes}")
    
    gen = HierarchicalDataGenerator(h, n_timesteps=n_total)
    Y_true = gen.generate_ground_truth()
    E = gen.generate_spatiotemporal_noise(spatial_rho=1.5, temporal_ar_coefs=[0.5], noise_scale=1.0)
    Y_hat = Y_true + E
    
    residuals_train = E[:, :n_train]
    Y_hat_test = Y_hat[:, n_train:]
    Y_true_test = Y_true[:, n_train:]
    
    S_sp = h.get_summing_matrix()
    S_total = baselines.build_spatiotemporal_s(S_sp, n_t_block)
    
    train_blocks = []
    for i in range(0, residuals_train.shape[1] - n_t_block + 1, n_t_block):
        block = residuals_train[:, i:i+n_t_block]
        train_blocks.append(block.flatten(order='F'))
    residuals_samples = np.array(train_blocks)
    
    results = []
    
    methods = [
        ('MinT_Shrink', 'baseline', None),
        ('GS-GLS (Spec)', 'gs', 'spectral'),
    ]
    
    for label, mode, submode in methods:
        try:
            t0 = time.time()
            if mode == 'baseline':
                model = BaselineReconciler(label, S_total)
                model.fit(residuals_samples)
            else:
                model = GSGLS(h, temporal_method=submode)
                model.fit(residuals_train)
            train_time = time.time() - t0
            
            t0 = time.time()
            mse_list = []
            for i in range(0, Y_hat_test.shape[1] - n_t_block + 1, n_t_block):
                y_hat_blk = Y_hat_test[:, i:i+n_t_block]
                y_true_blk = Y_true_test[:, i:i+n_t_block]
                
                if mode == 'baseline':
                    y_hat_flat = y_hat_blk.flatten(order='F')
                    y_tilde_flat = model.reconcile(y_hat_flat)
                    y_tilde_blk = y_tilde_flat.reshape((h.n_nodes, n_t_block), order='F')
                else:
                    y_tilde_blk = model.reconcile(y_hat_blk)
                    
                mse_list.append(np.mean((y_tilde_blk - y_true_blk)**2))
            infer_time = time.time() - t0
            
            results.append({
                'Scenario': name,
                'Nodes': h.n_nodes,
                'T': n_total,
                'Method': label,
                'Total Time': train_time + infer_time,
                'MSE': np.mean(mse_list)
            })
        except Exception as e:
            print(f"Error in {label}: {e}")
            
    return results

scenarios = [
    {'name': 'Small_Short', 'depth': 3, 'branch': 3, 'n_total': 300},
    {'name': 'Medium_Short', 'depth': 4, 'branch': 3, 'n_total': 300},
]

all_results = []
for sc in scenarios:
    res = run_experiment(sc)
    all_results.extend(res)

print(pd.DataFrame(all_results))
