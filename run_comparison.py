import numpy as np
import pandas as pd
import time
import random
import sys

# Add path for local modules
sys.path.append('.')

from hierarchy import Hierarchy
from data_generator import HierarchicalDataGenerator
from gs_gls import GSGLS
import baselines

# --- Random Hierarchy Generation ---
def generate_random_hierarchy(depth=3, branching_factor=3):
    """
    Generates a random tree structure.
    """
    structure = {}
    
    current_layer = ['Total']
    all_nodes = set(['Total'])
    
    node_ctr = 1
    
    for d in range(depth):
        next_layer = []
        for parent in current_layer:
            # Random branching
            n_children = random.randint(2, branching_factor)
            children = []
            for _ in range(n_children):
                child_name = f"Node_{d+1}_{node_ctr}"
                children.append(child_name)
                all_nodes.add(child_name)
                node_ctr += 1
            
            structure[parent] = children
            next_layer.extend(children)
        current_layer = next_layer
        
    return structure

# --- Robust Incoherence Check ---
def check_full_coherence(y, hierarchy):
    """
    Checks ALL parent-child constraints.
    y: (n_nodes x n_timesteps)
    """
    max_error = 0.0
    
    for parent, children in hierarchy.structure.items():
        parent_idx = hierarchy.node_to_idx[parent]
        child_indices = [hierarchy.node_to_idx[c] for c in children]
        
        # Sum children
        agg_children = np.sum(y[child_indices, :], axis=0)
        parent_val = y[parent_idx, :]
        
        err = np.max(np.abs(parent_val - agg_children))
        if err > max_error:
            max_error = err
    
    return max_error

# --- Main Execution ---

# Configuration for HARDER task
n_timesteps_train = 600
n_timesteps_test = 100
n_t_block = 10
# Large random hierarchy
print("Generating Random Hierarchy...")
random.seed(42)
structure = generate_random_hierarchy(depth=3, branching_factor=4)
h = Hierarchy(structure)
print(f"Hierarchy Created: {h.n_nodes} nodes (Bottom: {h.m_bottom})")

print("Generating Complex Data (Bias + Heteroscedasticity)...")
gen = HierarchicalDataGenerator(h, n_timesteps=n_timesteps_train + n_timesteps_test)
Y_true_all = gen.generate_ground_truth()

# Add COMPLEX noise
# Bias: 2.0 (Significant structural error)
# Heteroscedasticity: True (Variance grows)
noise_all = gen.generate_spatiotemporal_noise(
    spatial_rho=1.5, 
    temporal_ar_coefs=[0.6, 0.2], 
    noise_scale=2.0,
    bias_scale=3.0,
    heteroscedastic=True
)
Y_hat_all = Y_true_all + noise_all

# Split Train/Test
residuals_train = noise_all[:, :n_timesteps_train]
Y_hat_test = Y_hat_all[:, n_timesteps_train:]
Y_true_test = Y_true_all[:, n_timesteps_train:]

print(f"Train/Test Shapes: {residuals_train.shape}, {Y_hat_test.shape}")

# Prepare Baselines
S_sp = h.get_summing_matrix()
S_total = baselines.build_spatiotemporal_s(S_sp, n_t_block)

# Sample covariance for MinT
train_blocks = []
for i in range(0, residuals_train.shape[1] - n_t_block + 1, n_t_block):
    block = residuals_train[:, i:i+n_t_block]
    flat_block = block.flatten(order='F')
    train_blocks.append(flat_block)

residuals_samples = np.array(train_blocks)
n_samples_mint = residuals_samples.shape[0]
n_dim_mint = residuals_samples.shape[1]
print(f"MinT Samples: {n_samples_mint}, Dimension: {n_dim_mint}")

if n_dim_mint > n_samples_mint:
    print("WARNING: Dimension > Samples. MinT (Sample) will be singular/ill-conditioned.")

# Fit GS-GLS
print("Fitting GS-GLS...")
start_t = time.time()
gs_estimator = GSGLS(h)
try:
    gs_estimator.fit(residuals_train)
    gs_time = time.time() - start_t
    print(f"GS-GLS Fitted in {gs_time:.4f}s")
except Exception as e:
    print(f"GS-GLS Failed: {e}")

# Evaluation Loop
metrics = {'Method': [], 'MSE': [], 'MAE': [], 'Max_Incoherence': [], 'Time': []}

def eval_method(name, func, needs_train_data=False):
    print(f"Evaluating {name}...")
    mse_list = []
    mae_list = []
    incoh_list = []
    t_list = []
    
    for i in range(0, Y_hat_test.shape[1] - n_t_block + 1, n_t_block):
        y_hat_block = Y_hat_test[:, i:i+n_t_block]
        y_true_block = Y_true_test[:, i:i+n_t_block]
        y_hat_flat = y_hat_block.flatten(order='F')
        
        start_time = time.time()
        try:
            if name == 'GS-GLS':
                y_tilde_block = gs_estimator.reconcile(y_hat_block)
            else:
                if needs_train_data:
                    y_tilde_flat = func(y_hat_flat, residuals_samples, S_total)
                else:
                    y_tilde_flat = func(y_hat_flat, S_total)
                y_tilde_block = y_tilde_flat.reshape((h.n_nodes, n_t_block), order='F')
            
            end_time = time.time()
            
            # Accuracy
            mse = np.mean((y_tilde_block - y_true_block)**2)
            mae = np.mean(np.abs(y_tilde_block - y_true_block))
            
            # Full Structural Coherence Check
            max_incoh = check_full_coherence(y_tilde_block, h)
            
            mse_list.append(mse)
            mae_list.append(mae)
            incoh_list.append(max_incoh)
            t_list.append(end_time - start_time)
            
        except np.linalg.LinAlgError:
            print(f"  {name} Singularity Error")
            return np.nan, np.nan, np.nan, np.nan
        except Exception as e:
            print(f"  {name} Error: {e}")
            break
            
    return np.mean(mse_list), np.mean(mae_list), np.mean(incoh_list), np.mean(t_list)

# Run
methods = [
    ('OLS (Identity)', baselines.ols_identity, False),
    ('MinT (Sample)', baselines.mint_sample, True),
    ('MinT (Shrinkage)', baselines.mint_shrink, True),
    ('GS-GLS', None, False)
]

for name, func, needs_train in methods:
    m, ma, inc, t = eval_method(name, func, needs_train)
    metrics['Method'].append(name)
    metrics['MSE'].append(m)
    metrics['MAE'].append(ma)
    metrics['Max_Incoherence'].append(inc)
    metrics['Time'].append(t)

df_res = pd.DataFrame(metrics)
print("\nFinal Results (Hard Dataset):")
print(df_res.to_string())

print("\n--- Theory Explanation ---")
print("Why is Incoherence effectively zero?")
print("All methods evaluated are 'Projection' methods of the form y~ = S P y^.")
print("The output range of matrix S is, by definition, the space of coherent forecasts.")
print("Therefore, any result computed via multiplication by S satisfies the aggregation constraints exactly (within float precision).")
