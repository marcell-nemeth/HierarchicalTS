"""
Debug script to check MinT projection matrix properties
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('..') / 'src'))

import numpy as np
from hierarchy_builder import HierarchyBuilder
from data_generator import SyntheticDataGenerator
from mint_reconciliation import MinTReconciler
from evaluation import check_coherency

# Set seed
np.random.seed(42)

# Create hierarchy
hierarchy = HierarchyBuilder([1, 3, 9])
print(f"Hierarchy: {hierarchy.n_nodes} nodes, {hierarchy.n_bottom} bottom-level")
print(f"S shape: {hierarchy.S.shape}\n")

# Generate forecast errors
data_gen = SyntheticDataGenerator(hierarchy, seed=42)
forecast_errors = data_gen.generate_forecast_errors(n_samples=100, error_std=2.0)
print(f"Forecast errors shape: {forecast_errors.shape}\n")

# Fit reconciler
reconciler = MinTReconciler(hierarchy, method='sample')
reconciler.fit(forecast_errors)

# Check projection matrix property: PS = S
P = reconciler.get_projection_matrix()
S = hierarchy.S
PS = P @ S

print(f"P shape: {P.shape}")
print(f"S shape: {S.shape}")
print(f"PS shape: {PS.shape}")
print(f"\nMax |PS - S|: {np.max(np.abs(PS - S))}")
print(f"PS equals S (within tolerance)? {np.allclose(PS, S)}\n")

# Generate test forecasts and reconcile
historical = data_gen.generate_coherent_series(n_periods=120)
base_fc, true_vals = data_gen.generate_base_forecasts(historical, n_forecast=12)

print("Base forecasts coherency:")
base_coherency = check_coherency(base_fc.values, hierarchy)
print(f"  Is coherent: {base_coherency['is_coherent']}")
print(f"  Max violation: {base_coherency['max_violation']:.2e}\n")

# Reconcile
reconciled = reconciler.reconcile(base_fc.values)

print("Reconciled forecasts coherency:")
rec_coherency = check_coherency(reconciled, hierarchy)
print(f"  Is coherent: {rec_coherency['is_coherent']}")
print(f"  Max violation: {rec_coherency['max_violation']:.2e}")

if not rec_coherency['is_coherent']:
    print("\n✗ ISSUE: Reconciled forecasts are NOT coherent!")
    print("This indicates a bug in the projection matrix computation.")
else:
    print("\n✓ SUCCESS: Reconciled forecasts are coherent!")
