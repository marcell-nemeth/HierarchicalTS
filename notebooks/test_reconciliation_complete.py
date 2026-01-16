"""
Comprehensive test for MinT reconciliation fix
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('..') / 'src'))

import numpy as np
from hierarchy_builder import HierarchyBuilder
from data_generator import SyntheticDataGenerator
from mint_reconciliation import MinTReconciler
from evaluation import check_coherency, compute_hierarchy_metrics

# Set seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("MinT Reconciliation Comprehensive Test")
print("=" * 60 + "\n")

# Create hierarchy [1, 3, 9]
hierarchy = HierarchyBuilder([1, 3, 9])
print(f"✓ Created hierarchy: {hierarchy.n_nodes} nodes ({hierarchy.n_bottom} bottom)")

# Generate data
data_gen = SyntheticDataGenerator(hierarchy, seed=42)
historical_data = data_gen.generate_coherent_series(
    n_periods=120,
    trend_coef=0.05,
    seasonal_period=12,
    seasonal_amplitude=3.0,
    noise_std=1.5,
    ar_coef=0.6
)

# Generate incoherent base forecasts
base_forecasts, true_values = data_gen.generate_base_forecasts(
    historical_data,
    n_forecast=12,
    incoherency_std=2.0
)

# Generate forecast errors
forecast_errors = data_gen.generate_forecast_errors(n_samples=100, error_std=2.0)

# Initialize and fit reconciler
reconciler = MinTReconciler(hierarchy, method='sample')
reconciler.fit(forecast_errors)
print("✓ Fitted MinT reconciler\n")

# Test 1: Check projection matrix property PS = S
print("Test 1: Projection Matrix Property (PS = S)")
print("-" * 60)
P = reconciler.get_projection_matrix()
S = hierarchy.S
PS = P @ S

max_diff = np.max(np.abs(PS - S))
ps_equals_s = np.allclose(PS, S, atol=1e-6)

print(f"  Max |PS - S|: {max_diff:.2e}")
print(f"  PS equals S? {ps_equals_s}")
if ps_equals_s:
    print("  ✓ PASS: Projection matrix preserves aggregation structure\n")
else:
    print(f"  ✗ FAIL: PS !="}")
    print()

# Test 2: Check base forecasts coherency
print("Test 2: Base Forecasts Coherency")
print("-" * 60)
base_coherency = check_coherency(base_forecasts.values, hierarchy)
print(f"  Is coherent: {base_coherency['is_coherent']}")
print(f"  Max violation: {base_coherency['max_violation']:.2e}")
if not base_coherency['is_coherent']:
    print("  ✓ PASS: Base forecasts are incoherent (as expected)\n")
else:
    print("  ⚠ WARNING: Base forecasts are already coherent\n")

# Test 3: Reconcile and check coherency
print("Test 3: Reconciled Forecasts Coherency")
print("-" * 60)
reconciled = reconciler.reconcile(base_forecasts.values)
rec_coherency = check_coherency(reconciled, hierarchy)

print(f"  Is coherent: {rec_coherency['is_coherent']}")
print(f"  Max violation: {rec_coherency['max_violation']:.2e}")

if rec_coherency['is_coherent']:
    print("  ✓ PASS: Reconciled forecasts ARE coherent!\n")
else:
    print("  ✗ FAIL: Reconciled forecasts still incoherent\n")

# Test 4: Check error reduction
print("Test 4: Forecast Error Reduction")
print("-" * 60)

base_metrics = compute_hierarchy_metrics(base_forecasts.values, true_values.values, hierarchy)
rec_metrics = compute_hierarchy_metrics(reconciled, true_values.values, hierarchy)

base_rmse = base_metrics[base_metrics['node'] == 'Overall']['RMSE'].values[0]
rec_rmse = rec_metrics[rec_metrics['node'] == 'Overall']['RMSE'].values[0]
improvement = ((base_rmse - rec_rmse) / base_rmse * 100)

print(f"  Base RMSE:       {base_rmse:.4f}")
print(f"  Reconciled RMSE: {rec_rmse:.4f}")
print(f"  Improvement:     {improvement:.2f}%")

if improvement > 0:
    print("  ✓ PASS: Reconciliation REDUCED error!\n")
else:
    print("  ✗ FAIL: Reconciliation INCREASED error\n")

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)

all_pass = ps_equals_s and rec_coherency['is_coherent'] and improvement > 0

if all_pass:
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("\nMinT reconciliation is working correctly:")
    print("  • Projection matrix preserves aggregation structure")
    print("  • Reconciled forecasts are coherent")
    print("  • Reconciliation reduces forecast error")
else:
    print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("\nIssues found:")
    if not ps_equals_s:
        print("  • Projection matrix doesn't preserve aggregation")
    if not rec_coherency['is_coherent']:
        print("  • Reconciled forecasts are NOT coherent")
    if improvement <= 0:
        print("  • Reconciliation doesn't reduce error")

print("=" * 60)
