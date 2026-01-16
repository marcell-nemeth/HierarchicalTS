"""
Quick test script to verify MinT reconciliation fix
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path('..') / 'src'))

import numpy as np
from hierarchy_builder import HierarchyBuilder
from data_generator import SyntheticDataGenerator
from mint_reconciliation import MinTReconciler
from evaluation import compute_hierarchy_metrics

# Set seed for reproducibility
np.random.seed(42)

# Create hierarchy [1, 3, 9]
hierarchy = HierarchyBuilder([1, 3, 9])
print(f"Created hierarchy with {hierarchy.n_nodes} nodes ({hierarchy.n_bottom} bottom-level)")

# Generate data
data_gen = SyntheticDataGenerator(hierarchy, seed=42)
n_historical = 120
n_forecast = 12

# Generate coherent historical data
historical_data = data_gen.generate_coherent_series(
    n_periods=n_historical,
    trend_coef=0.05,
    seasonal_period=12,
    seasonal_amplitude=3.0,
    noise_std=1.5,
    ar_coef=0.6
)

# Generate incoherent base forecasts and true values
base_forecasts, true_values = data_gen.generate_base_forecasts(
    historical_data,
    n_forecast=n_forecast,
    incoherency_std=2.0
)

# Generate forecast errors for covariance estimation
forecast_errors = data_gen.generate_forecast_errors(
    n_samples=100,
    error_std=2.0
)

# Initialize and fit MinT reconciler
reconciler = MinTReconciler(hierarchy, method='sample')
reconciler.fit(forecast_errors)

# Reconcile forecasts
reconciled_forecasts = reconciler.reconcile(base_forecasts.values)

# Compute metrics
base_metrics = compute_hierarchy_metrics(
    base_forecasts.values, 
    true_values.values, 
    hierarchy
)
reconciled_metrics = compute_hierarchy_metrics(
    reconciled_forecasts, 
    true_values.values, 
    hierarchy
)

# Get overall metrics
base_rmse = base_metrics[base_metrics['node'] == 'Overall']['RMSE'].values[0]
rec_rmse = reconciled_metrics[reconciled_metrics['node'] == 'Overall']['RMSE'].values[0]

print(f"\nBase RMSE: {base_rmse:.4f}")
print(f"Reconciled RMSE: {rec_rmse:.4f}")
print(f"Improvement: {((base_rmse - rec_rmse) / base_rmse * 100):.2f}%")

if rec_rmse < base_rmse:
    print("\n✓ SUCCESS: Reconciliation REDUCED error as expected!")
else:
    print("\n✗ FAILED: Reconciliation INCREASED error (bug still present)")
