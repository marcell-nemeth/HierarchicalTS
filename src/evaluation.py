"""
Evaluation Module

Functions for evaluating hierarchical forecast reconciliation performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from hierarchy_builder import HierarchyBuilder


def check_coherency(
    forecasts: np.ndarray,
    hierarchy: HierarchyBuilder,
    tolerance: float = 1e-6
) -> Dict:
    """
    Check if forecasts satisfy hierarchical coherency constraints.
    
    A forecast is coherent if: parent_forecast = sum(children_forecasts)
    
    Parameters
    ----------
    forecasts : np.ndarray
        Forecasts with shape (n_periods, n_nodes) or (n_nodes,).
    hierarchy : HierarchyBuilder
        Hierarchy structure.
    tolerance : float
        Numerical tolerance for coherency check.
        
    Returns
    -------
    Dict
        Dictionary containing:
        - 'is_coherent': bool, whether forecasts are coherent
        - 'max_violation': float, maximum coherency violation
        - 'violations_by_node': dict, violations for each node
    """
    # Handle both 1D and 2D inputs
    input_1d = (forecasts.ndim == 1)
    if input_1d:
        forecasts = forecasts.reshape(1, -1)
    
    n_periods = forecasts.shape[0]
    violations = {}
    max_violation = 0.0
    
    # Check each non-bottom node
    for node_id in range(hierarchy.n_nodes):
        node_info = hierarchy.node_info[node_id]
        
        if not node_info['is_bottom'] and node_info['children']:
            # Get parent and children forecasts
            parent_forecast = forecasts[:, node_id]
            children_ids = node_info['children']
            children_sum = forecasts[:, children_ids].sum(axis=1)
            
            # Compute violation
            violation = np.abs(parent_forecast - children_sum)
            max_node_violation = np.max(violation)
            
            violations[node_info['name']] = {
                'max': max_node_violation,
                'mean': np.mean(violation)
            }
            
            max_violation = max(max_violation, max_node_violation)
    
    is_coherent = max_violation < tolerance
    
    return {
        'is_coherent': is_coherent,
        'max_violation': max_violation,
        'violations_by_node': violations
    }


def compute_hierarchy_metrics(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    hierarchy: HierarchyBuilder,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute forecast accuracy metrics for hierarchical forecasts.
    
    Parameters
    ----------
    forecasts : np.ndarray
        Forecasts with shape (n_periods, n_nodes).
    actuals : np.ndarray
        Actual values with shape (n_periods, n_nodes).
    hierarchy : HierarchyBuilder
        Hierarchy structure.
    metrics : Optional[List[str]]
        List of metrics to compute. Options: 'rmse', 'mae', 'mape', 'mase'.
        If None, computes all metrics.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each node and overall.
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'mape']
    
    n_nodes = hierarchy.n_nodes
    node_names = hierarchy.get_node_names()
    
    results = []
    
    for node_id in range(n_nodes):
        forecast = forecasts[:, node_id]
        actual = actuals[:, node_id]
        errors = forecast - actual
        
        node_metrics = {'node': node_names[node_id]}
        
        if 'rmse' in metrics:
            node_metrics['RMSE'] = np.sqrt(np.mean(errors ** 2))
        
        if 'mae' in metrics:
            node_metrics['MAE'] = np.mean(np.abs(errors))
        
        if 'mape' in metrics:
            # Avoid division by zero
            non_zero = actual != 0
            if np.any(non_zero):
                mape = np.mean(
                    np.abs(errors[non_zero] / actual[non_zero])
                ) * 100
            else:
                mape = np.nan
            node_metrics['MAPE'] = mape
        
        if 'mase' in metrics:
            # MASE requires scale from naive forecast
            # Using MAE of naive forecast as scale
            if len(actual) > 1:
                naive_mae = np.mean(np.abs(np.diff(actual)))
                if naive_mae > 0:
                    mase = node_metrics['MAE'] / naive_mae
                else:
                    mase = np.nan
            else:
                mase = np.nan
            node_metrics['MASE'] = mase
        
        results.append(node_metrics)
    
    df = pd.DataFrame(results)
    
    # Add overall metrics (average across all nodes)
    overall = {'node': 'Overall'}
    for metric in metrics:
        metric_upper = metric.upper()
        if metric_upper in df.columns:
            overall[metric_upper] = df[metric_upper].mean()
    results.append(overall)
    
    df = pd.DataFrame(results)
    
    return df


def reconciliation_improvement(
    base_forecasts: np.ndarray,
    reconciled_forecasts: np.ndarray,
    actuals: np.ndarray,
    hierarchy: HierarchyBuilder
) -> Dict:
    """
    Compute improvement from reconciliation.
    
    Parameters
    ----------
    base_forecasts : np.ndarray
        Base (incoherent) forecasts.
    reconciled_forecasts : np.ndarray
        Reconciled (coherent) forecasts.
    actuals : np.ndarray
        Actual values.
    hierarchy : HierarchyBuilder
        Hierarchy structure.
        
    Returns
    -------
    Dict
        Dictionary containing:
        - 'base_metrics': metrics for base forecasts
        - 'reconciled_metrics': metrics for reconciled forecasts
        - 'improvement': percentage improvement (negative = worse)
        - 'coherency_base': coherency check for base forecasts
        - 'coherency_reconciled': coherency check for reconciled forecasts
    """
    # Compute metrics
    base_metrics = compute_hierarchy_metrics(
        base_forecasts, actuals, hierarchy
    )
    reconciled_metrics = compute_hierarchy_metrics(
        reconciled_forecasts, actuals, hierarchy
    )
    
    # Compute improvement percentages
    improvement = {}
    for metric in ['RMSE', 'MAE', 'MAPE']:
        if metric in base_metrics.columns:
            base_val = base_metrics[base_metrics['node'] == 'Overall'][metric].values[0]
            rec_val = reconciled_metrics[reconciled_metrics['node'] == 'Overall'][metric].values[0]
            
            if base_val > 0:
                improvement[metric] = ((base_val - rec_val) / base_val) * 100
            else:
                improvement[metric] = 0.0
    
    # Check coherency
    coherency_base = check_coherency(base_forecasts, hierarchy)
    coherency_reconciled = check_coherency(reconciled_forecasts, hierarchy)
    
    return {
        'base_metrics': base_metrics,
        'reconciled_metrics': reconciled_metrics,
        'improvement': improvement,
        'coherency_base': coherency_base,
        'coherency_reconciled': coherency_reconciled
    }


def compute_metrics_by_level(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    hierarchy: HierarchyBuilder,
    metric: str = 'rmse'
) -> pd.DataFrame:
    """
    Compute metrics aggregated by hierarchy level.
    
    Parameters
    ----------
    forecasts : np.ndarray
        Forecasts with shape (n_periods, n_nodes).
    actuals : np.ndarray
        Actual values with shape (n_periods, n_nodes).
    hierarchy : HierarchyBuilder
        Hierarchy structure.
    metric : str
        Metric to compute ('rmse', 'mae', 'mape').
        
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics by level.
    """
    results = []
    
    for level in range(hierarchy.n_levels):
        level_nodes = hierarchy.get_level_indices(level)
        level_forecasts = forecasts[:, level_nodes]
        level_actuals = actuals[:, level_nodes]
        
        errors = level_forecasts - level_actuals
        
        if metric.lower() == 'rmse':
            value = np.sqrt(np.mean(errors ** 2))
        elif metric.lower() == 'mae':
            value = np.mean(np.abs(errors))
        elif metric.lower() == 'mape':
            non_zero = level_actuals != 0
            if np.any(non_zero):
                value = np.mean(
                    np.abs(errors[non_zero] / level_actuals[non_zero])
                ) * 100
            else:
                value = np.nan
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results.append({
            'level': level,
            'n_nodes': len(level_nodes),
            metric.upper(): value
        })
    
    return pd.DataFrame(results)
