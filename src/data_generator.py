"""
Synthetic Data Generator Module

This module generates synthetic hierarchical time series data with 
controllable temporal patterns and incoherency levels.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from hierarchy_builder import HierarchyBuilder


class SyntheticDataGenerator:
    """
    Generates synthetic hierarchical time series data.
    
    Features:
    - Configurable temporal patterns (trend, seasonality, AR)
    - Controllable incoherency for testing reconciliation methods
    - Coherent historical data and incoherent forecasts
    """
    
    def __init__(
        self,
        hierarchy: HierarchyBuilder,
        seed: Optional[int] = None
    ):
        """
        Initialize data generator.
        
        Parameters
        ----------
        hierarchy : HierarchyBuilder
            Hierarchy structure.
        seed : Optional[int]
            Random seed for reproducibility.
        """
        self.hierarchy = hierarchy
        self.rng = np.random.RandomState(seed)
        
    def generate_coherent_series(
        self,
        n_periods: int,
        trend_coef: float = 0.1,
        seasonal_period: int = 12,
        seasonal_amplitude: float = 2.0,
        noise_std: float = 1.0,
        ar_coef: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate coherent historical time series.
        
        Data is generated bottom-up: create bottom-level series with
        temporal patterns, then aggregate to upper levels using summing matrix.
        
        Parameters
        ----------
        n_periods : int
            Number of time periods.
        trend_coef : float
            Linear trend coefficient.
        seasonal_period : int
            Seasonality period (e.g., 12 for monthly).
        seasonal_amplitude : float
            Amplitude of seasonal component.
        noise_std : float
            Standard deviation of noise.
        ar_coef : float
            AR(1) coefficient for temporal dependence.
            
        Returns
        -------
        pd.DataFrame
            Coherent time series data with shape (n_periods, n_nodes).
        """
        n_bottom = self.hierarchy.n_bottom
        
        # Generate bottom-level series
        bottom_series = np.zeros((n_periods, n_bottom))
        
        for i in range(n_bottom):
            # Base level and random variation
            base_level = self.rng.uniform(10, 50)
            
            # Initialize AR process
            ar_component = np.zeros(n_periods)
            ar_component[0] = self.rng.normal(0, noise_std)
            
            for t in range(1, n_periods):
                ar_component[t] = (
                    ar_coef * ar_component[t-1] + 
                    self.rng.normal(0, noise_std)
                )
            
            # Trend component
            trend = trend_coef * np.arange(n_periods)
            
            # Seasonal component
            seasonal = seasonal_amplitude * np.sin(
                2 * np.pi * np.arange(n_periods) / seasonal_period
            )
            
            # Combine components
            bottom_series[:, i] = (
                base_level + trend + seasonal + ar_component
            )
        
        # Aggregate to all levels using summing matrix
        all_series = bottom_series @ self.hierarchy.S.T
        
        # Create DataFrame
        node_names = self.hierarchy.get_node_names()
        df = pd.DataFrame(
            all_series,
            columns=node_names,
            index=pd.RangeIndex(n_periods, name='period')
        )
        
        return df
    
    def generate_base_forecasts(
        self,
        historical_data: pd.DataFrame,
        n_forecast: int,
        incoherency_std: float = 2.0,
        forecast_noise_std: float = 1.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate incoherent base forecasts.
        
        Forecasts are created by:
        1. Extrapolating from historical data (simple method)
        2. Adding independent noise to each node to create incoherency
        
        Parameters
        ----------
        historical_data : pd.DataFrame
            Historical time series data.
        n_forecast : int
            Number of forecast periods.
        incoherency_std : float
            Standard deviation of incoherency noise added to each node.
        forecast_noise_std : float
            Standard deviation of general forecast noise.
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (base_forecasts, true_values)
            - base_forecasts: Incoherent base forecasts
            - true_values: Coherent "true" future values for evaluation
        """
        n_nodes = self.hierarchy.n_nodes
        
        # Generate true coherent future values (for evaluation)
        # Use simple extrapolation from last observations
        last_window = historical_data.values[-12:, :]  # Last 12 periods
        mean_level = np.mean(last_window, axis=0)
        trend = (last_window[-1, :] - last_window[0, :]) / 12
        
        # Generate coherent bottom-level forecasts
        n_bottom = self.hierarchy.n_bottom
        bottom_nodes = [
            i for i, info in self.hierarchy.node_info.items()
            if info['is_bottom']
        ]
        
        true_bottom = np.zeros((n_forecast, n_bottom))
        for i, node_id in enumerate(bottom_nodes):
            for h in range(n_forecast):
                # Simple trend extrapolation with small noise
                true_bottom[h, i] = (
                    mean_level[node_id] + 
                    trend[node_id] * (h + 1) +
                    self.rng.normal(0, forecast_noise_std * 0.5)
                )
        
        # Aggregate to get coherent values at all levels
        true_values = true_bottom @ self.hierarchy.S.T
        
        # Create incoherent base forecasts
        # Start from true values and add independent noise to each node
        base_forecasts = true_values.copy()
        for node_id in range(n_nodes):
            # Add incoherency noise
            incoherency_noise = self.rng.normal(
                0, incoherency_std, size=n_forecast
            )
            base_forecasts[:, node_id] += incoherency_noise
        
        # Convert to DataFrames
        node_names = self.hierarchy.get_node_names()
        forecast_index = pd.RangeIndex(
            len(historical_data),
            len(historical_data) + n_forecast,
            name='period'
        )
        
        base_forecasts_df = pd.DataFrame(
            base_forecasts,
            columns=node_names,
            index=forecast_index
        )
        
        true_values_df = pd.DataFrame(
            true_values,
            columns=node_names,
            index=forecast_index
        )
        
        return base_forecasts_df, true_values_df
    
    def generate_forecast_errors(
        self,
        n_samples: int,
        error_std: float = 2.0
    ) -> np.ndarray:
        """
        Generate forecast error samples for covariance estimation.
        
        This simulates historical forecast errors that would be used
        to estimate the covariance matrix for MinT reconciliation.
        
        Parameters
        ----------
        n_samples : int
            Number of error samples to generate.
        error_std : float
            Standard deviation of forecast errors.
            
        Returns
        -------
        np.ndarray
            Forecast errors with shape (n_samples, n_nodes).
        """
        n_nodes = self.hierarchy.n_nodes
        
        # Generate correlated errors
        # Bottom-level errors
        n_bottom = self.hierarchy.n_bottom
        bottom_errors = self.rng.normal(
            0, error_std, size=(n_samples, n_bottom)
        )
        
        # Aggregate errors using summing matrix
        # This ensures errors follow the hierarchical structure
        # (upper-level errors = sum of corresponding bottom-level errors)
        all_errors = bottom_errors @ self.hierarchy.S.T
        
        return all_errors
