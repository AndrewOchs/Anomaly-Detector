from typing import Tuple, Union

import numpy as np
import pandas as pd


def load_csv_data(filepath: str, datetime_column: str = None, 
                 value_column: str = None) -> pd.DataFrame:
    """
    Load time series data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        datetime_column (str, optional): Name of the datetime column
        value_column (str, optional): Name of the value column to analyze
        
    Returns:
        pandas.DataFrame: Loaded and preprocessed data
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # If datetime column specified, parse and set as index
    if datetime_column and datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.set_index(datetime_column)
    
    # If value column specified, ensure it exists
    if value_column and value_column in df.columns:
        return df[[value_column]]
    
    return df

def generate_synthetic_data(n_points: int = 1000, 
                           anomaly_percentage: float = 0.05) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic time series data with anomalies for testing.
    
    Args:
        n_points (int): Number of data points to generate
        anomaly_percentage (float): Percentage of anomalies to include
        
    Returns:
        Tuple: (DataFrame with time series data, array with anomaly indicators)
    """
    # Generate date range
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='h')
    
    # Generate normal time series with weekly seasonality
    hours = np.arange(n_points)
    weekly_pattern = np.sin(2 * np.pi * hours / (7 * 24))
    daily_pattern = np.sin(2 * np.pi * hours / 24)
    
    # Add some noise
    noise = np.random.normal(0, 0.5, n_points)
    
    # Combine patterns
    values = 10 + 3 * weekly_pattern + 2 * daily_pattern + noise
    
    # Create DataFrame
    df = pd.DataFrame({'timestamp': dates, 'value': values})
    df = df.set_index('timestamp')
    
    # Generate random anomalies
    n_anomalies = int(n_points * anomaly_percentage)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    
    # Add anomalies to the data
    anomaly_multiplier = np.random.choice([2, -2, 3, -3], n_anomalies)
    df.iloc[anomaly_indices, 0] += 5 * anomaly_multiplier
    
    # Create ground truth array
    ground_truth = np.zeros(n_points, dtype=bool)
    ground_truth[anomaly_indices] = True
    
    return df, ground_truth