import numpy as np
import pandas as pd

def z_score_detection(data, threshold=3.0):
    """
    Detect anomalies using Z-score method.
    
    Args:
        data (numpy.ndarray or pandas.Series): Time series data
        threshold (float): Z-score threshold for anomaly detection
        
    Returns:
        numpy.ndarray: Boolean array where True indicates an anomaly
    """
    if isinstance(data, pd.Series):
        data = data.values
        
    mean = np.mean(data)
    std = np.std(data)
    
    z_scores = np.abs((data - mean) / std)
    return z_scores > threshold

def iqr_detection(data, k=1.5):
    """
    Detect anomalies using Interquartile Range (IQR) method.
    
    Args:
        data (numpy.ndarray or pandas.Series): Time series data
        k (float): Multiplier for IQR to determine threshold
        
    Returns:
        numpy.ndarray: Boolean array where True indicates an anomaly
    """
    if isinstance(data, pd.Series):
        data = data.values
        
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    return (data < lower_bound) | (data > upper_bound)

def moving_average_detection(data, window=10, threshold=2.0):
    """
    Detect anomalies using moving average and standard deviation.
    
    Args:
        data (pandas.Series): Time series data
        window (int): Window size for moving average
        threshold (float): Number of standard deviations to consider as anomaly
        
    Returns:
        numpy.ndarray: Boolean array where True indicates an anomaly
    """
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
        
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    # Calculate upper and lower bounds
    upper_bound = rolling_mean + (rolling_std * threshold)
    lower_bound = rolling_mean - (rolling_std * threshold)
    
    # Identify anomalies
    anomalies = (data > upper_bound) | (data < lower_bound)
    
    # First few points won't have a rolling average, set them to False
    anomalies.iloc[:window-1] = False
    
    return anomalies.values