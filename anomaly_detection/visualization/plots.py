import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Union

def plot_time_series_with_anomalies(data: Union[pd.DataFrame, pd.Series], 
                                   anomalies: np.ndarray,
                                   title: str = "Time Series with Anomalies",
                                   use_plotly: bool = True):  # Default to True, param kept for compatibility
    """
    Plot time series data with highlighted anomalies using Plotly.
    
    Args:
        data: Time series data (DataFrame or Series)
        anomalies: Boolean array indicating anomalies
        title: Plot title
        use_plotly: Parameter kept for backward compatibility (always uses Plotly)
    
    Returns:
        Plotly figure
    """
    if isinstance(data, pd.DataFrame):
        # Assuming the first column contains the values
        series = data.iloc[:, 0]
    else:
        series = data
    
    # Create a new figure
    fig = go.Figure()
    
    # Add time series
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name='Time Series',
        line=dict(color='blue')
    ))
    
    # Add anomalies if there are any
    if np.any(anomalies):
        anomaly_indices = np.where(anomalies)[0]
        
        # Make sure anomaly_indices doesn't exceed data length
        anomaly_indices = anomaly_indices[anomaly_indices < len(series)]
        
        if len(anomaly_indices) > 0:
            fig.add_trace(go.Scatter(
                x=series.index[anomaly_indices],
                y=series.values[anomaly_indices],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10)
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        legend_title="Legend",
        hovermode="x unified"
    )
    
    return fig