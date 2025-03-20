import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

# Add the project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.algorithms.statistical import z_score_detection, iqr_detection, moving_average_detection
from anomaly_detection.utils.data_loader import generate_synthetic_data
from anomaly_detection.visualization.plots import plot_time_series_with_anomalies

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Generate synthetic data for demonstration
df, true_anomalies = generate_synthetic_data(n_points=1000, anomaly_percentage=0.05)

# Layout 
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Anomaly Detection Dashboard", className="text-center mb-4"),
            html.P("A visual dashboard for detecting anomalies in time series data using various algorithms.", 
                   className="text-center")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Detection Settings"),
                dbc.CardBody([
                    html.Label("Algorithm:"),
                    dcc.Dropdown(
                        id='algorithm-dropdown',
                        options=[
                            {'label': 'Z-Score', 'value': 'z_score'},
                            {'label': 'IQR Method', 'value': 'iqr'},
                            {'label': 'Moving Average', 'value': 'moving_avg'}
                        ],
                        value='z_score'
                    ),
                    html.Div(id='parameter-controls', className="mt-3")
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Data Information"),
                dbc.CardBody([
                    html.P(id='data-info'),
                    html.P(id='anomaly-info')
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Time Series Visualization"),
                dbc.CardBody([
                    dcc.Graph(id='time-series-graph'),
                ])
            ]),
        ], width=9)
    ])
], fluid=True)

# Callback to update parameter controls based on selected algorithm
@app.callback(
    Output('parameter-controls', 'children'),
    Input('algorithm-dropdown', 'value')
)
def update_parameter_controls(algorithm):
    if algorithm == 'z_score':
        return [
            html.Label("Z-Score Threshold:"),
            dcc.Slider(
                id='z-threshold-slider',
                min=1.0,
                max=5.0,
                step=0.1,
                value=3.0,
                marks={i: str(i) for i in range(1, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]
    elif algorithm == 'iqr':
        return [
            html.Label("IQR Multiplier (k):"),
            dcc.Slider(
                id='iqr-slider',
                min=0.5,
                max=3.0,
                step=0.1,
                value=1.5,
                marks={i/2: str(i/2) for i in range(1, 7)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]
    elif algorithm == 'moving_avg':
        return [
            html.Label("Window Size:"),
            dcc.Slider(
                id='window-slider',
                min=5,
                max=50,
                step=5,
                value=20,
                marks={i: str(i) for i in range(5, 51, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Label("Threshold (Std Devs):"),
            dcc.Slider(
                id='ma-threshold-slider',
                min=1.0,
                max=4.0,
                step=0.1,
                value=2.0,
                marks={i: str(i) for i in range(1, 5)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ]

# Callback to update the graph based on algorithm selection and parameters
@app.callback(
    [Output('time-series-graph', 'figure'),
     Output('data-info', 'children'),
     Output('anomaly-info', 'children')],
    [Input('algorithm-dropdown', 'value'),
     Input('z-threshold-slider', 'value'),
     Input('iqr-slider', 'value'),
     Input('window-slider', 'value'),
     Input('ma-threshold-slider', 'value')]
)
def update_graph(algorithm, z_threshold, iqr_k, window_size, ma_threshold):
    # Default values for parameters not in use
    z_threshold = z_threshold or 3.0
    iqr_k = iqr_k or 1.5
    window_size = window_size or 20
    ma_threshold = ma_threshold or 2.0
    
    # Detect anomalies based on selected algorithm
    if algorithm == 'z_score':
        anomalies = z_score_detection(df['value'], threshold=z_threshold)
        method_name = f"Z-Score (threshold = {z_threshold})"
    elif algorithm == 'iqr':
        anomalies = iqr_detection(df['value'], k=iqr_k)
        method_name = f"IQR (k = {iqr_k})"
    elif algorithm == 'moving_avg':
        anomalies = moving_average_detection(df['value'], window=window_size, threshold=ma_threshold)
        method_name = f"Moving Average (window = {window_size}, threshold = {ma_threshold})"
    
    # Create plot
    fig = plot_time_series_with_anomalies(
        df, 
        anomalies, 
        title=f"Anomaly Detection using {method_name}",
        use_plotly=True
    )
    
    # Count anomalies
    anomaly_count = np.sum(anomalies)
    
    # Information text
    data_info = f"Dataset: Synthetic time series with {len(df)} points"
    anomaly_info = f"Detected anomalies: {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)"
    
    return fig, data_info, anomaly_info

if __name__ == '__main__':
    app.run_server(debug=True)