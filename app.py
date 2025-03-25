import os
import sys
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_detection.algorithms.statistical import (iqr_detection,
                                                     moving_average_detection,
                                                     z_score_detection)
from anomaly_detection.utils.data_loader import generate_synthetic_data
from anomaly_detection.visualization.plots import plot_time_series_with_anomalies

print("Initializing Dash app...")
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Generate synthetic data
print("Generating synthetic data...")
df, true_anomalies = generate_synthetic_data(n_points=400, anomaly_percentage=0.05)
print(f"Data shape: {df.shape}")
print(f"Data sample:\n{df.head()}")
print("Data generated successfully.")

# Create the layout - IMPORTANT CHANGE: Include sliders directly in the layout
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
                    html.Div([
                        # Z-Score Controls
                        html.Div([
                            html.Label("Z-Score Threshold:"),
                            dcc.Slider(
                                id='z-threshold-slider',
                                min=1.0, max=5.0, step=0.1, value=3.0,
                                marks={i: str(i) for i in range(1, 6)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], id='z-score-controls', style={'display': 'block'}),
                        
                        # IQR Controls
                        html.Div([
                            html.Label("IQR Multiplier (k):"),
                            dcc.Slider(
                                id='iqr-slider',
                                min=0.5, max=3.0, step=0.1, value=1.5,
                                marks={i/2: str(i/2) for i in range(1, 7)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], id='iqr-controls', style={'display': 'none'}),
                        
                        # Moving Average Controls
                        html.Div([
                            html.Label("Window Size:"),
                            dcc.Slider(
                                id='window-slider',
                                min=5, max=50, step=5, value=20,
                                marks={i: str(i) for i in range(5, 51, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Label("Threshold (Std Devs):"),
                            dcc.Slider(
                                id='ma-threshold-slider',
                                min=1.0, max=4.0, step=0.1, value=2.0,
                                marks={i: str(i) for i in range(1, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], id='ma-controls', style={'display': 'none'})
                    ], className="mt-3")
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

# Callback to toggle visibility of parameter controls
@app.callback(
    [Output('z-score-controls', 'style'),
     Output('iqr-controls', 'style'),
     Output('ma-controls', 'style')],
    [Input('algorithm-dropdown', 'value')]
)
def update_control_visibility(algorithm):
    z_style = {'display': 'block' if algorithm == 'z_score' else 'none'}
    iqr_style = {'display': 'block' if algorithm == 'iqr' else 'none'}
    ma_style = {'display': 'block' if algorithm == 'moving_avg' else 'none'}
    return z_style, iqr_style, ma_style

# Callback to update the graph
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
    print(f"update_graph called with: algorithm={algorithm}, z_threshold={z_threshold}, "
          f"iqr_k={iqr_k}, window_size={window_size}, ma_threshold={ma_threshold}")
    
    if algorithm is None:
        algorithm = 'z_score'
    
    # Detect anomalies based on algorithm
    if algorithm == 'z_score':
        threshold = z_threshold if z_threshold is not None else 3.0
        anomalies = z_score_detection(df['value'], threshold=threshold)
        method_name = f"Z-Score (threshold = {threshold})"
    elif algorithm == 'iqr':
        k = iqr_k if iqr_k is not None else 1.5
        anomalies = iqr_detection(df['value'], k=k)
        method_name = f"IQR (k = {k})"
    elif algorithm == 'moving_avg':
        window = window_size if window_size is not None else 20
        threshold = ma_threshold if ma_threshold is not None else 2.0
        anomalies = moving_average_detection(df['value'], window=window, threshold=threshold)
        method_name = f"Moving Average (window = {window}, threshold = {threshold})"
    else:
        # Fallback
        threshold = z_threshold if z_threshold is not None else 3.0
        anomalies = z_score_detection(df['value'], threshold=threshold)
        method_name = f"Z-Score (threshold = {threshold})"
    
    # Create a default empty figure in case of errors
    default_fig = go.Figure()
    default_fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers'))
    default_fig.update_layout(title="No data available")
    
    try:
        # Create plot with error handling
        fig = plot_time_series_with_anomalies(
            df, 
            anomalies, 
            title=f"Anomaly Detection using {method_name}",
            use_plotly=True
        )
        print(f"Figure created successfully with {len(fig.data)} traces")
    except Exception as e:
        print(f"Error creating figure: {e}")
        fig = default_fig
    
    # Count anomalies
    anomaly_count = np.sum(anomalies)
    
    # Information text
    data_info = f"Dataset: Synthetic time series with {len(df)} points"
    anomaly_info = f"Detected anomalies: {anomaly_count} ({anomaly_count/len(df)*100:.2f}%)"
    
    return fig, data_info, anomaly_info

# Initialize with a default figure
def create_initial_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['value'],
        mode='lines',
        name='Time Series'
    ))
    fig.update_layout(title="Initial Time Series Data")
    return fig

# Run the app
if __name__ == '__main__':
    print("Starting Dash server...")
    app.run(debug=False)