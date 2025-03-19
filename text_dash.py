import dash
import dash_bootstrap_components as dbc
from dash import html

print("Creating test app...")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div("Hello World! This is a test dashboard.")

print("Starting server...")
if __name__ == '__main__':
    app.run_server(debug=True)