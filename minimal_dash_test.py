import time

import dash
from dash import html

print("Starting minimal Dash test...")
time.sleep(1)  # Add a small delay to see the progress
print("Creating Dash app...")
app = dash.Dash(__name__)
print("Setting up layout...")
app.layout = html.Div("Hello World - Minimal Dash Test")
print("Setup complete, starting server...")

if __name__ == '__main__':
    app.run(debug=False)  # Using debug=False for faster startup