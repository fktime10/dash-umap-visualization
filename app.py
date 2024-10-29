# app.py

import os
import json
import pickle
import logging
from typing import List
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data(file_path: str = 'data/processed_data.pkl') -> pd.DataFrame:
    """
    Load the precomputed processed data from a pickle file.
    """
    if not os.path.exists(file_path):
        logger.error(f"Processed data file not found: {file_path}")
        raise FileNotFoundError(f"Processed data file not found: {file_path}")

    logger.info(f"Loading processed data from {file_path}...")
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    logger.info("Processed data loaded successfully.")
    return df

def create_plotly_figure(df: pd.DataFrame) -> px.scatter:
    """
    Create a Plotly scatter plot from the DataFrame.
    """
    # Define color mapping: base colors with varying intensities
    color_mapping = {
        'Eval Prompt': '#ADD8E6',            # Light Blue
        'Eval Code': '#00008B',              # Dark Blue
        'User Intent (Train)': '#FFB6C1',     # Light Pink
        'Assistant Code (Train)': '#C71585',  # Dark Pink
        'TI_COS Intent': '#90EE90',          # Light Green
        'TI_COS Code': '#006400',            # Dark Green
        'User Intent (Test)': '#FFFFE0',      # Light Yellow
        'Assistant Code (Test)': '#FFD700',   # Dark Yellow/Gold
        'Project Intent': '#E6E6FA',         # Light Lavender
        'Project Code': '#800080'             # Dark Purple
    }

    fig = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='Label',
        title='2D UMAP Projection of Text Data',
        labels={'color': 'Data Label'},
        opacity=0.8,
        hover_data={
            'UMAP1': False,
            'UMAP2': False
        },
        color_discrete_map=color_mapping,
        custom_data=['Content', 'Label']  # Include content and label in customdata
    )

    # Update marker properties for better visibility
    fig.update_traces(marker=dict(line=dict(width=0.1, color='black')), marker_size=5)

    # Update layout to set background color to white
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend_title='Data Label',
        legend=dict(
            itemsizing='constant'
        ),
        width=1000,
        height=800
    )

    # Add grid lines for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig

# Initialize Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the server variable for deployments

# Load precomputed data
try:
    df = load_processed_data()
except FileNotFoundError as e:
    logger.error(e)
    df = pd.DataFrame(columns=['Content', 'Label', 'UMAP1', 'UMAP2'])

# Create Plotly figure
if not df.empty:
    fig = create_plotly_figure(df)
else:
    fig = px.scatter(title="No data available.")

# Define the app layout using Bootstrap components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H1("Interactive UMAP Visualization", className="text-center text-primary mb-4"),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='umap-scatter',
                figure=fig,
                style={'height': '80vh'}
            ),
            width=8
        ),
        dbc.Col([
            html.H4("Selected Content", className="text-center text-secondary"),
            html.Div(
                id='selected-content',
                style={
                    'whiteSpace': 'pre-wrap',
                    'wordWrap': 'break-word',
                    'border': '1px solid #ccc',
                    'padding': '10px',
                    'height': '80vh',
                    'overflowY': 'scroll',
                    'backgroundColor': '#f9f9f9'
                }
            )
        ], width=4)
    ])
], fluid=True)

# Define the callback to update the selected content
@app.callback(
    Output('selected-content', 'children'),
    Input('umap-scatter', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a data point to see its content here."
    try:
        # Extract the customdata
        point = clickData['points'][0]
        content = point['customdata'][0]
        label = point['customdata'][1]

        # Format the content using Bootstrap Card
        return dbc.Card([
            dbc.CardHeader(f"Label: {label}"),
            dbc.CardBody([
                html.P(content, className="card-text")
            ])
        ], color="light", outline=True)
    except (IndexError, KeyError, TypeError) as e:
        logger.error(f"Error accessing clickData: {e}")
        return "Error fetching content."

def main():
    """
    Main function to launch Dash app.
    """
    try:
        logger.info("Launching Dash app...")
        # Retrieve the port from environment variables or use default
        port = int(os.environ.get('PORT', 8050))
        debug_mode = os.environ.get('DEBUG', 'False') == 'True'

        # Run the Dash app
        app.run_server(host='0.0.0.0', port=port, debug=debug_mode)

    except Exception as e:
        logger.error(f"An error occurred while running the app: {e}")

if __name__ == "__main__":
    main()
