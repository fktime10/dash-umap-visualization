import json
import os
from typing import List, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import umap
import plotly.express as px
from dash import Dash, dcc, html, Output, Input
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths to data files
DATA_FILES = {
    "file2": os.path.join('data', 'scl_tia_importable_gpt4_cleaned_train.jsonl'),
    "file3": os.path.join('data', 'TI_COS_intents.jsonl'),
    "file4": os.path.join('data', 'scl_tia_importable_gpt4_cleaned_test.jsonl'),
    "file_lgf": os.path.join('data', 'eval_prompts_lgf.jsonl'),
    "file_project": os.path.join('data', 'projects_results.jsonl')
}

def load_jsonl_file(file_path: str) -> List[dict]:
    """
    Load a JSONL file and return a list of dictionaries.
    """
    data = []
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return data

    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line {line_number} in {file_path}")
    return data

def extract_content_file2(data: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Extract user intents and assistant codes from File 2.
    """
    user_contents = []
    assistant_codes = []
    for item in data:
        for message in item.get('messages', []):
            role = message.get('role')
            content = message.get('content', '')
            if role == 'user':
                user_contents.append(content)
            elif role == 'assistant' and 'FUNC' in content:
                assistant_codes.append(content)
    return user_contents, assistant_codes

def extract_content_ti_cos(data: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Extract prompts and responses from TI_COS dataset.
    """
    ti_cos_prompts = []
    ti_cos_responses = []
    for item in data:
        prompt = item.get('prompt', '')
        response = item.get('response_j', '')
        if prompt:
            ti_cos_prompts.append(prompt)
        if response:
            ti_cos_responses.append(response)
    return ti_cos_prompts, ti_cos_responses

def extract_content_project(data: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Extract project prompts and codes from Project data.
    """
    project_prompts = []
    project_codes = []
    for item in data:
        prompt = item.get('prompt', '')
        response = item.get('response_j', '')
        if prompt:
            project_prompts.append(prompt)
        if response:
            project_codes.append(response)
    return project_prompts, project_codes

def prepare_data() -> pd.DataFrame:
    """
    Load and extract data from all sources, then combine contents and labels.
    Returns a DataFrame with contents and corresponding labels.
    """
    # Load Project data
    project_data = load_jsonl_file(DATA_FILES["file_project"])
    project_prompts, project_codes = extract_content_project(project_data)

    # Load LGF data
    lgf_data = load_jsonl_file(DATA_FILES["file_lgf"])
    lgf_prompts = [entry.get('prompts', '') for entry in lgf_data]
    lgf_codes = [entry.get('codes', '') for entry in lgf_data]

    # Load File2 data
    data2 = load_jsonl_file(DATA_FILES["file2"])
    user_intents, assistant_codes = extract_content_file2(data2)

    # Load TI_COS data
    ti_cos_data = load_jsonl_file(DATA_FILES["file3"])
    ti_cos_prompts, ti_cos_responses = extract_content_ti_cos(ti_cos_data)

    # Load Test data
    data_test = load_jsonl_file(DATA_FILES["file4"])
    user_test_intents, assistant_test_codes = extract_content_file2(data_test)

    # Combine all contents
    all_contents = (
        lgf_prompts + lgf_codes +
        user_intents + assistant_codes +
        ti_cos_prompts + ti_cos_responses +
        user_test_intents + assistant_test_codes +
        project_prompts + project_codes
    )

    # Create corresponding labels
    all_labels = (
        ['Eval Prompt'] * len(lgf_prompts) +
        ['Eval Code'] * len(lgf_codes) +
        ['User Intent (Train)'] * len(user_intents) +
        ['Assistant Code (Train)'] * len(assistant_codes) +
        ['TI_COS Intent'] * len(ti_cos_prompts) +
        ['TI_COS Code'] * len(ti_cos_responses) +
        ['User Intent (Test)'] * len(user_test_intents) +
        ['Assistant Code (Test)'] * len(assistant_test_codes) +
        ['Project Intent'] * len(project_prompts) +
        ['Project Code'] * len(project_codes)
    )

    # Check for consistency
    if len(all_contents) != len(all_labels):
        raise ValueError("Mismatch between number of contents and labels.")

    # Create DataFrame
    df = pd.DataFrame({
        'Content': all_contents,
        'Label': all_labels
    })

    logger.info(f"Total contents loaded: {len(df)}")

    # Print a sample of the DataFrame to verify
    sample_df = df.head(10)
    logger.info("Sample data:")
    logger.info(sample_df)

    # Sanitize content to prevent display issues
    df['Content'] = df['Content'].astype(str)  # Ensure all content is string
    df['Content'] = df['Content'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))  # Remove newlines

    return df

def generate_embeddings(contents: List[str], model_name: str = 'paraphrase-MiniLM-L6-v2') -> pd.DataFrame:
    """
    Generate embeddings for the given contents using SentenceTransformer.
    """
    logger.info(f"Loading SentenceTransformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Error loading model '{model_name}': {e}")
        raise

    logger.info("Generating embeddings...")
    try:
        embeddings = model.encode(contents, show_progress_bar=True, convert_to_tensor=False)
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}")
        raise

    logger.info("Embeddings generated.")
    return pd.DataFrame(embeddings)

def reduce_dimensions_with_umap(
    embeddings: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,  # 2D for visualization
    metric: str = 'cosine',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Reduce the dimensionality of embeddings using UMAP with specified parameters.
    """
    logger.info("Reducing dimensionality with UMAP...")
    try:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        embedding = reducer.fit_transform(embeddings)
    except Exception as e:
        logger.error(f"Error during UMAP reduction: {e}")
        raise

    logger.info("UMAP reduction completed.")
    # Select components based on n_components
    if n_components == 3:
        df_embedding = pd.DataFrame(embedding[:, :3], columns=['UMAP1', 'UMAP2', 'UMAP3'])
    elif n_components == 2:
        df_embedding = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    else:
        raise ValueError("UMAP n_components must be 2 or 3.")
    return df_embedding

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

def main():
    """
    Main function to orchestrate data loading, processing, and launch Dash app.
    """
    try:
        # Step 1: Prepare Data
        logger.info("Loading and preparing data...")
        df = prepare_data()

        # Verify data is loaded
        if df.empty:
            logger.error("No data loaded. Exiting application.")
            return

        # Step 2: Generate Embeddings
        embedding_model = 'paraphrase-MiniLM-L6-v2'  # Ensure this model works
        embeddings = generate_embeddings(df['Content'].tolist(), model_name=embedding_model)

        # Step 3: Normalize Embeddings
        logger.info("Normalizing embeddings...")
        normalized_embeddings = normalize(embeddings, norm='l2')
        logger.info("Normalization completed.")

        # Step 4: Dimensionality Reduction with UMAP
        reduced_embeddings = reduce_dimensions_with_umap(
            normalized_embeddings,
            n_neighbors=15,    # Increased neighbors for better global structure
            min_dist=0.1,       # Smaller min_dist for tighter clusters
            n_components=2,     # 2D for visualization
            metric='cosine',
            random_state=42
        )

        # Add UMAP coordinates to DataFrame
        df['UMAP1'] = reduced_embeddings['UMAP1']
        df['UMAP2'] = reduced_embeddings['UMAP2']

        # Step 5: Create Plotly Figure
        fig = create_plotly_figure(df)

        # Step 6: Launch Dash App
        logger.info("Launching Dash app...")

        # Initialize Dash app
        app = Dash(__name__)
        server = app.server  # Expose the server variable for deployments

        # Define the app layout
        app.layout = html.Div([
            html.H1("Interactive UMAP Visualization"),
            html.Div([
                dcc.Graph(
                    id='umap-scatter',
                    figure=fig,
                    style={'width': '100%', 'height': '80vh'}
                )
            ], style={'width': '70%', 'display': 'inline-block'}),
            html.Div([
                html.H2("Selected Content"),
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
            ], style={'width': '28%', 'display': 'inline-block', 'paddingLeft': '2%', 'verticalAlign': 'top'})
        ], style={'padding': '20px'})

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

                # Format the content (Markdown)
                return f"**Label:** {label}\n\n**Content:**\n{content}"
            except (IndexError, KeyError, TypeError) as e:
                logger.error(f"Error accessing clickData: {e}")
                return "Error fetching content."

        # Run the Dash app
        app.run_server(debug=True)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
