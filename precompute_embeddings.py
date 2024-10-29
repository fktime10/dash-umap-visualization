import json
import os
import logging
from typing import List, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import umap
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths to data files using relative paths
DATA_FILES = {
    "file2": os.path.join('data', 'scl_tia_importable_gpt4_cleaned_train.jsonl'),
    "file3": os.path.join('data', 'TI_COS_intents.jsonl'),
    "file4": os.path.join('data', 'scl_tia_importable_gpt4_cleaned_test.jsonl'),
    "file_lgf": os.path.join('data', 'eval_prompts_lgf.jsonl'),
    "file_project": os.path.join('data', 'projects_results.jsonl')
}

def load_jsonl_file(file_path: str) -> List[dict]:
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
    project_data = load_jsonl_file(DATA_FILES["file_project"])
    project_prompts, project_codes = extract_content_project(project_data)

    lgf_data = load_jsonl_file(DATA_FILES["file_lgf"])
    lgf_prompts = [entry.get('prompts', '') for entry in lgf_data]
    lgf_codes = [entry.get('codes', '') for entry in lgf_data]

    data2 = load_jsonl_file(DATA_FILES["file2"])
    user_intents, assistant_codes = extract_content_file2(data2)

    ti_cos_data = load_jsonl_file(DATA_FILES["file3"])
    ti_cos_prompts, ti_cos_responses = extract_content_ti_cos(ti_cos_data)

    data_test = load_jsonl_file(DATA_FILES["file4"])
    user_test_intents, assistant_test_codes = extract_content_file2(data_test)

    all_contents = (
        lgf_prompts + lgf_codes +
        user_intents + assistant_codes +
        ti_cos_prompts + ti_cos_responses +
        user_test_intents + assistant_test_codes +
        project_prompts + project_codes
    )

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

    if len(all_contents) != len(all_labels):
        raise ValueError("Mismatch between number of contents and labels.")

    df = pd.DataFrame({
        'Content': all_contents,
        'Label': all_labels
    })

    logger.info(f"Total contents loaded: {len(df)}")
    sample_df = df.head(10)
    logger.info("Sample data:")
    logger.info(sample_df)

    df['Content'] = df['Content'].astype(str).apply(lambda x: x.replace('\n', ' ').replace('\r', ' '))
    return df

def generate_embeddings(contents: List[str], model: SentenceTransformer) -> pd.DataFrame:
    logger.info("Generating embeddings...")
    embeddings = model.encode(contents, show_progress_bar=True, convert_to_tensor=False)
    logger.info("Embeddings generated.")
    return pd.DataFrame(embeddings)

def reduce_dimensions_with_umap(
    embeddings: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = 'cosine',
    random_state: int = 42
) -> pd.DataFrame:
    logger.info("Reducing dimensionality with UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state
    )
    embedding = reducer.fit_transform(embeddings)
    logger.info("UMAP reduction completed.")
    if n_components == 3:
        df_embedding = pd.DataFrame(embedding[:, :3], columns=['UMAP1', 'UMAP2', 'UMAP3'])
    elif n_components == 2:
        df_embedding = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    else:
        raise ValueError("UMAP n_components must be 2 or 3.")
    return df_embedding

def main():
    try:
        df = prepare_data()

        if df.empty:
            logger.error("No data loaded. Exiting precompute script.")
            return

        # Initialize the model once
        model_name = 'paraphrase-MiniLM-L6-v2'
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        model = SentenceTransformer(model_name)

        # Generate embeddings
        embeddings = generate_embeddings(df['Content'].tolist(), model=model)

        # Normalize embeddings
        logger.info("Normalizing embeddings...")
        normalized_embeddings = normalize(embeddings, norm='l2')
        logger.info("Normalization completed.")

        # Dimensionality Reduction with UMAP
        reduced_embeddings = reduce_dimensions_with_umap(
            normalized_embeddings,
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='cosine',
            random_state=42
        )

        # Add UMAP coordinates to DataFrame
        df['UMAP1'] = reduced_embeddings['UMAP1']
        df['UMAP2'] = reduced_embeddings['UMAP2']

        # Save the processed DataFrame
        output_file = os.path.join('data', 'processed_data.pkl')
        logger.info(f"Saving processed data to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(df, f)
        logger.info("Data successfully saved.")

    except Exception as e:
        logger.error(f"An error occurred during precomputing: {e}")

if __name__ == "__main__":
    main()
