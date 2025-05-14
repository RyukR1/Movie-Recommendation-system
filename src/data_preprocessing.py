import pandas as pd
import numpy as np
import os
import zipfile
import requests
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URLs for MovieLens datasets
MOVIELENS_URL = {
    'small': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
    'full': 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
}

def download_dataset(dataset_size='small', data_dir='data'):
    """
    Download the MovieLens dataset if it doesn't exist.
    
    Parameters:
    -----------
    dataset_size: str, default='small'
        Size of the dataset to download ('small' or 'full')
    data_dir: str, default='data'
        Directory to store the data
    
    Returns:
    --------
    str: Path to the extracted dataset directory
    """
    if dataset_size not in MOVIELENS_URL:
        raise ValueError(f"Dataset size must be one of {list(MOVIELENS_URL.keys())}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine file paths
    zip_path = os.path.join(data_dir, f"ml-latest-{dataset_size}.zip")
    extract_dir = os.path.join(data_dir, f"ml-latest-{dataset_size}")
    
    # Check if dataset already exists
    if os.path.exists(extract_dir):
        logger.info(f"Dataset already exists at {extract_dir}")
        return extract_dir
    
    # Download the dataset
    logger.info(f"Downloading MovieLens {dataset_size} dataset...")
    response = requests.get(MOVIELENS_URL[dataset_size])
    
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the dataset
    logger.info(f"Extracting dataset to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    return extract_dir

def load_and_preprocess_data(data_dir, min_ratings=0):
    """
    Load and preprocess MovieLens dataset.
    
    Parameters:
    -----------
    data_dir: str
        Directory containing the MovieLens data files
    min_ratings: int, default=0
        Minimum number of ratings for a movie to be included
    
    Returns:
    --------
    tuple: (ratings_df, movies_df, users_df if exists)
    """
    logger.info("Loading and preprocessing data...")
    
    # Load ratings data
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    ratings_df = pd.read_csv(ratings_path)
    
    # Load movies data
    movies_path = os.path.join(data_dir, 'movies.csv')
    movies_df = pd.read_csv(movies_path)
    
    # Load links data (contains IMDb and TMDb IDs)
    links_path = os.path.join(data_dir, 'links.csv')
    if os.path.exists(links_path):
        links_df = pd.read_csv(links_path)
        movies_df = pd.merge(movies_df, links_df, on='movieId', how='left')
    
    # Load tags data if available
    tags_path = os.path.join(data_dir, 'tags.csv')
    if os.path.exists(tags_path):
        tags_df = pd.read_csv(tags_path)
        logger.info(f"Loaded {len(tags_df)} tag entries")
    
    # Filter out movies with too few ratings if requested
    if min_ratings > 0:
        movie_counts = ratings_df['movieId'].value_counts()
        popular_movies = movie_counts[movie_counts >= min_ratings].index
        ratings_df = ratings_df[ratings_df['movieId'].isin(popular_movies)]
        movies_df = movies_df[movies_df['movieId'].isin(popular_movies)]
        logger.info(f"Filtered to {len(movies_df)} movies with at least {min_ratings} ratings")
    
    # Parse genres into a separate column
    movies_df['genres_list'] = movies_df['genres'].apply(lambda x: x.split('|') if x != '(no genres listed)' else [])
    
    # Create a clean timestamp column
    ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    
    # Check if users data exists
    users_path = os.path.join(data_dir, 'users.csv')
    users_df = None
    if os.path.exists(users_path):
        users_df = pd.read_csv(users_path)
        logger.info(f"Loaded {len(users_df)} user profiles")
    
    logger.info(f"Loaded {len(ratings_df)} ratings for {len(movies_df)} movies")
    
    # Save processed data for later use
    processed_dir = os.path.join(os.path.dirname(data_dir), 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    ratings_df.to_csv(os.path.join(processed_dir, 'ratings_processed.csv'), index=False)
    movies_df.to_csv(os.path.join(processed_dir, 'movies_processed.csv'), index=False)
    
    if users_df is not None:
        return ratings_df, movies_df, users_df
    else:
        return ratings_df, movies_df

def create_train_test_split(ratings_df, test_size=0.2, random_state=42):
    """
    Split ratings data into training and test sets.
    
    Parameters:
    -----------
    ratings_df: DataFrame
        DataFrame containing user ratings
    test_size: float, default=0.2
        Proportion of data to use for testing
    random_state: int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple: (train_df, test_df)
    """
    logger.info(f"Creating train-test split with test_size={test_size}")
    
    train_df, test_df = train_test_split(
        ratings_df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Save train and test sets
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(processed_dir, 'ratings_train.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'ratings_test.csv'), index=False)
    
    logger.info(f"Train set: {len(train_df)} ratings, Test set: {len(test_df)} ratings")
    
    return train_df, test_df

def create_user_movie_matrix(ratings_df):
    """
    Create a user-movie ratings matrix.
    
    Parameters:
    -----------
    ratings_df: DataFrame
        DataFrame containing user ratings
    
    Returns:
    --------
    DataFrame: User-movie matrix with users as rows and movies as columns
    """
    user_movie_matrix = ratings_df.pivot(
        index='userId', 
        columns='movieId', 
        values='rating'
    ).fillna(0)
    
    logger.info(f"Created user-movie matrix with shape {user_movie_matrix.shape}")
    
    # Save the matrix
    processed_dir = os.path.join('data', 'processed')
    user_movie_matrix.to_csv(os.path.join(processed_dir, 'user_movie_matrix.csv'))
    
    return user_movie_matrix

def main():
    """
    Main function to execute the preprocessing pipeline.
    """
    # Download and extract dataset
    data_dir = download_dataset(dataset_size='small')
    
    # Load and preprocess data
    ratings_df, movies_df = load_and_preprocess_data(data_dir, min_ratings=5)
    
    # Create train-test split
    train_df, test_df = create_train_test_split(ratings_df)
    
    # Create user-movie matrix
    user_movie_matrix = create_user_movie_matrix(train_df)
    
    logger.info("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main() 