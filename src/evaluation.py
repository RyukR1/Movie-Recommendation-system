import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collaborative_filter import CollaborativeFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load the training and test datasets.
    
    Returns:
    --------
    tuple: (train_df, test_df, movies_df)
    """
    data_dir = os.path.join('data', 'processed')
    
    # Check if files exist
    train_file = os.path.join(data_dir, 'ratings_train.csv')
    test_file = os.path.join(data_dir, 'ratings_test.csv')
    movies_file = os.path.join(data_dir, 'movies_processed.csv')
    
    if not all(os.path.exists(f) for f in [train_file, test_file, movies_file]):
        logger.error("Required data files not found. Please run data_preprocessing.py first.")
        return None, None, None
    
    # Load the datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    movies_df = pd.read_csv(movies_file)
    
    logger.info(f"Loaded training set with {len(train_df)} ratings")
    logger.info(f"Loaded test set with {len(test_df)} ratings")
    
    return train_df, test_df, movies_df

def load_models():
    """
    Load the trained collaborative filtering models.
    
    Returns:
    --------
    tuple: (user_based_model, item_based_model)
    """
    models_dir = 'models'
    
    # Check if model files exist
    user_model_file = os.path.join(models_dir, 'user_based_model.pkl')
    item_model_file = os.path.join(models_dir, 'item_based_model.pkl')
    
    if not all(os.path.exists(f) for f in [user_model_file, item_model_file]):
        logger.error("Model files not found. Please run collaborative_filter.py first.")
        return None, None
    
    # Load the models
    user_model = CollaborativeFilter.load_model(user_model_file)
    item_model = CollaborativeFilter.load_model(item_model_file)
    
    logger.info("Loaded user-based and item-based collaborative filtering models")
    
    return user_model, item_model

def calculate_prediction_metrics(model, test_df, train_df=None, method='user'):
    """
    Calculate prediction accuracy metrics (RMSE, MAE) for a model.
    
    Parameters:
    -----------
    model: CollaborativeFilter
        Trained collaborative filtering model
    test_df: DataFrame
        Test dataset containing ratings
    train_df: DataFrame, default=None
        Training dataset (needed for item-based CF)
    method: str, default='user'
        Type of collaborative filtering ('user' or 'item')
        
    Returns:
    --------
    dict: Dictionary containing evaluation metrics
    """
    logger.info(f"Calculating prediction metrics for {method}-based model")
    
    # Initialize variables to store predictions and actual ratings
    y_true = []
    y_pred = []
    
    # Group test data by user
    user_groups = test_df.groupby('userId')
    
    # For each user in the test set
    for user_id, group in user_groups:
        # Get the user's test ratings
        test_ratings = group[['movieId', 'rating']]
        
        for _, row in test_ratings.iterrows():
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            # Get predicted rating
            try:
                if method == 'user':
                    # For user-based CF
                    # Check if user and movie exist in the training data
                    if user_id in model.mapping:
                        user_idx = model.mapping[user_id]
                        movie_indices = [i for i, mid in model.reverse_mapping.items() if mid == movie_id]
                        
                        if movie_indices:
                            movie_idx = movie_indices[0]
                            
                            # Find similar users
                            user_vector = model.data_matrix[user_idx].toarray().reshape(1, -1)
                            distances, indices = model.model.kneighbors(
                                user_vector, 
                                n_neighbors=min(model.n_neighbors+1, model.data_matrix.shape[0])
                            )
                            
                            # Skip the first entry (the user itself)
                            similar_user_indices = indices.flatten()[1:]
                            
                            # Calculate predicted rating based on similar users
                            ratings = []
                            sim_scores = []
                            
                            for sim_user_idx in similar_user_indices:
                                # Get the similar user's rating for this movie
                                rating = model.data_matrix[sim_user_idx, movie_idx]
                                
                                if rating > 0:
                                    # Calculate similarity score (inverse distance)
                                    sim_score = 1 / (1 + distances[0, np.where(indices[0] == sim_user_idx)[0][0]])
                                    ratings.append(rating)
                                    sim_scores.append(sim_score)
                            
                            if ratings:
                                # Weighted average of ratings from similar users
                                predicted_rating = np.average(ratings, weights=sim_scores)
                                y_true.append(actual_rating)
                                y_pred.append(predicted_rating)
                else:
                    # For item-based CF
                    # Need to get user's ratings from training data
                    if train_df is None:
                        continue
                    
                    user_train_ratings = train_df[train_df['userId'] == user_id]
                    
                    if user_train_ratings.empty:
                        continue
                    
                    # Get the indices of the user's rated movies in training
                    rated_movie_ids = user_train_ratings['movieId'].values
                    rated_movie_indices = [model.mapping.get(mid) for mid in rated_movie_ids 
                                          if mid in model.mapping]
                    
                    if not rated_movie_indices:
                        continue
                    
                    # Get the ratings given by the user in training
                    rated_movie_scores = user_train_ratings['rating'].values
                    
                    # Get index of the test movie if it exists in the model
                    movie_indices = [i for i, mid in model.reverse_mapping.items() if mid == movie_id]
                    if not movie_indices or movie_id in rated_movie_ids:
                        continue
                    
                    movie_idx = movie_indices[0]
                    
                    # Calculate similarity scores for all rated movies
                    sim_scores = []
                    ratings = []
                    
                    for idx, rm_idx in enumerate(rated_movie_indices):
                        # Get rating vector for the rated movie
                        movie_vector = model.data_matrix[rm_idx].toarray().reshape(1, -1)
                        
                        # Get rating vector for the test movie
                        test_movie_vector = model.data_matrix[movie_idx].toarray().reshape(1, -1)
                        
                        # Calculate cosine similarity
                        dot_product = np.dot(movie_vector, test_movie_vector.T)[0, 0]
                        norm_product = np.linalg.norm(movie_vector) * np.linalg.norm(test_movie_vector)
                        
                        if norm_product != 0:
                            similarity = dot_product / norm_product
                            sim_scores.append(similarity)
                            ratings.append(rated_movie_scores[idx])
                    
                    if sim_scores:
                        # Weighted average of ratings based on similarity
                        predicted_rating = np.average(ratings, weights=sim_scores)
                        y_true.append(actual_rating)
                        y_pred.append(predicted_rating)
            
            except Exception as e:
                logger.error(f"Error calculating prediction for user {user_id}, movie {movie_id}: {str(e)}")
    
    # Calculate metrics if we have predictions
    if y_true and y_pred:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        logger.info(f"{method.capitalize()}-based model - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return {
            'method': method,
            'rmse': rmse,
            'mae': mae,
            'num_predictions': len(y_true)
        }
    else:
        logger.warning(f"No predictions made for {method}-based model")
        return {
            'method': method,
            'rmse': None,
            'mae': None,
            'num_predictions': 0
        }

def calculate_recommendation_metrics(model, test_df, train_df, k=10, method='user'):
    """
    Calculate recommendation quality metrics (Precision@k, Recall@k).
    
    Parameters:
    -----------
    model: CollaborativeFilter
        Trained collaborative filtering model
    test_df: DataFrame
        Test dataset containing ratings
    train_df: DataFrame
        Training dataset
    k: int, default=10
        Number of recommendations to consider
    method: str, default='user'
        Type of collaborative filtering ('user' or 'item')
        
    Returns:
    --------
    dict: Dictionary containing evaluation metrics
    """
    logger.info(f"Calculating recommendation metrics for {method}-based model at k={k}")
    
    # Threshold to consider a rating as 'liked'
    # Assuming rating scale of 1-5, ratings >= 4 are considered 'liked'
    rating_threshold = 4.0
    
    # Initialize metrics
    precisions = []
    recalls = []
    
    # Group test data by user
    user_groups = test_df.groupby('userId')
    
    # For each user in the test set
    for user_id, group in user_groups:
        # Skip users not in training set
        if user_id not in model.mapping:
            continue
        
        # Get the movies the user liked in the test set
        liked_movies_test = set(group[group['rating'] >= rating_threshold]['movieId'].values)
        
        if not liked_movies_test:
            continue  # Skip users with no liked movies in test set
        
        # Get recommendations for the user
        if method == 'user':
            recommended_movies = model.recommend(user_id, n_recommendations=k)
        else:
            recommended_movies = model.recommend(user_id, train_df, n_recommendations=k)
        
        # Calculate precision and recall
        if recommended_movies:
            recommended_set = set(recommended_movies)
            relevant_and_recommended = len(liked_movies_test.intersection(recommended_set))
            
            precision = relevant_and_recommended / len(recommended_set) if recommended_set else 0
            recall = relevant_and_recommended / len(liked_movies_test) if liked_movies_test else 0
            
            precisions.append(precision)
            recalls.append(recall)
    
    # Calculate average precision and recall
    if precisions and recalls:
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        logger.info(f"{method.capitalize()}-based model - Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}")
        
        return {
            'method': method,
            'precision': avg_precision,
            'recall': avg_recall,
            'k': k
        }
    else:
        logger.warning(f"No recommendations evaluated for {method}-based model")
        return {
            'method': method,
            'precision': None,
            'recall': None,
            'k': k
        }

def visualize_results(prediction_metrics, recommendation_metrics, output_dir='results'):
    """
    Visualize evaluation results.
    
    Parameters:
    -----------
    prediction_metrics: list
        List of dictionaries containing prediction metrics
    recommendation_metrics: list
        List of dictionaries containing recommendation metrics
    output_dir: str, default='results'
        Directory to save visualization results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Convert metrics to DataFrames
    pred_df = pd.DataFrame(prediction_metrics)
    rec_df = pd.DataFrame(recommendation_metrics)
    
    # Visualization 1: Prediction Metrics Comparison
    if not pred_df.empty and pred_df['rmse'].notna().any():
        plt.figure(figsize=(10, 6))
        
        # Plot RMSE
        plt.subplot(1, 2, 1)
        sns.barplot(x='method', y='rmse', data=pred_df)
        plt.title('RMSE Comparison')
        plt.ylabel('RMSE (lower is better)')
        plt.xlabel('Model Type')
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        sns.barplot(x='method', y='mae', data=pred_df)
        plt.title('MAE Comparison')
        plt.ylabel('MAE (lower is better)')
        plt.xlabel('Model Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_metrics.png'))
        plt.close()
    
    # Visualization 2: Recommendation Metrics Comparison
    if not rec_df.empty and rec_df['precision'].notna().any():
        plt.figure(figsize=(10, 6))
        
        # Plot Precision
        plt.subplot(1, 2, 1)
        sns.barplot(x='method', y='precision', data=rec_df)
        plt.title(f'Precision@{rec_df["k"].iloc[0]} Comparison')
        plt.ylabel('Precision (higher is better)')
        plt.xlabel('Model Type')
        
        # Plot Recall
        plt.subplot(1, 2, 2)
        sns.barplot(x='method', y='recall', data=rec_df)
        plt.title(f'Recall@{rec_df["k"].iloc[0]} Comparison')
        plt.ylabel('Recall (higher is better)')
        plt.xlabel('Model Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'recommendation_metrics.png'))
        plt.close()
    
    # Save metrics to CSV
    pred_df.to_csv(os.path.join(output_dir, 'prediction_metrics.csv'), index=False)
    rec_df.to_csv(os.path.join(output_dir, 'recommendation_metrics.csv'), index=False)
    
    logger.info(f"Visualization results saved to {output_dir}")

def main():
    """
    Main function to execute the evaluation pipeline.
    """
    # Load data
    train_df, test_df, movies_df = load_data()
    
    if train_df is None or test_df is None:
        return
    
    # Load models
    user_model, item_model = load_models()
    
    if user_model is None or item_model is None:
        return
    
    # Calculate prediction metrics
    user_pred_metrics = calculate_prediction_metrics(user_model, test_df, method='user')
    item_pred_metrics = calculate_prediction_metrics(item_model, test_df, train_df, method='item')
    
    # Calculate recommendation metrics for different values of k
    k_values = [5, 10, 20]
    recommendation_metrics = []
    
    for k in k_values:
        user_rec_metrics = calculate_recommendation_metrics(user_model, test_df, train_df, k=k, method='user')
        item_rec_metrics = calculate_recommendation_metrics(item_model, test_df, train_df, k=k, method='item')
        recommendation_metrics.extend([user_rec_metrics, item_rec_metrics])
    
    # Visualize results
    visualize_results([user_pred_metrics, item_pred_metrics], recommendation_metrics)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 