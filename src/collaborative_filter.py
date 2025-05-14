import pandas as pd
import numpy as np
import os
import logging
import joblib
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollaborativeFilter:
    """
    A class implementing collaborative filtering recommendation algorithms.
    
    This class provides implementations for:
    1. User-based collaborative filtering
    2. Item-based collaborative filtering
    """
    
    def __init__(self, method='user', n_neighbors=10, min_ratings=5, metric='cosine'):
        """
        Initialize the collaborative filtering model.
        
        Parameters:
        -----------
        method: str, default='user'
            The type of collaborative filtering ('user' or 'item')
        n_neighbors: int, default=10
            Number of neighbors to consider for making recommendations
        min_ratings: int, default=5
            Minimum number of ratings for a user/item to be included
        metric: str, default='cosine'
            Distance metric to use for nearest neighbors algorithm
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.min_ratings = min_ratings
        self.metric = metric
        self.model = None
        self.data_matrix = None
        self.mapping = None
        self.reverse_mapping = None
        
        logger.info(f"Initialized {method}-based collaborative filtering model")
    
    def _preprocess_data(self, ratings_df):
        """
        Preprocess the ratings data for the collaborative filtering model.
        
        Parameters:
        -----------
        ratings_df: DataFrame
            DataFrame containing user ratings
            
        Returns:
        --------
        scipy.sparse.csr_matrix: Sparse matrix of ratings
        """
        # Filter out users/items with too few ratings
        if self.method == 'user':
            # For user-based CF, filter out users with too few ratings
            user_counts = ratings_df['userId'].value_counts()
            active_users = user_counts[user_counts >= self.min_ratings].index
            ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]
        else:
            # For item-based CF, filter out items with too few ratings
            item_counts = ratings_df['movieId'].value_counts()
            popular_items = item_counts[item_counts >= self.min_ratings].index
            ratings_df = ratings_df[ratings_df['movieId'].isin(popular_items)]
        
        # Create a user-item matrix
        if self.method == 'user':
            # For user-based CF, users are rows and items are columns
            pivot_table = ratings_df.pivot(
                index='userId', 
                columns='movieId', 
                values='rating'
            ).fillna(0)
            
            # Create mapping from original user IDs to matrix indices
            self.mapping = {user_id: i for i, user_id in enumerate(pivot_table.index)}
            self.reverse_mapping = {i: user_id for user_id, i in self.mapping.items()}
        else:
            # For item-based CF, items are rows and users are columns
            pivot_table = ratings_df.pivot(
                index='movieId', 
                columns='userId', 
                values='rating'
            ).fillna(0)
            
            # Create mapping from original movie IDs to matrix indices
            self.mapping = {movie_id: i for i, movie_id in enumerate(pivot_table.index)}
            self.reverse_mapping = {i: movie_id for movie_id, i in self.mapping.items()}
        
        # Convert to sparse matrix for efficiency
        self.data_matrix = csr_matrix(pivot_table.values)
        
        logger.info(f"Created data matrix with shape {self.data_matrix.shape}")
        return self.data_matrix
    
    def fit(self, ratings_df):
        """
        Fit the collaborative filtering model to the ratings data.
        
        Parameters:
        -----------
        ratings_df: DataFrame
            DataFrame containing user ratings
            
        Returns:
        --------
        self: The fitted model
        """
        # Preprocess the data
        data_matrix = self._preprocess_data(ratings_df)
        
        # Initialize and fit the nearest neighbors model
        self.model = NearestNeighbors(
            n_neighbors=self.n_neighbors+1,  # +1 because the item itself is included
            metric=self.metric,
            algorithm='brute'  # Use brute force for higher accuracy
        )
        self.model.fit(data_matrix)
        
        logger.info(f"Fitted {self.method}-based collaborative filtering model")
        return self
    
    def _get_user_based_recommendations(self, user_id, n_recommendations=5):
        """
        Get recommendations for a user based on user-based collaborative filtering.
        
        Parameters:
        -----------
        user_id: int
            User ID for which to make recommendations
        n_recommendations: int, default=5
            Number of recommendations to return
            
        Returns:
        --------
        list: List of recommended movie IDs
        """
        # Check if user exists in the training data
        if user_id not in self.mapping:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        # Get the user's index in the matrix
        user_idx = self.mapping[user_id]
        
        # Get the user's ratings vector
        user_vector = self.data_matrix[user_idx].toarray().reshape(1, -1)
        
        # Find similar users
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=self.n_neighbors+1)
        
        # Skip the first entry (the user itself)
        similar_user_indices = indices.flatten()[1:]
        
        # Get recommendations based on what similar users have rated highly
        recommendations = []
        for movie_idx in range(self.data_matrix.shape[1]):
            # Skip movies the user has already rated
            if user_vector[0, movie_idx] > 0:
                continue
            
            # Calculate predicted rating based on similar users
            ratings = []
            sim_scores = []
            
            for sim_user_idx in similar_user_indices:
                # Get the similar user's rating for this movie
                rating = self.data_matrix[sim_user_idx, movie_idx]
                
                if rating > 0:
                    # Calculate similarity score (inverse distance)
                    sim_score = 1 / (1 + distances[0, np.where(indices[0] == sim_user_idx)[0][0]])
                    ratings.append(rating)
                    sim_scores.append(sim_score)
            
            if ratings:
                # Weighted average of ratings from similar users
                predicted_rating = np.average(ratings, weights=sim_scores)
                recommendations.append((movie_idx, predicted_rating))
        
        # Sort recommendations by predicted rating and return top n
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [self.reverse_mapping[rec[0]] for rec in recommendations[:n_recommendations]]
    
    def _get_item_based_recommendations(self, user_id, ratings_df, n_recommendations=5):
        """
        Get recommendations for a user based on item-based collaborative filtering.
        
        Parameters:
        -----------
        user_id: int
            User ID for which to make recommendations
        ratings_df: DataFrame
            DataFrame containing user ratings
        n_recommendations: int, default=5
            Number of recommendations to return
            
        Returns:
        --------
        list: List of recommended movie IDs
        """
        # Get the user's rated movies
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if user_ratings.empty:
            logger.warning(f"No ratings found for user {user_id}")
            return []
        
        # Get the indices of the user's rated movies
        rated_movie_ids = user_ratings['movieId'].values
        rated_movie_indices = [self.mapping.get(movie_id) for movie_id in rated_movie_ids 
                               if movie_id in self.mapping]
        
        if not rated_movie_indices:
            logger.warning(f"None of user {user_id}'s rated movies were found in training data")
            return []
        
        # Get the ratings given by the user
        rated_movie_scores = user_ratings['rating'].values
        
        # Calculate recommendations
        recommendations = []
        
        # For each movie the user has rated
        for idx, movie_idx in enumerate(rated_movie_indices):
            # Find similar movies
            movie_vector = self.data_matrix[movie_idx].toarray().reshape(1, -1)
            distances, indices = self.model.kneighbors(movie_vector, n_neighbors=self.n_neighbors+1)
            
            # Skip the first entry (the movie itself)
            similar_movie_indices = indices.flatten()[1:]
            similar_movie_distances = distances.flatten()[1:]
            
            for sim_idx, sim_dist in zip(similar_movie_indices, similar_movie_distances):
                # Get the movie ID
                movie_id = self.reverse_mapping[sim_idx]
                
                # Skip if user has already rated this movie
                if movie_id in rated_movie_ids:
                    continue
                
                # Calculate similarity score (inverse distance)
                sim_score = 1 / (1 + sim_dist)
                
                # Weight the similarity by the user's rating of the original movie
                weighted_score = sim_score * rated_movie_scores[idx]
                
                # Add to recommendations list
                recommendations.append((movie_id, weighted_score))
        
        # Aggregate scores for the same movie and sort by score
        movie_scores = {}
        for movie_id, score in recommendations:
            if movie_id in movie_scores:
                movie_scores[movie_id] += score
            else:
                movie_scores[movie_id] = score
        
        # Sort by score and return top n
        sorted_recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [rec[0] for rec in sorted_recommendations[:n_recommendations]]
    
    def recommend(self, user_id, ratings_df=None, n_recommendations=5):
        """
        Get recommendations for a user.
        
        Parameters:
        -----------
        user_id: int
            User ID for which to make recommendations
        ratings_df: DataFrame, default=None
            DataFrame containing user ratings (required for item-based CF)
        n_recommendations: int, default=5
            Number of recommendations to return
            
        Returns:
        --------
        list: List of recommended movie IDs
        """
        if self.model is None:
            logger.error("Model has not been fitted yet")
            return []
        
        if self.method == 'user':
            return self._get_user_based_recommendations(user_id, n_recommendations)
        else:
            if ratings_df is None:
                logger.error("ratings_df is required for item-based collaborative filtering")
                return []
            return self._get_item_based_recommendations(user_id, ratings_df, n_recommendations)
    
    def save_model(self, file_path):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        file_path: str
            Path to save the model to
        """
        if self.model is None:
            logger.error("Model has not been fitted yet")
            return
        
        model_data = {
            'model': self.model,
            'data_matrix': self.data_matrix,
            'mapping': self.mapping,
            'reverse_mapping': self.reverse_mapping,
            'method': self.method,
            'n_neighbors': self.n_neighbors,
            'min_ratings': self.min_ratings,
            'metric': self.metric
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the model
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        file_path: str
            Path to load the model from
            
        Returns:
        --------
        CollaborativeFilter: Loaded model
        """
        # Load the model data
        model_data = joblib.load(file_path)
        
        # Create a new instance with the saved parameters
        instance = cls(
            method=model_data['method'],
            n_neighbors=model_data['n_neighbors'],
            min_ratings=model_data['min_ratings'],
            metric=model_data['metric']
        )
        
        # Restore the model state
        instance.model = model_data['model']
        instance.data_matrix = model_data['data_matrix']
        instance.mapping = model_data['mapping']
        instance.reverse_mapping = model_data['reverse_mapping']
        
        logger.info(f"Model loaded from {file_path}")
        return instance

def train_and_save_models(ratings_file, output_dir='models'):
    """
    Train and save both user-based and item-based collaborative filtering models.
    
    Parameters:
    -----------
    ratings_file: str
        Path to the processed ratings CSV file
    output_dir: str, default='models'
        Directory to save the trained models
    """
    # Load ratings data
    ratings_df = pd.read_csv(ratings_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train and save user-based model
    user_model = CollaborativeFilter(method='user', n_neighbors=20, min_ratings=5)
    user_model.fit(ratings_df)
    user_model.save_model(os.path.join(output_dir, 'user_based_model.pkl'))
    
    # Train and save item-based model
    item_model = CollaborativeFilter(method='item', n_neighbors=20, min_ratings=10)
    item_model.fit(ratings_df)
    item_model.save_model(os.path.join(output_dir, 'item_based_model.pkl'))
    
    logger.info("Models trained and saved successfully")

def main():
    """
    Main function to execute the model training pipeline.
    """
    # Path to processed ratings file
    ratings_file = os.path.join('data', 'processed', 'ratings_processed.csv')
    
    # Check if file exists
    if not os.path.exists(ratings_file):
        logger.error(f"Ratings file not found: {ratings_file}")
        logger.info("Please run data_preprocessing.py first")
        return
    
    # Train and save models
    train_and_save_models(ratings_file)

if __name__ == "__main__":
    main() 