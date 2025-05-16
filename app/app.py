from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from datetime import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from models import db, User, Review, WatchedMovie

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-goes-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movie_recommender.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database tables function
def create_tables():
    db.create_all()

# Load movie data - in a production environment, this would come from a database
@app.route('/api/load_data', methods=['GET'])
def load_data():
    try:
        # Sample movie data - in a real scenario, this would be loaded from a dataset/database
        movies_data = {
            'movies': [
                {'id': 1, 'title': 'The Shawshank Redemption', 'genres': 'Drama', 'description': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.'},
                {'id': 2, 'title': 'The Godfather', 'genres': 'Crime, Drama', 'description': 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.'},
                {'id': 3, 'title': 'The Dark Knight', 'genres': 'Action, Crime, Drama', 'description': 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.'},
                {'id': 4, 'title': 'Pulp Fiction', 'genres': 'Crime, Drama', 'description': 'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.'},
                {'id': 5, 'title': 'The Lord of the Rings: The Return of the King', 'genres': 'Adventure, Fantasy', 'description': 'Gandalf and Aragorn lead the World of Men against Sauron\'s army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring.'},
                {'id': 6, 'title': 'Forrest Gump', 'genres': 'Drama, Romance', 'description': 'The presidencies of Kennedy and Johnson, the events of Vietnam, Watergate, and other historical events unfold through the perspective of an Alabama man with an IQ of 75.'},
                {'id': 7, 'title': 'Inception', 'genres': 'Action, Adventure, Sci-Fi', 'description': 'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.'},
                {'id': 8, 'title': 'The Matrix', 'genres': 'Action, Sci-Fi', 'description': 'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.'},
                {'id': 9, 'title': 'Star Wars: Episode V - The Empire Strikes Back', 'genres': 'Action, Adventure, Fantasy', 'description': 'After the Rebels are brutally overpowered by the Empire on the ice planet Hoth, Luke Skywalker begins Jedi training with Yoda, while his friends are pursued by Darth Vader.'},
                {'id': 10, 'title': 'Interstellar', 'genres': 'Adventure, Drama, Sci-Fi', 'description': 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.'},
            ]
        }
        
        # Save to a temporary file for recommendation system to use
        with open('app/models/movies_data.json', 'w') as f:
            json.dump(movies_data, f)
            
        return jsonify({"status": "success", "message": "Data loaded successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Authentication routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not all([username, email, password, confirm_password]):
            flash('All fields are required.')
            return render_template('register.html', now=datetime.now())
        
        if password != confirm_password:
            flash('Passwords do not match.')
            return render_template('register.html', now=datetime.now())
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return render_template('register.html', now=datetime.now())
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered.')
            return render_template('register.html', now=datetime.now())
            
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html', now=datetime.now())

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False) == 'on'
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html', now=datetime.now())

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    # Get user's reviews and watched movies
    reviews = Review.query.filter_by(user_id=current_user.id).all()
    watched_movies = WatchedMovie.query.filter_by(user_id=current_user.id).all()
    
    # Get movie details for each review and watched movie
    movies = get_movies_data()
    
    review_details = []
    for review in reviews:
        movie = next((m for m in movies if m['id'] == review.movie_id), None)
        if movie:
            review_details.append({
                'review': review,
                'movie': movie
            })
    
    watched_details = []
    for watched in watched_movies:
        movie = next((m for m in movies if m['id'] == watched.movie_id), None)
        if movie:
            watched_details.append({
                'watched': watched,
                'movie': movie
            })
    
    return render_template('profile.html', 
                           reviews=review_details, 
                           watched_movies=watched_details,
                           now=datetime.now())

# API endpoint for recommendations
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        num_recommendations = data.get('num_recommendations', 10)
        
        recommendations = get_recommendations(movie_id, num_recommendations)
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Get all movies API endpoint
@app.route('/api/movies', methods=['GET'])
def api_get_movies():
    try:
        # Load the movie data
        movies = get_movies_data()
        
        return jsonify({
            "status": "success",
            "movies": movies
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Search movies by title API endpoint
@app.route('/api/search', methods=['GET'])
def api_search_movies():
    try:
        query = request.args.get('query', '').lower()
        
        # Load the movie data
        movies = get_movies_data()
        
        # Filter movies by title
        results = [movie for movie in movies if query in movie['title'].lower()]
        
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Review API endpoints
@app.route('/api/reviews/<int:movie_id>', methods=['GET'])
def api_get_movie_reviews(movie_id):
    try:
        reviews = Review.query.filter_by(movie_id=movie_id).all()
        
        review_data = []
        for review in reviews:
            user = User.query.get(review.user_id)
            review_data.append({
                'id': review.id,
                'user_id': review.user_id,
                'username': user.username,
                'rating': review.rating,
                'comment': review.comment,
                'created_at': review.created_at.isoformat()
            })
        
        return jsonify({
            "status": "success",
            "reviews": review_data
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/reviews', methods=['POST'])
@login_required
def api_add_review():
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        
        # Validate input
        if not movie_id or not rating:
            return jsonify({"status": "error", "message": "Movie ID and rating are required"}), 400
        
        # Check if user already reviewed this movie
        existing_review = Review.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        
        if existing_review:
            # Update existing review
            existing_review.rating = rating
            existing_review.comment = comment
            db.session.commit()
            
            return jsonify({
                "status": "success",
                "message": "Review updated successfully",
                "review_id": existing_review.id
            })
        else:
            # Create new review
            review = Review(
                user_id=current_user.id,
                movie_id=movie_id,
                rating=rating,
                comment=comment
            )
            db.session.add(review)
            db.session.commit()
            
            return jsonify({
                "status": "success",
                "message": "Review added successfully",
                "review_id": review.id
            })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Watched movies API endpoints
@app.route('/api/watched', methods=['POST'])
@login_required
def api_add_watched():
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        
        # Check if already marked as watched
        existing = WatchedMovie.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
        
        if not existing:
            # Add to watched movies
            watched = WatchedMovie(
                user_id=current_user.id,
                movie_id=movie_id
            )
            db.session.add(watched)
            db.session.commit()
        
        return jsonify({
            "status": "success",
            "message": "Movie marked as watched"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Web UI routes
@app.route('/')
def index():
    try:
        movies = get_movies_data()
        return render_template('index.html', movies=movies, now=datetime.now())
    except Exception as e:
        return render_template('error.html', error=str(e), now=datetime.now())

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    try:
        movies = get_movies_data()
        
        # Find the movie by id
        movie = next((m for m in movies if m['id'] == movie_id), None)
        
        if not movie:
            return redirect(url_for('index'))
        
        # Get recommendations
        recommendations = get_recommendations(movie_id, 10)
        
        # Get reviews for this movie
        reviews = Review.query.filter_by(movie_id=movie_id).all()
        
        review_data = []
        for review in reviews:
            user = User.query.get(review.user_id)
            review_data.append({
                'id': review.id,
                'user_id': review.user_id,
                'username': user.username,
                'rating': review.rating,
                'comment': review.comment,
                'created_at': review.created_at
            })
        
        # Check if user has watched this movie
        watched = False
        user_review = None
        
        if current_user.is_authenticated:
            watched = WatchedMovie.query.filter_by(
                user_id=current_user.id, 
                movie_id=movie_id
            ).first() is not None
            
            user_review = Review.query.filter_by(
                user_id=current_user.id,
                movie_id=movie_id
            ).first()
        
        return render_template(
            'movie_detail.html', 
            movie=movie, 
            recommendations=recommendations, 
            reviews=review_data,
            watched=watched,
            user_review=user_review,
            now=datetime.now()
        )
    except Exception as e:
        return render_template('error.html', error=str(e), now=datetime.now())

@app.route('/search')
def search_movies():
    try:
        query = request.args.get('query', '').strip()
        
        if not query:
            return redirect(url_for('index'))
        
        # Load the movie data
        movies = get_movies_data()
        
        # Filter movies by title
        results = [movie for movie in movies if query.lower() in movie['title'].lower()]
        
        return render_template('search_results.html', results=results, query=query, now=datetime.now())
    except Exception as e:
        return render_template('error.html', error=str(e), now=datetime.now())

# Helper functions
def get_movies_data():
    # Check if the data file exists
    if not os.path.exists('app/models/movies_data.json'):
        # Initialize the data
        with app.test_client() as client:
            client.get('/api/load_data')
    
    # Load the movie data
    with open('app/models/movies_data.json', 'r') as f:
        movies_data = json.load(f)
    
    return movies_data['movies']

def get_recommendations(movie_id, num_recommendations=10):
    # Load movie data
    movies = get_movies_data()
    
    # Convert to DataFrame
    df = pd.DataFrame(movies)
    
    # Create a text feature combining genres and description for content-based filtering
    df['content'] = df['genres'] + ' ' + df['description']
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the movie that matches the id
    idx = df[df['id'] == movie_id].index[0]
    
    # Get the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the most similar movies (excluding the movie itself)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top movies
    return df.iloc[movie_indices].to_dict(orient='records')

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('app/models', exist_ok=True)
    
    # Create database tables
    with app.app_context():
        create_tables()
    
    app.run(debug=True, port=5000) 