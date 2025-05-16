# Movie Recommendation System

A collaborative filtering-based movie recommendation system that suggests movies to users based on their preferences and similarity to other users.

## Project Overview

This project implements a movie recommendation system using collaborative filtering techniques. Collaborative filtering is based on the idea that users who have liked similar items in the past will likely like similar items in the future. The system analyzes user ratings to identify patterns and relationships between users and movies.

## Features

- User-based collaborative filtering
- Item-based collaborative filtering
- Interactive web interface using Streamlit
- Data visualization of recommendation patterns
- Movie search functionality
- Personalized recommendations based on user ratings
- User authentication system (registration, login, profile)
- Review and rating system for movies
- "Watched movies" tracking functionality
- Database to store user data, reviews, and watched movies history

## Project Structure

```
movie-recommendation-system/
│
├── data/                      # Data directory
│   ├── movies.csv             # Movie information
│   └── ratings.csv            # User ratings
│
├── models/                    # Trained models
│   └── collaborative_model.pkl
│
├── src/                       # Source code
│   ├── data_preprocessing.py  # Data cleaning and preparation
│   ├── collaborative_filter.py # Collaborative filtering implementation
│   └── evaluation.py          # Model evaluation metrics
│
├── notebooks/                 # Jupyter notebooks for exploration
│   └── model_development.ipynb
│
├── app/                       # Web application
│   ├── app.py                 # Flask application
│   ├── models.py              # Database models
│   ├── streamlit_app.py       # Streamlit interface
│   ├── static/                # Static files (CSS, JS)
│   └── templates/             # HTML templates
│   
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Requirements

### Software Requirements
- **Python**: 3.8 or higher
- **Dependencies**:
  - NumPy (1.24.3 or higher)
  - Pandas (2.0.3 or higher)
  - scikit-learn (1.3.0 or higher)
  - Streamlit (1.25.0 or higher)
  - Flask (2.3.3 or higher)
  - Flask-SQLAlchemy (3.0.5 or higher)
  - Flask-Login (0.6.2 or higher)
  - Werkzeug (2.3.7 or higher)
  - Matplotlib (3.7.2 or higher)
  - Seaborn (0.12.2 or higher)

### Hardware Requirements
- **Processor**: Dual-core processor, 2.0 GHz or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: At least 500MB for the application and dependencies
- **Operating System**: Windows 10+, macOS 10.14+, or Linux

## Usage

### Data Preparation

1. Download the MovieLens dataset or use your own movie rating data
2. Run the data preprocessing script:
```
python src/data_preprocessing.py
```

### Model Training

Train the collaborative filtering model:
```
python src/collaborative_filter.py
```

### Running the Web App

#### Flask Application (with user authentication and review system)
```
python app/app.py
```
The application will run on http://localhost:5000

#### Streamlit Interface
```
streamlit run app/streamlit_app.py
```
The Streamlit app will run on http://localhost:8501

## User Authentication System

The system includes:
- User registration with email verification
- User login with secure password handling
- User profiles with review history and watched movies
- Personalized recommendations based on user behavior

## Review and Rating System

Users can:
- Rate movies on a scale of 1-5 stars
- Write reviews for movies
- View other users' reviews and ratings
- Edit or update their own reviews

## Watched Movies Tracking

The system tracks:
- Movies marked as watched by users
- Date and time when movies were marked as watched
- Recommendations based on watch history

## Methods Used

### User-Based Collaborative Filtering
Identifies users with similar preferences to the target user and recommends items that these similar users have liked.

### Item-Based Collaborative Filtering
Identifies relationships between items based on user ratings and recommends items similar to those the user has previously liked.

## Evaluation Metrics

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Precision and Recall at K
- User satisfaction through feedback

## Future Improvements

- Hybrid recommendation system combining collaborative and content-based filtering
- Implementation of matrix factorization techniques
- Deep learning models for better prediction accuracy
- A/B testing to compare recommendation strategies
- Social features (sharing, following other users)
- Enhanced analytics dashboard for user behavior

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [Streamlit](https://streamlit.io/) for the web application framework 
- [Flask](https://flask.palletsprojects.com/) for the backend framework
- [SQLAlchemy](https://www.sqlalchemy.org/) for database ORM 