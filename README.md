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
│   └── streamlit_app.py       # Streamlit interface
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

Launch the Streamlit application:
```
streamlit run app/streamlit_app.py
```

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [Streamlit](https://streamlit.io/) for the web application framework 