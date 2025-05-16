import streamlit as st
import pandas as pd
import requests
import json

# Set page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("Movie Recommendation System")
st.write("Discover movies you might enjoy based on your preferences!")

# API endpoints
API_URL = "http://localhost:5000/api"

# Function to get all movies
@st.cache_data
def get_movies():
    try:
        response = requests.get(f"{API_URL}/movies")
        if response.status_code == 200:
            return response.json()["movies"]
        else:
            st.error("Failed to fetch movies from API")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []

# Function to get recommendations
def get_recommendations(movie_id, num_recommendations=10):
    try:
        payload = {
            "movie_id": movie_id,
            "num_recommendations": num_recommendations
        }
        response = requests.post(f"{API_URL}/recommend", json=payload)
        if response.status_code == 200:
            return response.json()["recommendations"]
        else:
            st.error("Failed to get recommendations")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []

# Function to search movies
def search_movies(query):
    try:
        response = requests.get(f"{API_URL}/search?query={query}")
        if response.status_code == 200:
            return response.json()["results"]
        else:
            st.error("Failed to search movies")
            return []
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return []

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Browse Movies", "Search Movies"])

# Add recommendation settings
st.sidebar.title("Settings")
num_recommendations = st.sidebar.slider("Number of recommendations", 5, 20, 10)

if page == "Browse Movies":
    # Get all movies
    movies = get_movies()
    
    # Display movies in a grid
    st.header("Browse Movies")
    
    # Create columns
    cols = st.columns(3)
    
    for i, movie in enumerate(movies):
        with cols[i % 3]:
            st.subheader(movie["title"])
            st.write(f"**Genres:** {movie['genres']}")
            st.write(movie["description"][:150] + "..." if len(movie["description"]) > 150 else movie["description"])
            
            if st.button(f"Get Recommendations for {movie['title']}", key=f"rec_{movie['id']}"):
                recommendations = get_recommendations(movie["id"], num_recommendations)
                
                st.write("### Recommended Movies:")
                for rec in recommendations:
                    st.write(f"- **{rec['title']}** ({rec['genres']})")

elif page == "Search Movies":
    st.header("Search Movies")
    
    # Search input
    query = st.text_input("Enter movie title to search")
    
    if query:
        results = search_movies(query)
        
        if results:
            st.write(f"Found {len(results)} results for '{query}'")
            
            for movie in results:
                st.subheader(movie["title"])
                st.write(f"**Genres:** {movie['genres']}")
                st.write(movie["description"])
                
                if st.button(f"Get Recommendations for {movie['title']}", key=f"search_rec_{movie['id']}"):
                    recommendations = get_recommendations(movie["id"], num_recommendations)
                    
                    st.write("### Recommended Movies:")
                    for rec in recommendations:
                        st.write(f"- **{rec['title']}** ({rec['genres']})")
        else:
            st.write(f"No results found for '{query}'")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2023 Movie Recommendation System") 