import streamlit as st
import pandas as pd
import ast  # for parsing lists from strings (genres/keywords)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem !important;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem !important;
        color: #1E88E5;
        margin-top: 1rem;
    }
    .movie-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .similarity-text {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load the dataset
# Use the correct path to the dataset file
file_path = os.path.join(os.path.dirname(__file__), 'tmdb_5000_movies.csv')
try:
    movies_df = pd.read_csv(file_path)
except FileNotFoundError:
    # Fallback to current directory if file not found in script directory
    movies_df = pd.read_csv('tmdb_5000_movies.csv')

# 2. Feature Engineering: parse genres and keywords, combine with overview
# Fill NaN overviews with empty string
movies_df['overview'] = movies_df['overview'].fillna('')

# Define helper function to parse the 'genres' and 'keywords' columns (which are JSON strings)
def parse_list(col):
    try:
        # Use ast.literal_eval to safely evaluate the string to a Python list/dict structure
        items = ast.literal_eval(col)
        # Extract the 'name' of each item (genre or keyword) into a list
        names = [item['name'] for item in items]
        return " ".join(name.replace(" ", "") for name in names)  # remove spaces in multi-word names for consistency
    except Exception as e:
        return ""  # return empty string if any issue (e.g., empty or invalid format)

# Apply the parsing to 'genres' and 'keywords' columns
movies_df['genres'] = movies_df['genres'].fillna('[]').apply(parse_list)
movies_df['keywords'] = movies_df['keywords'].fillna('[]').apply(parse_list)

# Combine genres, keywords, and overview into a single text string for each movie
movies_df['combined_content'] = movies_df['genres'] + " " + movies_df['keywords'] + " " + movies_df['overview']

# 3. Vectorize the combined content using TF-IDF and compute cosine similarity
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies_df['combined_content'])
# Compute cosine similarity matrix from TF-IDF feature vectors
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Construct reverse mapping from movie title to index for quick lookup
# (drop_duplicates in case of duplicate titles to avoid any index issues)
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# Enhanced recommendation function: return titles and similarity scores
def get_recommendations(title, top_n=5):
    # If title not in dataset, return empty list
    if title not in indices:
        return [], []
    # Get the index of the given movie
    idx = indices[title]
    # Get cosine similarity scores for this movie with all others
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Skip the first item (it is the movie itself with similarity 1.0) and take the next top_n
    sim_scores = sim_scores[1: top_n+1]
    # Get the movie indices and scores for the top similarities
    movie_indices = [i for i, score in sim_scores]
    similarity_scores = [score for i, score in sim_scores]
    # Return the titles and similarity scores of the top_n similar movies
    return movies_df['title'].iloc[movie_indices].tolist(), similarity_scores

# Get movie details function
def get_movie_details(title):
    if title not in indices:
        return None
    idx = indices[title]
    return {
        'title': movies_df['title'].iloc[idx],
        'overview': movies_df['overview'].iloc[idx],
        'genres': movies_df['genres'].iloc[idx].replace(" ", ", "),
        'popularity': movies_df['popularity'].iloc[idx] if 'popularity' in movies_df.columns else None,
        'vote_average': movies_df['vote_average'].iloc[idx] if 'vote_average' in movies_df.columns else None,
        'release_date': movies_df['release_date'].iloc[idx] if 'release_date' in movies_df.columns else None
    }

# 4. Streamlit UI
st.markdown("<h1 class='main-header'>🎬 Ultimate Movie Recommendation System</h1>", unsafe_allow_html=True)

# Sidebar for filters and options
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/movie.png", width=100)
    st.markdown("## Recommendation Settings")
    
    # Number of recommendations slider
    num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
    
    # Genre filter (if available)
    all_genres = set()
    for genre_list in movies_df['genres'].str.split():
        all_genres.update(genre_list)
    all_genres = sorted(list(all_genres))
    
    if all_genres:
        selected_genres = st.multiselect("Filter by genres (optional)", all_genres)
    
    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses content-based filtering to recommend movies similar to your selection.
    
    Features:
    - Similarity scores
    - Movie details
    - Visual recommendations
    
    Data source: TMDB 5000 Movie Dataset
    """)

# Main content area
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<h2 class='subheader'>Select a Movie</h2>", unsafe_allow_html=True)
    
    # Movie selection dropdown with search
    movie_list = movies_df['title'].tolist()
    selected_movie = st.selectbox("", movie_list, index=0)
    
    # Display selected movie details
    movie_details = get_movie_details(selected_movie)
    if movie_details:
        st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
        st.markdown(f"### {movie_details['title']}")
        
        if movie_details['release_date']:
            st.markdown(f"**Release Date:** {movie_details['release_date']}")
        
        if movie_details['vote_average']:
            st.markdown(f"**Rating:** ⭐ {movie_details['vote_average']}/10")
        
        st.markdown(f"**Genres:** {movie_details['genres']}")
        
        st.markdown("**Overview:**")
        st.markdown(f"{movie_details['overview']}")
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 class='subheader'>Recommendations</h2>", unsafe_allow_html=True)
    
    # Get recommendations for the selected movie
    recommendations, similarity_scores = get_recommendations(selected_movie, top_n=num_recommendations)
    
    if recommendations:
        # Create a dataframe for visualization
        rec_df = pd.DataFrame({
            'Movie': recommendations,
            'Similarity Score': [round(score * 100, 1) for score in similarity_scores]
        })
        
        # Display recommendations with similarity scores
        for i, (movie, score) in enumerate(zip(recommendations, similarity_scores)):
            rec_details = get_movie_details(movie)
            if rec_details:
                st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
                st.markdown(f"### {i+1}. {movie}")
                
                # Similarity score progress bar
                similarity_percentage = score * 100
                st.markdown(f"<p class='similarity-text'>Similarity: {similarity_percentage:.1f}%</p>", unsafe_allow_html=True)
                st.progress(score)
                
                # Movie details
                if rec_details['release_date']:
                    st.markdown(f"**Release Date:** {rec_details['release_date']}")
                
                if rec_details['vote_average']:
                    st.markdown(f"**Rating:** ⭐ {rec_details['vote_average']}/10")
                
                st.markdown(f"**Genres:** {rec_details['genres']}")
                
                with st.expander("Overview"):
                    st.write(rec_details['overview'])
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualization of similarity scores
        st.markdown("<h3 class='subheader'>Similarity Comparison</h3>", unsafe_allow_html=True)
        fig = px.bar(
            rec_df, 
            x='Similarity Score', 
            y='Movie',
            orientation='h',
            color='Similarity Score',
            color_continuous_scale='Viridis',
            title=f"Movies Similar to {selected_movie}"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No recommendations found. The movie might not be in the dataset.")

# Footer
st.markdown("---")
st.markdown("### How it works")
st.markdown("""
This recommendation system uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to find movies with similar content.

1. **Content Analysis**: We analyze movie genres, keywords, and plot overviews
2. **Vectorization**: Convert text data into numerical vectors using TF-IDF
3. **Similarity Calculation**: Compute cosine similarity between movies
4. **Recommendation**: Present the most similar movies with similarity scores
""")
