# Import libraries
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load the data
def load_data(dataset):
    df = pd.read_csv(dataset)
    df.head()

# Preprocess the data
def preprocess_data(df):
    # Replace NaN values with empty string for specified columns
    df[['Collaboration', 'Character-centric', 'Series']] = df[['Collaboration', 'Character-centric', 'Series']].fillna("")

    # Handle potential NaN in 'Discount' before converting to string in create_similarity_matrix
    df['Discount'] = df['Discount'].fillna(0)

    return df

# Create TF-IDF vectorizer and compute cosine similarity matrix
def create_tfidf_matrix(df, weights=None):
    # Columns to use for content features
    text_cols = ['Characters', 'Type', 'Tags', 'Series', 'Collaboration', 'Character-centric']
    phrase_cols = ['Characters', 'Type', 'Series', 'Collaboration', 'Character-centric']

    # Handle normal phrase columns
    for col in phrase_cols:
        df[col] = df[col].fillna('').astype(str).str.lower().str.replace(r'[\s,]+', '_', regex=True)
        # Remove any trailing underscores that might result from ", "
        df[col] = df[col].str.replace(r'_+', '_', regex=True).str.strip('_')
        # Handle 'none' string from fillna/conversion
        df[col] = df[col].replace('none', '')

    # Handle tags
    col = 'Tags'
    df[col] = df[col].fillna('').astype(str).str.lower()
    df[col] = df[col].apply(lambda x:
        ' '.join([
            # Strip whitespace, replace internal spaces with _
            tag.strip().replace(' ', '_') for tag in x.split(',') if tag.strip()
        ])
    )
    # Clean up any double underscores
    df[col] = df[col].str.replace(r'_+', '_', regex=True)

    # Initialize weights if without
    if weights is None:
        weights = {
            'star_char': 6,
            'type': 3,
            'series': 2,
            'baseline': 1
        }

    # Build the weighted content soup string by repeating the data
    df['content_soup'] = (
        (df['Character-centric'] + ' ') * weights.get('star_char', 6) +
        (df['Type'] + ' ') * weights.get('type', 3) +
        (df['Series'] + ' ') * weights.get('series', 2) +
        (df['Characters'] + ' ') * weights.get('baseline', 1) +
        (df['Tags'] + ' ') * weights.get('baseline', 1) +
        (df['Collaboration'] + ' ') * weights.get('baseline', 1)
    )

    # Clean up soup: remove extra spaces
    df['content_soup'] = df['content_soup'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Create a mapping from 'Item #' to its index
    item_id_to_index = pd.Series(df.index, index=df['Item #']).to_dict()

    # Compute TF-IDF feature matrix from context soup
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content_soup'])

    return tfidf_matrix, tfidf, item_id_to_index

def get_recommendations(user_ratings, df, tfidf_matrix, item_id_to_index, top_n=10):
    # Create an empty user profile vector
    user_profile = np.zeros(tfidf_matrix.shape[1])

    # Check if any valid ratings were provided
    valid_ratings_found = False

    # Build the user's profile based on their ratings
    for item_id, rating in user_ratings.items():
        item_id = str(item_id)
        if item_id in item_id_to_index:
            idx = item_id_to_index[item_id]

            # Convert rating to a weight (-2 to +2 scale)
            weight = rating - 3.0

            if weight != 0:
                valid_ratings_found = True
                # Get the item's TF-IDF vector
                item_vector = tfidf_matrix[idx].toarray().flatten()
                # Add the weighted vector to the user's profile
                user_profile += item_vector * weight
        else:
            print(f"Warning: Item # '{item_id}' not found in the database.")

    # If no valid ratings (all neutral or no items found), return empty
    if not valid_ratings_found:
        print("No recommendations to generate. Please provide non-neutral ratings (1, 2, 4, or 5).")
        return pd.DataFrame(columns=['Item #', 'Title', 'Type', 'Character-centric' 'Similarity'])

    # Reshape profile to 2D array for cosine_similarity function
    user_profile_sparse = csr_matrix(user_profile)

    # Calculate similarity between the user's profile and all items
    cos_sim_scores = cosine_similarity(user_profile_sparse, tfidf_matrix).flatten()

    # Create a DataFrame of items and their similarity scores
    df_scores = pd.DataFrame({
        'Item #': df['Item #'],
        'Title': df['Title'],
        'Type': df['Type'],
        'Character-centric': df['Character-centric'],
        'Similarity': cos_sim_scores
    })

    # Get list of items the user has already rated
    rated_item_ids = [str(k) for k in user_ratings.keys()]

    # Filter out items the user has already rated
    df_recommendations = df_scores[~df_scores['Item #'].isin(rated_item_ids)]

    # Sort by similarity to get the top recommendations
    df_recommendations = df_recommendations.sort_values(by='Similarity', ascending=False)

    return df_recommendations.head(top_n)

def generate_random_ratings(df, num_products):
    random_item_indices = random.sample(range(len(df)), num_products)
    random_items = df.iloc[random_item_indices]

    my_ratings = {}
    print("Randomly picked products and their ratings:")
    for index, row in random_items.iterrows():
        item_id = row['Item #']
        title = row['Title']
        item_type = row['Type']
        rating = random.randint(1, 5)
        my_ratings[item_id] = rating
        print(f"  - (Item #{item_id}, {item_type}) {title}: {rating}")
    return my_ratings

