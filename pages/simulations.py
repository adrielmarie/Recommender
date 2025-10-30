import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import rec_systems as recommender
import random
from modules.nav import Navbar

@st.cache_resource
def load_all():
    try:
        df = pd.read_csv('sanrio_products.csv')
        df['Item #'] = df['Item #'].astype(str)
    except FileNotFoundError:
        st.error("Error: 'sanrio_products.csv' not found. Make sure it's in the root directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

    if df is not None:
        df_processed = recommender.preprocess_data(df.copy())
        tfidf_matrix, _, item_id_to_index = recommender.create_tfidf_matrix(df_processed)
        return df, df_processed, item_id_to_index, tfidf_matrix
    
    return None, None, None, None

def run_simulation(df, df_processed, item_id_to_index, tfidf_matrix):
    # Pick a random number of products to rate (2-9)
    num_to_rate = random.randint(2, 9)
    
    # Get random ratings
    random_item_indices = random.sample(range(len(df)), num_to_rate)
    random_items = df.iloc[random_item_indices]
    
    user_ratings = {}
    rated_items_details = []  # Details for display
    
    for _, row in random_items.iterrows():
        item_id = row['Item #']
        rating = random.randint(1, 5)
        user_ratings[item_id] = rating
        rated_items_details.append({
            "Title": row['Title'],
            "Item #": item_id,
            "Type": row['Type'],
            "Character-centric": row['Character-centric'],
            "Rating": rating
        })
        
    # Get recommendations
    recommendations = recommender.get_recommendations(
        user_ratings,
        df_processed,
        tfidf_matrix,
        item_id_to_index,
        top_n=10
    )
    
    return rated_items_details, recommendations

st.title("🧪 Simulations")
st.markdown("""
This page runs 5 random experiments for the non-LLM recommender every time it's loaded. 
Each experiment simulates a user rating 2-9 products randomly and shows the top 10 recommendations.
""")
st.button("Re-run Simulations")

Navbar()

# Load all data
df, df_processed, item_id_to_index, tfidf_matrix = load_all()

if df is None:
    st.error("Simulation could not run. Check data file 'sanrio_products.csv'.")
else:
    # Run five experiments
    for i in range(1, 6):
        st.divider()
        st.header(f"User #{i}")
        
        rated_items, recommendations = run_simulation(df, df_processed, item_id_to_index, tfidf_matrix)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Simulated Ratings ({len(rated_items)} items)")
            
            rated_df = pd.DataFrame(rated_items).set_index('Item #')
            # Rename column for display
            rated_df.rename(columns={'Character-centric': 'Featured Character'}, inplace=True)
            
            st.dataframe(rated_df, use_container_width=True)
            
        with col2:
            st.subheader("Top 10 Recommendations")
            if recommendations.empty:
                st.info("No recommendations were generated for this simulation (e.g., all ratings were neutral).")
            else:
                # Display recommendations
                rec_display = recommendations[['Item #', 'Title', 'Type', 'Character-centric', 'Similarity']].copy()
                # Rename column for display
                rec_display.rename(columns={'Character-centric': 'Featured Character'}, inplace=True)
                
                st.dataframe(rec_display.set_index('Item #'), use_container_width=True)