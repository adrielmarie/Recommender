import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import random
import asyncio

# Import your recommender systems
import rec_systems
import rec_llm as llm_recommender

from modules.nav import Navbar

@st.cache_resource
def load_llm_data():
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
        df_processed = rec_systems.preprocess_data(df.copy())
        tfidf_matrix, tfidf_vectorizer, item_id_to_index = llm_recommender.create_tfidf_matrix(df_processed)
        return df, df_processed, tfidf_matrix, tfidf_vectorizer, item_id_to_index
    
    return None, None, None, None

async def run_llm_simulation(experiment_num, df, df_llm_copy, tfidf_matrix, tfidf_vectorizer, item_id_to_index, api_key):
    st.subheader(f"Experiment {experiment_num}: Rating {random.randint(3, 9)} Products")

    # Generate random ratings
    with st.spinner(f"[Exp {experiment_num}] Generating random ratings..."):
        num_to_rate = random.randint(3, 9)
        random_items = df.sample(num_to_rate)
        
        rated_items_data = [] # For LLM prompt
        rated_item_ids = []   # For filtering

        for _, row in random_items.iterrows():
            rating = random.randint(1, 5)
            item_id = row['Item #']
            rated_item_ids.append(item_id)
            
            # Collect data for the LLM prompt
            rated_items_data.append({
                "Item #": item_id,
                "rating": rating,
                "Title": row['Title'],
                "Type": row['Type'],
                "Character-centric": row['Character-centric'],
                "Series": row['Series'],
                "Tags": row['Tags']
            })
        
        rated_items_df = pd.DataFrame(rated_items_data)

    # LLM recommender
    try:
        # Get AI's inferred profile
        with st.spinner(f"[Exp {experiment_num}] Calling Gemini API..."):
            llm_profile = await llm_recommender.get_llm_profile(rated_items_df, api_key)
        
        # Get recommendations based on that profile
        with st.spinner(f"[Exp {experiment_num}] Calculating recommendations..."):
            print("item_id_to_index", item_id_to_index)
            print("rated_item_ids", rated_item_ids)
            
            recommendations = llm_recommender.get_llm_recommendations(
                llm_profile,
                df_llm_copy,
                tfidf_matrix,
                tfidf_vectorizer,
                rated_item_ids
            )
        
        # Display Results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Simulated Ratings:**")
            st.dataframe(rated_items_df[['Title', 'Type', 'rating']])
            with st.expander("Show AI-Inferred Profile (JSON)"):
                st.json(llm_profile)
        
        with col2:
            st.write("**Top 10 Recommendations:**")
            if recommendations.empty:
                st.warning("No recommendations found with > 0 similarity.")
            else:
                st.dataframe(recommendations)

    except Exception as e:
        st.error(f"Error in Experiment {experiment_num}: {e}")
        st.json(llm_profile) # Print profile for debugging

st.title("LLM Recommender Simulations")
st.write("""
This page runs 5 separate simulations of the LLM-based recommender.
Each experiment randomly selects 3-9 products, gives them random 1-5 star ratings,
and then calls the Gemini API to infer a user profile and generate recommendations.
""")

Navbar()

# Load all data
data = load_llm_data()

if data[0] is not None:
    df, df_llm_copy, tfidf_matrix, tfidf_vectorizer, item_id_to_index = data

    # Get API Key
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY not found in st.secrets. Please add it to .streamlit/secrets.toml")
    else:
        # This is the main async function that will run all experiments
        async def run_all_simulations():
            for i in range(5):
                st.divider()
                await run_llm_simulation(
                    i + 1,
                    df, 
                    df_llm_copy, 
                    tfidf_matrix, 
                    tfidf_vectorizer, 
                    item_id_to_index, 
                    api_key
                )

        if st.button("Run 5 LLM Simulations", type="primary"):
            # Run the async function that contains the loop
            asyncio.run(run_all_simulations())
else:
    st.error("Failed to load data. Please check file paths and try again.")