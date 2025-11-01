import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import random
import asyncio
import httpx

# Import your recommender systems
import rec_systems
import rec_llm as llm_recommender

from modules.nav import Navbar

# Define number of products to rate for each experiment
PRODUCT_COUNTS = [3, 5, 10, 15, 20]

@st.cache_resource
def load_llm_data():
    try:
        df = pd.read_csv('sanrio_products.csv')
        df['Item #'] = df['Item #'].astype(str)
    except FileNotFoundError:
        st.error("Error: 'sanrio_products.csv' not found. Make sure it's in the root directory.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None
    
    if df is not None:
        df_processed = rec_systems.preprocess_data(df.copy())
        tfidf_matrix, tfidf_vectorizer, item_id_to_index = llm_recommender.create_tfidf_matrix(df_processed)
        return df, df_processed, tfidf_matrix, tfidf_vectorizer, item_id_to_index
    
    return None, None, None, None, None

async def run_llm_simulation(experiment_num, products_to_rate, df, df_llm_copy, tfidf_matrix, tfidf_vectorizer, item_id_to_index, api_key):
    st.header(f"Experiment {experiment_num}: {products_to_rate} Ratings")
    
    # Randomly select products and assign random 1-5 star ratings
    try:
        # Get a random sample of product indices
        random_indices = random.sample(range(len(df_llm_copy)), products_to_rate)
        rated_items_df = df_llm_copy.iloc[random_indices].copy()
        
        # Assign random ratings (1-5 stars)
        rated_items_df['rating'] = [random.randint(1, 5) for _ in range(products_to_rate)]
        
        rated_item_ids = rated_items_df['Item #'].tolist()
        
        st.subheader("Simulated Ratings")
        st.dataframe(rated_items_df[['Item #', 'Title', 'Type', 'Character-centric', 'rating']].sort_values(by='rating', ascending=False), hide_index=True)

    except ValueError as e:
        st.error(f"Error in product selection for Experiment {experiment_num}: Not enough products to sample {products_to_rate} items.")
        return
    except Exception as e:
        st.error(f"Unexpected error during product selection in Experiment {experiment_num}: {e}")
        return

    # Call the LLM to get the user profile
    st.subheader("Generated LLM Profile")
    try:
        with st.spinner(f"Calling Gemini API to infer profile based on {products_to_rate} ratings..."):
            llm_profile, duration = await llm_recommender.get_llm_profile(rated_items_df, api_key)
            st.metric(label="API Call Duration", value=f"{duration:.2f} seconds")
            with st.expander("Show/Hide JSON"):
                st.json(llm_profile)
    except httpx.ConnectTimeout as e:
        st.error(f"Experiment {experiment_num} Failed: LLM API connection timed out. Check your network or API Key.")
        llm_profile = None
    except Exception as e:
        st.error(f"Experiment {experiment_num} Failed: Error in get_llm_profile: {e}")
        llm_profile = None

    # Generate recommendations using the LLM profile
    if llm_profile:
        st.subheader("LLM-Based Recommendations")
        try:
            recommendations = llm_recommender.get_llm_recommendations(
                llm_profile,
                df,
                tfidf_matrix,
                tfidf_vectorizer,
                rated_item_ids
            )

            # Display the top 10 recommendations
            st.dataframe(recommendations.head(10)[['Item #', 'Title', 'Character-centric', 'Type', 'Similarity']], hide_index=True)

        except Exception as e:
            st.error(f"Error in get_llm_recommendations for Experiment {experiment_num}: {e}")
            st.json(llm_profile) # Print profile for debugging

st.title("LLM Recommender Simulations")
st.write(f"""
This page runs **{len(PRODUCT_COUNTS)}** separate simulations of the LLM-based recommender.
Each experiment progressively selects **{PRODUCT_COUNTS}** products, gives them random 1-5 star ratings,
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
            for i, count in enumerate(PRODUCT_COUNTS):
                st.divider()
                await run_llm_simulation(
                    i + 1,
                    count,
                    df, 
                    df_llm_copy, 
                    tfidf_matrix, 
                    tfidf_vectorizer, 
                    item_id_to_index, 
                    api_key
                )

        if st.button("Run Progressive Simulations", type="primary"):
            st.info("Simulations running... Please wait for the LLM API calls to complete.")
            asyncio.run(run_all_simulations())
            st.success("All simulations complete!")