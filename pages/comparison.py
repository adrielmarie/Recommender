import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import asyncio

import rec_systems as recommender
import rec_llm as llm_recommender

from modules.nav import Navbar

# Load data
@st.cache_resource
def load_all_data():
    try:
        df = pd.read_csv('sanrio_products.csv')
        df['Item #'] = df['Item #'].astype(str)
    except FileNotFoundError:
        st.error("Error: 'sanrio_products.csv' not found.")
        return (None,) * 8
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return (None,) * 8

    has_image_url = 'Image URL' in df.columns

    # Load traditional recommender system
    df_processed_trad = recommender.preprocess_data(df.copy())
    trad_models = recommender.create_tfidf_matrix(df_processed_trad, weights=None)
    
    # Load LLM-based recommender system
    df_processed_llm = recommender.preprocess_data(df.copy())
    llm_models = llm_recommender.create_tfidf_matrix(df_processed_llm)

    return df, df_processed_trad, trad_models, df_processed_llm, llm_models, has_image_url

def get_image_url(item_row, has_image_url_column):
    if has_image_url_column and 'Image URL' in item_row and pd.notna(item_row['Image URL']):
        return item_row['Image URL']
    # Placeholder image
    title = item_row.get('Title', 'No+Title').replace(' ', '+')
    return f"https://placehold.co/400x400/eeeeee/cccccc?text={title}"

def display_recommendations_with_images(st_col, rec_df, full_df, has_image_url, title_col='Title'):
    if rec_df.empty:
        st_col.info("No recommendations found.")
        return
        
    # Merge with the original 'df' to get image URLs
    rec_df_with_images = rec_df.head(10).merge(full_df, on='Item #', how='left')
    
    for index, row in rec_df_with_images.iterrows():
        st_col.divider()
        col1, col2 = st_col.columns([1, 3])
        
        with col1:
            img_url = get_image_url(row, has_image_url)
            col1.image(img_url, width='stretch')
        with col2:
            # Use 'Title_x' if it exists (from merge), otherwise fall back to 'Title' or 'title_col'
            display_title = row.get(f'{title_col}_x', row.get(title_col, "No Title"))
            col2.subheader(f"{index + 1}. {display_title}")
            
            if 'Similarity' in row:
                safe_similarity = max(0.0, row['Similarity'])
                col2.progress(safe_similarity, text=f"Match: {row['Similarity']*100:.2f}%")
            
            info = f"**Type:** {row.get('Type_y', row.get('Type', 'N/A'))}  \n**Characters:** {row.get('Characters', 'N/A')}"
            col2.write(info)

def initialize_session_state():
    defaults = {
        'user_ratings': {},
        'show_results': False,
        'trad_recs_cache': None,
        'trad_profile_cache': None,
        'trad_time_cache': 0.0,
        'llm_recs_cache': None,
        'llm_profile_cache': None,
        'llm_time_cache': 0.0,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_session_state():
    keys_to_reset = [
        'user_ratings', 'show_results',
        'trad_recs_cache', 'trad_profile_cache', 'trad_time_cache',
        'llm_recs_cache', 'llm_profile_cache', 'llm_time_cache'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    # Re-initialize after deleting
    initialize_session_state()

async def run_recommendations(api_key, full_df, trad_data, llm_data):
    user_ratings = st.session_state.user_ratings
    
    # Unpack data
    df_processed_trad, (tfidf_matrix_trad, tfidf_vectorizer_trad, item_id_ind_trad) = trad_data
    df_processed_llm, (tfidf_matrix_llm, tfidf_vectorizer_llm, _) = llm_data

    # Run traditional recommender
    if st.session_state.trad_recs_cache is None:
        with st.spinner("Running Traditional Recommender..."):
            try:
                recs, profile, duration = recommender.get_recommendations(
                    user_ratings,
                    df_processed_trad, 
                    tfidf_matrix_trad, 
                    tfidf_vectorizer_trad,
                    item_id_ind_trad
                )
                st.session_state.trad_recs_cache = recs
                st.session_state.trad_profile_cache = profile
                st.session_state.trad_time_cache = duration
            except Exception as e:
                st.error(f"Traditional Recommender Error: {e}")

    # Run LLM recommender
    if st.session_state.llm_recs_cache is None:
        with st.spinner("Running LLM Recommender (Calling Gemini API)..."):
            try:
                # Build rated_items_df required by get_llm_profile
                rated_item_ids = list(user_ratings.keys())
                rated_items_df = df_processed_llm[df_processed_llm['Item #'].isin(rated_item_ids)].copy()
                # Add the 'rating' column from session state
                rated_items_df['rating'] = rated_items_df['Item #'].map(user_ratings)

                # Get LLM profile
                llm_profile, llm_duration = await llm_recommender.get_llm_profile(
                    rated_items_df, api_key
                )
                
                # Get LLM recommendations
                llm_recs = llm_recommender.get_llm_recommendations(
                    llm_profile,
                    df_processed_llm,
                    tfidf_matrix_llm,
                    tfidf_vectorizer_llm,
                    rated_item_ids
                )
                
                # Save to cache
                st.session_state.llm_recs_cache = llm_recs
                st.session_state.llm_profile_cache = llm_profile
                st.session_state.llm_time_cache = llm_duration
            except Exception as e:
                st.error(f"LLM Recommender Error: {e}")

# Main app
st.title("Recommender Comparison")
st.write("""
Build your own user profile and get recommendations from the traditional and LLM-based recommender! Maximum of 20 products. Check the product catalog for options.
""")
Navbar()

# Load all data and models
data_load_result = load_all_data()

if data_load_result[0] is not None:
    df, df_processed_trad, trad_models, df_processed_llm, llm_models, has_image_url = data_load_result
    
    # Get API Key
    api_key = st.secrets.get("GEMINI_API_KEY")

    # Ensure session state is set up
    initialize_session_state()

    if not st.session_state.show_results:
        # Input ratings
        st.header("Add Product Ratings")

        # Form for adding a new rating
        with st.form(key="add_rating_form"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                item_id_input = st.text_input("Product Item ID:")
            with col2:
                rating_input = st.slider("Rating:", 1, 5, 3)
            with col3:
                # Spacers
                st.write("")
                st.write("")
                submitted = st.form_submit_button("Add Rating", width='stretch')
            
            if submitted:
                if not item_id_input:
                    st.error("Please enter an Item ID.")
                elif item_id_input not in df['Item #'].values:
                    st.error(f"Item ID '{item_id_input}' not found.")
                elif item_id_input in st.session_state.user_ratings:
                    st.warning(f"Item '{item_id_input}' is already in list.")
                elif len(st.session_state.user_ratings) >= 20:
                    st.error("You have reached a maximum of 20 rated items.")
                else:
                    st.session_state.user_ratings[item_id_input] = rating_input
                    st.success(f"Added '{item_id_input}' with rating {rating_input} stars.")

        # Display current list of rated items
        st.subheader(f"Your Rated Items: ({len(st.session_state.user_ratings)} / 20)")
        if not st.session_state.user_ratings:
            st.info("No items rated yet. Use the form above to add items.")
        else:
            rated_items_list = []
            for item_id, rating in st.session_state.user_ratings.items():
                item_details = df[df['Item #'] == item_id].iloc[0]
                rated_items_list.append({
                    "Item #": item_id,
                    "Title": item_details['Title'],
                    "Your Rating": rating
                })
            
            st.dataframe(pd.DataFrame(rated_items_list), width="stretch")
            
            # Button to remove the last item
            if st.button("Remove Last Item"):
                if st.session_state.user_ratings:
                    last_item_id = list(st.session_state.user_ratings.keys())[-1]
                    del st.session_state.user_ratings[last_item_id]
                    st.rerun()

        # Button to trigger recommendations
        st.divider()
        if st.button("Get Recommendations", type="primary", width="stretch", disabled=not st.session_state.user_ratings):
            st.session_state.show_results = True
            st.rerun()

    else:
        st.header("Results")
        
        # Run the async function to populate cache
        trad_data = (df_processed_trad, trad_models)
        llm_data = (df_processed_llm, llm_models)
        asyncio.run(run_recommendations(api_key, df, trad_data, llm_data))

        col1, col2 = st.columns(2)
        
        # Traditional Recommender
        with col1:
            st.subheader("Traditional Recommender")
            st.metric("Calculation Time", f"{st.session_state.trad_time_cache:.4f} seconds")

            with st.expander("User Profile (Top 5 Tokens)"):
                profile = st.session_state.trad_profile_cache
                if profile:
                    st.write("**Top 'Love' Tokens:**")
                    st.dataframe(profile.get("loves", []), width="stretch")
                    st.write("**Top 'Hate' Tokens:**")
                    st.dataframe(profile.get("hates", []), width="stretch")
                else:
                    st.write("No profile was generated.")
            
            st.subheader("Top 10 Recommendations")
            display_recommendations_with_images(
                col1, 
                st.session_state.trad_recs_cache, 
                df, 
                has_image_url,
                title_col='Title'
            )

        # LLM Recommender
        with col2:
            st.subheader("LLM Recommender")
            st.metric("API Call Time", f"{st.session_state.llm_time_cache:.2f} seconds")

            with st.expander("User Profile"):
                profile = st.session_state.llm_profile_cache
                if profile:
                    st.json(profile)
                else:
                    st.write("No profile was generated.")

            st.subheader("Top 10 Recommendations")
            display_recommendations_with_images(
                col2,
                st.session_state.llm_recs_cache,
                df,
                has_image_url,
                title_col='Title'
            )
        
        st.divider()
        if st.button("Start Over & Clear Results", type="primary", width='stretch'):
            reset_session_state()
            st.rerun()
else:
    st.error("Failed to load essential data. The application cannot start.")