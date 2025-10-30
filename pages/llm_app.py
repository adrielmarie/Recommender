import streamlit as st
st.set_page_config(layout="centered")

import pandas as pd
import asyncio
from rec_systems import preprocess_data, create_tfidf_matrix
from rec_llm import get_llm_profile, get_llm_recommendations
from modules.nav import Navbar

@st.cache_resource
def load_all():
    try:
        df = pd.read_csv('sanrio_products.csv')
        df['Item #'] = df['Item #'].astype(str)
    except FileNotFoundError:
        st.error("Error: 'sanrio_products.csv' not found.")
        return None, None, None, None, None, False
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, False

    has_image_url = 'Image URL' in df.columns

    if df is not None:
        df_processed = preprocess_data(df.copy())
        tfidf_matrix, tfidf_vectorizer, item_id_to_index = create_tfidf_matrix(df_processed, weights=None)
        
        return df, df_processed, item_id_to_index, tfidf_matrix, tfidf_vectorizer, has_image_url
    
    return None, None, None, None, None, False

def get_image_url(item_row, has_image_url_column):
    """Helper function to get image URL or a placeholder."""
    if has_image_url_column and pd.notna(item_row['Image URL']):
        return item_row['Image URL']
    else:
        # placeholder URL
        title_text = item_row['Title'].split(' ')[0]
        return f"https://placehold.co/600x400/EEE/333?text={title_text}"

def show_rating_screen(df, num_rated, has_image_url):
    st.title("Sanrio Store Recommender — with LLM")
    st.write("Rate five items first to get recommendations!")
    st.progress(num_rated / 5, text=f"Item {num_rated + 1} of 5")
    
    # Get or set the current item to rate
    item_id = st.session_state.get('current_item_id')
    
    # Find a new item if none or if the current one already rated
    if not item_id or item_id in st.session_state.user_ratings:
        rated_ids = set(st.session_state.user_ratings.keys())
        available_items = df[~df['Item #'].isin(rated_ids)]
        
        if available_items.empty:
            st.warning("You've rated all available items!")
            st.info("Click 'Show Recommendations' to see your results.")
            # Add a button to skip to recommendations
            if st.button("Show Recommendations Now"):
                st.session_state.show_recs = True
                st.session_state.current_item_id = None # Clear current item
                st.rerun()
            return

        # Get a new random item and store its ID in the session state
        random_item = available_items.sample(1).iloc[0]
        item_id = random_item['Item #']
        st.session_state.current_item_id = item_id
    else:
        # Get the item row from the stored item_id
        random_item_series = df[df['Item #'] == item_id]
        if random_item_series.empty:
             st.session_state.current_item_id = None
             st.rerun()
             return
        random_item = random_item_series.iloc[0]

    item_title = random_item['Title']
    item_image_url = get_image_url(random_item, has_image_url)

    # Display the item
    st.header(item_title)
    st.image(item_image_url, caption=item_title, width='stretch')
    
    # Create a simple description
    description = f"**Type:** {random_item['Type']}  \n"
    if pd.notna(random_item['Characters']):
        description += f"**Characters:** {random_item['Characters']}"
    if pd.notna(random_item['Description']):
        description += f"\n\n{random_item['Description']}"
    st.info(description)

    st.subheader("How would you rate this item?")
    
    # Display rating buttons
    cols = st.columns(5)
    
    for i in range(1, 6):
        with cols[i-1]:
            if st.button(f"⭐️ {i}", key=f"rate_{item_id}_{i}", use_container_width=True):
                st.session_state.user_ratings[item_id] = i
                st.session_state.current_item_id = None
                
                # Check if we've reached 5 ratings
                if len(st.session_state.user_ratings) >= 5:
                    st.session_state.show_recs = True
                
                st.rerun()

def show_recommendation_screen(df, df_processed, tfidf_matrix, item_id_to_index, tfidf_vectorizer, has_image_url):
    """
    Calculates and displays recommendations using the LLM.
    """
    st.title("Your LLM Recommendations!")
    num_rated = len(st.session_state.user_ratings)
    st.write(f"Based on your {num_rated} ratings, you might also like these:")

    st.subheader("Your Rated Items")
    
    rated_ids = list(st.session_state.user_ratings.keys())
    rated_items_df = df[df['Item #'].isin(rated_ids)].copy()
    rated_items_df['rating'] = rated_items_df['Item #'].apply(lambda x: st.session_state.user_ratings[x])
    
    # Display rated items side-by-side
    cols = st.columns(len(rated_items_df))
    for i, (_, row) in enumerate(rated_items_df.iterrows()):
        with cols[i]:
            st.image(get_image_url(row, has_image_url), caption=f"Rated: {row['rating']} ⭐️", width='stretch')
            st.caption(row['Title'])
    
    # Get API Key
    api_key = st.secrets["GEMINI_API_KEY"]
    
    # Create profile with LLM
    with st.spinner("Analyzing your ratings with Gemini..."):
        try:
            llm_profile = asyncio.run(get_llm_profile(rated_items_df, api_key))
        except Exception as e:
            st.error(f"Error calling LLM: {e}")
            if st.button("Try again"):
                 st.rerun()
            return
    
    # Calculate recommendations from profile
    with st.spinner("Calculating your personalized recommendations..."):
        recommendations = get_llm_recommendations(
            llm_profile,
            df_processed,
            tfidf_matrix,
            tfidf_vectorizer,
            rated_ids,
            top_n=10
        )
        
    # Display LLM's analysis
    st.subheader("AI Analysis of Your Profile:")
    with st.expander("Show/Hide JSON"):
        st.json(llm_profile)

    # Similar to app.py onwards
    st.subheader("Top Recommended Items")
    
    if recommendations.empty:
        st.info("No strong recommendations found. Try rating a few more items.")
    else:
        # Merge to get image URLs and full data from the original df
        rec_df = recommendations.merge(df, on='Item #', how='left')

        # Filter out for positive similarities (already done in llm_recommender, but good to double check)
        rec_df_positive = rec_df[rec_df['Similarity'] > 0]

        if rec_df_positive.empty:
            st.warning("No relevant recommendations found. Try adjusting your ratings!")
            if st.button("Start Over"):
                st.session_state.user_ratings = {}
                st.session_state.show_recs = False
                st.session_state.current_item_id = None
                st.rerun()
            return
        
        for _, row in rec_df_positive.iterrows():
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(get_image_url(row, has_image_url), width='stretch')
            
            with col2:
                # 'Title_y' is the title from the original dataframe after merge
                st.subheader(row['Title_y'])

                # Clip the similarity score before using it in st.progress
                safe_similarity = max(0.0, row['Similarity'])
                st.progress(safe_similarity, text=f"Match: {row['Similarity']*100:.2f}%")
                
                # Show some extra info
                info = f"**Type:** {row['Type_y']}  \n**Characters:** {row['Characters']}"
                st.write(info)

    st.divider()
    if st.button("Start Over & Rate Again", type="primary"):
        # Clear all session state to restart
        st.session_state.user_ratings = {}
        st.session_state.show_recs = False
        st.session_state.current_item_id = None
        st.rerun()

Navbar()

# Load data
df, df_processed, item_id_to_index, tfidf_matrix, tfidf_vectorizer, has_image_url = load_all()

if df is not None:
    # Initialize session state
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    if 'show_recs' not in st.session_state:
        st.session_state.show_recs = False
    if 'current_item_id' not in st.session_state:
        st.session_state.current_item_id = None

    num_rated = len(st.session_state.user_ratings)

    # Show recommendations
    if num_rated >= 5 or st.session_state.show_recs:
        show_recommendation_screen(
            df, df_processed, tfidf_matrix, item_id_to_index, tfidf_vectorizer, has_image_url
        )
    
    # Show rating screen
    else:
        show_rating_screen(df, num_rated, has_image_url)