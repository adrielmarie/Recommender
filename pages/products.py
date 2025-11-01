import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
from modules.nav import Navbar
import math

st.title("Available Products")
Navbar()

PRODUCTS_PER_PAGE = 50 # Number of items to load at a time
COLUMNS = 4 # Number of columns in the product grid

# LOad data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('sanrio_products.csv')
        df['Item #'] = df['Item #'].astype(str)
        # Fill NaN values for filter columns to make them selectable
        df['Character-centric'] = df['Character-centric'].fillna('Unknown')
        df['Type'] = df['Type'].fillna('Unknown')
        df['Image URL'] = df['Image URL'].fillna('')
        return df
    except FileNotFoundError:
        st.error("Error: 'sanrio_products.csv' not found. Make sure it's in the root directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        return None

@st.cache_data
def get_filter_options(df):
    """Gets unique, sorted lists for filter dropdowns."""
    char_options = sorted(df['Character-centric'].unique())
    type_options = sorted(df['Type'].unique())
    
    char_options.insert(0, 'All')
    type_options.insert(0, 'All')
    
    return char_options, type_options

def main():
    df = load_data()
    if df is None:
        return

    char_options, type_options = get_filter_options(df)

    # Manages infinite scroll
    if 'items_to_show' not in st.session_state:
        st.session_state.items_to_show = PRODUCTS_PER_PAGE
    # Track filters to reset the page count on change
    if 'last_char_filter' not in st.session_state:
        st.session_state.last_char_filter = 'All'
    if 'last_type_filter' not in st.session_state:
        st.session_state.last_type_filter = 'All'

    st.sidebar.header("Filters")
    char_filter = st.sidebar.selectbox("Filter by Character", char_options)
    type_filter = st.sidebar.selectbox("Filter by Type", type_options)

    # FILTERING LOGIC
    # If filters change, reset the number of items to show
    if (char_filter != st.session_state.last_char_filter or 
        type_filter != st.session_state.last_type_filter):
        st.session_state.items_to_show = PRODUCTS_PER_PAGE
        st.session_state.last_char_filter = char_filter
        st.session_state.last_type_filter = type_filter
        st.rerun()

    # Apply filters to the dataframe
    filtered_df = df.copy()
    if char_filter != 'All':
        filtered_df = filtered_df[filtered_df['Character-centric'] == char_filter]
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == type_filter]

    st.write(f"Showing **{min(st.session_state.items_to_show, len(filtered_df))}** of **{len(filtered_df)}** matching products.")

    # PRODUCT GRID DISPLAY
    # Get the slice of products to display based on session state
    products_to_display = filtered_df.iloc[:st.session_state.items_to_show]
    
    num_rows = math.ceil(len(products_to_display) / COLUMNS)

    for i in range(num_rows):
        cols = st.columns(COLUMNS)
        
        # Get the products for the current row
        row_products = products_to_display.iloc[i*COLUMNS : (i+1)*COLUMNS]

        for j, (index, product) in enumerate(row_products.iterrows()):
            with cols[j]:
                with st.container(border=True):
                    # Display image with a placeholder
                    if product['Image URL']:
                        st.image(product['Image URL'], width=450)
                    else:
                        st.image("https://placehold.co/400x400/eeeeee/cccccc?text=No+Image", width=450)
                    
                    # Display Title and Category
                    st.markdown(f"**{product['Title']}**")
                    st.write(f"Type: {product['Type']}")
                    st.caption(f"Item #: {product['Item #']}")

    if st.session_state.items_to_show < len(filtered_df):
        st.divider()
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Load More Products", use_container_width=True, type="primary"):
                st.session_state.items_to_show += PRODUCTS_PER_PAGE
                st.rerun()
    else:
        st.success("You've reached the end of the list!")

if __name__ == "__main__":
    main()