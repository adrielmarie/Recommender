import streamlit as st

def Navbar():
    with st.sidebar:
        st.page_link('app.py', label='Interactive Demo')
        st.page_link('pages/llm_app.py', label='LLM Recommender Demo')
        st.page_link('pages/simulations.py', label='Simulation')