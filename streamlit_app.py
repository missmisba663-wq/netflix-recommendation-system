"""Netflix Recommendation System - Streamlit entry point.

Run locally:  streamlit run streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="NetflixRec · AI Recommender", page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

from src.ui import setup_page, sidebar_profile  # noqa: E402

setup_page()

pages = {
    "Discover": [
        st.Page("app_pages/home.py", title="Home", icon="🏠", default=True),
        st.Page("app_pages/recommend.py", title="Get Recommendations", icon="✨"),
        st.Page("app_pages/predictor.py", title="AI Genre Predictor", icon="🔮"),
        st.Page("app_pages/browse.py", title="Browse Catalogue", icon="🔍"),
        st.Page("app_pages/watchlist.py", title="My List", icon="❤️"),
    ],
    "Insights": [
        st.Page("app_pages/analytics.py", title="Analytics Dashboard", icon="📊"),
        st.Page("app_pages/model.py", title="ML Model Insights", icon="🧠"),
        st.Page("app_pages/about.py", title="About the Project", icon="ℹ️"),
    ],
}
nav = st.navigation(pages)
sidebar_profile()
nav.run()
