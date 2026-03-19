import streamlit as st

from app.ui import apply_app_styles

st.set_page_config(
    page_title="Soundtrack Popularity Explorer",
    layout="wide",
)

apply_app_styles()

st.title("Soundtrack Popularity Explorer")
st.write(
    """
    Interactive Streamlit companion to the soundtrack popularity analysis pipeline.

    Use the pages in the left sidebar to explore correlation and, later,
    regression, distribution, and category-specific visualizations.
    """
)