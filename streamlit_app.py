import streamlit as st

from app.ui import apply_app_styles


def render_home() -> None:
    """Render the landing page for the app."""
    st.title("Soundtrack Popularity Explorer")
    st.write(
        """
        Interactive Streamlit companion to the soundtrack popularity analysis
        project.

        Use the navigation in the left sidebar to explore album-level patterns,
        statistical modeling results, track-level data exploration, and
        album–track relationship analysis.
        """
    )


st.set_page_config(
    page_title="Soundtrack Popularity Explorer",
    layout="wide",
)

apply_app_styles()

navigation = st.navigation(
    {
        "⚪ Home": [
            st.Page(
                render_home,
                title="Overview",
                default=True,
            ),
        ],
        "🟦 Album Analysis": [
            st.Page("pages/1_Dataset_Explorer.py", title="Dataset Explorer"),
            st.Page("pages/2_Distribution_Explorer.py", title="Distribution Explorer"),
            st.Page(
                "pages/3_Group_Comparison_Explorer.py",
                title="Group Comparison Explorer",
            ),
            st.Page("pages/4_Relationship_Explorer.py", title="Relationship Explorer"),
            st.Page("pages/5_Concentration_Explorer.py", title="Concentration Explorer"),
            st.Page("pages/6_Cooccurrence_Explorer.py", title="Co-occurrence Explorer"),
            st.Page("pages/7_Cross_Entity_Explorer.py", title="Cross-Entity Explorer"),
        ],
        "🟪 Modeling & Statistical Analysis": [
            st.Page("pages/8_Correlation_Explorer.py", title="Correlation Explorer"),
            st.Page("pages/9_Ridge_Explorer.py", title="Ridge Explorer"),
            st.Page("pages/10_Regression_Explorer.py", title="Regression Explorer"),
        ],
        "🟩 Track Data Exploration": [
            st.Page(
                "pages/30_Track_Data_Explorer.py",
                title="Track Data Explorer",
            ),
        ],
        "🟨 Album–Track Analysis": [
            st.Page(
                "pages/20_Track_Structure_Explorer.py",
                title="Track Structure Explorer",
            ),
            st.Page(
                "pages/21_Track_Album_Relationship_Explorer.py",
                title="Track–Album Relationship Explorer",
            ),
            st.Page(
                "pages/22_Track_Cohesion_Explorer.py",
                title="Track Cohesion Explorer",
            ),
        ],
    }
)

navigation.run()