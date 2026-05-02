from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui import apply_app_styles

IMAGE_DIR = Path("images/findings")
APP_BASE_URL = "https://soundtrack-popularity-explorer.streamlit.app"


def get_findings_image_path(filename: str) -> str | None:
    """Return a findings image path when the screenshot exists."""
    path = IMAGE_DIR / filename
    return str(path) if path.exists() else None


def render_findings_image(filename: str, caption: str) -> None:
    """Render a findings screenshot from images/findings."""
    path = get_findings_image_path(filename)

    if path:
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Missing screenshot: `{IMAGE_DIR / filename}`")


def explorer_url(page_name: str) -> str:
    """Build a Streamlit page URL from the visible page route."""
    return f"{APP_BASE_URL}/{page_name}"


def render_explorer_button(label: str, page_name: str) -> None:
    """Render an external link button to a deployed Streamlit explorer page."""
    st.link_button(label, explorer_url(page_name))


def render_home() -> None:
    """Render the landing page, recommended explorers, key findings, and visual playground."""
    st.title("Soundtrack Popularity Explorer")

    st.write(
        """
        Welcome to an interactive companion for a University of Michigan applied
        data science milestone project on film soundtrack popularity. The project
        combines film metadata, soundtrack album data, Last.fm listening metrics,
        awards data, and track-level audio features to explore what drives
        soundtrack performance from 2015–2025.
        """
    )

    st.write(
        """
        The central question: **what drives soundtrack popularity — film context,
        album structure, musical characteristics, or track-level behavior?**
        """
    )

    st.caption(
        "Data sources include TMDB, MusicBrainz, Last.fm, awards data, Spotify IDs, "
        "and Soundnet audio features via RapidAPI."
    )

    st.divider()

    st.markdown("### Start here")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎯 Big-picture exploration")
        st.write(
            """
            Start with the dataset, distribution, and concentration pages to
            understand the shape of the soundtrack universe.
            """
        )
        render_explorer_button("Open Dataset Explorer", "Dataset_Explorer")
        render_explorer_button("Open Distribution Explorer", "Distribution_Explorer")
        render_explorer_button("Open Concentration Explorer", "Concentration_Explorer")

    with col2:
        st.markdown("#### 📈 Modeling & drivers")
        st.write(
            """
            Use the regression explorers to compare album-level and track-level
            predictors of popularity.
            """
        )
        render_explorer_button("Open Album Regression Explorer", "Album_Regression_Explorer")
        render_explorer_button("Open Track Regression Explorer", "Track_Regression_Explorer")

    with col3:
        st.markdown("#### 🎨 Visually rich explorers")
        st.write(
            """
            Try these pages for some of the most visual, interactive views of the
            soundtrack ecosystem.
            """
        )
        render_explorer_button("Open Co-occurrence Explorer", "Cooccurrence_Explorer")
        render_explorer_button("Open Cross-Entity Explorer", "Cross_Entity_Explorer")
        render_explorer_button("Open Track Sequence Explorer", "Track_Sequence_Breakpoints_Explorer")

    st.divider()

    st.markdown("## Key Findings")

    st.write(
        """
        The app points to a layered story: soundtrack success is shaped by film
        visibility, genre context, concentrated listening behavior, album cohesion,
        and track-level structure.
        """
    )

    st.markdown("### 1. Film visibility sets the baseline")
    st.write(
        """
        Film popularity and related exposure measures show a reliable positive
        association with soundtrack listenership. Soundtracks benefit from the
        audience reach of the films they are attached to.
        """
    )
    render_findings_image(
        "album_regression_global_film_visibility.png",
        "Album regression: film visibility and exposure signals",
    )
    render_explorer_button("Explore in Album Regression Explorer", "Album_Regression_Explorer")

    st.markdown("### 2. Drivers are more predictable within genres")
    st.write(
        """
        Album-level models explain soundtrack popularity more effectively when
        restricted to individual genres than when applied across all films. The
        global model has lower explanatory power, while film genre focused models
        such as Action, Comedy, and Drama perform better.
        """
    )

    r2_df = pd.DataFrame(
        {
            "Model Scope": ["All Films", "Action", "Comedy", "Drama"],
            "R²": [0.16, 0.31, 0.23, 0.21],
        }
    )
    st.dataframe(r2_df, hide_index=True, use_container_width=True)
    st.caption(
        "Genre-specific models suggest that soundtrack success follows different "
        "internal patterns depending on film context."
    )
    render_explorer_button("Explore in Album Regression Explorer", "Album_Regression_Explorer")

    st.markdown("### 3. Soundtrack popularity is winner-take-most")
    st.write(
        """
        Listening is rarely distributed evenly. Within composer portfolios, a small
        number of albums capture a disproportionate share of total listeners,
        producing a strong winner-take-most pattern.
        """
    )
    render_findings_image(
        "lorenz_drama_concentration.png",
        "Lorenz curve: listener concentration within composer portfolios",
    )
    render_explorer_button("Explore in Concentration Explorer", "Concentration_Explorer")

    st.markdown("### 4. Cohesion has a consistent but modest effect")

    st.write(
        """
        Cohesion matters most through **variability**, not a single musical trait.
        In the metric-family comparison, all eight cohesion metrics point in the
        same direction: lower variability is associated with higher album listeners.
        Energy variability is the clearest example, while instrumentalness,
        danceability, and happiness variability show the same general pattern.

        The low / medium / high bin view translates that relationship into a simpler
        takeaway: albums with lower energy variability have higher average
        listenership, but the differences are modest.
        """
    )

    render_findings_image(
        "cohesion_low_medium_high.png",
        "Track cohesion: cohesion dimensions and low-, medium-, high-variability view",
    )

    render_explorer_button("Explore in Track Cohesion Explorer", "Track_Cohesion_Explorer")

    st.markdown("### 5. The strongest tracks are disproportionately positioned first")
    st.write(
        """
        The most popular track in a soundtrack is much more likely to appear near
        the beginning of the album. Track 1 is the most common location for the
        strongest track, followed by a sharp drop-off across later positions.
        """
    )
    render_findings_image(
        "strongest_track_position.png",
        "Track structure: where the strongest track appears",
    )
    render_explorer_button("Explore in Track Structure Explorer", "Track_Structure_Explorer")

    st.markdown("### 6. Track-level performance is more predictable")
    st.write(
        """
        Track-level playcount is more predictable than album-level listenership
        (R² ≈ 0.32 vs. about 0.16 in the global album model). The strongest
        positive signals include Pop, Rock, Film Popularity, and track-position
        indicators such as first and last track. Later relative track position is
        negative, reinforcing that track sequencing matters.
        """
    )
    render_findings_image(
        "track_regression_top_predictors.png",
        "Track regression: strongest predictors of track playcount",
    )
    render_explorer_button("Explore in Track Regression Explorer", "Track_Regression_Explorer")

    st.markdown("### 7. Soundtrack sequencing follows genre-specific energy arcs")

    st.write(
        """
        Track energy does not move randomly across an album. In the sequence ribbon
        view, Action, Comedy, and Drama soundtracks follow distinct energy paths
        across normalized track position: Comedy starts higher and tapers, Action
        builds toward later peaks, and Drama starts lower before gradually rising.

        This suggests that soundtrack sequencing has a genre-specific shape. The
        album horizon chart in the Visualization Playground below lets you inspect
        those album-level energy patterns in more detail.
        """
    )

    render_findings_image(
        "sequence_alluvial_ribbons.png",
        "Track Sequence Explorer: genre-level energy arcs across normalized track position",
    )

    render_explorer_button("Explore in Track Sequence Explorer", "Track_Sequence_Explorer")

    st.divider()

    st.markdown("## Visualization Playground")

    st.write(
        """
        These views showcase some of the app's more advanced visual forms. They are
        designed less as final answers and more as interactive tools for spotting
        structure, flow, overlap, and sequence patterns in the soundtrack ecosystem.
        """
    )

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("### Chord chart: award category overlap")
        st.write(
            """
            A chord chart shows relationships between categories as curved links around
            a circle. Here, it helps reveal which soundtrack-related award categories
            tend to co-occur across films.
            """
        )
        render_findings_image(
            "award_chord_chart.png",
            "Chord chart from the Co-occurrence Explorer",
        )
        render_explorer_button("Open Co-occurrence Explorer", "Co-occurrence_Explorer")

    with viz_col2:
        st.markdown("### Ridge plot: listener distribution shifts")
        st.write(
            """
            A ridge plot stacks distributions so you can compare how listener counts
            shift across feature groups. Here, it highlights whether award or creator
            signals correspond to higher album listenership.
            """
        )
        render_findings_image(
            "awards_ridge.png",
            "Ridge plot from the Album Ridge Explorer",
        )
        render_explorer_button("Open Album Ridge Explorer", "Album_Ridge_Explorer")

    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        st.markdown("### Sankey diagram: genre-to-composer flow")
        st.write(
            """
            A Sankey diagram shows how records flow across linked categories. Here,
            it traces connections from film genre to composer to album genre, making
            cross-entity structure visible.
            """
        )
        render_findings_image(
            "film_genre_composer_album_genre_sankey.png",
            "Sankey diagram from the Cross-Entity Explorer",
        )
        render_explorer_button("Open Cross-Entity Explorer", "Cross-Entity_Explorer")

    with viz_col4:
        st.markdown("### Horizon chart: album-level sequence patterns")
        st.write(
            """
            A horizon chart compresses many time- or sequence-based profiles into
            compact rows. Here, it shows where album energy rises or falls across
            normalized track position.
            """
        )
        render_findings_image(
            "track_sequence_horizon.png",
            "Horizon chart from the Track Sequence Explorer",
        )
        render_explorer_button("Open Track Sequence Explorer", "Track_Sequence_Explorer")

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
            st.Page("pages/8_Album_Correlation_Explorer.py", title="Album Correlation Explorer"),
            st.Page("pages/9_Album_Ridge_Explorer.py", title="Album Ridge Explorer"),
            st.Page("pages/10_Album_Regression_Explorer.py", title="Album Regression Explorer"),
            st.Page(
                "pages/40_Track_Correlation_Explorer.py",
                title="Track Correlation Explorer",
            ),
            st.Page(
                "pages/41_Track_Ridge_Explorer.py",
                title="Track Ridge Explorer",
            ),
            st.Page(
                "pages/42_Track_Regression_Explorer.py",
                title="Track Regression Explorer",
            ),
        ],
        "🟩 Track Data Exploration": [
            st.Page(
                "pages/30_Track_Dataset_Explorer.py",
                title="Track Dataset Explorer",
            ),
            st.Page(
                "pages/31_Track_Distribution_Explorer.py",
                title="Track Distribution Explorer",
            ),
            st.Page(
                "pages/32_Track_Comparison_Explorer.py",
                title="Track Comparison Explorer",
            ),
            st.Page(
                "pages/33_Track_Relationship_Explorer.py",
                title="Track Relationship Explorer",
            ),
            st.Page(
                "pages/34_Track_Ecosystem_Explorer.py",
                title="Track Ecosystem Explorer",
            ),
            st.Page(
                "pages/35_Track_Sequence_Breakpoints_Explorer.py",
                title="Track Sequence Explorer",
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