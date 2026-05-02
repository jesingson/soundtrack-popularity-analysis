from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui import apply_app_styles


IMAGE_DIR = Path("images/findings")


def render_image(filename: str, caption: str) -> None:
    """Render a findings image if the asset exists."""
    path = IMAGE_DIR / filename

    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Add image asset: `{path}`")


def render_finding(
    number: int,
    title: str,
    claim: str,
    why_it_matters: str,
    image_filename: str | None = None,
    image_caption: str | None = None,
) -> None:
    """Render one findings card."""
    st.markdown(
        f"""
        <div style="
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        ">
            <div style="font-size: 0.8rem; color: #6b7280; font-weight: 600;">
                Finding {number}
            </div>
            <h3 style="margin-top: 0.15rem;">{title}</h3>
            <p><strong>Claim:</strong> {claim}</p>
            <p><strong>Why it matters:</strong> {why_it_matters}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if image_filename:
        render_image(image_filename, image_caption or title)


def render_r2_comparison() -> None:
    """Render the album regression R-squared comparison table."""
    r2_df = pd.DataFrame(
        {
            "Model Scope": ["All Films", "Action", "Comedy", "Drama"],
            "R²": [0.16, 0.31, 0.23, 0.21],
        }
    )

    st.dataframe(
        r2_df,
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    """Render the Key Findings page."""
    apply_app_styles()

    st.title("Key Findings: What Drives Soundtrack Popularity?")
    st.write(
        """
        Soundtrack popularity is not driven by one universal formula. Album-level
        success is shaped by film visibility, genre context, concentrated listener
        behavior, album cohesion, and track-level structure.
        """
    )

    st.markdown("### Executive Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Album-level signal", "Context matters")
    col2.metric("Track-level signal", "Structure matters")
    col3.metric("Best modeling lens", "Genre-specific")

    st.divider()

    render_finding(
        number=1,
        title="Film visibility sets the baseline",
        claim=(
            "Film popularity and related exposure measures show a reliable positive "
            "association with soundtrack listenership."
        ),
        why_it_matters=(
            "Soundtracks benefit from the audience reach of the films they are attached to. "
            "Before album structure or musical characteristics matter, visibility matters."
        ),
        image_filename="album_regression_film_visibility.png",
        image_caption="Album regression: film visibility and exposure signals",
    )

    render_finding(
        number=2,
        title="Drivers of popularity are more predictable within genres",
        claim=(
            "Album-level models explain soundtrack popularity more effectively when "
            "restricted to individual genres than when applied across all films."
        ),
        why_it_matters=(
            "A single global model obscures genre-specific dynamics. Action, Comedy, "
            "and Drama soundtracks appear to follow different internal patterns."
        ),
    )

    st.markdown("#### Genre-specific model comparison")
    render_r2_comparison()

    render_finding(
        number=3,
        title="Soundtrack popularity is winner-take-most",
        claim=(
            "Within composer portfolios, listener attention is highly concentrated in "
            "a small number of albums."
        ),
        why_it_matters=(
            "Soundtrack success is not evenly distributed. A few breakout albums account "
            "for a disproportionate share of listeners."
        ),
        image_filename="lorenz_concentration.png",
        image_caption="Lorenz curve: listener concentration within composer portfolios",
    )

    render_finding(
        number=4,
        title="Cohesion has a consistent but modest effect",
        claim=(
            "More cohesive soundtracks tend to perform slightly better, but cohesion is "
            "not a dominant driver."
        ),
        why_it_matters=(
            "A consistent album sound may help, but it does not explain soundtrack "
            "popularity on its own."
        ),
        image_filename="cohesion_variability_bins.png",
        image_caption="Track cohesion: low-, medium-, and high-variability albums",
    )

    render_finding(
        number=5,
        title="The strongest tracks are disproportionately positioned first",
        claim=(
            "The most popular track in a soundtrack is much more likely to appear at "
            "the beginning of the album."
        ),
        why_it_matters=(
            "Soundtracks often appear front-loaded around a standout opening track, "
            "reinforcing the importance of early engagement."
        ),
        image_filename="strongest_track_position.png",
        image_caption="Track structure: where the strongest track appears",
    )

    render_finding(
        number=6,
        title="Track performance is driven by clearer measurable signals",
        claim=(
            "Track-level playcount is more predictable than album-level listenership."
        ),
        why_it_matters=(
            "Album success is diffuse and context-dependent, but individual tracks show "
            "clearer relationships with genre, film exposure, and track position."
        ),
        image_filename="track_regression_top_predictors.png",
        image_caption="Track regression: strongest predictors of track playcount",
    )


if __name__ == "__main__":
    main()