import re
import streamlit as st


DISPLAY_LABELS = {
    "film_vote_count": "Film Vote Count",
    "film_popularity": "Film Popularity",
    "film_budget": "Film Budget",
    "film_revenue": "Film Revenue",
    "film_rating": "Film Rating",
    "days_since_film_release": "Days Since Film Release",
    "film_runtime_min": "Film Runtime (Min)",
    "days_since_album_release": "Days Since Album Release",
    "n_tracks": "Track Count",
    "composer_album_count": "Composer Album Count",
    "ambient_experimental": "Ambient / Experimental",
    "classical_orchestral": "Classical / Orchestral",
    "electronic": "Electronic",
    "hip_hop_rnb": "Hip-Hop / R&B",
    "pop": "Pop",
    "rock": "Rock",
    "world_folk": "World / Folk",
    "us_score_nominee_count": "US Score Nominations",
    "us_song_nominee_count": "US Song Nominations",
    "bafta_nominee": "BAFTA Nominee",
    "log_lfm_album_listeners": "Log Album Listeners",
    "album_genre_group": "Album Genre",
    "film_genre_group": "Film Genre",
    "album_us_release_year": "Album Release Year",
    "composer_primary_clean": "Composer",
    "release_group_name": "Soundtrack",
    "album_title": "Album Title",
    "soundtrack_title": "Soundtrack Title",
    "film_title": "Film Title",
    "tmdb_id": "TMDB ID",
    "release_group_mbid": "Release Group MBID",
    "lfm_album_listeners": "Album Listeners",
    "lfm_album_playcount": "Album Playcount",
    "label_names": "Label",
    "film_genres": "Film Genres",
    "album_genres_display": "Album Genres",
    "album_release_lag_days": "Album vs Film Lag (Days)",
    "film_release_date": "Film Release Date",
    "album_us_release_date": "Album Release Date",
    "oscar_score_nominee": "Oscar Score Nominee",
    "oscar_song_nominee": "Oscar Song Nominee",
    "globes_score_nominee": "Globes Score Nominee",
    "globes_song_nominee": "Globes Song Nominee",
    "critics_score_nominee": "Critics Score Nominee",
    "critics_song_nominee": "Critics Song Nominee",
    "bafta_score_nominee": "BAFTA Score Nominee",
    "film_year": "Film Year",
    "plot_value": "Displayed Value",
}


def snake_to_title(name: str) -> str:
    """
    Convert a snake_case field name into a readable title.

    Args:
        name: Raw column or feature name.

    Returns:
        str: Human-readable title-cased label.
    """
    if not name:
        return name

    label = name.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label.title()


def get_display_label(name: str) -> str:
    """
    Return a human-readable label for a field name.

    Uses the curated label registry when available and falls back to a
    generic snake_case-to-title conversion.

    Args:
        name: Raw column or feature name.

    Returns:
        str: Display-friendly label.
    """
    if name is None:
        return ""
    return DISPLAY_LABELS.get(name, snake_to_title(name))


def get_display_labels(names: list[str]) -> dict[str, str]:
    """
    Build a mapping from raw field names to readable labels.

    Args:
        names: Raw field names.

    Returns:
        dict[str, str]: Mapping suitable for DataFrame rename operations.
    """
    return {name: get_display_label(name) for name in names}


def rename_columns_for_display(df):
    """
    Return a copy of a dataframe with readable display labels as columns.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with renamed columns.
    """
    return df.rename(columns=get_display_labels(list(df.columns)))


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        section.main .block-container h1 {
            font-size: 2rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.5rem !important;
        }
        section.main .block-container h2 {
            font-size: 1.4rem !important;
            line-height: 1.2 !important;
            margin-top: 0.75rem !important;
            margin-bottom: 0.35rem !important;
        }
        section.main .block-container h3 {
            font-size: 1.15rem !important;
            line-height: 1.2 !important;
            margin-top: 0.6rem !important;
            margin-bottom: 0.25rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
            line-height: 1.0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )