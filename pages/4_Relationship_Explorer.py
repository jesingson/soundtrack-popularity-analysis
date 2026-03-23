import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import data_processing as dp
import regression_analysis as reg
from app.app_controls import get_scatter_controls
from app.app_data import (
    load_analysis_data,
    load_explorer_data,
    load_source_data,
)
from app.ui import (
    apply_app_styles,
    get_display_label,
    rename_columns_for_display,
)

TOOLTIP_METADATA_CANDIDATES = [
    "release_group_name",
    "album_title",
    "soundtrack_title",
    "film_title",
    "composer_primary_clean",
    "album_us_release_year",
]

EXCLUDED_EXPLORER_COLS = {
    "tmdb_id",
    "barcode",
    "rg_rating_count",
    "film_ingested_at",
    "match_method",
    "matched_at",
    "us_date_has_missing_month",
    "us_date_has_missing_day",
    "us_any_event_missing_month",
    "us_any_event_missing_day",
    "keep_row",
    "vote_count_above_500",
    "canonical_rule",
}

MAX_COLOR_CARDINALITY = 20

ALBUM_GENRE_FLAG_COLS = [
    "ambient_experimental",
    "classical_orchestral",
    "electronic",
    "hip_hop_rnb",
    "pop",
    "rock",
    "world_folk",
]

FILM_GENRE_FLAG_COLS = [
    "film_is_action",
    "film_is_adventure",
    "film_is_animation",
    "film_is_comedy",
    "film_is_crime",
    "film_is_documentary",
    "film_is_drama",
    "film_is_family",
    "film_is_fantasy",
    "film_is_history",
    "film_is_horror",
    "film_is_music",
    "film_is_mystery",
    "film_is_romance",
    "film_is_science_fiction",
    "film_is_tv_movie",
    "film_is_thriller",
    "film_is_war",
    "film_is_western",
]

GENRE_LABEL_MAP = {
    "ambient_experimental": "Ambient / Experimental",
    "classical_orchestral": "Classical / Orchestral",
    "electronic": "Electronic",
    "hip_hop_rnb": "Hip-Hop / R&B",
    "pop": "Pop",
    "rock": "Rock",
    "world_folk": "World / Folk",
    "film_is_action": "Action",
    "film_is_adventure": "Adventure",
    "film_is_animation": "Animation",
    "film_is_comedy": "Comedy",
    "film_is_crime": "Crime",
    "film_is_documentary": "Documentary",
    "film_is_drama": "Drama",
    "film_is_family": "Family",
    "film_is_fantasy": "Fantasy",
    "film_is_history": "History",
    "film_is_horror": "Horror",
    "film_is_music": "Music",
    "film_is_mystery": "Mystery",
    "film_is_romance": "Romance",
    "film_is_science_fiction": "Science Fiction",
    "film_is_tv_movie": "TV Movie",
    "film_is_thriller": "Thriller",
    "film_is_war": "War",
    "film_is_western": "Western",
}

ALBUM_GENRE_DOMAIN = [
    "Ambient / Experimental",
    "Classical / Orchestral",
    "Electronic",
    "Hip-Hop / R&B",
    "Pop",
    "Rock",
    "World / Folk",
    "Multi-genre",
    "Unknown",
]

FILM_GENRE_DOMAIN = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "TV Movie",
    "Thriller",
    "War",
    "Western",
    "Multi-genre",
    "Unknown",
]


def build_feature_rank_lookup(
    ranking_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Build a lookup dictionary for ranked feature metadata.

    Args:
        ranking_df: Ranked feature dataframe.

    Returns:
        dict[str, dict]: Mapping from feature name to ranking row data.
    """
    return ranking_df.set_index("feature").to_dict(orient="index")


def pick_available_metadata_cols(
    albums_df: pd.DataFrame,
) -> list[str]:
    """
    Select descriptive metadata columns that actually exist in albums_df.

    Args:
        albums_df: Raw album dataframe.

    Returns:
        list[str]: Available descriptive columns for tooltip use.
    """
    return [col for col in TOOLTIP_METADATA_CANDIDATES if col in albums_df.columns]

def is_id_like_column(col_name: str) -> bool:
    """Return True if a column name looks like an ID/key field."""
    col = col_name.lower()
    id_markers = [
        "id",
        "mbid",
        "_key",
        "spotify_",
        "musicbrainz_",
    ]
    return any(marker in col for marker in id_markers)

def derive_multi_label_group(
    df: pd.DataFrame,
    flag_cols: list[str],
    label_map: dict[str, str],
    output_col: str,
) -> pd.DataFrame:
    """
    Collapse multi-label genre flags into a single grouping column.

    Rules:
        - one positive flag -> that genre label
        - multiple positive flags -> "Multi-genre"
        - no positive flags -> "Unknown"

    Args:
        df: Source dataframe.
        flag_cols: Binary genre flag columns.
        label_map: Mapping from flag column to display label.
        output_col: Name of the derived grouping column.

    Returns:
        pd.DataFrame: Dataframe with the derived grouping column added.
    """
    available_cols = [col for col in flag_cols if col in df.columns]

    if not available_cols:
        df[output_col] = "Unknown"
        return df

    def assign_group(row: pd.Series) -> str:
        active_cols = [col for col in available_cols if row[col] == 1]
        if len(active_cols) == 1:
            return label_map[active_cols[0]]
        if len(active_cols) > 1:
            return "Multi-genre"
        return "Unknown"

    df[output_col] = df[available_cols].apply(assign_group, axis=1)
    return df


def build_relationship_explorer_df(
    explorer_source_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the dataframe used by the freeform relationship explorer.

    Args:
        explorer_source_df: Rich album-level exploration dataframe.

    Returns:
        pd.DataFrame: Album-level freeform relationship dataframe with
        derived grouping columns added.
    """
    explorer_df = explorer_source_df.copy()

    explorer_df = derive_multi_label_group(
        df=explorer_df,
        flag_cols=ALBUM_GENRE_FLAG_COLS,
        label_map=GENRE_LABEL_MAP,
        output_col="album_genre_group",
    )

    explorer_df = derive_multi_label_group(
        df=explorer_df,
        flag_cols=FILM_GENRE_FLAG_COLS,
        label_map=GENRE_LABEL_MAP,
        output_col="film_genre_group",
    )

    return explorer_df


def get_freeform_numeric_options(
    explorer_df: pd.DataFrame,
) -> list[str]:
    """
    Return numeric columns eligible for freeform X/Y selection.

    Args:
        explorer_df: Merged relationship explorer dataframe.

    Returns:
        list[str]: Safe numeric columns for freeform scatterplots.
    """
    numeric_cols = explorer_df.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    usable_cols = []
    for col in numeric_cols:
        if col in EXCLUDED_EXPLORER_COLS or is_id_like_column(col):
            continue
        if explorer_df[col].dropna().nunique() < 2:
            continue
        usable_cols.append(col)

    preferred_front = [
        "lfm_album_listeners",
        "lfm_album_playcount",
        "n_tracks",
        "album_release_lag_days",
        "days_since_album_release",
        "days_since_film_release",
        "film_vote_count",
        "film_popularity",
        "film_rating",
        "film_runtime_min",
        "film_budget",
        "film_revenue",
        "composer_album_count",
        "us_score_nominee_count",
        "us_song_nominee_count",
        "bafta_nominee",
        dp.TARGET_COL,
    ]

    ordered = [c for c in preferred_front if c in usable_cols]
    ordered.extend([c for c in usable_cols if c not in ordered])

    return ordered


def get_color_options(
    explorer_df: pd.DataFrame,
) -> list[str]:
    """
    Return low-cardinality columns eligible for color encoding.

    Args:
        explorer_df: Merged relationship explorer dataframe.

    Returns:
        list[str]: Safe categorical/binary columns for color grouping.
    """
    color_candidates = []

    for col in explorer_df.columns:
        non_null = explorer_df[col].dropna()
        if non_null.empty:
            continue

        nunique = non_null.nunique()
        dtype = explorer_df[col].dtype

        if (
                pd.api.types.is_object_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
        ):
            if (
                    col not in EXCLUDED_EXPLORER_COLS
                    and not is_id_like_column(col)
                    and 2 <= nunique <= MAX_COLOR_CARDINALITY
            ):
                color_candidates.append(col)


        elif pd.api.types.is_bool_dtype(dtype) or (
                pd.api.types.is_numeric_dtype(dtype) and nunique <= 5
        ):
            if col not in EXCLUDED_EXPLORER_COLS and not is_id_like_column(col):
                color_candidates.append(col)

    preferred_front = [
        "album_genre_group",
        "film_genre_group",
        "composer_primary_clean",
        "album_us_release_year",
        "ambient_experimental",
        "classical_orchestral",
        "electronic",
        "hip_hop_rnb",
        "pop",
        "rock",
        "world_folk",
        "bafta_nominee",
        "film_is_action",
        "film_is_adventure",
        "film_is_animation",
        "film_is_comedy",
        "film_is_crime",
        "film_is_documentary",
        "film_is_drama",
        "film_is_family",
        "film_is_fantasy",
        "film_is_history",
        "film_is_horror",
        "film_is_music",
        "film_is_mystery",
        "film_is_romance",
        "film_is_science_fiction",
        "film_is_tv_movie",
        "film_is_thriller",
        "film_is_war",
        "film_is_western",
    ]

    ordered = [c for c in preferred_front if c in color_candidates]
    ordered.extend([c for c in color_candidates if c not in ordered])

    return ordered


def build_freeform_scatter_data(
    explorer_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    transform_x: str = "None",
    transform_y: str = "None",
    apply_jitter: bool = False,
    jitter_strength: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """
    Build freeform scatterplot data for arbitrary numeric X and Y columns.

    Args:
        explorer_df: Album-level relationship explorer dataframe.
        x_col: Selected x-axis column.
        y_col: Selected y-axis column.
        color_col: Optional color grouping column.
        transform_x: X-axis transform choice ("None" or "Log1p").
        transform_y: Y-axis transform choice ("None" or "Log1p").
        apply_jitter: Whether to add jitter for display only.
        jitter_strength: Relative jitter strength used for display only.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None, dict]:
            1. Plot dataframe
            2. Optional fitted-line dataframe
            3. Summary metrics
    """
    required_cols = [x_col, y_col]
    if color_col and color_col != "None":
        required_cols.append(color_col)

    metadata_cols = [
        col for col in TOOLTIP_METADATA_CANDIDATES
        if col in explorer_df.columns
    ]
    required_cols.extend(metadata_cols)
    required_cols = list(dict.fromkeys(required_cols))

    plot_df = explorer_df[required_cols].copy().dropna(subset=[x_col, y_col])

    if len(plot_df) < 2:
        raise ValueError(
            "At least two non-null rows are required to build the freeform scatterplot."
        )

    plot_df["x_raw"] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df["y_raw"] = pd.to_numeric(plot_df[y_col], errors="coerce")

    if transform_x == "Log1p":
        if (plot_df["x_raw"] < 0).any():
            raise ValueError(
                f"X-axis field '{x_col}' contains negative values and cannot use Log1p."
            )
        plot_df["x_value"] = np.log1p(plot_df["x_raw"])
    else:
        plot_df["x_value"] = plot_df["x_raw"]

    if transform_y == "Log1p":
        if (plot_df["y_raw"] < 0).any():
            raise ValueError(
                f"Y-axis field '{y_col}' contains negative values and cannot use Log1p."
            )
        plot_df["y_value"] = np.log1p(plot_df["y_raw"])
    else:
        plot_df["y_value"] = plot_df["y_raw"]

    plot_df = plot_df.dropna(subset=["x_value", "y_value"])

    if len(plot_df) < 2:
        raise ValueError(
            "At least two valid rows remain after applying transforms."
        )

    x = plot_df["x_value"].to_numpy()
    y = plot_df["y_value"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    x_min, x_max = float(x.min()), float(x.max())

    line_df = pd.DataFrame({
        "x_plot": [x_min, x_max],
        "y_plot": [slope * x_min + intercept, slope * x_max + intercept],
    })

    pearson_r = float(plot_df["x_value"].corr(plot_df["y_value"], method="pearson"))

    plot_df["x_plot"] = plot_df["x_value"]
    plot_df["y_plot"] = plot_df["y_value"]

    if apply_jitter and jitter_strength > 0:
        x_range = float(plot_df["x_value"].max() - plot_df["x_value"].min())
        y_range = float(plot_df["y_value"].max() - plot_df["y_value"].min())

        x_scale = x_range if x_range > 0 else 1.0
        y_scale = y_range if y_range > 0 else 1.0

        rng = np.random.default_rng(42)
        plot_df["x_plot"] = plot_df["x_value"] + rng.normal(
            loc=0.0,
            scale=x_scale * jitter_strength,
            size=len(plot_df),
        )
        plot_df["y_plot"] = plot_df["y_value"] + rng.normal(
            loc=0.0,
            scale=y_scale * jitter_strength,
            size=len(plot_df),
        )

    x_display = get_display_label(x_col)
    y_display = get_display_label(y_col)

    metrics = {
        "x_col": x_col,
        "y_col": y_col,
        "rows_used": len(plot_df),
        "pearson_r": pearson_r,
        "r_squared": pearson_r ** 2,
        "color_col": color_col,
        "transform_x": transform_x,
        "transform_y": transform_y,
        "apply_jitter": apply_jitter,
        "jitter_strength": jitter_strength,
        "x_axis_title": (
            f"log1p({x_display})" if transform_x == "Log1p" else x_display
        ),
        "y_axis_title": (
            f"log1p({y_display})" if transform_y == "Log1p" else y_display
        ),
    }

    return plot_df, line_df, metrics


def create_guided_scatter_chart(
    plot_df: pd.DataFrame,
    line_df: pd.DataFrame,
    metrics: dict,
    feature_rank: dict,
    show_trendline: bool,
) -> alt.Chart:
    """
    Create the guided scatterplot with richer tooltips.

    Args:
        plot_df: Scatterplot point dataframe.
        line_df: Fitted line dataframe.
        metrics: Metrics dictionary returned by the helper.
        feature_rank: Ranking metadata for the selected feature.
        show_trendline: Whether to overlay the fitted line.

    Returns:
        alt.Chart: Layered Altair scatterplot and fitted line.
    """
    feature_label = get_display_label(metrics["feature_col"])
    target_label = get_display_label(metrics["target_col"])

    tooltip_fields = []

    if "film_title" in plot_df.columns:
        tooltip_fields.append(alt.Tooltip("film_title:N", title="Film"))

    if "release_group_name" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("release_group_name:N", title="Soundtrack")
        )
    elif "album_title" in plot_df.columns:
        tooltip_fields.append(alt.Tooltip("album_title:N", title="Soundtrack"))
    elif "soundtrack_title" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("soundtrack_title:N", title="Soundtrack")
        )

    if "composer_primary_clean" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("composer_primary_clean:N", title="Composer")
        )

    tooltip_fields.extend(
        [
            alt.Tooltip(
                "x_raw_value:Q",
                title=f"{feature_label} (Raw)",
                format=",.3f",
            ),
            alt.Tooltip(
                "x_value:Q",
                title=metrics["x_axis_label"],
                format=".3f",
            ),
            alt.Tooltip(
                "y_value:Q",
                title=target_label,
                format=".3f",
            ),
        ]
    )

    points = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.25, size=40)
        .encode(
            x=alt.X("x_value:Q", title=metrics["x_axis_label"]),
            y=alt.Y("y_value:Q", title=target_label),
            tooltip=tooltip_fields,
        )
    )

    title_text = f"{feature_label} vs {target_label}"
    subtitle_text = (
        f"Rank #{feature_rank['rank']} by absolute Pearson correlation | "
        f"r = {feature_rank['corr']:.3f} | "
        f"R² = {feature_rank['r_squared']:.3f}"
    )

    chart = points

    if show_trendline:
        line = (
            alt.Chart(line_df)
            .mark_line(strokeWidth=3)
            .encode(
                x="x_value:Q",
                y="y_value:Q",
            )
        )
        chart = points + line

    return chart.properties(
        width=750,
        height=500,
        title={
            "text": title_text,
            "subtitle": [subtitle_text],
        },
    )


def create_freeform_scatter_chart(
    plot_df: pd.DataFrame,
    line_df: pd.DataFrame | None,
    metrics: dict,
    show_trendline: bool,
) -> alt.Chart:
    """
    Create a freeform scatterplot for arbitrary numeric X and Y columns.

    Args:
        plot_df: Plot dataframe.
        line_df: Optional fitted line dataframe.
        metrics: Summary metrics dictionary.
        show_trendline: Whether to overlay the fitted line.

    Returns:
        alt.Chart: Scatterplot, optionally with trendline.
    """
    tooltip_fields = []

    if "film_title" in plot_df.columns:
        tooltip_fields.append(alt.Tooltip("film_title:N", title="Film"))

    if "release_group_name" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("release_group_name:N", title="Soundtrack")
        )
    elif "album_title" in plot_df.columns:
        tooltip_fields.append(alt.Tooltip("album_title:N", title="Soundtrack"))
    elif "soundtrack_title" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("soundtrack_title:N", title="Soundtrack")
        )

    if "composer_primary_clean" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("composer_primary_clean:N", title="Composer")
        )

    tooltip_fields.extend(
        [
            alt.Tooltip(
                "x_raw:Q",
                title=f"{get_display_label(metrics['x_col'])} (Raw)",
                format=",.3f",
            ),
            alt.Tooltip(
                "y_raw:Q",
                title=f"{get_display_label(metrics['y_col'])} (Raw)",
                format=",.3f",
            ),
        ]
    )

    if metrics["transform_x"] != "None":
        tooltip_fields.append(
            alt.Tooltip(
                "x_value:Q",
                title=metrics["x_axis_title"],
                format=",.3f",
            )
        )

    if metrics["transform_y"] != "None":
        tooltip_fields.append(
            alt.Tooltip(
                "y_value:Q",
                title=metrics["y_axis_title"],
                format=",.3f",
            )
        )

    if metrics["color_col"] and metrics["color_col"] != "None":
        color_label = get_display_label(metrics["color_col"])
        if pd.api.types.is_numeric_dtype(plot_df[metrics["color_col"]]):
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metrics['color_col']}:Q",
                    title=color_label,
                    format=",.3f",
                )
            )
        else:
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metrics['color_col']}:N",
                    title=color_label,
                )
            )

    base = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.35, size=45)
        .encode(
            x=alt.X("x_plot:Q", title=metrics["x_axis_title"]),
            y=alt.Y("y_plot:Q", title=metrics["y_axis_title"]),
            tooltip=tooltip_fields,
        )
    )

    if metrics["color_col"] and metrics["color_col"] != "None":
        if metrics["color_col"] == "album_genre_group":
            points = base.encode(
                color=alt.Color(
                    "album_genre_group:N",
                    title=get_display_label("album_genre_group"),
                    scale=alt.Scale(domain=ALBUM_GENRE_DOMAIN),
                )
            )
        elif metrics["color_col"] == "film_genre_group":
            points = base.encode(
                color=alt.Color(
                    "film_genre_group:N",
                    title=get_display_label("film_genre_group"),
                    scale=alt.Scale(domain=FILM_GENRE_DOMAIN),
                )
            )
        else:
            points = base.encode(
                color=alt.Color(
                    f"{metrics['color_col']}:N",
                    title=get_display_label(metrics["color_col"]),
                )
            )
    else:
        points = base

    title_text = f"{metrics['x_axis_title']} vs {metrics['y_axis_title']}"
    subtitle_parts = [
        f"Rows used: {metrics['rows_used']:,}",
        f"Pearson r = {metrics['pearson_r']:.3f}",
        f"R² = {metrics['r_squared']:.3f}",
    ]

    if metrics["color_col"] and metrics["color_col"] != "None":
        subtitle_parts.append(
            f"Color = {get_display_label(metrics['color_col'])}"
        )

    if metrics["apply_jitter"]:
        subtitle_parts.append(
            f"Jitter = {metrics['jitter_strength']:.3f} (display only)"
        )

    if (
        metrics["transform_x"] != "None"
        or metrics["transform_y"] != "None"
    ):
        subtitle_parts.append("Metrics and fitted line use displayed scale")

    chart = points

    if show_trendline and line_df is not None:
        line = (
            alt.Chart(line_df)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X("x_plot:Q"),
                y=alt.Y("y_plot:Q"),
            )
        )
        chart = points + line

    return chart.properties(
        width=750,
        height=500,
        title={
            "text": title_text,
            "subtitle": [" | ".join(subtitle_parts)],
        },
    )


def render_guided_summary_metrics(
    metrics: dict,
    feature_rank: dict,
) -> None:
    """
    Render headline metrics for the guided feature view.

    Args:
        metrics: Metrics dictionary from the scatter helper.
        feature_rank: Ranking metadata for the selected feature.
    """
    direction = feature_rank["direction"].title()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Selected Feature", get_display_label(metrics["feature_col"]))

    with col2:
        st.metric("Rank", f"#{feature_rank['rank']}")

    with col3:
        st.metric("Rows Used", f"{metrics['rows_used']:,}")

    with col4:
        st.metric("Pearson r", f"{feature_rank['corr']:.3f}")

    with col5:
        st.metric("Univariate R²", f"{feature_rank['r_squared']:.3f}")

    st.caption(
        f"Direction: {direction}. "
        f"X-axis shown on transformed modeling scale; raw values remain in the tooltip."
    )


def render_freeform_summary_metrics(
    metrics: dict,
) -> None:
    """
    Render headline metrics for freeform mode.

    Args:
        metrics: Freeform metrics dictionary.
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("X-Axis", get_display_label(metrics["x_col"]))

    with col2:
        st.metric("Y-Axis", get_display_label(metrics["y_col"]))

    with col3:
        st.metric("Rows Used", f"{metrics['rows_used']:,}")

    with col4:
        st.metric("Pearson r", f"{metrics['pearson_r']:.3f}")

    with col5:
        color_value = (
            get_display_label(metrics["color_col"])
            if metrics["color_col"] != "None"
            else "None"
        )
        st.metric("Color", color_value)

    st.caption(
        "Each point represents one album. Freeform mode uses album-level numeric fields "
        "for X and Y and optional low-cardinality grouping for color."
    )


def main() -> None:
    """
    Run the relationship explorer page.
    """
    st.set_page_config(
        page_title="Relationship Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Relationship Explorer")
    st.write(
        """
        Explore album-level relationships in two ways. Guided mode follows the
        regression-oriented feature ranking against `log_lfm_album_listeners`.
        Freeform mode lets you compare any safe album-level numeric X and Y
        fields, with an optional color grouping.

        Each point represents one album.
        """
    )

    albums_df, _ = load_source_data()
    album_analytics_df = load_analysis_data()
    explorer_source_df = load_explorer_data()

    ranking_df = reg.build_scatterplot_feature_ranking(
        album_analytics_df=album_analytics_df,
        target_col=dp.TARGET_COL,
        method="pearson",
    )
    rank_lookup = build_feature_rank_lookup(ranking_df)

    metadata_cols = pick_available_metadata_cols(albums_df)

    explorer_df = build_relationship_explorer_df(
        explorer_source_df=explorer_source_df,
    )

    guided_feature_options = ranking_df["feature"].tolist()
    freeform_numeric_options = get_freeform_numeric_options(explorer_df)
    color_options = get_color_options(explorer_df)

    freeform_default_y = (
        "lfm_album_listeners"
        if "lfm_album_listeners" in freeform_numeric_options
        else dp.TARGET_COL
    )

    controls = get_scatter_controls(
        guided_feature_options=guided_feature_options,
        freeform_numeric_options=freeform_numeric_options,
        color_options=color_options,
        default_y=freeform_default_y,
    )

    if controls["mode"] == "Guided":
        selected_feature = controls["selected_feature"]
        feature_rank = rank_lookup[selected_feature]

        plot_df, line_df, metrics = reg.build_exploratory_scatter_data(
            album_analytics_df=album_analytics_df,
            feature_col=selected_feature,
            metadata_df=albums_df,
            metadata_cols=metadata_cols,
            id_cols=["tmdb_id", "release_group_mbid"],
            target_col=dp.TARGET_COL,
        )

        render_guided_summary_metrics(
            metrics=metrics,
            feature_rank=feature_rank,
        )

        st.subheader("Scatterplot")
        chart = create_guided_scatter_chart(
            plot_df=plot_df,
            line_df=line_df,
            metrics=metrics,
            feature_rank=feature_rank,
            show_trendline=controls["show_trendline"],
        )
        st.altair_chart(chart, width="stretch")

        if controls["show_data_table"]:
            st.subheader("Scatterplot Source Data")
            st.dataframe(rename_columns_for_display(plot_df), width="stretch")

        if controls["show_feature_ranking"]:
            st.subheader("Ranked Feature Table")
            st.dataframe(
                rename_columns_for_display(ranking_df.round(3)),
                width="stretch",
            )

    else:
        x_col = controls["x_col"]
        y_col = controls["y_col"]
        color_col = controls["color_col"]

        if x_col == y_col:
            st.warning("Please choose different X-axis and Y-axis fields.")
            return

        plot_df, line_df, metrics = build_freeform_scatter_data(
            explorer_df=explorer_df,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            transform_x=controls["transform_x"],
            transform_y=controls["transform_y"],
            apply_jitter=controls["apply_jitter"],
            jitter_strength=controls["jitter_strength"],
        )

        render_freeform_summary_metrics(metrics)

        st.subheader("Scatterplot")
        chart = create_freeform_scatter_chart(
            plot_df=plot_df,
            line_df=line_df,
            metrics=metrics,
            show_trendline=controls["show_trendline"],
        )
        st.altair_chart(chart, width="stretch")

        if controls["show_data_table"]:
            st.subheader("Scatterplot Source Data")
            st.dataframe(rename_columns_for_display(plot_df), width="stretch")


if __name__ == "__main__":
    main()