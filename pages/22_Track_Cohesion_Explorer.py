from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import get_global_filter_controls
from app.app_data import load_track_audio_cohesion_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_film_year_bucket,
    add_standard_multivalue_groups,
    get_clean_composer_options,
    get_global_filter_inputs,
    rename_and_dedupe_for_display,
    select_unique_existing_columns,
)
from app.ui import apply_app_styles


METRIC_SPECS = {
    "Energy cohesion — track-to-track spread": {
        "col": "energy_variance_proxy_std",
        "family": "Energy cohesion",
        "short_label": "Energy variability",
        "description": (
            "Measures how much track energy changes across the album. "
            "Lower values mean a more consistent energy profile."
        ),
        "default_rank": 1,
    },
    "Rhythmic cohesion — danceability spread": {
        "col": "danceability_variance_proxy_std",
        "family": "Rhythmic cohesion",
        "short_label": "Danceability variability",
        "description": (
            "Measures how much danceability shifts across tracks. "
            "Lower values mean a more rhythmically consistent soundtrack."
        ),
        "default_rank": 2,
    },
    "Rhythmic cohesion — tempo spread": {
        "col": "tempo_variance_proxy_std",
        "family": "Rhythmic cohesion",
        "short_label": "Tempo variability",
        "description": (
            "Measures how much tempo fluctuates across tracks. "
            "Lower values mean a more rhythmically stable soundtrack."
        ),
        "default_rank": 3,
    },
    "Rhythmic cohesion — tempo range": {
        "col": "tempo_range",
        "family": "Rhythmic cohesion",
        "short_label": "Tempo range",
        "description": (
            "Measures the spread between the slowest and fastest visible tracks. "
            "Lower values indicate a tighter tempo band."
        ),
        "default_rank": 4,
    },
    "Mood cohesion — happiness spread": {
        "col": "happiness_variance_proxy_std",
        "family": "Mood cohesion",
        "short_label": "Happiness variability",
        "description": (
            "Measures how much emotional brightness or mood changes across tracks. "
            "Lower values mean a more tonally consistent album."
        ),
        "default_rank": 5,
    },
    "Texture cohesion — instrumentalness spread": {
        "col": "instrumentalness_variance_proxy_std",
        "family": "Texture cohesion",
        "short_label": "Instrumentalness variability",
        "description": (
            "Measures how much the soundtrack shifts between more vocal and more "
            "instrumental tracks. Lower values indicate a more consistent texture."
        ),
        "default_rank": 6,
    },
    "Harmonic cohesion — key entropy": {
        "col": "key_entropy",
        "family": "Harmonic cohesion",
        "short_label": "Key entropy",
        "description": (
            "Measures how dispersed the album is across musical keys. "
            "Lower values indicate stronger key consistency."
        ),
        "default_rank": 7,
    },
    "Harmonic cohesion — mode entropy": {
        "col": "mode_entropy",
        "family": "Harmonic cohesion",
        "short_label": "Mode entropy",
        "description": (
            "Measures how mixed the album is between major and minor mode. "
            "Lower values indicate stronger harmonic consistency."
        ),
        "default_rank": 8,
    },
}

ALBUM_OUTCOME_OPTIONS = {
    "Album listeners": "lfm_album_listeners",
    "Album playcount": "lfm_album_playcount",
}

COLOR_OPTIONS = {
    "Album genre": "album_genre_group",
    "Film genre": "film_genre_group",
    "Film year bucket": "film_year_bucket",
    "Percent instrumental tracks": "pct_instrumental_tracks",
    "Composer": "composer_primary_clean",
    "None": None,
}


def add_cohesion_display_fields(cohesion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add page-level grouped display fields used for filtering and coloring.

    Args:
        cohesion_df: Album-level cohesion dataframe.

    Returns:
        pd.DataFrame: Copy with grouped fields added.
    """
    df = add_standard_multivalue_groups(cohesion_df)
    df = add_film_year_bucket(df)

    if "composer_primary_clean" in df.columns:
        df["composer_primary_clean"] = (
            df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


def get_track_cohesion_controls(composer_options: list[str]) -> dict:
    """
    Build sidebar controls for the Track Cohesion Explorer.

    Args:
        composer_options: Available composer values.

    Returns:
        dict: Selected page controls.
    """
    st.sidebar.header("Track Cohesion Controls")

    metric_labels = sorted(
        METRIC_SPECS.keys(),
        key=lambda x: METRIC_SPECS[x]["default_rank"],
    )

    cohesion_metric_label = st.sidebar.selectbox(
        "Cohesion metric",
        options=metric_labels,
        index=0,
        help=(
            "Choose which within-album cohesion dimension to compare "
            "against popularity and dominance."
        ),
    )

    album_outcome_label = st.sidebar.selectbox(
        "Album outcome",
        options=list(ALBUM_OUTCOME_OPTIONS.keys()),
        index=0,
        help="Choose the album-level success metric for the popularity chart.",
    )

    color_label = st.sidebar.selectbox(
        "Color by",
        options=list(COLOR_OPTIONS.keys()),
        index=0,
    )

    selected_composers = st.sidebar.multiselect(
        "Composers",
        options=composer_options,
        default=[],
        help="Optionally restrict the page to a selected composer subset.",
    )

    use_log_scale = st.sidebar.checkbox(
        "Log-scale album outcome",
        value=True,
        help="Apply log10 scaling to the album outcome chart.",
    )

    min_tracks = st.sidebar.slider(
        "Minimum tracks per album",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Hide albums with too few observed tracks for stable cohesion estimates.",
    )

    show_strength_chart = st.sidebar.checkbox(
        "Show metric strength comparison",
        value=True,
    )

    show_binned_view = st.sidebar.checkbox(
        "Show low / medium / high bins",
        value=True,
    )

    show_table = st.sidebar.checkbox(
        "Show source table",
        value=False,
    )

    return {
        "cohesion_metric_label": cohesion_metric_label,
        "cohesion_metric_col": METRIC_SPECS[cohesion_metric_label]["col"],
        "cohesion_metric_family": METRIC_SPECS[cohesion_metric_label]["family"],
        "cohesion_metric_short_label": METRIC_SPECS[cohesion_metric_label]["short_label"],
        "cohesion_metric_description": METRIC_SPECS[cohesion_metric_label]["description"],
        "album_outcome_label": album_outcome_label,
        "album_outcome_col": ALBUM_OUTCOME_OPTIONS[album_outcome_label],
        "color_label": color_label,
        "color_col": COLOR_OPTIONS[color_label],
        "selected_composers": selected_composers,
        "use_log_scale": use_log_scale,
        "min_tracks": min_tracks,
        "show_strength_chart": show_strength_chart,
        "show_binned_view": show_binned_view,
        "show_table": show_table,
    }


def filter_cohesion_df(
    cohesion_df: pd.DataFrame,
    global_controls: dict,
    controls: dict,
) -> pd.DataFrame:
    """
    Apply global and page-level filters to the cohesion dataframe.

    Args:
        cohesion_df: Album-level cohesion dataframe.
        global_controls: Shared global filter values.
        controls: Page control values.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df = filter_dataset(cohesion_df, global_controls).copy()

    if controls["selected_composers"]:
        df = df[
            df["composer_primary_clean"].isin(controls["selected_composers"])
        ].copy()

    if "n_tracks" in df.columns:
        df = df[df["n_tracks"] >= controls["min_tracks"]].copy()

    required_cols = [
        controls["cohesion_metric_col"],
        controls["album_outcome_col"],
        "top_to_total_listeners",
    ]
    required_cols = [col for col in required_cols if col in df.columns]

    df = df.dropna(subset=required_cols).copy()

    outcome_col = controls["album_outcome_col"]
    if controls["use_log_scale"] and outcome_col in df.columns:
        df = df[df[outcome_col] > 0].copy()
        df["album_outcome_display"] = np.log10(df[outcome_col])
        df["album_outcome_display_label"] = f"{controls['album_outcome_label']} (log10)"
    else:
        df["album_outcome_display"] = df[outcome_col]
        df["album_outcome_display_label"] = controls["album_outcome_label"]

    return df


def summarize_metric_relationships(
    plot_df: pd.DataFrame,
    outcome_col: str = "album_outcome_display",
) -> pd.DataFrame:
    """
    Summarize how each cohesion metric relates to the displayed outcome.

    Args:
        plot_df: Filtered cohesion dataframe.
        outcome_col: Outcome column used for correlation.

    Returns:
        pd.DataFrame: One row per available cohesion metric.
    """
    rows = []

    for metric_label, spec in METRIC_SPECS.items():
        metric_col = spec["col"]
        if metric_col not in plot_df.columns or outcome_col not in plot_df.columns:
            continue

        subset = plot_df[[metric_col, outcome_col]].dropna().copy()
        if subset.empty:
            continue

        corr = subset[metric_col].corr(subset[outcome_col], method="pearson")
        if pd.isna(corr):
            continue

        rows.append(
            {
                "metric_label": metric_label,
                "metric_short_label": spec["short_label"],
                "metric_family": spec["family"],
                "metric_col": metric_col,
                "pearson_r": float(corr),
                "abs_pearson_r": float(abs(corr)),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    return summary_df.sort_values(
        ["abs_pearson_r", "metric_label"],
        ascending=[False, True],
    ).reset_index(drop=True)


def build_variability_bins(
    plot_df: pd.DataFrame,
    cohesion_col: str,
) -> pd.DataFrame:
    """
    Create low / medium / high variability bins for the selected cohesion metric.

    Args:
        plot_df: Filtered cohesion dataframe.
        cohesion_col: Selected cohesion metric column.

    Returns:
        pd.DataFrame: Copy with a cohesion_bin column added.
    """
    df = plot_df.copy()

    valid = df[cohesion_col].dropna()
    if valid.empty or valid.nunique() < 3:
        df["cohesion_bin"] = "Unbinned"
        return df

    q1 = valid.quantile(1 / 3)
    q2 = valid.quantile(2 / 3)

    def assign_bin(val: float) -> str:
        if pd.isna(val):
            return "Unknown"
        if val <= q1:
            return "Low variability"
        if val <= q2:
            return "Medium variability"
        return "High variability"

    df["cohesion_bin"] = df[cohesion_col].apply(assign_bin)
    return df


def build_binned_summary_df(
    plot_df: pd.DataFrame,
    raw_outcome_col: str,
    display_outcome_col: str,
) -> pd.DataFrame:
    """
    Summarize album outcomes by low / medium / high variability bin.

    Args:
        plot_df: Binned cohesion dataframe.
        raw_outcome_col: Raw outcome column.
        display_outcome_col: Displayed outcome column.

    Returns:
        pd.DataFrame: One row per cohesion bin.
    """
    if "cohesion_bin" not in plot_df.columns:
        return pd.DataFrame()

    summary_df = (
        plot_df.groupby("cohesion_bin", as_index=False)
        .agg(
            album_count=("cohesion_bin", "size"),
            mean_raw_outcome=(raw_outcome_col, "mean"),
            median_raw_outcome=(raw_outcome_col, "median"),
            mean_display_outcome=(display_outcome_col, "mean"),
            median_display_outcome=(display_outcome_col, "median"),
            mean_top_track_share=("top_to_total_listeners", "mean"),
        )
    )

    order = ["Low variability", "Medium variability", "High variability", "Unbinned"]
    summary_df["bin_order"] = summary_df["cohesion_bin"].map(
        {name: idx for idx, name in enumerate(order)}
    )
    summary_df = summary_df.sort_values("bin_order").drop(columns=["bin_order"])
    return summary_df


def build_cohesion_scope_caption(
    plot_df: pd.DataFrame,
    controls: dict,
) -> str:
    """
    Build a short caption summarizing the current analysis scope.

    Args:
        plot_df: Filtered dataframe used in the charts.
        controls: Page control values.

    Returns:
        str: Human-readable scope caption.
    """
    parts = [
        f"{len(plot_df):,} albums in view",
        f"cohesion metric: {controls['cohesion_metric_short_label'].lower()}",
        f"album outcome: {controls['album_outcome_label'].lower()}",
        f"minimum {controls['min_tracks']} observed tracks per album",
    ]

    if controls["color_label"] != "None":
        parts.append(f"color: {controls['color_label'].lower()}")

    if controls["selected_composers"]:
        parts.append(f"{len(controls['selected_composers'])} selected composers")

    if controls["use_log_scale"]:
        parts.append("log10-scaled album outcome")

    return "Current scope: " + "; ".join(parts) + "."


def get_variability_explainer(controls: dict) -> str:
    """
    Explain what the selected cohesion metric means.

    Args:
        controls: Page control values.

    Returns:
        str: Human-readable explainer.
    """
    return (
        f"**{controls['cohesion_metric_family']}** · "
        f"{controls['cohesion_metric_description']} "
        "Lower values indicate greater within-album consistency."
    )


def classify_relationship_direction(corr: float) -> str:
    """
    Convert a correlation into a qualitative interpretation.

    Args:
        corr: Pearson correlation.

    Returns:
        str: Interpretation label.
    """
    if pd.isna(corr):
        return "No stable relationship"
    if corr <= -0.30:
        return "Clear negative relationship"
    if corr <= -0.10:
        return "Mild negative relationship"
    if corr >= 0.30:
        return "Clear positive relationship"
    if corr >= 0.10:
        return "Mild positive relationship"
    return "Weak / near-flat relationship"


def build_track_cohesion_insight_summary(
    plot_df: pd.DataFrame,
    controls: dict,
    metric_summary_df: pd.DataFrame,
) -> dict[str, str]:
    """
    Build top-row insight cards for the Track Cohesion Explorer.

    Args:
        plot_df: Filtered cohesion dataframe.
        controls: Page control values.
        metric_summary_df: Cross-metric relationship summary.

    Returns:
        dict[str, str]: Titles, values, and captions for the top-row cards.
    """
    if plot_df.empty:
        return {
            "card1_title": "Selected Relationship",
            "card1_value": "None",
            "card1_caption": "No albums remain under the current filters.",
            "card2_title": "Direction",
            "card2_value": "None",
            "card2_caption": "No relationship is visible in the current view.",
            "card3_title": "Best Predictor",
            "card3_value": "None",
            "card3_caption": "No cohesion metric stands out in the current view.",
        }

    cohesion_col = controls["cohesion_metric_col"]

    corr = plot_df[cohesion_col].corr(plot_df["album_outcome_display"])

    relationship_value = f"r = {corr:.3f}" if pd.notna(corr) else "NA"
    direction_value = classify_relationship_direction(corr)

    if metric_summary_df.empty:
        best_value = "None"
        best_caption = "No cohesion metric stands out."
    else:
        best_row = metric_summary_df.iloc[0]
        best_value = best_row["metric_short_label"]
        best_caption = (
            f"{best_row['metric_short_label']} shows the strongest relationship "
            f"with {controls['album_outcome_label'].lower()} (r = {best_row['pearson_r']:.3f}). "
            "This means this dimension of cohesion is the most informative in the current view."
        )

    return {
        "card1_title": "Selected Relationship",
        "card1_value": relationship_value,
        "card1_caption": (
            f"{controls['cohesion_metric_short_label']} vs {controls['album_outcome_label'].lower()}. "
            "This means the chart is showing how changes in cohesion relate to album performance."
        ),
        "card2_title": "Direction",
        "card2_value": direction_value,
        "card2_caption": (
            "This means negative relationships favor more cohesive albums, "
            "while positive relationships favor more varied albums."
        ),
        "card3_title": "Best Predictor",
        "card3_value": best_value,
        "card3_caption": best_caption,
    }


def render_track_cohesion_insight_cards(
    plot_df: pd.DataFrame,
    controls: dict,
    metric_summary_df: pd.DataFrame,
) -> None:
    """
    Render top-row insight cards.

    Args:
        plot_df: Filtered cohesion dataframe.
        controls: Page control values.
        metric_summary_df: Cross-metric relationship summary.
    """
    insights = build_track_cohesion_insight_summary(
        plot_df=plot_df,
        controls=controls,
        metric_summary_df=metric_summary_df,
    )

    st.markdown("### 🧠 Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(insights["card1_title"], insights["card1_value"])
        st.caption(insights["card1_caption"])

    with col2:
        st.metric(insights["card2_title"], insights["card2_value"])
        st.caption(insights["card2_caption"])

    with col3:
        st.metric(insights["card3_title"], insights["card3_value"])
        st.caption(insights["card3_caption"])


def build_page_level_narrative(
    plot_df: pd.DataFrame,
    controls: dict,
    metric_summary_df: pd.DataFrame,
) -> str:
    """
    Build an executive-style narrative summary for the current page state.

    Args:
        plot_df: Filtered cohesion dataframe.
        controls: Page control values.
        metric_summary_df: Cross-metric relationship summary.

    Returns:
        str: Narrative caption.
    """
    if plot_df.empty:
        return "No albums remain under the current filters."

    corr = plot_df[controls["cohesion_metric_col"]].corr(
        plot_df["album_outcome_display"]
    )

    if pd.isna(corr):
        main_read = (
            f"There is no stable relationship between "
            f"{controls['cohesion_metric_short_label'].lower()} and "
            f"{controls['album_outcome_label'].lower()} in this view."
        )
    elif corr <= -0.20:
        main_read = (
            f"More cohesive albums tend to have higher "
            f"{controls['album_outcome_label'].lower()}."
        )
    elif corr >= 0.20:
        main_read = (
            f"More varied albums tend to have higher "
            f"{controls['album_outcome_label'].lower()}."
        )
    else:
        main_read = (
            f"{controls['cohesion_metric_short_label']} shows only a weak relationship "
            f"with {controls['album_outcome_label'].lower()}."
        )

    if metric_summary_df.empty:
        cross = ""
    else:
        neg = (metric_summary_df["pearson_r"] < 0).sum()
        total = len(metric_summary_df)
        best = metric_summary_df.iloc[0]

        cross = (
            f" Across cohesion metrics, {neg} of {total} show a negative slope. "
            f"The strongest signal comes from {best['metric_short_label'].lower()}."
        )

    return (
        f"💡 {main_read}{cross} "
        "This means cohesion can matter, but its impact depends on the specific audio dimension."
    )


def _build_color_encoding(color_col: str, color_label: str):
    """
    Build a color encoding for Altair charts.

    Args:
        color_col: Column used for color.
        color_label: Display label for color.

    Returns:
        alt.Color | None: Altair color encoding or None.
    """
    if color_col is None:
        return None

    if color_col == "pct_instrumental_tracks":
        return alt.Color(f"{color_col}:Q", title=color_label)

    return alt.Color(f"{color_col}:N", title=color_label)


def plot_cohesion_vs_album_outcome(
    plot_df: pd.DataFrame,
    controls: dict,
) -> alt.Chart:
    """
    Plot cohesion versus album success.

    Args:
        plot_df: Filtered cohesion dataframe.
        controls: Page control values.

    Returns:
        alt.Chart: Scatter chart.
    """
    cohesion_col = controls["cohesion_metric_col"]
    color_col = controls["color_col"]
    y_title = plot_df["album_outcome_display_label"].iloc[0]

    tooltip = [
        alt.Tooltip("film_title:N", title="Film"),
        alt.Tooltip("album_title:N", title="Album"),
        alt.Tooltip("composer_primary_clean:N", title="Composer"),
        alt.Tooltip(
            f"{cohesion_col}:Q",
            title=controls["cohesion_metric_short_label"],
            format=".3f",
        ),
        alt.Tooltip(
            f"{controls['album_outcome_col']}:Q",
            title=controls["album_outcome_label"],
            format=",.0f",
        ),
        alt.Tooltip("album_outcome_display:Q", title=y_title, format=".3f"),
        alt.Tooltip(
            "top_to_total_listeners:Q",
            title="Top Track / Total Listeners",
            format=".3f",
        ),
        alt.Tooltip("n_tracks:Q", title="Track Count", format=",.0f"),
    ]

    if color_col is not None:
        tooltip.append(
            alt.Tooltip(
                f"{color_col}:{'Q' if color_col == 'pct_instrumental_tracks' else 'N'}",
                title=controls["color_label"],
            )
        )

    base = alt.Chart(plot_df).mark_circle(
        size=55,
        opacity=0.35,
    ).encode(
        x=alt.X(
            f"{cohesion_col}:Q",
            title=controls["cohesion_metric_short_label"],
        ),
        y=alt.Y(
            "album_outcome_display:Q",
            title=y_title,
        ),
        tooltip=tooltip,
    )

    color_encoding = _build_color_encoding(color_col, controls["color_label"])
    if color_encoding is not None:
        base = base.encode(color=color_encoding)

    trend = (
        alt.Chart(plot_df)
        .transform_regression(cohesion_col, "album_outcome_display")
        .mark_line(strokeDash=[6, 4], opacity=0.85)
        .encode(
            x=alt.X(f"{cohesion_col}:Q"),
            y=alt.Y("album_outcome_display:Q"),
        )
    )

    return (base + trend).properties(height=420)


def plot_cohesion_vs_top_track_dominance(
    plot_df: pd.DataFrame,
    controls: dict,
) -> alt.Chart:
    """
    Plot cohesion versus top-track dominance.

    Args:
        plot_df: Filtered cohesion dataframe.
        controls: Page control values.

    Returns:
        alt.Chart: Scatter chart.
    """
    cohesion_col = controls["cohesion_metric_col"]
    color_col = controls["color_col"]

    tooltip = [
        alt.Tooltip("film_title:N", title="Film"),
        alt.Tooltip("album_title:N", title="Album"),
        alt.Tooltip("composer_primary_clean:N", title="Composer"),
        alt.Tooltip(
            f"{cohesion_col}:Q",
            title=controls["cohesion_metric_short_label"],
            format=".3f",
        ),
        alt.Tooltip(
            "top_to_total_listeners:Q",
            title="Top Track / Total Listeners",
            format=".3f",
        ),
        alt.Tooltip(
            "lfm_album_listeners:Q",
            title="Album Listeners",
            format=",.0f",
        ),
        alt.Tooltip("n_tracks:Q", title="Track Count", format=",.0f"),
    ]

    if color_col is not None:
        tooltip.append(
            alt.Tooltip(
                f"{color_col}:{'Q' if color_col == 'pct_instrumental_tracks' else 'N'}",
                title=controls["color_label"],
            )
        )

    base = alt.Chart(plot_df).mark_circle(
        size=55,
        opacity=0.35,
    ).encode(
        x=alt.X(
            f"{controls['cohesion_metric_col']}:Q",
            title=controls["cohesion_metric_short_label"],
        ),
        y=alt.Y(
            "top_to_total_listeners:Q",
            title="Top Track / Total Listeners",
        ),
        tooltip=tooltip,
    )

    color_encoding = _build_color_encoding(color_col, controls["color_label"])
    if color_encoding is not None:
        base = base.encode(color=color_encoding)

    trend = (
        alt.Chart(plot_df)
        .transform_regression(controls["cohesion_metric_col"], "top_to_total_listeners")
        .mark_line(strokeDash=[6, 4], opacity=0.85)
        .encode(
            x=alt.X(f"{controls['cohesion_metric_col']}:Q"),
            y=alt.Y("top_to_total_listeners:Q"),
        )
    )

    return (base + trend).properties(height=420)


def plot_metric_strength_chart(
    metric_summary_df: pd.DataFrame,
    outcome_label: str,
) -> alt.Chart:
    """
    Plot cohesion metrics ranked by relationship strength.

    Args:
        metric_summary_df: Cross-metric summary dataframe.
        outcome_label: Selected outcome label.

    Returns:
        alt.Chart: Horizontal bar chart.
    """
    plot_df = metric_summary_df.copy()

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "metric_short_label:N",
                sort=alt.SortField(field="abs_pearson_r", order="descending"),
                title="Cohesion metric",
            ),
            x=alt.X(
                "pearson_r:Q",
                title=f"Pearson r vs {outcome_label}",
            ),
            color=alt.Color(
                "metric_family:N",
                title="Metric family",
            ),
            tooltip=[
                alt.Tooltip("metric_short_label:N", title="Metric"),
                alt.Tooltip("metric_family:N", title="Family"),
                alt.Tooltip("pearson_r:Q", title="Pearson r", format=".3f"),
                alt.Tooltip("abs_pearson_r:Q", title="Abs Pearson r", format=".3f"),
            ],
        )
        .properties(height=320)
    )


def plot_binned_outcome_chart(
    binned_summary_df: pd.DataFrame,
    display_outcome_label: str,
) -> alt.Chart:
    """
    Plot mean displayed album outcome across low / medium / high variability bins.

    Args:
        binned_summary_df: Binned summary dataframe.
        display_outcome_label: Display outcome label.

    Returns:
        alt.Chart: Bar chart.
    """
    order = ["Low variability", "Medium variability", "High variability", "Unbinned"]

    return (
        alt.Chart(binned_summary_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "cohesion_bin:N",
                sort=order,
                title="Variability bin",
            ),
            y=alt.Y(
                "mean_display_outcome:Q",
                title=f"Mean {display_outcome_label}",
            ),
            tooltip=[
                alt.Tooltip("cohesion_bin:N", title="Variability bin"),
                alt.Tooltip("album_count:Q", title="Albums", format=",.0f"),
                alt.Tooltip("mean_raw_outcome:Q", title="Mean Raw Outcome", format=",.0f"),
                alt.Tooltip("median_raw_outcome:Q", title="Median Raw Outcome", format=",.0f"),
                alt.Tooltip("mean_display_outcome:Q", title=f"Mean {display_outcome_label}", format=".3f"),
                alt.Tooltip("median_display_outcome:Q", title=f"Median {display_outcome_label}", format=".3f"),
                alt.Tooltip("mean_top_track_share:Q", title="Mean Top Track / Total Listeners", format=".3f"),
            ],
        )
        .properties(height=320)
    )

def build_popularity_supporting_insight(
    plot_df: pd.DataFrame,
    controls: dict,
) -> str:
    """
    Build a supporting insight for cohesion vs album popularity.

    Args:
        plot_df: Filtered cohesion dataframe.
        controls: Page control values.

    Returns:
        str: Supporting sentence.
    """
    if plot_df.empty:
        return "No popularity insight is available."

    corr = plot_df[controls["cohesion_metric_col"]].corr(
        plot_df["album_outcome_display"]
    )

    if pd.isna(corr):
        return "No stable relationship is visible."

    if corr <= -0.20:
        read = "more cohesive albums tend to perform better"
    elif corr >= 0.20:
        read = "more varied albums tend to perform better"
    else:
        read = "cohesion has only a weak relationship with performance"

    return (
        f"💡 In this view, {read} (r = {corr:.3f}). "
        f"This means changes in {controls['cohesion_metric_short_label'].lower()} "
        f"are associated with differences in {controls['album_outcome_label'].lower()}."
    )

def build_dominance_supporting_insight(
    plot_df: pd.DataFrame,
    controls: dict,
) -> str:
    """
        Build a supporting insight for cohesion vs top-track dominance.

        Args:
            plot_df: Filtered cohesion dataframe.
            controls: Page control values.

        Returns:
            str: Supporting sentence.
        """
    if plot_df.empty:
        return "No dominance insight is available."

    corr = plot_df[controls["cohesion_metric_col"]].corr(
        plot_df["top_to_total_listeners"]
    )

    if pd.isna(corr):
        return "No stable relationship is visible."

    if corr <= -0.20:
        read = "more cohesive albums tend to concentrate listening into fewer tracks"
    elif corr >= 0.20:
        read = "more varied albums tend to concentrate listening into fewer tracks"
    else:
        read = "cohesion has only a weak relationship with track dominance"

    return (
        f"💡 In this view, {read} (r = {corr:.3f}). "
        "This means cohesion may influence whether an album relies on a standout track."
    )

def build_metric_strength_supporting_insight(
    metric_summary_df: pd.DataFrame,
    outcome_label: str,
) -> str:
    """
        Build a supporting insight for the metric-strength ranking chart.

        Args:
            metric_summary_df: Cross-metric relationship summary.
            outcome_label: Selected outcome label.

        Returns:
            str: Supporting sentence.
        """
    if metric_summary_df.empty:
        return "No cross-metric comparison is available."

    best = metric_summary_df.iloc[0]
    neg = (metric_summary_df["pearson_r"] < 0).sum()
    total = len(metric_summary_df)

    return (
        f"💡 {best['metric_short_label']} is the strongest predictor of "
        f"{outcome_label.lower()} (r = {best['pearson_r']:.3f}). "
        f"{neg} of {total} metrics show a negative relationship. "
        "This means some dimensions of cohesion matter more than others."
    )

def build_binned_supporting_insight(
    binned_summary_df: pd.DataFrame,
    display_outcome_label: str,
) -> str:
    """
        Build a reader-facing insight for the variability-bin view.

        Args:
            binned_summary_df: Binned summary dataframe.
            display_outcome_label: Display outcome label.

        Returns:
            str: Supporting sentence.
        """
    if binned_summary_df.empty:
        return "No variability-bin summary is available."

    lookup = {row["cohesion_bin"]: row for _, row in binned_summary_df.iterrows()}

    low = lookup.get("Low variability")
    high = lookup.get("High variability")

    if low is None or high is None:
        return "The current sample does not support a full comparison."

    low_val = float(low["mean_display_outcome"])
    high_val = float(high["mean_display_outcome"])

    label = display_outcome_label.lower().replace(" (log10)", "")

    if abs(low_val - high_val) < 0.05:
        return (
            "💡 Low- and high-variability albums perform similarly on average. "
            "This means this cohesion metric does not strongly separate higher- and lower-performing albums."
        )

    if low_val > high_val:
        return (
            f"💡 Albums with lower variability have higher average {label}. "
            "This means more cohesive soundtracks tend to perform better in the current view."
        )

    return (
        f"💡 Albums with higher variability have higher average {label}. "
        "This means more varied soundtracks may perform better for this metric in the current view."
    )

def main() -> None:
    """Render the Track Cohesion Explorer."""
    st.set_page_config(
        page_title="Track Cohesion Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Cohesion Explorer")
    st.write(
        """
        Examine whether soundtrack success is associated with sonic cohesion or
        diversity. This page rolls track-level audio variation up to the album
        level so you can compare cohesion metrics against album popularity and
        top-track dominance.
        """
    )

    cohesion_df = load_track_audio_cohesion_data()
    cohesion_df = add_cohesion_display_fields(cohesion_df)

    filter_inputs = get_global_filter_inputs(cohesion_df)
    composer_options = get_clean_composer_options(cohesion_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    controls = get_track_cohesion_controls(
        composer_options=composer_options,
    )

    plot_df = filter_cohesion_df(
        cohesion_df=cohesion_df,
        global_controls=global_controls,
        controls=controls,
    )

    if plot_df.empty:
        st.warning("No albums remain after applying the current filters.")
        return

    metric_summary_df = summarize_metric_relationships(
        plot_df=plot_df,
        outcome_col="album_outcome_display",
    )

    binned_df = build_variability_bins(
        plot_df=plot_df,
        cohesion_col=controls["cohesion_metric_col"],
    )
    binned_summary_df = build_binned_summary_df(
        plot_df=binned_df,
        raw_outcome_col=controls["album_outcome_col"],
        display_outcome_col="album_outcome_display",
    )

    display_outcome_label = plot_df["album_outcome_display_label"].iloc[0]

    st.caption(build_cohesion_scope_caption(plot_df, controls))

    st.markdown("**What does this metric mean?**")
    st.caption(get_variability_explainer(controls))

    render_track_cohesion_insight_cards(
        plot_df=plot_df,
        controls=controls,
        metric_summary_df=metric_summary_df,
    )

    st.caption(
        build_page_level_narrative(
            plot_df=plot_df,
            controls=controls,
            metric_summary_df=metric_summary_df,
        )
    )

    st.subheader("Cohesion vs Album Success")
    st.altair_chart(
        plot_cohesion_vs_album_outcome(plot_df, controls),
        width="stretch",
    )
    st.caption(build_popularity_supporting_insight(plot_df, controls))

    st.subheader("Cohesion vs Top-Track Dominance")
    st.altair_chart(
        plot_cohesion_vs_top_track_dominance(plot_df, controls),
        width="stretch",
    )
    st.caption(build_dominance_supporting_insight(plot_df, controls))

    if controls["show_strength_chart"] and not metric_summary_df.empty:
        st.subheader("Which Cohesion Dimension Matters Most?")
        st.altair_chart(
            plot_metric_strength_chart(
                metric_summary_df=metric_summary_df,
                outcome_label=controls["album_outcome_label"],
            ),
            width="stretch",
        )
        st.caption(
            build_metric_strength_supporting_insight(
                metric_summary_df=metric_summary_df,
                outcome_label=controls["album_outcome_label"],
            )
        )

    if controls["show_binned_view"] and not binned_summary_df.empty:
        st.subheader("Low / Medium / High Variability View")
        st.altair_chart(
            plot_binned_outcome_chart(
                binned_summary_df=binned_summary_df,
                display_outcome_label=display_outcome_label,
            ),
            width="stretch",
        )
        st.caption(
            build_binned_supporting_insight(
                binned_summary_df=binned_summary_df,
                display_outcome_label=display_outcome_label,
            )
        )

    if controls["show_table"]:
        preferred_cols = [
            "film_title",
            "album_title",
            "composer_primary_clean",
            "film_year",
            "film_year_bucket",
            "album_genre_group",
            "film_genre_group",
            "n_tracks",
            controls["cohesion_metric_col"],
            controls["album_outcome_col"],
            "top_to_total_listeners",
            "pct_instrumental_tracks",
            "energy_mean",
            "danceability_mean",
            "instrumentalness_mean",
            "tempo_mean",
            "tempo_range",
            "key_entropy",
            "mode_entropy",
        ]

        table_cols = select_unique_existing_columns(plot_df, preferred_cols)
        table_df = rename_and_dedupe_for_display(plot_df[table_cols])

        st.subheader("Source Data")
        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()