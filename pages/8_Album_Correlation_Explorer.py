import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

import analysis as an
from app.app_controls import get_correlation_controls
from app.data_filters import filter_dataset
from app.explorer_shared import (
    get_clean_composer_options,
    get_global_filter_inputs,
    rename_and_dedupe_for_display,
)
from app.app_data import load_explorer_data
from app.ui import (
    apply_app_styles,
    get_display_label,
    rename_columns_for_display,
)

EXCLUDED_CORRELATION_FEATURES = {
    "tmdb_id",
    "release_group_mbid",
}


def filter_correlation_features(
    corr_df_plot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove page-level excluded features from the lollipop dataframe.
    """
    if corr_df_plot.empty:
        return corr_df_plot

    feature_col = _get_lollipop_feature_col(corr_df_plot)
    return corr_df_plot[
        ~corr_df_plot[feature_col].astype(str).isin(EXCLUDED_CORRELATION_FEATURES)
    ].copy()


def filter_correlation_matrix(
    corr_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove page-level excluded features from the correlation matrix.
    """
    if corr_matrix.empty:
        return corr_matrix

    keep = [
        col for col in corr_matrix.columns
        if str(col) not in EXCLUDED_CORRELATION_FEATURES
    ]

    if len(keep) < 2:
        return corr_matrix

    return corr_matrix.loc[keep, keep].copy()


def apply_display_labels_to_lollipop_df(
    corr_df_plot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace raw feature names with ui.py display labels in the lollipop dataframe.
    """
    if corr_df_plot.empty:
        return corr_df_plot

    plot_df = corr_df_plot.copy()
    feature_col = _get_lollipop_feature_col(plot_df)
    plot_df[feature_col] = plot_df[feature_col].astype(str).map(get_display_label)
    return plot_df


def apply_display_labels_to_corr_matrix(
    corr_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace raw feature names with ui.py display labels in the correlation matrix.
    """
    if corr_matrix.empty:
        return corr_matrix

    display_matrix = corr_matrix.copy()
    display_matrix.index = [get_display_label(str(col)) for col in display_matrix.index]
    display_matrix.columns = [get_display_label(str(col)) for col in display_matrix.columns]
    return display_matrix

def get_correlation_view_explainer(method: str) -> str:
    """
    Explain how to interpret the selected correlation method.
    """
    if method == "pearson":
        return (
            "Pearson correlation emphasizes linear relationships and is more sensitive "
            "to scale and extreme values."
        )

    return (
        "Spearman correlation emphasizes monotonic rank-order relationships and is "
        "generally less sensitive to outliers and non-normal scaling."
    )

def build_correlation_context_caption(
    method: str,
    ranking_mode: str,
    top_n: int,
    rows_loaded: int,
    heatmap_scope: str,
    heatmap_top_n: int,
) -> str:
    """
    Build a natural-language scope caption for the correlation view.
    """
    if ranking_mode == "Positive only":
        ranking_text = "top positive target relationships"
    elif ranking_mode == "Negative only":
        ranking_text = "top negative target relationships"
    else:
        ranking_text = "top absolute target relationships"

    if heatmap_scope == "Top ranked features only":
        heatmap_text = (
            f"Heatmap is limited to the top {heatmap_top_n:,} visible ranked features."
        )
    else:
        heatmap_text = f"Heatmap scope is set to {heatmap_scope.lower()}."

    return (
        f"Using {method.title()} correlation across {rows_loaded:,} album rows, "
        f"with the lollipop chart showing the top {top_n:,} {ranking_text}. "
        f"{heatmap_text}"
    )


def _get_lollipop_feature_col(corr_df_plot: pd.DataFrame) -> str:
    """
    Infer the feature-name column in the lollipop dataframe.
    """
    for col in ["feature", "predictor", "variable", "name"]:
        if col in corr_df_plot.columns:
            return col

    object_cols = corr_df_plot.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        return object_cols[0]

    return corr_df_plot.columns[0]


def _get_lollipop_corr_col(corr_df_plot: pd.DataFrame) -> str:
    """
    Infer the signed correlation column in the lollipop dataframe.
    """
    for col in ["corr", "correlation", "pearson_r", "spearman_r", "r"]:
        if col in corr_df_plot.columns:
            return col

    numeric_cols = corr_df_plot.select_dtypes(include=["number"]).columns.tolist()
    preferred = [
        col for col in numeric_cols
        if col not in {"rank", "abs_corr", "abs_correlation", "r_squared"}
    ]
    if preferred:
        return preferred[0]

    return numeric_cols[0]

def apply_lollipop_ranking_mode(
    corr_df_plot: pd.DataFrame,
    ranking_mode: str,
) -> pd.DataFrame:
    """
    Apply the selected lollipop ranking mode to the prepared correlation dataframe.

    Args:
        corr_df_plot: Prepared lollipop dataframe.
        ranking_mode: One of Absolute, Positive only, Negative only.

    Returns:
        pd.DataFrame: Filtered and sorted lollipop dataframe.
    """
    if corr_df_plot.empty:
        return corr_df_plot

    feature_col = _get_lollipop_feature_col(corr_df_plot)
    corr_col = _get_lollipop_corr_col(corr_df_plot)

    working_df = corr_df_plot.copy()
    working_df["abs_corr"] = working_df[corr_col].abs()

    if ranking_mode == "Positive only":
        working_df = working_df[working_df[corr_col] > 0].copy()
        working_df = working_df.sort_values(
            [corr_col, feature_col],
            ascending=[False, True],
        )
    elif ranking_mode == "Negative only":
        working_df = working_df[working_df[corr_col] < 0].copy()
        working_df = working_df.sort_values(
            [corr_col, feature_col],
            ascending=[True, True],
        )
    else:
        working_df = working_df.sort_values(
            ["abs_corr", corr_col],
            ascending=[False, False],
        )

    return working_df.reset_index(drop=True)

def build_correlation_insight_summary(
    corr_df_plot: pd.DataFrame,
) -> list[tuple[str, str, str]] | None:
    """
    Build top-line insight cards from the visible lollipop dataframe.
    """
    if corr_df_plot.empty:
        return None

    feature_col = _get_lollipop_feature_col(corr_df_plot)
    corr_col = _get_lollipop_corr_col(corr_df_plot)

    working_df = corr_df_plot[[feature_col, corr_col]].copy()
    working_df["abs_corr"] = working_df[corr_col].abs()

    top_visible = working_df.iloc[0]
    max_abs = float(working_df["abs_corr"].max())

    positive_count = int((working_df[corr_col] > 0).sum())
    negative_count = int((working_df[corr_col] < 0).sum())

    if max_abs >= 0.60:
        strength_label = "Strong"
    elif max_abs >= 0.35:
        strength_label = "Moderate"
    elif max_abs >= 0.15:
        strength_label = "Weak–Moderate"
    else:
        strength_label = "Weak"

    return [
        (
            "Top Visible Feature",
            str(top_visible[feature_col]),
            f"r = {top_visible[corr_col]:.3f}",
        ),
        (
            "Visible Direction Mix",
            f"{positive_count} + / {negative_count} −",
            "Counts within the currently visible lollipop ranking.",
        ),
        (
            "Strongest Visible Signal",
            strength_label,
            f"Max |r| = {max_abs:.3f}",
        ),
    ]


def render_correlation_insight_cards(
    corr_df_plot: pd.DataFrame,
) -> None:
    """
    Render top-line correlation insight cards.
    """
    insights = build_correlation_insight_summary(corr_df_plot)
    if not insights:
        st.info("No lollipop features remain under the current settings.")
        return

    st.markdown("### 🧠 Key Insights")
    cols = st.columns(3)

    for i, (title, value, caption) in enumerate(insights):
        with cols[i]:
            st.metric(title, value)
            st.caption(caption)

def build_lollipop_supporting_insight(
    corr_df_plot: pd.DataFrame,
    method: str,
    ranking_mode: str,
) -> str:
    """
    Build a data-reactive supporting insight for the lollipop chart.
    """
    if corr_df_plot.empty:
        return ""

    feature_col = _get_lollipop_feature_col(corr_df_plot)
    corr_col = _get_lollipop_corr_col(corr_df_plot)

    working_df = corr_df_plot[[feature_col, corr_col]].copy()
    working_df["abs_corr"] = working_df[corr_col].abs()
    ranked = working_df.reset_index(drop=True)

    top = ranked.iloc[0]
    max_abs = float(top["abs_corr"])

    if len(ranked) >= 2:
        second = ranked.iloc[1]
        gap = float(top["abs_corr"] - second["abs_corr"])
        gap_text = f"The gap vs the next visible feature is {gap:.3f}."
    else:
        gap_text = "Only one feature is visible in the current ranking."

    if max_abs >= 0.60:
        strength = "strong"
    elif max_abs >= 0.35:
        strength = "moderate"
    elif max_abs >= 0.15:
        strength = "weak-to-moderate"
    else:
        strength = "weak"

    direction = (
        "positive" if top[corr_col] > 0
        else "negative" if top[corr_col] < 0
        else "near-zero"
    )

    if ranking_mode == "Positive only":
        mode_text = "Within the visible positive-only ranking"
    elif ranking_mode == "Negative only":
        mode_text = "Within the visible negative-only ranking"
    else:
        mode_text = "Within the visible absolute ranking"

    return (
        f"💡 {mode_text}, the strongest {method.title()} relationship is "
        f"{top[feature_col]} with a {strength} {direction} association "
        f"(r = {top[corr_col]:.3f}). {gap_text}"
    )


def build_heatmap_supporting_insight(
    corr_matrix: pd.DataFrame,
    method: str,
) -> str:
    """
    Build a short supporting insight for the correlation heatmap.

    Args:
        corr_matrix: Correlation matrix used in the heatmap.
        method: Selected correlation method.

    Returns:
        str: Supporting insight sentence.
    """
    if corr_matrix is None or corr_matrix.empty:
        return "💡 No heatmap insight is available."

    numeric_matrix = corr_matrix.select_dtypes(include="number").copy()

    if numeric_matrix.empty or numeric_matrix.shape[0] < 2:
        return "💡 Not enough features remain to summarize pairwise relationships."

    abs_matrix = numeric_matrix.abs().copy()
    diag_mask = pd.DataFrame(
        np.eye(len(abs_matrix), dtype=bool),
        index=abs_matrix.index,
        columns=abs_matrix.columns,
    )
    abs_matrix = abs_matrix.mask(diag_mask)

    stacked_abs = abs_matrix.stack()
    stacked_abs = stacked_abs.dropna()

    if stacked_abs.empty:
        return "💡 Not enough off-diagonal correlations remain to summarize."

    top_pair = stacked_abs.sort_values(ascending=False).index[0]
    top_value = float(stacked_abs.loc[top_pair])

    strength_label = (
        "very strong"
        if top_value >= 0.70 else
        "strong"
        if top_value >= 0.50 else
        "moderate"
        if top_value >= 0.30 else
        "weak"
    )

    return (
        f"💡 The strongest off-diagonal {method.lower()} relationship in the "
        f"visible heatmap is between **{top_pair[0]}** and **{top_pair[1]}** "
        f"(absolute correlation {top_value:.2f}), which is {strength_label}."
    )


def build_heatmap_feature_subset(
    corr_df_plot: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    heatmap_scope: str,
    heatmap_top_n: int,
) -> pd.DataFrame:
    """Select the feature subset shown in the album heatmap."""
    if corr_matrix is None or corr_matrix.empty:
        return corr_matrix

    excluded_heatmap_features = {
        "lfm_album_listeners",
        "lfm_album_playcount",
        "log_lfm_album_listeners",
        "log_lfm_album_playcount",
    }

    working_corr_matrix = corr_matrix.copy()
    keep_cols = [
        col for col in working_corr_matrix.columns
        if col not in excluded_heatmap_features
    ]

    if len(keep_cols) >= 2:
        working_corr_matrix = working_corr_matrix.loc[keep_cols, keep_cols].copy()
    else:
        return corr_matrix

    ranked_features = (
        corr_df_plot["feature"].head(heatmap_top_n).tolist()
        if "feature" in corr_df_plot.columns
        else []
    )

    success_anchor_features = [
        col for col in [
            "lfm_album_listeners",
            "lfm_album_playcount",
            "log_lfm_album_listeners",
            "log_lfm_album_playcount",
            "album_cohesion_score",
            "n_tracks",
            "composer_album_count",
        ]
        if col in working_corr_matrix.columns
    ]

    album_release_structure_features = [
        col for col in [
            "days_since_album_release",
            "days_since_film_release",
            "album_release_lag_days",
            "n_tracks",
            "composer_album_count",
            "album_cohesion_score",
            "album_cohesion_has_audio_data",
        ]
        if col in corr_matrix.columns
    ]

    album_genre_features = [
        col for col in [
            "ambient_experimental",
            "classical_orchestral",
            "electronic",
            "hip_hop_rnb",
            "pop",
            "rock",
            "world_folk",
        ]
        if col in corr_matrix.columns
    ]

    film_genre_features = [
        col for col in working_corr_matrix.columns
        if col.startswith("film_is_")
    ]

    awards_features = [
        col for col in [
            "oscar_score_nominee",
            "oscar_score_winner",
            "oscar_song_nominee",
            "oscar_song_winner",
            "globes_score_nominee",
            "globes_score_winner",
            "globes_song_nominee",
            "globes_song_winner",
            "critics_score_nominee",
            "critics_score_winner",
            "critics_song_nominee",
            "critics_song_winner",
            "bafta_score_nominee",
            "bafta_score_winner",
            "us_score_nominee_count",
            "us_song_nominee_count",
            "bafta_nominee",
        ]
        if col in corr_matrix.columns
    ]

    context_continuous_features = [
        col for col in [
            "film_vote_count",
            "film_popularity",
            "film_budget",
            "film_revenue",
            "film_rating",
            "days_since_film_release",
            "film_runtime_min",
            "days_since_album_release",
            "n_tracks",
            "composer_album_count",
            "album_cohesion_score",
        ]
        if col in corr_matrix.columns
    ]

    context_binary_features = [
        col for col in corr_matrix.columns
        if (
            col.startswith("film_is_")
            or col in {
                "album_cohesion_has_audio_data",
                "oscar_score_nominee",
                "oscar_score_winner",
                "oscar_song_nominee",
                "oscar_song_winner",
                "globes_score_nominee",
                "globes_score_winner",
                "globes_song_nominee",
                "globes_song_winner",
                "critics_score_nominee",
                "critics_score_winner",
                "critics_song_nominee",
                "critics_song_winner",
                "bafta_score_nominee",
                "bafta_score_winner",
            }
        )
    ]

    scope_map = {
        "Top ranked features only": ranked_features,
        "Success anchors only": success_anchor_features,
        "Album release / structure": album_release_structure_features,
        "Album genres": album_genre_features,
        "Film genres": film_genre_features,
        "Awards": awards_features,
        "Context continuous": context_continuous_features,
        "Context binary": context_binary_features,
        "All visible features": list(working_corr_matrix.columns),
    }

    selected_features = scope_map.get(heatmap_scope, ranked_features)
    selected_features = [
        feature for feature in selected_features
        if feature in working_corr_matrix.index and feature in working_corr_matrix.columns
    ]

    if len(selected_features) < 2:
        fallback_features = [
            feature for feature in ranked_features
            if feature in working_corr_matrix.index and feature in working_corr_matrix.columns
        ]
        selected_features = fallback_features[: max(2, min(len(fallback_features), heatmap_top_n))]

    if len(selected_features) < 2:
        return working_corr_matrix

    return working_corr_matrix.loc[selected_features, selected_features]

def render_lollipop_section(
    corr_df_plot: pd.DataFrame,
    method: str,
    ranking_mode: str,
    show_table: bool,
) -> None:
    """
    Render the lollipop chart section using the precomputed page dataframe.

    Args:
        corr_df_plot: Pre-filtered lollipop dataframe already limited to the
            visible Top-N features for the current page state.
        method: Correlation method.
        ranking_mode: Current lollipop ranking mode.
        show_table: Whether to show the underlying dataframe.
    """
    st.subheader("Feature Correlations with Album Popularity")

    if corr_df_plot.empty:
        st.info("No lollipop features remain under the current settings.")
        return

    if ranking_mode == "Positive only":
        st.caption("Showing only the strongest positive relationships with the target.")
    elif ranking_mode == "Negative only":
        st.caption("Showing only the strongest negative relationships with the target.")
    else:
        st.caption("Showing the strongest relationships by absolute correlation magnitude.")

    display_df = apply_display_labels_to_lollipop_df(corr_df_plot)

    chart = an.plot_lollipop_chart(display_df)
    st.altair_chart(chart, width="stretch")
    st.caption(build_lollipop_supporting_insight(display_df, method, ranking_mode))

    if show_table:
        st.dataframe(
            rename_columns_for_display(display_df),
            width="stretch",
        )


def render_heatmap_section(
    corr_matrix: pd.DataFrame,
    method: str,
    show_table: bool,
) -> None:
    """
    Render the correlation heatmap section.

    Args:
        corr_matrix: Precomputed correlation matrix.
        method: Correlation method.
        show_table: Whether to show the underlying matrix and long table.
    """
    st.subheader(f"Correlation Heatmap ({method.title()})")
    st.caption(
        "Use the heatmap scope control to focus on compact album-level feature sets "
        "instead of viewing the full matrix all at once."
    )

    if corr_matrix is None or corr_matrix.empty:
        st.info("No correlation matrix available for the current selection.")
        return

    display_corr_matrix = apply_display_labels_to_corr_matrix(corr_matrix)

    feature_count = display_corr_matrix.shape[0]

    if feature_count <= 12:
        chart_height = 500
        label_font_size = 12
    elif feature_count <= 20:
        chart_height = 700
        label_font_size = 11
    elif feature_count <= 30:
        chart_height = 900
        label_font_size = 10
    else:
        chart_height = min(1400, 28 * feature_count)
        label_font_size = 9

    heatmap_chart = an.plot_correlation_heatmap(
        corr_matrix=display_corr_matrix,
        title=f"Correlation Heatmap ({method.title()})",
    ).properties(
        height=chart_height,
    )

    heatmap_chart = heatmap_chart.configure_axis(
        labelFontSize=label_font_size,
        titleFontSize=12,
    )

    heatmap_chart = heatmap_chart.configure_view(
        stroke=None
    )

    st.altair_chart(heatmap_chart, width="stretch")

    heatmap_insight = build_heatmap_supporting_insight(display_corr_matrix, method)
    if heatmap_insight:
        st.caption(heatmap_insight)

    if show_table:
        st.dataframe(
            rename_and_dedupe_for_display(display_corr_matrix.reset_index()),
            width="stretch",
        )

def build_album_correlation_scope_caption(
    filtered_df: pd.DataFrame,
    controls: dict,
) -> str:
    """Describe the current filtered album scope in plain English."""
    parts: list[str] = [f"{len(filtered_df):,} visible albums"]

    year_min, year_max = controls["year_range"]
    parts.append(f"film years {year_min}–{year_max}")

    if controls.get("selected_film_genres"):
        parts.append(
            f"{len(controls['selected_film_genres'])} film genre filter(s)"
        )

    if controls.get("selected_album_genres"):
        parts.append(
            f"{len(controls['selected_album_genres'])} album genre filter(s)"
        )

    if controls.get("selected_composers"):
        parts.append(
            f"{len(controls['selected_composers'])} composer filter(s)"
        )

    if controls.get("selected_labels"):
        parts.append(
            f"{len(controls['selected_labels'])} label filter(s)"
        )

    if controls.get("search_text", "").strip():
        parts.append("text search active")

    if controls.get("min_tracks", 1) > 1:
        parts.append(f"minimum {controls['min_tracks']} tracks")

    if controls.get("listeners_only", False):
        parts.append("listener availability required")

    return "Current album scope: " + " · ".join(parts) + "."


def render_album_correlation_scope_metrics(
    filtered_df: pd.DataFrame,
    corr_df_plot: pd.DataFrame,
    corr_matrix: pd.DataFrame,
) -> None:
    """Render top-level metrics for the current album slice."""
    visible_predictors = len(corr_df_plot)
    heatmap_features = corr_matrix.shape[0] if corr_matrix is not None else 0

    top_abs_corr = None
    if corr_df_plot is not None and not corr_df_plot.empty:
        corr_col = _get_lollipop_corr_col(corr_df_plot)
        top_abs_corr = float(corr_df_plot[corr_col].abs().max())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Visible Albums", f"{len(filtered_df):,}")

    with col2:
        st.metric("Visible Ranked Predictors", f"{visible_predictors:,}")

    with col3:
        st.metric(
            "Strongest Visible |r|",
            "n/a" if top_abs_corr is None else f"{top_abs_corr:.3f}",
        )

    st.caption(
        f"Heatmap currently includes {heatmap_features:,} feature(s) "
        "after applying the selected scope and ranking settings."
    )


def build_top_redundancy_pairs_table(
    corr_matrix: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return strongest off-diagonal absolute correlation pairs."""
    if corr_matrix is None or corr_matrix.empty or corr_matrix.shape[0] < 2:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    cols = list(corr_matrix.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            feature_a = cols[i]
            feature_b = cols[j]
            corr_value = float(corr_matrix.iloc[i, j])
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": corr_value,
                    "abs_correlation": abs(corr_value),
                }
            )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["abs_correlation", "correlation"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )


def render_redundancy_section(corr_matrix: pd.DataFrame) -> None:
    """Render a small redundancy handoff section for album modeling."""
    st.subheader("Redundancy Check for Album Modeling")

    redundancy_df = build_top_redundancy_pairs_table(corr_matrix, top_n=5)

    if redundancy_df.empty:
        st.info("Not enough features remain to summarize redundant predictors.")
        return

    top_pair = redundancy_df.iloc[0]

    st.caption(
        "Use this section to spot overlapping album-level predictors before "
        "interpreting Album Ridge or Album Regression results."
    )
    st.caption(
        f"The strongest visible overlap is **{get_display_label(str(top_pair['feature_a']))}** "
        f"vs **{get_display_label(str(top_pair['feature_b']))}** "
        f"(absolute correlation = {top_pair['abs_correlation']:.3f})."
    )

    display_df = redundancy_df.copy()
    display_df["feature_a"] = display_df["feature_a"].astype(str).map(get_display_label)
    display_df["feature_b"] = display_df["feature_b"].astype(str).map(get_display_label)

    st.dataframe(
        rename_and_dedupe_for_display(display_df),
        width="stretch",
    )

def main() -> None:
    """
    Run the album correlation explorer page.
    """
    st.set_page_config(
        page_title="Album Correlation Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Album Correlation Explorer")
    st.caption(
        "Identify album-level signals tied to soundtrack popularity and inspect "
        "redundant predictors before moving into album ridge and regression."
    )

    album_analytics_df = load_explorer_data().copy()

    global_filter_inputs = get_global_filter_inputs(album_analytics_df)
    composer_options = get_clean_composer_options(album_analytics_df)

    label_options = sorted(
        album_analytics_df["label_names"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    controls = get_correlation_controls(
        min_year=global_filter_inputs["min_year"],
        max_year=global_filter_inputs["max_year"],
        film_genre_options=global_filter_inputs["film_genre_options"],
        album_genre_options=global_filter_inputs["album_genre_options"],
        composer_options=composer_options,
        label_options=label_options,
        include_global_filters=True,
        include_composers=True,
        include_labels=True,
        include_search=True,
        include_min_tracks=True,
        include_listeners_only=True,
        default_listeners_only=True,
        min_tracks_min=1,
        min_tracks_max=40,
        default_min_tracks=1,
    )

    filtered_album_df = filter_dataset(album_analytics_df, controls)

    if filtered_album_df.empty:
        st.warning("No albums remain after the selected global filters.")
        return

    raw_lollipop_df = an.prepare_lollipop_data(
        album_analytics_df=filtered_album_df,
        target_col=controls["target_col"],
        method=controls["method"],
    ).copy()

    raw_lollipop_df = filter_correlation_features(raw_lollipop_df)
    ranked_lollipop_df = apply_lollipop_ranking_mode(
        raw_lollipop_df,
        controls["ranking_mode"],
    )
    corr_df_plot = ranked_lollipop_df.head(controls["top_n"]).copy()

    full_corr_matrix = an.compute_correlation_matrix(
        album_analytics_df=filtered_album_df,
        method=controls["method"],
    )
    full_corr_matrix = filter_correlation_matrix(full_corr_matrix)

    corr_matrix = build_heatmap_feature_subset(
        corr_df_plot=corr_df_plot,
        corr_matrix=full_corr_matrix,
        heatmap_scope=controls["heatmap_scope"],
        heatmap_top_n=controls["heatmap_top_n"],
    )

    target_label = (
        "log album listeners"
        if controls["target_col"] == "log_lfm_album_listeners"
        else "log album playcount"
    )

    st.caption(
        f"{get_correlation_view_explainer(controls['method'])} "
        f"Current target: {target_label}. "
        f"{build_album_correlation_scope_caption(filtered_album_df, controls)}"
    )

    st.caption(
        build_correlation_context_caption(
            method=controls["method"],
            ranking_mode=controls["ranking_mode"],
            top_n=controls["top_n"],
            rows_loaded=len(filtered_album_df),
            heatmap_scope=controls["heatmap_scope"],
            heatmap_top_n=controls["heatmap_top_n"],
        )
    )

    render_album_correlation_scope_metrics(
        filtered_df=filtered_album_df,
        corr_df_plot=corr_df_plot,
        corr_matrix=corr_matrix,
    )

    render_correlation_insight_cards(
        apply_display_labels_to_lollipop_df(corr_df_plot)
    )

    render_lollipop_section(
        corr_df_plot=corr_df_plot,
        method=controls["method"],
        ranking_mode=controls["ranking_mode"],
        show_table=controls["show_lollipop_table"],
    )

    st.divider()

    render_heatmap_section(
        corr_matrix=corr_matrix,
        method=controls["method"],
        show_table=controls["show_heatmap_table"],
    )

    st.divider()

    render_redundancy_section(corr_matrix)

    st.caption(
        "These are pairwise associations within the currently selected album "
        "slice and should be interpreted as descriptive, not causal."
    )


if __name__ == "__main__":
    main()