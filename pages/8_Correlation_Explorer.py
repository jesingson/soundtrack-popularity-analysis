import streamlit as st
import numpy as np
import pandas as pd

import analysis as an
from app.app_controls import get_correlation_controls
from app.app_data import load_analysis_data
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
    top_n: int,
    rows_loaded: int,
    heatmap_scope: str,
    heatmap_top_n: int,
) -> str:
    """
    Build a natural-language scope caption for the correlation view.
    """
    if heatmap_scope == "Top lollipop features only":
        heatmap_text = f"Heatmap is limited to the top {heatmap_top_n:,} lollipop features."
    else:
        heatmap_text = "Heatmap uses the full feature matrix."

    return (
        f"Using {method.title()} correlation across {rows_loaded:,} album rows, "
        f"with the lollipop chart showing the top {top_n:,} target relationships. "
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


def build_correlation_insight_summary(
    corr_df_plot: pd.DataFrame,
) -> list[tuple[str, str, str]] | None:
    """
    Build top-line insight cards from the lollipop dataframe.
    """
    if corr_df_plot.empty:
        return None

    feature_col = _get_lollipop_feature_col(corr_df_plot)
    corr_col = _get_lollipop_corr_col(corr_df_plot)

    working_df = corr_df_plot[[feature_col, corr_col]].copy()
    working_df["abs_corr"] = working_df[corr_col].abs()

    top_abs_row = working_df.sort_values(
        ["abs_corr", corr_col],
        ascending=[False, False],
    ).iloc[0]

    positive_df = working_df[working_df[corr_col] > 0]
    negative_df = working_df[working_df[corr_col] < 0]

    if not positive_df.empty:
        top_positive = positive_df.sort_values(corr_col, ascending=False).iloc[0]
        pos_value = str(top_positive[feature_col])
        pos_caption = f"r = {top_positive[corr_col]:.3f}"
    else:
        pos_value = "None"
        pos_caption = "No positive features in view"

    if not negative_df.empty:
        top_negative = negative_df.sort_values(corr_col, ascending=True).iloc[0]
        neg_value = str(top_negative[feature_col])
        neg_caption = f"r = {top_negative[corr_col]:.3f}"
    else:
        neg_value = "None"
        neg_caption = "No negative features in view"

    return [
        (
            "Top Positive Feature",
            pos_value,
            pos_caption,
        ),
        (
            "Top Negative Feature",
            neg_value,
            neg_caption,
        ),
        (
            "Largest Absolute Correlation",
            str(top_abs_row[feature_col]),
            f"|r| = {top_abs_row['abs_corr']:.3f}",
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
    ranked = working_df.sort_values(
        ["abs_corr", corr_col],
        ascending=[False, False],
    ).reset_index(drop=True)

    top = ranked.iloc[0]
    max_abs = float(top["abs_corr"])

    if len(ranked) >= 2:
        second = ranked.iloc[1]
        gap = float(top["abs_corr"] - second["abs_corr"])
        gap_text = f"The gap vs the next feature is {gap:.3f}."
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

    return (
        f"💡 The strongest visible {method.title()} relationship is "
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
    """
    Optionally restrict the heatmap to the top lollipop features.
    """
    if heatmap_scope != "Top lollipop features only":
        return corr_matrix

    if corr_df_plot.empty or corr_matrix.empty:
        return corr_matrix

    feature_col = _get_lollipop_feature_col(corr_df_plot)
    top_features = (
        corr_df_plot[feature_col]
        .astype(str)
        .head(heatmap_top_n)
        .tolist()
    )

    available_features = [
        feature for feature in top_features
        if feature in corr_matrix.index and feature in corr_matrix.columns
    ]

    if len(available_features) < 2:
        return corr_matrix

    return corr_matrix.loc[available_features, available_features].copy()

def render_lollipop_section(
    corr_df_plot: pd.DataFrame,
    method: str,
    show_table: bool,
) -> None:
    """
    Render the lollipop chart section using precomputed data.
    """
    st.subheader("Feature Correlations with Album Popularity")

    display_df = apply_display_labels_to_lollipop_df(corr_df_plot)

    chart = an.plot_lollipop_chart(display_df)
    st.altair_chart(chart, width="stretch")

    render_correlation_insight_cards(
        apply_display_labels_to_lollipop_df(corr_df_plot)
    )

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
    st.subheader("Correlation Heatmap")

    display_corr_matrix = apply_display_labels_to_corr_matrix(corr_matrix)

    heatmap = (
        an.plot_correlation_heatmap(
            corr_matrix=display_corr_matrix,
            title=f"Correlation Heatmap ({method.title()})",
        )
        .properties(
            width=950,
            height=950,
        )
        .configure_axisX(
            labelAngle=-45,
            labelFontSize=9,
            labelLimit=300,
            labelOverlap=False,
        )
        .configure_axisY(
            labelFontSize=9,
            labelLimit=220,
            labelOverlap=False,
        )
    )

    st.altair_chart(heatmap, width="stretch")
    st.caption(build_heatmap_supporting_insight(corr_matrix, method))

    if show_table:
        st.write("Correlation matrix")
        st.dataframe(
            rename_columns_for_display(display_corr_matrix),
            width="stretch",
        )

        st.write("Long-form heatmap source")
        st.dataframe(
            rename_columns_for_display(an.corr_to_long(display_corr_matrix)),
            width="stretch",
        )

def main() -> None:
    """
    Run the correlation explorer page.
    """
    st.set_page_config(
        page_title="Correlation Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Correlation Explorer")
    st.write(
        """
        Explore pairwise feature relationships and feature-level correlations
        with soundtrack album popularity.
        """
    )

    controls = get_correlation_controls()
    album_analytics_df = load_analysis_data()

    raw_lollipop_df = an.prepare_lollipop_data(
        album_analytics_df=album_analytics_df,
        target_col="log_lfm_album_listeners",
        method=controls["method"],
    ).copy()

    raw_lollipop_df = filter_correlation_features(raw_lollipop_df)
    corr_df_plot = raw_lollipop_df.head(controls["top_n"]).copy()

    full_corr_matrix = an.compute_correlation_matrix(
        album_analytics_df=album_analytics_df,
        method=controls["method"],
    )
    full_corr_matrix = filter_correlation_matrix(full_corr_matrix)

    corr_matrix = build_heatmap_feature_subset(
        corr_df_plot=corr_df_plot,
        corr_matrix=full_corr_matrix,
        heatmap_scope=controls["heatmap_scope"],
        heatmap_top_n=controls["heatmap_top_n"],
    )

    st.caption(get_correlation_view_explainer(controls["method"]))

    st.markdown("**View Context**")
    st.caption(
        build_correlation_context_caption(
            method=controls["method"],
            top_n=controls["top_n"],
            rows_loaded=len(album_analytics_df),
            heatmap_scope=controls["heatmap_scope"],
            heatmap_top_n=controls["heatmap_top_n"],
        )
    )

    render_correlation_insight_cards(
        apply_display_labels_to_lollipop_df(corr_df_plot)
    )

    render_lollipop_section(
        corr_df_plot=corr_df_plot,
        method=controls["method"],
        show_table=controls["show_lollipop_table"],
    )

    st.divider()

    render_heatmap_section(
        corr_matrix=corr_matrix,
        method=controls["method"],
        show_table=controls["show_heatmap_table"],
    )

    st.caption(
        "These are pairwise associations and should be interpreted as descriptive, not causal."
    )


if __name__ == "__main__":
    main()