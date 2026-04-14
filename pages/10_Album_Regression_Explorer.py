import streamlit as st

import numpy as np
import pandas as pd
import streamlit as st

import regression_analysis as reg
import regression_visualization as reg_viz
from app.app_controls import (
    get_global_filter_controls,
    get_regression_controls,
)
from app.app_data import load_analysis_data, load_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import get_global_filter_inputs
from app.ui import (
    apply_app_styles,
    get_display_label,
    rename_columns_for_display,
)

ALBUM_REGRESSION_TARGET_CANDIDATES = [
    "log_lfm_album_listeners",
    "log_lfm_album_playcount",
]

EXCLUDED_MODEL_COLS = {
    "album_cohesion_has_audio_data",
}

def attach_regression_filter_metadata(
    album_analytics_df,
    album_explorer_df,
):
    """
    Merge explorer-style global-filter metadata and missing outcome fields
    onto the analysis dataframe.

    This keeps the regression pipeline based on the analysis dataframe
    while enriching it with:
    - shared global-filter metadata
    - album playcount target fields if they are missing
    """
    analysis_df = album_analytics_df.copy()
    explorer_df = album_explorer_df.copy()

    merge_keys = [
        col for col in ["release_group_mbid", "tmdb_id"]
        if col in analysis_df.columns and col in explorer_df.columns
    ]

    if not merge_keys:
        return analysis_df

    candidate_metadata_cols = [
        "release_group_mbid",
        "tmdb_id",
        "film_year",
        "film_genres",
        "album_genres_display",
        "lfm_album_listeners",
        "lfm_album_playcount",
        "log_lfm_album_listeners",
        "log_lfm_album_playcount",
    ]

    metadata_cols = [
        col for col in candidate_metadata_cols
        if col in explorer_df.columns
    ]

    cols_to_add = [
        col for col in metadata_cols
        if col in merge_keys or col not in analysis_df.columns
    ]

    if len(cols_to_add) <= len(merge_keys):
        merged_df = analysis_df.copy()
    else:
        metadata_df = (
            explorer_df[cols_to_add]
            .drop_duplicates(subset=merge_keys)
            .copy()
        )

        merged_df = analysis_df.merge(
            metadata_df,
            on=merge_keys,
            how="left",
            validate="1:1",
        )

    if (
        "log_lfm_album_playcount" not in merged_df.columns
        and "lfm_album_playcount" in merged_df.columns
    ):
        merged_df["log_lfm_album_playcount"] = np.log1p(
            pd.to_numeric(merged_df["lfm_album_playcount"], errors="coerce")
        )

    if (
        "log_lfm_album_listeners" not in merged_df.columns
        and "lfm_album_listeners" in merged_df.columns
    ):
        merged_df["log_lfm_album_listeners"] = np.log1p(
            pd.to_numeric(merged_df["lfm_album_listeners"], errors="coerce")
        )

    return merged_df

def render_regression_metrics(regression_results: dict) -> None:
    """
    Render headline regression metrics.

    Args:
        regression_results: Full regression pipeline results dictionary.
    """
    feature_config = regression_results["feature_config"]
    filter_results = regression_results["filter_results"]
    ols_results = regression_results["ols_results"]

    continuous_kept = len(filter_results["kept_continuous"])
    continuous_total = len(feature_config["continuous_features"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Modeling rows", f"{ols_results['n_rows']:,}")

    with col2:
        st.metric("Predictor count", f"{ols_results['n_predictors']:,}")

    with col3:
        st.metric(
            "Continuous predictors kept",
            f"{continuous_kept} / {continuous_total}",
        )


def render_coefficient_chart(results):
    """
    Build and render the coefficient plot.

    Args:
        results: Fitted statsmodels OLS results object.

    Returns:
        pd.DataFrame: Tidy coefficient dataframe used by the chart.
    """
    coef_df = reg.build_coefficient_plot_df(results)
    chart = reg_viz.create_coefficient_whisker_chart(coef_df)
    st.altair_chart(chart, width="stretch")
    return coef_df


def render_filter_summary(filter_results: dict) -> None:
    """
    Render continuous-feature filtering results.

    Args:
        filter_results: Filter-stage regression results dictionary.
    """
    st.subheader("Feature Filtering Summary")

    st.write(
        """
        Continuous predictors are screened using their absolute Pearson
        correlation with the target. Binary predictors are retained
        without this filtering step.
        """
    )

    st.caption(
        "This is a light screening step: continuous features with very weak "
        "linear signal are dropped before the later transform and OLS stages."
    )

    st.dataframe(
        rename_columns_for_display(filter_results["summary_df"].round(3)),
        width="stretch",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Kept continuous predictors**")
        st.write(filter_results["kept_continuous"])
    with col2:
        st.write("**Dropped continuous predictors**")
        st.write(filter_results["dropped_continuous"])


def render_transform_summary(transform_results: dict) -> None:
    """
    Render regression transform summary.

    Args:
        transform_results: Transform-stage regression results dictionary.
    """
    st.subheader("Transform Summary")
    st.caption(
        "Continuous predictors are standardized for comparability, while selected "
        "heavy-tailed exposure variables are log-transformed before modeling."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Continuous predictors standardized",
            len(transform_results["x_cont"]),
        )

    with col2:
        st.metric(
            "Binary predictors unchanged",
            len(transform_results["x_bin"]),
        )

    with col3:
        st.metric(
            "Logged predictors",
            len(transform_results["logged_predictors"]),
        )

    st.write("**Logged predictors**")
    st.write(transform_results["logged_predictors"])


def render_finalize_summary(finalize_results: dict) -> None:
    """
    Render final predictor cleanup summary.

    Args:
        finalize_results: Finalize-stage regression results dictionary.
    """
    st.subheader("Final Predictor Cleanup")
    st.caption(
        "This final cleanup removes a few predictors to reduce redundancy and "
        "ensures genre indicators are fully numeric before OLS fitting."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dropped columns**")
        st.write(finalize_results["dropped_columns"])

    with col2:
        st.write("**Film genre columns cast to int**")
        st.write(finalize_results["film_genre_cols"])


def render_model_summary(results) -> None:
    """
    Render the full OLS summary text.

    Args:
        results: Fitted statsmodels OLS results object.
    """
    st.subheader("Full OLS Summary")
    st.text(results.summary().as_text())

def build_regression_scope_caption(
    regression_results: dict,
    target_col: str,
) -> str:
    """
    Build a short caption describing the current regression scope.

    Args:
        regression_results: Full regression pipeline results dictionary.

    Returns:
        str: Human-readable scope caption.
    """
    feature_config = regression_results["feature_config"]
    filter_results = regression_results["filter_results"]
    ols_results = regression_results["ols_results"]

    continuous_kept = len(filter_results["kept_continuous"])
    continuous_total = len(feature_config["continuous_features"])

    return (
        f"Current scope: target = {get_display_label(target_col)}, "
        f"{ols_results['n_rows']:,} modeled rows, "
        f"{ols_results['n_predictors']:,} final predictors, and "
        f"{continuous_kept} of {continuous_total} continuous candidates "
        "retained after the initial screening step."
    )


def build_coefficient_insight_summary(coef_df) -> dict[str, str]:
    """
    Build three narrative insight cards from the coefficient dataframe.

    Args:
        coef_df: Tidy coefficient dataframe from build_coefficient_plot_df().

    Returns:
        dict[str, str]: Titles, values, and captions for three insight cards.
    """
    if coef_df.empty:
        return {
            "card1_title": "Largest Absolute Effect",
            "card1_value": "None",
            "card1_caption": "No coefficients are available.",
            "card2_title": "Direction Balance",
            "card2_value": "None",
            "card2_caption": "No positive or negative effects are available.",
            "card3_title": "CI Clarity",
            "card3_value": "None",
            "card3_caption": "No confidence-interval information is available.",
        }

    ranked_df = coef_df.sort_values(
        ["abs_coef", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    top_row = ranked_df.iloc[0]
    top_feature = str(top_row["feature"])
    top_coef = float(top_row["coef"])

    positive_count = int((coef_df["coef"] > 0).sum())
    negative_count = int((coef_df["coef"] < 0).sum())

    not_crossing_zero = int((~coef_df["crosses_zero"]).sum())
    total_coefficients = int(len(coef_df))
    not_crossing_share = (
        not_crossing_zero / total_coefficients
        if total_coefficients > 0 else 0.0
    )

    if top_coef > 0:
        top_caption = (
            f"'{top_feature}' has the largest absolute coefficient and is "
            f"positive ({top_coef:.3f})."
        )
    elif top_coef < 0:
        top_caption = (
            f"'{top_feature}' has the largest absolute coefficient and is "
            f"negative ({top_coef:.3f})."
        )
    else:
        top_caption = (
            f"'{top_feature}' has the largest absolute coefficient, but its "
            f"estimated effect is effectively zero ({top_coef:.3f})."
        )

    return {
        "card1_title": "Largest Absolute Effect",
        "card1_value": top_feature,
        "card1_caption": top_caption,
        "card2_title": "Direction Balance",
        "card2_value": f"{positive_count} + / {negative_count} −",
        "card2_caption": (
            "This counts how many modeled coefficients are positive versus "
            "negative in the fitted multivariate regression."
        ),
        "card3_title": "CI Clarity",
        "card3_value": f"{not_crossing_zero}/{total_coefficients}",
        "card3_caption": (
            f"{not_crossing_share:.1%} of coefficients have confidence "
            "intervals that do not cross 0."
        ),
    }


def render_coefficient_insight_cards(coef_df) -> None:
    """
    Render the top-row coefficient insight cards.

    Args:
        coef_df: Tidy coefficient dataframe.
    """
    insights = build_coefficient_insight_summary(coef_df)

    st.markdown("### 🧠 Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            insights["card1_title"],
            insights["card1_value"],
        )
        st.caption(insights["card1_caption"])

    with col2:
        st.metric(
            insights["card2_title"],
            insights["card2_value"],
        )
        st.caption(insights["card2_caption"])

    with col3:
        st.metric(
            insights["card3_title"],
            insights["card3_value"],
        )
        st.caption(insights["card3_caption"])


def build_coefficient_supporting_insight(coef_df, target_col: str) -> str:
    """
    Build a short supporting sentence for the coefficient plot.

    Args:
        coef_df: Tidy coefficient dataframe.

    Returns:
        str: Supporting chart insight.
    """
    if coef_df.empty:
        return "No coefficient insight is available."

    ranked_df = coef_df.sort_values(
        ["abs_coef", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    top_feature = str(ranked_df.iloc[0]["feature"])
    top_abs = float(ranked_df.iloc[0]["abs_coef"])

    median_abs = float(coef_df["abs_coef"].median())
    not_crossing_zero = int((~coef_df["crosses_zero"]).sum())
    total_coefficients = int(len(coef_df))

    return (
        f"💡 The coefficient plot for {get_display_label(target_col).lower()} is led by "
        f"'{top_feature}' (absolute coefficient {top_abs:.3f}). "
        f"The median absolute effect across all predictors is {median_abs:.3f}, and "
        f"{not_crossing_zero} of {total_coefficients} coefficients have "
        "intervals that do not cross 0."
    )

def build_album_modeling_takeaway(
    coef_df,
    target_col: str,
) -> str:
    """
    Build a short modeling takeaway from the visible coefficient table.
    """
    if coef_df.empty:
        return (
            f"No fitted model takeaway is available for "
            f"{get_display_label(target_col).lower()}."
        )

    strong_effects = int((coef_df["abs_coef"] >= coef_df["abs_coef"].median()).sum())
    clear_effects = int((~coef_df["crosses_zero"]).sum())

    return (
        f"The fitted album model suggests that {get_display_label(target_col).lower()} "
        f"is shaped by several overlapping predictors rather than one dominant driver. "
        f"{clear_effects} coefficients have confidence intervals that do not cross 0, "
        f"and {strong_effects} predictors sit at or above the model’s median absolute effect size."
    )

def main() -> None:
    """
    Run the album regression explorer page.
    """
    st.set_page_config(
        page_title="Album Regression Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Album Regression Explorer")
    st.caption(
        "Model album-level popularity while preserving the staged regression workflow "
        "used in the analysis pipeline."
    )

    album_analytics_df = load_analysis_data()
    album_explorer_df = load_explorer_data()

    regression_filter_df = attach_regression_filter_metadata(
        album_analytics_df=album_analytics_df,
        album_explorer_df=album_explorer_df,
    )

    filter_inputs = get_global_filter_inputs(regression_filter_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    filtered_album_df = filter_dataset(regression_filter_df, global_controls).copy()

    if filtered_album_df.empty:
        st.warning("No albums remain under the current global filters.")
        st.stop()

    available_target_options = [
        col for col in ALBUM_REGRESSION_TARGET_CANDIDATES
        if col in filtered_album_df.columns
    ]

    exclude_feature_options = sorted(
        [
            col for col in filtered_album_df.columns
            if col not in {
            "tmdb_id",
            "release_group_mbid",
            "film_year",
            "film_genres",
            "album_genres_display",
            "lfm_album_listeners",
            "lfm_album_playcount",
            "log_lfm_album_listeners",
            "log_lfm_album_playcount",
            "album_cohesion_has_audio_data",
        }
        ]
    )

    controls = get_regression_controls(
        target_options=available_target_options,
        exclude_feature_options=exclude_feature_options,
    )

    target_col = controls.get("target_col", "log_lfm_album_listeners")
    threshold = controls["threshold"]
    excluded_features = list(set(EXCLUDED_MODEL_COLS | set(controls["excluded_features"])))

    if target_col not in filtered_album_df.columns:
        st.error(
            f"The selected regression target '{target_col}' is not available in the current album dataframe."
        )
        st.stop()

    regression_results = reg.run_regression_pipeline(
        album_analytics_df=filtered_album_df,
        target_col=target_col,
        threshold=threshold,
        excluded_features=excluded_features,
    )

    filter_results = regression_results["filter_results"]
    transform_results = regression_results["transform_results"]
    finalize_results = regression_results["finalize_results"]
    ols_results = regression_results["ols_results"]
    results = ols_results["results"]

    n_rows = ols_results["n_rows"]
    n_predictors = ols_results["n_predictors"]

    if n_rows < 30:
        st.error(
            f"""
            ⚠️ Not enough data to run a stable regression.

            Current selection produces only **{n_rows} albums**, which is too small
            for meaningful statistical modeling.

            👉 Try broadening your filters.
            """
        )
        st.stop()

    if n_rows < (5 * n_predictors):
        st.warning(
            f"""
            ⚠️ Regression may be unstable under the current filters.

            - Rows: {n_rows}
            - Predictors: {n_predictors}

            As a rule of thumb, you want at least **5–10 observations per predictor**.
            """
        )

    quality_flag = "Good"
    if n_rows < 30:
        quality_flag = "Insufficient"
    elif n_rows < (5 * n_predictors):
        quality_flag = "Weak"

    st.caption(
        build_regression_scope_caption(
            regression_results=regression_results,
            target_col=target_col,
        )
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modeling rows", f"{ols_results['n_rows']:,}")
    with col2:
        st.metric("Predictor count", f"{ols_results['n_predictors']:,}")
    with col3:
        st.metric(
            "Continuous predictors kept",
            f"{len(filter_results['kept_continuous'])} / {len(regression_results['feature_config']['continuous_features'])}",
        )
    with col4:
        st.metric("Model quality", quality_flag)

    st.subheader("Coefficient Plot")
    st.caption(
        "Dots show coefficient estimates, whiskers show 95% confidence intervals, "
        "and the vertical zero line marks no estimated effect. These are partial "
        "associations within the fitted multivariate model, not causal effects."
    )

    coef_df = render_coefficient_chart(results)

    render_coefficient_insight_cards(coef_df)

    st.caption(
        build_coefficient_supporting_insight(
            coef_df,
            target_col=target_col,
        )
    )

    st.markdown("### 📈 Modeling Takeaway")
    st.caption(
        build_album_modeling_takeaway(
            coef_df,
            target_col=target_col,
        )
    )

    if controls["show_coefficient_table"]:
        st.subheader("Coefficient Dataframe")
        st.dataframe(
            rename_columns_for_display(coef_df.round(3)),
            width="stretch",
        )

    if controls["show_filter_summary"]:
        render_filter_summary(filter_results)

    if controls["show_transform_summary"]:
        render_transform_summary(transform_results)

    if controls["show_finalize_summary"]:
        render_finalize_summary(finalize_results)

    if controls["show_model_summary"]:
        render_model_summary(results)


if __name__ == "__main__":
    main()