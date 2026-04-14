from __future__ import annotations

import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_regression_controls,
)
from app.app_data import load_track_data_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    get_global_filter_inputs,
    get_track_page_display_label,
    rename_track_page_columns_for_display,
)
from app.ui import apply_app_styles
from track_regression_analysis import (
    TRACK_REGRESSION_TARGET_OPTIONS,
    build_track_coefficient_plot_df,
    run_track_regression_pipeline,
)
from track_regression_visualization import create_track_coefficient_whisker_chart


def build_track_regression_scope_caption(
    regression_results: dict,
    target_col: str,
    include_context_controls: bool,
) -> str:
    feature_config = regression_results["feature_config"]
    filter_results = regression_results["filter_results"]
    ols_results = regression_results["ols_results"]

    continuous_kept = len(filter_results["kept_continuous"])
    continuous_total = len(feature_config["continuous_features"])

    context_status = (
        "with film & album controls"
        if include_context_controls
        else "track-only model"
    )

    return (
        f"Current scope ({context_status}): target = {get_track_page_display_label(target_col)}, "
        f"{ols_results['n_rows']:,} modeled rows, "
        f"{ols_results['n_predictors']:,} final predictors, and "
        f"{continuous_kept} of {continuous_total} continuous candidates "
        f"retained after the initial screening step."
    )

def build_track_coefficient_insight_summary(coef_df) -> dict[str, str]:
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
            f"'{get_track_page_display_label(top_feature)}' has the largest absolute coefficient "
            f"and is positive ({top_coef:.3f})."
        )
    elif top_coef < 0:
        top_caption = (
            f"'{get_track_page_display_label(top_feature)}' has the largest absolute coefficient "
            f"and is negative ({top_coef:.3f})."
        )
    else:
        top_caption = (
            f"'{get_track_page_display_label(top_feature)}' has the largest absolute coefficient "
            f"but its estimated effect is effectively zero ({top_coef:.3f})."
        )

    return {
        "card1_title": "Largest Absolute Effect",
        "card1_value": get_track_page_display_label(top_feature),
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


def render_track_coefficient_insight_cards(coef_df) -> None:
    insights = build_track_coefficient_insight_summary(coef_df)

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


def build_track_coefficient_supporting_insight(coef_df, target_col: str) -> str:
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
        f"💡 The coefficient plot for {get_track_page_display_label(target_col).lower()} is led by "
        f"'{get_track_page_display_label(top_feature)}' (absolute coefficient {top_abs:.3f}). "
        f"The median absolute effect across all predictors is {median_abs:.3f}, and "
        f"{not_crossing_zero} of {total_coefficients} coefficients have intervals "
        f"that do not cross 0."
    )


def build_track_modeling_takeaway(coef_df, target_col: str) -> str:
    if coef_df.empty:
        return (
            f"No fitted model takeaway is available for "
            f"{get_track_page_display_label(target_col).lower()}."
        )

    strong_effects = int((coef_df["abs_coef"] >= coef_df["abs_coef"].median()).sum())
    clear_effects = int((~coef_df["crosses_zero"]).sum())

    return (
        f"The fitted track model suggests that {get_track_page_display_label(target_col).lower()} "
        f"is shaped by several overlapping predictors rather than one dominant driver. "
        f"{clear_effects} coefficients have confidence intervals that do not cross 0, "
        f"and {strong_effects} predictors sit at or above the model’s median absolute effect size."
    )


def main() -> None:
    st.set_page_config(
        page_title="Track Regression Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Regression Explorer")
    st.write(
        """
        Model track-level performance while cleanly separating native track features
        from attached film and album context. This page shows which predictors retain
        directional signal inside a multivariate regression and how those estimates
        change when broader film and album controls are included.
        """
    )

    track_df = load_track_data_explorer_data()

    filter_inputs = get_global_filter_inputs(track_df)
    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    filtered_track_df = filter_dataset(track_df, global_controls).copy()

    if filtered_track_df.empty:
        st.warning("No tracks remain under the current global filters.")
        st.stop()

    exclude_feature_options = sorted(
        [
            col for col in filtered_track_df.columns
            if col not in {
            "tmdb_id",
            "release_group_mbid",
            "film_year",
            "film_genres",
            "album_genres_display",
            "lfm_track_listeners",
            "lfm_track_playcount",
            "spotify_popularity",
            "log_lfm_track_listeners",
            "log_lfm_track_playcount",
            "album_cohesion_has_audio_data",
            "track_audio_feature_count",
            "track_has_any_audio_features",
        }
        ]
    )

    regression_controls = get_track_regression_controls(
        target_options=TRACK_REGRESSION_TARGET_OPTIONS,
        exclude_feature_options=exclude_feature_options,
    )

    target_col = regression_controls["target_col"]
    threshold = regression_controls["threshold"]
    use_context_controls = regression_controls["include_context_controls"]
    excluded_features = regression_controls["excluded_features"]
    show_coefficient_table = regression_controls["show_coefficient_table"]
    show_filter_summary = regression_controls["show_filter_summary"]
    show_transform_summary = regression_controls["show_transform_summary"]
    show_finalize_summary = regression_controls["show_finalize_summary"]
    show_model_summary = regression_controls["show_model_summary"]

    regression_results = run_track_regression_pipeline(
        track_df=filtered_track_df,
        target_col=target_col,
        threshold=threshold,
        include_context_controls=use_context_controls,
        global_controls=global_controls,
        excluded_features=excluded_features,
    )

    n_rows = regression_results["ols_results"]["n_rows"]
    n_predictors = regression_results["ols_results"]["n_predictors"]

    # Hard failure: not enough data to run regression
    if n_rows < 30:
        st.error(
            f"""
            ⚠️ Not enough data to run a stable regression.

            Current selection produces only **{n_rows} tracks**, which is too small
            for meaningful statistical modeling.

            👉 Try broadening your filters (e.g., remove a genre or expand the year range).
            """
        )
        st.stop()

    # Soft warning: model is underpowered
    if n_rows < (5 * n_predictors):
        st.warning(
            f"""
            ⚠️ Regression may be unstable under the current filters.

            - Rows: {n_rows}
            - Predictors: {n_predictors}

            As a rule of thumb, you want at least **5–10 observations per predictor**.

            👉 Consider reducing filters or unchecking *"Include film & album controls"*.
            """
        )

    filter_results = regression_results["filter_results"]
    transform_results = regression_results["transform_results"]
    finalize_results = regression_results["finalize_results"]
    ols_results = regression_results["ols_results"]
    results = ols_results["results"]

    n_rows = ols_results["n_rows"]
    n_predictors = ols_results["n_predictors"]

    quality_flag = "Good"
    if n_rows < 30:
        quality_flag = "Insufficient"
    elif n_rows < (5 * n_predictors):
        quality_flag = "Weak"

    st.caption(
        build_track_regression_scope_caption(
            regression_results=regression_results,
            target_col=target_col,
            include_context_controls=use_context_controls,
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
        "and the vertical zero line marks no estimated effect. Track, film, and album "
        "predictors are shown together, but their labels are prefixed so the source of "
        "each signal remains visually clear. These are partial associations within the "
        "fitted model, not causal effects."
    )

    coef_df = build_track_coefficient_plot_df(results)
    chart = create_track_coefficient_whisker_chart(coef_df, target_col=target_col)
    st.altair_chart(chart, width="stretch")

    render_track_coefficient_insight_cards(coef_df)

    st.caption(build_track_coefficient_supporting_insight(coef_df, target_col))

    st.markdown("### 📈 Modeling Takeaway")
    st.caption(build_track_modeling_takeaway(coef_df, target_col))

    if show_coefficient_table:
        st.subheader("Coefficient Dataframe")
        st.dataframe(
            rename_track_page_columns_for_display(coef_df.round(3)),
            width="stretch",
        )

    if show_filter_summary:
        st.subheader("Feature Filtering Summary")
        st.dataframe(
            rename_track_page_columns_for_display(filter_results["summary_df"].round(3)),
            width="stretch",
        )

    if show_transform_summary:
        st.subheader("Transform Summary")
        st.write("Logged predictors")
        st.write(transform_results["logged_predictors"])

    if show_finalize_summary:
        st.subheader("Final Predictor Cleanup")
        st.write(finalize_results["dropped_columns"])

    if show_model_summary:
        st.subheader("Full OLS Summary")
        st.text(results.summary().as_text())


if __name__ == "__main__":
    main()