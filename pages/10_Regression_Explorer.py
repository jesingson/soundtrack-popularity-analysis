import streamlit as st

import regression_analysis as reg
import regression_visualization as reg_viz
from app.app_controls import get_regression_controls
from app.app_data import load_regression_results
from app.ui import apply_app_styles, rename_columns_for_display


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
def build_regression_scope_caption(regression_results: dict) -> str:
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
        f"Current scope: {ols_results['n_rows']:,} modeled rows, "
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


def build_coefficient_supporting_insight(coef_df) -> str:
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
        f"💡 The coefficient plot is led by '{top_feature}' "
        f"(absolute coefficient {top_abs:.3f}). The median absolute effect "
        f"across all predictors is {median_abs:.3f}, and "
        f"{not_crossing_zero} of {total_coefficients} coefficients have "
        "intervals that do not cross 0."
    )

def main() -> None:
    """
    Run the regression explorer page.
    """
    st.set_page_config(
        page_title="Regression Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Regression Explorer")
    st.write(
        """
        This page walks through the staged regression workflow used to model
        soundtrack popularity. The coefficient plot is the main interpretation
        view: it shows each predictor's estimated direction and size in the
        multivariate model, along with uncertainty from the 95% confidence interval.
        """
    )

    controls = get_regression_controls()
    regression_results = load_regression_results()

    filter_results = regression_results["filter_results"]
    transform_results = regression_results["transform_results"]
    finalize_results = regression_results["finalize_results"]
    ols_results = regression_results["ols_results"]
    results = ols_results["results"]

    st.caption(
        build_regression_scope_caption(regression_results)
    )

    render_regression_metrics(regression_results)

    st.subheader("Coefficient Plot")
    st.caption(
        "Dots show coefficient estimates, whiskers show 95% confidence intervals, "
        "and the vertical zero line marks no estimated effect. These are partial "
        "associations within the fitted model, not causal effects."
    )
    coef_df = render_coefficient_chart(results)

    render_coefficient_insight_cards(coef_df)

    st.caption(
        build_coefficient_supporting_insight(coef_df)
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