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
    st.subheader("OLS Summary")
    st.text(results.summary().as_text())


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
        This page exposes the staged regression workflow, including
        feature screening, modeling transforms, final predictor cleanup,
        and the fitted OLS coefficient view.
        """
    )

    controls = get_regression_controls()
    regression_results = load_regression_results()

    filter_results = regression_results["filter_results"]
    transform_results = regression_results["transform_results"]
    finalize_results = regression_results["finalize_results"]
    ols_results = regression_results["ols_results"]
    results = ols_results["results"]

    render_regression_metrics(regression_results)

    st.subheader("Coefficient Plot")
    coef_df = render_coefficient_chart(results)

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