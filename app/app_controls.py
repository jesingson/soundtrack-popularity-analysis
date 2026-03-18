import streamlit as st


def get_correlation_controls() -> dict:
    """
    Render sidebar controls for the correlation explorer.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Correlation Controls")

    method = st.sidebar.selectbox(
        "Correlation method",
        options=["pearson", "spearman"],
        index=0,
    )

    top_n = st.sidebar.slider(
        "Top N features for lollipop chart",
        min_value=5,
        max_value=25,
        value=15,
        step=1,
    )

    show_lollipop_table = st.sidebar.checkbox(
        "Show lollipop data table",
        value=False,
    )

    show_heatmap_table = st.sidebar.checkbox(
        "Show heatmap source table",
        value=False,
    )

    return {
        "method": method,
        "top_n": top_n,
        "show_lollipop_table": show_lollipop_table,
        "show_heatmap_table": show_heatmap_table,
    }


def get_scatter_controls(feature_options: list[str]) -> dict:
    """
    Render sidebar controls for the scatterplot explorer.

    Args:
        feature_options: Ordered list of selectable feature names.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Scatterplot Controls")

    selected_feature = st.sidebar.selectbox(
        "X-axis feature",
        options=feature_options,
        index=0,
    )

    show_data_table = st.sidebar.checkbox(
        "Show scatterplot data table",
        value=False,
    )

    show_feature_ranking = st.sidebar.checkbox(
        "Show ranked feature table",
        value=True,
    )

    return {
        "selected_feature": selected_feature,
        "show_data_table": show_data_table,
        "show_feature_ranking": show_feature_ranking,
    }


def get_regression_controls() -> dict:
    """
    Render sidebar controls for the regression explorer.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Regression Controls")

    show_filter_summary = st.sidebar.checkbox(
        "Show feature filtering summary",
        value=True,
    )

    show_transform_summary = st.sidebar.checkbox(
        "Show transform summary",
        value=True,
    )

    show_finalize_summary = st.sidebar.checkbox(
        "Show final predictor cleanup",
        value=True,
    )

    show_coefficient_table = st.sidebar.checkbox(
        "Show coefficient dataframe",
        value=False,
    )

    show_model_summary = st.sidebar.checkbox(
        "Show full OLS summary",
        value=True,
    )

    return {
        "show_filter_summary": show_filter_summary,
        "show_transform_summary": show_transform_summary,
        "show_finalize_summary": show_finalize_summary,
        "show_coefficient_table": show_coefficient_table,
        "show_model_summary": show_model_summary,
    }