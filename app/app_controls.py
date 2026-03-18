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

def get_ridge_controls(
    preset_labels: dict[str, str],
    preset_keys: list[str],
    all_available_features: list[str],
    default_custom_features: list[str],
) -> dict:
    """
    Render sidebar controls for the ridge explorer.

    Args:
        preset_labels: Mapping from preset key to display label.
        preset_keys: Ordered list of preset keys.
        all_available_features: Full set of available ridge features.
        default_custom_features: Default feature list to preselect when
            the user chooses the custom preset.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Ridge Explorer Controls")

    preset_key = st.sidebar.selectbox(
        "Preset",
        options=preset_keys,
        index=0,
        format_func=lambda x: preset_labels[x],
    )

    top_n = st.sidebar.slider(
        "Top-N size for dynamic presets",
        min_value=5,
        max_value=12,
        value=8,
        step=1,
    )

    if preset_key == "custom":
        selected_features = st.sidebar.multiselect(
            "Custom features",
            options=all_available_features,
            default=[
                f for f in default_custom_features
                if f in all_available_features
            ],
        )
    else:
        selected_features = []

    st.sidebar.markdown("---")

    with st.sidebar.expander("Advanced density controls", expanded=False):
        bins = st.slider(
            "Bins",
            min_value=40,
            max_value=120,
            value=80,
            step=10,
        )
        smooth_window = st.slider(
            "Smoothing window",
            min_value=3,
            max_value=21,
            value=11,
            step=2,
        )
        min_group_n = st.slider(
            "Minimum group size",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
        )

    show_order_table = st.sidebar.checkbox(
        "Show ordering table",
        value=True,
    )

    show_density_table = st.sidebar.checkbox(
        "Show density dataframe sample",
        value=False,
    )

    show_ridge_long_sample = st.sidebar.checkbox(
        "Show ridge_long sample",
        value=False,
    )

    return {
        "preset_key": preset_key,
        "top_n": top_n,
        "selected_features": selected_features,
        "bins": bins,
        "smooth_window": smooth_window,
        "min_group_n": min_group_n,
        "show_order_table": show_order_table,
        "show_density_table": show_density_table,
        "show_ridge_long_sample": show_ridge_long_sample,
    }