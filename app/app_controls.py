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


def get_scatter_controls(
    guided_feature_options: list[str],
    freeform_numeric_options: list[str],
    color_options: list[str],
    default_y: str,
) -> dict:
    """
    Render sidebar controls for the relationship explorer.

    Args:
        guided_feature_options: Ranked continuous features used in Guided mode.
        freeform_numeric_options: Numeric columns available for freeform X/Y.
        color_options: Optional categorical fields available for color encoding.
        default_y: Default y-axis field for freeform mode.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Relationship Controls")

    mode = st.sidebar.radio(
        "Explorer mode",
        options=["Guided", "Freeform"],
        index=0,
        help=(
            "Guided uses the regression-oriented ranked feature view. "
            "Freeform lets you choose any album-level numeric X and Y."
        ),
    )

    if mode == "Guided":
        selected_feature = st.sidebar.selectbox(
            "X-axis feature",
            options=guided_feature_options,
            index=0,
        )

        show_trendline = st.sidebar.checkbox(
            "Show fitted line",
            value=True,
        )

        show_feature_ranking = st.sidebar.checkbox(
            "Show ranked feature table",
            value=True,
        )

        x_col = selected_feature
        y_col = default_y
        color_col = "None"
        transform_x = "None"
        transform_y = "None"
        apply_jitter = False
        jitter_strength = 0.0

    else:
        default_x_index = 0
        default_y_index = (
            freeform_numeric_options.index(default_y)
            if default_y in freeform_numeric_options
            else 0
        )

        x_col = st.sidebar.selectbox(
            "X-axis",
            options=freeform_numeric_options,
            index=default_x_index,
        )

        y_col = st.sidebar.selectbox(
            "Y-axis",
            options=freeform_numeric_options,
            index=default_y_index,
        )

        color_col = st.sidebar.selectbox(
            "Color grouping",
            options=["None"] + color_options,
            index=0,
        )

        st.sidebar.markdown("---")
        st.sidebar.caption("Display and scale controls")

        transform_x = st.sidebar.selectbox(
            "Transform X",
            options=["None", "Log1p"],
            index=0,
        )

        transform_y = st.sidebar.selectbox(
            "Transform Y",
            options=["None", "Log1p"],
            index=0,
        )

        apply_jitter = st.sidebar.checkbox(
            "Jitter points",
            value=False,
            help=(
                "Adds small random offsets for display only. "
                "Useful when X or Y has many repeated values."
            ),
        )

        if apply_jitter:
            jitter_strength = st.sidebar.slider(
                "Jitter strength",
                min_value=0.000,
                max_value=0.050,
                value=0.008,
                step=0.001,
                format="%.3f",
            )
        else:
            jitter_strength = 0.0

        show_trendline = st.sidebar.checkbox(
            "Show fitted line",
            value=True,
        )

        show_feature_ranking = False
        selected_feature = None

    show_data_table = st.sidebar.checkbox(
        "Show scatterplot data table",
        value=False,
    )

    return {
        "mode": mode,
        "selected_feature": selected_feature,
        "x_col": x_col,
        "y_col": y_col,
        "color_col": color_col,
        "transform_x": transform_x,
        "transform_y": transform_y,
        "apply_jitter": apply_jitter,
        "jitter_strength": jitter_strength,
        "show_trendline": show_trendline,
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