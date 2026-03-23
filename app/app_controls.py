import streamlit as st
from app.ui import get_display_label

# Global Filter Controls (multiple pages)
def get_global_filter_controls(
    min_year: int,
    max_year: int,
    film_genre_options: list[str],
    album_genre_options: list[str],
) -> dict:
    st.sidebar.markdown("### Global Filters")

    year_range = st.sidebar.slider(
        "Film year",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )

    film_genres = st.sidebar.multiselect(
        "Film genres",
        options=film_genre_options,
        default=[],
    )

    album_genres = st.sidebar.multiselect(
        "Album genres",
        options=album_genre_options,
        default=[],
    )

    return {
        "year_range": year_range,
        "selected_film_genres": film_genres,
        "selected_album_genres": album_genres,
    }

# PAGE 1 Controls
def get_dataset_controls(
    min_year: int,
    max_year: int,
    film_genre_options: list[str],
    album_genre_options: list[str],
    composer_options: list[str],
    label_options: list[str],
) -> dict:
    """
    Render sidebar controls for the dataset explorer.

    Args:
        min_year: Minimum film year available in the data.
        max_year: Maximum film year available in the data.
        film_genre_options: Sorted list of available film genre labels.
        album_genre_options: Sorted list of available album genre labels.
        composer_options: Sorted list of available composer names.
        label_options: Sorted list of available label names.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Global Filters")

    year_range = st.sidebar.slider(
        "Film year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )

    selected_film_genres = st.sidebar.multiselect(
        "Film genres",
        options=film_genre_options,
        default=[],
        help="Leave blank to include all film genres.",
    )

    selected_album_genres = st.sidebar.multiselect(
        "Album genres",
        options=album_genre_options,
        default=[],
        help="Leave blank to include all album genres.",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Dataset Controls")

    selected_composers = st.sidebar.multiselect(
        "Composers",
        options=composer_options,
        default=[],
        help="Leave blank to include all composers.",
    )

    selected_labels = st.sidebar.multiselect(
        "Labels",
        options=label_options,
        default=[],
        help="Leave blank to include all labels.",
    )

    search_text = st.sidebar.text_input(
        "Search album, film, composer, or label",
        value="",
        help=(
            "Case-insensitive text search across album title, film title, "
            "composer, and label."
        ),
    )

    min_tracks = st.sidebar.slider(
        "Minimum number of tracks",
        min_value=1,
        max_value=40,
        value=1,
        step=1,
    )

    listeners_only = st.sidebar.checkbox(
        "Only show albums with Last.fm listener data",
        value=True,
    )

    show_data_table = st.sidebar.checkbox(
        "Show full filtered table",
        value=True,
    )

    return {
        "year_range": year_range,
        "selected_film_genres": selected_film_genres,
        "selected_album_genres": selected_album_genres,
        "selected_composers": selected_composers,
        "selected_labels": selected_labels,
        "search_text": search_text,
        "min_tracks": min_tracks,
        "listeners_only": listeners_only,
        "show_data_table": show_data_table,
    }

# PAGE 2 Controls
def get_distribution_controls(
    numeric_options: list[str],
    group_options: list[str],
    group_value_options_map: dict[str, list[str]],
) -> dict:
    """
    Render sidebar controls for the Distribution Explorer.

    Args:
        numeric_options: Numeric columns eligible for distribution analysis.
        group_options: Grouping fields eligible for comparison.
        group_value_options_map: Mapping from grouping field to selectable
            group values.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Distribution Controls")

    metric = st.sidebar.selectbox(
        "Metric",
        options=numeric_options,
        index=0,
        format_func=get_display_label,
    )

    view_type = st.sidebar.selectbox(
        "View type",
        options=["Histogram", "Density", "CDF"],
        index=0,
    )

    use_log = st.sidebar.checkbox(
        "Log scale (log10)",
        value=True,
        help="Applies log10 to positive values only.",
    )

    group_var = st.sidebar.selectbox(
        "Group by",
        options=["None"] + group_options,
        index=0,
        format_func=lambda x: "None" if x == "None" else get_display_label(x),
    )

    selected_groups = []
    top_n = None

    if group_var != "None":
        selected_groups = st.sidebar.multiselect(
            f"Select {get_display_label(group_var)} values",
            options=group_value_options_map.get(group_var, []),
            default=[],
            help=(
                "Type to search specific values. Leave blank to use the top N "
                "groups by album count."
            ),
        )

        if not selected_groups:
            top_n = st.sidebar.slider(
                "Top N groups to display",
                min_value=3,
                max_value=15,
                value=8,
                step=1,
            )

    bins = st.sidebar.slider(
        "Bins",
        min_value=20,
        max_value=100,
        value=40,
        step=5,
    )

    show_table = st.sidebar.checkbox(
        "Show data table",
        value=False,
    )

    return {
        "metric": metric,
        "view_type": view_type,
        "use_log": use_log,
        "group_var": group_var,
        "selected_groups": selected_groups,
        "top_n": top_n,
        "bins": bins,
        "show_table": show_table,
    }

# PAGE 3 Controls

def get_group_comparison_controls(
    numeric_options: list[str],
    group_options: list[str],
    group_value_options_map: dict[str, list[str]],
) -> dict:
    """
    Render sidebar controls for the Group Comparison Explorer.

    Args:
        numeric_options: Numeric columns eligible for comparison.
        group_options: Grouping fields eligible for comparison.
        group_value_options_map: Mapping from grouping field to selectable
            group values.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Group Comparison Controls")

    metric = st.sidebar.selectbox(
        "Metric",
        options=numeric_options,
        index=0,
        format_func=get_display_label,
    )

    view_mode = st.sidebar.selectbox(
        "View mode",
        options=["Boxplot", "Violin", "Bar Ranking"],
        index=0,
    )

    group_var = st.sidebar.selectbox(
        "Group by",
        options=group_options,
        index=0,
        format_func=get_display_label,
    )

    selected_groups = st.sidebar.multiselect(
        f"Select {get_display_label(group_var)} values",
        options=group_value_options_map.get(group_var, []),
        default=[],
        help=(
            "Type to search specific values. Leave blank to use the top N "
            "groups by album count."
        ),
    )

    top_n = None
    if not selected_groups:
        top_n = st.sidebar.slider(
            "Top N groups to display",
            min_value=3,
            max_value=15,
            value=8,
            step=1,
        )

    min_group_size = st.sidebar.slider(
        "Minimum albums per group",
        min_value=3,
        max_value=50,
        value=5,
        step=1,
        help="Exclude groups with fewer than this many albums.",
    )

    stratify_by = "None"
    max_strata = 4

    if view_mode in ["Boxplot", "Violin"]:
        use_log = st.sidebar.checkbox(
            "Log scale (log10)",
            value=True,
            help="Applies log10 to positive values only.",
        )
        ranking_stat = None
    else:
        use_log = False
        ranking_stat = st.sidebar.selectbox(
            "Ranking statistic",
            options=["Median", "Mean", "Total", "Count"],
            index=0,
        )

        stratify_by = st.sidebar.selectbox(
            "Stratify by",
            options=[
                "None",
                "album_genre_group",
                "film_genre_group",
                "album_us_release_year",
                "bafta_nominee",
                "oscar_score_nominee",
                "oscar_song_nominee",
                "globes_score_nominee",
                "globes_song_nominee",
                "critics_score_nominee",
                "critics_song_nominee",
            ],
            index=0,
            format_func=lambda x: "None" if x == "None" else get_display_label(x),
            help="Split each ranked bar into stacked segments by a second category.",
        )

        if stratify_by != "None":
            max_strata = st.sidebar.slider(
                "Maximum strata to display",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
                help="Keep only the largest strata and roll the rest into Others.",
            )

    show_table = st.sidebar.checkbox(
        "Show data table",
        value=False,
    )

    show_points = False
    if view_mode in ["Boxplot", "Violin"]:
        show_points = st.sidebar.checkbox(
            "Show individual album points",
            value=False,
        )

    genre_mode = "Collapsed genre groups"
    if (
        view_mode in ["Boxplot", "Violin"]
        and group_var in ["album_genre_group", "film_genre_group"]
    ):
        genre_mode = st.sidebar.radio(
            "Genre handling",
            options=[
                "Collapsed genre groups",
                "Include albums in all matching genres",
            ],
            index=0,
            help=(
                "Collapsed genre groups keeps one genre label per album. "
                "Include albums in all matching genres lets the same album "
                "appear in multiple genre distributions."
            ),
        )

    return {
        "metric": metric,
        "view_mode": view_mode,
        "group_var": group_var,
        "selected_groups": selected_groups,
        "top_n": top_n,
        "min_group_size": min_group_size,
        "use_log": use_log,
        "ranking_stat": ranking_stat,
        "show_table": show_table,
        "show_points": show_points,
        "stratify_by": stratify_by,
        "max_strata": max_strata,
        "genre_mode": genre_mode,
    }

# PAGE 4 Controls
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
            format_func=get_display_label,
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
            format_func=get_display_label,
        )

        y_col = st.sidebar.selectbox(
            "Y-axis",
            options=freeform_numeric_options,
            index=default_y_index,
            format_func=get_display_label,
        )

        color_col = st.sidebar.selectbox(
            "Color grouping",
            options=["None"] + color_options,
            index=0,
            format_func=lambda x: "None" if x == "None" else get_display_label(x),
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

# PAGE 8 Controls
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

# PAGE 9 Controls
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

# PAGE 10 Controls
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
            format_func=get_display_label,
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