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

# PAGE 5 controls
def get_concentration_controls(
    metric_options: list[str],
    group_options: list[str],
    group_value_options_map: dict[str, list[str]],
) -> dict:
    """
    Render sidebar controls for the Concentration Explorer.

    Args:
        metric_options: Numeric album-level metrics eligible for
            concentration analysis.
        group_options: Grouping fields eligible for concentration analysis.
        group_value_options_map: Mapping from grouping field to selectable
            group values.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Concentration Controls")

    metric = st.sidebar.selectbox(
        "Album performance metric",
        options=metric_options,
        index=0,
        format_func=get_display_label,
        help=(
            "Choose the album-level performance measure used to compute "
            "group totals, Top-K shares, and Gini."
        ),
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
            max_value=20,
            value=8,
            step=1,
        )

    min_group_size = st.sidebar.slider(
        "Minimum albums per group",
        min_value=2,
        max_value=20,
        value=3,
        step=1,
    )

    ranking_metric = st.sidebar.selectbox(
        "Group ranking statistic",
        options=[
            "total_metric",
            "gini",
            "top_1_share",
            "top_3_share",
            "top_5_share",
        ],
        index=1,
        format_func=get_display_label,
        help=(
            "Total Metric, Top-K Share, and Gini are computed from the "
            "selected album performance metric."
        ),
    )

    default_lorenz_groups = selected_groups[:6] if selected_groups else []
    lorenz_groups = st.sidebar.multiselect(
        "Lorenz curve groups",
        options=group_value_options_map.get(group_var, []),
        default=[],
        max_selections=6,
        help=(
            "Choose up to 6 displayed groups for the Lorenz curve. "
            "If left blank, the top 6 displayed groups by the selected "
            "group ranking statistic will be used."
        ),
    )

    histogram_bins = st.sidebar.slider(
        "Histogram bins",
        min_value=8,
        max_value=30,
        value=16,
        step=1,
    )

    show_summary_table = st.sidebar.checkbox(
        "Show concentration summary table",
        value=True,
    )

    show_detail_table = st.sidebar.checkbox(
        "Show group drilldown table",
        value=True,
    )

    return {
        "metric": metric,
        "group_var": group_var,
        "selected_groups": selected_groups,
        "top_n": top_n,
        "min_group_size": min_group_size,
        "ranking_metric": ranking_metric,
        "lorenz_groups": lorenz_groups,
        "histogram_bins": histogram_bins,
        "show_summary_table": show_summary_table,
        "show_detail_table": show_detail_table,
    }

# PAGE 6 Controls
def get_cooccurrence_controls(
    relationship_options: list[str],
) -> dict:
    """
    Render sidebar controls for the Co-occurrence Explorer.

    Args:
        relationship_options: Available self-entity relationship modes.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Co-occurrence Controls")

    relationship_type = st.sidebar.radio(
        "Relationship type",
        options=relationship_options,
        index=0,
        help=(
            "Choose which within-entity co-occurrence network to explore."
        ),
    )

    min_edge_count = st.sidebar.slider(
        "Minimum co-occurrence",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help="Hide weak relationships below this album count.",
    )

    edge_metric = st.sidebar.selectbox(
        "Edge metric",
        options=[
            "Count",
            "% of source",
            "% of target",
            "Jaccard similarity",
            "Lift",
        ],
        index=0,
        help=(
            "Choose how pair strength is measured in the chord thickness, "
            "ranking chart, and summary table."
        ),
    )

    top_n_relationships = st.sidebar.slider(
        "Number of ranked relationships",
        min_value=5,
        max_value=21,
        value=10,
        step=1,
    )

    show_edge_table = st.sidebar.checkbox(
        "Show relationship summary table",
        value=True,
    )

    show_album_table = st.sidebar.checkbox(
        "Show selected-relationship album table",
        value=True,
    )

    return {
        "relationship_type": relationship_type,
        "min_edge_count": min_edge_count,
        "edge_metric": edge_metric,
        "top_n_edges": top_n_relationships,
        "show_edge_table": show_edge_table,
        "show_album_table": show_album_table,
    }


# PAGE 7 Controls
def get_cross_entity_controls(
    relationship_options: list[str],
) -> dict:
    """
    Render sidebar controls for the Cross-Entity Explorer.

    Args:
        relationship_options: Available cross-entity relationship modes.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Cross-Entity Controls")

    relationship_type = st.sidebar.radio(
        "Relationship type",
        options=relationship_options,
        index=0,
        help=(
            "Choose a directional single-hop flow or a constrained multi-hop "
            "path view."
        ),
    )

    min_edge_count = st.sidebar.slider(
        "Minimum relationship count",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
        help="Hide weak visible flows or paths below this contribution count.",
    )

    edge_metric = st.sidebar.selectbox(
        "Relationship metric",
        options=[
            "Count",
            "% of source",
            "% of target",
            "Jaccard similarity",
            "Lift",
        ],
        index=0,
        help=(
            "Choose how visible relationship strength is measured in the Sankey, "
            "ranking chart, and summary table."
        ),
    )

    top_n_relationships = st.sidebar.slider(
        "Number of ranked relationships",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("High-cardinality controls")

    min_composer_count = st.sidebar.slider(
        "Minimum composer frequency",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Hide composers that appear on too few filtered albums.",
    )

    top_n_composers = st.sidebar.slider(
        "Top N composers",
        min_value=5,
        max_value=50,
        value=15,
        step=1,
        help="Keep only the most frequent visible composers.",
    )

    min_label_count = st.sidebar.slider(
        "Minimum label frequency",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Hide labels that appear on too few filtered albums.",
    )

    top_n_labels = st.sidebar.slider(
        "Top N labels",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Keep only the most frequent visible labels.",
    )

    show_edge_table = st.sidebar.checkbox(
        "Show relationship summary table",
        value=True,
    )

    show_album_table = st.sidebar.checkbox(
        "Show selected-relationship album table",
        value=True,
    )

    return {
        "relationship_type": relationship_type,
        "min_edge_count": min_edge_count,
        "edge_metric": edge_metric,
        "top_n_edges": top_n_relationships,
        "min_composer_count": min_composer_count,
        "top_n_composers": top_n_composers,
        "min_label_count": min_label_count,
        "top_n_labels": top_n_labels,
        "show_edge_table": show_edge_table,
        "show_album_table": show_album_table,
    }

# PAGE 8 Controls
def get_correlation_controls() -> dict:
    """
    Render sidebar controls for the Correlation Explorer.

    Returns:
        dict: Control values for the page.
    """
    st.sidebar.header("Correlation Explorer Controls")

    method = st.sidebar.radio(
        "Correlation method",
        options=["pearson", "spearman"],
        index=0,
        help=(
            "Pearson emphasizes linear relationships. "
            "Spearman emphasizes monotonic rank-order relationships."
        ),
    )

    ranking_mode = st.sidebar.selectbox(
        "Lollipop ranking mode",
        options=["Absolute", "Positive only", "Negative only"],
        index=0,
        help=(
            "Absolute shows the strongest relationships by magnitude. "
            "Positive only and Negative only restrict the ranking to one direction."
        ),
    )

    top_n = st.sidebar.slider(
        "Top features in lollipop chart",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Heatmap Controls")

    heatmap_scope = st.sidebar.radio(
        "Heatmap feature scope",
        options=["Full matrix", "Top lollipop features only"],
        index=0,
        help=(
            "Use the full feature matrix, or limit the heatmap to the same top-ranked "
            "features shown in the lollipop chart."
        ),
    )

    heatmap_top_n = st.sidebar.slider(
        "Features in reduced heatmap",
        min_value=5,
        max_value=25,
        value=12,
        step=1,
        disabled=(heatmap_scope != "Top lollipop features only"),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tables")

    show_lollipop_table = st.sidebar.checkbox(
        "Show lollipop source table",
        value=False,
    )

    show_heatmap_table = st.sidebar.checkbox(
        "Show heatmap source tables",
        value=False,
    )

    return {
        "method": method,
        "ranking_mode": ranking_mode,
        "top_n": top_n,
        "heatmap_scope": heatmap_scope,
        "heatmap_top_n": heatmap_top_n,
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


# PAGE 20 Controls
def get_track_structure_controls(
    metric_options: list[str],
    color_options: list[str],
    composer_options: list[str],
) -> dict:
    """
    Render sidebar controls for the Track Structure Explorer.

    Args:
        metric_options: Track-level numeric metrics eligible for analysis.
        color_options: Optional categorical fields available for scatter
            color encoding.
        composer_options: Available composer names for optional filtering.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Track Structure Controls")

    metric = st.sidebar.selectbox(
        "Track performance metric",
        options=metric_options,
        index=0,
        format_func=get_display_label,
    )

    transform_y = st.sidebar.selectbox(
        "Transform Y",
        options=["None", "Log1p"],
        index=1,
        help="Apply log1p to positive values for display and summaries.",
    )
    analysis_basis = st.sidebar.selectbox(
        "Analysis basis",
        options=["Raw metric", "Within-album share"],
        index=0,
        help=(
            "Raw metric shows absolute track performance. "
            "Within-album share shows each track's share of its album's visible total."
        ),
    )

    max_track_position = st.sidebar.slider(
        "Maximum track position",
        min_value=5,
        max_value=40,
        value=20,
        step=1,
        help="Restrict analysis to earlier track positions if desired.",
    )

    min_tracks_per_album = st.sidebar.slider(
        "Minimum tracks per album",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Exclude very short albums from the track-structure analysis.",
    )

    selected_composers = st.sidebar.multiselect(
        "Composers to compare",
        options=composer_options,
        default=[],
        max_selections=12,
        help=(
            "Optionally limit the page to a focused set of composers. "
            "Leave blank to include all composers."
        ),
    )

    summary_stat = st.sidebar.selectbox(
        "Summary statistic",
        options=["Median", "Mean", "Both"],
        index=0,
    )

    show_ribbon = st.sidebar.checkbox(
        "Show interquartile ribbon",
        value=True,
        help="Display the 25th to 75th percentile band around the summary.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Scatter controls")

    if selected_composers:
        st.sidebar.info(
            "Scatterplot colors are locked to composer when composers are selected."
        )
        color_col = "composer_primary_clean"
    else:
        color_col = st.sidebar.selectbox(
            "Color grouping",
            options=["None"] + color_options,
            index=0,
            format_func=lambda x: "None" if x == "None" else get_display_label(x),
        )

    apply_jitter = st.sidebar.checkbox(
        "Jitter points",
        value=True,
        help="Adds small horizontal offsets to reduce overplotting.",
    )

    if apply_jitter:
        jitter_strength = st.sidebar.slider(
            "Jitter strength",
            min_value=0.00,
            max_value=0.30,
            value=0.25,
            step=0.01,
        )
    else:
        jitter_strength = 0.0

    show_trendline = st.sidebar.checkbox(
        "Show fitted line",
        value=False,
    )

    show_scatter = st.sidebar.checkbox(
        "Show scatterplot",
        value=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Album cohesion controls")

    cohesion_metric = st.sidebar.selectbox(
        "Cohesion metric",
        options=[
            "top_track_share",
            "top_track_to_median_ratio",
            "track_metric_std_dev",
        ],
        index=1,
        format_func=get_display_label,
        help=(
            "Rank albums by how concentrated or uneven track performance is "
            "within the soundtrack."
        ),
    )

    top_n_albums = st.sidebar.slider(
        "Top N albums",
        min_value=5,
        max_value=25,
        value=15,
        step=1,
    )

    show_cohesion_table = st.sidebar.checkbox(
        "Show cohesion table",
        value=False,
    )

    show_album_track_table = st.sidebar.checkbox(
        "Show album drilldown table",
        value=False,
    )

    show_summary_table = st.sidebar.checkbox(
        "Show position summary table",
        value=False,
    )

    show_track_table = st.sidebar.checkbox(
        "Show filtered track table",
        value=False,
    )

    return {
        "metric": metric,
        "transform_y": transform_y,
        "analysis_basis": analysis_basis,
        "max_track_position": max_track_position,
        "min_tracks_per_album": min_tracks_per_album,
        "selected_composers": selected_composers,
        "summary_stat": summary_stat,
        "show_ribbon": show_ribbon,
        "color_col": color_col,
        "apply_jitter": apply_jitter,
        "jitter_strength": jitter_strength,
        "show_trendline": show_trendline,
        "show_scatter": show_scatter,
        "show_summary_table": show_summary_table,
        "show_track_table": show_track_table,
        "cohesion_metric": cohesion_metric,
        "top_n_albums": top_n_albums,
        "show_cohesion_table": show_cohesion_table,
        "show_album_track_table": show_album_track_table,
    }

# Page 21 Controls
def get_track_album_relationship_controls(
    composer_options: list[str],
    label_options: list[str],
) -> dict:
    """
    Render sidebar controls for the Track–Album Relationship Explorer.

    Shared global filters (year + film genres + album genres) should be
    collected separately through get_global_filter_controls(...). This
    function only owns Page 21-specific controls plus the optional
    composer/label restrictions that are commonly useful on track pages.

    Args:
        composer_options: Available composer values.
        label_options: Available label values.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Track–Album Controls")

    comparison_metric = st.sidebar.selectbox(
        "Comparison metric",
        options=["listeners", "playcount"],
        index=0,
        format_func=lambda x: "Listeners" if x == "listeners" else "Playcount",
        help=(
            "Compare album-level and track-level Last.fm metrics using either "
            "listeners or playcount."
        ),
    )

    track_aggregation = st.sidebar.selectbox(
        "Track aggregation",
        options=["max", "mean", "median", "top3"],
        index=0,
        format_func=lambda x: {
            "max": "Top track",
            "mean": "Mean track",
            "median": "Median track",
            "top3": "Top 3 sum",
        }[x],
        help="Choose how track performance should be summarized at the album level.",
    )

    dominance_metric = st.sidebar.selectbox(
        "Dominance metric",
        options=["top_to_album", "top_to_total"],
        index=1,
        format_func=lambda x: {
            "top_to_album": "Top track / Album",
            "top_to_total": "Top track / Total track",
        }[x],
        help="Choose how track dominance should be measured.",
    )

    color_mode = st.sidebar.selectbox(
        "Color points by",
        options=[
            "Dominance Bucket",
            "Album Genre",
            "Film Genre",
            "Film Year",
        ],
        index=0,
    )

    use_log_scale = st.sidebar.checkbox(
        "Use log scale",
        value=True,
        help="Apply log scaling to the scatterplot axes where appropriate.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Entity Filters")

    selected_composers = st.sidebar.multiselect(
        "Composers",
        options=composer_options,
        default=[],
    )

    selected_labels = st.sidebar.multiselect(
        "Labels",
        options=label_options,
        default=[],
    )

    st.sidebar.markdown("---")

    show_data_table = st.sidebar.checkbox(
        "Show source data",
        value=False,
    )

    return {
        "comparison_metric": comparison_metric,
        "track_aggregation": track_aggregation,
        "dominance_metric": dominance_metric,
        "color_mode": color_mode,
        "use_log_scale": use_log_scale,
        "selected_composers": selected_composers,
        "selected_labels": selected_labels,
        "show_data_table": show_data_table,
    }

# Page 22 Controls
def get_track_cohesion_controls(
    composer_options: list[str],
    metric_specs: dict[str, dict],
    album_outcome_options: dict[str, str],
    color_options: dict[str, str | None],
) -> dict:
    """
    Render sidebar controls for the Track Cohesion Explorer.

    Args:
        composer_options: Available composer values.
        metric_specs: Metric metadata registry for cohesion metrics.
        album_outcome_options: Mapping of album outcome labels to columns.
        color_options: Mapping of color labels to columns.

    Returns:
        dict: Selected page controls.
    """
    st.sidebar.header("Track Cohesion Controls")

    metric_labels = sorted(
        metric_specs.keys(),
        key=lambda x: metric_specs[x]["default_rank"],
    )

    cohesion_metric_label = st.sidebar.selectbox(
        "Cohesion metric",
        options=metric_labels,
        index=0,
        help=(
            "Choose which within-album cohesion dimension to compare "
            "against popularity and dominance."
        ),
    )

    album_outcome_label = st.sidebar.selectbox(
        "Album outcome",
        options=list(album_outcome_options.keys()),
        index=0,
        help="Choose the album-level success metric for the popularity chart.",
    )

    color_label = st.sidebar.selectbox(
        "Color by",
        options=list(color_options.keys()),
        index=0,
    )

    selected_composers = st.sidebar.multiselect(
        "Composers",
        options=composer_options,
        default=[],
        help="Optionally restrict the page to a selected composer subset.",
    )

    use_log_scale = st.sidebar.checkbox(
        "Log-scale album outcome",
        value=True,
        help="Apply log10 scaling to the album outcome chart.",
    )

    min_tracks = st.sidebar.slider(
        "Minimum tracks per album",
        min_value=1,
        max_value=20,
        value=3,
        step=1,
        help="Hide albums with too few observed tracks for stable cohesion estimates.",
    )

    show_strength_chart = st.sidebar.checkbox(
        "Show metric strength comparison",
        value=True,
    )

    show_binned_view = st.sidebar.checkbox(
        "Show low / medium / high bins",
        value=True,
    )

    show_table = st.sidebar.checkbox(
        "Show source table",
        value=False,
    )

    return {
        "cohesion_metric_label": cohesion_metric_label,
        "cohesion_metric_col": metric_specs[cohesion_metric_label]["col"],
        "cohesion_metric_family": metric_specs[cohesion_metric_label]["family"],
        "cohesion_metric_short_label": metric_specs[cohesion_metric_label]["short_label"],
        "cohesion_metric_description": metric_specs[cohesion_metric_label]["description"],
        "album_outcome_label": album_outcome_label,
        "album_outcome_col": album_outcome_options[album_outcome_label],
        "color_label": color_label,
        "color_col": color_options[color_label],
        "selected_composers": selected_composers,
        "use_log_scale": use_log_scale,
        "min_tracks": min_tracks,
        "show_strength_chart": show_strength_chart,
        "show_binned_view": show_binned_view,
        "show_table": show_table,
    }

# Page 30 controls
def get_track_data_explorer_controls(
    metric_options: list[str],
    group_options: list[str],
    group_value_options_map: dict[str, list[str]],
    composer_options: list[str],
) -> dict:
    """
    Render sidebar controls for the Track Data Explorer.

    Args:
        metric_options: Track-level numeric metrics eligible for the main
            metric panel.
        group_options: Grouping fields eligible for grouped distributions.
        group_value_options_map: Mapping from grouping field to selectable
            group values.
        composer_options: Available composer values.

    Returns:
        dict: Selected control values.
    """
    st.sidebar.header("Track Data Explorer Controls")

    metric = st.sidebar.selectbox(
        "Track metric",
        options=metric_options,
        index=0,
        format_func=get_display_label,
    )

    non_log_safe_metrics = {
        "loudness",
        "relative_track_position",
        "track_number",
        "key",
        "mode",
        "camelot_number",
        "camelot_mode",
    }

    log_is_allowed = metric not in non_log_safe_metrics

    use_log = st.sidebar.checkbox(
        "Log scale (log10)",
        value=(metric in {"lfm_track_listeners", "lfm_track_playcount"} and log_is_allowed),
        disabled=not log_is_allowed,
        help=(
            "Apply log10 to positive-valued metrics."
            if log_is_allowed
            else "Log scale is disabled for this metric because it includes non-positive, bounded, or already-log-like values."
        ),
    )
    if not log_is_allowed:
        st.sidebar.caption(
            f"Log scale is unavailable for {get_display_label(metric)}."
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
                "groups by track count."
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

    st.sidebar.markdown("---")
    st.sidebar.subheader("Track Filters")

    selected_composers = st.sidebar.multiselect(
        "Composers",
        options=composer_options,
        default=[],
    )

    search_text = st.sidebar.text_input(
        "Search track, album, film, composer, or label",
        value="",
        help=(
            "Case-insensitive text search across album title, film title, "
            "composer, and label."
        ),
    )

    max_track_position = st.sidebar.slider(
        "Maximum track position",
        min_value=5,
        max_value=40,
        value=20,
        step=1,
        help="Restrict analysis to earlier track positions if desired.",
    )

    min_album_listeners = st.sidebar.number_input(
        "Minimum album listeners",
        min_value=0,
        value=0,
        step=100,
        help="Exclude tracks from albums below this listener threshold.",
    )

    audio_only = st.sidebar.checkbox(
        "Only tracks with any audio features",
        value=False,
    )

    bins = st.sidebar.slider(
        "Histogram bins",
        min_value=20,
        max_value=100,
        value=40,
        step=5,
    )

    show_data_table = st.sidebar.checkbox(
        "Show source data table",
        value=True,
    )

    return {
        "metric": metric,
        "use_log": use_log,
        "group_var": group_var,
        "selected_groups": selected_groups,
        "top_n": top_n,
        "selected_composers": selected_composers,
        "max_track_position": max_track_position,
        "min_album_listeners": min_album_listeners,
        "audio_only": audio_only,
        "bins": bins,
        "show_data_table": show_data_table,
        "search_text": search_text,
    }