import streamlit as st

import ridge_analysis as ridge
import ridge_visualization as ridge_viz
from app.app_controls import get_global_filter_controls, get_ridge_controls
from app.app_data import load_analysis_data, load_explorer_data, load_ridge_dynamic_groups
from app.data_filters import filter_dataset
from app.explorer_shared import get_global_filter_inputs
from app.ui import apply_app_styles, rename_columns_for_display


PRESET_LABELS = {
    "core_static": "Core Static (Notebook Set)",
    "top_pearson": "Top Pearson Features",
    "top_spearman": "Top Spearman Features",
    "top_regression": "Top Regression Features",
    "exposure_and_timing": "Exposure and Timing",
    "structure_and_quality": "Structure and Quality",
    "awards_and_creator": "Awards and Creator",
    "film_genres": "Film Genres",
    "album_genres": "Album Genres",
    "custom": "Custom Selection",
}

EXCLUDED_RIDGE_FEATURES = {
    "album_cohesion_has_audio_data",
    "audio_feature_count",
    "has_any_audio_features",
}

@st.cache_data
def build_ridge_presets(
    album_analytics_df,
    top_n: int,
) -> dict[str, list[str]]:
    """
    Build the full ridge preset registry for the page.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        top_n: Number of features to retain in dynamic top-N presets.

    Returns:
        dict[str, list[str]]: Preset name to feature-list mapping.
    """
    dynamic_groups = load_ridge_dynamic_groups(top_n=top_n)

    preset_registry = {
        "core_static": ridge.DEFAULT_RIDGE_GROUPS["core_static"],
        "top_pearson": dynamic_groups["top_pearson"],
        "top_spearman": dynamic_groups["top_spearman"],
        "top_regression": dynamic_groups["top_regression"],
        "exposure_and_timing": ridge.DEFAULT_RIDGE_GROUPS["exposure_and_timing"],
        "structure_and_quality": ridge.DEFAULT_RIDGE_GROUPS["structure_and_quality"],
        "awards_and_creator": ridge.DEFAULT_RIDGE_GROUPS["awards_and_creator"],
        "film_genres": ridge.DEFAULT_RIDGE_GROUPS["film_genres"],
        "album_genres": ridge.DEFAULT_RIDGE_GROUPS["album_genres"],
    }

    available_cols = set(album_analytics_df.columns)
    cleaned_registry = {
        preset_name: [
            col for col in cols
            if col in available_cols and col not in EXCLUDED_RIDGE_FEATURES
        ]
        for preset_name, cols in preset_registry.items()
    }

    return cleaned_registry


@st.cache_data
def build_ridge_outputs_for_features(
    album_analytics_df,
    selected_features: tuple[str, ...],
    bins: int,
    smooth_window: int,
    min_group_n: int,
):
    """
    Build Phase 1 and Phase 2 ridge outputs for a selected feature set.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        selected_features: Feature tuple selected for the ridge chart.
        bins: Number of bins for density estimation.
        smooth_window: Odd smoothing window for density estimation.
        min_group_n: Minimum rows required for a feature-group density.

    Returns:
        tuple[dict, dict]: Phase 1 ridge outputs and Phase 2 ridge outputs.
    """
    feature_groups = {"selected": list(selected_features)}

    ridge_config = ridge.get_ridge_feature_config(
        albums_df=album_analytics_df,
        y_col="log_lfm_album_listeners",
        feature_groups=feature_groups,
        feature_labels=ridge.DEFAULT_FEATURE_LABELS,
    )

    viz_df = ridge.build_ridge_viz_df(
        albums_df=album_analytics_df,
        ridge_config=ridge_config,
    )
    viz_df = ridge.add_ridge_group_columns(
        viz_df=viz_df,
        ridge_config=ridge_config,
    )
    ridge_long = ridge.build_ridge_long_df(
        viz_df=viz_df,
        y_col=ridge_config["y_col"],
    )

    ridge_outputs = {
        "ridge_config": ridge_config,
        "viz_df": viz_df,
        "ridge_long": ridge_long,
        "dynamic_groups": {},
    }

    phase2_outputs = ridge.build_ridge_phase2_outputs(
        ridge_outputs=ridge_outputs,
        bins=bins,
        smooth_window=smooth_window,
        min_group_n=min_group_n,
        order_method="median_gap",
        row_gap=2.5,
        density_scale=11.0,
        label_y_offset=0.0,
    )

    return ridge_outputs, phase2_outputs

def build_ridge_scope_caption(
    controls: dict,
    selected_features: list[str],
    n_ridges: int,
    n_albums: int,
) -> str:
    """
    Build a short caption describing the current ridge scope.

    Args:
        controls: Sidebar control selections.
        selected_features: Selected ridge features.
        n_ridges: Number of rendered ridge rows.

    Returns:
        str: Human-readable scope caption.
    """
    preset_label = PRESET_LABELS.get(
        controls["preset_key"],
        controls["preset_key"],
    )

    mode_phrase = (
        "custom feature selection"
        if controls["preset_key"] == "custom"
        else f"preset '{preset_label}'"
    )

    return (
        f"Current scope: {mode_phrase}, {n_albums:,} visible albums, showing "
        f"{n_ridges} binary feature splits with {controls['bins']} density bins, "
        f"smoothing window {controls['smooth_window']}, and minimum group size "
        f"{controls['min_group_n']}."
    )


def build_ridge_insight_summary(
    order_df,
) -> dict[str, str]:
    """
    Build a compact narrative summary from the ridge ordering dataframe.

    Args:
        order_df: Ridge ordering dataframe indexed by feature label, with
            columns 'No', 'Yes', and 'median_gap'.

    Returns:
        dict[str, str]: Titles, values, and captions for three insight cards.
    """
    if order_df.empty:
        return {
            "card1_title": "Largest Median Gap",
            "card1_value": "None",
            "card1_caption": "No ridge features are available under the current settings.",
            "card2_title": "Direction of Separation",
            "card2_value": "None",
            "card2_caption": "No directional split is available.",
            "card3_title": "Overall Pattern",
            "card3_value": "None",
            "card3_caption": "No overall ridge pattern is available.",
        }

    top_feature_label = str(order_df.index[0])
    top_row = order_df.iloc[0]

    top_gap = float(top_row["median_gap"])
    top_no = float(top_row["No"])
    top_yes = float(top_row["Yes"])

    if top_yes > top_no:
        direction_value = "Yes above No"
        direction_caption = (
            f"For the top split, '{top_feature_label}', the Yes group median "
            f"({top_yes:.3f}) is above the No group median ({top_no:.3f})."
        )
    elif top_yes < top_no:
        direction_value = "No above Yes"
        direction_caption = (
            f"For the top split, '{top_feature_label}', the No group median "
            f"({top_no:.3f}) is above the Yes group median ({top_yes:.3f})."
        )
    else:
        direction_value = "Tie"
        direction_caption = (
            f"For the top split, '{top_feature_label}', the Yes and No group "
            f"medians are equal at {top_yes:.3f}."
        )

    mean_gap = float(order_df["median_gap"].mean())
    median_gap = float(order_df["median_gap"].median())

    return {
        "card1_title": "Largest Median Gap",
        "card1_value": top_feature_label,
        "card1_caption": (
            f"Top-ranked split with a median gap of {top_gap:.3f}."
        ),
        "card2_title": "Direction of Separation",
        "card2_value": direction_value,
        "card2_caption": direction_caption,
        "card3_title": "Overall Pattern",
        "card3_value": f"Avg gap {mean_gap:.3f}",
        "card3_caption": (
            f"Across visible features, the median gap is {median_gap:.3f} at the median, "
            "so the page is showing moderate separation concentrated near the top rows."
        ),
    }


def render_ridge_insight_cards(order_df) -> None:
    """
    Render the three narrative insight cards for the ridge page.

    Args:
        order_df: Ridge ordering dataframe.
    """
    insights = build_ridge_insight_summary(order_df)

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


def build_ridge_supporting_insight(order_df) -> str:
    """
    Build a short supporting sentence for the main ridge chart.

    Args:
        order_df: Ridge ordering dataframe.

    Returns:
        str: Short chart-supporting insight.
    """
    if order_df.empty:
        return "No ridge ordering insight is available."

    top_feature_label = str(order_df.index[0])
    top_gap = float(order_df.iloc[0]["median_gap"])

    bottom_feature_label = str(order_df.index[-1])
    bottom_gap = float(order_df.iloc[-1]["median_gap"])

    return (
        f"💡 Rows are ordered by median separation, so '{top_feature_label}' "
        f"shows the strongest visible split ({top_gap:.3f}), while "
        f"'{bottom_feature_label}' is the weakest visible split "
        f"({bottom_gap:.3f}) under the current settings."
    )


def build_selected_feature_caption(
    controls: dict,
    selected_features: list[str],
) -> str:
    """
    Build a compact caption listing the selected raw feature columns.

    Args:
        controls: Sidebar control selections.
        selected_features: Selected feature column names.

    Returns:
        str: Human-readable feature list caption.
    """
    if not selected_features:
        return "No selected features."

    if controls["preset_key"] == "custom":
        prefix = "Selected custom features"
    else:
        prefix = "Selected preset features"

    return f"{prefix}: {', '.join(selected_features)}."

def attach_ridge_filter_metadata(
    album_analytics_df,
    album_explorer_df,
):
    """
    Merge explorer-style global filter metadata onto the analysis dataframe.

    This lets the ridge page use shared global filters (film year, film genres,
    album genres) without changing the underlying analysis dataset used by the
    ridge pipeline.

    Args:
        album_analytics_df: Narrow analysis-ready album dataframe.
        album_explorer_df: Rich explorer dataframe containing filter metadata.

    Returns:
        pd.DataFrame: Analysis dataframe with filter metadata columns attached.
    """
    merge_keys = [
        col for col in ["release_group_mbid", "tmdb_id"]
        if col in album_analytics_df.columns and col in album_explorer_df.columns
    ]

    metadata_cols = [
        col for col in [
            "release_group_mbid",
            "tmdb_id",
            "film_year",
            "film_genres",
            "album_genres_display",
        ]
        if col in album_explorer_df.columns
    ]

    if not merge_keys or len(metadata_cols) <= len(merge_keys):
        return album_analytics_df.copy()

    metadata_df = (
        album_explorer_df[metadata_cols]
        .drop_duplicates(subset=merge_keys)
        .copy()
    )

    merged_df = album_analytics_df.merge(
        metadata_df,
        on=merge_keys,
        how="left",
        validate="1:1",
    )

    return merged_df

def main() -> None:
    """
    Run the Ridge Explorer Streamlit page.
    """
    st.set_page_config(
        page_title="Ridge Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Ridge Explorer")
    st.write(
        """
        Compare soundtrack listener distributions across binary feature splits.
        This page is designed to show which feature-based Yes/No groupings most
        clearly separate listener outcomes, while preserving the validated
        ridge-density pipeline used in the notebook workflow.
        """
    )

    album_analytics_df = load_analysis_data()
    album_explorer_df = load_explorer_data()

    ridge_filter_df = attach_ridge_filter_metadata(
        album_analytics_df=album_analytics_df,
        album_explorer_df=album_explorer_df,
    )

    filter_inputs = get_global_filter_inputs(ridge_filter_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    filtered_album_df = filter_dataset(ridge_filter_df, global_controls).copy()

    if filtered_album_df.empty:
        st.warning("No albums remain under the current global filters.")
        st.stop()

    initial_top_n = 8
    preset_registry = build_ridge_presets(
        album_analytics_df=filtered_album_df,
        top_n=initial_top_n,
    )

    all_available_features = sorted(
        {
            feature
            for cols in ridge.DEFAULT_RIDGE_GROUPS.values()
            for feature in cols
            if feature in filtered_album_df.columns
        }
    )

    preset_keys = list(PRESET_LABELS.keys())

    controls = get_ridge_controls(
        preset_labels=PRESET_LABELS,
        preset_keys=preset_keys,
        all_available_features=all_available_features,
        default_custom_features=ridge.DEFAULT_RIDGE_GROUPS["core_static"],
    )

    preset_registry = build_ridge_presets(
        album_analytics_df=filtered_album_df,
        top_n=controls["top_n"],
    )

    if controls["preset_key"] != "custom":
        selected_features = preset_registry.get(controls["preset_key"], [])
    else:
        selected_features = controls["selected_features"]

    if not selected_features:
        st.warning("Select at least one feature to render the ridge plot.")
        st.stop()

    ridge_outputs, phase2_outputs = build_ridge_outputs_for_features(
        album_analytics_df=filtered_album_df,
        selected_features=tuple(selected_features),
        bins=controls["bins"],
        smooth_window=controls["smooth_window"],
        min_group_n=controls["min_group_n"],
    )

    order_df = phase2_outputs["order_df"]

    n_ridges = len(phase2_outputs["labels_df"])
    chart_height = max(500, n_ridges * 32)

    subtitle_lines = [
        f"Preset: {PRESET_LABELS[controls['preset_key']]} | Features shown: {n_ridges}",
        "Rows are ordered by absolute median listener gap between Yes and No groups.",
    ]

    chart = ridge_viz.create_ridge_chart(
        ridge_chart_df=phase2_outputs["ridge_chart_df"],
        labels_df=phase2_outputs["labels_df"],
        title_text="Listener distributions by feature group",
        subtitle_lines=subtitle_lines,
        width=780,
        height=chart_height,
        left_padding=300,
    )

    st.caption(
        build_ridge_scope_caption(
            controls=controls,
            selected_features=selected_features,
            n_ridges=n_ridges,
            n_albums=len(filtered_album_df),
        )
    )

    render_ridge_insight_cards(order_df)

    st.caption(
        build_selected_feature_caption(
            controls=controls,
            selected_features=selected_features,
        )
    )

    st.altair_chart(chart, width="stretch")

    st.caption(
        build_ridge_supporting_insight(order_df)
    )

    if controls["show_order_table"]:
        st.subheader("Feature Ordering Table")
        st.dataframe(
            rename_columns_for_display(phase2_outputs["order_df"].round(3)),
            width="stretch",
        )

    if controls["show_density_table"]:
        st.subheader("Density Dataframe Sample")
        st.dataframe(
            rename_columns_for_display(
                phase2_outputs["ridge_density_df"].head(50)
            ),
            width="stretch",
        )

    if controls["show_ridge_long_sample"]:
        st.subheader("ridge_long Sample")
        st.dataframe(
            rename_columns_for_display(ridge_outputs["ridge_long"].head(50)),
            width="stretch",
        )


if __name__ == "__main__":
    main()