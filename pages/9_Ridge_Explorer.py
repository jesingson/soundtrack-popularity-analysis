import streamlit as st

import ridge_analysis as ridge
import ridge_visualization as ridge_viz
from app.app_controls import get_ridge_controls
from app.app_data import load_analysis_data, load_ridge_dynamic_groups
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
        preset_name: [col for col in cols if col in available_cols]
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
        This view uses the same validated ridge-density pipeline as the notebook,
        including precomputed density curves and median-gap ordering.
        """
    )

    album_analytics_df = load_analysis_data()

    initial_top_n = 8
    preset_registry = build_ridge_presets(
        album_analytics_df=album_analytics_df,
        top_n=initial_top_n,
    )

    all_available_features = sorted(
        {
            feature
            for cols in ridge.DEFAULT_RIDGE_GROUPS.values()
            for feature in cols
            if feature in album_analytics_df.columns
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
        album_analytics_df=album_analytics_df,
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
        album_analytics_df=album_analytics_df,
        selected_features=tuple(selected_features),
        bins=controls["bins"],
        smooth_window=controls["smooth_window"],
        min_group_n=controls["min_group_n"],
    )

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

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Features shown", n_ridges)
    with metric_col2:
        st.metric("Density bins", controls["bins"])
    with metric_col3:
        st.metric("Minimum group size", controls["min_group_n"])

    st.altair_chart(chart, width="stretch")

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