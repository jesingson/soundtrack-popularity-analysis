from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import regression_analysis as reg

# /***************************************
#  *             Phase 1                 *
#  * Feature grouping, cleaning, binary  *
#  * group assignment and longform ridge *
#  * dataframe creation                  *
#  ***************************************/

DEFAULT_RIDGE_GROUPS = {
    "core_static": [
        "film_vote_count",
        "film_popularity",
        "days_since_album_release",
        "n_tracks",
        "film_runtime_min",
        "composer_album_count",
        "film_rating",
        "album_cohesion_score",
        "us_score_nominee_count",
        "us_song_nominee_count",
        "bafta_nominee",
    ],
    "exposure_and_timing": [
        "film_vote_count",
        "film_popularity",
        "film_budget",
        "film_revenue",
        "days_since_film_release",
        "days_since_album_release",
    ],
    "structure_and_quality": [
        "n_tracks",
        "film_runtime_min",
        "composer_album_count",
        "film_rating",
        "album_cohesion_score",
    ],
    "awards_and_creator": [
        "composer_album_count",
        "us_score_nominee_count",
        "us_song_nominee_count",
        "bafta_nominee",
    ],
    "film_genres": [
        "film_is_action",
        "film_is_adventure",
        "film_is_animation",
        "film_is_comedy",
        "film_is_crime",
        "film_is_documentary",
        "film_is_drama",
        "film_is_family",
        "film_is_fantasy",
        "film_is_history",
        "film_is_horror",
        "film_is_music",
        "film_is_mystery",
        "film_is_romance",
        "film_is_science_fiction",
        "film_is_tv_movie",
        "film_is_thriller",
        "film_is_war",
        "film_is_western",
    ],
    "album_genres": [
        "ambient_experimental",
        "classical_orchestral",
        "electronic",
        "hip_hop_rnb",
        "pop",
        "rock",
        "world_folk",
    ],
}

DEFAULT_FEATURE_LABELS = {
    "film_vote_count": "Film exposure above median",
    "film_popularity": "Film popularity above median",
    "days_since_album_release": "Album released longer ago than median",
    "composer_album_count": "Composer appears on many albums (above median)",
    "film_rating": "Film rating above median",
    "n_tracks": "Album has many tracks (above median)",
    "film_runtime_min": "Film runtime above median",
    "us_score_nominee_count": "Any score nominations",
    "us_song_nominee_count": "Any song nominations",
    "bafta_nominee": "BAFTA nominee",
    "album_cohesion_score": "Album cohesion above median",
    "album_cohesion_has_audio_data": "Album has enough audio data for cohesion",
}

DEFAULT_FEATURE_LABELS.update({
    "film_budget": "Film budget above median",
    "film_revenue": "Film revenue above median",
    "days_since_film_release": "Film released longer ago than median",

    "film_is_action": "Action film",
    "film_is_adventure": "Adventure film",
    "film_is_animation": "Animated film",
    "film_is_comedy": "Comedy film",
    "film_is_crime": "Crime film",
    "film_is_documentary": "Documentary film",
    "film_is_drama": "Drama film",
    "film_is_family": "Family film",
    "film_is_fantasy": "Fantasy film",
    "film_is_history": "History film",
    "film_is_horror": "Horror film",
    "film_is_music": "Musical film",
    "film_is_mystery": "Mystery film",
    "film_is_romance": "Romance film",
    "film_is_science_fiction": "Science fiction film",
    "film_is_tv_movie": "TV movie",
    "film_is_thriller": "Thriller film",
    "film_is_war": "War film",
    "film_is_western": "Western film",

    "ambient_experimental": "Ambient / experimental soundtrack",
    "classical_orchestral": "Classical / orchestral soundtrack",
    "electronic": "Electronic soundtrack",
    "hip_hop_rnb": "Hip-hop / R&B soundtrack",
    "pop": "Pop soundtrack",
    "rock": "Rock soundtrack",
    "world_folk": "World / folk soundtrack",
})


def get_ridge_feature_config(
    albums_df: pd.DataFrame,
    y_col: str = "log_lfm_album_listeners",
    feature_groups: dict[str, list[str]] | None = None,
    feature_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Build the ridge-plot feature configuration from available album columns.

    This helper starts from the configured ridge feature groups, keeps only
    columns that actually exist in the provided dataframe, and classifies the
    surviving features into three downstream handling types:

    - continuous-like features, which will later be split into above/below
      median groups
    - award count features, which will later be collapsed into none vs any
      recognition
    - binary flag features, such as genre indicators and BAFTA nominee, which
      will later be converted into True/False style ridge groups

    The function also returns the ordered feature list and a filtered label map
    so downstream ridge helpers can stay fully data-driven.

    Args:
        albums_df: Source album-level dataframe.
        y_col: Outcome column used throughout the ridge workflow.
        feature_groups: Optional override for grouped ridge feature definitions.
            If omitted, the default static + curated ridge groups are used.
        feature_labels: Optional override for human-readable feature labels.
            If omitted, the default label map is used.

    Returns:
        dict[str, Any]: Configuration dictionary containing:
            - "y_col": outcome column name
            - "feature_groups_available": ridge groups after removing features
              not present in the dataframe
            - "all_features": de-duplicated ordered list of all selected ridge
              features
            - "keep_cols": columns required for ridge preparation
            - "feature_labels": available human-readable labels keyed by feature
            - "cont_like_features": features that should receive median splits
            - "award_count_features": award count fields handled as none vs any
            - "binary_flag_features": binary indicator fields handled as
              True/False groups

    Raises:
        ValueError: If the outcome column is not present in the dataframe.
    """
    if y_col not in albums_df.columns:
        raise ValueError(
            f"Outcome column '{y_col}' was not found in the albums dataframe."
        )

    if feature_groups is None:
        feature_groups = DEFAULT_RIDGE_GROUPS

    if feature_labels is None:
        feature_labels = DEFAULT_FEATURE_LABELS

    feature_groups_available: dict[str, list[str]] = {}
    for group_name, cols in feature_groups.items():
        available = [col for col in cols if col in albums_df.columns]
        if available:
            feature_groups_available[group_name] = available

    all_features = []
    for cols in feature_groups_available.values():
        for col in cols:
            if col not in all_features:
                all_features.append(col)

    keep_cols = [y_col] + all_features

    award_count_features = [
        col for col in all_features if col in {
            "us_score_nominee_count",
            "us_song_nominee_count",
        }
    ]

    binary_flag_features = [
        col for col in all_features
        if (
            col.startswith("film_is_")
            or col in {
                "bafta_nominee",
                "ambient_experimental",
                "classical_orchestral",
                "electronic",
                "hip_hop_rnb",
                "pop",
                "rock",
                "world_folk",
            }
        )
    ]

    cont_like_features = [
        col for col in all_features
        if col not in award_count_features and col not in binary_flag_features
    ]

    available_feature_labels = {
        col: feature_labels.get(col, col)
        for col in all_features
    }

    return {
        "y_col": y_col,
        "feature_groups_available": feature_groups_available,
        "all_features": all_features,
        "keep_cols": keep_cols,
        "feature_labels": available_feature_labels,
        "cont_like_features": cont_like_features,
        "award_count_features": award_count_features,
        "binary_flag_features": binary_flag_features,
    }


def build_ridge_viz_df(
    albums_df: pd.DataFrame,
    ridge_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build the cleaned working dataframe used for ridge-plot preparation.

    This function performs the notebook-style preprocessing needed before any
    ridge grouping or reshaping occurs. It:

    - selects the outcome and all configured ridge features
    - coerces the outcome to numeric and drops rows without valid outcome data
    - coerces continuous-like features to numeric
    - coerces award count fields to numeric and fills missing values with zero
    - standardizes binary flag features into stable 0/1 integer values

    The returned dataframe remains at the album level and is intended to feed
    the later group-column creation step.

    Args:
        albums_df: Source album dataframe.
        ridge_config: Configuration dictionary returned by
            ``get_ridge_feature_config()``.

    Returns:
        pd.DataFrame: Cleaned album-level ridge working dataframe.
    """
    y_col = ridge_config["y_col"]
    keep_cols = ridge_config["keep_cols"]
    cont_like_features = ridge_config["cont_like_features"]
    award_count_features = ridge_config["award_count_features"]
    binary_flag_features = ridge_config["binary_flag_features"]

    viz_df = albums_df[keep_cols].copy()

    viz_df[y_col] = pd.to_numeric(viz_df[y_col], errors="coerce")
    viz_df = viz_df.dropna(subset=[y_col]).copy()

    for col in cont_like_features:
        if col in viz_df.columns:
            viz_df[col] = pd.to_numeric(viz_df[col], errors="coerce")

    for col in award_count_features:
        if col in viz_df.columns:
            viz_df[col] = pd.to_numeric(viz_df[col], errors="coerce").fillna(0)

    for col in binary_flag_features:
        if col in viz_df.columns:
            viz_df[col] = (
                viz_df[col]
                .fillna(False)
                .infer_objects(copy=False)
                .astype(bool)
                .astype(int)
            )

    return viz_df


def add_ridge_group_columns(
    viz_df: pd.DataFrame,
    ridge_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Add binary comparison columns used in the ridge-plot workflow.

    This helper converts each selected ridge feature into a two-group label
    suitable for comparing outcome distributions in a ridgeline chart:

    - continuous-like features become "Above median" vs "Below median"
    - award count features become "1+ (recognized)" vs "0 (none)"
    - binary flag features become "True" vs "False"

    The result remains an album-level dataframe but now includes one `*_group`
    column per selected ridge feature, making it ready for long-format stacking.

    Args:
        viz_df: Cleaned ridge working dataframe.
        ridge_config: Configuration dictionary returned by
            ``get_ridge_feature_config()``.

    Returns:
        pd.DataFrame: Ridge working dataframe with `*_group` columns added.
    """
    viz_df = viz_df.copy()

    cont_like_features = ridge_config["cont_like_features"]
    award_count_features = ridge_config["award_count_features"]
    binary_flag_features = ridge_config["binary_flag_features"]

    for col in cont_like_features:
        if col in viz_df.columns:
            median_val = viz_df[col].median(skipna=True)
            viz_df[f"{col}_group"] = np.where(
                viz_df[col] >= median_val,
                "Above median",
                "Below median",
            )

    for col in award_count_features:
        if col in viz_df.columns:
            viz_df[f"{col}_group"] = np.where(
                viz_df[col] >= 1,
                "1+ (recognized)",
                "0 (none)",
            )

    for col in binary_flag_features:
        if col in viz_df.columns:
            viz_df[f"{col}_group"] = np.where(
                viz_df[col] == 1,
                "True",
                "False",
            )

    return viz_df

def build_ridge_long_df(
    viz_df: pd.DataFrame,
    y_col: str = "log_lfm_album_listeners",
) -> pd.DataFrame:
    """
    Convert the grouped ridge dataframe into long format.

    Args:
        viz_df: Ridge dataframe containing outcome + *_group columns.
        y_col: Outcome column.

    Returns:
        pd.DataFrame: Long-form dataframe with columns:
            - y_col
            - feature
            - group
    """
    if y_col not in viz_df.columns:
        raise ValueError(
            f"Outcome column '{y_col}' was not found in viz_df."
        )

    group_cols = [col for col in viz_df.columns if col.endswith("_group")]
    if not group_cols:
        raise ValueError(
            "No *_group columns were found. Run add_ridge_group_columns() first."
        )

    ridge_long = (
        viz_df[[y_col] + group_cols]
        .melt(
            id_vars=[y_col],
            value_vars=group_cols,
            var_name="feature",
            value_name="group",
        )
        .dropna(subset=[y_col, "group"])
        .copy()
    )

    ridge_long["feature"] = ridge_long["feature"].str.replace(
        "_group$",
        "",
        regex=True,
    )

    ridge_long["group"] = pd.Categorical(
        ridge_long["group"],
        categories=[
            "Below median",
            "Above median",
            "0 (none)",
            "1+ (recognized)",
            "False",
            "True",
        ],
        ordered=True,
    )

    return ridge_long


def normalize_ridge_group_label(group_value: Any) -> str:
    """
    Normalize mixed binary encodings into a stable Yes/No label.

    Args:
        group_value: Raw group label.

    Returns:
        str: 'Yes' if the condition is met, otherwise 'No'.
    """
    s = str(group_value).lower()
    if ("above" in s) or ("1+" in s) or (s == "true"):
        return "Yes"
    return "No"

def build_dynamic_ridge_groups(
    albums_df: pd.DataFrame,
    y_col: str = "log_lfm_album_listeners",
    top_n: int = 8,
) -> dict[str, list[str]]:
    """
    Build data-driven ridge feature groups for interactive exploration.

    This helper derives dynamic ridge presets directly from the current
    analysis-ready dataframe rather than relying on hardcoded notebook choices.
    It creates three ranked feature groups:

    - ``top_pearson``: strongest continuous features by absolute Pearson
      correlation with the target
    - ``top_spearman``: strongest continuous features by absolute Spearman
      correlation with the target
    - ``top_regression``: strongest modeled predictors by absolute regression
      coefficient magnitude from the fitted OLS workflow

    These groups are intended to complement, not replace, the curated static
    ridge groups such as ``core_static``.

    Args:
        albums_df: Analysis-ready album dataframe.
        y_col: Outcome column used to rank features.
        top_n: Number of features to retain in each dynamic group.

    Returns:
        dict[str, list[str]]: Dictionary containing dynamic ridge group names
        mapped to ordered feature lists.
    """
    pearson_rank = reg.build_scatterplot_feature_ranking(
        album_analytics_df=albums_df,
        target_col=y_col,
        method="pearson",
    )

    spearman_rank = reg.build_scatterplot_feature_ranking(
        album_analytics_df=albums_df,
        target_col=y_col,
        method="spearman",
    )

    regression_results = reg.run_regression_pipeline(
        album_analytics_df=albums_df,
        target_col=y_col,
        threshold=0.05,
    )
    coef_df = reg.build_coefficient_plot_df(
        regression_results["ols_results"]["results"]
    )

    return {
        "top_pearson": pearson_rank["feature"].head(top_n).tolist(),
        "top_spearman": spearman_rank["feature"].head(top_n).tolist(),
        "top_regression": coef_df["feature"].head(top_n).tolist(),
    }


def build_ridge_prep_outputs(
    albums_df: pd.DataFrame,
    y_col: str = "log_lfm_album_listeners",
    feature_groups: dict[str, list[str]] | None = None,
    feature_labels: dict[str, str] | None = None,
    top_n: int = 8,
) -> dict[str, Any]:
    """
    Run the full first-stage ridge preparation workflow.

    This convenience wrapper combines the static ridge groups with
    data-driven dynamic groups, builds the ridge configuration, prepares the
    cleaned working dataframe, adds binary group columns, and reshapes the
    result into the long format needed for downstream density estimation.

    It is intended as the main entry point for validating ridge inputs before
    moving on to density precomputation, feature ordering, and chart layout.

    Args:
        albums_df: Analysis-ready album dataframe.
        y_col: Outcome column used throughout the ridge workflow.
        feature_groups: Optional override for static ridge group definitions.
            If omitted, the default grouped presets are used.
        feature_labels: Optional override for feature-label mappings.
        top_n: Number of features to retain in each dynamic ranking-based
            ridge group.

    Returns:
        dict[str, Any]: Dictionary containing:
            - "ridge_config": resolved ridge configuration
            - "viz_df": cleaned album-level ridge working dataframe
            - "ridge_long": long-form ridge dataframe
            - "dynamic_groups": computed Pearson/Spearman/regression presets
    """
    if feature_groups is None:
        feature_groups = DEFAULT_RIDGE_GROUPS.copy()
    else:
        feature_groups = feature_groups.copy()

    dynamic_groups = build_dynamic_ridge_groups(
        albums_df=albums_df,
        y_col=y_col,
        top_n=top_n,
    )
    feature_groups.update(dynamic_groups)

    ridge_config = get_ridge_feature_config(
        albums_df=albums_df,
        y_col=y_col,
        feature_groups=feature_groups,
        feature_labels=feature_labels,
    )
    viz_df = build_ridge_viz_df(albums_df, ridge_config)
    viz_df = add_ridge_group_columns(viz_df, ridge_config)
    ridge_long = build_ridge_long_df(viz_df, y_col=y_col)

    return {
        "ridge_config": ridge_config,
        "viz_df": viz_df,
        "ridge_long": ridge_long,
        "dynamic_groups": dynamic_groups,
    }


# /***************************************
#  *             Phase 2                 *
#  * Density computations, ordering and  *
#  * Chart-ready dataframes
#  ***************************************/

def build_ridge_density_df(
    ridge_long: pd.DataFrame,
    y_col: str = "log_lfm_album_listeners",
    bins: int = 80,
    smooth_window: int = 11,
    min_group_n: int = 10,
) -> pd.DataFrame:
    """
    Precompute ridge density curves on a shared x-grid.

    This helper converts the long-form ridge input dataframe into a
    plot-ready density table by estimating a smoothed histogram-based
    density for each (feature, group) pair.

    A shared x-grid is used across all features and groups so that the
    resulting ridge curves remain directly comparable. Small groups are
    skipped to avoid unstable or misleading densities.

    Args:
        ridge_long: Long-form ridge dataframe containing the outcome,
            feature, and group columns.
        y_col: Outcome column whose distribution is plotted in each ridge.
        bins: Number of x bins used for the shared density grid.
        smooth_window: Odd-valued moving-average window used to smooth
            the histogram-based densities.
        min_group_n: Minimum number of observations required to retain a
            feature-group pair in the density output.

    Returns:
        pd.DataFrame: Plot-ready density dataframe with columns:
            - feature
            - group
            - x
            - density
            - n_obs

    Raises:
        ValueError: If required columns are missing or smoothing settings
            are invalid.
    """
    required_cols = {y_col, "feature", "group"}
    missing_cols = [col for col in required_cols if col not in ridge_long.columns]
    if missing_cols:
        raise ValueError(
            f"ridge_long is missing required columns: {missing_cols}"
        )

    if bins < 5:
        raise ValueError("bins must be at least 5 for density estimation.")

    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError(
            "smooth_window must be a positive odd integer, such as 5, 7, or 11."
        )

    df = ridge_long.dropna(subset=[y_col, "feature", "group"]).copy()

    if df.empty:
        raise ValueError(
            "No valid rows remain after dropping missing values from ridge_long."
        )

    x_min = float(df[y_col].min())
    x_max = float(df[y_col].max())

    bin_edges = np.linspace(x_min, x_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    kernel = np.ones(smooth_window) / smooth_window

    density_rows = []

    for (feature, group), g in df.groupby(["feature", "group"], observed=True):
        x = g[y_col].to_numpy()

        if len(x) < min_group_n:
            continue

        dens, _ = np.histogram(x, bins=bin_edges, density=True)
        dens_smooth = np.convolve(dens, kernel, mode="same")

        density_rows.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "group": group,
                    "x": bin_centers,
                    "density": dens_smooth,
                    "n_obs": len(x),
                }
            )
        )

    if not density_rows:
        raise ValueError(
            "No feature-group pairs met the minimum sample size required "
            "for density estimation."
        )

    ridge_density_df = pd.concat(density_rows, ignore_index=True)
    return ridge_density_df

def compute_ridge_feature_order(
    viz_df: pd.DataFrame,
    feature_labels: dict[str, str],
    y_col: str = "log_lfm_album_listeners",
    order_method: str = "median_gap",
) -> pd.DataFrame:
    """
    Compute a principled feature ordering for ridge-plot stacking.

    The default ordering ranks features by the absolute difference in
    median outcome between the condition-met group and the condition-not-met
    group. This highlights the features with the clearest separation in
    soundtrack listener distributions.

    Args:
        viz_df: Album-level ridge working dataframe containing the outcome
            and feature-specific `*_group` columns.
        feature_labels: Mapping from raw feature names to display labels.
        y_col: Outcome column used in the ridge plot.
        order_method: Ordering strategy to apply. Currently supported:
            - "median_gap": absolute difference between median Yes and No
              outcome values

    Returns:
        pd.DataFrame: Ordering summary dataframe with one row per feature
        label and columns such as:
            - No
            - Yes
            - median_gap

    Raises:
        ValueError: If required columns are missing or the requested order
            method is unsupported.
    """
    if y_col not in viz_df.columns:
        raise ValueError(f"Outcome column '{y_col}' was not found in viz_df.")

    if order_method != "median_gap":
        raise ValueError(
            f"Unsupported order_method '{order_method}'. "
            f"Currently supported: ['median_gap']"
        )

    group_cols = [
        f"{feature}_group"
        for feature in feature_labels.keys()
        if f"{feature}_group" in viz_df.columns
    ]

    if not group_cols:
        raise ValueError(
            "No ridge group columns were found for the supplied feature labels."
        )

    order_long = (
        viz_df[[y_col] + group_cols]
        .melt(
            id_vars=y_col,
            var_name="feature_group_col",
            value_name="group_raw",
        )
        .assign(
            feature=lambda d: d["feature_group_col"].str.replace(
                "_group$",
                "",
                regex=True,
            ),
            feature_label=lambda d: d["feature"].map(feature_labels),
            group_std=lambda d: d["group_raw"].apply(normalize_ridge_group_label),
        )
    )

    order_df = (
        order_long
        .groupby(["feature_label", "group_std"], observed=True)[y_col]
        .median()
        .unstack("group_std")
    )

    if "Yes" not in order_df.columns:
        order_df["Yes"] = np.nan
    if "No" not in order_df.columns:
        order_df["No"] = np.nan

    order_df = order_df.assign(
        signed_gap=lambda d: d["Yes"] - d["No"],
        median_gap=lambda d: (d["Yes"] - d["No"]).abs(),
    ).sort_values("median_gap", ascending=False)

    return order_df

def compute_ridge_feature_order(
    viz_df: pd.DataFrame,
    feature_labels: dict[str, str],
    y_col: str = "log_lfm_album_listeners",
    order_method: str = "median_gap",
) -> pd.DataFrame:
    """
    Compute a principled feature ordering for ridge-plot stacking.

    The default ordering ranks features by the absolute difference in
    median outcome between the condition-met group and the condition-not-met
    group. This highlights the features with the clearest separation in
    soundtrack listener distributions.

    Args:
        viz_df: Album-level ridge working dataframe containing the outcome
            and feature-specific `*_group` columns.
        feature_labels: Mapping from raw feature names to display labels.
        y_col: Outcome column used in the ridge plot.
        order_method: Ordering strategy to apply. Currently supported:
            - "median_gap": absolute difference between median Yes and No
              outcome values

    Returns:
        pd.DataFrame: Ordering summary dataframe with one row per feature
        label and columns such as:
            - No
            - Yes
            - median_gap

    Raises:
        ValueError: If required columns are missing or the requested order
            method is unsupported.
    """
    if y_col not in viz_df.columns:
        raise ValueError(f"Outcome column '{y_col}' was not found in viz_df.")

    if order_method != "median_gap":
        raise ValueError(
            f"Unsupported order_method '{order_method}'. "
            f"Currently supported: ['median_gap']"
        )

    group_cols = [
        f"{feature}_group"
        for feature in feature_labels.keys()
        if f"{feature}_group" in viz_df.columns
    ]

    if not group_cols:
        raise ValueError(
            "No ridge group columns were found for the supplied feature labels."
        )

    order_long = (
        viz_df[[y_col] + group_cols]
        .melt(
            id_vars=y_col,
            var_name="feature_group_col",
            value_name="group_raw",
        )
        .assign(
            feature=lambda d: d["feature_group_col"].str.replace(
                "_group$",
                "",
                regex=True,
            ),
            feature_label=lambda d: d["feature"].map(feature_labels),
            group_std=lambda d: d["group_raw"].apply(normalize_ridge_group_label),
        )
    )

    order_df = (
        order_long
        .groupby(["feature_label", "group_std"], observed=True)[y_col]
        .median()
        .unstack("group_std")
    )

    if "Yes" not in order_df.columns:
        order_df["Yes"] = np.nan
    if "No" not in order_df.columns:
        order_df["No"] = np.nan

    order_df = order_df.assign(
        median_gap=lambda d: (d["Yes"] - d["No"]).abs()
    ).sort_values("median_gap", ascending=False)

    return order_df

def build_ridge_chart_df(
    ridge_density_df: pd.DataFrame,
    feature_labels: dict[str, str],
    feature_order: list[str],
    row_gap: float = 2.5,
    density_scale: float = 11.0,
    label_y_offset: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build chart-ready ridge and label dataframes.

    This helper attaches display labels, standardizes group labels into
    Yes/No, applies a supplied vertical feature order, and computes the
    y-baseline / y-top coordinates used to draw ridgeline areas.

    It also returns a compact labels dataframe used for left-side feature
    labels and horizontal baseline rules.

    Args:
        ridge_density_df: Precomputed density dataframe returned by
            ``build_ridge_density_df()``.
        feature_labels: Mapping from raw feature names to human-readable
            display labels.
        feature_order: Ordered list of feature labels, typically derived
            from ``compute_ridge_feature_order()``.
        row_gap: Vertical spacing between ridge rows.
        density_scale: Visual multiplier applied to density heights.
        label_y_offset: Optional vertical offset for label placement.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            1. chart-ready ridge dataframe
            2. labels dataframe for text labels and baseline rules

    Raises:
        ValueError: If feature labels are missing or no usable rows remain.
    """
    if ridge_density_df.empty:
        raise ValueError("ridge_density_df is empty.")

    ridge_chart_df = ridge_density_df.copy()
    ridge_chart_df["feature_label"] = ridge_chart_df["feature"].map(feature_labels)

    ridge_chart_df = ridge_chart_df.dropna(subset=["feature_label"]).copy()
    if ridge_chart_df.empty:
        raise ValueError(
            "No ridge density rows matched the provided feature_labels mapping."
        )

    ridge_chart_df["group_std"] = ridge_chart_df["group"].apply(
        normalize_ridge_group_label
    )

    feature_order_clean = [
        feature_label
        for feature_label in feature_order
        if feature_label in ridge_chart_df["feature_label"].unique()
    ]

    if not feature_order_clean:
        raise ValueError(
            "None of the supplied feature_order labels were found in the "
            "ridge density dataframe."
        )

    n_features = len(feature_order_clean)
    feature_label_to_idx = {
        feature_label: (n_features - 1 - idx)
        for idx, feature_label in enumerate(feature_order_clean)
    }

    ridge_chart_df["feature_idx"] = ridge_chart_df["feature_label"].map(
        feature_label_to_idx
    )

    ridge_chart_df["y0"] = ridge_chart_df["feature_idx"] * row_gap
    ridge_chart_df["y1"] = ridge_chart_df["y0"] + (
        ridge_chart_df["density"] * density_scale
    )

    ridge_chart_df = ridge_chart_df.sort_values(
        ["feature_idx", "group_std", "x"]
    ).reset_index(drop=True)

    labels_df = (
        ridge_chart_df[["feature_label", "y0"]]
        .drop_duplicates()
        .set_index("feature_label")
        .loc[feature_order_clean]
        .reset_index()
        .copy()
    )
    labels_df["y_label"] = labels_df["y0"] + label_y_offset

    return ridge_chart_df, labels_df

def build_ridge_phase2_outputs(
    ridge_outputs: dict[str, Any],
    bins: int = 80,
    smooth_window: int = 11,
    min_group_n: int = 10,
    order_method: str = "median_gap",
    row_gap: float = 2.5,
    density_scale: float = 11.0,
    label_y_offset: float = 0.0,
) -> dict[str, Any]:
    """
    Run the second-stage ridge workflow after Phase 1 prep.

    This convenience wrapper takes the Phase 1 ridge outputs, precomputes
    density curves, derives a feature ordering, and builds the chart-ready
    ridge and label tables needed for Altair rendering.

    Args:
        ridge_outputs: Dictionary returned by ``build_ridge_prep_outputs()``.
        bins: Number of x bins used for density estimation.
        smooth_window: Odd-valued smoothing window for density estimation.
        min_group_n: Minimum observations required per feature-group pair.
        order_method: Ordering strategy for ridge rows.
        row_gap: Vertical spacing between ridge rows.
        density_scale: Visual scaling factor for density heights.
        label_y_offset: Optional vertical offset for feature labels.

    Returns:
        dict[str, Any]: Dictionary containing:
            - ridge_density_df
            - order_df
            - feature_order
            - ridge_chart_df
            - labels_df
    """
    ridge_config = ridge_outputs["ridge_config"]
    viz_df = ridge_outputs["viz_df"]
    ridge_long = ridge_outputs["ridge_long"]

    ridge_density_df = build_ridge_density_df(
        ridge_long=ridge_long,
        y_col=ridge_config["y_col"],
        bins=bins,
        smooth_window=smooth_window,
        min_group_n=min_group_n,
    )

    order_df = compute_ridge_feature_order(
        viz_df=viz_df,
        feature_labels=ridge_config["feature_labels"],
        y_col=ridge_config["y_col"],
        order_method=order_method,
    )

    feature_order = order_df.index.tolist()

    ridge_chart_df, labels_df = build_ridge_chart_df(
        ridge_density_df=ridge_density_df,
        feature_labels=ridge_config["feature_labels"],
        feature_order=feature_order,
        row_gap=row_gap,
        density_scale=density_scale,
        label_y_offset=label_y_offset,
    )

    return {
        "ridge_density_df": ridge_density_df,
        "order_df": order_df,
        "feature_order": feature_order,
        "ridge_chart_df": ridge_chart_df,
        "labels_df": labels_df,
    }