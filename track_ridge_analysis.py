from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.ui import get_display_label


TRACK_RIDGE_CONDITION_LABEL_OVERRIDES = {
    "energy": "Energy above median",
    "danceability": "Danceability above median",
    "happiness": "Happiness above median",
    "acousticness": "Acousticness above median",
    "instrumentalness": "Instrumentalness above median",
    "speechiness": "Speechiness above median",
    "liveness": "Liveness above median",
    "tempo": "Tempo above median",
    "loudness": "Loudness above median",
    "duration_seconds": "Track duration above median",
    "relative_track_position": "Later track position (above median)",
    "track_intensity_score": "High intensity archetype",
    "track_acoustic_orchestral_score": "High acoustic / orchestral archetype",
    "track_speech_texture_score": "High speech texture archetype",
    "is_first_track": "Opening track",
    "is_last_track": "Closing track",
    "is_first_three_tracks": "In first 3 tracks",
    "is_first_five_tracks": "In first 5 tracks",
    "is_instrumental": "Instrumental track",
    "is_high_energy": "High-energy track",
    "is_high_happiness": "High-happiness track",
    "is_major_mode": "Major-mode track",
    "has_any_audio_features": "Has audio features",
    "film_vote_count": "Film vote count above median",
    "film_popularity": "Film popularity above median",
    "film_rating": "Film rating above median",
    "film_runtime_min": "Film runtime above median",
    "days_since_film_release": "Film released longer ago than median",
    "n_tracks": "Album has many tracks (above median)",
    "album_release_lag_days": "Album released later than median",
    "composer_album_count": "Composer appears on many albums (above median)",
    "album_cohesion_score": "Album cohesion above median",

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

    "bafta_nominee": "BAFTA nominee",
    "us_score_nominee_count": "Any score nominations",
    "us_song_nominee_count": "Any song nominations",
}

TRACK_RIDGE_TARGET_OPTIONS = [
    "log_lfm_track_playcount",
    "log_lfm_track_listeners",
    "spotify_popularity",
]

TRACK_DEFAULT_RIDGE_GROUPS = {
    "core_audio": [
        "energy",
        "danceability",
        "happiness",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "tempo",
        "loudness",
        "duration_seconds",
    ],
    "position_and_sequence": [
        "is_first_track",
        "is_last_track",
        "is_first_three_tracks",
        "is_first_five_tracks",
        "relative_track_position",
    ],
    "track_type_and_mood": [
        "is_instrumental",
        "is_high_energy",
        "is_high_happiness",
        "is_major_mode",
        "acousticness",
        "speechiness",
    ],
    "archetypes": [
        "track_intensity_score",
        "track_acoustic_orchestral_score",
        "track_speech_texture_score",
    ],
    "film_and_album_context": [
        "film_vote_count",
        "film_popularity",
        "film_rating",
        "film_runtime_min",
        "days_since_film_release",
        "n_tracks",
        "album_release_lag_days",
        "composer_album_count",
        "album_cohesion_score",
    ],

    "context_genres_and_awards": [
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
        "ambient_experimental",
        "classical_orchestral",
        "electronic",
        "hip_hop_rnb",
        "pop",
        "rock",
        "world_folk",
        "bafta_nominee",
        "us_score_nominee_count",
        "us_song_nominee_count",
    ],
}

TRACK_BINARY_FLAG_FEATURES = {
    "is_first_track",
    "is_last_track",
    "is_first_three_tracks",
    "is_first_five_tracks",
    "is_instrumental",
    "is_high_energy",
    "is_high_happiness",
    "is_major_mode",
    "has_any_audio_features",
}


def get_track_ridge_condition_label(feature: str) -> str:
    """
    Return the user-facing ridge-row label for a feature split.

    This differs from a plain display label because ridge rows represent
    binary conditions such as "above median" or "track is opening track",
    not just raw field names.

    Args:
        feature: Raw track feature name.

    Returns:
        str: Human-readable ridge condition label.
    """
    return TRACK_RIDGE_CONDITION_LABEL_OVERRIDES.get(
        feature,
        f"{get_display_label(feature)} above median",
    )


def get_track_ridge_feature_config(
    track_df: pd.DataFrame,
    y_col: str = "log_lfm_track_playcount",
    feature_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Build the track-ridge feature configuration from available track columns.

    This helper mirrors the album ridge config pattern:
    - keeps only features that actually exist in the dataframe
    - classifies them into binary flags vs continuous-like features
    - prepares the keep_cols and label map needed downstream

    Args:
        track_df: Track-level dataframe.
        y_col: Selected track-level outcome column.
        feature_groups: Optional override for grouped ridge feature definitions.

    Returns:
        dict[str, Any]: Configuration dictionary for the track ridge workflow.

    Raises:
        ValueError: If the selected outcome column is not present.
    """
    if y_col not in track_df.columns:
        raise ValueError(
            f"Outcome column '{y_col}' was not found in the track dataframe."
        )

    if feature_groups is None:
        feature_groups = TRACK_DEFAULT_RIDGE_GROUPS

    feature_groups_available: dict[str, list[str]] = {}
    for group_name, cols in feature_groups.items():
        available = [col for col in cols if col in track_df.columns]
        if available:
            feature_groups_available[group_name] = available

    all_features: list[str] = []
    for cols in feature_groups_available.values():
        for col in cols:
            if col not in all_features:
                all_features.append(col)

    keep_cols = [y_col] + all_features

    binary_flag_features = [
        col for col in all_features
        if col in TRACK_BINARY_FLAG_FEATURES
    ]

    cont_like_features = [
        col for col in all_features
        if col not in binary_flag_features
    ]

    available_feature_labels = {
        col: get_track_ridge_condition_label(col)
        for col in all_features
    }

    return {
        "y_col": y_col,
        "feature_groups_available": feature_groups_available,
        "all_features": all_features,
        "keep_cols": keep_cols,
        "feature_labels": available_feature_labels,
        "cont_like_features": cont_like_features,
        "binary_flag_features": binary_flag_features,
    }


def build_track_ridge_viz_df(
    track_df: pd.DataFrame,
    ridge_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build the cleaned working dataframe used for track ridge preparation.

    This:
    - selects the outcome + selected ridge features
    - coerces the outcome to numeric and drops invalid rows
    - coerces continuous-like features to numeric
    - standardizes binary flag features into stable 0/1 values

    Args:
        track_df: Source track-level dataframe.
        ridge_config: Configuration dictionary returned by
            get_track_ridge_feature_config().

    Returns:
        pd.DataFrame: Cleaned track-level ridge working dataframe.
    """
    y_col = ridge_config["y_col"]
    keep_cols = ridge_config["keep_cols"]
    cont_like_features = ridge_config["cont_like_features"]
    binary_flag_features = ridge_config["binary_flag_features"]

    viz_df = track_df[keep_cols].copy()

    viz_df[y_col] = pd.to_numeric(viz_df[y_col], errors="coerce")
    viz_df = viz_df.dropna(subset=[y_col]).copy()

    for col in cont_like_features:
        if col in viz_df.columns:
            viz_df[col] = pd.to_numeric(viz_df[col], errors="coerce")

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


def add_track_ridge_group_columns(
    viz_df: pd.DataFrame,
    ridge_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Add binary comparison columns used in the track ridge workflow.

    Rules:
    - continuous-like features become "Above median" vs "Below median"
    - binary flag features become "True" vs "False"

    Args:
        viz_df: Cleaned track ridge working dataframe.
        ridge_config: Configuration dictionary returned by
            get_track_ridge_feature_config().

    Returns:
        pd.DataFrame: Ridge working dataframe with *_group columns added.
    """
    viz_df = viz_df.copy()

    cont_like_features = ridge_config["cont_like_features"]
    binary_flag_features = ridge_config["binary_flag_features"]

    for col in cont_like_features:
        if col in viz_df.columns:
            median_val = viz_df[col].median(skipna=True)
            viz_df[f"{col}_group"] = np.where(
                viz_df[col] >= median_val,
                "Above median",
                "Below median",
            )

    for col in binary_flag_features:
        if col in viz_df.columns:
            viz_df[f"{col}_group"] = np.where(
                viz_df[col] == 1,
                "True",
                "False",
            )

    return viz_df


def build_track_ridge_long_df(
    viz_df: pd.DataFrame,
    y_col: str = "log_lfm_track_playcount",
) -> pd.DataFrame:
    """
    Convert the grouped track ridge dataframe into long format.

    Args:
        viz_df: Ridge dataframe containing outcome + *_group columns.
        y_col: Selected track-level outcome column.

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
            "No *_group columns were found. Run add_track_ridge_group_columns() first."
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
    if ("above" in s) or (s == "true"):
        return "Yes"
    return "No"


def build_track_ridge_prep_outputs(
    track_df: pd.DataFrame,
    y_col: str = "log_lfm_track_playcount",
    feature_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Run the full Phase 1 track ridge preparation workflow.

    Args:
        track_df: Track-level dataframe.
        y_col: Selected track-level outcome column.
        feature_groups: Optional override for static ridge groups.

    Returns:
        dict[str, Any]: Dictionary containing:
            - ridge_config
            - viz_df
            - ridge_long
    """
    ridge_config = get_track_ridge_feature_config(
        track_df=track_df,
        y_col=y_col,
        feature_groups=feature_groups,
    )
    viz_df = build_track_ridge_viz_df(track_df, ridge_config)
    viz_df = add_track_ridge_group_columns(viz_df, ridge_config)
    ridge_long = build_track_ridge_long_df(viz_df, y_col=y_col)

    return {
        "ridge_config": ridge_config,
        "viz_df": viz_df,
        "ridge_long": ridge_long,
    }

def build_track_ridge_density_df(
    ridge_long: pd.DataFrame,
    y_col: str = "log_lfm_track_playcount",
    bins: int = 80,
    smooth_window: int = 11,
    min_group_n: int = 10,
) -> pd.DataFrame:
    """
    Precompute ridge density curves on a shared x-grid for track outcomes.

    This mirrors the album ridge density pipeline:
    - uses a shared x-grid across all features/groups
    - estimates histogram-based densities
    - smooths them with a moving average
    - skips unstable small groups

    Args:
        ridge_long: Long-form ridge dataframe containing outcome, feature,
            and group columns.
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


def compute_track_ridge_feature_order(
    viz_df: pd.DataFrame,
    feature_labels: dict[str, str],
    y_col: str = "log_lfm_track_playcount",
    order_method: str = "median_gap",
) -> pd.DataFrame:
    """
    Compute a principled feature ordering for track ridge stacking.

    The default ordering ranks features by the absolute difference in
    median outcome between the condition-met group and the condition-not-met
    group.

    Args:
        viz_df: Track-level ridge working dataframe containing the outcome
            and feature-specific `*_group` columns.
        feature_labels: Mapping from raw feature names to ridge-row labels.
        y_col: Outcome column used in the ridge plot.
        order_method: Ordering strategy to apply. Currently supported:
            - "median_gap": absolute difference between median Yes and No
              outcome values

    Returns:
        pd.DataFrame: Ordering summary dataframe with one row per feature
        label and columns:
            - No
            - Yes
            - signed_gap
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


def build_track_ridge_chart_df(
    ridge_density_df: pd.DataFrame,
    feature_labels: dict[str, str],
    feature_order: list[str],
    row_gap: float = 2.5,
    density_scale: float = 11.0,
    label_y_offset: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build chart-ready ridge and label dataframes for track ridge rendering.

    This mirrors the album workflow:
    - attaches display labels
    - standardizes group labels into Yes/No
    - applies the supplied vertical feature order
    - computes y-baseline / y-top coordinates

    Args:
        ridge_density_df: Precomputed density dataframe returned by
            build_track_ridge_density_df().
        feature_labels: Mapping from raw feature names to human-readable
            ridge labels.
        feature_order: Ordered list of feature labels, typically derived
            from compute_track_ridge_feature_order().
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


def build_track_ridge_phase2_outputs(
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
    Run the Phase 2 track ridge workflow after Phase 1 prep.

    This convenience wrapper takes the Phase 1 outputs, precomputes
    density curves, derives a feature ordering, and builds the chart-ready
    ridge and label tables needed for Altair rendering.

    Args:
        ridge_outputs: Dictionary returned by build_track_ridge_prep_outputs().
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

    ridge_density_df = build_track_ridge_density_df(
        ridge_long=ridge_long,
        y_col=ridge_config["y_col"],
        bins=bins,
        smooth_window=smooth_window,
        min_group_n=min_group_n,
    )

    order_df = compute_track_ridge_feature_order(
        viz_df=viz_df,
        feature_labels=ridge_config["feature_labels"],
        y_col=ridge_config["y_col"],
        order_method=order_method,
    )

    feature_order = order_df.index.tolist()

    ridge_chart_df, labels_df = build_track_ridge_chart_df(
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