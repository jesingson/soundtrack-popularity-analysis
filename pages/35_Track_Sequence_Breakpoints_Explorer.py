from __future__ import annotations

import re
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_sequence_controls,
)
from app.app_data import load_track_data_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_film_year_bucket,
    add_standard_multivalue_groups,
    get_global_filter_inputs,
    get_track_page_display_label,
    rename_track_page_columns_for_display,
    select_unique_existing_columns,
)
from app.ui import apply_app_styles


SEQUENCE_DETAIL_ALBUM_COLS = [
    "film_title",
    "album_title",
    "composer_primary_clean",
    "label_names",
    "film_year",
    "film_genres",
    "album_genres_display",
    "n_tracks",
    "lfm_album_listeners",
    "lfm_album_playcount",
]

AUDIO_FEATURE_OPTIONS = [
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
    "mode",
    "key",
]

COMPARISON_DIMENSIONS = [
    "Film genres",
    "Album genres",
    "Composer",
    "Film year",
    "Labels",
    "Album cohesion group",
]

COHESION_GROUP_ORDER = [
    "Low variability",
    "Medium variability",
    "High variability",
]

PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
]


def split_multivalue_cell(value: object) -> list[str]:
    """Split pipe- or comma-delimited cells into clean unique values."""
    if pd.isna(value):
        return []

    parts = re.split(r"\s*\|\s*|\s*,\s*", str(value))
    cleaned = []
    seen = set()

    for part in parts:
        part = part.strip()
        if part and part not in seen:
            cleaned.append(part)
            seen.add(part)

    return cleaned


def add_track_sequence_display_fields(track_df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped display fields used for Page 35."""
    df = add_standard_multivalue_groups(track_df)
    df = add_film_year_bucket(df)

    for col in ["composer_primary_clean", "label_names"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    return df


def filter_track_sequence_df(
    track_df: pd.DataFrame,
    global_controls: dict,
    controls: dict,
) -> pd.DataFrame:
    """Apply shared global filters plus Page 35-specific filters."""
    merged_controls = {
        **global_controls,
        "search_text": controls.get("search_text", ""),
    }

    filtered = filter_dataset(track_df, merged_controls).copy()

    if "track_number" in filtered.columns:
        filtered = filtered[
            filtered["track_number"].fillna(0) <= controls["max_track_position"]
        ].copy()

    if "lfm_album_listeners" in filtered.columns:
        filtered = filtered[
            filtered["lfm_album_listeners"].fillna(0) >= controls["min_album_listeners"]
        ].copy()

    if controls.get("audio_only", False) and controls["metric"] in filtered.columns:
        filtered = filtered.dropna(subset=[controls["metric"]]).copy()

    return filtered


def build_sequence_base_df(
    df: pd.DataFrame,
    metric: str,
    n_bins: int,
) -> pd.DataFrame:
    """
    Normalize track position within album.

    Output grain:
        one row per visible track
    """
    required_cols = [
        "release_group_mbid",
        "tmdb_id",
        "track_number",
        "album_title",
        "film_title",
        "film_year",
        "film_genres",
        "album_genres_display",
        "composer_primary_clean",
        "label_names",
        "lfm_album_listeners",
        metric,
    ]
    keep_cols = [col for col in required_cols if col in df.columns]

    working = df[keep_cols].copy()
    working = working.dropna(subset=["track_number", metric]).copy()

    working["album_max_track"] = (
        working.groupby("release_group_mbid")["track_number"].transform("max")
    )
    working = working[working["album_max_track"].fillna(0) >= 2].copy()

    working["pos_norm"] = (
        (working["track_number"] - 1)
        / (working["album_max_track"] - 1)
    ).clip(lower=0.0, upper=1.0)

    working["pos_bin"] = (
        (working["pos_norm"] * (n_bins - 1)).round().astype(int)
    )
    working["pos_bin_norm"] = working["pos_bin"] / (n_bins - 1)

    return working


def add_album_cohesion_group(
    base_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Add low / medium / high variability bins per album for the selected metric.

    This mirrors the low/medium/high variability logic used on the Track Cohesion
    Explorer, but applies it to the currently selected audio feature. :contentReference[oaicite:1]{index=1}
    """
    df = base_df.copy()

    album_var_df = (
        df.groupby(["release_group_mbid", "tmdb_id"], as_index=False)[metric]
        .agg(metric_std="std")
    )
    album_var_df["metric_std"] = album_var_df["metric_std"].fillna(0.0)

    valid = album_var_df["metric_std"].dropna()
    if valid.empty or valid.nunique() < 3:
        album_var_df["album_cohesion_group"] = "Unbinned"
    else:
        q1 = valid.quantile(1 / 3)
        q2 = valid.quantile(2 / 3)

        def assign_bin(val: float) -> str:
            if pd.isna(val):
                return "Unknown"
            if val <= q1:
                return "Low variability"
            if val <= q2:
                return "Medium variability"
            return "High variability"

        album_var_df["album_cohesion_group"] = album_var_df["metric_std"].apply(assign_bin)

    df = df.merge(
        album_var_df[["release_group_mbid", "tmdb_id", "album_cohesion_group"]],
        on=["release_group_mbid", "tmdb_id"],
        how="left",
        validate="m:1",
    )

    return df


def build_grouped_sequence_df(
    base_df: pd.DataFrame,
    controls: dict,
) -> pd.DataFrame:
    """
    Expand / assign the chosen grouping dimension into a unified group_value column.
    """
    metric = controls["metric"]
    comparison_dimension = controls["comparison_dimension"]

    df = base_df.copy()

    if comparison_dimension == "Film genres":
        df["group_value"] = df["film_genres"].apply(split_multivalue_cell)
        df = df.explode("group_value")
        df = df[df["group_value"].notna() & (df["group_value"] != "")].copy()

    elif comparison_dimension == "Album genres":
        df["group_value"] = df["album_genres_display"].apply(split_multivalue_cell)
        df = df.explode("group_value")
        df = df[df["group_value"].notna() & (df["group_value"] != "")].copy()

    elif comparison_dimension == "Composer":
        df["group_value"] = (
            df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        df = df[
            df["group_value"].ne("")
            & df["group_value"].ne("Unknown")
            ].copy()

    elif comparison_dimension == "Film year":
        df = df[df["film_year"].notna()].copy()
        df["group_value"] = df["film_year"].astype(int).astype(str)


    elif comparison_dimension == "Labels":
        df["group_value"] = df["label_names"].apply(split_multivalue_cell)
        df = df.explode("group_value")
        df["group_value"] = df["group_value"].fillna("").astype(str).str.strip()
        df = df[
            df["group_value"].ne("")
            & df["group_value"].ne("Unknown")
            ].copy()

    elif comparison_dimension == "Album cohesion group":
        df = add_album_cohesion_group(df, metric=metric)
        df["group_value"] = df["album_cohesion_group"].fillna("Unknown").astype(str)

    else:
        df["group_value"] = "All"

    return df


def build_group_options(
    df: pd.DataFrame,
    controls: dict,
) -> list[str]:
    """
    Determine available group values based on album-level presence.
    """
    if df.empty or "group_value" not in df.columns:
        return []

    working = df.copy()

    # explode list-valued group columns if needed
    first_valid = working["group_value"].dropna()
    if not first_valid.empty and isinstance(first_valid.iloc[0], list):
        working = working.explode("group_value")

    working["group_value"] = (
        working["group_value"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    working = working[
        working["group_value"].ne("")
        & working["group_value"].ne("Unknown")
    ].copy()

    if working.empty:
        return []

    group_counts_df = (
        working.groupby("group_value", as_index=False)["release_group_mbid"]
        .nunique()
        .rename(columns={"release_group_mbid": "album_count"})
    )

    group_counts_df = group_counts_df[
        group_counts_df["album_count"] >= controls["min_albums_per_group"]
    ].copy()

    if group_counts_df.empty:
        return []

    if controls["comparison_dimension"] in ["Composer", "Labels"]:
        group_counts_df = group_counts_df.sort_values(
            ["album_count", "group_value"],
            ascending=[False, True],
        )

    elif controls["comparison_dimension"] == "Film year":
        group_counts_df["sort_year"] = pd.to_numeric(
            group_counts_df["group_value"],
            errors="coerce",
        )
        group_counts_df = (
            group_counts_df.sort_values(["sort_year", "group_value"])
            .drop(columns=["sort_year"])
        )

    elif controls["comparison_dimension"] == "Album cohesion group":
        group_counts_df["sort_order"] = group_counts_df["group_value"].map(
            {name: i for i, name in enumerate(COHESION_GROUP_ORDER)}
        )
        group_counts_df = (
            group_counts_df.sort_values(["sort_order", "group_value"])
            .drop(columns=["sort_order"])
        )

    else:
        group_counts_df = group_counts_df.sort_values(
            ["album_count", "group_value"],
            ascending=[False, True],
        )

    return group_counts_df["group_value"].tolist()

def render_group_selector(
    group_options: list[str],
    controls: dict,
) -> list[str]:
    """Render a dynamic group multiselect after comparison-dimension options are known."""
    if not group_options:
        return []

    default_groups = group_options[: min(3, len(group_options))]

    if controls["comparison_dimension"] == "Album genres":
        preferred = ["Classical/Orchestral", "Pop", "Electronic"]
        preferred_present = [g for g in preferred if g in group_options]
        if preferred_present:
            default_groups = preferred_present

    group_search = st.sidebar.text_input(
        "Filter group options",
        value="",
        help="Type to narrow the available group list before selecting groups.",
    )

    filtered_group_options = group_options
    if group_search.strip():
        search_text = group_search.strip().lower()
        filtered_group_options = [
            g for g in group_options
            if search_text in str(g).lower()
        ]

    selected_groups = st.sidebar.multiselect(
        "Groups to compare",
        options=filtered_group_options,
        default=[g for g in default_groups if g in filtered_group_options],
        help="Select up to the maximum visible group limit.",
    )

    if not selected_groups:
        selected_groups = default_groups

    return selected_groups[: controls["max_groups"]]

def aggregate_sequence_df(
    grouped_df: pd.DataFrame,
    controls: dict,
) -> pd.DataFrame:
    """
    Aggregate the selected audio feature by group and normalized position bin.

    Notes:
    - The ribbon is always built around the same center statistic used by the line.
    - This version intentionally removes the overly wide legacy band options.
    - Supported ribbon widths:
        * Middle 5%
        * Middle 10%
        * Middle 20%
        * IQR
        * Mean ± 0.25 SD
        * Mean ± 0.5 SD
    """
    metric = controls["metric"]

    if grouped_df.empty:
        return pd.DataFrame(
            columns=[
                "group_value",
                "pos_bin",
                "pos_bin_norm",
                "center_value",
                "lower_value",
                "upper_value",
                "n_tracks",
                "n_albums",
            ]
        )

    agg_df = (
        grouped_df.groupby(["group_value", "pos_bin", "pos_bin_norm"], as_index=False)
        .agg(
            mean_value=(metric, "mean"),
            median_value=(metric, "median"),
            std_value=(metric, "std"),
            p25_value=(metric, lambda x: x.quantile(0.25)),
            p40_value=(metric, lambda x: x.quantile(0.40)),
            p45_value=(metric, lambda x: x.quantile(0.45)),
            p475_value=(metric, lambda x: x.quantile(0.475)),
            p525_value=(metric, lambda x: x.quantile(0.525)),
            p55_value=(metric, lambda x: x.quantile(0.55)),
            p60_value=(metric, lambda x: x.quantile(0.60)),
            p75_value=(metric, lambda x: x.quantile(0.75)),
            n_tracks=(metric, "size"),
            n_albums=("release_group_mbid", "nunique"),
        )
        .sort_values(["group_value", "pos_bin"])
        .reset_index(drop=True)
    )

    agg_df["std_value"] = agg_df["std_value"].fillna(0.0)

    center_mode = controls["center_stat"]
    if controls["metric"] == "mode":
        center_mode = "Mean"
    ribbon_mode = controls["ribbon_range"]

    if center_mode == "Median":
        agg_df["center_value"] = agg_df["median_value"]

        if ribbon_mode == "Middle 5%":
            agg_df["lower_value"] = agg_df["p475_value"]
            agg_df["upper_value"] = agg_df["p525_value"]
        elif ribbon_mode == "Middle 10%":
            agg_df["lower_value"] = agg_df["p45_value"]
            agg_df["upper_value"] = agg_df["p55_value"]
        elif ribbon_mode == "Middle 20%":
            agg_df["lower_value"] = agg_df["p40_value"]
            agg_df["upper_value"] = agg_df["p60_value"]
        elif ribbon_mode == "IQR":
            agg_df["lower_value"] = agg_df["p25_value"]
            agg_df["upper_value"] = agg_df["p75_value"]
        elif ribbon_mode == "Mean ± 0.25 SD":
            half_width = 0.25 * agg_df["std_value"]
            agg_df["lower_value"] = agg_df["center_value"] - half_width
            agg_df["upper_value"] = agg_df["center_value"] + half_width
        elif ribbon_mode == "Mean ± 0.5 SD":
            half_width = 0.5 * agg_df["std_value"]
            agg_df["lower_value"] = agg_df["center_value"] - half_width
            agg_df["upper_value"] = agg_df["center_value"] + half_width
        else:
            agg_df["lower_value"] = agg_df["p45_value"]
            agg_df["upper_value"] = agg_df["p55_value"]

        if controls["metric"] == "mode":
            agg_df["lower_value"] = agg_df["lower_value"].clip(lower=0, upper=1)
            agg_df["upper_value"] = agg_df["upper_value"].clip(lower=0, upper=1)

    else:
        agg_df["center_value"] = agg_df["mean_value"]

        if ribbon_mode == "Middle 5%":
            half_width = (agg_df["p525_value"] - agg_df["p475_value"]) / 2
        elif ribbon_mode == "Middle 10%":
            half_width = (agg_df["p55_value"] - agg_df["p45_value"]) / 2
        elif ribbon_mode == "Middle 20%":
            half_width = (agg_df["p60_value"] - agg_df["p40_value"]) / 2
        elif ribbon_mode == "IQR":
            half_width = (agg_df["p75_value"] - agg_df["p25_value"]) / 2
        elif ribbon_mode == "Mean ± 0.25 SD":
            half_width = 0.25 * agg_df["std_value"]
        elif ribbon_mode == "Mean ± 0.5 SD":
            half_width = 0.5 * agg_df["std_value"]
        else:
            half_width = (agg_df["p55_value"] - agg_df["p45_value"]) / 2

        agg_df["lower_value"] = agg_df["center_value"] - half_width
        agg_df["upper_value"] = agg_df["center_value"] + half_width

    if controls["smoothing_window"] > 1:
        for col in ["center_value", "lower_value", "upper_value"]:
            agg_df[col] = (
                agg_df.groupby("group_value")[col]
                .transform(
                    lambda s: s.rolling(
                        window=controls["smoothing_window"],
                        min_periods=1,
                        center=True,
                    ).mean()
                )
            )

    return agg_df

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def create_sequence_ribbon_chart(
    agg_df: pd.DataFrame,
    controls: dict,
) -> go.Figure:
    """
    Create a single multi-group alluvial-style ribbon chart.
    """
    fig = go.Figure()

    if agg_df.empty:
        fig.update_layout(
            height=620,
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

    groups = agg_df["group_value"].drop_duplicates().tolist()

    # make cohesion groups visually ordered if present
    if controls["comparison_dimension"] == "Album cohesion group":
        groups = [g for g in COHESION_GROUP_ORDER if g in groups]

    metric_label = get_track_page_display_label(controls["metric"])

    for i, group in enumerate(groups):
        gdf = agg_df[agg_df["group_value"] == group].sort_values("pos_bin_norm").copy()
        color = PALETTE[i % len(PALETTE)]

        show_ribbon = controls["show_ribbon"] and controls["metric"] != "mode"

        if show_ribbon:
            fig.add_trace(
                go.Scatter(
                    x=list(gdf["pos_bin_norm"]) + list(gdf["pos_bin_norm"])[::-1],
                    y=list(gdf["upper_value"]) + list(gdf["lower_value"])[::-1],
                    fill="toself",
                    fillcolor=hex_to_rgba(color, 0.12),
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=str(group),
                    name=str(group),
                )
            )

        fig.add_trace(
            go.Scatter(
                x=gdf["pos_bin_norm"],
                y=gdf["center_value"],
                mode="lines",
                line=dict(
                    width=4,
                    color=color,
                    shape="spline",
                    smoothing=1.15,
                ),
                name=str(group),
                legendgroup=str(group),
                customdata=gdf[["lower_value", "upper_value", "n_tracks", "n_albums"]],
                hovertemplate=(
                    "Group: %{fullData.name}<br>"
                    "Normalized Position: %{x:.2f}<br>"
                    f"{metric_label}: " + "%{y:.3f}<br>"
                    "Ribbon: %{customdata[0]:.3f} – %{customdata[1]:.3f}<br>"
                    "Tracks: %{customdata[2]:,.0f}<br>"
                    "Albums: %{customdata[3]:,.0f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=620,
        template="plotly_dark",
        margin=dict(l=50, r=40, t=40, b=40),
        hovermode="x unified",
        legend_title_text=controls["comparison_dimension"].rstrip("s"),
        xaxis=dict(
            title="Normalized Track Position",
            tickformat=".1f",
            range=[0, 1],
        ),
        yaxis=dict(
            title=metric_label,
            tickmode="array" if controls["metric"] == "mode" else "auto",
            tickvals=[0, 0.5, 1] if controls["metric"] == "mode" else None,
            ticktext=["Minor (0)", "Mixed", "Major (1)"] if controls["metric"] == "mode" else None,
        ),
    )

    return fig

def assign_horizon_group_value(
    df: pd.DataFrame,
    controls: dict,
) -> pd.DataFrame:
    """
    Assign a single comparison-group value per album for the horizon chart.

    This preserves multi-membership for the alluvial chart, but forces each
    album to appear only once in the horizon chart.
    """
    out = df.copy()
    comparison_dimension = controls["comparison_dimension"]

    def first_clean_token(value: object) -> str:
        tokens = split_multivalue_cell(value)
        return tokens[0] if tokens else ""

    if comparison_dimension == "Album genres":
        source_col = "album_genres_display"

    elif comparison_dimension == "Film genres":
        source_col = "film_genres"

    elif comparison_dimension == "Composer":
        source_col = "composer_primary_clean"

    elif comparison_dimension == "Labels":
        source_col = "label_names"

    else:
        return out

    if source_col not in out.columns:
        return out

    # Collapse to one canonical value per album
    primary_group_df = (
        out.groupby(["release_group_mbid", "tmdb_id"], as_index=False)
        .agg(source_value=(source_col, "first"))
    )

    primary_group_df["horizon_group_value"] = (
        primary_group_df["source_value"]
        .apply(first_clean_token)
        .astype(str)
        .str.strip()
    )

    out = out.merge(
        primary_group_df[["release_group_mbid", "tmdb_id", "horizon_group_value"]],
        on=["release_group_mbid", "tmdb_id"],
        how="left",
        validate="m:1",
    )

    out["group_value"] = out["horizon_group_value"]
    out = out.drop(columns=["horizon_group_value"], errors="ignore")

    # Remove blank / unknown group assignments
    out = out[
        out["group_value"].fillna("").astype(str).str.strip().ne("")
        & out["group_value"].fillna("").astype(str).str.strip().ne("Unknown")
    ].copy()

    return out

def build_album_horizon_df(
    grouped_df: pd.DataFrame,
    controls: dict,
    selected_groups: list[str],
) -> pd.DataFrame:
    """
    Build one smoothed, densified sequence per album for the selected groups.

    Output grain:
        one row per (group_value, album, x_norm)
    """
    metric = controls["metric"]

    working = grouped_df.copy()
    working = assign_horizon_group_value(
        df=working,
        controls=controls,
    )

    working = working[working["group_value"].isin(selected_groups)].copy()
    if working.empty:
        return pd.DataFrame()

    # After collapsing to a single horizon group, remove duplicate album/bin rows
    # created by the exploded alluvial grouping.
    working = (
        working.sort_values(["release_group_mbid", "tmdb_id", "pos_bin", "group_value"])
        .drop_duplicates(
            subset=[
                "group_value",
                "release_group_mbid",
                "tmdb_id",
                "album_title",
                "film_title",
                "lfm_album_listeners",
                "pos_bin",
                "pos_bin_norm",
            ]
        )
        .copy()
    )

    group_cols = [
        "group_value",
        "release_group_mbid",
        "tmdb_id",
        "album_title",
        "film_title",
        "film_genres",
        "album_genres_display",
        "composer_primary_clean",
        "label_names",
        "lfm_album_listeners",
        "pos_bin",
        "pos_bin_norm",
    ]
    group_cols = [col for col in group_cols if col in working.columns]

    album_seq_df = (
        working.groupby(group_cols, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "raw_value"})
        .sort_values(["group_value", "release_group_mbid", "pos_bin_norm"])
        .reset_index(drop=True)
    )

    if controls["smoothing_window"] > 1:
        album_seq_df["raw_value"] = (
            album_seq_df.groupby(["group_value", "release_group_mbid"])["raw_value"]
            .transform(
                lambda s: s.rolling(
                    window=controls["smoothing_window"],
                    min_periods=1,
                    center=True,
                ).mean()
            )
        )

    dense_x = np.linspace(0, 1, 161)
    rows = []

    group_cols = [
        "group_value",
        "release_group_mbid",
        "tmdb_id",
        "album_title",
        "film_title",
        "film_genres",
        "album_genres_display",
        "composer_primary_clean",
        "label_names",
        "lfm_album_listeners",
    ]
    group_cols = [col for col in group_cols if col in album_seq_df.columns]

    for keys, sdf in album_seq_df.groupby(group_cols, dropna=False):
        sdf = sdf.sort_values("pos_bin_norm").copy()

        x = sdf["pos_bin_norm"].to_numpy(dtype=float)
        y = sdf["raw_value"].to_numpy(dtype=float)

        if len(x) == 0:
            continue

        if len(np.unique(x)) == 1:
            dense_y = np.repeat(y[0], len(dense_x))
        else:
            unique_df = (
                pd.DataFrame({"x": x, "y": y})
                .groupby("x", as_index=False)["y"]
                .mean()
                .sort_values("x")
            )
            dense_y = np.interp(
                dense_x,
                unique_df["x"].to_numpy(dtype=float),
                unique_df["y"].to_numpy(dtype=float),
            )

        dense_series = pd.Series(dense_y)
        dense_y = (
            dense_series.rolling(window=7, min_periods=1, center=True).mean()
            .to_numpy()
        )

        key_map = dict(zip(group_cols, keys))

        dense_df = pd.DataFrame(
            {
                "group_value": key_map.get("group_value", ""),
                "release_group_mbid": key_map.get("release_group_mbid", ""),
                "tmdb_id": key_map.get("tmdb_id", ""),
                "album_title": key_map.get("album_title", ""),
                "film_title": key_map.get("film_title", ""),
                "film_genres": key_map.get("film_genres", ""),
                "album_genres_display": key_map.get("album_genres_display", ""),
                "composer_primary_clean": key_map.get("composer_primary_clean", ""),
                "label_names": key_map.get("label_names", ""),
                "lfm_album_listeners": key_map.get("lfm_album_listeners", np.nan),
                "pos_bin_norm": dense_x,
                "raw_value": dense_y,
            }
        )
        rows.append(dense_df)

    if not rows:
        return pd.DataFrame()

    album_seq_df = pd.concat(rows, ignore_index=True)

    if controls["horizon_normalization"] == "Album-centered shape":
        album_seq_df["baseline"] = (
            album_seq_df.groupby(["group_value", "release_group_mbid"])["raw_value"]
            .transform("median")
        )
    else:
        album_seq_df["baseline"] = (
            album_seq_df.groupby(["group_value", "pos_bin_norm"])["raw_value"]
            .transform("median")
        )

    album_seq_df["centered_value"] = album_seq_df["raw_value"] - album_seq_df["baseline"]

    return album_seq_df

def filter_top_albums_for_horizon(
    album_horizon_df: pd.DataFrame,
    controls: dict,
) -> pd.DataFrame:
    """
    Keep the top N albums per selected group for the horizon chart.
    """
    if album_horizon_df.empty:
        return album_horizon_df

    sort_mode = controls["horizon_sort"]

    album_rank_df = (
        album_horizon_df[
            [
                "group_value",
                "release_group_mbid",
                "tmdb_id",
                "album_title",
                "film_title",
                "lfm_album_listeners",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    if sort_mode == "Group, then album title":
        album_rank_df = album_rank_df.sort_values(
            ["group_value", "album_title", "lfm_album_listeners"],
            ascending=[True, True, False],
            na_position="last",
        )
        album_rank_df["group_rank"] = album_rank_df.groupby("group_value").cumcount() + 1
        album_rank_df = album_rank_df[
            album_rank_df["group_rank"] <= controls["albums_per_group"]
        ].copy()

    elif sort_mode == "Album title (global)":
        album_rank_df = album_rank_df.sort_values(
            ["album_title", "lfm_album_listeners"],
            ascending=[True, False],
            na_position="last",
        )
        album_rank_df["global_rank"] = album_rank_df.groupby("group_value").cumcount() + 1
        album_rank_df = album_rank_df[
            album_rank_df["global_rank"] <= controls["albums_per_group"]
        ].copy()

    elif sort_mode == "Album listeners (global)":
        album_rank_df = album_rank_df.sort_values(
            ["lfm_album_listeners", "album_title"],
            ascending=[False, True],
            na_position="last",
        )
        album_rank_df["global_rank"] = album_rank_df.groupby("group_value").cumcount() + 1
        album_rank_df = album_rank_df[
            album_rank_df["global_rank"] <= controls["albums_per_group"]
        ].copy()

    else:
        # Default: Group, then album listeners
        album_rank_df = album_rank_df.sort_values(
            ["group_value", "lfm_album_listeners", "album_title"],
            ascending=[True, False, True],
            na_position="last",
        )
        album_rank_df["group_rank"] = album_rank_df.groupby("group_value").cumcount() + 1
        album_rank_df = album_rank_df[
            album_rank_df["group_rank"] <= controls["albums_per_group"]
        ].copy()

    keep_keys = album_rank_df[["group_value", "release_group_mbid", "tmdb_id"]].copy()

    out = album_horizon_df.merge(
        keep_keys,
        on=["group_value", "release_group_mbid", "tmdb_id"],
        how="inner",
        validate="m:1",
    )

    return out

def get_auto_horizon_band_count(
    horizon_df: pd.DataFrame,
    value_col: str,
    max_bands: int = 6,
) -> int:
    """
    Choose a sensible number of horizon bands automatically based on the
    robust amplitude of the centered data.
    """
    if horizon_df.empty:
        return 3

    scale_max = horizon_df[value_col].abs().quantile(0.95)
    if pd.isna(scale_max) or scale_max <= 0:
        return 3

    q75 = horizon_df[value_col].abs().quantile(0.75)
    if pd.isna(q75) or q75 <= 0:
        return 3

    ratio = scale_max / q75 if q75 > 0 else 1.0

    if ratio >= 3.5:
        return min(6, max_bands)
    if ratio >= 2.7:
        return min(5, max_bands)
    if ratio >= 1.9:
        return min(4, max_bands)
    return 3

def build_folded_horizon_bands(
    horizon_df: pd.DataFrame,
    value_col: str,
    bands: int = 3,
) -> pd.DataFrame:
    """
    Convert centered album trajectories into horizon bands.

    At any x-position, a point contributes to either the positive bands
    or the negative bands, never both.
    """
    if horizon_df.empty:
        return pd.DataFrame()

    df = horizon_df.copy()

    scale_max = df[value_col].abs().max()
    if pd.isna(scale_max) or scale_max <= 0:
        scale_max = 1e-6

    df["clipped_value"] = df[value_col]
    df["pos_component"] = df["clipped_value"].clip(lower=0)
    df["neg_component"] = (-df["clipped_value"]).clip(lower=0)

    band_height = scale_max / bands
    rows = []

    keep_cols = [
        "group_value",
        "release_group_mbid",
        "tmdb_id",
        "album_title",
        "film_title",
        "film_genres",
        "album_genres_display",
        "composer_primary_clean",
        "label_names",
        "lfm_album_listeners",
        "pos_bin_norm",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]

    for band_idx in range(bands):
        lower = band_idx * band_height
        upper = (band_idx + 1) * band_height

        pos_mag = df["pos_component"].clip(lower=lower, upper=upper) - lower
        pos_mag = pos_mag.clip(lower=0)

        neg_mag = df["neg_component"].clip(lower=lower, upper=upper) - lower
        neg_mag = neg_mag.clip(lower=0)

        pos_mask = pos_mag > 0
        if pos_mask.any():
            pos_df = df.loc[pos_mask, keep_cols].copy()
            pos_df["band_value"] = pos_mag.loc[pos_mask] / band_height
            pos_df["band"] = band_idx
            pos_df["sign"] = "pos"
            rows.append(pos_df)

        neg_mask = neg_mag > 0
        if neg_mask.any():
            neg_df = df.loc[neg_mask, keep_cols].copy()
            neg_df["band_value"] = neg_mag.loc[neg_mask] / band_height
            neg_df["band"] = band_idx
            neg_df["sign"] = "neg"
            rows.append(neg_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

def split_contiguous_runs(
    df: pd.DataFrame,
    value_col: str,
) -> list[pd.DataFrame]:
    """
    Split a dataframe into contiguous runs where value_col > 0.
    """
    if df.empty:
        return []

    working = df.sort_values("pos_bin_norm").copy()
    mask = working[value_col] > 0
    if not mask.any():
        return []

    working = working.loc[mask].copy()
    if working.empty:
        return []

    gap = working["pos_bin_norm"].diff().fillna(0)
    run_id = (gap > 0.02).cumsum()

    return [g.copy() for _, g in working.groupby(run_id)]

def wrap_tick_label(text: str, width: int = 42, max_lines: int = 2) -> str:
    """Wrap a long y-axis label to at most two lines for display."""
    if not text:
        return ""

    wrapped = textwrap.wrap(str(text), width=width)
    if len(wrapped) <= max_lines:
        return "<br>".join(wrapped)

    kept = wrapped[:max_lines]
    kept[-1] = kept[-1].rstrip(" .,;-") + "…"
    return "<br>".join(kept)

def create_album_horizon_chart(
    band_df: pd.DataFrame,
    controls: dict,
    band_steps: int,
) -> go.Figure:
    """
    Render a homework-style horizon chart:
    - positive bands rise from the bottom of the row rectangle
    - negative bands descend from the top of the SAME row rectangle
    - both signs occupy the same album row area
    - grouped blocks get headers and separator lines
    - row-level hover is handled by invisible hover rails, not the polygons
    """
    fig = go.Figure()

    if band_df.empty:
        fig.update_layout(
            height=420,
            template="plotly_dark",
            margin=dict(l=240, r=40, t=40, b=40),
        )
        return fig

    pos_palette = ["#cfefff", "#a9dcff", "#74bdf2", "#3f8fd8", "#1f5fb8", "#123c84"]
    neg_palette = ["#ffe082", "#ffc247", "#ff9f1c", "#f46d43", "#d73027", "#8f1d1d"]

    pos_colors = [pos_palette[min(i, len(pos_palette) - 1)] for i in range(band_steps)]
    neg_colors = [neg_palette[min(i, len(neg_palette) - 1)] for i in range(band_steps)]

    row_cols = [
        "group_value",
        "release_group_mbid",
        "tmdb_id",
        "album_title",
        "film_title",
        "film_genres",
        "album_genres_display",
        "composer_primary_clean",
        "label_names",
        "lfm_album_listeners",
    ]
    row_cols = [col for col in row_cols if col in band_df.columns]

    row_df = (
        band_df[row_cols]
        .drop_duplicates()
        .copy()
    )

    def _fmt(val):
        if pd.isna(val):
            return ""
        return str(val)

    for col in [
        "film_genres",
        "album_genres_display",
        "composer_primary_clean",
        "label_names",
    ]:
        if col in row_df.columns:
            row_df[col] = row_df[col].apply(_fmt)

    sort_mode = controls["horizon_sort"]
    if sort_mode == "Group, then album title":
        row_df = row_df.sort_values(
            ["group_value", "album_title", "lfm_album_listeners"],
            ascending=[True, True, False],
            na_position="last",
        )
    elif sort_mode == "Album title (global)":
        row_df = row_df.sort_values(
            ["album_title", "lfm_album_listeners"],
            ascending=[True, False],
            na_position="last",
        )
    elif sort_mode == "Album listeners (global)":
        row_df = row_df.sort_values(
            ["lfm_album_listeners", "album_title"],
            ascending=[False, True],
            na_position="last",
        )
    else:
        row_df = row_df.sort_values(
            ["group_value", "lfm_album_listeners", "album_title"],
            ascending=[True, False, True],
            na_position="last",
        )

    row_df = row_df.reset_index(drop=True)

    row_spacing = 1.35
    group_gap = 0.9

    row_positions = []
    group_header_positions = []
    current_y = (len(row_df) - 1) * row_spacing

    prev_group = None
    for _, row in row_df.iterrows():
        group_value = row["group_value"]

        if prev_group is not None and group_value != prev_group:
            current_y -= group_gap

        row_positions.append(current_y)

        if group_value != prev_group:
            group_header_positions.append((group_value, current_y + 0.52))

        current_y -= row_spacing
        prev_group = group_value

    row_df["row_index"] = row_positions

    band_df = band_df.merge(
        row_df[["group_value", "release_group_mbid", "tmdb_id", "row_index"]],
        on=["group_value", "release_group_mbid", "tmdb_id"],
        how="left",
        validate="m:1",
    )

    row_height = 0.92
    row_bottom_offset = row_height / 2
    row_top_offset = row_height / 2

    for band_idx in range(band_steps):
        band_slice = band_df[band_df["band"] == band_idx].copy()
        if band_slice.empty:
            continue

        pos_slice = band_slice[band_slice["sign"] == "pos"].copy()
        for _, gdf in pos_slice.groupby("release_group_mbid", dropna=False):
            gdf = gdf.sort_values("pos_bin_norm").copy()

            for run_df in split_contiguous_runs(gdf, "band_value"):
                row_y = run_df["row_index"].iloc[0]

                row_bottom = row_y - row_bottom_offset
                lower = pd.Series(row_bottom, index=run_df.index)
                upper = row_bottom + run_df["band_value"] * row_height

                fig.add_trace(
                    go.Scatter(
                        x=list(run_df["pos_bin_norm"]) + list(run_df["pos_bin_norm"])[::-1],
                        y=list(upper) + list(lower)[::-1],
                        fill="toself",
                        fillcolor=pos_colors[band_idx],
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        neg_slice = band_slice[band_slice["sign"] == "neg"].copy()
        for _, gdf in neg_slice.groupby("release_group_mbid", dropna=False):
            gdf = gdf.sort_values("pos_bin_norm").copy()

            for run_df in split_contiguous_runs(gdf, "band_value"):
                row_y = run_df["row_index"].iloc[0]

                row_top = row_y + row_top_offset
                upper = pd.Series(row_top, index=run_df.index)
                lower = row_top - run_df["band_value"] * row_height

                fig.add_trace(
                    go.Scatter(
                        x=list(run_df["pos_bin_norm"]) + list(run_df["pos_bin_norm"])[::-1],
                        y=list(upper) + list(lower)[::-1],
                        fill="toself",
                        fillcolor=neg_colors[band_idx],
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Stable row-level hover rails across the full x-range.
    compare_label = controls["comparison_dimension"].rstrip("s")

    for _, row in row_df.iterrows():
        hover_x = np.linspace(0, 1, 25)
        hover_customdata = np.array(
            [
                [
                    str(row.get("album_title", "")),
                    str(row.get("group_value", "")),
                    str(row.get("film_title", "")),
                    str(row.get("film_genres", "")),
                    str(row.get("album_genres_display", "")),
                    str(row.get("composer_primary_clean", "")),
                    str(row.get("label_names", "")),
                    row.get("lfm_album_listeners", np.nan),
                ]
            ] * len(hover_x)
        )

        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=[row["row_index"]] * len(hover_x),
                mode="markers",
                marker=dict(
                    size=16,
                    color="rgba(0,0,0,0)",
                ),
                customdata=hover_customdata,
                hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        + f"{compare_label}: "
                        + "%{customdata[1]}<br>"
                        + "Film: %{customdata[2]}<br>"
                        + "Film genres: %{customdata[3]}<br>"
                        + "Album genres: %{customdata[4]}<br>"
                        + "Composer(s): %{customdata[5]}<br>"
                        + "Label(s): %{customdata[6]}<br>"
                        + "Album listeners: %{customdata[7]:,.0f}<br>"
                        + "Normalized position: %{x:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    tickvals = row_df["row_index"].tolist()
    ticktext = [
        wrap_tick_label(str(row["album_title"]), width=42, max_lines=2)
        for _, row in row_df.iterrows()
    ]

    subtitle = (
        "Album-centered rows emphasize each album's internal shape."
        if controls["horizon_normalization"] == "Album-centered shape"
        else "Group-centered rows emphasize deviation from the selected group's average profile."
    )

    annotations = [
        dict(
            text=subtitle,
            x=0,
            xref="paper",
            y=1.05,
            yref="paper",
            showarrow=False,
            xanchor="left",
            font=dict(size=12),
        )
    ]

    for group_value, y_pos in group_header_positions:
        annotations.append(
            dict(
                text=f"<b>{group_value}</b>",
                x=0,
                xref="paper",
                y=y_pos,
                yref="y",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=12),
            )
        )

    group_shapes = []
    for i in range(1, len(row_df)):
        curr_group = row_df.iloc[i]["group_value"]
        prev_group = row_df.iloc[i - 1]["group_value"]

        if curr_group != prev_group:
            boundary_y = (row_df.iloc[i]["row_index"] + row_df.iloc[i - 1]["row_index"]) / 2
            group_shapes.append(
                dict(
                    type="line",
                    xref="paper",
                    yref="y",
                    x0=0,
                    x1=1,
                    y0=boundary_y,
                    y1=boundary_y,
                    line=dict(
                        color="rgba(255,255,255,0.10)",
                        width=1,
                    ),
                )
            )

    fig.update_layout(
        height=max(840, int(42 * len(row_df) + 140)),
        template="plotly_dark",
        margin=dict(l=250, r=40, t=60, b=40),
        hovermode="closest",
        xaxis=dict(
            title="Normalized Track Position",
            tickformat=".1f",
            range=[0, 1],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Albums",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        annotations=annotations,
        shapes=group_shapes,
    )

    return fig

def build_horizon_takeaway(
    album_horizon_df: pd.DataFrame,
    controls: dict,
) -> str:
    """Build a reactive narrative takeaway for the album horizon chart."""
    if album_horizon_df.empty:
        return "No album-level horizon data remains under the current settings."

    latest_positive_df = (
        album_horizon_df[album_horizon_df["centered_value"] > 0]
        .sort_values(["pos_bin_norm", "centered_value"], ascending=[False, False])
    )
    deepest_negative_df = (
        album_horizon_df.sort_values(["centered_value", "pos_bin_norm"], ascending=[True, True])
    )

    latest_positive = latest_positive_df.iloc[0]
    deepest_negative = deepest_negative_df.iloc[0]

    if controls["horizon_normalization"] == "Album-centered shape":
        return (
            f"In the album-centered horizon view, {latest_positive['album_title']} shows the latest strong positive crest "
            f"within its own sequence shape near normalized position {latest_positive['pos_bin_norm']:.2f}, while "
            f"{deepest_negative['album_title']} posts the deepest relative trough ({deepest_negative['centered_value']:.3f})."
        )

    return (
        f"In the group-centered horizon view, {latest_positive['album_title']} diverges latest and most positively from its "
        f"{latest_positive['group_value']} reference profile near normalized position {latest_positive['pos_bin_norm']:.2f}, "
        f"while {deepest_negative['album_title']} falls furthest below its group baseline ({deepest_negative['centered_value']:.3f})."
    )

def build_horizon_insight_cards(
    album_horizon_df: pd.DataFrame,
    controls: dict,
) -> list[tuple[str, str, str]]:
    """Build top-level insight cards for the horizon chart."""
    if album_horizon_df.empty:
        return [
            ("Latest Positive Breakaway", "None", "No album-level horizon data available."),
            ("Deepest Negative Trough", "None", "No album-level horizon data available."),
            ("Most Volatile Album", "None", "No album-level horizon data available."),
        ]

    latest_positive_df = (
        album_horizon_df[album_horizon_df["centered_value"] > 0]
        .sort_values(["pos_bin_norm", "centered_value"], ascending=[False, False])
    )

    deepest_negative_df = (
        album_horizon_df.sort_values(["centered_value", "pos_bin_norm"], ascending=[True, True])
    )

    volatility_df = (
        album_horizon_df.groupby(
            ["group_value", "release_group_mbid", "album_title"],
            as_index=False,
        )["centered_value"]
        .agg(
            peak_value="max",
            trough_value="min",
        )
    )
    volatility_df["swing"] = volatility_df["peak_value"] - volatility_df["trough_value"]

    latest_positive = latest_positive_df.iloc[0]
    deepest_negative = deepest_negative_df.iloc[0]
    most_volatile = volatility_df.sort_values(
        ["swing", "album_title"],
        ascending=[False, True],
    ).iloc[0]

    return [
        (
            "Latest Positive Breakaway",
            str(latest_positive["album_title"]),
            f"{latest_positive['group_value']} peaks above its reference profile near position {latest_positive['pos_bin_norm']:.2f}.",
        ),
        (
            "Deepest Negative Trough",
            str(deepest_negative["album_title"]),
            f"{deepest_negative['group_value']} drops furthest below its reference profile at {deepest_negative['centered_value']:.3f}.",
        ),
        (
            "Most Volatile Album",
            str(most_volatile["album_title"]),
            f"{most_volatile['group_value']} shows the widest within-row swing at {most_volatile['swing']:.3f}.",
        ),
    ]


def render_horizon_insight_cards(
    album_horizon_df: pd.DataFrame,
    controls: dict,
) -> None:
    """Render horizon-specific insight cards."""
    cards = build_horizon_insight_cards(album_horizon_df, controls)
    cols = st.columns(3)

    for i, (title, value, caption) in enumerate(cards):
        with cols[i]:
            st.metric(title, value)
            st.caption(caption)

def build_sequence_insight_cards(
    agg_df: pd.DataFrame,
    controls: dict,
) -> list[tuple[str, str, str]]:
    """Build top-level insight cards for the grouped sequence view."""
    if agg_df.empty:
        return [
            ("Strongest Opening Group", "None", "No sequence data available."),
            ("Strongest Ending Group", "None", "No sequence data available."),
            ("Largest Net Shift", "None", "No sequence data available."),
        ]

    first_bin = agg_df["pos_bin"].min()
    last_bin = agg_df["pos_bin"].max()

    opening_df = (
        agg_df[agg_df["pos_bin"] == first_bin]
        .sort_values(["center_value", "group_value"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ending_df = (
        agg_df[agg_df["pos_bin"] == last_bin]
        .sort_values(["center_value", "group_value"], ascending=[False, True])
        .reset_index(drop=True)
    )

    shift_df = (
        agg_df.groupby("group_value", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "start_value": g.sort_values("pos_bin").iloc[0]["center_value"],
                    "end_value": g.sort_values("pos_bin").iloc[-1]["center_value"],
                    "peak_value": g["center_value"].max(),
                    "trough_value": g["center_value"].min(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    group_labels = agg_df["group_value"].drop_duplicates().reset_index(drop=True)
    shift_df["group_value"] = group_labels
    shift_df["net_shift"] = shift_df["end_value"] - shift_df["start_value"]
    shift_df["abs_shift"] = shift_df["net_shift"].abs()
    shift_df["range_value"] = shift_df["peak_value"] - shift_df["trough_value"]

    opening = opening_df.iloc[0]
    ending = ending_df.iloc[0]
    mover = shift_df.sort_values(
        ["abs_shift", "group_value"],
        ascending=[False, True],
    ).iloc[0]

    direction = "upward" if mover["net_shift"] > 0 else "downward"

    return [
        (
            "Strongest Opening Group",
            str(opening["group_value"]),
            f"Starts highest at {opening['center_value']:.3f} on the opening normalized track position.",
        ),
        (
            "Strongest Ending Group",
            str(ending["group_value"]),
            f"Finishes highest at {ending['center_value']:.3f} on the closing normalized track position.",
        ),
        (
            "Largest Net Shift",
            str(mover["group_value"]),
            f"Moves {direction} by {mover['abs_shift']:.3f} from start to finish.",
        ),
    ]

def render_insight_cards(
    agg_df: pd.DataFrame,
    controls: dict,
) -> None:
    """Render top insight cards."""
    cards = build_sequence_insight_cards(agg_df, controls)
    cols = st.columns(3)

    for i, (title, value, caption) in enumerate(cards):
        with cols[i]:
            st.metric(title, value)
            st.caption(caption)

def build_sequence_takeaway(
    agg_df: pd.DataFrame,
    controls: dict,
) -> str:
    """Build a reactive narrative takeaway for the alluvial ribbon chart."""
    if agg_df.empty:
        return "No sequence data remains under the current settings."

    first_bin = agg_df["pos_bin"].min()
    mid_bin = agg_df["pos_bin"].median()
    last_bin = agg_df["pos_bin"].max()

    opening = (
        agg_df[agg_df["pos_bin"] == first_bin]
        .sort_values(["center_value", "group_value"], ascending=[False, True])
        .iloc[0]
    )
    ending = (
        agg_df[agg_df["pos_bin"] == last_bin]
        .sort_values(["center_value", "group_value"], ascending=[False, True])
        .iloc[0]
    )

    shift_df = (
        agg_df.groupby("group_value", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "start_value": g.sort_values("pos_bin").iloc[0]["center_value"],
                    "end_value": g.sort_values("pos_bin").iloc[-1]["center_value"],
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    group_labels = agg_df["group_value"].drop_duplicates().reset_index(drop=True)
    shift_df["group_value"] = group_labels
    shift_df["net_shift"] = shift_df["end_value"] - shift_df["start_value"]
    shift_df["abs_shift"] = shift_df["net_shift"].abs()

    mover = shift_df.sort_values(
        ["abs_shift", "group_value"],
        ascending=[False, True],
    ).iloc[0]

    direction = "rises" if mover["net_shift"] > 0 else "falls"
    metric_label = get_track_page_display_label(controls["metric"]).lower()

    return (
        f"Within the current filtered comparison set, {opening['group_value']} opens with the highest "
        f"{metric_label} profile ({opening['center_value']:.3f}), while {ending['group_value']} ends highest "
        f"({ending['center_value']:.3f}). The biggest overall mover is {mover['group_value']}, which {direction} "
        f"by {mover['abs_shift']:.3f} from the opening to the closing normalized track position."
    )

def build_filter_context_caption(
    controls: dict,
    visible_groups: int,
    visible_albums: int,
    visible_tracks: int,
    visible_bins: int,
) -> str:
    """Build a compact analysis-scope caption."""
    parts = [
        f"{visible_tracks:,} tracks",
        f"{visible_albums:,} albums",
        f"{visible_groups:,} groups",
        f"{visible_bins:,} normalized position bins",
        f"feature: {get_track_page_display_label(controls['metric']).lower()}",
        f"compare by: {controls['comparison_dimension'].lower()}",
        f"center: {controls['center_stat'].lower()}",
        f"band: {controls['ribbon_range'].lower()}",
    ]

    if controls["min_album_listeners"] > 0:
        parts.append(f"min album listeners {controls['min_album_listeners']:,}")

    parts.append(f"max track position {controls['max_track_position']}")

    return "Current scope: " + " | ".join(parts) + "."

def build_metric_explainer(metric: str) -> str:
    """
    Return a short user-facing explainer for the selected audio feature.
    """
    explainers = {
        "energy": (
            "Energy is a 0–1 measure of intensity and activity. "
            "Higher values indicate more forceful, energetic tracks."
        ),
        "danceability": (
            "Danceability is a 0–1 measure of rhythmic steadiness and groove. "
            "Higher values indicate tracks that feel more danceable."
        ),
        "happiness": (
            "Happiness is a 0–1 mood-brightness measure derived from valence-like data. "
            "Higher values indicate brighter, more positive-sounding tracks."
        ),
        "acousticness": (
            "Acousticness is a 0–1 measure of how acoustic a track sounds. "
            "Higher values indicate less electronic or synthetic texture."
        ),
        "instrumentalness": (
            "Instrumentalness is a 0–1 measure of how likely a track is to be instrumental. "
            "Higher values indicate less vocal presence."
        ),
        "speechiness": (
            "Speechiness is a 0–1 measure of spoken-word content. "
            "Higher values indicate more speech-like vocal texture."
        ),
        "liveness": (
            "Liveness is a 0–1 measure of how live or performance-like a track sounds. "
            "Higher values suggest a more live-recorded feel."
        ),
        "tempo": (
            "Tempo is measured in beats per minute (BPM). "
            "Higher values indicate faster tracks."
        ),
        "loudness": (
            "Loudness is measured in decibels (dB), so values are usually negative. "
            "Values closer to 0 are louder, while more negative values are quieter."
        ),
        "duration_seconds": (
            "Duration is shown in seconds per track. "
            "Higher values indicate longer tracks."
        ),
        "mode": (
            "Mode is a binary harmonic label: 1 = major and 0 = minor. "
            "When the chart uses a mean, values between 0 and 1 represent the share "
            "of tracks in major mode. When the chart uses a median, lines may collapse "
            "to exactly 0 or 1 because the majority mode wins."
        ),
        "key": (
            "Key is encoded numerically from 0 to 11, representing pitch classes "
            "(for example C through B). It is not an ordered performance scale, so "
            "changes reflect tonal movement rather than higher or lower quality."
        ),
    }

    return explainers.get(
        metric,
        "This metric traces how the selected audio feature changes across normalized track position."
    )

def build_sequence_comparison_album_table(
    grouped_df: pd.DataFrame,
    selected_groups: list[str],
) -> pd.DataFrame:
    """Build a simple album table for the currently visible comparison scope."""
    if grouped_df.empty:
        return pd.DataFrame()

    working = grouped_df.copy()
    if selected_groups:
        working = working[working["group_value"].isin(selected_groups)].copy()

    group_keys = ["group_value", "release_group_mbid", "tmdb_id"]
    album_cols = select_unique_existing_columns(
        working,
        group_keys + SEQUENCE_DETAIL_ALBUM_COLS,
    )

    out = working[album_cols].drop_duplicates(subset=group_keys).copy()

    if "lfm_album_listeners" in out.columns:
        out = out.sort_values(
            ["group_value", "lfm_album_listeners"],
            ascending=[True, False],
            na_position="last",
        )

    return out


def build_group_value_column(df: pd.DataFrame, comparison_dimension: str) -> pd.Series:
    """Create group_value column BEFORE track aggregation."""

    if comparison_dimension == "Film genres":
        return df["film_genres"].apply(split_multivalue_cell)


    elif comparison_dimension == "Album genres":
        return df["album_genres_display"].apply(split_multivalue_cell)

    elif comparison_dimension == "Composer":
        return (
            df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    elif comparison_dimension == "Film year":
        out = pd.Series(index=df.index, dtype="object")
        valid_years = pd.to_numeric(df["film_year"], errors="coerce")
        out.loc[valid_years.notna()] = valid_years.loc[valid_years.notna()].astype(int).astype(str)
        return out

    elif comparison_dimension == "Labels":
        return df["label_names"].apply(split_multivalue_cell)

    elif comparison_dimension == "Album cohesion group":
        df = add_album_cohesion_group(df.copy(), metric=AUDIO_FEATURE_OPTIONS[0])
        return df["album_cohesion_group"]

    return pd.Series(dtype=str)

def main() -> None:
    """Render the Track Sequence Breakpoints Explorer."""
    st.set_page_config(
        page_title="Track Sequence Breakpoints Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Sequence Breakpoints Explorer")
    st.caption(
        "Compare how a selected audio feature evolves across normalized track position using multi-group alluvial-style ribbons."
    )

    track_df = load_track_data_explorer_data()
    track_df = add_track_sequence_display_fields(track_df)

    filter_inputs = get_global_filter_inputs(track_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    # First-pass controls need options, so bootstrap with broad visible group options
    bootstrap_controls = {
        "metric": AUDIO_FEATURE_OPTIONS[0],
        "comparison_dimension": "Film genres",
        "selected_groups": [],
        "max_groups": 4,
        "min_albums_per_group": 5,
        "n_bins": 20,
        "smoothing_window": 3,
        "center_stat": "Median",
        "ribbon_range": "Middle 20%",
        "show_ribbon": True,
        "search_text": "",
        "max_track_position": 20,
        "min_album_listeners": 0,
        "audio_only": True,
    }

    bootstrap_filtered_df = filter_track_sequence_df(
        track_df=track_df,
        global_controls=global_controls,
        controls=bootstrap_controls,
    )

    bootstrap_base_df = build_sequence_base_df(
        df=bootstrap_filtered_df,
        metric=bootstrap_controls["metric"],
        n_bins=bootstrap_controls["n_bins"],
    )

    bootstrap_grouped_df = build_grouped_sequence_df(
        base_df=bootstrap_base_df,
        controls=bootstrap_controls,
    )

    initial_group_options = build_group_options(
        df=bootstrap_grouped_df,
        controls=bootstrap_controls,
    )

    controls = get_track_sequence_controls(
        comparison_dimension_options=COMPARISON_DIMENSIONS,
        audio_feature_options=AUDIO_FEATURE_OPTIONS,
    )

    # Apply real filters and rebuild with the selected controls
    filtered_df = filter_track_sequence_df(
        track_df=track_df,
        global_controls=global_controls,
        controls=controls,
    )

    if filtered_df.empty:
        st.warning("No tracks remain under the current filters.")
        return

    base_df = build_sequence_base_df(
        df=filtered_df,
        metric=controls["metric"],
        n_bins=controls["n_bins"],
    )

    if base_df.empty:
        st.warning("No normalized sequence rows remain under the current settings.")
        return

    grouped_df = build_grouped_sequence_df(
        base_df=base_df,
        controls=controls,
    )

    if grouped_df.empty:
        st.warning("No grouped sequence rows remain under the current settings.")
        return

    available_group_options = build_group_options(
        filtered_df.assign(
            group_value=build_group_value_column(filtered_df, controls["comparison_dimension"])
        ),
        controls,
    )

    selected_groups = render_group_selector(
        group_options=available_group_options,
        controls=controls,
    )

    grouped_df = grouped_df[grouped_df["group_value"].isin(selected_groups)].copy()

    if grouped_df.empty:
        st.warning("No groups remain after applying the current selection and group thresholds.")
        return

    agg_df = aggregate_sequence_df(
        grouped_df=grouped_df,
        controls=controls,
    )

    if agg_df.empty:
        st.warning("No aggregated sequence data remains under the current settings.")
        return

    visible_tracks = len(grouped_df)
    visible_albums = grouped_df["release_group_mbid"].nunique()
    visible_groups = agg_df["group_value"].nunique()
    visible_bins = agg_df["pos_bin"].nunique()

    st.caption(
        build_filter_context_caption(
            controls=controls,
            visible_groups=visible_groups,
            visible_albums=visible_albums,
            visible_tracks=visible_tracks,
            visible_bins=visible_bins,
        )
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tracks", f"{visible_tracks:,}")
    c2.metric("Albums", f"{visible_albums:,}")
    c3.metric("Groups", f"{visible_groups:,}")
    c4.metric("Bins", f"{visible_bins:,}")
    c5.metric("Feature", get_track_page_display_label(controls["metric"]))

    st.markdown("### 🧠 Key Insights")
    render_insight_cards(agg_df, controls)

    st.markdown("### Sequence Alluvial Ribbons")
    fig = create_sequence_ribbon_chart(
        agg_df=agg_df,
        controls=controls,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        key="page35_sequence_alluvial_chart",
    )
    st.caption(build_sequence_takeaway(agg_df, controls))
    st.markdown("**What does this feature mean?**")
    st.caption(build_metric_explainer(controls["metric"]))

    if controls["show_horizon"]:
        album_horizon_df = build_album_horizon_df(
            grouped_df=grouped_df,
            controls=controls,
            selected_groups=selected_groups,
        )

        album_horizon_df = filter_top_albums_for_horizon(
            album_horizon_df=album_horizon_df,
            controls=controls,
        )

        auto_band_count = get_auto_horizon_band_count(
            horizon_df=album_horizon_df,
            value_col="centered_value",
        )

        band_df = build_folded_horizon_bands(
            horizon_df=album_horizon_df,
            value_col="centered_value",
            bands=auto_band_count,
        )

        st.markdown("### Horizon View Insights")
        render_horizon_insight_cards(
            album_horizon_df=album_horizon_df,
            controls=controls,
        )

        st.markdown("### Album Horizon Chart")
        horizon_fig = create_album_horizon_chart(
            band_df=band_df,
            controls=controls,
            band_steps=auto_band_count,
        )
        st.plotly_chart(
            horizon_fig,
            width="stretch",
            key="page35_album_horizon_chart",
        )
        st.caption(build_horizon_takeaway(album_horizon_df, controls))

        with st.expander("Inspect horizon chart data"):
            debug_album_options = sorted(album_horizon_df["album_title"].dropna().unique().tolist())

            debug_album = st.selectbox(
                "Choose an album to inspect",
                options=debug_album_options,
                index=0,
                key="page35_horizon_debug_album",
            )

            st.caption(
                "Horizon rows use one canonical group assignment per album for readability. "
                "The detail tables below preserve the full multi-genre, multi-composer, and label metadata."
            )

            st.markdown("**Album-level centered horizon input**")
            horizon_input_cols = [
                "group_value",
                "album_title",
                "film_title",
                "film_genres",
                "album_genres_display",
                "composer_primary_clean",
                "label_names",
                "lfm_album_listeners",
                "pos_bin_norm",
                "raw_value",
                "baseline",
                "centered_value",
            ]
            horizon_input_cols = [col for col in horizon_input_cols if col in album_horizon_df.columns]

            st.dataframe(
                album_horizon_df[
                    album_horizon_df["album_title"] == debug_album
                    ][horizon_input_cols].sort_values("pos_bin_norm"),
                width="stretch",
                hide_index=True,
            )

            st.markdown("**Folded horizon bands**")
            folded_band_cols = [
                "group_value",
                "album_title",
                "film_title",
                "pos_bin_norm",
                "sign",
                "band",
                "band_value",
            ]
            folded_band_cols = [col for col in folded_band_cols if col in band_df.columns]

            st.dataframe(
                band_df[
                    band_df["album_title"] == debug_album
                    ][folded_band_cols].sort_values(["band", "sign", "pos_bin_norm"]),
                width="stretch",
                hide_index=True,
            )

    st.divider()

    st.markdown("### Visible Album Comparison Scope")
    st.caption(
        "These albums contribute to the currently visible ribbon comparison."
    )

    album_table = build_sequence_comparison_album_table(
        grouped_df=grouped_df,
        selected_groups=selected_groups,
    )

    if album_table.empty:
        st.info("No albums remain in the current comparison scope.")
    else:
        st.dataframe(
            rename_track_page_columns_for_display(album_table),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()