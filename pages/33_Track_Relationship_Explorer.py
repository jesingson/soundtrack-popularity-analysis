from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_relationship_controls,
)
from app.app_data import load_track_data_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_film_year_bucket,
    add_standard_multivalue_groups,
    add_key_mode_label,
    get_clean_composer_options,
    get_global_filter_inputs,
    rename_and_dedupe_for_display,
    get_track_numeric_options,
    get_track_group_options,
    get_track_page_display_label,
)
from app.ui import apply_app_styles

TOOLTIP_COLUMNS = [
    "film_title",
    "album_title",
    "track_title",
    "track_number",
    "track_position_bucket",
    "relative_track_position",
    "track_type",
    "composer_primary_clean",
    "film_year",
    "film_genres",
    "album_genres_display",
    "award_category",
    "mode_label",
    "key_label",
    "key_mode_label",
    "lfm_track_listeners",
    "lfm_track_playcount",
    "track_share_of_album_listeners",
    "track_share_of_album_playcount",
    "spotify_popularity",
    "energy",
    "danceability",
    "happiness",
    "instrumentalness",
    "acousticness",
    "liveness",
    "speechiness",
    "tempo",
    "loudness",
    "duration_seconds",
]


def add_track_relationship_display_fields(track_df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped display fields used by the Track Relationship Explorer."""
    df = add_standard_multivalue_groups(track_df)
    df = add_film_year_bucket(df)
    df = add_key_mode_label(df)

    if "composer_primary_clean" in df.columns:
        df["composer_primary_clean"] = (
            df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "mode" in df.columns:
        df["mode_label"] = df["mode"].map({1.0: "Major", 0.0: "Minor"}).fillna("Unknown")

    if "key_label" in df.columns:
        df["key_label"] = (
            df["key_label"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    if "is_instrumental" in df.columns:
        df["track_type"] = np.where(
            pd.to_numeric(df["is_instrumental"], errors="coerce").fillna(0).astype(int) == 1,
            "Instrumental",
            "Vocal / Mixed",
        )
    elif "instrumentalness" in df.columns:
        df["track_type"] = np.where(
            df["instrumentalness"] >= 0.70,
            "Instrumental",
            "Vocal / Mixed",
        )
    else:
        df["track_type"] = "Unknown"

    return df


def filter_track_relationship_df(
    track_df: pd.DataFrame,
    global_controls: dict,
    controls: dict,
) -> pd.DataFrame:
    """Apply shared global filters plus Page 33-specific filters."""
    merged_controls = {
        **global_controls,
        "selected_composers": controls.get("selected_composers", []),
        "search_text": controls.get("search_text", ""),
    }

    filtered = filter_dataset(track_df, merged_controls).copy()

    if "track_number" in filtered.columns:
        filtered = filtered[
            filtered["track_number"] <= controls["max_track_position"]
        ].copy()

    if "lfm_album_listeners" in filtered.columns:
        filtered = filtered[
            filtered["lfm_album_listeners"].fillna(0) >= controls["min_album_listeners"]
        ].copy()

    if controls.get("audio_only", False):
        required_audio_cols = [
            col for col in [
                "energy",
                "danceability",
                "happiness",
                "instrumentalness",
                "tempo",
                "loudness",
            ]
            if col in filtered.columns
        ]
        if required_audio_cols:
            filtered = filtered.dropna(subset=required_audio_cols).copy()

    return filtered


def build_relationship_context_caption(controls: dict) -> str:
    """Build a natural-language caption describing the current relationship view."""
    base = (
        f"Comparing {get_track_page_display_label(controls['x_col'])} vs "
        f"{get_track_page_display_label(controls['y_col'])}"
    )

    extras = []
    extras.append(f"view = {controls['chart_mode']}")

    if controls["color_col"] != "None":
        extras.append(f"color = {get_track_page_display_label(controls['color_col'])}")

    if controls["size_col"] != "None":
        extras.append(f"bubble size = {get_track_page_display_label(controls['size_col'])}")

    if controls["show_median_lines"]:
        extras.append("median reference lines shown")

    if controls["transform_x"] != "None" or controls["transform_y"] != "None":
        transform_bits = []
        if controls["transform_x"] != "None":
            transform_bits.append(f"X: {controls['transform_x']}")
        if controls["transform_y"] != "None":
            transform_bits.append(f"Y: {controls['transform_y']}")
        extras.append("transforms = " + ", ".join(transform_bits))

    if controls["apply_jitter"]:
        extras.append(f"jitter = {controls['jitter_strength']:.3f}")

    extras.append(
        "fitted line shown" if controls["show_trendline"] else "fitted line hidden"
    )

    return base + (" | " + " | ".join(extras) if extras else "") + "."


def build_freeform_scatter_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    size_col: str | None = None,
    transform_x: str = "None",
    transform_y: str = "None",
    apply_jitter: bool = False,
    jitter_strength: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Build freeform scatterplot data for arbitrary track-level numeric X and Y columns."""
    required_cols = [x_col, y_col]

    if size_col and size_col != "None":
        required_cols.append(size_col)

    if color_col and color_col != "None":
        required_cols.append(color_col)

    metadata_cols = [col for col in TOOLTIP_COLUMNS if col in df.columns]
    required_cols.extend(metadata_cols)
    required_cols = list(dict.fromkeys(required_cols))

    plot_df = df[required_cols].copy().dropna(subset=[x_col, y_col])

    if len(plot_df) < 2:
        raise ValueError("At least two non-null rows are required to build the scatterplot.")

    plot_df["x_raw"] = pd.to_numeric(plot_df[x_col], errors="coerce")
    plot_df["y_raw"] = pd.to_numeric(plot_df[y_col], errors="coerce")

    if size_col and size_col != "None":
        plot_df["size_raw"] = pd.to_numeric(plot_df[size_col], errors="coerce")
    else:
        plot_df["size_raw"] = np.nan

    zero_means_missing_cols = {"film_budget", "film_revenue"}

    if x_col in zero_means_missing_cols:
        plot_df.loc[plot_df["x_raw"] <= 0, "x_raw"] = np.nan

    if y_col in zero_means_missing_cols:
        plot_df.loc[plot_df["y_raw"] <= 0, "y_raw"] = np.nan

    if size_col in zero_means_missing_cols:
        plot_df.loc[plot_df["size_raw"] <= 0, "size_raw"] = np.nan

    if transform_x == "Log1p":
        if (plot_df["x_raw"] < 0).any():
            raise ValueError(f"X-axis field '{x_col}' contains negative values and cannot use Log1p.")
        plot_df["x_value"] = np.log1p(plot_df["x_raw"])
    else:
        plot_df["x_value"] = plot_df["x_raw"]

    if transform_y == "Log1p":
        if (plot_df["y_raw"] < 0).any():
            raise ValueError(f"Y-axis field '{y_col}' contains negative values and cannot use Log1p.")
        plot_df["y_value"] = np.log1p(plot_df["y_raw"])
    else:
        plot_df["y_value"] = plot_df["y_raw"]

    plot_df = plot_df.dropna(subset=["x_value", "y_value"]).copy()

    if len(plot_df) < 2:
        raise ValueError("At least two valid rows remain after applying transforms.")

    x = plot_df["x_value"].to_numpy()
    y = plot_df["y_value"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    x_min, x_max = float(x.min()), float(x.max())

    line_df = pd.DataFrame({
        "x_plot": [x_min, x_max],
        "y_plot": [slope * x_min + intercept, slope * x_max + intercept],
    })

    pearson_r = float(plot_df["x_value"].corr(plot_df["y_value"], method="pearson"))

    plot_df["x_plot"] = plot_df["x_value"]
    plot_df["y_plot"] = plot_df["y_value"]

    if apply_jitter and jitter_strength > 0:
        x_range = float(plot_df["x_value"].max() - plot_df["x_value"].min())
        y_range = float(plot_df["y_value"].max() - plot_df["y_value"].min())

        x_scale = x_range if x_range > 0 else 1.0
        y_scale = y_range if y_range > 0 else 1.0

        rng = np.random.default_rng(42)
        plot_df["x_plot"] = plot_df["x_value"] + rng.normal(
            loc=0.0,
            scale=x_scale * jitter_strength,
            size=len(plot_df),
        )
        plot_df["y_plot"] = plot_df["y_value"] + rng.normal(
            loc=0.0,
            scale=y_scale * jitter_strength,
            size=len(plot_df),
        )
    if size_col and size_col != "None":
        size_series = plot_df["size_raw"].clip(lower=0)

        if size_series.notna().sum() > 0:
            upper_cap = float(size_series.quantile(0.98))
            if upper_cap <= 0:
                upper_cap = float(size_series.max())
        else:
            upper_cap = 1.0

        if not np.isfinite(upper_cap) or upper_cap <= 0:
            upper_cap = 1.0

        plot_df["size_scaled_raw"] = size_series.clip(upper=upper_cap)
    else:
        plot_df["size_scaled_raw"] = np.nan

    x_median = float(plot_df["x_value"].median())
    y_median = float(plot_df["y_value"].median())

    x_display = get_track_page_display_label(x_col)
    y_display = get_track_page_display_label(y_col)

    metrics = {
        "x_col": x_col,
        "y_col": y_col,
        "rows_used": len(plot_df),
        "pearson_r": pearson_r,
        "r_squared": pearson_r ** 2,
        "color_col": color_col,
        "transform_x": transform_x,
        "transform_y": transform_y,
        "apply_jitter": apply_jitter,
        "jitter_strength": jitter_strength,
        "x_axis_title": f"log1p({x_display})" if transform_x == "Log1p" else x_display,
        "y_axis_title": f"log1p({y_display})" if transform_y == "Log1p" else y_display,
        "slope": slope,
        "direction": (
            "Positive" if pearson_r > 0.05
            else "Negative" if pearson_r < -0.05
            else "Near-zero"
        ),
        "size_col": size_col,
        "size_scale_max": float(plot_df["size_scaled_raw"].max()) if size_col and size_col != "None" else None,
        "x_median": x_median,
        "y_median": y_median,
    }

    return plot_df, line_df, metrics


def build_relationship_insight_summary(metrics: dict) -> list[tuple[str, str, str]]:
    """Build key insight cards for the relationship explorer."""
    pearson_r = float(metrics["pearson_r"])

    color_value = (
        get_track_page_display_label(metrics["color_col"])
        if metrics["color_col"] and metrics["color_col"] != "None"
        else "None"
    )

    direction = (
        "Positive" if pearson_r > 0.05
        else "Negative" if pearson_r < -0.05
        else "Near-zero"
    )

    return [
        ("Pearson r", f"{pearson_r:.3f}", f"{direction} association"),
        ("Rows Used", f"{metrics['rows_used']:,}", "Tracks in the plotted view"),
        ("Color Grouping", color_value, "Categorical overlay"),
    ]


def render_relationship_insight_cards(metrics: dict) -> None:
    """Render educational insight cards for the relationship view."""
    st.markdown("### 🧠 Key Insights")

    pearson_r = float(metrics["pearson_r"])
    r2 = float(metrics["r_squared"])

    strength = (
        "Strong" if abs(pearson_r) >= 0.5
        else "Moderate" if abs(pearson_r) >= 0.2
        else "Weak"
    )

    direction = (
        "Positive" if pearson_r > 0.05
        else "Negative" if pearson_r < -0.05
        else "Near-zero"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Relationship", f"{strength} {direction}")
        st.caption(f"Pearson r = {pearson_r:.3f}")

    with col2:
        st.metric("Explained Variance", f"{r2:.3f}")
        st.caption("R² = share of variation explained by a linear fit.")

    with col3:
        st.metric("Tracks", f"{metrics['rows_used']:,}")
        st.caption("Rows contributing to the visible scatter.")


def build_relationship_supporting_insight(metrics: dict, controls: dict) -> str:
    """Build an educational interpretation of the scatterplot."""
    pearson_r = float(metrics["pearson_r"])

    x_label = get_track_page_display_label(metrics["x_col"]).lower()
    y_label = get_track_page_display_label(metrics["y_col"]).lower()

    strength = (
        "strong" if abs(pearson_r) >= 0.5
        else "moderate" if abs(pearson_r) >= 0.2
        else "weak"
    )

    direction = (
        "positive" if pearson_r > 0.05
        else "negative" if pearson_r < -0.05
        else "near-zero"
    )

    base = (
        f"💡 This scatter shows a {strength} {direction} relationship between "
        f"{x_label} and {y_label} (r = {pearson_r:.3f}). "
    )

    if abs(pearson_r) < 0.1:
        base += (
            "There is little consistent linear relationship — the variables largely vary independently."
        )
    elif pearson_r > 0:
        base += (
            f"As {x_label} increases, {y_label} tends to increase as well, "
            "though the spread shows how consistent that relationship is."
        )
    else:
        base += (
            f"As {x_label} increases, {y_label} tends to decrease, "
            "with variation indicating how strong or noisy the relationship is."
        )

    if controls["color_col"] != "None":
        color_label = get_track_page_display_label(controls["color_col"]).lower()
        base += (
            f" Coloring by {color_label} helps reveal whether distinct subgroups follow different patterns."
        )

    return base

def build_relationship_mode_supporting_insight(
    metrics: dict,
    controls: dict,
) -> str:
    """Build a short interpretation tailored to the selected relationship chart mode."""
    chart_mode = controls.get("chart_mode", "Scatter")

    x_label = get_track_page_display_label(metrics["x_col"]).lower()
    y_label = get_track_page_display_label(metrics["y_col"]).lower()

    if chart_mode == "Density Heatmap":
        return (
            f"💡 This density view shows where tracks are most concentrated in the "
            f"{x_label} × {y_label} space. Darker or more saturated cells indicate "
            f"regions where many tracks overlap, which is especially useful when the "
            f"scatterplot becomes too dense to read point-by-point."
        )

    if chart_mode == "Quantile Heatmap":
        stat = controls.get("quantile_stat", "Mean Y")
        return (
            f"💡 This heatmap summarizes {stat.lower()} across the visible "
            f"{x_label} × {y_label} space. It helps reveal whether stronger values "
            f"of {y_label} cluster in particular regions rather than following a single "
            f"clean linear relationship."
        )

    return build_relationship_supporting_insight(metrics, controls)

def build_relationship_chart_insight(
    plot_df: pd.DataFrame,
    metrics: dict,
    controls: dict,
) -> str:
    """Build a data-reactive insight tied to the active relationship chart mode."""
    chart_mode = controls.get("chart_mode", "Scatter")
    x_label = get_track_page_display_label(metrics["x_col"]).lower()
    y_label = get_track_page_display_label(metrics["y_col"]).lower()

    rows_used = int(metrics.get("rows_used", 0))
    pearson_r = float(metrics.get("pearson_r", np.nan))

    if plot_df.empty:
        return "No visible relationship insight is available under the current filters."

    if chart_mode == "Density Heatmap":
        x_q10 = float(plot_df["x_value"].quantile(0.10))
        x_q90 = float(plot_df["x_value"].quantile(0.90))
        y_q10 = float(plot_df["y_value"].quantile(0.10))
        y_q90 = float(plot_df["y_value"].quantile(0.90))

        x_iqr = float(plot_df["x_value"].quantile(0.75) - plot_df["x_value"].quantile(0.25))
        y_iqr = float(plot_df["y_value"].quantile(0.75) - plot_df["y_value"].quantile(0.25))

        if abs(pearson_r) >= 0.5:
            trend_phrase = "a strong overall linear structure"
        elif abs(pearson_r) >= 0.2:
            trend_phrase = "a moderate overall linear structure"
        else:
            trend_phrase = "only a weak overall linear structure"

        return (
            f"💡 Under the current filters, most visible tracks fall within the central "
            f"{x_label} range of about {x_q10:.2f} to {x_q90:.2f} and the central "
            f"{y_label} range of about {y_q10:.2f} to {y_q90:.2f}. The cloud shows "
            f"{trend_phrase}, with an interquartile spread of {x_iqr:.2f} on {x_label} "
            f"and {y_iqr:.2f} on {y_label}, which helps indicate whether the visible "
            f"distribution is tightly concentrated or broadly diffuse."
        )

    if chart_mode == "Quantile Heatmap":
        bins = int(controls.get("heatmap_bins", 30))
        quantile_stat = controls.get("quantile_stat", "Mean Y")

        temp_df = plot_df[["x_value", "y_value", "y_raw"]].copy()
        temp_df["x_bin"] = pd.cut(temp_df["x_value"], bins=bins, duplicates="drop")
        temp_df["y_bin"] = pd.cut(temp_df["y_value"], bins=bins, duplicates="drop")

        grouped = (
            temp_df.dropna(subset=["x_bin", "y_bin"])
            .groupby(["x_bin", "y_bin"], observed=False)
            .agg(
                n=("y_raw", "size"),
                mean_y=("y_raw", "mean"),
                median_y=("y_raw", "median"),
            )
            .reset_index()
        )

        if grouped.empty:
            return "💡 The visible data is too sparse to form stable heatmap regions under the current binning."

        if quantile_stat == "Median Y":
            value_col = "median_y"
            stat_label = f"median {y_label}"
        elif quantile_stat == "Count":
            value_col = "n"
            stat_label = "track count"
        else:
            value_col = "mean_y"
            stat_label = f"mean {y_label}"

        top_cell = grouped.sort_values([value_col, "n"], ascending=[False, False]).iloc[0]
        coverage = float(top_cell["n"]) / rows_used if rows_used > 0 else np.nan

        x_bin_label = str(top_cell["x_bin"])
        y_bin_label = str(top_cell["y_bin"])

        if quantile_stat == "Count":
            return (
                f"💡 The densest visible region contains {int(top_cell['n']):,} tracks "
                f"({coverage:.1%} of the plotted sample), concentrated around "
                f"{x_label} bin {x_bin_label} and {y_label} bin {y_bin_label}. "
                f"That suggests the relationship is anchored by one especially common region "
                f"rather than being evenly distributed across the space."
            )

        return (
            f"💡 The strongest visible heatmap region is in the {x_label} bin {x_bin_label} "
            f"and {y_label} bin {y_bin_label}, where the cell-level {stat_label} reaches "
            f"{float(top_cell[value_col]):.2f} across {int(top_cell['n']):,} tracks "
            f"({coverage:.1%} of the plotted sample). That indicates the highest observed "
            f"outcomes are concentrated in a specific pocket of the visible relationship space "
            f"rather than spread evenly across all tracks."
        )

    x_q10 = float(plot_df["x_value"].quantile(0.10))
    x_q90 = float(plot_df["x_value"].quantile(0.90))
    y_q10 = float(plot_df["y_value"].quantile(0.10))
    y_q90 = float(plot_df["y_value"].quantile(0.90))

    if pearson_r > 0.2:
        direction_phrase = "higher visible X values tend to align with higher visible Y values"
    elif pearson_r < -0.2:
        direction_phrase = "higher visible X values tend to align with lower visible Y values"
    else:
        direction_phrase = "the visible relationship is weak, so similar X values often map to a wide range of Y outcomes"

    return (
        f"💡 Most visible tracks lie between about {x_q10:.2f} and {x_q90:.2f} on "
        f"{x_label} and between {y_q10:.2f} and {y_q90:.2f} on {y_label}. Under the "
        f"current filters, {direction_phrase}, which helps explain whether the cloud forms "
        f"a coherent band or a much looser scatter."
    )

def build_relationship_light_guide(
    metrics: dict,
    controls: dict,
) -> str:
    """Return a minimal interpretation hint for non-scatter relationship views."""
    chart_mode = controls.get("chart_mode", "Scatter")
    x_label = get_track_page_display_label(metrics["x_col"]).lower()
    y_label = get_track_page_display_label(metrics["y_col"]).lower()

    if chart_mode == "Density Heatmap":
        return (
            f"Darker cells indicate where many tracks share similar "
            f"{x_label} and {y_label} values."
        )

    if chart_mode == "Quantile Heatmap":
        stat = controls.get("quantile_stat", "Mean Y").lower()
        return (
            f"Color reflects {stat} within each region of the visible "
            f"{x_label} × {y_label} space."
        )

    return ""

def create_freeform_scatter_chart(
    plot_df: pd.DataFrame,
    line_df: pd.DataFrame | None,
    metrics: dict,
    controls: dict,
) -> alt.Chart:
    """Create a freeform scatterplot for arbitrary numeric X and Y columns."""
    tooltip_fields = []

    for col in ["film_title", "album_title", "track_title", "composer_primary_clean"]:
        if col in plot_df.columns:
            tooltip_fields.append(
                alt.Tooltip(f"{col}:N", title=get_track_page_display_label(col))
            )

    tooltip_fields.extend(
        [
            alt.Tooltip(
                "x_raw:Q",
                title=f"{get_track_page_display_label(metrics['x_col'])} (Raw)",
                format=",.3f",
            ),
            alt.Tooltip(
                "y_raw:Q",
                title=f"{get_track_page_display_label(metrics['y_col'])} (Raw)",
                format=",.3f",
            ),
        ]
    )

    if metrics["transform_x"] != "None":
        tooltip_fields.append(
            alt.Tooltip(
                "x_value:Q",
                title=metrics["x_axis_title"],
                format=",.3f",
            )
        )

    if metrics["transform_y"] != "None":
        tooltip_fields.append(
            alt.Tooltip(
                "y_value:Q",
                title=metrics["y_axis_title"],
                format=",.3f",
            )
        )

    if metrics["size_col"] and metrics["size_col"] != "None":
        tooltip_fields.append(
            alt.Tooltip(
                "size_raw:Q",
                title=get_track_page_display_label(metrics["size_col"]),
                format=",.3f",
            )
        )

    if metrics["color_col"] and metrics["color_col"] != "None":
        color_label = get_track_page_display_label(metrics["color_col"])
        if pd.api.types.is_numeric_dtype(plot_df[metrics["color_col"]]):
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metrics['color_col']}:Q",
                    title=color_label,
                    format=",.3f",
                )
            )
        else:
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metrics['color_col']}:N",
                    title=color_label,
                )
            )

    base = alt.Chart(plot_df).mark_circle(
        opacity=0.45,
        filled=True,
        stroke="white",
        strokeWidth=0.2,
    ).encode(
        x=alt.X("x_plot:Q", title=metrics["x_axis_title"]),
        y=alt.Y("y_plot:Q", title=metrics["y_axis_title"]),
        tooltip=tooltip_fields,
    )

    color_legend_orient = "right"
    if metrics["color_col"] and metrics["color_col"] != "None":
        n_color_groups = plot_df[metrics["color_col"]].dropna().nunique()
        if n_color_groups > 8:
            color_legend_orient = "bottom"

    if metrics["color_col"] and metrics["color_col"] != "None":
        points = base.encode(
            color=alt.Color(
                f"{metrics['color_col']}:N",
                title=get_track_page_display_label(metrics["color_col"]),
                legend=alt.Legend(orient=color_legend_orient),
            )
        )
    else:
        points = base.encode(
            color=alt.value("#4c78a8")
        )

    show_size_legend = True
    if metrics["color_col"] and metrics["color_col"] != "None":
        n_color_groups = plot_df[metrics["color_col"]].dropna().nunique()
        if n_color_groups > 8:
            show_size_legend = False

    if metrics["size_col"] and metrics["size_col"] != "None":
        points = points.encode(
            size=alt.Size(
                "size_scaled_raw:Q",
                title=get_track_page_display_label(metrics["size_col"]),
                scale=alt.Scale(
                    type="sqrt",
                    range=[4, controls["bubble_max_size"]],
                    domain=[0, metrics["size_scale_max"]]
                    if metrics["size_scale_max"] else None,
                    clamp=True,
                ),
                legend=(
                    alt.Legend(
                        orient="right",
                        title=get_track_page_display_label(metrics["size_col"]),
                    )
                    if show_size_legend
                    else None
                ),
            )
        )

    layers = [points]

    if controls["show_median_lines"]:
        x_rule_df = pd.DataFrame({"x_plot": [metrics["x_median"]]})
        y_rule_df = pd.DataFrame({"y_plot": [metrics["y_median"]]})

        x_rule = (
            alt.Chart(x_rule_df)
            .mark_rule(
                color="#B8C4D6",
                strokeDash=[6, 4],
                strokeWidth=1.5,
                opacity=0.9,
            )
            .encode(x="x_plot:Q")
        )
        y_rule = (
            alt.Chart(y_rule_df)
            .mark_rule(
                color="#B8C4D6",
                strokeDash=[6, 4],
                strokeWidth=1.5,
                opacity=0.9,
            )
            .encode(y="y_plot:Q")
        )

        layers.extend([x_rule, y_rule])

    if controls["show_trendline"] and line_df is not None:
        line = (
            alt.Chart(line_df)
            .mark_line(strokeWidth=3)
            .encode(
                x="x_plot:Q",
                y="y_plot:Q",
            )
        )
        layers.append(line)

    subtitle_parts = [
        f"r = {metrics['pearson_r']:.3f}",
        f"R² = {metrics['r_squared']:.3f}",
        f"{metrics['rows_used']:,} tracks",
    ]

    if metrics["color_col"] and metrics["color_col"] != "None":
        subtitle_parts.append(
            f"Color = {get_track_page_display_label(metrics['color_col'])}"
        )

    if metrics["size_col"] and metrics["size_col"] != "None":
        if show_size_legend:
            subtitle_parts.append(
                f"Size = {get_track_page_display_label(metrics['size_col'])}"
            )
        else:
            subtitle_parts.append(
                f"Size = {get_track_page_display_label(metrics['size_col'])} (tooltip only)"
            )

    if controls["show_median_lines"]:
        subtitle_parts.append("Median lines shown")

    if metrics["apply_jitter"]:
        subtitle_parts.append(
            f"Jitter = {metrics['jitter_strength']:.3f} (display only)"
        )

    if (
        metrics["transform_x"] != "None"
        or metrics["transform_y"] != "None"
    ):
        subtitle_parts.append("Metrics and fitted line use displayed scale")

    return alt.layer(*layers).properties(
        width=750,
        height=500,
        title={
            "text": (
                f"{get_track_page_display_label(metrics['x_col'])} vs "
                f"{get_track_page_display_label(metrics['y_col'])}"
            ),
            "subtitle": [" | ".join(subtitle_parts)],
        },
    )


def build_supporting_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Build a supporting table tied to the active scatterplot."""
    cols = [col for col in TOOLTIP_COLUMNS if col in plot_df.columns]
    cols += [col for col in ["x_raw", "y_raw", "x_value", "y_value"] if col in plot_df.columns]
    cols = list(dict.fromkeys(cols))
    return rename_and_dedupe_for_display(plot_df[cols].head(200))

def create_density_heatmap_chart(
    plot_df: pd.DataFrame,
    metrics: dict,
    bins: int,
) -> alt.Chart:
    """Create a dense relationship view using a 2D binned heatmap of track counts."""
    x_label = metrics["x_axis_title"]
    y_label = metrics["y_axis_title"]

    subtitle = (
        f"{metrics['rows_used']:,} tracks | denser cells indicate where tracks cluster most"
    )

    binned = (
        alt.Chart(plot_df)
        .transform_bin(
            as_=["x_bin_start", "x_bin_end"],
            field="x_value",
            bin=alt.Bin(maxbins=bins),
        )
        .transform_bin(
            as_=["y_bin_start", "y_bin_end"],
            field="y_value",
            bin=alt.Bin(maxbins=bins),
        )
    )

    return (
        binned
        .mark_rect()
        .encode(
            x=alt.X("x_bin_start:Q", bin="binned", title=x_label),
            x2="x_bin_end:Q",
            y=alt.Y("y_bin_start:Q", bin="binned", title=y_label),
            y2="y_bin_end:Q",
            color=alt.Color("count():Q", title="Track count"),
            tooltip=[
                alt.Tooltip("x_bin_start:Q", title=f"{x_label} from", format=",.3f"),
                alt.Tooltip("x_bin_end:Q", title=f"{x_label} to", format=",.3f"),
                alt.Tooltip("y_bin_start:Q", title=f"{y_label} from", format=",.3f"),
                alt.Tooltip("y_bin_end:Q", title=f"{y_label} to", format=",.3f"),
                alt.Tooltip("count():Q", title="Tracks"),
            ],
        )
        .properties(
            width=750,
            height=500,
            title={
                "text": (
                    f"{get_track_page_display_label(metrics['x_col'])} vs "
                    f"{get_track_page_display_label(metrics['y_col'])}"
                ),
                "subtitle": [subtitle],
            },
        )
    )

def create_quantile_heatmap_chart(
    plot_df: pd.DataFrame,
    metrics: dict,
    bins: int,
    quantile_stat: str,
) -> alt.Chart:
    """Create a binned summary heatmap for the visible relationship space."""
    x_label = metrics["x_axis_title"]
    y_label = metrics["y_axis_title"]

    if quantile_stat == "Median Y":
        color_encoding = alt.Color(
            "median(y_raw):Q",
            title=f"Median {get_track_page_display_label(metrics['y_col'])}",
        )
        tooltip_fields = [
            alt.Tooltip("count():Q", title="Tracks"),
            alt.Tooltip(
                "median(y_raw):Q",
                title=f"Median {get_track_page_display_label(metrics['y_col'])}",
                format=",.3f",
            ),
        ]
        subtitle = "Cells summarize the median raw Y value within each visible region."
    elif quantile_stat == "Count":
        color_encoding = alt.Color(
            "count():Q",
            title="Track count",
        )
        tooltip_fields = [
            alt.Tooltip("count():Q", title="Tracks"),
        ]
        subtitle = "Cells summarize how many visible tracks fall in each region."
    else:
        color_encoding = alt.Color(
            "mean(y_raw):Q",
            title=f"Mean {get_track_page_display_label(metrics['y_col'])}",
        )
        tooltip_fields = [
            alt.Tooltip("count():Q", title="Tracks"),
            alt.Tooltip(
                "mean(y_raw):Q",
                title=f"Mean {get_track_page_display_label(metrics['y_col'])}",
                format=",.3f",
            ),
        ]
        subtitle = "Cells summarize the mean raw Y value within each visible region."

    binned = (
        alt.Chart(plot_df)
        .transform_bin(
            as_=["x_bin_start", "x_bin_end"],
            field="x_value",
            bin=alt.Bin(maxbins=bins),
        )
        .transform_bin(
            as_=["y_bin_start", "y_bin_end"],
            field="y_value",
            bin=alt.Bin(maxbins=bins),
        )
    )

    return (
        binned
        .mark_rect()
        .encode(
            x=alt.X("x_bin_start:Q", bin="binned", title=x_label),
            x2="x_bin_end:Q",
            y=alt.Y("y_bin_start:Q", bin="binned", title=y_label),
            y2="y_bin_end:Q",
            color=color_encoding,
            tooltip=[
                alt.Tooltip("x_bin_start:Q", title=f"{x_label} from", format=",.3f"),
                alt.Tooltip("x_bin_end:Q", title=f"{x_label} to", format=",.3f"),
                alt.Tooltip("y_bin_start:Q", title=f"{y_label} from", format=",.3f"),
                alt.Tooltip("y_bin_end:Q", title=f"{y_label} to", format=",.3f"),
                *tooltip_fields,
            ],
        )
        .properties(
            width=750,
            height=500,
            title={
                "text": (
                    f"{get_track_page_display_label(metrics['x_col'])} vs "
                    f"{get_track_page_display_label(metrics['y_col'])}"
                ),
                "subtitle": [subtitle],
            },
        )
    )

def main() -> None:
    st.set_page_config(
        page_title="Track Relationship Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Relationship Explorer")
    st.write(
        """
        Compare two track-level variables directly and explore how their relationship
        changes with optional transforms, color groupings, and fitted lines.
        """
    )

    track_df = load_track_data_explorer_data()
    track_df = add_track_relationship_display_fields(track_df)

    filter_inputs = get_global_filter_inputs(track_df)
    composer_options = get_clean_composer_options(track_df)

    st.sidebar.header("Track Relationship Controls")
    include_context_features = st.sidebar.checkbox(
        "Include film & album context",
        value=False,
        help=(
            "Adds film- and album-level numeric and grouping variables to the X/Y "
            "and color selectors while keeping native track variables first."
        ),
    )

    numeric_options = get_track_numeric_options(
        track_df,
        include_context_features=include_context_features,
    )
    color_options = get_track_group_options(
        track_df,
        include_context_features=include_context_features,
    )

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    controls = get_track_relationship_controls(
        numeric_options=numeric_options,
        color_options=color_options,
        composer_options=composer_options,
    )

    controls["include_context_features"] = include_context_features

    filtered_df = filter_track_relationship_df(
        track_df=track_df,
        global_controls=global_controls,
        controls=controls,
    )

    if filtered_df.empty:
        st.warning("No tracks remain under the current filters.")
        return

    st.markdown("**Filter Context**")
    st.caption(build_relationship_context_caption(controls))

    try:
        plot_df, line_df, metrics = build_freeform_scatter_data(
            df=filtered_df,
            x_col=controls["x_col"],
            y_col=controls["y_col"],
            color_col=controls["color_col"],
            size_col=controls["size_col"],
            transform_x=controls["transform_x"],
            transform_y=controls["transform_y"],
            apply_jitter=controls["apply_jitter"],
            jitter_strength=controls["jitter_strength"],
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    render_relationship_insight_cards(metrics)

    st.markdown("### Relationship View")

    if controls["chart_mode"] == "Density Heatmap":
        chart = create_density_heatmap_chart(
            plot_df=plot_df,
            metrics=metrics,
            bins=controls["heatmap_bins"],
        )
    elif controls["chart_mode"] == "Quantile Heatmap":
        chart = create_quantile_heatmap_chart(
            plot_df=plot_df,
            metrics=metrics,
            bins=controls["heatmap_bins"],
            quantile_stat=controls["quantile_stat"],
        )
    else:
        chart = create_freeform_scatter_chart(
            plot_df=plot_df,
            line_df=line_df,
            metrics=metrics,
            controls=controls,
        )

    st.altair_chart(chart, width="stretch")

    st.caption(
        build_relationship_chart_insight(
            plot_df=plot_df,
            metrics=metrics,
            controls=controls,
        )
    )

    if controls["chart_mode"] != "Scatter":
        st.caption(build_relationship_light_guide(metrics, controls))

    if controls["show_data_table"]:
        st.markdown("### Supporting Table")
        st.caption(
            "This table is tied to the active relationship view and shows the source rows behind the current chart."
        )
        table_df = build_supporting_table(plot_df)
        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()