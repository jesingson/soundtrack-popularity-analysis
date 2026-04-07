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
)
from app.ui import apply_app_styles, get_display_label

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
        f"Comparing {get_display_label(controls['x_col'])} vs "
        f"{get_display_label(controls['y_col'])}"
    )

    extras = []

    if controls["color_col"] != "None":
        extras.append(f"color = {get_display_label(controls['color_col'])}")

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
    transform_x: str = "None",
    transform_y: str = "None",
    apply_jitter: bool = False,
    jitter_strength: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Build freeform scatterplot data for arbitrary track-level numeric X and Y columns."""
    required_cols = [x_col, y_col]
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

    x_display = get_display_label(x_col)
    y_display = get_display_label(y_col)

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
    }

    return plot_df, line_df, metrics


def build_relationship_insight_summary(metrics: dict) -> list[tuple[str, str, str]]:
    """Build key insight cards for the relationship explorer."""
    pearson_r = float(metrics["pearson_r"])

    color_value = (
        get_display_label(metrics["color_col"])
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

    x_label = get_display_label(metrics["x_col"]).lower()
    y_label = get_display_label(metrics["y_col"]).lower()

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
        color_label = get_display_label(controls["color_col"]).lower()
        base += (
            f" Coloring by {color_label} helps reveal whether distinct subgroups follow different patterns."
        )

    return base


def create_freeform_scatter_chart(
    plot_df: pd.DataFrame,
    line_df: pd.DataFrame | None,
    metrics: dict,
    show_trendline: bool,
) -> alt.Chart:
    """Create a freeform scatterplot for arbitrary numeric X and Y columns."""
    tooltip_fields = []

    for col in ["film_title", "album_title", "track_title", "composer_primary_clean"]:
        if col in plot_df.columns:
            tooltip_fields.append(
                alt.Tooltip(f"{col}:N", title=get_display_label(col))
            )

    tooltip_fields.extend(
        [
            alt.Tooltip(
                "x_raw:Q",
                title=f"{get_display_label(metrics['x_col'])} (Raw)",
                format=",.3f",
            ),
            alt.Tooltip(
                "y_raw:Q",
                title=f"{get_display_label(metrics['y_col'])} (Raw)",
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

    if metrics["color_col"] and metrics["color_col"] != "None":
        if pd.api.types.is_numeric_dtype(plot_df[metrics["color_col"]]):
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metrics['color_col']}:Q",
                    title=get_display_label(metrics["color_col"]),
                    format=",.3f",
                )
            )
        else:
            tooltip_fields.append(
                alt.Tooltip(
                    f"{metrics['color_col']}:N",
                    title=get_display_label(metrics["color_col"]),
                )
            )

    base = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.35, size=45)
        .encode(
            x=alt.X("x_plot:Q", title=metrics["x_axis_title"]),
            y=alt.Y("y_plot:Q", title=metrics["y_axis_title"]),
            tooltip=tooltip_fields,
        )
    )

    if metrics["color_col"] and metrics["color_col"] != "None":
        points = base.encode(
            color=alt.Color(
                f"{metrics['color_col']}:N",
                title=get_display_label(metrics["color_col"]),
            )
        )
    else:
        points = base

    chart = points

    if show_trendline and line_df is not None:
        line = (
            alt.Chart(line_df)
            .mark_line(strokeWidth=3)
            .encode(
                x="x_plot:Q",
                y="y_plot:Q",
            )
        )
        chart = chart + line

    subtitle = (
        f"r = {metrics['pearson_r']:.3f} | R² = {metrics['r_squared']:.3f} | "
        f"{metrics['rows_used']:,} tracks"
    )

    return chart.properties(
        width=750,
        height=500,
        title={
            "text": f"{get_display_label(metrics['x_col'])} vs {get_display_label(metrics['y_col'])}",
            "subtitle": [subtitle],
        },
    )


def build_supporting_table(plot_df: pd.DataFrame) -> pd.DataFrame:
    """Build a supporting table tied to the active scatterplot."""
    cols = [col for col in TOOLTIP_COLUMNS if col in plot_df.columns]
    cols += [col for col in ["x_raw", "y_raw", "x_value", "y_value"] if col in plot_df.columns]
    cols = list(dict.fromkeys(cols))
    return rename_and_dedupe_for_display(plot_df[cols].head(200))


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

    numeric_options = get_track_numeric_options(track_df)
    color_options = get_track_group_options(track_df)

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
    chart = create_freeform_scatter_chart(
        plot_df=plot_df,
        line_df=line_df,
        metrics=metrics,
        show_trendline=controls["show_trendline"],
    )
    st.altair_chart(chart, width="stretch")

    st.caption(build_relationship_supporting_insight(metrics, controls))
    st.caption(
        "Look for slope (direction), spread (consistency), and clustering (group structure). "
        "Tighter bands indicate more predictable relationships."
    )

    if controls["show_data_table"]:
        st.markdown("### Supporting Table")
        st.caption(
            "This table is tied to the active scatterplot and shows the source rows behind the current relationship view."
        )
        table_df = build_supporting_table(plot_df)
        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()