from __future__ import annotations

import altair as alt
import pandas as pd

from app.ui import get_display_label


def create_single_track_feature_density_chart(
    ridge_density_df: pd.DataFrame,
    feature: str,
    y_col: str = "log_lfm_track_playcount",
    title_text: str = "Density check (single track feature)",
    subtitle_text: str = "Precomputed in pandas",
) -> alt.Chart:
    """
    Build a simple density comparison chart for one track feature.

    Args:
        ridge_density_df: Precomputed ridge density dataframe.
        feature: Raw feature name to filter on.
        y_col: Track outcome column used on the x-axis.
        title_text: Main chart title.
        subtitle_text: Subtitle shown below the title.

    Returns:
        alt.Chart: Single-feature density area chart.
    """
    one = ridge_density_df[ridge_density_df["feature"] == feature].copy()

    if one.empty:
        raise ValueError(
            f"No density rows were found for feature '{feature}'."
        )

    density = (
        alt.Chart(one)
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X("x:Q", title=get_display_label(y_col)),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("group:N", title="Group"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("x:Q", title=get_display_label(y_col), format=".2f"),
                alt.Tooltip("density:Q", title="Density", format=".3f"),
                alt.Tooltip("n_obs:Q", title="Group size"),
            ],
        )
        .properties(
            title={
                "text": title_text,
                "subtitle": subtitle_text,
            }
        )
    )

    return density


def create_track_ridge_chart(
    ridge_chart_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    y_col: str = "log_lfm_track_playcount",
    title_text: str = "Track success distributions by feature group",
    subtitle_lines: list[str] | None = None,
    width: int = 780,
    height: int = 600,
    left_padding: int = 300,
) -> alt.Chart:
    """
    Build the final track ridgeline chart using the same layered approach
    that already works for the album ridge chart.

    Args:
        ridge_chart_df: Chart-ready ridge dataframe containing at least:
            feature_label, group_std, x, y0, y1, density.
        labels_df: Label dataframe containing at least:
            feature_label, y_label.
        y_col: Track-level outcome column used on the x-axis.
        title_text: Main chart title.
        subtitle_lines: Optional subtitle lines.
        width: Chart width in pixels.
        height: Chart height in pixels.
        left_padding: Left chart padding for longer labels.

    Returns:
        alt.Chart: Final layered ridgeline chart.
    """
    if subtitle_lines is None:
        subtitle_lines = [
            "Rows are ordered by absolute median gap between Yes and No groups.",
            f"X-axis shows {get_display_label(y_col).lower()}."
        ]

    feature_labels_ordered = labels_df["feature_label"].tolist()

    base = alt.Chart(ridge_chart_df).encode(
        x=alt.X(
            "x:Q",
            title=get_display_label(y_col),
            scale=alt.Scale(zero=False),
            axis=alt.Axis(
                labelColor="#E5E7EB",
                titleColor="#E5E7EB",
                gridColor="#374151",
            ),
        ),
        y=alt.Y("y1:Q", axis=None),
        y2="y0:Q",
        color=alt.Color(
            "group_std:N",
            sort=["No", "Yes"],
            legend=alt.Legend(
                title="Condition met",
                titleColor="#E5E7EB",
                labelColor="#E5E7EB",
            ),
        ),
        tooltip=[
            alt.Tooltip("feature_label:N", title="Feature"),
            alt.Tooltip("group_std:N", title="Condition met"),
            alt.Tooltip("x:Q", title=get_display_label(y_col), format=".2f"),
            alt.Tooltip("density:Q", title="Density", format=".3f"),
            alt.Tooltip("n_obs:Q", title="Group size"),
        ],
    )

    layers = [
        alt.layer(
            base.transform_filter(alt.datum.feature_label == fl).mark_area(
                fillOpacity=0.6,
                stroke=None,
            ),
            base.transform_filter(alt.datum.feature_label == fl).mark_area(
                fillOpacity=0,
                stroke="#E5E7EB",
                strokeWidth=1.5,
            ),
        )
        for fl in feature_labels_ordered
    ]

    ridges = alt.layer(*layers).properties(
        width=width,
        height=height,
        title={
            "text": title_text,
            "subtitle": subtitle_lines,
        },
    )

    labels = alt.Chart(labels_df).mark_text(
        align="right",
        baseline="middle",
        dx=-8,
        color="#E5E7EB",
    ).encode(
        x=alt.value(0),
        y=alt.Y("y_label:Q", axis=None),
        text="feature_label:N",
    )

    baselines = alt.Chart(labels_df).mark_rule(
        opacity=0.18,
        color="#9CA3AF",
    ).encode(
        y="y_label:Q"
    )

    final_ridge = (
        labels + ridges + baselines
    ).properties(
        title=alt.TitleParams(
            text=title_text,
            subtitle=subtitle_lines,
            color="#E5E7EB",
            subtitleColor="#D1D5DB",
        )
    ).configure_view(
        stroke=None
    ).configure(
        padding={"left": left_padding, "right": 20, "top": 20, "bottom": 20}
    ).configure_axis(
        labelColor="#E5E7EB",
        titleColor="#E5E7EB",
        gridColor="#374151",
        domainColor="#6B7280",
        tickColor="#6B7280",
    ).configure_legend(
        titleColor="#E5E7EB",
        labelColor="#E5E7EB",
    )

    return final_ridge