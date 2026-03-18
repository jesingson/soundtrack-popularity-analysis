import altair as alt
import pandas as pd
import streamlit as st

import data_processing as dp
import regression_analysis as reg
from app.app_controls import get_scatter_controls
from app.app_data import load_analysis_data, load_source_data


TOOLTIP_METADATA_CANDIDATES = [
    "release_group_name",
    "album_title",
    "soundtrack_title",
    "film_title",
    "composer_primary_clean",
    "album_us_release_year",
]


def build_feature_rank_lookup(
    ranking_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Build a lookup dictionary for ranked feature metadata.

    Args:
        ranking_df: Ranked feature dataframe.

    Returns:
        dict[str, dict]: Mapping from feature name to ranking row data.
    """
    return ranking_df.set_index("feature").to_dict(orient="index")


def pick_available_metadata_cols(
    albums_df: pd.DataFrame,
) -> list[str]:
    """
    Select descriptive metadata columns that actually exist in albums_df.

    Args:
        albums_df: Raw album dataframe.

    Returns:
        list[str]: Available descriptive columns for tooltip use.
    """
    return [col for col in TOOLTIP_METADATA_CANDIDATES if col in albums_df.columns]


def create_exploratory_scatter_chart(
    plot_df: pd.DataFrame,
    line_df: pd.DataFrame,
    metrics: dict,
    feature_rank: dict,
) -> alt.Chart:
    """
    Create the exploratory scatterplot with richer tooltips.

    Args:
        plot_df: Scatterplot point dataframe.
        line_df: Fitted line dataframe.
        metrics: Metrics dictionary returned by the helper.
        feature_rank: Ranking metadata for the selected feature.

    Returns:
        alt.Chart: Layered Altair scatterplot and fitted line.
    """
    tooltip_fields = []

    if "film_title" in plot_df.columns:
        tooltip_fields.append(alt.Tooltip("film_title:N", title="Film"))

    if "release_group_name" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("release_group_name:N", title="Soundtrack")
        )
    elif "album_title" in plot_df.columns:
        tooltip_fields.append(alt.Tooltip("album_title:N", title="Soundtrack"))
    elif "soundtrack_title" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("soundtrack_title:N", title="Soundtrack")
        )

    if "composer_primary_clean" in plot_df.columns:
        tooltip_fields.append(
            alt.Tooltip("composer_primary_clean:N", title="Composer")
        )

    tooltip_fields.extend(
        [
            alt.Tooltip(
                "x_raw_value:Q",
                title=f"{metrics['feature_col']} (raw)",
                format=",.3f",
            ),
            alt.Tooltip(
                "x_value:Q",
                title=metrics["x_axis_label"],
                format=".3f",
            ),
            alt.Tooltip(
                "y_value:Q",
                title=metrics["target_col"],
                format=".3f",
            ),
        ]
    )

    points = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.25, size=40)
        .encode(
            x=alt.X("x_value:Q", title=metrics["x_axis_label"]),
            y=alt.Y("y_value:Q", title="Log soundtrack listeners"),
            tooltip=tooltip_fields,
        )
    )

    line = (
        alt.Chart(line_df)
        .mark_line(strokeWidth=3)
        .encode(
            x="x_value:Q",
            y="y_value:Q",
        )
    )

    title_text = (
        f"{metrics['feature_col']} vs {metrics['target_col']}"
    )
    subtitle_text = (
        f"Rank #{feature_rank['rank']} by absolute Pearson correlation | "
        f"r = {feature_rank['corr']:.3f} | "
        f"R² = {feature_rank['r_squared']:.3f}"
    )

    return (points + line).properties(
        width=750,
        height=500,
        title={
            "text": title_text,
            "subtitle": [subtitle_text],
        },
    )


def render_summary_metrics(
    metrics: dict,
    feature_rank: dict,
) -> None:
    """
    Render headline metrics for the selected feature.

    Args:
        metrics: Metrics dictionary from the scatter helper.
        feature_rank: Ranking metadata for the selected feature.
    """
    direction = feature_rank["direction"].title()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Selected feature", metrics["feature_col"])

    with col2:
        st.metric("Rank", f"#{feature_rank['rank']}")

    with col3:
        st.metric("Rows used", f"{metrics['rows_used']:,}")

    with col4:
        st.metric("Pearson r", f"{feature_rank['corr']:.3f}")

    with col5:
        st.metric("Univariate R²", f"{feature_rank['r_squared']:.3f}")

    st.caption(
        f"Direction: {direction}. "
        f"X-axis shown on transformed modeling scale; raw values remain in the tooltip."
    )


def main() -> None:
    """
    Run the scatterplot explorer page.
    """
    st.set_page_config(
        page_title="Scatterplot Explorer",
        layout="wide",
    )

    st.title("Scatterplot Explorer")
    st.write(
        """
        Explore how individual continuous features relate to soundtrack
        popularity. Features are ranked by their univariate Pearson
        relationship with `log_lfm_album_listeners`, while the chart uses
        the transformed regression-scale version of the selected feature
        when appropriate.
        """
    )

    albums_df, _ = load_source_data()
    album_analytics_df = load_analysis_data()

    ranking_df = reg.build_scatterplot_feature_ranking(
        album_analytics_df=album_analytics_df,
        target_col=dp.TARGET_COL,
        method="pearson",
    )

    feature_options = ranking_df["feature"].tolist()
    controls = get_scatter_controls(feature_options)

    selected_feature = controls["selected_feature"]
    rank_lookup = build_feature_rank_lookup(ranking_df)
    feature_rank = rank_lookup[selected_feature]

    metadata_cols = pick_available_metadata_cols(albums_df)

    plot_df, line_df, metrics = reg.build_exploratory_scatter_data(
        album_analytics_df=album_analytics_df,
        feature_col=selected_feature,
        metadata_df=albums_df,
        metadata_cols=metadata_cols,
        id_cols=["tmdb_id", "release_group_mbid"],
        target_col=dp.TARGET_COL,
    )

    render_summary_metrics(
        metrics=metrics,
        feature_rank=feature_rank,
    )

    st.subheader("Scatterplot")
    chart = create_exploratory_scatter_chart(
        plot_df=plot_df,
        line_df=line_df,
        metrics=metrics,
        feature_rank=feature_rank,
    )
    st.altair_chart(chart, use_container_width=True)

    if controls["show_data_table"]:
        st.subheader("Scatterplot source data")
        st.dataframe(plot_df, use_container_width=True)

    if controls["show_feature_ranking"]:
        st.subheader("Ranked feature table")
        st.dataframe(ranking_df.round(3), use_container_width=True)


if __name__ == "__main__":
    main()