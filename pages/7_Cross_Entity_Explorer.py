from __future__ import annotations

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.app_controls import (
    get_cross_entity_controls,
    get_global_filter_controls,
)
from app.app_data import load_explorer_data
from app.data_filters import filter_dataset, split_multivalue_genres
from app.ui import (
    apply_app_styles,
    rename_columns_for_display,
)

DETAIL_COLS = [
    "album_title",
    "film_title",
    "composer_primary_clean",
    "label_names",
    "film_year",
    "n_tracks",
    "lfm_album_listeners",
    "lfm_album_playcount",
    "album_genres_display",
    "film_genres",
]


def get_edge_metric_column(metric_name: str) -> str:
    """
    Map the UI-selected relationship metric label to the dataframe column.

    Args:
        metric_name: User-facing metric label from the sidebar.

    Returns:
        str: Dataframe column name.
    """
    metric_map = {
        "Count": "count",
        "% of film genre": "pct_source",
        "% of album genre": "pct_target",
        "Jaccard similarity": "jaccard",
        "Lift": "lift",
    }
    return metric_map.get(metric_name, "count")


def get_edge_metric_display_label(metric_name: str) -> str:
    """
    Return a readable axis/tooltip label for the selected metric.

    Args:
        metric_name: User-facing metric label.

    Returns:
        str: Display-ready metric label.
    """
    return metric_name


def build_cross_entity_explorer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the page-specific dataframe used by the Cross-Entity Explorer.

    Args:
        df: Album-level explorer dataframe.

    Returns:
        pd.DataFrame: Copy with numeric film year coerced cleanly.
    """
    out_df = df.copy()

    if "film_year" in out_df.columns:
        out_df["film_year"] = pd.to_numeric(
            out_df["film_year"],
            errors="coerce",
        )

    return out_df


def build_film_album_flow_df(
    df: pd.DataFrame,
    min_flow_count: int,
    selected_film_genres: list[str] | None = None,
    selected_album_genres: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the directional film-genre to album-genre flow dataframe.

    Each filtered album may contribute to multiple flows if it has multiple
    film genres and/or multiple album genres. When selected film or album
    genres are provided, the exploded source/target lists are restricted to
    those values so the Sankey respects the visible filter intent directly.

    Args:
        df: Filtered album-level dataframe.
        min_flow_count: Minimum raw count required to retain a flow.
        selected_film_genres: Optional film genres to keep on the source side.
        selected_album_genres: Optional album genres to keep on the target side.

    Returns:
        pd.DataFrame: One row per retained film→album genre flow with
        count- and similarity-based metrics.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "pair_label",
                "count",
                "source_count",
                "target_count",
                "union_count",
                "pct_source",
                "pct_target",
                "jaccard",
                "lift",
            ]
        )

    selected_film_set = set(selected_film_genres or [])
    selected_album_set = set(selected_album_genres or [])

    flow_rows = []

    for _, row in df.iterrows():
        film_values = []
        album_values = []

        if "film_genres" in row and pd.notna(row["film_genres"]):
            film_values = split_multivalue_genres(pd.Series([row["film_genres"]]))

        if "album_genres_display" in row and pd.notna(row["album_genres_display"]):
            album_values = split_multivalue_genres(
                pd.Series([row["album_genres_display"]])
            )

        if selected_film_set:
            film_values = [genre for genre in film_values if genre in selected_film_set]

        if selected_album_set:
            album_values = [
                genre for genre in album_values if genre in selected_album_set
            ]

        if not film_values or not album_values:
            continue

        for film_genre in film_values:
            for album_genre in album_values:
                flow_rows.append(
                    {
                        "source": film_genre,
                        "target": album_genre,
                        "album_title": row.get("album_title"),
                        "film_title": row.get("film_title"),
                    }
                )

    if not flow_rows:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "pair_label",
                "count",
                "source_count",
                "target_count",
                "union_count",
                "pct_source",
                "pct_target",
                "jaccard",
                "lift",
            ]
        )

    flow_detail_df = pd.DataFrame(flow_rows)

    pair_counts = (
        flow_detail_df.groupby(["source", "target"])
        .size()
        .reset_index(name="count")
    )

    pair_counts = pair_counts[pair_counts["count"] >= min_flow_count].copy()
    if pair_counts.empty:
        return pair_counts

    source_counts = flow_detail_df.groupby("source").size().to_dict()
    target_counts = flow_detail_df.groupby("target").size().to_dict()

    total_pairs = int(len(flow_detail_df))

    pair_counts["source_count"] = pair_counts["source"].map(source_counts).astype(int)
    pair_counts["target_count"] = pair_counts["target"].map(target_counts).astype(int)
    pair_counts["union_count"] = (
        pair_counts["source_count"] + pair_counts["target_count"] - pair_counts["count"]
    )

    pair_counts["pct_source"] = pair_counts["count"] / pair_counts["source_count"]
    pair_counts["pct_target"] = pair_counts["count"] / pair_counts["target_count"]
    pair_counts["jaccard"] = pair_counts["count"] / pair_counts["union_count"]

    p_source = pair_counts["source_count"] / total_pairs
    p_target = pair_counts["target_count"] / total_pairs
    p_both = pair_counts["count"] / total_pairs
    expected = p_source * p_target
    pair_counts["lift"] = p_both / expected

    pair_counts["pair_label"] = (
        pair_counts["source"].astype(str) + " → " + pair_counts["target"].astype(str)
    )

    pair_counts = pair_counts.sort_values(
        ["count", "pair_label"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return pair_counts


def build_film_album_detail_table(
    df: pd.DataFrame,
    film_genre: str,
    album_genre: str,
) -> pd.DataFrame:
    """
    Build the album drilldown table for one selected film→album flow.

    Args:
        df: Filtered album-level dataframe.
        film_genre: Selected film genre.
        album_genre: Selected album genre.

    Returns:
        pd.DataFrame: Albums that contain both the selected film genre and
        selected album genre.
    """
    if df.empty:
        return pd.DataFrame()

    detail_df = df[
        df["film_genres"].apply(
            lambda value: pd.notna(value)
            and film_genre in split_multivalue_genres(pd.Series([value]))
        ) &
        df["album_genres_display"].apply(
            lambda value: pd.notna(value)
            and album_genre in split_multivalue_genres(pd.Series([value]))
        )
    ].copy()

    if detail_df.empty:
        return detail_df

    if "lfm_album_listeners" in detail_df.columns:
        detail_df = detail_df.sort_values(
            "lfm_album_listeners",
            ascending=False,
            na_position="last",
        )

    cols = [col for col in DETAIL_COLS if col in detail_df.columns]
    return detail_df[cols].copy()


def create_top_flows_chart(
    flow_df: pd.DataFrame,
    top_n_edges: int,
    metric_name: str,
) -> alt.Chart:
    """
    Create a horizontal ranking chart of the strongest film→album flows.

    Args:
        flow_df: Film→album flow dataframe.
        top_n_edges: Number of top flows to show.
        metric_name: User-selected metric label.

    Returns:
        alt.Chart: Horizontal bar chart.
    """
    if flow_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    metric_col = get_edge_metric_column(metric_name)
    metric_label = get_edge_metric_display_label(metric_name)

    plot_df = flow_df.sort_values(
        [metric_col, "count", "pair_label"],
        ascending=[False, False, True],
    ).head(top_n_edges).copy()

    tooltip = [
        alt.Tooltip("pair_label:N", title="Flow"),
        alt.Tooltip("count:Q", title="Albums in Flow", format=",.0f"),
        alt.Tooltip("pct_source:Q", title="% of Film Genre", format=".1%"),
        alt.Tooltip("pct_target:Q", title="% of Album Genre", format=".1%"),
        alt.Tooltip("jaccard:Q", title="Jaccard", format=".3f"),
        alt.Tooltip("lift:Q", title="Lift", format=".2f"),
    ]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric_col}:Q", title=metric_label),
            y=alt.Y(
                "pair_label:N",
                sort=alt.SortField(field=metric_col, order="descending"),
                title="Film → Album Flow",
            ),
            tooltip=tooltip,
        )
        .properties(
            width=750,
            height=max(320, min(800, top_n_edges * 36)),
            title={
                "text": "Top Film → Album Genre Flows",
                "subtitle": [f"Ranked by {metric_label.lower()}."],
            },
        )
    )
    return chart


def render_film_album_sankey(
    flow_df: pd.DataFrame,
    metric_name: str,
) -> None:
    """
    Render the Film Genre → Album Genre Sankey chart with meaningful node hover.

    Args:
        flow_df: Film→album flow dataframe.
        metric_name: User-selected metric label.
    """
    if flow_df.empty:
        st.info("No film-to-album flows remain under the current filters.")
        return

    metric_col = get_edge_metric_column(metric_name)

    source_nodes = sorted(flow_df["source"].unique().tolist())
    target_nodes = sorted(flow_df["target"].unique().tolist())
    all_nodes = source_nodes + target_nodes

    node_index = {name: idx for idx, name in enumerate(all_nodes)}

    source_idx = [node_index[value] for value in flow_df["source"]]
    target_idx = [node_index[value] for value in flow_df["target"]]
    values = flow_df[metric_col].astype(float).tolist()

    link_hover = [
        (
            f"Film genre: {row['source']}<br>"
            f"Album genre: {row['target']}<br>"
            f"Albums in flow: {int(row['count'])}<br>"
            f"% of film genre: {row['pct_source']:.1%}<br>"
            f"% of album genre: {row['pct_target']:.1%}<br>"
            f"Jaccard: {row['jaccard']:.3f}<br>"
            f"Lift: {row['lift']:.2f}"
        )
        for _, row in flow_df.iterrows()
    ]

    film_totals = flow_df.groupby("source")["count"].sum().to_dict()
    album_totals = flow_df.groupby("target")["count"].sum().to_dict()
    total_flow = flow_df["count"].sum()

    node_album_counts = []
    node_pct = []

    for node in all_nodes:
        if node in film_totals:
            count = film_totals[node]
        else:
            count = album_totals.get(node, 0)

        node_album_counts.append(count)
        node_pct.append(count / total_flow if total_flow > 0 else 0)

    node_hover = [
        (
            f"Genre: {name}<br>"
            f"Albums connected: {node_album_counts[i]:,}<br>"
            f"% of total flows: {node_pct[i]:.1%}"
        )
        for i, name in enumerate(all_nodes)
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(255,255,255,0.2)", width=0.5),
                    label=all_nodes,
                    color=(["#4C78A8"] * len(source_nodes)) +
                          (["#F58518"] * len(target_nodes)),
                    customdata=node_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=source_idx,
                    target=target_idx,
                    value=values,
                    customdata=link_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Film Genre → Album Genre Sankey ({metric_name})",
        font=dict(color="white"),
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, width="stretch")


def render_summary_metrics(
    filtered_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    metric_name: str,
    relationship_label: str,
) -> None:
    """
    Render headline page metrics.

    Args:
        filtered_df: Globally filtered dataframe.
        relationship_df: Flow dataframe for the selected mode.
        metric_name: User-selected metric label.
        relationship_label: Human-readable label for the relationship family.
    """
    metric_col = get_edge_metric_column(metric_name)

    album_count = int(len(filtered_df))
    relationship_count = int(len(relationship_df))
    source_count = int(relationship_df["source"].nunique()) if not relationship_df.empty else 0
    target_count = int(relationship_df["target"].nunique()) if not relationship_df.empty else 0

    ranked_df = relationship_df.sort_values(
        [metric_col, "count", "pair_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top_relationship = ranked_df.iloc[0]["pair_label"] if not ranked_df.empty else "None"
    top_metric_value = float(ranked_df.iloc[0][metric_col]) if not ranked_df.empty else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Albums in View", f"{album_count:,}")

    with col2:
        st.metric("Sources in View", f"{source_count:,}")

    with col3:
        st.metric("Targets in View", f"{target_count:,}")

    with col4:
        st.metric(relationship_label, f"{relationship_count:,}")

    with col5:
        if metric_col in {"pct_source", "pct_target"}:
            st.metric(metric_name, f"{top_metric_value:.1%}")
        elif metric_col == "jaccard":
            st.metric("Top Jaccard", f"{top_metric_value:.3f}")
        elif metric_col == "lift":
            st.metric("Top Lift", f"{top_metric_value:.2f}")
        else:
            st.metric("Top Count", f"{top_metric_value:,.0f}")

    st.caption(f"Top relationship: {top_relationship}")


def main() -> None:
    """
    Render the Cross-Entity Explorer page.
    """
    apply_app_styles()

    st.title("Cross-Entity Explorer")
    st.write(
        """
        Explore directional relationships across different entity families.
        This page focuses on how film genres map into soundtrack album genres.
        """
    )

    explorer_df = load_explorer_data()
    cross_df = build_cross_entity_explorer_df(explorer_df)

    min_year = int(cross_df["film_year"].dropna().min())
    max_year = int(cross_df["film_year"].dropna().max())

    film_genre_options = split_multivalue_genres(cross_df["film_genres"])
    album_genre_options = split_multivalue_genres(
        cross_df["album_genres_display"]
    )

    global_filters = get_global_filter_controls(
        min_year=min_year,
        max_year=max_year,
        film_genre_options=film_genre_options,
        album_genre_options=album_genre_options,
    )

    controls = get_cross_entity_controls()

    filtered_df = filter_dataset(cross_df, global_filters)

    if filtered_df.empty:
        st.warning("No albums remain after applying the current global filters.")
        return

    flow_df = build_film_album_flow_df(
        df=filtered_df,
        min_flow_count=controls["min_edge_count"],
        selected_film_genres=global_filters.get("selected_film_genres"),
        selected_album_genres=global_filters.get("selected_album_genres"),
    )

    if flow_df.empty:
        st.warning(
            "No film-to-album flows remain after applying the current filters "
            "and minimum flow threshold."
        )
        return

    render_summary_metrics(
        filtered_df=filtered_df,
        relationship_df=flow_df,
        metric_name=controls["edge_metric"],
        relationship_label="Flows Shown",
    )

    st.subheader("Film Genre → Album Genre Sankey")
    render_film_album_sankey(
        flow_df=flow_df,
        metric_name=controls["edge_metric"],
    )
    st.caption(
        "A single album may contribute to multiple flows when it has multiple "
        "film genres and/or multiple album genres."
    )

    st.subheader("Top Film → Album Flows")
    top_flows_chart = create_top_flows_chart(
        flow_df=flow_df,
        top_n_edges=controls["top_n_edges"],
        metric_name=controls["edge_metric"],
    )
    st.altair_chart(top_flows_chart, width="stretch")

    if controls["show_edge_table"]:
        st.markdown("#### Flow Summary Table")

        metric_col = get_edge_metric_column(controls["edge_metric"])

        flow_table = flow_df.sort_values(
            [metric_col, "count", "pair_label"],
            ascending=[False, False, True],
        ).copy()

        flow_table = flow_table.rename(
            columns={
                "source": "Film Genre",
                "target": "Album Genre",
                "pair_label": "Flow",
                "count": "Albums in Flow",
                "source_count": "Albums with Film Genre",
                "target_count": "Albums with Album Genre",
                "union_count": "Albums in Union",
                "pct_source": "% of Film Genre",
                "pct_target": "% of Album Genre",
                "jaccard": "Jaccard",
                "lift": "Lift",
            }
        )

        st.dataframe(
            flow_table,
            width="stretch",
            hide_index=True,
        )

    st.subheader("Inspect a Film → Album Flow")
    flow_options = flow_df["pair_label"].tolist()
    selected_flow_label = st.selectbox(
        "Film → album flow",
        options=flow_options,
        index=0,
        key="film_album_flow_select",
    )

    selected_row = flow_df[flow_df["pair_label"] == selected_flow_label].iloc[0]
    selected_film_genre = selected_row["source"]
    selected_album_genre = selected_row["target"]

    st.caption(
        f"Showing albums with film genre **{selected_film_genre}** and "
        f"album genre **{selected_album_genre}**."
    )

    detail_df = build_film_album_detail_table(
        df=filtered_df,
        film_genre=selected_film_genre,
        album_genre=selected_album_genre,
    )

    if detail_df.empty:
        st.info("No albums match the selected film-to-album flow.")
        return

    if controls["show_album_table"]:
        st.dataframe(
            rename_columns_for_display(detail_df),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()