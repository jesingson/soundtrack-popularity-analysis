from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_cooccurrence_controls,
    get_global_filter_controls,
)
from app.app_data import load_explorer_data
from app.data_filters import filter_dataset, split_multivalue_genres
from app.ui import (
    apply_app_styles,
    get_display_label,
    rename_columns_for_display,
)

try:
    import holoviews as hv
    from bokeh.models import GraphRenderer, HoverTool
    from holoviews import opts
    from streamlit_bokeh import streamlit_bokeh

    hv.extension("bokeh")
    HOLOVIEWS_AVAILABLE = True
except ImportError:
    HOLOVIEWS_AVAILABLE = False


GENRE_FLAGS = [
    "ambient_experimental",
    "classical_orchestral",
    "electronic",
    "hip_hop_rnb",
    "pop",
    "rock",
    "world_folk",
]

GENRE_COLOR_MAP = {
    "ambient_experimental": "#7922CC",
    "classical_orchestral": "#1195B2",
    "electronic": "#CC0000",
    "hip_hop_rnb": "#CE7E00",
    "pop": "#1F6F5B",
    "rock": "#3F1D5C",
    "world_folk": "#8C4A00",
}

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

AWARD_FLAGS = [
    "oscar_score_nominee",
    "oscar_song_nominee",
    "globes_score_nominee",
    "globes_song_nominee",
    "critics_score_nominee",
    "critics_song_nominee",
    "bafta_score_nominee",
]

AWARD_LABEL_MAP = {
    "oscar_score_nominee": "Oscar Score",
    "oscar_song_nominee": "Oscar Song",
    "globes_score_nominee": "Globes Score",
    "globes_song_nominee": "Globes Song",
    "critics_score_nominee": "Critics Score",
    "critics_song_nominee": "Critics Song",
    "bafta_score_nominee": "BAFTA Score",
}

FILM_GENRE_PALETTE = [
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]

AWARD_COLOR_MAP = {
    "oscar_score_nominee": "#D4AF37",
    "oscar_song_nominee": "#F2C94C",
    "globes_score_nominee": "#56CCF2",
    "globes_song_nominee": "#2D9CDB",
    "critics_score_nominee": "#BB6BD9",
    "critics_song_nominee": "#9B51E0",
    "bafta_score_nominee": "#27AE60",
}


def build_cooccurrence_explorer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the page-specific dataframe used by the Co-occurrence Explorer.

    Args:
        df: Album-level explorer dataframe.

    Returns:
        pd.DataFrame: Copy with canonical genre flags coerced to clean integers.
    """
    out_df = df.copy()

    for col in GENRE_FLAGS:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(
                out_df[col],
                errors="coerce",
            ).fillna(0).astype(int)
        else:
            out_df[col] = 0

    for col in AWARD_FLAGS:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(
                out_df[col],
                errors="coerce",
            ).fillna(0).astype(int)
        else:
            out_df[col] = 0

    if "film_year" in out_df.columns:
        out_df["film_year"] = pd.to_numeric(
            out_df["film_year"],
            errors="coerce",
        )

    return out_df


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
        "% of source": "pct_source",
        "% of target": "pct_target",
        "% of source genre": "pct_source",
        "% of target genre": "pct_target",
        "Jaccard similarity": "jaccard",
        "Lift": "lift",
    }
    return metric_map.get(metric_name, "count")


def get_edge_metric_display_label(metric_name: str) -> str:
    """
    Return a readable axis or tooltip label for the selected edge metric.

    Args:
        metric_name: User-facing metric label from the sidebar.

    Returns:
        str: Display-ready metric label.
    """
    return metric_name


def get_award_display_label(name: str) -> str:
    """
    Return a readable label for an award flag.

    Args:
        name: Raw award flag column name.

    Returns:
        str: Display-friendly award label.
    """
    return AWARD_LABEL_MAP.get(name, get_display_label(name))

def get_film_genre_display_label(name: str) -> str:
    """
    Return a readable label for a film genre value.

    Args:
        name: Raw film genre label.

    Returns:
        str: Display-friendly film genre label.
    """
    return str(name)


def build_film_genre_color_map(genres: list[str]) -> dict[str, str]:
    """
    Build a deterministic color mapping for film genres.

    Args:
        genres: Sorted film genre labels.

    Returns:
        dict[str, str]: Mapping from film genre label to hex color.
    """
    return {
        genre: FILM_GENRE_PALETTE[i % len(FILM_GENRE_PALETTE)]
        for i, genre in enumerate(sorted(genres))
    }

def get_relationship_type_options() -> list[str]:
    """
    Return available relationship modes for the page.

    Returns:
        list[str]: Supported relationship modes.
    """
    return [
        "Album Genre ↔ Album Genre",
        "Film Genre ↔ Film Genre",
        "Awards ↔ Awards",
    ]


def build_album_genre_edge_df(
    df: pd.DataFrame,
    min_edge_count: int,
) -> pd.DataFrame:
    """
    Build the undirected genre co-occurrence edge dataframe with multiple
    edge-strength metrics.

    Args:
        df: Filtered album-level dataframe.
        min_edge_count: Minimum raw co-occurrence threshold for retaining an edge.

    Returns:
        pd.DataFrame: One row per retained genre pair with count- and
        similarity-based metrics.
    """
    available_flags = [col for col in GENRE_FLAGS if col in df.columns]
    if len(available_flags) < 2:
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

    genre_matrix = df[available_flags].fillna(0).astype(int)
    cooccurrence = genre_matrix.T @ genre_matrix
    total_albums = int(len(df))

    genre_counts = {
        genre: int(genre_matrix[genre].sum())
        for genre in available_flags
    }

    edges = []
    for i, source in enumerate(available_flags):
        for j, target in enumerate(available_flags):
            if j <= i:
                continue

            count = int(cooccurrence.loc[source, target])
            if count < min_edge_count:
                continue

            source_count = int(genre_counts[source])
            target_count = int(genre_counts[target])
            union_count = source_count + target_count - count

            pct_source = count / source_count if source_count > 0 else 0.0
            pct_target = count / target_count if target_count > 0 else 0.0
            jaccard = count / union_count if union_count > 0 else 0.0

            p_source = source_count / total_albums if total_albums > 0 else 0.0
            p_target = target_count / total_albums if total_albums > 0 else 0.0
            p_both = count / total_albums if total_albums > 0 else 0.0

            expected = p_source * p_target
            lift = p_both / expected if expected > 0 else 0.0

            edges.append(
                {
                    "source": source,
                    "target": target,
                    "pair_label": (
                        f"{get_display_label(source)} + "
                        f"{get_display_label(target)}"
                    ),
                    "count": count,
                    "source_count": source_count,
                    "target_count": target_count,
                    "union_count": union_count,
                    "pct_source": pct_source,
                    "pct_target": pct_target,
                    "jaccard": jaccard,
                    "lift": lift,
                }
            )

    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        return edge_df

    edge_df = edge_df.sort_values(
        ["count", "pair_label"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return edge_df


def build_album_genre_node_df(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the node dataframe for the album-genre chord chart.

    Args:
        edge_df: Genre-pair edge dataframe.

    Returns:
        pd.DataFrame: One row per node used in the chart.
    """
    if edge_df.empty:
        return pd.DataFrame(columns=["name", "label", "color"])

    used_genres = sorted(set(edge_df["source"]).union(edge_df["target"]))
    nodes_df = pd.DataFrame({"name": used_genres})
    nodes_df["label"] = nodes_df["name"].map(get_display_label)
    nodes_df["color"] = nodes_df["name"].map(GENRE_COLOR_MAP)
    return nodes_df


def build_edge_detail_album_table(
    df: pd.DataFrame,
    source_genre: str,
    target_genre: str,
) -> pd.DataFrame:
    """
    Build the album drilldown table for one selected genre pair.

    Args:
        df: Filtered album-level dataframe.
        source_genre: First canonical genre flag.
        target_genre: Second canonical genre flag.

    Returns:
        pd.DataFrame: Albums tagged with both genres.
    """
    if source_genre not in df.columns or target_genre not in df.columns:
        return pd.DataFrame()

    detail_df = df[
        (df[source_genre] == 1) &
        (df[target_genre] == 1)
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

def build_film_genre_edge_df(
    df: pd.DataFrame,
    min_edge_count: int,
) -> pd.DataFrame:
    """
    Build the undirected film-genre co-occurrence edge dataframe with multiple
    edge-strength metrics.

    Args:
        df: Filtered album-level dataframe.
        min_edge_count: Minimum raw co-occurrence threshold for retaining an edge.

    Returns:
        pd.DataFrame: One row per retained film-genre pair with count- and
        similarity-based metrics.
    """
    if df.empty or "film_genres" not in df.columns:
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

    exploded_rows = []

    for _, row in df.iterrows():
        genres = split_multivalue_genres(pd.Series([row["film_genres"]]))
        if not genres:
            continue

        for genre in genres:
            exploded_rows.append(
                {
                    "album_id": row.name,
                    "genre": genre,
                }
            )

    if not exploded_rows:
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

    exploded_df = pd.DataFrame(exploded_rows)

    genre_counts = (
        exploded_df.groupby("genre")["album_id"]
        .nunique()
        .to_dict()
    )

    genres_sorted = sorted(genre_counts.keys())
    total_albums = int(df.shape[0])

    # Build per-album genre sets for co-occurrence counting
    album_to_genres = (
        exploded_df.groupby("album_id")["genre"]
        .apply(lambda s: sorted(set(s)))
        .to_dict()
    )

    pair_counts: dict[tuple[str, str], int] = {}
    for genre_list in album_to_genres.values():
        for i, source in enumerate(genre_list):
            for j, target in enumerate(genre_list):
                if j <= i:
                    continue
                pair_key = (source, target)
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    edges = []
    for (source, target), count in pair_counts.items():
        if count < min_edge_count:
            continue

        source_count = int(genre_counts.get(source, 0))
        target_count = int(genre_counts.get(target, 0))
        union_count = source_count + target_count - count

        pct_source = count / source_count if source_count > 0 else 0.0
        pct_target = count / target_count if target_count > 0 else 0.0
        jaccard = count / union_count if union_count > 0 else 0.0

        p_source = source_count / total_albums if total_albums > 0 else 0.0
        p_target = target_count / total_albums if total_albums > 0 else 0.0
        p_both = count / total_albums if total_albums > 0 else 0.0

        expected = p_source * p_target
        lift = p_both / expected if expected > 0 else 0.0

        edges.append(
            {
                "source": source,
                "target": target,
                "pair_label": f"{source} + {target}",
                "count": count,
                "source_count": source_count,
                "target_count": target_count,
                "union_count": union_count,
                "pct_source": pct_source,
                "pct_target": pct_target,
                "jaccard": jaccard,
                "lift": lift,
            }
        )

    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        return edge_df

    edge_df = edge_df.sort_values(
        ["count", "pair_label"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return edge_df


def build_film_genre_node_df(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the node dataframe for the film-genre chord chart.

    Args:
        edge_df: Film-genre edge dataframe.

    Returns:
        pd.DataFrame: One row per node used in the chart.
    """
    if edge_df.empty:
        return pd.DataFrame(columns=["name", "label", "color"])

    used_genres = sorted(set(edge_df["source"]).union(edge_df["target"]))
    color_map = build_film_genre_color_map(used_genres)

    nodes_df = pd.DataFrame({"name": used_genres})
    nodes_df["label"] = nodes_df["name"].map(get_film_genre_display_label)
    nodes_df["color"] = nodes_df["name"].map(color_map)
    return nodes_df


def build_film_genre_detail_table(
    df: pd.DataFrame,
    source_genre: str,
    target_genre: str,
) -> pd.DataFrame:
    """
    Build the album drilldown table for one selected film-genre pair.

    Args:
        df: Filtered album-level dataframe.
        source_genre: First film genre.
        target_genre: Second film genre.

    Returns:
        pd.DataFrame: Albums tagged with both film genres.
    """
    if "film_genres" not in df.columns:
        return pd.DataFrame()

    detail_df = df[
        df["film_genres"].apply(
            lambda value: (
                pd.notna(value)
                and source_genre in split_multivalue_genres(pd.Series([value]))
                and target_genre in split_multivalue_genres(pd.Series([value]))
            )
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

def build_award_edge_df(
    df: pd.DataFrame,
    min_edge_count: int,
) -> pd.DataFrame:
    """
    Build the undirected award co-occurrence edge dataframe with multiple
    edge-strength metrics.

    Args:
        df: Filtered album-level dataframe.
        min_edge_count: Minimum raw co-occurrence threshold for retaining an edge.

    Returns:
        pd.DataFrame: One row per retained award pair with count- and
        similarity-based metrics.
    """
    available_flags = [col for col in AWARD_FLAGS if col in df.columns]
    if len(available_flags) < 2:
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

    award_matrix = df[available_flags].fillna(0).astype(int)
    cooccurrence = award_matrix.T @ award_matrix
    total_albums = int(len(df))

    award_counts = {
        award: int(award_matrix[award].sum())
        for award in available_flags
    }

    edges = []
    for i, source in enumerate(available_flags):
        for j, target in enumerate(available_flags):
            if j <= i:
                continue

            count = int(cooccurrence.loc[source, target])
            if count < min_edge_count:
                continue

            source_count = int(award_counts[source])
            target_count = int(award_counts[target])
            union_count = source_count + target_count - count

            pct_source = count / source_count if source_count > 0 else 0.0
            pct_target = count / target_count if target_count > 0 else 0.0
            jaccard = count / union_count if union_count > 0 else 0.0

            p_source = source_count / total_albums if total_albums > 0 else 0.0
            p_target = target_count / total_albums if total_albums > 0 else 0.0
            p_both = count / total_albums if total_albums > 0 else 0.0

            expected = p_source * p_target
            lift = p_both / expected if expected > 0 else 0.0

            edges.append(
                {
                    "source": source,
                    "target": target,
                    "pair_label": (
                        f"{get_award_display_label(source)} + "
                        f"{get_award_display_label(target)}"
                    ),
                    "count": count,
                    "source_count": source_count,
                    "target_count": target_count,
                    "union_count": union_count,
                    "pct_source": pct_source,
                    "pct_target": pct_target,
                    "jaccard": jaccard,
                    "lift": lift,
                }
            )

    edge_df = pd.DataFrame(edges)
    if edge_df.empty:
        return edge_df

    edge_df = edge_df.sort_values(
        ["count", "pair_label"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return edge_df


def build_award_node_df(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the node dataframe for the award chord chart.

    Args:
        edge_df: Award-pair edge dataframe.

    Returns:
        pd.DataFrame: One row per node used in the chart.
    """
    if edge_df.empty:
        return pd.DataFrame(columns=["name", "label", "color"])

    used_awards = sorted(set(edge_df["source"]).union(edge_df["target"]))
    nodes_df = pd.DataFrame({"name": used_awards})
    nodes_df["label"] = nodes_df["name"].map(get_award_display_label)
    nodes_df["color"] = nodes_df["name"].map(AWARD_COLOR_MAP)
    return nodes_df


def build_award_detail_table(
    df: pd.DataFrame,
    source_award: str,
    target_award: str,
) -> pd.DataFrame:
    """
    Build the album drilldown table for one selected award pair.

    Args:
        df: Filtered album-level dataframe.
        source_award: First award flag.
        target_award: Second award flag.

    Returns:
        pd.DataFrame: Albums associated with both award flags.
    """
    if source_award not in df.columns or target_award not in df.columns:
        return pd.DataFrame()

    detail_df = df[
        (df[source_award].fillna(0).astype(int) == 1) &
        (df[target_award].fillna(0).astype(int) == 1)
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
    award_cols = [col for col in AWARD_FLAGS if col in detail_df.columns]
    return detail_df[cols + award_cols].copy()


def create_top_edges_chart(
    edge_df: pd.DataFrame,
    top_n_edges: int,
    metric_name: str,
) -> alt.Chart:
    """
    Create a horizontal ranking chart of the strongest co-occurring pairs.

    Args:
        edge_df: Pair edge dataframe.
        top_n_edges: Number of top edges to show.
        metric_name: User-selected metric label.

    Returns:
        alt.Chart: Horizontal bar chart.
    """
    if edge_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    metric_col = get_edge_metric_column(metric_name)
    metric_label = get_edge_metric_display_label(metric_name)

    plot_df = edge_df.sort_values(
        [metric_col, "count", "pair_label"],
        ascending=[False, False, True],
    ).head(top_n_edges).copy()

    tooltip = [
        alt.Tooltip("pair_label:N", title="Pair"),
        alt.Tooltip("count:Q", title="Albums with Both", format=",.0f"),
        alt.Tooltip("pct_source:Q", title="% of Source", format=".1%"),
        alt.Tooltip("pct_target:Q", title="% of Target", format=".1%"),
        alt.Tooltip("jaccard:Q", title="Jaccard", format=".3f"),
        alt.Tooltip("lift:Q", title="Lift", format=".2f"),
    ]

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{metric_col}:Q",
                title=metric_label,
            ),
            y=alt.Y(
                "pair_label:N",
                sort=alt.SortField(field=metric_col, order="descending"),
                title="Relationship Pair",
            ),
            tooltip=tooltip,
        )
        .properties(
            width=750,
            height=max(320, min(700, top_n_edges * 36)),
            title={
                "text": "Top Co-occurrence Relationships",
                "subtitle": [f"Ranked by {metric_label.lower()}."],
            },
        )
    )
    return chart


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
        relationship_df: Edge dataframe for the selected mode.
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


def render_genre_chord(
    edge_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    metric_name: str,
) -> None:
    """
    Render the album-genre chord chart using HoloViews/Bokeh.

    Args:
        edge_df: Genre-pair edge dataframe.
        nodes_df: Node dataframe.
        filtered_df: Globally filtered album-level dataframe.
        metric_name: User-selected edge metric label.
    """
    if edge_df.empty:
        st.info("No genre co-occurrence edges remain under the current filters.")
        return

    if not HOLOVIEWS_AVAILABLE:
        st.warning(
            "HoloViews/Bokeh is not installed, so the chord chart cannot be "
            "rendered. You can still use the ranking chart and drilldown table "
            "below."
        )
        return

    metric_col = get_edge_metric_column(metric_name)
    total_albums = int(len(filtered_df))

    edges_plot = edge_df[["source", "target", "count"]].copy()
    edges_plot["edge_color"] = edge_df["source"].map(GENRE_COLOR_MAP)
    edges_plot["metric_value"] = edge_df[metric_col].astype(float)

    max_metric = float(edges_plot["metric_value"].max()) if not edges_plot.empty else 0.0
    if max_metric > 0:
        edges_plot["line_width_plot"] = (
            1.5 + (edges_plot["metric_value"] / max_metric) * 10.0
        )
    else:
        edges_plot["line_width_plot"] = 2.0

    chord = hv.Chord((edges_plot, hv.Dataset(nodes_df, "name"))).opts(
        opts.Chord(
            width=800,
            height=800,
            labels="label",
            label_text_font_size="10pt",
            label_text_color="white",
            bgcolor="#0b1020",
            node_color="color",
            edge_color="edge_color",
            edge_alpha=0.28,
            edge_line_width="line_width_plot",
            node_size=16,
            title=f"Album Genre Co-occurrence Chord Diagram ({metric_name})",
            tools=[],
        )
    )

    plot = hv.render(chord, backend="bokeh")
    plot.background_fill_color = "#0b1020"
    plot.border_fill_color = "#0b1020"
    plot.outline_line_color = None
    plot.title.text_color = "white"

    graph_renderer = None
    for renderer in plot.renderers:
        if isinstance(renderer, GraphRenderer):
            graph_renderer = renderer
            break

    if graph_renderer is not None:
        edge_source = graph_renderer.edge_renderer.data_source

        index_to_label = {
            idx: label
            for idx, label in enumerate(nodes_df["label"].tolist())
        }
        index_to_name = {
            idx: name
            for idx, name in enumerate(nodes_df["name"].tolist())
        }

        edge_source.data["source_label"] = [
            index_to_label.get(int(idx), str(idx))
            for idx in edge_source.data["start"]
        ]
        edge_source.data["target_label"] = [
            index_to_label.get(int(idx), str(idx))
            for idx in edge_source.data["end"]
        ]

        ranked_edges = edge_df.sort_values(
            [metric_col, "count", "pair_label"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        edge_metric_map = {}
        for rank_idx, (_, row) in enumerate(ranked_edges.iterrows(), start=1):
            pair_key = tuple(sorted((row["source"], row["target"])))
            edge_metric_map[pair_key] = {
                "count": int(row["count"]),
                "source_count": int(row["source_count"]),
                "target_count": int(row["target_count"]),
                "pct_filtered": row["count"] / total_albums if total_albums > 0 else 0.0,
                "pct_source": float(row["pct_source"]),
                "pct_target": float(row["pct_target"]),
                "jaccard": float(row["jaccard"]),
                "lift": float(row["lift"]),
                "edge_rank": rank_idx,
                "metric_value": float(row[metric_col]),
            }

        pair_metrics = [
            edge_metric_map.get(
                tuple(
                    sorted(
                        (
                            index_to_name.get(int(start_idx), str(start_idx)),
                            index_to_name.get(int(end_idx), str(end_idx)),
                        )
                    )
                ),
                {
                    "count": 0,
                    "source_count": 0,
                    "target_count": 0,
                    "pct_filtered": 0.0,
                    "pct_source": 0.0,
                    "pct_target": 0.0,
                    "jaccard": 0.0,
                    "lift": 0.0,
                    "edge_rank": 0,
                    "metric_value": 0.0,
                },
            )
            for start_idx, end_idx in zip(
                edge_source.data["start"],
                edge_source.data["end"],
            )
        ]

        edge_source.data["cooccurrence_count"] = [metric["count"] for metric in pair_metrics]
        edge_source.data["source_count"] = [metric["source_count"] for metric in pair_metrics]
        edge_source.data["target_count"] = [metric["target_count"] for metric in pair_metrics]
        edge_source.data["pct_filtered"] = [metric["pct_filtered"] for metric in pair_metrics]
        edge_source.data["pct_source"] = [metric["pct_source"] for metric in pair_metrics]
        edge_source.data["pct_target"] = [metric["pct_target"] for metric in pair_metrics]
        edge_source.data["jaccard"] = [metric["jaccard"] for metric in pair_metrics]
        edge_source.data["lift"] = [metric["lift"] for metric in pair_metrics]
        edge_source.data["edge_rank"] = [metric["edge_rank"] for metric in pair_metrics]
        edge_source.data["metric_value"] = [metric["metric_value"] for metric in pair_metrics]
        edge_source.data["metric_name"] = [metric_name] * len(pair_metrics)

        hover = HoverTool(
            renderers=[graph_renderer.edge_renderer],
            tooltips=[
                ("Source", "@source_label"),
                ("Target", "@target_label"),
                ("Albums with both", "@cooccurrence_count{,}"),
                ("% of filtered albums", "@pct_filtered{0.0%}"),
                ("Albums with source genre", "@source_count{,}"),
                ("Albums with target genre", "@target_count{,}"),
                ("% of source with target", "@pct_source{0.0%}"),
                ("% of target with source", "@pct_target{0.0%}"),
                ("Jaccard", "@jaccard{0.000}"),
                ("Lift", "@lift{0.00}"),
                ("Edge rank", "@edge_rank"),
                ("Selected metric", "@metric_name"),
                ("Metric value", "@metric_value{0.000}"),
            ],
        )
        plot.add_tools(hover)
        plot.toolbar.active_inspect = hover

    streamlit_bokeh(
        plot,
        use_container_width=True,
        theme="streamlit",
        key="cooccurrence_genre_chord",
    )

def render_film_genre_chord(
    edge_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    metric_name: str,
) -> None:
    """
    Render the film-genre chord chart using HoloViews/Bokeh.

    Args:
        edge_df: Film-genre edge dataframe.
        nodes_df: Node dataframe.
        filtered_df: Globally filtered album-level dataframe.
        metric_name: User-selected edge metric label.
    """
    if edge_df.empty:
        st.info("No film-genre co-occurrence edges remain under the current filters.")
        return

    if not HOLOVIEWS_AVAILABLE:
        st.warning(
            "HoloViews/Bokeh is not installed, so the chord chart cannot be "
            "rendered. You can still use the ranking chart and drilldown table "
            "below."
        )
        return

    metric_col = get_edge_metric_column(metric_name)
    total_albums = int(len(filtered_df))
    color_map = build_film_genre_color_map(nodes_df["name"].tolist())

    edges_plot = edge_df[["source", "target", "count"]].copy()
    edges_plot["edge_color"] = edge_df["source"].map(color_map)
    edges_plot["metric_value"] = edge_df[metric_col].astype(float)

    max_metric = float(edges_plot["metric_value"].max()) if not edges_plot.empty else 0.0
    if max_metric > 0:
        edges_plot["line_width_plot"] = (
            1.5 + (edges_plot["metric_value"] / max_metric) * 10.0
        )
    else:
        edges_plot["line_width_plot"] = 2.0

    chord = hv.Chord((edges_plot, hv.Dataset(nodes_df, "name"))).opts(
        opts.Chord(
            width=800,
            height=800,
            labels="label",
            label_text_font_size="10pt",
            label_text_color="white",
            bgcolor="#0b1020",
            node_color="color",
            edge_color="edge_color",
            edge_alpha=0.28,
            edge_line_width="line_width_plot",
            node_size=16,
            title=f"Film Genre Co-occurrence Chord Diagram ({metric_name})",
            tools=[],
        )
    )

    plot = hv.render(chord, backend="bokeh")
    plot.background_fill_color = "#0b1020"
    plot.border_fill_color = "#0b1020"
    plot.outline_line_color = None
    plot.title.text_color = "white"

    graph_renderer = None
    for renderer in plot.renderers:
        if isinstance(renderer, GraphRenderer):
            graph_renderer = renderer
            break

    if graph_renderer is not None:
        edge_source = graph_renderer.edge_renderer.data_source

        index_to_label = {
            idx: label
            for idx, label in enumerate(nodes_df["label"].tolist())
        }
        index_to_name = {
            idx: name
            for idx, name in enumerate(nodes_df["name"].tolist())
        }

        edge_source.data["source_label"] = [
            index_to_label.get(int(idx), str(idx))
            for idx in edge_source.data["start"]
        ]
        edge_source.data["target_label"] = [
            index_to_label.get(int(idx), str(idx))
            for idx in edge_source.data["end"]
        ]

        ranked_edges = edge_df.sort_values(
            [metric_col, "count", "pair_label"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        edge_metric_map = {}
        for rank_idx, (_, row) in enumerate(ranked_edges.iterrows(), start=1):
            pair_key = tuple(sorted((row["source"], row["target"])))
            edge_metric_map[pair_key] = {
                "count": int(row["count"]),
                "source_count": int(row["source_count"]),
                "target_count": int(row["target_count"]),
                "pct_filtered": row["count"] / total_albums if total_albums > 0 else 0.0,
                "pct_source": float(row["pct_source"]),
                "pct_target": float(row["pct_target"]),
                "jaccard": float(row["jaccard"]),
                "lift": float(row["lift"]),
                "edge_rank": rank_idx,
                "metric_value": float(row[metric_col]),
            }

        pair_metrics = [
            edge_metric_map.get(
                tuple(
                    sorted(
                        (
                            index_to_name.get(int(start_idx), str(start_idx)),
                            index_to_name.get(int(end_idx), str(end_idx)),
                        )
                    )
                ),
                {
                    "count": 0,
                    "source_count": 0,
                    "target_count": 0,
                    "pct_filtered": 0.0,
                    "pct_source": 0.0,
                    "pct_target": 0.0,
                    "jaccard": 0.0,
                    "lift": 0.0,
                    "edge_rank": 0,
                    "metric_value": 0.0,
                },
            )
            for start_idx, end_idx in zip(
                edge_source.data["start"],
                edge_source.data["end"],
            )
        ]

        edge_source.data["cooccurrence_count"] = [metric["count"] for metric in pair_metrics]
        edge_source.data["source_count"] = [metric["source_count"] for metric in pair_metrics]
        edge_source.data["target_count"] = [metric["target_count"] for metric in pair_metrics]
        edge_source.data["pct_filtered"] = [metric["pct_filtered"] for metric in pair_metrics]
        edge_source.data["pct_source"] = [metric["pct_source"] for metric in pair_metrics]
        edge_source.data["pct_target"] = [metric["pct_target"] for metric in pair_metrics]
        edge_source.data["jaccard"] = [metric["jaccard"] for metric in pair_metrics]
        edge_source.data["lift"] = [metric["lift"] for metric in pair_metrics]
        edge_source.data["edge_rank"] = [metric["edge_rank"] for metric in pair_metrics]
        edge_source.data["metric_value"] = [metric["metric_value"] for metric in pair_metrics]
        edge_source.data["metric_name"] = [metric_name] * len(pair_metrics)

        hover = HoverTool(
            renderers=[graph_renderer.edge_renderer],
            tooltips=[
                ("Source", "@source_label"),
                ("Target", "@target_label"),
                ("Albums with both", "@cooccurrence_count{,}"),
                ("% of filtered albums", "@pct_filtered{0.0%}"),
                ("Albums with source genre", "@source_count{,}"),
                ("Albums with target genre", "@target_count{,}"),
                ("% of source with target", "@pct_source{0.0%}"),
                ("% of target with source", "@pct_target{0.0%}"),
                ("Jaccard", "@jaccard{0.000}"),
                ("Lift", "@lift{0.00}"),
                ("Edge rank", "@edge_rank"),
                ("Selected metric", "@metric_name"),
                ("Metric value", "@metric_value{0.000}"),
            ],
        )
        plot.add_tools(hover)
        plot.toolbar.active_inspect = hover

    streamlit_bokeh(
        plot,
        use_container_width=True,
        theme="streamlit",
        key="cooccurrence_film_genre_chord",
    )

def render_award_chord(
    edge_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    metric_name: str,
) -> None:
    """
    Render the award co-occurrence chord chart using HoloViews/Bokeh.

    Args:
        edge_df: Award-pair edge dataframe.
        nodes_df: Award node dataframe.
        filtered_df: Globally filtered album-level dataframe.
        metric_name: User-selected edge metric label.
    """
    if edge_df.empty:
        st.info("No award co-occurrence edges remain under the current filters.")
        return

    if not HOLOVIEWS_AVAILABLE:
        st.warning(
            "HoloViews/Bokeh is not installed, so the chord chart cannot be "
            "rendered. You can still use the ranking chart and drilldown table "
            "below."
        )
        return

    metric_col = get_edge_metric_column(metric_name)
    total_albums = int(len(filtered_df))

    edges_plot = edge_df[["source", "target", "count"]].copy()
    edges_plot["edge_color"] = edge_df["source"].map(AWARD_COLOR_MAP)
    edges_plot["metric_value"] = edge_df[metric_col].astype(float)

    max_metric = float(edges_plot["metric_value"].max()) if not edges_plot.empty else 0.0
    if max_metric > 0:
        edges_plot["line_width_plot"] = (
            1.5 + (edges_plot["metric_value"] / max_metric) * 10.0
        )
    else:
        edges_plot["line_width_plot"] = 2.0

    chord = hv.Chord((edges_plot, hv.Dataset(nodes_df, "name"))).opts(
        opts.Chord(
            width=800,
            height=800,
            labels="label",
            label_text_font_size="10pt",
            label_text_color="white",
            bgcolor="#0b1020",
            node_color="color",
            edge_color="edge_color",
            edge_alpha=0.28,
            edge_line_width="line_width_plot",
            node_size=16,
            title=f"Awards Co-occurrence Chord Diagram ({metric_name})",
            tools=[],
        )
    )

    plot = hv.render(chord, backend="bokeh")
    plot.background_fill_color = "#0b1020"
    plot.border_fill_color = "#0b1020"
    plot.outline_line_color = None
    plot.title.text_color = "white"

    graph_renderer = None
    for renderer in plot.renderers:
        if isinstance(renderer, GraphRenderer):
            graph_renderer = renderer
            break

    if graph_renderer is not None:
        edge_source = graph_renderer.edge_renderer.data_source

        index_to_label = {
            idx: label
            for idx, label in enumerate(nodes_df["label"].tolist())
        }
        index_to_name = {
            idx: name
            for idx, name in enumerate(nodes_df["name"].tolist())
        }

        edge_source.data["source_label"] = [
            index_to_label.get(int(idx), str(idx))
            for idx in edge_source.data["start"]
        ]
        edge_source.data["target_label"] = [
            index_to_label.get(int(idx), str(idx))
            for idx in edge_source.data["end"]
        ]

        ranked_edges = edge_df.sort_values(
            [metric_col, "count", "pair_label"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        edge_metric_map = {}
        for rank_idx, (_, row) in enumerate(ranked_edges.iterrows(), start=1):
            pair_key = tuple(sorted((row["source"], row["target"])))
            edge_metric_map[pair_key] = {
                "count": int(row["count"]),
                "source_count": int(row["source_count"]),
                "target_count": int(row["target_count"]),
                "pct_filtered": row["count"] / total_albums if total_albums > 0 else 0.0,
                "pct_source": float(row["pct_source"]),
                "pct_target": float(row["pct_target"]),
                "jaccard": float(row["jaccard"]),
                "lift": float(row["lift"]),
                "edge_rank": rank_idx,
                "metric_value": float(row[metric_col]),
            }

        pair_metrics = [
            edge_metric_map.get(
                tuple(
                    sorted(
                        (
                            index_to_name.get(int(start_idx), str(start_idx)),
                            index_to_name.get(int(end_idx), str(end_idx)),
                        )
                    )
                ),
                {
                    "count": 0,
                    "source_count": 0,
                    "target_count": 0,
                    "pct_filtered": 0.0,
                    "pct_source": 0.0,
                    "pct_target": 0.0,
                    "jaccard": 0.0,
                    "lift": 0.0,
                    "edge_rank": 0,
                    "metric_value": 0.0,
                },
            )
            for start_idx, end_idx in zip(
                edge_source.data["start"],
                edge_source.data["end"],
            )
        ]

        edge_source.data["cooccurrence_count"] = [metric["count"] for metric in pair_metrics]
        edge_source.data["source_count"] = [metric["source_count"] for metric in pair_metrics]
        edge_source.data["target_count"] = [metric["target_count"] for metric in pair_metrics]
        edge_source.data["pct_filtered"] = [metric["pct_filtered"] for metric in pair_metrics]
        edge_source.data["pct_source"] = [metric["pct_source"] for metric in pair_metrics]
        edge_source.data["pct_target"] = [metric["pct_target"] for metric in pair_metrics]
        edge_source.data["jaccard"] = [metric["jaccard"] for metric in pair_metrics]
        edge_source.data["lift"] = [metric["lift"] for metric in pair_metrics]
        edge_source.data["edge_rank"] = [metric["edge_rank"] for metric in pair_metrics]
        edge_source.data["metric_value"] = [metric["metric_value"] for metric in pair_metrics]
        edge_source.data["metric_name"] = [metric_name] * len(pair_metrics)

        hover = HoverTool(
            renderers=[graph_renderer.edge_renderer],
            tooltips=[
                ("Source", "@source_label"),
                ("Target", "@target_label"),
                ("Albums with both", "@cooccurrence_count{,}"),
                ("% of filtered albums", "@pct_filtered{0.0%}"),
                ("Albums with source award", "@source_count{,}"),
                ("Albums with target award", "@target_count{,}"),
                ("% of source with target", "@pct_source{0.0%}"),
                ("% of target with source", "@pct_target{0.0%}"),
                ("Jaccard", "@jaccard{0.000}"),
                ("Lift", "@lift{0.00}"),
                ("Edge rank", "@edge_rank"),
                ("Selected metric", "@metric_name"),
                ("Metric value", "@metric_value{0.000}"),
            ],
        )
        plot.add_tools(hover)
        plot.toolbar.active_inspect = hover

    streamlit_bokeh(
        plot,
        use_container_width=True,
        theme="streamlit",
        key="cooccurrence_award_chord",
    )


def main() -> None:
    """
    Render the Co-occurrence Explorer page.
    """
    apply_app_styles()

    st.title("Co-occurrence Explorer")
    st.write(
        """
        Explore self-entity co-occurrence structure across soundtrack albums.
        This page focuses on relationships within the same entity family, such
        as album genres co-occurring with other album genres, or award
        categories co-occurring with other award categories.
        """
    )

    explorer_df = load_explorer_data()
    cooccurrence_df = build_cooccurrence_explorer_df(explorer_df)

    min_year = int(cooccurrence_df["film_year"].dropna().min())
    max_year = int(cooccurrence_df["film_year"].dropna().max())

    film_genre_options = split_multivalue_genres(cooccurrence_df["film_genres"])
    album_genre_options = split_multivalue_genres(
        cooccurrence_df["album_genres_display"]
    )

    global_filters = get_global_filter_controls(
        min_year=min_year,
        max_year=max_year,
        film_genre_options=film_genre_options,
        album_genre_options=album_genre_options,
    )

    controls = get_cooccurrence_controls(
        relationship_options=get_relationship_type_options(),
    )

    filtered_df = filter_dataset(cooccurrence_df, global_filters)

    if filtered_df.empty:
        st.warning("No albums remain after applying the current global filters.")
        return

    relationship_type = controls["relationship_type"]

    if relationship_type == "Album Genre ↔ Album Genre":
        edge_df = build_album_genre_edge_df(
            df=filtered_df,
            min_edge_count=controls["min_edge_count"],
        )

        if edge_df.empty:
            st.warning(
                "No genre-pair edges remain after applying the current filters "
                "and minimum co-occurrence threshold."
            )
            return

        nodes_df = build_album_genre_node_df(edge_df)

        render_summary_metrics(
            filtered_df=filtered_df,
            relationship_df=edge_df,
            metric_name=controls["edge_metric"],
            relationship_label="Edges Shown",
        )

        st.subheader("Chord Diagram")
        render_genre_chord(
            edge_df=edge_df,
            nodes_df=nodes_df,
            filtered_df=filtered_df,
            metric_name=controls["edge_metric"],
        )
        st.caption(
            "Each ribbon represents an album-genre pair retained under the "
            "current minimum co-occurrence threshold. Ribbon thickness reflects "
            "the selected edge metric."
        )

        st.subheader("Top Genre Pairings")
        top_edges_chart = create_top_edges_chart(
            edge_df=edge_df,
            top_n_edges=controls["top_n_edges"],
            metric_name=controls["edge_metric"],
        )
        st.altair_chart(top_edges_chart, width="stretch")

        if controls["show_edge_table"]:
            st.markdown("#### Relationship Summary Table")

            metric_col = get_edge_metric_column(controls["edge_metric"])

            edge_table = edge_df.sort_values(
                [metric_col, "count", "pair_label"],
                ascending=[False, False, True],
            ).copy()

            edge_table["source"] = edge_table["source"].map(get_display_label)
            edge_table["target"] = edge_table["target"].map(get_display_label)

            edge_table = edge_table.rename(
                columns={
                    "source": "Source Genre",
                    "target": "Target Genre",
                    "pair_label": "Genre Pair",
                    "count": "Albums with Both",
                    "source_count": "Albums with Source Genre",
                    "target_count": "Albums with Target Genre",
                    "union_count": "Albums in Union",
                    "pct_source": "% of Source Genre",
                    "pct_target": "% of Target Genre",
                    "jaccard": "Jaccard",
                    "lift": "Lift",
                }
            )

            st.dataframe(
                edge_table,
                width="stretch",
                hide_index=True,
            )

        st.subheader("Inspect a Genre Pair")
        pair_options = edge_df["pair_label"].tolist()
        selected_pair_label = st.selectbox(
            "Genre pair",
            options=pair_options,
            index=0,
            key="genre_pair_select",
        )

        selected_row = edge_df[edge_df["pair_label"] == selected_pair_label].iloc[0]
        source_genre = selected_row["source"]
        target_genre = selected_row["target"]

        st.caption(
            f"Showing albums tagged with both "
            f"**{get_display_label(source_genre)}** and "
            f"**{get_display_label(target_genre)}**."
        )

        detail_df = build_edge_detail_album_table(
            df=filtered_df,
            source_genre=source_genre,
            target_genre=target_genre,
        )

        if detail_df.empty:
            st.info("No albums match the selected genre pair.")
            return

        if controls["show_album_table"]:
            st.dataframe(
                rename_columns_for_display(detail_df),
                width="stretch",
                hide_index=True,
            )

    elif relationship_type == "Film Genre ↔ Film Genre":
        edge_df = build_film_genre_edge_df(
            df=filtered_df,
            min_edge_count=controls["min_edge_count"],
        )

        if edge_df.empty:
            st.warning(
                "No film-genre pair edges remain after applying the current "
                "filters and minimum co-occurrence threshold."
            )
            return

        nodes_df = build_film_genre_node_df(edge_df)

        render_summary_metrics(
            filtered_df=filtered_df,
            relationship_df=edge_df,
            metric_name=controls["edge_metric"],
            relationship_label="Film-Genre Pairs Shown",
        )

        st.subheader("Film Genre Chord Diagram")
        render_film_genre_chord(
            edge_df=edge_df,
            nodes_df=nodes_df,
            filtered_df=filtered_df,
            metric_name=controls["edge_metric"],
        )
        st.caption(
            "Each ribbon represents a pair of film genres that co-occur on the "
            "same filtered soundtrack albums. Ribbon thickness reflects the "
            "selected edge metric."
        )

        st.subheader("Top Film Genre Pairings")
        top_edges_chart = create_top_edges_chart(
            edge_df=edge_df,
            top_n_edges=controls["top_n_edges"],
            metric_name=controls["edge_metric"],
        )
        st.altair_chart(top_edges_chart, width="stretch")

        if controls["show_edge_table"]:
            st.markdown("#### Film Genre Relationship Summary Table")

            metric_col = get_edge_metric_column(controls["edge_metric"])

            edge_table = edge_df.sort_values(
                [metric_col, "count", "pair_label"],
                ascending=[False, False, True],
            ).copy()

            edge_table = edge_table.rename(
                columns={
                    "source": "Source Film Genre",
                    "target": "Target Film Genre",
                    "pair_label": "Film Genre Pair",
                    "count": "Albums with Both",
                    "source_count": "Albums with Source Genre",
                    "target_count": "Albums with Target Genre",
                    "union_count": "Albums in Union",
                    "pct_source": "% of Source Genre",
                    "pct_target": "% of Target Genre",
                    "jaccard": "Jaccard",
                    "lift": "Lift",
                }
            )

            st.dataframe(
                edge_table,
                width="stretch",
                hide_index=True,
            )

        st.subheader("Inspect a Film Genre Pair")
        pair_options = edge_df["pair_label"].tolist()
        selected_pair_label = st.selectbox(
            "Film genre pair",
            options=pair_options,
            index=0,
            key="film_genre_pair_select",
        )

        selected_row = edge_df[edge_df["pair_label"] == selected_pair_label].iloc[0]
        source_genre = selected_row["source"]
        target_genre = selected_row["target"]

        st.caption(
            f"Showing albums associated with both "
            f"**{source_genre}** and **{target_genre}**."
        )

        detail_df = build_film_genre_detail_table(
            df=filtered_df,
            source_genre=source_genre,
            target_genre=target_genre,
        )

        if detail_df.empty:
            st.info("No albums match the selected film genre pair.")
            return

        if controls["show_album_table"]:
            st.dataframe(
                rename_columns_for_display(detail_df),
                width="stretch",
                hide_index=True,
            )

    elif relationship_type == "Awards ↔ Awards":
        award_df = build_award_edge_df(
            df=filtered_df,
            min_edge_count=controls["min_edge_count"],
        )

        if award_df.empty:
            st.warning(
                "No award-pair edges remain after applying the current filters "
                "and minimum co-occurrence threshold."
            )
            return

        award_nodes_df = build_award_node_df(award_df)

        render_summary_metrics(
            filtered_df=filtered_df,
            relationship_df=award_df,
            metric_name=controls["edge_metric"],
            relationship_label="Award Pairs Shown",
        )

        st.subheader("Awards Chord Diagram")
        render_award_chord(
            edge_df=award_df,
            nodes_df=award_nodes_df,
            filtered_df=filtered_df,
            metric_name=controls["edge_metric"],
        )
        st.caption(
            "Each ribbon represents a pair of award categories that co-occur on "
            "the same filtered soundtrack albums. Ribbon thickness reflects the "
            "selected edge metric."
        )

        st.subheader("Top Award Pairings")
        top_awards_chart = create_top_edges_chart(
            edge_df=award_df,
            top_n_edges=controls["top_n_edges"],
            metric_name=controls["edge_metric"],
        )
        st.altair_chart(top_awards_chart, width="stretch")

        if controls["show_edge_table"]:
            st.markdown("#### Award Relationship Summary Table")

            metric_col = get_edge_metric_column(controls["edge_metric"])

            award_table = award_df.sort_values(
                [metric_col, "count", "pair_label"],
                ascending=[False, False, True],
            ).copy()

            award_table["source"] = award_table["source"].map(get_award_display_label)
            award_table["target"] = award_table["target"].map(get_award_display_label)

            award_table = award_table.rename(
                columns={
                    "source": "Source Award",
                    "target": "Target Award",
                    "pair_label": "Award Pair",
                    "count": "Albums with Both",
                    "source_count": "Albums with Source Award",
                    "target_count": "Albums with Target Award",
                    "union_count": "Albums in Union",
                    "pct_source": "% of Source Award",
                    "pct_target": "% of Target Award",
                    "jaccard": "Jaccard",
                    "lift": "Lift",
                }
            )

            st.dataframe(
                award_table,
                width="stretch",
                hide_index=True,
            )

        st.subheader("Inspect an Award Pair")
        pair_options = award_df["pair_label"].tolist()
        selected_pair_label = st.selectbox(
            "Award pair",
            options=pair_options,
            index=0,
            key="award_pair_select",
        )

        selected_row = award_df[award_df["pair_label"] == selected_pair_label].iloc[0]
        source_award = selected_row["source"]
        target_award = selected_row["target"]

        st.caption(
            f"Showing albums associated with both "
            f"**{get_award_display_label(source_award)}** and "
            f"**{get_award_display_label(target_award)}**."
        )

        detail_df = build_award_detail_table(
            df=filtered_df,
            source_award=source_award,
            target_award=target_award,
        )

        if detail_df.empty:
            st.info("No albums match the selected award pair.")
            return

        if controls["show_album_table"]:
            st.dataframe(
                rename_columns_for_display(detail_df),
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    main()