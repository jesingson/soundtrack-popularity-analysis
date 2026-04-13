from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from app.app_controls import (
    get_global_filter_controls,
    get_track_ecosystem_controls,
)
from app.app_data import load_track_data_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_film_year_bucket,
    add_standard_multivalue_groups,
    get_clean_composer_options,
    get_global_filter_inputs,
    get_track_page_display_label,
    rename_track_page_columns_for_display,
    select_unique_existing_columns,
)
from app.ui import apply_app_styles


CORE_ARCHETYPE_COLS = [
    "track_intensity_band",
    "track_acoustic_orchestral_band",
    "track_speech_texture_band",
]

BOOLEAN_ARCHETYPE_COLS = [
    "is_instrumental",
    "is_high_energy",
    "is_high_happiness",
]

PAIR_DETAIL_ALBUM_COLS = [
    "film_title",
    "album_title",
    "composer_primary_clean",
    "label_names",
    "film_year",
    "film_genres",
    "album_genres_display",
    "award_category",
    "n_tracks",
    "lfm_album_listeners",
    "lfm_album_playcount",
    "album_cohesion_score",
]

PAIR_DETAIL_TRACK_COLS = [
    "film_title",
    "album_title",
    "track_title",
    "track_number",
    "track_position_bucket",
    "composer_primary_clean",
    "lfm_track_listeners",
    "lfm_track_playcount",
    "spotify_popularity",
    "track_intensity_band",
    "track_acoustic_orchestral_band",
    "track_speech_texture_band",
    "is_instrumental",
    "is_high_energy",
    "is_high_happiness",
]


def add_track_ecosystem_display_fields(track_df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped display fields used by the Track Ecosystem Explorer."""
    df = add_standard_multivalue_groups(track_df)
    df = add_film_year_bucket(df)

    if "composer_primary_clean" in df.columns:
        df["composer_primary_clean"] = (
            df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "label_names" in df.columns:
        df["label_names"] = (
            df["label_names"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


def filter_track_ecosystem_df(
    track_df: pd.DataFrame,
    global_controls: dict,
    controls: dict,
) -> pd.DataFrame:
    """Apply shared global filters plus Page 34-specific track filters."""
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
            col for col in ["energy", "danceability", "happiness", "instrumentalness"]
            if col in filtered.columns
        ]
        if required_audio_cols:
            filtered = filtered.dropna(subset=required_audio_cols).copy()

    return filtered


def build_filter_context_caption(controls: dict) -> str:
    """Build a short caption describing the current filter scope."""
    parts = []

    if controls.get("selected_composers"):
        shown = ", ".join(controls["selected_composers"][:3])
        if len(controls["selected_composers"]) > 3:
            shown += ", ..."
        parts.append(f"Composers: {shown}")

    if controls["min_album_listeners"] > 0:
        parts.append(f"Min album listeners: {controls['min_album_listeners']:,}")

    parts.append(f"Max track position: {controls['max_track_position']}")

    if controls["audio_only"]:
        parts.append("Only tracks with core audio features")

    return " | ".join(parts)


def get_active_archetype_labels(
    df: pd.DataFrame,
    ecosystem_mode: str,
) -> dict[str, pd.Series]:
    """
    Build a dict of album-presence-ready archetype labels.

    Returns mapping:
        label -> boolean Series at track grain
    """
    archetypes: dict[str, pd.Series] = {}

    clean_map = {
        "track_intensity_band": "Intensity",
        "track_acoustic_orchestral_band": "Acoustic",
        "track_speech_texture_band": "Speech",
    }

    for col in CORE_ARCHETYPE_COLS:
        if col not in df.columns:
            continue

        display_family = clean_map.get(col, col)
        series = df[col].fillna("Unknown").astype(str)

        # Deliberately exclude "Medium" because it dominates and washes out the page.
        for level in ["High", "Low"]:
            label = f"{display_family}: {level}"
            archetypes[label] = series.eq(level)

    if ecosystem_mode == "Core archetypes + boolean flags":
        boolean_label_map = {
            "is_instrumental": "Instrumental",
            "is_high_energy": "High Energy",
            "is_high_happiness": "High Happiness",
        }

        for col in BOOLEAN_ARCHETYPE_COLS:
            if col not in df.columns:
                continue

            display_label = boolean_label_map.get(col, col)
            values = pd.to_numeric(df[col], errors="coerce").fillna(0)
            archetypes[display_label] = values.eq(1)

    return archetypes


def build_album_archetype_presence_df(
    df: pd.DataFrame,
    ecosystem_mode: str,
) -> pd.DataFrame:
    """Collapse track-grain archetypes into one-row-per-album presence flags."""
    group_keys = ["release_group_mbid", "tmdb_id"]

    archetype_map = get_active_archetype_labels(df, ecosystem_mode)
    if not archetype_map:
        return pd.DataFrame(columns=group_keys)

    base_cols = [col for col in group_keys if col in df.columns]
    presence_df = df[base_cols].drop_duplicates().copy()

    working_df = df[base_cols].copy()
    for label, mask in archetype_map.items():
        working_df[label] = mask.astype(int)

    grouped = (
        working_df.groupby(group_keys, dropna=False)
        .max()
        .reset_index()
    )

    return grouped


def build_pair_metrics_df(
    album_presence_df: pd.DataFrame,
    min_album_count: int,
) -> pd.DataFrame:
    """Build album-level archetype pair metrics."""
    group_keys = ["release_group_mbid", "tmdb_id"]
    archetype_cols = [col for col in album_presence_df.columns if col not in group_keys]

    if len(archetype_cols) < 2:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "pair_label",
                "count",
                "source_count",
                "target_count",
                "pct_albums",
                "lift",
            ]
        )

    total_albums = len(album_presence_df)
    rows = []

    for i, source in enumerate(archetype_cols):
        source_mask = album_presence_df[source].fillna(0).astype(int) == 1
        source_count = int(source_mask.sum())

        if source_count == 0:
            continue

        for target in archetype_cols[i + 1:]:
            target_mask = album_presence_df[target].fillna(0).astype(int) == 1
            target_count = int(target_mask.sum())

            if target_count == 0:
                continue

            both_count = int((source_mask & target_mask).sum())
            if both_count < min_album_count:
                continue

            pct_albums = both_count / total_albums if total_albums > 0 else 0.0

            p_source = source_count / total_albums if total_albums > 0 else 0.0
            p_target = target_count / total_albums if total_albums > 0 else 0.0
            p_both = both_count / total_albums if total_albums > 0 else 0.0
            expected = p_source * p_target
            lift = p_both / expected if expected > 0 else 0.0

            rows.append(
                {
                    "source": source,
                    "target": target,
                    "pair_label": f"{source} + {target}",
                    "count": both_count,
                    "source_count": source_count,
                    "target_count": target_count,
                    "pct_albums": pct_albums,
                    "lift": lift,
                    "lift_delta": lift - 1.0,
                }
            )

    pair_df = pd.DataFrame(rows)

    if pair_df.empty:
        return pair_df

    return pair_df.sort_values(
        ["lift", "count", "pair_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def get_pair_metric_column(metric_name: str) -> str:
    """Map UI metric label to pair dataframe column."""
    return {
        "Lift": "lift",
        "Lift vs Random": "lift_delta",  # <-- ADD
        "Count": "count",
        "% of albums": "pct_albums",
    }[metric_name]


def build_matrix_df(pair_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Build long-form matrix dataframe for the ecosystem heatmap."""
    if pair_df.empty:
        return pd.DataFrame(columns=["source", "target", "metric_value"])

    metric_col = get_pair_metric_column(metric_name)

    cols = [
        "source",
        "target",
        "count",
        "pct_albums",
        "lift",
        "lift_delta",
    ]
    existing_cols = [col for col in cols if col in pair_df.columns]

    forward = pair_df[existing_cols].copy()
    forward["metric_value"] = forward[metric_col]

    reverse = forward.copy()
    reverse["source"], reverse["target"] = reverse["target"], reverse["source"]

    matrix_df = pd.concat([forward, reverse], ignore_index=True)

    return matrix_df

def get_ecosystem_label_order(pair_df: pd.DataFrame) -> list[str]:
    """Return a stable, human-readable ordering for ecosystem labels."""
    preferred_order = [
        "Intensity: High",
        "Intensity: Low",
        "Acoustic: High",
        "Acoustic: Low",
        "Speech: High",
        "Speech: Low",
        "Instrumental",
        "High Energy",
        "High Happiness",
    ]

    visible = set(pair_df["source"]).union(pair_df["target"])
    ordered = [label for label in preferred_order if label in visible]

    extras = sorted(visible - set(ordered))
    return ordered + extras

def create_ecosystem_matrix_chart(pair_df: pd.DataFrame, metric_name: str) -> alt.Chart:
    """Create the Track Ecosystem Matrix heatmap."""
    matrix_df = build_matrix_df(pair_df, metric_name)
    labels = get_ecosystem_label_order(pair_df)

    chart_size = max(480, min(820, 95 * len(labels)))

    if metric_name == "Lift vs Random":
        max_value = float(matrix_df["metric_value"].max()) if not matrix_df.empty else 0.0
        color = alt.Color(
            "metric_value:Q",
            scale=alt.Scale(
                domain=[0.0, max(0.02, max_value)],
                scheme="blues",
            ),
            title=metric_name,
        )
    elif metric_name == "Lift":
        max_value = float(matrix_df["metric_value"].max()) if not matrix_df.empty else 1.0
        color = alt.Color(
            "metric_value:Q",
            scale=alt.Scale(
                domain=[1.0, max(1.02, max_value)],
                scheme="blues",
            ),
            title=metric_name,
        )
    elif metric_name == "% of albums":
        max_value = float(matrix_df["metric_value"].max()) if not matrix_df.empty else 0.0
        color = alt.Color(
            "metric_value:Q",
            scale=alt.Scale(
                domain=[0.0, max_value],
                scheme="blues",
            ),
            title=metric_name,
        )
    else:
        max_value = float(matrix_df["metric_value"].max()) if not matrix_df.empty else 0.0
        color = alt.Color(
            "metric_value:Q",
            scale=alt.Scale(
                domain=[0.0, max_value],
                scheme="blues",
            ),
            title=metric_name,
        )

    return (
        alt.Chart(matrix_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "source:N",
                sort=labels,
                title=None,
                axis=alt.Axis(
                    labelAngle=-45,
                    labelLimit=400,
                    labelOverlap=False,
                ),
            ),
            y=alt.Y(
                "target:N",
                sort=labels,
                title=None,
                axis=alt.Axis(
                    labelLimit=240,
                    labelOverlap=False,
                ),
            ),
            color=color,
            tooltip=[
                alt.Tooltip("source:N", title="Archetype A"),
                alt.Tooltip("target:N", title="Archetype B"),
                alt.Tooltip("count:Q", title="Albums with Both", format=",.0f"),
                alt.Tooltip("pct_albums:Q", title="% of Albums", format=".1%"),
                alt.Tooltip("lift:Q", title="Lift", format=".2f"),
                alt.Tooltip("lift_delta:Q", title="Lift vs Random", format=".3f"),
            ],
        )
        .properties(
            width=chart_size,
            height=chart_size,
        )
    )


def create_top_pairs_chart(
    pair_df: pd.DataFrame,
    metric_name: str,
    top_n_pairs: int,
) -> alt.Chart:
    """Create ranked archetype-pair chart."""
    metric_col = get_pair_metric_column(metric_name)
    plot_df = pair_df.sort_values(
        [metric_col, "count", "pair_label"],
        ascending=[False, False, True],
    ).head(top_n_pairs).copy()

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "pair_label:N",
                sort=alt.SortField(field=metric_col, order="descending"),
                title="Archetype Pair",
            ),
            x=alt.X(f"{metric_col}:Q", title=metric_name),
            tooltip=[
                alt.Tooltip("pair_label:N", title="Pair"),
                alt.Tooltip("count:Q", title="Albums with Both", format=",.0f"),
                alt.Tooltip("pct_albums:Q", title="% of Albums", format=".1%"),
                alt.Tooltip("lift:Q", title="Lift", format=".2f"),
                alt.Tooltip("lift_delta:Q", title="Lift vs Random", format=".3f"),
            ],
        )
        .properties(
            height=max(320, 34 * len(plot_df)),
            title="Top Archetype Pairings",
        )
    )


def build_selected_pair_album_table(
    filtered_df: pd.DataFrame,
    album_presence_df: pd.DataFrame,
    source_label: str,
    target_label: str,
) -> pd.DataFrame:
    """Return albums containing both selected archetype labels."""
    group_keys = ["release_group_mbid", "tmdb_id"]

    keep_keys = album_presence_df[
        (album_presence_df[source_label] == 1) & (album_presence_df[target_label] == 1)
    ][group_keys].drop_duplicates()

    if keep_keys.empty:
        return pd.DataFrame()

    album_cols = select_unique_existing_columns(filtered_df, group_keys + PAIR_DETAIL_ALBUM_COLS)
    album_df = (
        filtered_df[album_cols]
        .drop_duplicates(subset=group_keys)
        .merge(keep_keys, on=group_keys, how="inner")
    )

    if "lfm_album_listeners" in album_df.columns:
        album_df = album_df.sort_values(
            "lfm_album_listeners",
            ascending=False,
            na_position="last",
        )

    return album_df


def build_selected_pair_track_table(
    filtered_df: pd.DataFrame,
    album_presence_df: pd.DataFrame,
    source_label: str,
    target_label: str,
    ecosystem_mode: str,
) -> pd.DataFrame:
    """Return track rows from albums containing both selected archetype labels."""
    group_keys = ["release_group_mbid", "tmdb_id"]

    keep_keys = album_presence_df[
        (album_presence_df[source_label] == 1) & (album_presence_df[target_label] == 1)
    ][group_keys].drop_duplicates()

    if keep_keys.empty:
        return pd.DataFrame()

    track_df = filtered_df.merge(keep_keys, on=group_keys, how="inner")

    keep_cols = select_unique_existing_columns(track_df, PAIR_DETAIL_TRACK_COLS)
    track_df = track_df[keep_cols].copy()

    if "lfm_track_listeners" in track_df.columns:
        track_df = track_df.sort_values(
            "lfm_track_listeners",
            ascending=False,
            na_position="last",
        )

    return track_df

def build_dominant_supporting_df(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build dominant → supporting archetype relationships at the album level.

    Returns one row per (dominant, supporting) pairing with count.
    """
    if track_df.empty:
        return pd.DataFrame(columns=["dominant", "supporting", "count"])

    # List your archetype columns here (same ones used earlier)
    archetype_cols = [
        "track_intensity_band",
        "track_acoustic_orchestral_band",
        "track_speech_texture_band",
    ]

    rows = []

    # Group by album
    for album_id, group in track_df.groupby("release_group_mbid"):
        values = []

        for col in archetype_cols:
            if col not in group.columns:
                continue

            vals = [
                v for v in group[col].dropna().astype(str).tolist()
                if v in ["High", "Low"]
            ]
            clean_map = {
                "track_intensity_band": "Intensity",
                "track_acoustic_orchestral_band": "Acoustic",
                "track_speech_texture_band": "Speech",
            }

            display = clean_map.get(col, col)

            values.extend([f"{display}: {v}" for v in vals])

        if not values:
            continue

        # Count frequencies
        value_counts = pd.Series(values).value_counts()

        dominant = value_counts.idxmax()

        # Supporting = all other unique values
        supporting_values = set(value_counts.index) - {dominant}

        for s in supporting_values:
            rows.append(
                {
                    "dominant": dominant,
                    "supporting": s,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["dominant", "supporting", "count"])

    df = pd.DataFrame(rows)

    out = (
        df.groupby(["dominant", "supporting"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    # Keep only top flows
    TOP_N = 15
    out = out.head(TOP_N)

    return out

def build_dominant_supporting_display_df(dom_sup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add display fields and simple concentration metrics for dominant → supporting flows.
    """
    if dom_sup_df.empty:
        return pd.DataFrame(
            columns=[
                "dominant",
                "supporting",
                "count",
                "relationship_label",
                "dominant_total",
                "pct_of_dominant",
            ]
        )

    out = dom_sup_df.copy()

    out["relationship_label"] = (
        out["dominant"].astype(str) + " → " + out["supporting"].astype(str)
    )

    dominant_totals = (
        out.groupby("dominant", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "dominant_total"})
    )

    out = out.merge(
        dominant_totals,
        on="dominant",
        how="left",
        validate="m:1",
    )

    out["pct_of_dominant"] = out["count"] / out["dominant_total"]

    out = out.sort_values(
        ["count", "relationship_label"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return out

def create_dominant_supporting_sankey(df: pd.DataFrame) -> go.Figure:
    """
    Create a centered Sankey for dominant → supporting archetypes.
    """
    if df.empty:
        return go.Figure()

    plot_df = (
        df.sort_values(["count", "dominant", "supporting"], ascending=[False, True, True])
        .head(10)
        .copy()
    )

    dominant_nodes = plot_df["dominant"].drop_duplicates().tolist()
    supporting_nodes = plot_df["supporting"].drop_duplicates().tolist()

    all_nodes = dominant_nodes + supporting_nodes
    node_index = {name: i for i, name in enumerate(all_nodes)}

    source = plot_df["dominant"].map(node_index).tolist()
    target = plot_df["supporting"].map(node_index).tolist()
    values = plot_df["count"].tolist()

    n_left = len(dominant_nodes)
    n_right = len(supporting_nodes)

    def centered_positions(n: int, center: float = 0.5, span: float = 0.42) -> list[float]:
        """
        Place n nodes evenly in a vertically centered band.
        """
        if n <= 1:
            return [center]
        low = center - span / 2
        high = center + span / 2
        step = (high - low) / (n - 1)
        return [low + i * step for i in range(n)]

    # Center both columns vertically inside the chart
    left_y = centered_positions(n_left, center=0.5, span=0.42)
    right_y = centered_positions(n_right, center=0.5, span=0.30)

    x_positions = [0.18] * n_left + [0.82] * n_right
    y_positions = left_y + right_y

    node_hover = []
    for label in all_nodes:
        if label in dominant_nodes:
            total = int(plot_df.loc[plot_df["dominant"] == label, "count"].sum())
            role = "Dominant"
        else:
            total = int(plot_df.loc[plot_df["supporting"] == label, "count"].sum())
            role = "Supporting"

        node_hover.append(
            f"{role} archetype: {label}<br>"
            f"Visible flow count: {total:,}"
        )

    link_hover = [
        f"Dominant: {row['dominant']}<br>"
        f"Supporting: {row['supporting']}<br>"
        f"Albums: {int(row['count']):,}"
        for _, row in plot_df.iterrows()
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    label=all_nodes,
                    x=x_positions,
                    y=y_positions,
                    pad=18,
                    thickness=16,
                    line=dict(color="rgba(255,255,255,0.18)", width=0.6),
                    customdata=node_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=values,
                    color="rgba(220,220,220,0.72)",
                    customdata=link_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(
        height=560,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="white"),
    )

    return fig

def create_dominant_supporting_chart(
    dom_sup_display_df: pd.DataFrame,
    top_n: int = 10,
) -> alt.Chart:
    """
    Create a ranked bar chart of dominant → supporting structures.
    """
    if dom_sup_display_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    plot_df = dom_sup_display_df.head(top_n).copy()

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "relationship_label:N",
                sort=alt.SortField(field="count", order="descending"),
                title="Dominant → Supporting Structure",
                axis=alt.Axis(labelLimit=320),
            ),
            x=alt.X("count:Q", title="Albums"),
            tooltip=[
                alt.Tooltip("dominant:N", title="Dominant"),
                alt.Tooltip("supporting:N", title="Supporting"),
                alt.Tooltip("count:Q", title="Albums", format=",.0f"),
                alt.Tooltip("pct_of_dominant:Q", title="% of Dominant Albums", format=".1%"),
            ],
        )
        .properties(
            height=max(280, min(520, 36 * len(plot_df))),
        )
    )

def build_dominant_supporting_takeaway(dom_sup_display_df: pd.DataFrame) -> str:
    """
    Build a short interpretation of the visible dominant → supporting structure.
    """
    if dom_sup_display_df.empty:
        return "No dominant-supporting structures remain under the current settings."

    top_row = dom_sup_display_df.iloc[0]
    total_visible = int(dom_sup_display_df["count"].sum())
    top_share = (
        float(top_row["count"]) / total_visible
        if total_visible > 0 else 0.0
    )

    return (
        f"Under the current filters, the most common visible structure is "
        f"{top_row['dominant']} → {top_row['supporting']} "
        f"({int(top_row['count']):,} albums, {top_share:.1%} of visible dominant-supporting flows)."
    )

def build_dominant_supporting_album_table(
    filtered_df: pd.DataFrame,
    dominant_label: str,
    supporting_label: str,
) -> pd.DataFrame:
    """
    Build a simple album table for albums containing both the dominant and supporting labels.
    """
    if filtered_df.empty:
        return pd.DataFrame()

    group_keys = ["release_group_mbid", "tmdb_id"]

    archetype_map = get_active_archetype_labels(
        filtered_df,
        ecosystem_mode="Core archetypes + boolean flags",
    )

    if dominant_label not in archetype_map or supporting_label not in archetype_map:
        return pd.DataFrame()

    working_df = filtered_df[group_keys].copy()
    working_df["dominant_match"] = archetype_map[dominant_label].astype(int)
    working_df["supporting_match"] = archetype_map[supporting_label].astype(int)

    album_flags = (
        working_df.groupby(group_keys, dropna=False)[["dominant_match", "supporting_match"]]
        .max()
        .reset_index()
    )

    keep_keys = album_flags[
        (album_flags["dominant_match"] == 1) &
        (album_flags["supporting_match"] == 1)
    ][group_keys].drop_duplicates()

    if keep_keys.empty:
        return pd.DataFrame()

    album_cols = select_unique_existing_columns(
        filtered_df,
        group_keys + PAIR_DETAIL_ALBUM_COLS,
    )

    out = (
        filtered_df[album_cols]
        .drop_duplicates(subset=group_keys)
        .merge(keep_keys, on=group_keys, how="inner")
    )

    if "lfm_album_listeners" in out.columns:
        out = out.sort_values(
            "lfm_album_listeners",
            ascending=False,
            na_position="last",
        )

    return out

def build_view_context_caption(
    controls: dict,
    filtered_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> str:
    """Build a compact scope caption."""
    visible_albums = filtered_df[["release_group_mbid", "tmdb_id"]].drop_duplicates().shape[0]

    return (
        f"{controls['ecosystem_mode']} | "
        f"{visible_albums:,} albums | "
        f"min pair count {controls['min_album_count']} | "
        f"lift ≥ {controls['min_lift']:.2f} | "
        f"ranked by {controls['pair_metric']}"
    )


def build_insight_cards(
    pair_df: pd.DataFrame,
    dom_sup_display_df: pd.DataFrame,
    metric_name: str,
) -> list[tuple[str, str, str]]:
    """Build top-line insight cards."""
    if pair_df.empty:
        top_pair_value = "None"
        top_pair_caption = "No pairings remain."
        median_lift_value = "None"
        median_lift_caption = "No lift insight remains."
    else:
        metric_col = get_pair_metric_column(metric_name)
        ranked = pair_df.sort_values(
            [metric_col, "count", "pair_label"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        top_metric = ranked.iloc[0]
        median_lift = float(pair_df["lift"].median())

        if metric_name == "Lift vs Random":
            top_pair_metric_text = f"{metric_name} = {top_metric[metric_col]:.3f}"
        elif metric_name == "Lift":
            top_pair_metric_text = f"{metric_name} = {top_metric[metric_col]:.2f}"
        elif metric_name == "% of albums":
            top_pair_metric_text = f"{metric_name} = {top_metric[metric_col]:.1%}"
        else:
            top_pair_metric_text = f"{metric_name} = {top_metric[metric_col]:,.0f}"

        top_pair_value = str(top_metric["pair_label"])
        top_pair_caption = top_pair_metric_text
        median_lift_value = f"{median_lift:.2f}"
        median_lift_caption = "Median lift across visible pairings"

    if dom_sup_display_df.empty:
        top_structure_value = "None"
        top_structure_caption = "No dominant-supporting structures remain."
    else:
        top_structure = dom_sup_display_df.iloc[0]
        top_structure_value = str(top_structure["relationship_label"])
        top_structure_caption = (
            f"{int(top_structure['count']):,} albums | "
            f"{top_structure['pct_of_dominant']:.1%} of dominant albums"
        )

    return [
        ("Top Pair", top_pair_value, top_pair_caption),
        ("Top Structure", top_structure_value, top_structure_caption),
        ("Typical Lift", median_lift_value, median_lift_caption),
    ]


def render_insight_cards(
    pair_df: pd.DataFrame,
    dom_sup_display_df: pd.DataFrame,
    metric_name: str,
) -> None:
    """Render page insight cards."""
    cards = build_insight_cards(
        pair_df=pair_df,
        dom_sup_display_df=dom_sup_display_df,
        metric_name=metric_name,
    )
    cols = st.columns(3)

    for i, (title, value, caption) in enumerate(cards):
        with cols[i]:
            st.metric(title, value)
            st.caption(caption)

def build_ecosystem_takeaway(pair_df: pd.DataFrame) -> str:
    """Build a short interpretation of the visible ecosystem structure."""
    if pair_df.empty:
        return "No archetype pairings remain under the current settings."

    median_lift = float(pair_df["lift"].median())
    max_lift = float(pair_df["lift"].max())
    strong_pairs = int((pair_df["lift"] >= 1.05).sum())

    if max_lift < 1.05:
        return (
            "Most visible archetype pairings occur at rates very close to chance "
            "(lift near 1.00). That suggests soundtrack albums are not strongly "
            "organized around a few distinctive archetype combinations."
        )

    if strong_pairs <= 3:
        return (
            f"Only a small number of visible archetype pairings stand out materially "
            f"(max lift = {max_lift:.2f}, median lift = {median_lift:.2f}). "
            "That suggests weak but non-random ecosystem structure rather than a "
            "strongly segmented album archetype system."
        )

    return (
        f"Several visible archetype pairings stand out above chance "
        f"(max lift = {max_lift:.2f}, median lift = {median_lift:.2f}), "
        "suggesting that some soundtrack albums do exhibit recurring track-type ecosystems."
    )

def main() -> None:
    """Render the Track Ecosystem Explorer."""
    st.set_page_config(
        page_title="Track Ecosystem Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Ecosystem Explorer")
    st.caption(
        "See which track archetype pairings appear together within albums more often than expected."
    )

    track_df = load_track_data_explorer_data()
    track_df = add_track_ecosystem_display_fields(track_df)

    filter_inputs = get_global_filter_inputs(track_df)
    composer_options = get_clean_composer_options(track_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    controls = get_track_ecosystem_controls(
        composer_options=composer_options,
    )

    filtered_df = filter_track_ecosystem_df(
        track_df=track_df,
        global_controls=global_controls,
        controls=controls,
    )

    if filtered_df.empty:
        st.warning("No tracks remain under the current filters.")
        return

    album_presence_df = build_album_archetype_presence_df(
        filtered_df,
        ecosystem_mode=controls["ecosystem_mode"],
    )

    pair_df = build_pair_metrics_df(
        album_presence_df=album_presence_df,
        min_album_count=controls["min_album_count"],
    )

    pair_df = pair_df[
        pair_df["lift"] >= controls["min_lift"]
    ].copy()

    if pair_df.empty:
        st.warning("No archetype pairings remain under the current settings.")
        return

    dom_sup_df = build_dominant_supporting_df(filtered_df)
    dom_sup_display_df = build_dominant_supporting_display_df(dom_sup_df)

    st.caption(
        build_filter_context_caption(controls)
        + " | "
        + build_view_context_caption(
            controls=controls,
            filtered_df=filtered_df,
            pair_df=pair_df,
        )
    )

    visible_tracks = len(filtered_df)
    visible_albums = filtered_df[["release_group_mbid", "tmdb_id"]].drop_duplicates().shape[0]
    visible_pairs = len(pair_df)
    visible_archetypes = len([col for col in album_presence_df.columns if col not in ["release_group_mbid", "tmdb_id"]])

    top_pair = pair_df.sort_values(
        [get_pair_metric_column(controls["pair_metric"]), "count", "pair_label"],
        ascending=[False, False, True],
    ).iloc[0]["pair_label"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tracks", f"{visible_tracks:,}")
    c2.metric("Albums", f"{visible_albums:,}")
    c3.metric("States", f"{visible_archetypes:,}")
    c4.metric("Pairs", f"{visible_pairs:,}")
    c5.metric("Top Pair", top_pair)

    st.markdown("### 🧠 Key Insights")
    render_insight_cards(
        pair_df=pair_df,
        dom_sup_display_df=dom_sup_display_df,
        metric_name=controls["pair_metric"],
    )
    st.caption(build_ecosystem_takeaway(pair_df))

    st.markdown("### Track Ecosystem Matrix")
    st.altair_chart(
        create_ecosystem_matrix_chart(pair_df, controls["pair_metric"]),
        width="stretch",
    )
    st.caption(
        "Darker cells indicate archetype pairs that appear together more strongly under the selected metric."
    )

    st.divider()

    # ==============================
    # Dominant → Supporting section
    # ==============================

    st.markdown("### Dominant → Supporting Archetypes")
    st.caption(
        "Read left to right: dominant album archetype → supporting archetypes commonly present alongside it."
    )

    if dom_sup_display_df.empty:
        st.info("No dominant-supporting relationships found.")
    else:
        sankey_fig = create_dominant_supporting_sankey(dom_sup_display_df)
        st.plotly_chart(sankey_fig, width="stretch")

        st.caption(build_dominant_supporting_takeaway(dom_sup_display_df))

        st.markdown("#### Top Dominant → Supporting Structures")
        st.altair_chart(
            create_dominant_supporting_chart(
                dom_sup_display_df=dom_sup_display_df,
                top_n=10,
            ),
            width="stretch",
        )

        st.markdown("#### Inspect a Dominant → Supporting Structure")

        selected_structure_label = st.selectbox(
            "Dominant → supporting structure",
            options=dom_sup_display_df["relationship_label"].tolist(),
            index=0,
            key="dominant_supporting_structure_select",
        )

        selected_structure_row = dom_sup_display_df[
            dom_sup_display_df["relationship_label"] == selected_structure_label
        ].iloc[0]

        dominant_label = selected_structure_row["dominant"]
        supporting_label = selected_structure_row["supporting"]

        st.caption(
            f"Showing albums containing both **{dominant_label}** and **{supporting_label}**."
        )

        dom_sup_album_table = build_dominant_supporting_album_table(
            filtered_df=filtered_df,
            dominant_label=dominant_label,
            supporting_label=supporting_label,
        )

        if dom_sup_album_table.empty:
            st.info("No albums match the selected dominant-supporting structure.")
        else:
            st.dataframe(
                rename_track_page_columns_for_display(dom_sup_album_table),
                width="stretch",
                hide_index=True,
            )

    st.divider()

    st.markdown("### Top Archetype Pairings")
    st.altair_chart(
        create_top_pairs_chart(
            pair_df=pair_df,
            metric_name=controls["pair_metric"],
            top_n_pairs=controls["top_n_pairs"],
        ),
        width="stretch",
    )

    if controls["show_pair_table"]:
        st.markdown("### Pair Summary Table")
        pair_display = pair_df.rename(
            columns={
                "source": "Archetype A",
                "target": "Archetype B",
                "pair_label": "Archetype Pair",
                "count": "Albums with Both",
                "source_count": "Albums with A",
                "target_count": "Albums with B",
                "pct_albums": "% of Albums",
                "lift": "Lift",
            }
        )
        st.dataframe(pair_display, width="stretch", hide_index=True)

    st.divider()

    st.markdown("### Inspect a Pair")
    selected_pair_label = st.selectbox(
        "Archetype pair",
        options=pair_df["pair_label"].tolist(),
        index=0,
    )

    selected_row = pair_df[pair_df["pair_label"] == selected_pair_label].iloc[0]
    source_label = selected_row["source"]
    target_label = selected_row["target"]

    st.caption(
        f"Showing albums that contain both **{source_label}** and **{target_label}**."
    )

    album_table = build_selected_pair_album_table(
        filtered_df=filtered_df,
        album_presence_df=album_presence_df,
        source_label=source_label,
        target_label=target_label,
    )

    if controls["show_album_table"]:
        st.markdown("#### Albums with Selected Pair")
        if album_table.empty:
            st.info("No albums match the selected archetype pair.")
        else:
            st.dataframe(
                rename_track_page_columns_for_display(album_table),
                width="stretch",
                hide_index=True,
            )

    if controls["show_track_table"]:
        track_table = build_selected_pair_track_table(
            filtered_df=filtered_df,
            album_presence_df=album_presence_df,
            source_label=source_label,
            target_label=target_label,
            ecosystem_mode=controls["ecosystem_mode"],
        )

        st.markdown("#### Track Rows from Matching Albums")
        if track_table.empty:
            st.info("No tracks are available for the selected archetype pair.")
        else:
            st.dataframe(
                rename_track_page_columns_for_display(track_table),
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    main()