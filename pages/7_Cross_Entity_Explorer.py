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
    "label_names_clean",
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

RELATIONSHIP_SPECS = {
    "Film Genre → Album Genre": {
        "mode": "single_hop",
        "source_type": "multivalue",
        "target_type": "multivalue",
        "source_col": "film_genres",
        "target_col": "album_genres_display",
        "source_label": "Film Genre",
        "target_label": "Album Genre",
        "relationship_unit_singular": "flow",
        "relationship_unit_plural": "flows",
    },
    "Award → Album Genre": {
        "mode": "single_hop",
        "source_type": "binary_flags",
        "target_type": "multivalue",
        "source_cols": AWARD_FLAGS,
        "target_col": "album_genres_display",
        "source_label": "Award",
        "target_label": "Album Genre",
        "relationship_unit_singular": "flow",
        "relationship_unit_plural": "flows",
    },
    "Film Genre → Award": {
        "mode": "single_hop",
        "source_type": "multivalue",
        "target_type": "binary_flags",
        "source_col": "film_genres",
        "target_cols": AWARD_FLAGS,
        "source_label": "Film Genre",
        "target_label": "Award",
        "relationship_unit_singular": "flow",
        "relationship_unit_plural": "flows",
    },
    "Film Genre → Composer": {
        "mode": "single_hop",
        "source_type": "multivalue",
        "target_type": "single_value",
        "source_col": "film_genres",
        "target_col": "composer_primary_clean",
        "source_label": "Film Genre",
        "target_label": "Composer",
        "relationship_unit_singular": "flow",
        "relationship_unit_plural": "flows",
    },
    "Film Genre → Label": {
        "mode": "single_hop",
        "source_type": "multivalue",
        "target_type": "multivalue",
        "source_col": "film_genres",
        "target_col": "label_names_clean",
        "source_label": "Film Genre",
        "target_label": "Label",
        "relationship_unit_singular": "flow",
        "relationship_unit_plural": "flows",
    },
    "Award → Film Genre → Album Genre": {
        "mode": "three_hop",
        "source_type": "binary_flags",
        "middle_type": "multivalue",
        "target_type": "multivalue",
        "source_cols": AWARD_FLAGS,
        "middle_col": "film_genres",
        "target_col": "album_genres_display",
        "source_label": "Award",
        "middle_label": "Film Genre",
        "target_label": "Album Genre",
        "relationship_unit_singular": "path",
        "relationship_unit_plural": "paths",
    },
    "Film Genre → Composer → Album Genre": {
        "mode": "three_hop",
        "source_type": "multivalue",
        "middle_type": "single_value",
        "target_type": "multivalue",
        "source_col": "film_genres",
        "middle_col": "composer_primary_clean",
        "target_col": "album_genres_display",
        "source_label": "Film Genre",
        "middle_label": "Composer",
        "target_label": "Album Genre",
        "relationship_unit_singular": "path",
        "relationship_unit_plural": "paths",
    },
    "Film Genre → Label → Album Genre": {
        "mode": "three_hop",
        "source_type": "multivalue",
        "middle_type": "multivalue",
        "target_type": "multivalue",
        "source_col": "film_genres",
        "middle_col": "label_names_clean",
        "target_col": "album_genres_display",
        "source_label": "Film Genre",
        "middle_label": "Label",
        "target_label": "Album Genre",
        "relationship_unit_singular": "path",
        "relationship_unit_plural": "paths",
    },
}


def get_award_display_label(name: str) -> str:
    """
    Return a readable label for an award flag.

    Args:
        name: Raw award flag column name.

    Returns:
        str: Display-friendly award label.
    """
    return AWARD_LABEL_MAP.get(name, name)


def pluralize(label: str) -> str:
    """
    Return a simple pluralized form of an entity label.

    Args:
        label: Singular entity label.

    Returns:
        str: Pluralized label.
    """
    if label.endswith("y"):
        return label[:-1] + "ies"
    return label + "s"


def get_relationship_type_options() -> list[str]:
    """
    Return supported cross-entity relationship types.

    Returns:
        list[str]: Relationship labels shown in the UI.
    """
    return list(RELATIONSHIP_SPECS.keys())


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
        "Jaccard similarity": "jaccard",
        "Lift": "lift",
    }
    return metric_map.get(metric_name, "count")


def get_edge_metric_display_label(metric_name: str) -> str:
    """
    Return a readable axis or tooltip label for the selected metric.

    Args:
        metric_name: User-facing metric label.

    Returns:
        str: Display-ready metric label.
    """
    return metric_name


def format_metric_value(value: float, metric_col: str) -> str:
    """
    Format a metric value for display.

    Args:
        value: Numeric metric value.
        metric_col: Backing dataframe metric column.

    Returns:
        str: Formatted metric value.
    """
    if metric_col in {"pct_source", "pct_target"}:
        return f"{value:.1%}"
    if metric_col == "jaccard":
        return f"{value:.3f}"
    if metric_col == "lift":
        return f"{value:.2f}"
    return f"{value:,.0f}"


def build_cross_entity_explorer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the page-specific dataframe used by the Cross-Entity Explorer.

    Args:
        df: Album-level explorer dataframe.

    Returns:
        pd.DataFrame: Copy with numeric film year, award flags, and cleaned
        high-cardinality fields prepared for relationship views.
    """
    out_df = df.copy()

    if "film_year" in out_df.columns:
        out_df["film_year"] = pd.to_numeric(
            out_df["film_year"],
            errors="coerce",
        )

    for col in AWARD_FLAGS:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(
                out_df[col],
                errors="coerce",
            ).fillna(0).astype(int)
        else:
            out_df[col] = 0

    if "composer_primary_clean" not in out_df.columns:
        out_df["composer_primary_clean"] = ""
    else:
        out_df["composer_primary_clean"] = (
            out_df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "label_names_clean" not in out_df.columns:
        out_df["label_names_clean"] = ""
    else:
        out_df["label_names_clean"] = (
            out_df["label_names_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return out_df


def extract_multivalue_values(value: object) -> list[str]:
    """
    Extract cleaned values from a pipe- or comma-delimited string.

    Args:
        value: Raw dataframe cell value.

    Returns:
        list[str]: Cleaned distinct values.
    """
    if pd.isna(value):
        return []

    return split_multivalue_genres(pd.Series([value]))


def extract_binary_flag_values(
    row: pd.Series,
    cols: list[str],
) -> list[str]:
    """
    Extract active binary flag columns from a row.

    Args:
        row: Dataframe row.
        cols: Candidate binary flag columns.

    Returns:
        list[str]: Names of active flags.
    """
    active = []
    for col in cols:
        if col in row and pd.notna(row[col]) and int(row[col]) == 1:
            active.append(col)
    return active


def extract_values_from_row(
    row: pd.Series,
    value_type: str,
    col: str | None = None,
    cols: list[str] | None = None,
) -> list[str]:
    """
    Extract source, middle, or target values from a row based on the
    configured type.

    Args:
        row: Dataframe row.
        value_type: One of 'multivalue', 'binary_flags', or 'single_value'.
        col: Single column for multivalue or single-value extraction.
        cols: Multiple binary flag columns.

    Returns:
        list[str]: Extracted values.
    """
    if value_type == "multivalue":
        if col is None:
            return []
        return extract_multivalue_values(row.get(col))

    if value_type == "binary_flags":
        return extract_binary_flag_values(row, cols or [])

    if value_type == "single_value":
        if col is None:
            return []
        value = row.get(col)
        if pd.isna(value):
            return []
        value = str(value).strip()
        if not value:
            return []
        return [value]

    return []

def filter_high_cardinality_entities(df, controls, spec):
    """
    Filter composers and labels BEFORE relationship construction.
    """

    df = df.copy()

    # Composer filtering
    if (
        spec.get("source_label") == "Composer" or
        spec.get("middle_label") == "Composer" or
        spec.get("target_label") == "Composer"
    ):
        counts = df["composer_primary_clean"].value_counts()

        keep = counts[
            (counts >= controls["min_composer_count"])
        ].head(controls["top_n_composers"]).index

        df = df[df["composer_primary_clean"].isin(keep)]

    # Label filtering
    if (
        spec.get("source_label") == "Label" or
        spec.get("middle_label") == "Label" or
        spec.get("target_label") == "Label"
    ):
        exploded = (
            df.assign(label=df["label_names_clean"].str.split(r"\s*\|\s*"))
            .explode("label")
        )

        counts = exploded["label"].value_counts()

        keep = counts[
            (counts >= controls["min_label_count"])
        ].head(controls["top_n_labels"]).index

        df = exploded[exploded["label"].isin(keep)]

        # collapse back
        df = df.groupby(df.index).first()

    return df

def apply_relationship_side_filters(
    values: list[str],
    selected_values: list[str] | None,
) -> list[str]:
    """
    Restrict exploded values to a selected subset when provided.

    Args:
        values: Extracted values.
        selected_values: Optional selected values from global filters.

    Returns:
        list[str]: Filtered values.
    """
    if not selected_values:
        return values

    selected_set = set(selected_values)
    return [value for value in values if value in selected_set]


def get_selected_values_for_label(
    label: str,
    global_filters: dict,
) -> list[str] | None:
    """
    Return selected global filter values matching a semantic entity label.

    Args:
        label: Semantic label such as Film Genre or Album Genre.
        global_filters: Shared global filter selections.

    Returns:
        list[str] | None: Matching selected values when applicable.
    """
    if label == "Film Genre":
        return global_filters.get("selected_film_genres")
    if label == "Album Genre":
        return global_filters.get("selected_album_genres")
    return None


def get_selected_values_for_relationship(
    spec: dict,
    global_filters: dict,
) -> dict[str, list[str] | None]:
    """
    Determine which selected global filters should constrain each visible side
    of the current relationship type.

    Args:
        spec: Relationship spec.
        global_filters: Shared global filter selections.

    Returns:
        dict[str, list[str] | None]: Selected values for source/middle/target.
    """
    selections = {
        "source": get_selected_values_for_label(
            spec["source_label"],
            global_filters,
        ),
        "target": get_selected_values_for_label(
            spec["target_label"],
            global_filters,
        ),
    }

    if spec["mode"] == "three_hop":
        selections["middle"] = get_selected_values_for_label(
            spec["middle_label"],
            global_filters,
        )

    return selections


def build_single_hop_flow_df(
    df: pd.DataFrame,
    spec: dict,
    min_flow_count: int,
    selected_source_values: list[str] | None = None,
    selected_target_values: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a generic single-hop source→target flow dataframe.

    Args:
        df: Filtered album-level dataframe.
        spec: Relationship config.
        min_flow_count: Minimum raw count required to retain a flow.
        selected_source_values: Optional selected source-side values.
        selected_target_values: Optional selected target-side values.

    Returns:
        pd.DataFrame: One row per retained source→target flow with count- and
        similarity-based metrics.
    """
    empty_cols = [
        "source",
        "target",
        "relationship_label",
        "count",
        "source_count",
        "target_count",
        "union_count",
        "pct_source",
        "pct_target",
        "jaccard",
        "lift",
    ]

    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    flow_rows = []

    for _, row in df.iterrows():
        source_values = extract_values_from_row(
            row=row,
            value_type=spec["source_type"],
            col=spec.get("source_col"),
            cols=spec.get("source_cols"),
        )
        target_values = extract_values_from_row(
            row=row,
            value_type=spec["target_type"],
            col=spec.get("target_col"),
            cols=spec.get("target_cols"),
        )

        source_values = apply_relationship_side_filters(
            values=source_values,
            selected_values=selected_source_values,
        )
        target_values = apply_relationship_side_filters(
            values=target_values,
            selected_values=selected_target_values,
        )

        if not source_values or not target_values:
            continue

        for source_value in source_values:
            for target_value in target_values:
                flow_rows.append(
                    {
                        "source": source_value,
                        "target": target_value,
                    }
                )

    if not flow_rows:
        return pd.DataFrame(columns=empty_cols)

    flow_detail_df = pd.DataFrame(flow_rows)

    relationship_df = (
        flow_detail_df.groupby(["source", "target"])
        .size()
        .reset_index(name="count")
    )

    relationship_df = relationship_df[
        relationship_df["count"] >= min_flow_count
    ].copy()

    if relationship_df.empty:
        return pd.DataFrame(columns=empty_cols)

    source_counts = flow_detail_df.groupby("source").size().to_dict()
    target_counts = flow_detail_df.groupby("target").size().to_dict()
    total_pairs = int(len(flow_detail_df))

    relationship_df["source_count"] = relationship_df["source"].map(source_counts).astype(int)
    relationship_df["target_count"] = relationship_df["target"].map(target_counts).astype(int)
    relationship_df["union_count"] = (
        relationship_df["source_count"] +
        relationship_df["target_count"] -
        relationship_df["count"]
    )

    relationship_df["pct_source"] = (
        relationship_df["count"] / relationship_df["source_count"]
    )
    relationship_df["pct_target"] = (
        relationship_df["count"] / relationship_df["target_count"]
    )
    relationship_df["jaccard"] = (
        relationship_df["count"] / relationship_df["union_count"]
    )

    p_source = relationship_df["source_count"] / total_pairs
    p_target = relationship_df["target_count"] / total_pairs
    p_both = relationship_df["count"] / total_pairs
    expected = p_source * p_target
    relationship_df["lift"] = p_both / expected

    relationship_df["relationship_label"] = (
        relationship_df["source"].astype(str) +
        " → " +
        relationship_df["target"].astype(str)
    )

    relationship_df = relationship_df.sort_values(
        ["count", "relationship_label"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return relationship_df


def build_three_hop_path_df(
    df: pd.DataFrame,
    spec: dict,
    min_flow_count: int,
    selected_source_values: list[str] | None = None,
    selected_middle_values: list[str] | None = None,
    selected_target_values: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a constrained three-hop source→middle→target path dataframe.

    Args:
        df: Filtered album-level dataframe.
        spec: Relationship config.
        min_flow_count: Minimum raw count required to retain a path.
        selected_source_values: Optional selected source-side values.
        selected_middle_values: Optional selected middle-side values.
        selected_target_values: Optional selected target-side values.

    Returns:
        pd.DataFrame: One row per retained source→middle→target path with
        count- and similarity-based metrics.
    """
    empty_cols = [
        "source",
        "middle",
        "target",
        "relationship_label",
        "count",
        "source_count",
        "target_count",
        "union_count",
        "pct_source",
        "pct_target",
        "jaccard",
        "lift",
    ]

    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    path_rows = []

    for _, row in df.iterrows():
        source_values = extract_values_from_row(
            row=row,
            value_type=spec["source_type"],
            col=spec.get("source_col"),
            cols=spec.get("source_cols"),
        )
        middle_values = extract_values_from_row(
            row=row,
            value_type=spec["middle_type"],
            col=spec.get("middle_col"),
            cols=spec.get("middle_cols"),
        )
        target_values = extract_values_from_row(
            row=row,
            value_type=spec["target_type"],
            col=spec.get("target_col"),
            cols=spec.get("target_cols"),
        )

        source_values = apply_relationship_side_filters(
            values=source_values,
            selected_values=selected_source_values,
        )
        middle_values = apply_relationship_side_filters(
            values=middle_values,
            selected_values=selected_middle_values,
        )
        target_values = apply_relationship_side_filters(
            values=target_values,
            selected_values=selected_target_values,
        )

        if not source_values or not middle_values or not target_values:
            continue

        for source_value in source_values:
            for middle_value in middle_values:
                for target_value in target_values:
                    path_rows.append(
                        {
                            "source": source_value,
                            "middle": middle_value,
                            "target": target_value,
                        }
                    )

    if not path_rows:
        return pd.DataFrame(columns=empty_cols)

    path_detail_df = pd.DataFrame(path_rows)

    relationship_df = (
        path_detail_df.groupby(["source", "middle", "target"])
        .size()
        .reset_index(name="count")
    )

    relationship_df = relationship_df[
        relationship_df["count"] >= min_flow_count
    ].copy()

    if relationship_df.empty:
        return pd.DataFrame(columns=empty_cols)

    source_counts = path_detail_df.groupby("source").size().to_dict()
    target_counts = path_detail_df.groupby("target").size().to_dict()
    total_paths = int(len(path_detail_df))

    relationship_df["source_count"] = relationship_df["source"].map(source_counts).astype(int)
    relationship_df["target_count"] = relationship_df["target"].map(target_counts).astype(int)
    relationship_df["union_count"] = (
        relationship_df["source_count"] +
        relationship_df["target_count"] -
        relationship_df["count"]
    )

    relationship_df["pct_source"] = (
        relationship_df["count"] / relationship_df["source_count"]
    )
    relationship_df["pct_target"] = (
        relationship_df["count"] / relationship_df["target_count"]
    )
    relationship_df["jaccard"] = (
        relationship_df["count"] / relationship_df["union_count"]
    )

    p_source = relationship_df["source_count"] / total_paths
    p_target = relationship_df["target_count"] / total_paths
    p_both = relationship_df["count"] / total_paths
    expected = p_source * p_target
    relationship_df["lift"] = p_both / expected

    relationship_df["relationship_label"] = (
        relationship_df["source"].astype(str) +
        " → " +
        relationship_df["middle"].astype(str) +
        " → " +
        relationship_df["target"].astype(str)
    )

    relationship_df = relationship_df.sort_values(
        ["count", "relationship_label"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return relationship_df


def display_value(
    raw_value: str,
    side_label: str,
) -> str:
    """
    Convert raw source, middle, or target values into display-friendly labels.

    Args:
        raw_value: Raw value.
        side_label: Semantic side label, such as Award or Film Genre.

    Returns:
        str: Display-friendly value.
    """
    if side_label == "Award":
        return get_award_display_label(raw_value)

    return raw_value


def get_metric_side_labels(
    source_label: str,
    target_label: str,
) -> dict[str, str]:
    """
    Return relationship-aware labels for source/target percentage metrics.

    Args:
        source_label: Source-side entity label.
        target_label: Target-side entity label.

    Returns:
        dict[str, str]: Display labels for metric-related UI text.
    """
    return {
        "pct_source": f"% of {source_label}",
        "pct_target": f"% of {target_label}",
    }


def build_filter_context_caption(
    relationship_type: str,
    global_filters: dict,
) -> str:
    """
    Build a short caption explaining which global filters are currently shaping
    the album universe behind the visible relationships.

    Args:
        relationship_type: Selected relationship mode.
        global_filters: Shared global filter selections.

    Returns:
        str: Caption describing the active context.
    """
    active_parts = []

    year_range = global_filters.get("year_range")
    if year_range:
        active_parts.append(
            f"film years {year_range[0]}–{year_range[1]}"
        )

    if global_filters.get("selected_film_genres"):
        active_parts.append(
            f"film genres: {', '.join(global_filters['selected_film_genres'])}"
        )

    if global_filters.get("selected_album_genres"):
        active_parts.append(
            f"album genres: {', '.join(global_filters['selected_album_genres'])}"
        )

    if not active_parts:
        return (
            f"Showing all albums in the current dataset scope for "
            f"{relationship_type.lower()}."
        )

    return (
        f"Visible relationships for {relationship_type.lower()} are currently "
        f"shaped by these global filters: {'; '.join(active_parts)}."
    )

def prune_multivalue_column_to_allowed(
    series: pd.Series,
    allowed_values: set[str],
) -> pd.Series:
    """
    Keep only allowed values inside a pipe-delimited multivalue field.

    Args:
        series: Multivalue string series.
        allowed_values: Set of permitted values.

    Returns:
        pd.Series: Cleaned pipe-delimited strings containing only kept values.
    """
    def _prune(value: object) -> str:
        values = extract_multivalue_values(value)
        kept = [item for item in values if item in allowed_values]
        return " | ".join(kept)

    return series.apply(_prune)


def filter_high_cardinality_entities(
    df: pd.DataFrame,
    controls: dict,
    spec: dict,
) -> pd.DataFrame:
    """
    Restrict high-cardinality composer and label entities before relationship
    construction so the visible Sankey and rankings remain interpretable.

    Args:
        df: Globally filtered dataframe.
        controls: Cross-entity page controls.
        spec: Relationship config.

    Returns:
        pd.DataFrame: Further filtered dataframe with long-tail composers/labels
        removed from relevant columns.
    """
    out_df = df.copy()

    visible_labels = {
        spec.get("source_label"),
        spec.get("middle_label"),
        spec.get("target_label"),
    }

    if "Composer" in visible_labels and "composer_primary_clean" in out_df.columns:
        composer_counts = (
            out_df["composer_primary_clean"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
        )

        keep_composers = composer_counts[
            composer_counts >= controls["min_composer_count"]
        ].head(controls["top_n_composers"]).index

        out_df = out_df[
            out_df["composer_primary_clean"].isin(keep_composers)
        ].copy()

    if "Label" in visible_labels and "label_names_clean" in out_df.columns:
        exploded_labels = (
            out_df["label_names_clean"]
            .fillna("")
            .astype(str)
            .str.split(r"\s*\|\s*", regex=True)
            .explode()
            .str.strip()
        )

        exploded_labels = exploded_labels[
            exploded_labels.notna() & (exploded_labels != "")
        ]

        label_counts = exploded_labels.value_counts()

        keep_labels = set(
            label_counts[label_counts >= controls["min_label_count"]]
            .head(controls["top_n_labels"])
            .index
            .tolist()
        )

        out_df["label_names_clean"] = prune_multivalue_column_to_allowed(
            out_df["label_names_clean"],
            keep_labels,
        )

        out_df = out_df[
            out_df["label_names_clean"].fillna("").astype(str).str.strip() != ""
        ].copy()

    return out_df

def build_detail_table(
    df: pd.DataFrame,
    spec: dict,
    selected_source: str,
    selected_target: str,
    selected_middle: str | None = None,
) -> pd.DataFrame:
    """
    Build a generic album drilldown table for a selected relationship.

    Args:
        df: Filtered album-level dataframe.
        spec: Relationship config.
        selected_source: Raw selected source value.
        selected_target: Raw selected target value.
        selected_middle: Raw selected middle value for three-hop paths.

    Returns:
        pd.DataFrame: Matching albums.
    """
    if df.empty:
        return pd.DataFrame()

    detail_df = df.copy()

    if spec["source_type"] == "multivalue":
        source_col = spec["source_col"]
        detail_df = detail_df[
            detail_df[source_col].apply(
                lambda value: selected_source in extract_multivalue_values(value)
            )
        ]
    else:
        detail_df = detail_df[
            detail_df[selected_source].fillna(0).astype(int) == 1
        ]

    if spec["mode"] == "three_hop" and selected_middle is not None:
        middle_col = spec["middle_col"]
        detail_df = detail_df[
            detail_df[middle_col].apply(
                lambda value: selected_middle in extract_multivalue_values(value)
            )
        ]

    if spec["target_type"] == "multivalue":
        target_col = spec["target_col"]
        detail_df = detail_df[
            detail_df[target_col].apply(
                lambda value: selected_target in extract_multivalue_values(value)
            )
        ]
    else:
        detail_df = detail_df[
            detail_df[selected_target].fillna(0).astype(int) == 1
        ]

    if detail_df.empty:
        return detail_df

    if "lfm_album_listeners" in detail_df.columns:
        detail_df = detail_df.sort_values(
            "lfm_album_listeners",
            ascending=False,
            na_position="last",
        )

    cols = [col for col in DETAIL_COLS if col in detail_df.columns]

    if (
        spec["source_label"] == "Award" or
        spec.get("middle_label") == "Award" or
        spec["target_label"] == "Award"
    ):
        award_cols = [col for col in AWARD_FLAGS if col in detail_df.columns]
        cols = cols + [col for col in award_cols if col not in cols]

    return detail_df[cols].copy()


def build_relationship_label_display(
    row: pd.Series,
    spec: dict,
) -> str:
    """
    Build a display-friendly relationship label for a row.

    Args:
        row: Relationship dataframe row.
        spec: Relationship config.

    Returns:
        str: Display label.
    """
    source_display = display_value(row["source"], spec["source_label"])
    target_display = display_value(row["target"], spec["target_label"])

    if spec["mode"] == "three_hop":
        middle_display = display_value(row["middle"], spec["middle_label"])
        return f"{source_display} → {middle_display} → {target_display}"

    return f"{source_display} → {target_display}"


def create_top_relationships_chart(
    relationship_df: pd.DataFrame,
    spec: dict,
    top_n_edges: int,
    metric_name: str,
) -> alt.Chart:
    """
    Create a horizontal ranking chart of the strongest visible flows or paths.

    Args:
        relationship_df: Relationship dataframe.
        spec: Relationship config.
        top_n_edges: Number of top relationships to show.
        metric_name: User-selected metric label.

    Returns:
        alt.Chart: Horizontal bar chart.
    """
    if relationship_df.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

    metric_col = get_edge_metric_column(metric_name)
    metric_label = get_edge_metric_display_label(metric_name)
    metric_side_labels = get_metric_side_labels(
        spec["source_label"],
        spec["target_label"],
    )

    plot_df = relationship_df.sort_values(
        [metric_col, "count", "relationship_label"],
        ascending=[False, False, True],
    ).head(top_n_edges).copy()

    plot_df["relationship_label_display"] = plot_df.apply(
        lambda row: build_relationship_label_display(row, spec),
        axis=1,
    )

    tooltip = [
        alt.Tooltip("relationship_label_display:N", title="Relationship"),
        alt.Tooltip("count:Q", title="Contributions", format=",.0f"),
        alt.Tooltip(
            "pct_source:Q",
            title=metric_side_labels["pct_source"],
            format=".1%",
        ),
        alt.Tooltip(
            "pct_target:Q",
            title=metric_side_labels["pct_target"],
            format=".1%",
        ),
        alt.Tooltip("jaccard:Q", title="Jaccard", format=".3f"),
        alt.Tooltip("lift:Q", title="Lift", format=".2f"),
    ]

    title_prefix = (
        f"Top {spec['source_label']} → {spec['middle_label']} → {spec['target_label']} Paths"
        if spec["mode"] == "three_hop"
        else f"Top {spec['source_label']} → {spec['target_label']} Flows"
    )

    y_title = None

    chart_width = 900 if spec["mode"] == "three_hop" else 750

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{metric_col}:Q",
                title=metric_label,
            ),
            y=alt.Y(
                "relationship_label_display:N",
                sort=alt.SortField(field=metric_col, order="descending"),
                title=y_title,
                axis=alt.Axis(
                    labelLimit=420 if spec["mode"] == "three_hop" else 260,
                ),
            ),
            tooltip=tooltip,
        )
        .properties(
            width=chart_width,
            height=max(320, min(900, top_n_edges * 38)),
            title={
                "text": title_prefix,
                "subtitle": [f"Ranked by {metric_label.lower()}."],
            },
        )
    )


def get_metric_explainer(
    metric_name: str,
    spec: dict,
) -> str:
    """
    Return a short explanation for the selected relationship metric.

    Args:
        metric_name: User-selected metric label.
        spec: Relationship config.

    Returns:
        str: Short interpretation string.
    """
    relationship_phrase = (
        f"{spec['source_label'].lower()} → "
        f"{spec['middle_label'].lower()} → "
        f"{spec['target_label'].lower()} path"
        if spec["mode"] == "three_hop"
        else f"{spec['source_label'].lower()} → {spec['target_label'].lower()} flow"
    )

    explainers = {
        "Count": (
            f"Count = the number of visible {relationship_phrase} contributions "
            "in the filtered view."
        ),
        "% of source": (
            f"% of source = among all visible contributions from a "
            f"{spec['source_label'].lower()}, the share going to this visible "
            f"{spec['target_label'].lower()} relationship."
        ),
        "% of target": (
            f"% of target = among all visible contributions reaching a "
            f"{spec['target_label'].lower()}, the share coming from this "
            f"{spec['source_label'].lower()} relationship."
        ),
        "Jaccard similarity": (
            "Jaccard = overlap divided by union. Higher values indicate a tighter "
            "relationship relative to the overall visible footprint."
        ),
        "Lift": (
            "Lift = observed pairing rate divided by expected rate under "
            "independence. Values above 1 indicate a stronger-than-expected relationship."
        ),
    }
    return explainers.get(metric_name, metric_name)


def build_relationship_insight_summary(
    relationship_df: pd.DataFrame,
    spec: dict,
    metric_name: str,
) -> dict[str, str]:
    """
    Build a narrative summary from the current relationship dataframe.

    Args:
        relationship_df: Relationship dataframe.
        spec: Relationship config.
        metric_name: User-selected metric label.

    Returns:
        dict[str, str]: Top insight strings.
    """
    if relationship_df.empty:
        return {
            "top_relationship_title": "Top Relationship",
            "top_relationship_value": "None",
            "top_relationship_caption": "No visible relationships remain under the current settings.",
            "lift_title": "Strongest Lift",
            "lift_value": "None",
            "lift_caption": "No lift insight available.",
            "concentration_title": f"Most Concentrated {spec['source_label']}",
            "concentration_value": "None",
            "concentration_caption": "No concentration insight available.",
        }

    metric_col = get_edge_metric_column(metric_name)

    ranked_metric_df = relationship_df.sort_values(
        [metric_col, "count", "relationship_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    top_metric_row = ranked_metric_df.iloc[0]

    ranked_lift_df = relationship_df.sort_values(
        ["lift", "count", "relationship_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    top_lift_row = ranked_lift_df.iloc[0]

    source_rank_df = relationship_df.sort_values(
        ["source", "count", "relationship_label"],
        ascending=[True, False, True],
    ).copy()
    top_per_source = source_rank_df.groupby("source", as_index=False).first()
    dominant_source_row = top_per_source.sort_values(
        ["pct_source", "count", "relationship_label"],
        ascending=[False, False, True],
    ).iloc[0]

    return {
        "top_relationship_title": "Top Relationship",
        "top_relationship_value": build_relationship_label_display(
            top_metric_row,
            spec,
        ),
        "top_relationship_caption": (
            f"Leads on {metric_name.lower()} at "
            f"{format_metric_value(top_metric_row[metric_col], metric_col)}."
        ),
        "lift_title": "Strongest Lift",
        "lift_value": build_relationship_label_display(
            top_lift_row,
            spec,
        ),
        "lift_caption": (
            f"Lift {top_lift_row['lift']:.2f}; "
            f"{top_lift_row['count']:,} visible contributions."
        ),
        "concentration_title": f"Most Concentrated {spec['source_label']}",
        "concentration_value": build_relationship_label_display(
            dominant_source_row,
            spec,
        ),
        "concentration_caption": (
            f"{dominant_source_row['pct_source']:.1%} of visible contributions "
            f"from {display_value(dominant_source_row['source'], spec['source_label'])} "
            "flow through this strongest visible relationship."
        ),
    }


def render_insight_cards(
    relationship_df: pd.DataFrame,
    spec: dict,
    metric_name: str,
) -> None:
    """
    Render the top-row insight cards.

    Args:
        relationship_df: Relationship dataframe.
        spec: Relationship config.
        metric_name: User-selected metric label.
    """
    insights = build_relationship_insight_summary(
        relationship_df=relationship_df,
        spec=spec,
        metric_name=metric_name,
    )

    st.markdown("### 🧠 Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            insights["top_relationship_title"],
            insights["top_relationship_value"],
        )
        st.caption(insights["top_relationship_caption"])

    with col2:
        st.metric(
            insights["lift_title"],
            insights["lift_value"],
        )
        st.caption(insights["lift_caption"])

    with col3:
        st.metric(
            insights["concentration_title"],
            insights["concentration_value"],
        )
        st.caption(insights["concentration_caption"])


def build_sankey_supporting_insight(
    relationship_df: pd.DataFrame,
    spec: dict,
) -> str:
    """
    Build a short supporting insight for the Sankey section.

    Args:
        relationship_df: Relationship dataframe.
        spec: Relationship config.

    Returns:
        str: Supporting insight sentence.
    """
    if relationship_df.empty:
        return "No visible pattern remains to summarize."

    top_relationship_by_source = (
        relationship_df.sort_values(
            ["source", "count", "relationship_label"],
            ascending=[True, False, True],
        )
        .groupby("source", as_index=False)
        .first()
    )

    mean_top_share = float(top_relationship_by_source["pct_source"].mean())
    median_top_share = float(top_relationship_by_source["pct_source"].median())

    relationship_word = "path" if spec["mode"] == "three_hop" else "flow"

    return (
        f"💡 Across visible {pluralize(spec['source_label']).lower()}, the top "
        f"{relationship_word} accounts for an average of {mean_top_share:.1%} "
        f"of each source's visible contributions (median {median_top_share:.1%}), "
        "suggesting the visible structure is fairly concentrated rather than evenly distributed."
    )


def build_ranked_chart_supporting_insight(
    relationship_df: pd.DataFrame,
    spec: dict,
    top_n_edges: int,
) -> str:
    """
    Build a short supporting insight for the ranked relationship chart.

    Args:
        relationship_df: Relationship dataframe.
        spec: Relationship config.
        top_n_edges: Number of ranked relationships shown.

    Returns:
        str: Supporting insight sentence.
    """
    if relationship_df.empty:
        return "No ranked relationship insight is available."

    ranked_df = relationship_df.sort_values(
        ["count", "relationship_label"],
        ascending=[False, True],
    ).reset_index(drop=True)

    total_relationship_contributions = int(ranked_df["count"].sum())
    top_n_contributions = int(ranked_df.head(top_n_edges)["count"].sum())
    top_n_share = (
        top_n_contributions / total_relationship_contributions
        if total_relationship_contributions > 0 else 0.0
    )

    top_row = ranked_df.iloc[0]
    top_relationship_display = build_relationship_label_display(
        top_row,
        spec,
    )

    relationship_word = "paths" if spec["mode"] == "three_hop" else "flows"

    return (
        f"💡 The top {min(top_n_edges, len(ranked_df))} visible {relationship_word} "
        f"account for {top_n_share:.1%} of all visible contributions. The single "
        f"largest visible relationship is {top_relationship_display} "
        f"({top_row['count']:,} contributions)."
    )


def render_summary_metrics(
    filtered_df: pd.DataFrame,
    relationship_df: pd.DataFrame,
    spec: dict,
    metric_name: str,
) -> None:
    """
    Render headline page metrics.

    Args:
        filtered_df: Globally filtered dataframe.
        relationship_df: Relationship dataframe for the selected mode.
        spec: Relationship config.
        metric_name: User-selected metric label.
    """
    metric_col = get_edge_metric_column(metric_name)

    album_count = int(len(filtered_df))
    relationship_count = int(len(relationship_df))
    source_count = (
        int(relationship_df["source"].nunique())
        if not relationship_df.empty else 0
    )
    target_count = (
        int(relationship_df["target"].nunique())
        if not relationship_df.empty else 0
    )
    total_relationship_contributions = (
        int(relationship_df["count"].sum())
        if not relationship_df.empty else 0
    )

    ranked_df = relationship_df.sort_values(
        [metric_col, "count", "relationship_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top_metric_value = (
        float(ranked_df.iloc[0][metric_col])
        if not ranked_df.empty else 0.0
    )

    top_relationship_display = "None"
    if not ranked_df.empty:
        top_relationship_display = build_relationship_label_display(
            ranked_df.iloc[0],
            spec,
        )

    relationship_count_label = (
        "Paths Shown" if spec["mode"] == "three_hop" else "Flows Shown"
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Albums in View", f"{album_count:,}")

    with col2:
        st.metric(
            f"{pluralize(spec['source_label'])} in View",
            f"{source_count:,}",
        )

    with col3:
        st.metric(
            f"{pluralize(spec['target_label'])} in View",
            f"{target_count:,}",
        )

    with col4:
        st.metric(relationship_count_label, f"{relationship_count:,}")

    with col5:
        st.metric(
            "Total Contributions",
            f"{total_relationship_contributions:,}",
        )

    with col6:
        st.metric(
            f"Top {metric_name}",
            format_metric_value(top_metric_value, metric_col),
        )

    st.caption(
        "Top visible relationship: "
        f"{top_relationship_display}. "
        "Contributions can exceed album count because one album may contribute "
        "to multiple visible relationships."
    )


def build_sankey_link_df(
    relationship_df: pd.DataFrame,
    spec: dict,
    metric_name: str,
) -> tuple[pd.DataFrame, list[str], list[str], list[str] | None]:
    """
    Build link data for a two-column or three-column Sankey.

    Args:
        relationship_df: Relationship dataframe.
        spec: Relationship config.
        metric_name: User-selected metric label.

    Returns:
        tuple[pd.DataFrame, list[str], list[str], list[str] | None]:
            Link dataframe plus raw source, target, and optional middle nodes.
    """
    metric_col = get_edge_metric_column(metric_name)

    source_nodes_raw = sorted(relationship_df["source"].unique().tolist())
    target_nodes_raw = sorted(relationship_df["target"].unique().tolist())

    if spec["mode"] == "single_hop":
        link_df = relationship_df[["source", "target", "count"]].copy()

        if metric_col == "count":
            link_df["metric_value"] = link_df["count"].astype(float)
        else:
            link_df["metric_value"] = relationship_df[metric_col].astype(float).values

        return link_df, source_nodes_raw, target_nodes_raw, None

    middle_nodes_raw = sorted(relationship_df["middle"].unique().tolist())

    source_middle_df = (
        relationship_df.groupby(["source", "middle"], as_index=False)
        .agg(
            count=("count", "sum"),
            metric_value=(metric_col, "sum"),
        )
        .rename(columns={"middle": "target"})
    )

    source_middle_df["layer"] = "source_middle"

    middle_target_df = (
        relationship_df.groupby(["middle", "target"], as_index=False)
        .agg(
            count=("count", "sum"),
            metric_value=(metric_col, "sum"),
        )
        .rename(columns={"middle": "source"})
    )

    middle_target_df["layer"] = "middle_target"

    link_df = pd.concat(
        [source_middle_df, middle_target_df],
        ignore_index=True,
    )

    return link_df, source_nodes_raw, target_nodes_raw, middle_nodes_raw


def render_sankey(
    relationship_df: pd.DataFrame,
    spec: dict,
    metric_name: str,
) -> None:
    """
    Render a generic source→target or source→middle→target Sankey chart.

    Args:
        relationship_df: Relationship dataframe.
        spec: Relationship config.
        metric_name: User-selected metric label.
    """
    if relationship_df.empty:
        st.info("No visible relationships remain under the current filters.")
        return

    link_df, source_nodes_raw, target_nodes_raw, middle_nodes_raw = (
        build_sankey_link_df(
            relationship_df=relationship_df,
            spec=spec,
            metric_name=metric_name,
        )
    )

    if spec["mode"] == "single_hop":
        source_nodes_display = [
            display_value(value, spec["source_label"])
            for value in source_nodes_raw
        ]
        target_nodes_display = [
            display_value(value, spec["target_label"])
            for value in target_nodes_raw
        ]

        all_nodes_display = source_nodes_display + target_nodes_display
        source_index = {
            name: idx for idx, name in enumerate(source_nodes_raw)
        }
        target_index = {
            name: idx + len(source_nodes_raw)
            for idx, name in enumerate(target_nodes_raw)
        }

        source_idx = [source_index[value] for value in link_df["source"]]
        target_idx = [target_index[value] for value in link_df["target"]]

        node_contribution_counts = []
        node_pct = []
        node_role = []

        source_totals = relationship_df.groupby("source")["count"].sum().to_dict()
        target_totals = relationship_df.groupby("target")["count"].sum().to_dict()
        total_contributions = int(relationship_df["count"].sum())

        for raw_node in source_nodes_raw:
            count = int(source_totals.get(raw_node, 0))
            node_contribution_counts.append(count)
            node_pct.append(
                count / total_contributions if total_contributions > 0 else 0.0
            )
            node_role.append(spec["source_label"])

        for raw_node in target_nodes_raw:
            count = int(target_totals.get(raw_node, 0))
            node_contribution_counts.append(count)
            node_pct.append(
                count / total_contributions if total_contributions > 0 else 0.0
            )
            node_role.append(spec["target_label"])

        node_hover = [
            (
                f"{node_role[i]}: {all_nodes_display[i]}<br>"
                f"Visible contributions touching node: "
                f"{node_contribution_counts[i]:,}<br>"
                f"% of visible contributions: {node_pct[i]:.1%}"
            )
            for i in range(len(all_nodes_display))
        ]

        link_hover = [
            (
                f"{spec['source_label']}: {display_value(row['source'], spec['source_label'])}<br>"
                f"{spec['target_label']}: {display_value(row['target'], spec['target_label'])}<br>"
                f"Contributions: {int(row['count']):,}<br>"
                f"Selected metric weight: {format_metric_value(row['metric_value'], 'count') if metric_name == 'Count' else format_metric_value(row['metric_value'], get_edge_metric_column(metric_name))}"
            )
            for _, row in link_df.iterrows()
        ]

        node_colors = (
            ["#4C78A8"] * len(source_nodes_raw) +
            ["#F58518"] * len(target_nodes_raw)
        )

    else:
        middle_nodes_display = [
            display_value(value, spec["middle_label"])
            for value in middle_nodes_raw or []
        ]
        source_nodes_display = [
            display_value(value, spec["source_label"])
            for value in source_nodes_raw
        ]
        target_nodes_display = [
            display_value(value, spec["target_label"])
            for value in target_nodes_raw
        ]

        all_nodes_display = (
            source_nodes_display +
            middle_nodes_display +
            target_nodes_display
        )

        source_index = {
            name: idx for idx, name in enumerate(source_nodes_raw)
        }
        middle_index = {
            name: idx + len(source_nodes_raw)
            for idx, name in enumerate(middle_nodes_raw or [])
        }
        target_index = {
            name: idx + len(source_nodes_raw) + len(middle_nodes_raw or [])
            for idx, name in enumerate(target_nodes_raw)
        }

        source_idx = []
        target_idx = []

        for _, row in link_df.iterrows():
            if row["layer"] == "source_middle":
                source_idx.append(source_index[row["source"]])
                target_idx.append(middle_index[row["target"]])
            else:
                source_idx.append(middle_index[row["source"]])
                target_idx.append(target_index[row["target"]])

        total_contributions = int(relationship_df["count"].sum())

        source_totals = relationship_df.groupby("source")["count"].sum().to_dict()
        middle_totals = (
            relationship_df.groupby("middle")["count"].sum().to_dict()
        )
        target_totals = relationship_df.groupby("target")["count"].sum().to_dict()

        node_contribution_counts = []
        node_pct = []
        node_role = []

        for raw_node in source_nodes_raw:
            count = int(source_totals.get(raw_node, 0))
            node_contribution_counts.append(count)
            node_pct.append(
                count / total_contributions if total_contributions > 0 else 0.0
            )
            node_role.append(spec["source_label"])

        for raw_node in middle_nodes_raw or []:
            count = int(middle_totals.get(raw_node, 0))
            node_contribution_counts.append(count)
            node_pct.append(
                count / total_contributions if total_contributions > 0 else 0.0
            )
            node_role.append(spec["middle_label"])

        for raw_node in target_nodes_raw:
            count = int(target_totals.get(raw_node, 0))
            node_contribution_counts.append(count)
            node_pct.append(
                count / total_contributions if total_contributions > 0 else 0.0
            )
            node_role.append(spec["target_label"])

        node_hover = [
            (
                f"{node_role[i]}: {all_nodes_display[i]}<br>"
                f"Visible path contributions touching node: "
                f"{node_contribution_counts[i]:,}<br>"
                f"% of visible path contributions: {node_pct[i]:.1%}"
            )
            for i in range(len(all_nodes_display))
        ]

        link_hover = []
        for _, row in link_df.iterrows():
            if row["layer"] == "source_middle":
                hover = (
                    f"{spec['source_label']}: {display_value(row['source'], spec['source_label'])}<br>"
                    f"{spec['middle_label']}: {display_value(row['target'], spec['middle_label'])}<br>"
                    f"Visible path contributions through pair: {int(row['count']):,}<br>"
                    f"Selected metric weight: {format_metric_value(row['metric_value'], 'count') if metric_name == 'Count' else format_metric_value(row['metric_value'], get_edge_metric_column(metric_name))}"
                )
            else:
                hover = (
                    f"{spec['middle_label']}: {display_value(row['source'], spec['middle_label'])}<br>"
                    f"{spec['target_label']}: {display_value(row['target'], spec['target_label'])}<br>"
                    f"Visible path contributions through pair: {int(row['count']):,}<br>"
                    f"Selected metric weight: {format_metric_value(row['metric_value'], 'count') if metric_name == 'Count' else format_metric_value(row['metric_value'], get_edge_metric_column(metric_name))}"
                )
            link_hover.append(hover)

        node_colors = (
            ["#4C78A8"] * len(source_nodes_raw) +
            ["#72B7B2"] * len(middle_nodes_raw or []) +
            ["#F58518"] * len(target_nodes_raw)
        )

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(
                        color="rgba(255,255,255,0.2)",
                        width=0.5,
                    ),
                    label=all_nodes_display,
                    color=node_colors,
                    customdata=node_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                link=dict(
                    source=source_idx,
                    target=target_idx,
                    value=link_df["metric_value"].astype(float).tolist(),
                    customdata=link_hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )

    title_text = (
        f"{spec['source_label']} → {spec['middle_label']} → {spec['target_label']} Sankey ({metric_name})"
        if spec["mode"] == "three_hop"
        else f"{spec['source_label']} → {spec['target_label']} Sankey ({metric_name})"
    )

    fig.update_layout(
        title=title_text,
        font=dict(color="white"),
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        height=720 if spec["mode"] == "three_hop" else 700,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    st.plotly_chart(fig, width="stretch")


def main() -> None:
    """
    Render the Cross-Entity Explorer page.
    """
    apply_app_styles()

    st.title("Cross-Entity Explorer")
    st.write(
        """
        Explore directional relationships across different entity families.
        This page supports both single-hop flows and a constrained multi-hop
        path view across awards, film genres, and album genres.
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

    controls = get_cross_entity_controls(
        relationship_options=get_relationship_type_options(),
    )

    relationship_type = controls["relationship_type"]
    spec = RELATIONSHIP_SPECS[relationship_type]

    filtered_df = filter_dataset(cross_df, global_filters)
    filtered_df = filter_high_cardinality_entities(
        df=filtered_df,
        controls=controls,
        spec=spec,
    )

    if filtered_df.empty:
        st.warning(
            "No albums remain after applying the current global filters. "
            "Try broadening the year range or clearing one of the genre filters."
        )
        return

    selected_values = get_selected_values_for_relationship(
        spec=spec,
        global_filters=global_filters,
    )

    if spec["mode"] == "single_hop":
        relationship_df = build_single_hop_flow_df(
            df=filtered_df,
            spec=spec,
            min_flow_count=controls["min_edge_count"],
            selected_source_values=selected_values["source"],
            selected_target_values=selected_values["target"],
        )
    else:
        relationship_df = build_three_hop_path_df(
            df=filtered_df,
            spec=spec,
            min_flow_count=controls["min_edge_count"],
            selected_source_values=selected_values["source"],
            selected_middle_values=selected_values["middle"],
            selected_target_values=selected_values["target"],
        )

    if relationship_df.empty:
        st.warning(
            "No visible relationships remain after applying the current filters "
            "and minimum threshold. Try lowering the threshold or broadening "
            "the selected genres."
        )
        return

    st.caption(
        get_metric_explainer(
            metric_name=controls["edge_metric"],
            spec=spec,
        )
    )

    st.markdown("**Filter Context**")
    st.caption(
        build_filter_context_caption(
            relationship_type=relationship_type,
            global_filters=global_filters,
        )
    )

    render_insight_cards(
        relationship_df=relationship_df,
        spec=spec,
        metric_name=controls["edge_metric"],
    )

    st.markdown("---")

    render_summary_metrics(
        filtered_df=filtered_df,
        relationship_df=relationship_df,
        spec=spec,
        metric_name=controls["edge_metric"],
    )

    sankey_title = (
        f"{spec['source_label']} → {spec['middle_label']} → {spec['target_label']} Sankey"
        if spec["mode"] == "three_hop"
        else f"{spec['source_label']} → {spec['target_label']} Sankey"
    )
    st.subheader(sankey_title)
    render_sankey(
        relationship_df=relationship_df,
        spec=spec,
        metric_name=controls["edge_metric"],
    )
    st.caption(
        "A single album may contribute to multiple visible relationships when it has "
        "multiple relevant source, intermediate, and/or target values."
    )
    st.caption(
        build_sankey_supporting_insight(
            relationship_df=relationship_df,
            spec=spec,
        )
    )

    ranked_title = (
        f"Top {spec['source_label']} → {spec['middle_label']} → {spec['target_label']} Paths"
        if spec["mode"] == "three_hop"
        else f"Top {spec['source_label']} → {spec['target_label']} Flows"
    )
    st.subheader(ranked_title)
    top_relationships_chart = create_top_relationships_chart(
        relationship_df=relationship_df,
        spec=spec,
        top_n_edges=controls["top_n_edges"],
        metric_name=controls["edge_metric"],
    )
    st.altair_chart(top_relationships_chart, width="stretch")
    st.caption(
        build_ranked_chart_supporting_insight(
            relationship_df=relationship_df,
            spec=spec,
            top_n_edges=controls["top_n_edges"],
        )
    )

    if controls["show_edge_table"]:
        st.markdown("#### Relationship Summary Table")

        metric_col = get_edge_metric_column(controls["edge_metric"])
        metric_side_labels = get_metric_side_labels(
            spec["source_label"],
            spec["target_label"],
        )

        relationship_table = relationship_df.sort_values(
            [metric_col, "count", "relationship_label"],
            ascending=[False, False, True],
        ).copy()

        relationship_table["source"] = relationship_table["source"].apply(
            lambda value: display_value(value, spec["source_label"])
        )
        relationship_table["target"] = relationship_table["target"].apply(
            lambda value: display_value(value, spec["target_label"])
        )

        if spec["mode"] == "three_hop":
            relationship_table["middle"] = relationship_table["middle"].apply(
                lambda value: display_value(value, spec["middle_label"])
            )
            relationship_table["relationship_label"] = (
                relationship_table["source"] + " → " +
                relationship_table["middle"] + " → " +
                relationship_table["target"]
            )
        else:
            relationship_table["relationship_label"] = (
                relationship_table["source"] + " → " + relationship_table["target"]
            )

        rename_map = {
            "source": spec["source_label"],
            "target": spec["target_label"],
            "relationship_label": "Relationship",
            "count": "Contributions",
            "source_count": f"Visible Contributions from {spec['source_label']}",
            "target_count": f"Visible Contributions to {spec['target_label']}",
            "union_count": "Visible Contributions in Union",
            "pct_source": metric_side_labels["pct_source"],
            "pct_target": metric_side_labels["pct_target"],
            "jaccard": "Jaccard",
            "lift": "Lift",
        }

        if spec["mode"] == "three_hop":
            rename_map["middle"] = spec["middle_label"]

        relationship_table = relationship_table.rename(columns=rename_map)

        st.dataframe(
            relationship_table,
            width="stretch",
            hide_index=True,
        )

    relationship_prompt = (
        f"{spec['source_label']} → {spec['middle_label']} → {spec['target_label']} path"
        if spec["mode"] == "three_hop"
        else f"{spec['source_label']} → {spec['target_label']} flow"
    )
    st.subheader(f"Inspect a {relationship_prompt}")
    relationship_options = relationship_df["relationship_label"].tolist()
    selected_relationship_label = st.selectbox(
        relationship_prompt,
        options=relationship_options,
        index=0,
        key="cross_entity_relationship_select",
    )

    selected_row = relationship_df[
        relationship_df["relationship_label"] == selected_relationship_label
    ].iloc[0]

    selected_source = selected_row["source"]
    selected_target = selected_row["target"]
    selected_middle = selected_row["middle"] if spec["mode"] == "three_hop" else None

    if spec["mode"] == "three_hop":
        st.caption(
            f"Showing albums with {spec['source_label'].lower()} "
            f"**{display_value(selected_source, spec['source_label'])}**, "
            f"{spec['middle_label'].lower()} "
            f"**{display_value(selected_middle, spec['middle_label'])}**, and "
            f"{spec['target_label'].lower()} "
            f"**{display_value(selected_target, spec['target_label'])}**."
        )
    else:
        st.caption(
            f"Showing albums with {spec['source_label'].lower()} "
            f"**{display_value(selected_source, spec['source_label'])}** and "
            f"{spec['target_label'].lower()} "
            f"**{display_value(selected_target, spec['target_label'])}**."
        )

    detail_df = build_detail_table(
        df=filtered_df,
        spec=spec,
        selected_source=selected_source,
        selected_target=selected_target,
        selected_middle=selected_middle,
    )

    if detail_df.empty:
        st.info("No albums match the selected relationship.")
        return

    if controls["show_album_table"]:
        detail_display_df = detail_df.copy()

        for col in AWARD_FLAGS:
            if col in detail_display_df.columns:
                detail_display_df[col] = detail_display_df[col].map(
                    lambda x: "Yes" if pd.notna(x) and int(x) == 1 else ""
                )

        st.dataframe(
            rename_columns_for_display(detail_display_df),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()