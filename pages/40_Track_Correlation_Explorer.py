from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.app_controls import (
    get_global_filter_controls,
    get_track_correlation_controls,
)
from app.app_data import load_track_data_explorer_data
from app.data_filters import filter_dataset
from app.explorer_shared import (
    add_film_year_bucket,
    add_standard_multivalue_groups,
    get_clean_composer_options,
    get_global_filter_inputs,
    rename_and_dedupe_for_display,
    select_unique_existing_columns,
)
from app.ui import apply_app_styles, get_display_label


TRACK_SUCCESS_ANCHORS = [
    "lfm_track_listeners",
    "lfm_track_playcount",
    "spotify_popularity",
]

TRACK_PREDICTOR_FEATURES = [
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
    "relative_track_position",
]

TRACK_CORRELATION_FEATURES = TRACK_SUCCESS_ANCHORS + TRACK_PREDICTOR_FEATURES

CORE_AUDIO_REQUIRED_COLS = [
    "energy",
    "danceability",
    "happiness",
    "instrumentalness",
]

SOURCE_DISPLAY_COLUMNS = [
    "film_title",
    "album_title",
    "track_title",
    "track_number",
    "track_position_bucket",
    "relative_track_position",
    "composer_primary_clean",
    "label_names",
    "film_year",
    "film_genres",
    "album_genres_display",
    "lfm_track_listeners",
    "lfm_track_playcount",
    "spotify_popularity",
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
]


def add_track_correlation_display_fields(track_df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped display fields used by the Track Correlation Explorer."""
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


def filter_track_correlation_dataset(
    df: pd.DataFrame,
    controls: dict,
) -> pd.DataFrame:
    """Apply shared filters plus page-specific track filters."""
    filtered = filter_dataset(df, controls).copy()

    if "track_number" in filtered.columns:
        filtered = filtered[
            filtered["track_number"] <= controls["max_track_position"]
        ].copy()

    if "lfm_album_listeners" in filtered.columns:
        filtered = filtered[
            filtered["lfm_album_listeners"].fillna(0) >= controls["min_album_listeners"]
        ].copy()

    if controls.get("audio_only", False):
        required_cols = [col for col in CORE_AUDIO_REQUIRED_COLS if col in filtered.columns]
        if required_cols:
            filtered = filtered.dropna(subset=required_cols).copy()

    return filtered


def build_filter_context_caption(controls: dict) -> str:
    """Build a short caption describing the current filter scope."""
    parts = []

    year_min, year_max = controls["year_range"]
    parts.append(f"Film years {year_min}–{year_max}")

    if controls.get("selected_film_genres"):
        shown = ", ".join(controls["selected_film_genres"][:5])
        if len(controls["selected_film_genres"]) > 5:
            shown += ", ..."
        parts.append(f"Film genres: {shown}")

    if controls.get("selected_album_genres"):
        shown = ", ".join(controls["selected_album_genres"][:5])
        if len(controls["selected_album_genres"]) > 5:
            shown += ", ..."
        parts.append(f"Album genres: {shown}")

    if controls.get("selected_composers"):
        shown = ", ".join(controls["selected_composers"][:3])
        if len(controls["selected_composers"]) > 3:
            shown += ", ..."
        parts.append(f"Composers: {shown}")

    if controls["min_album_listeners"] > 0:
        parts.append(f"Min album listeners: {controls['min_album_listeners']:,}")

    parts.append(f"Max track position: {controls['max_track_position']}")
    parts.append(f"Method: {controls['method'].title()}")
    parts.append(f"Anchor: {get_display_label(controls['anchor_metric'])}")

    if controls["audio_only"]:
        parts.append("Only tracks with core audio features")

    return " | ".join(parts)


def get_available_correlation_features(df: pd.DataFrame) -> list[str]:
    """Return the correlation feature subset available in the dataframe."""
    return [col for col in TRACK_CORRELATION_FEATURES if col in df.columns]


def build_numeric_correlation_df(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Select and coerce correlation features to numeric."""
    if not feature_cols:
        return pd.DataFrame()

    numeric_df = df[feature_cols].copy()

    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

    return numeric_df


def compute_correlation_matrix(
    numeric_df: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Compute pairwise correlation matrix."""
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return pd.DataFrame()

    corr_matrix = numeric_df.corr(method=method)
    corr_matrix = corr_matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")

    keep = [col for col in corr_matrix.columns if col in corr_matrix.index]
    if len(keep) < 2:
        return pd.DataFrame()

    return corr_matrix.loc[keep, keep].copy()


def restrict_heatmap_scope(
    corr_matrix: pd.DataFrame,
    heatmap_scope: str,
) -> pd.DataFrame:
    """Optionally restrict the heatmap to predictor features only."""
    if corr_matrix.empty:
        return corr_matrix

    if heatmap_scope != "Predictors only":
        return corr_matrix

    predictor_cols = [
        col for col in TRACK_PREDICTOR_FEATURES
        if col in corr_matrix.index and col in corr_matrix.columns
    ]
    if len(predictor_cols) < 2:
        return corr_matrix

    return corr_matrix.loc[predictor_cols, predictor_cols].copy()


def corr_to_long(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide correlation matrix into long form."""
    if corr_matrix.empty:
        return pd.DataFrame(columns=["feature_x", "feature_y", "correlation"])

    return (
        corr_matrix.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_x", "level_1": "feature_y", 0: "correlation"})
    )


def build_anchor_ranking_df(
    corr_matrix: pd.DataFrame,
    anchor_metric: str,
    ranking_mode: str,
    top_n: int,
) -> pd.DataFrame:
    """Build ranked correlations to the selected anchor metric."""
    if corr_matrix.empty or anchor_metric not in corr_matrix.columns:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])

    ranking_df = (
        corr_matrix[[anchor_metric]]
        .reset_index()
        .rename(columns={"index": "feature", anchor_metric: "correlation"})
    )

    ranking_df = ranking_df[ranking_df["feature"] != anchor_metric].copy()
    ranking_df["abs_correlation"] = ranking_df["correlation"].abs()

    if ranking_mode == "Positive only":
        ranking_df = ranking_df[ranking_df["correlation"] > 0].copy()
        ranking_df = ranking_df.sort_values(
            ["correlation", "feature"],
            ascending=[False, True],
        )
    elif ranking_mode == "Negative only":
        ranking_df = ranking_df[ranking_df["correlation"] < 0].copy()
        ranking_df = ranking_df.sort_values(
            ["correlation", "feature"],
            ascending=[True, True],
        )
    else:
        ranking_df = ranking_df.sort_values(
            ["abs_correlation", "correlation"],
            ascending=[False, False],
        )

    return ranking_df.head(top_n).reset_index(drop=True)


def build_predictor_redundancy_df(
    corr_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Build a ranked table of strongest predictor-predictor correlations."""
    predictor_cols = [
        col for col in TRACK_PREDICTOR_FEATURES
        if col in corr_matrix.index and col in corr_matrix.columns
    ]
    if len(predictor_cols) < 2:
        return pd.DataFrame(
            columns=["feature_1", "feature_2", "correlation", "abs_correlation", "risk_band"]
        )

    predictor_matrix = corr_matrix.loc[predictor_cols, predictor_cols].copy()

    rows = []
    for i, col_1 in enumerate(predictor_matrix.columns):
        for col_2 in predictor_matrix.columns[i + 1:]:
            corr_value = predictor_matrix.loc[col_1, col_2]
            if pd.isna(corr_value):
                continue

            abs_corr = abs(float(corr_value))
            if abs_corr >= 0.80:
                risk_band = "Very high"
            elif abs_corr >= 0.65:
                risk_band = "Moderate"
            else:
                risk_band = "Lower"

            rows.append(
                {
                    "feature_1": col_1,
                    "feature_2": col_2,
                    "correlation": float(corr_value),
                    "abs_correlation": abs_corr,
                    "risk_band": risk_band,
                }
            )

    redundancy_df = pd.DataFrame(rows)
    if redundancy_df.empty:
        return redundancy_df

    return redundancy_df.sort_values(
        ["abs_correlation", "correlation"],
        ascending=[False, False],
    ).reset_index(drop=True)


def classify_correlation_strength(value: float) -> str:
    """Convert absolute correlation magnitude into a readable label."""
    abs_value = abs(value)

    if abs_value >= 0.80:
        return "Very strong"
    if abs_value >= 0.65:
        return "Strong"
    if abs_value >= 0.40:
        return "Moderate"
    if abs_value >= 0.20:
        return "Weak–moderate"
    return "Weak"


def get_least_connected_predictor(
    corr_matrix: pd.DataFrame,
) -> tuple[str, float] | None:
    """Return the predictor with the weakest average absolute correlation."""
    predictor_cols = [
        col for col in TRACK_PREDICTOR_FEATURES
        if col in corr_matrix.index and col in corr_matrix.columns
    ]
    if len(predictor_cols) < 2:
        return None

    predictor_matrix = corr_matrix.loc[predictor_cols, predictor_cols].abs().copy()
    np.fill_diagonal(predictor_matrix.values, np.nan)

    mean_abs = predictor_matrix.mean(axis=1, skipna=True).dropna()
    if mean_abs.empty:
        return None

    weakest = mean_abs.sort_values(ascending=True).index[0]
    return weakest, float(mean_abs.loc[weakest])


def build_archetype_opportunity_text(
    redundancy_df: pd.DataFrame,
) -> str:
    """Build a short archetype-oriented interpretation from the strongest predictor pair."""
    if redundancy_df.empty:
        return "No clear archetype candidate pair is visible under the current filters."

    top_pair = redundancy_df.iloc[0]
    f1 = str(top_pair["feature_1"])
    f2 = str(top_pair["feature_2"])
    abs_corr = float(top_pair["abs_correlation"])

    pair_set = {f1, f2}

    if pair_set == {"energy", "loudness"}:
        family = "an intensity dimension"
    elif pair_set == {"acousticness", "instrumentalness"}:
        family = "a score-like acoustic texture dimension"
    elif pair_set == {"danceability", "happiness"}:
        family = "an upbeat / accessible feel dimension"
    else:
        family = "a possible shared track-character dimension"

    return (
        f"{get_display_label(f1)} and {get_display_label(f2)} move together strongly "
        f"(|r| = {abs_corr:.2f}), suggesting {family} that could later be turned "
        f"into an archetype lens on other track pages."
    )


def build_view_context_caption(
    controls: dict,
    filtered_df: pd.DataFrame,
    feature_cols: list[str],
) -> str:
    """Build a natural-language scope caption for the page."""
    predictor_count = len([col for col in feature_cols if col in TRACK_PREDICTOR_FEATURES])
    anchor_count = len([col for col in feature_cols if col in TRACK_SUCCESS_ANCHORS])

    return (
        f"Using {controls['method'].title()} correlation across {len(filtered_df):,} visible tracks, "
        f"with {anchor_count} success anchors and {predictor_count} candidate predictors. "
        f"The ranked view is anchored to {get_display_label(controls['anchor_metric'])}, "
        f"and the heatmap scope is set to {controls['heatmap_scope'].lower()}."
    )


def build_heatmap_supporting_insight(
    corr_matrix: pd.DataFrame,
) -> str:
    """Build a supporting insight for the heatmap."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return "💡 Not enough variables remain to summarize the visible matrix."

    long_df = corr_to_long(corr_matrix)
    long_df = long_df[long_df["feature_x"] != long_df["feature_y"]].copy()

    long_df["pair_key"] = long_df.apply(
        lambda row: "||".join(sorted([str(row["feature_x"]), str(row["feature_y"])])),
        axis=1,
    )
    long_df = long_df.sort_values("pair_key").drop_duplicates("pair_key")
    long_df["abs_correlation"] = long_df["correlation"].abs()

    if long_df.empty:
        return "💡 No off-diagonal relationships remain to summarize."

    top_pair = long_df.sort_values("abs_correlation", ascending=False).iloc[0]
    strength = classify_correlation_strength(float(top_pair["correlation"]))

    return (
        f"💡 The strongest visible off-diagonal relationship is between "
        f"**{get_display_label(str(top_pair['feature_x']))}** and "
        f"**{get_display_label(str(top_pair['feature_y']))}** "
        f"(r = {top_pair['correlation']:.2f}), which is {strength.lower()}."
    )


def build_anchor_supporting_insight(
    ranking_df: pd.DataFrame,
    anchor_metric: str,
    method: str,
    ranking_mode: str,
) -> str:
    """Build a supporting insight for the anchor-ranking section."""
    if ranking_df.empty:
        return "💡 No ranked relationships remain under the current settings."

    top_row = ranking_df.iloc[0]
    feature = get_display_label(str(top_row["feature"]))
    corr_value = float(top_row["correlation"])
    strength = classify_correlation_strength(corr_value).lower()
    direction = "positive" if corr_value > 0 else "negative"

    if ranking_mode == "Positive only":
        prefix = "Within the positive-only ranking"
    elif ranking_mode == "Negative only":
        prefix = "Within the negative-only ranking"
    else:
        prefix = "Within the visible ranking"

    return (
        f"💡 {prefix}, **{feature}** has the strongest {method.title()} association "
        f"with **{get_display_label(anchor_metric)}** "
        f"(r = {corr_value:.2f}), which is a {strength} {direction} relationship."
    )


def build_redundancy_supporting_insight(
    redundancy_df: pd.DataFrame,
) -> str:
    """Build a supporting insight for the multicollinearity watchlist."""
    if redundancy_df.empty:
        return "💡 Not enough predictor features remain to screen for redundancy."

    top_row = redundancy_df.iloc[0]
    f1 = get_display_label(str(top_row["feature_1"]))
    f2 = get_display_label(str(top_row["feature_2"]))
    abs_corr = float(top_row["abs_correlation"])
    risk_band = str(top_row["risk_band"])

    return (
        f"💡 The tightest predictor pair is **{f1}** and **{f2}** "
        f"(|r| = {abs_corr:.2f}), which falls in the **{risk_band.lower()}** "
        f"multicollinearity watch band for later regression work."
    )


def build_insight_cards(
    ranking_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    anchor_metric: str,
) -> list[tuple[str, str, str]]:
    """Build the three top-line insight cards."""
    cards: list[tuple[str, str, str]] = []

    if not ranking_df.empty:
        top_anchor = ranking_df.iloc[0]
        cards.append(
            (
                "Strongest Success Signal",
                get_display_label(str(top_anchor["feature"])),
                f"r = {top_anchor['correlation']:.2f} vs {get_display_label(anchor_metric)}",
            )
        )
    else:
        cards.append(
            (
                "Strongest Success Signal",
                "Not available",
                "No ranked anchor relationships remain.",
            )
        )

    if not redundancy_df.empty:
        top_pair = redundancy_df.iloc[0]
        cards.append(
            (
                "Top Archetype Opportunity",
                f"{get_display_label(str(top_pair['feature_1']))} + {get_display_label(str(top_pair['feature_2']))}",
                f"|r| = {top_pair['abs_correlation']:.2f}",
            )
        )
    else:
        cards.append(
            (
                "Top Archetype Opportunity",
                "Not available",
                "No strong predictor pair is visible.",
            )
        )

    weakest = get_least_connected_predictor(corr_matrix)
    if weakest is not None:
        weakest_feature, weakest_avg = weakest
        cards.append(
            (
                "Least Connected Predictor",
                get_display_label(weakest_feature),
                f"Avg |r| = {weakest_avg:.2f}",
            )
        )
    else:
        cards.append(
            (
                "Least Connected Predictor",
                "Not available",
                "Not enough predictors remain to summarize.",
            )
        )

    return cards


def create_heatmap_chart(corr_matrix: pd.DataFrame) -> alt.Chart:
    """Create an Altair heatmap from the correlation matrix."""
    display_matrix = corr_matrix.copy()
    display_matrix.index = [get_display_label(str(col)) for col in display_matrix.index]
    display_matrix.columns = [get_display_label(str(col)) for col in display_matrix.columns]

    long_df = corr_to_long(display_matrix)

    return (
        alt.Chart(long_df)
        .mark_rect()
        .encode(
            x=alt.X("feature_x:N", title=None, sort=list(display_matrix.columns)),
            y=alt.Y("feature_y:N", title=None, sort=list(display_matrix.index)),
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(domain=[-1, 1]),
                title="Correlation",
            ),
            tooltip=[
                alt.Tooltip("feature_x:N", title="Feature X"),
                alt.Tooltip("feature_y:N", title="Feature Y"),
                alt.Tooltip("correlation:Q", title="Correlation", format=".2f"),
            ],
        )
        .properties(
            height=600,
            title="Track Correlation Heatmap",
        )
        .configure_axisX(
            labelAngle=-45,
            labelFontSize=10,
            labelLimit=220,
            labelOverlap=False,
        )
        .configure_axisY(
            labelFontSize=10,
            labelLimit=220,
            labelOverlap=False,
        )
    )



def create_anchor_ranking_chart(
    ranking_df: pd.DataFrame,
    anchor_metric: str,
) -> alt.Chart:
    """Create a bar chart of ranked anchor associations."""
    plot_df = ranking_df.copy()
    plot_df["feature_label"] = plot_df["feature"].astype(str).map(get_display_label)

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "feature_label:N",
                sort=alt.SortField(field="abs_correlation", order="descending"),
                title="Feature",
            ),
            x=alt.X("correlation:Q", title="Correlation"),
            tooltip=[
                alt.Tooltip("feature_label:N", title="Feature"),
                alt.Tooltip("correlation:Q", title="Correlation", format=".2f"),
                alt.Tooltip("abs_correlation:Q", title="Absolute Correlation", format=".2f"),
            ],
        )
        .properties(
            height=max(280, 34 * len(plot_df)),
            title=f"Top Associations to {get_display_label(anchor_metric)}",
        )
    )


def create_redundancy_chart(
    redundancy_df: pd.DataFrame,
    top_n: int = 10,
) -> alt.Chart:
    """Create a bar chart of strongest predictor-predictor correlations."""
    plot_df = redundancy_df.head(top_n).copy()
    plot_df["pair_label"] = (
        plot_df["feature_1"].astype(str).map(get_display_label)
        + " ↔ "
        + plot_df["feature_2"].astype(str).map(get_display_label)
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "pair_label:N",
                sort=alt.SortField(field="abs_correlation", order="descending"),
                title="Predictor Pair",
            ),
            x=alt.X("abs_correlation:Q", title="Absolute Correlation"),
            tooltip=[
                alt.Tooltip("pair_label:N", title="Pair"),
                alt.Tooltip("correlation:Q", title="Correlation", format=".2f"),
                alt.Tooltip("risk_band:N", title="Risk Band"),
            ],
        )
        .properties(
            height=max(280, 34 * len(plot_df)),
            title="Predictor Redundancy / Multicollinearity Watchlist",
        )
    )


def get_safe_source_display_columns(df: pd.DataFrame) -> list[str]:
    """Return the subset of preferred source-table columns present in the dataframe."""
    return [col for col in SOURCE_DISPLAY_COLUMNS if col in df.columns]


def main() -> None:
    """Render the Track Correlation Explorer."""
    st.set_page_config(
        page_title="Track Correlation Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Correlation Explorer")
    st.write(
        """
        Explore how track-level success metrics and audio features move together
        across the visible track set. This page supports three goals:
        understanding track structure, surfacing candidate archetype dimensions,
        and flagging highly correlated predictors before regression work.
        """
    )

    track_df = load_track_data_explorer_data()
    track_df = add_track_correlation_display_fields(track_df)

    filter_inputs = get_global_filter_inputs(track_df)
    composer_options = get_clean_composer_options(track_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    local_controls = get_track_correlation_controls(
        composer_options=composer_options,
    )

    controls = {
        **global_controls,
        **local_controls,
    }

    filtered_df = filter_track_correlation_dataset(track_df, controls)

    if filtered_df.empty:
        st.warning("No tracks remain under the current filters.")
        return

    feature_cols = get_available_correlation_features(filtered_df)
    numeric_df = build_numeric_correlation_df(filtered_df, feature_cols)
    corr_matrix = compute_correlation_matrix(
        numeric_df=numeric_df,
        method=controls["method"],
    )

    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        st.warning("Not enough numeric variables remain to compute correlations.")
        return

    heatmap_matrix = restrict_heatmap_scope(
        corr_matrix=corr_matrix,
        heatmap_scope=controls["heatmap_scope"],
    )

    ranking_df = build_anchor_ranking_df(
        corr_matrix=corr_matrix,
        anchor_metric=controls["anchor_metric"],
        ranking_mode=controls["ranking_mode"],
        top_n=controls["top_n"],
    )
    if not ranking_df.empty:
        ranking_df["anchor_metric"] = controls["anchor_metric"]

    redundancy_df = build_predictor_redundancy_df(corr_matrix)

    st.markdown("**Filter Context**")
    st.caption(build_filter_context_caption(controls))

    st.markdown("**View Context**")
    st.caption(
        build_view_context_caption(
            controls=controls,
            filtered_df=filtered_df,
            feature_cols=feature_cols,
        )
    )

    track_count = len(filtered_df)
    feature_count = len(feature_cols)
    usable_predictor_count = len(
        [col for col in TRACK_PREDICTOR_FEATURES if col in corr_matrix.columns]
    )
    high_risk_pairs = int((redundancy_df["abs_correlation"] >= 0.80).sum()) if not redundancy_df.empty else 0
    moderate_plus_pairs = int((redundancy_df["abs_correlation"] >= 0.65).sum()) if not redundancy_df.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Visible Tracks", f"{track_count:,}")
    c2.metric("Correlation Features", f"{feature_count:,}")
    c3.metric("Predictor Features", f"{usable_predictor_count:,}")
    c4.metric("Very High-Risk Pairs", f"{high_risk_pairs:,}")
    c5.metric("Moderate+ Redundancy Pairs", f"{moderate_plus_pairs:,}")

    st.markdown("### 🧠 Key Insights")
    insight_cards = build_insight_cards(
        ranking_df=ranking_df,
        redundancy_df=redundancy_df,
        corr_matrix=corr_matrix,
        anchor_metric=controls["anchor_metric"],
    )
    insight_cols = st.columns(3)
    for i, (title, value, caption) in enumerate(insight_cards):
        with insight_cols[i]:
            st.metric(title, value)
            st.caption(caption)

    st.markdown("### Correlation Heatmap")
    st.caption(
        "Use the matrix to identify broad relationship structure, promising archetype families, "
        "and highly redundant predictor clusters."
    )
    st.altair_chart(create_heatmap_chart(heatmap_matrix), width="stretch")
    st.caption(build_heatmap_supporting_insight(heatmap_matrix))

    st.divider()

    st.markdown("### Ranked Associations to Selected Success Anchor")
    st.caption(
        "This view shows which visible features are most associated with the selected track-success metric."
    )

    if ranking_df.empty:
        st.info("No ranked anchor relationships remain under the current settings.")
    else:
        st.altair_chart(
            create_anchor_ranking_chart(
                ranking_df=ranking_df,
                anchor_metric=controls["anchor_metric"],
            ),
            width="stretch",
        )
        st.caption(
            build_anchor_supporting_insight(
                ranking_df=ranking_df,
                anchor_metric=controls["anchor_metric"],
                method=controls["method"],
                ranking_mode=controls["ranking_mode"],
            )
        )

        if controls["show_ranked_table"]:
            ranked_display = ranking_df.copy()
            ranked_display["feature"] = ranked_display["feature"].astype(str).map(get_display_label)
            ranked_display = ranked_display.rename(
                columns={
                    "feature": "Feature",
                    "correlation": "Correlation",
                    "abs_correlation": "Absolute Correlation",
                    "anchor_metric": "Anchor Metric",
                }
            )
            ranked_display["Anchor Metric"] = ranked_display["Anchor Metric"].astype(str).map(get_display_label)
            st.dataframe(ranked_display, width="stretch", hide_index=True)

    st.divider()

    st.markdown("### Archetype + Multicollinearity Watchlist")
    st.caption(
        "These are the strongest predictor-predictor relationships. They can suggest either "
        "a shared archetype dimension or a future multicollinearity problem in regression."
    )

    if redundancy_df.empty:
        st.info("Not enough predictor features remain to evaluate redundancy.")
    else:
        st.altair_chart(create_redundancy_chart(redundancy_df, top_n=controls["top_n"]), width="stretch")
        st.caption(build_redundancy_supporting_insight(redundancy_df))
        st.caption(f"💡 Archetype interpretation: {build_archetype_opportunity_text(redundancy_df)}")

        if controls["show_redundancy_table"]:
            redundancy_display = redundancy_df.copy()
            redundancy_display["feature_1"] = redundancy_display["feature_1"].astype(str).map(get_display_label)
            redundancy_display["feature_2"] = redundancy_display["feature_2"].astype(str).map(get_display_label)
            redundancy_display = redundancy_display.rename(
                columns={
                    "feature_1": "Feature 1",
                    "feature_2": "Feature 2",
                    "correlation": "Correlation",
                    "abs_correlation": "Absolute Correlation",
                    "risk_band": "Risk Band",
                }
            )
            st.dataframe(redundancy_display, width="stretch", hide_index=True)

    if controls["show_source_table"]:
        st.divider()
        st.markdown("### Source Track Table")
        st.caption("These are the currently visible source tracks behind the page.")
        source_cols = select_unique_existing_columns(
            filtered_df,
            get_safe_source_display_columns(filtered_df),
        )
        display_df = rename_and_dedupe_for_display(filtered_df[source_cols].copy())
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.caption(
        "These are descriptive pairwise associations. Strong relationships can suggest "
        "shared structure, candidate archetypes, or redundancy risk, but they do not by themselves establish causality."
    )


if __name__ == "__main__":
    main()