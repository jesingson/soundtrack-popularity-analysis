from __future__ import annotations

import streamlit as st

from app.ui import apply_app_styles
from app.app_data import load_track_data_explorer_data
from app.app_controls import (
    get_global_filter_controls,
    get_track_ridge_controls,
)
from app.explorer_shared import (
    get_global_filter_inputs,
    get_track_page_display_label,
    rename_track_page_columns_for_display,
)
from app.data_filters import filter_dataset

from track_ridge_analysis import (
    TRACK_DEFAULT_RIDGE_GROUPS,
    TRACK_RIDGE_TARGET_OPTIONS,
    build_track_ridge_prep_outputs,
    build_track_ridge_phase2_outputs,
)
from track_ridge_visualization import create_track_ridge_chart



PRESET_LABELS = {
    "core_audio": "Core Audio",
    "position_and_sequence": "Position and Sequence",
    "track_type_and_mood": "Track Type and Mood",
    "archetypes": "Archetypes",
    "film_and_album_context": "Film and Album Context",
    "context_genres_and_awards": "Context Genres and Awards",
    "custom": "Custom Selection",
}


def build_track_ridge_scope_caption(
    target_col: str,
    preset_key: str,
    selected_features: list[str],
    n_ridges: int,
    bins: int,
    smooth_window: int,
    min_group_n: int,
) -> str:
    """Build a short caption describing the current ridge scope."""
    preset_label = PRESET_LABELS.get(preset_key, preset_key)

    mode_phrase = (
        "custom feature selection"
        if preset_key == "custom"
        else f"preset '{preset_label}'"
    )

    return (
        f"Current scope: target = {get_track_page_display_label(target_col)}, {mode_phrase}, "
        f"showing {n_ridges} binary feature splits with {bins} density bins, "
        f"smoothing window {smooth_window}, and minimum group size {min_group_n}."
    )


def build_selected_feature_caption(
    preset_key: str,
    selected_features: list[str],
) -> str:
    """Build a compact caption listing the selected raw feature columns."""
    if not selected_features:
        return "No selected features."

    prefix = "Selected custom features" if preset_key == "custom" else "Selected preset features"
    display_features = [
        get_track_page_display_label(feature)
        for feature in selected_features
    ]
    return f"{prefix}: {', '.join(display_features)}."


def build_track_ridge_insight_summary(
    order_df,
    target_col: str,
) -> dict[str, str]:
    """Build a compact narrative summary from the track ridge ordering dataframe."""
    if order_df.empty:
        return {
            "card1_title": "Strongest Separator",
            "card1_value": "None",
            "card1_caption": "No ridge features are available under the current settings.",
            "card2_title": "Direction of Shift",
            "card2_value": "None",
            "card2_caption": "No directional split is available.",
            "card3_title": "Overall Separation",
            "card3_value": "None",
            "card3_caption": "No overall ridge pattern is available.",
        }

    top_feature_label = str(order_df.index[0])
    top_row = order_df.iloc[0]

    top_gap = float(top_row["median_gap"])
    top_no = float(top_row["No"])
    top_yes = float(top_row["Yes"])

    if top_yes > top_no:
        direction_value = "Yes above No"
        direction_caption = (
            f"For the strongest visible split, tracks in the Yes group have a higher "
                            f"median {get_track_page_display_label(target_col).lower()} ({top_yes:.3f}) than the "
            f"No group ({top_no:.3f})."
        )
    elif top_yes < top_no:
        direction_value = "No above Yes"
        direction_caption = (
            f"For the strongest visible split, tracks in the No group have a higher "
                            f"median {get_track_page_display_label(target_col).lower()} ({top_no:.3f}) than the "
            f"Yes group ({top_yes:.3f})."
        )
    else:
        direction_value = "Tie"
        direction_caption = (
            f"For the strongest visible split, the Yes and No groups have equal "
                        f"median {get_track_page_display_label(target_col).lower()} at {top_yes:.3f}."
        )

    mean_gap = float(order_df["median_gap"].mean())
    median_gap = float(order_df["median_gap"].median())

    return {
        "card1_title": "Strongest Separator",
        "card1_value": get_track_page_display_label(top_feature_label),
        "card1_caption": (
                        f"Top-ranked split by median {get_track_page_display_label(target_col).lower()} gap "
            f"({top_gap:.3f})."
        ),
        "card2_title": "Direction of Shift",
        "card2_value": direction_value,
        "card2_caption": direction_caption,
        "card3_title": "Overall Separation",
        "card3_value": f"Avg gap {mean_gap:.3f}",
        "card3_caption": (
            f"Across visible features, the median gap is {median_gap:.3f}, so separation "
            f"is concentrated more strongly in the top rows than across the full set."
        ),
    }


def render_track_ridge_insight_cards(
    order_df,
    target_col: str,
) -> None:
    """Render the three narrative insight cards for the track ridge page."""
    insights = build_track_ridge_insight_summary(
        order_df=order_df,
        target_col=target_col,
    )

    st.markdown("### 🧠 Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(insights["card1_title"], insights["card1_value"])
        st.caption(insights["card1_caption"])

    with col2:
        st.metric(insights["card2_title"], insights["card2_value"])
        st.caption(insights["card2_caption"])

    with col3:
        st.metric(insights["card3_title"], insights["card3_value"])
        st.caption(insights["card3_caption"])


def build_track_ridge_supporting_insight(order_df, target_col: str) -> str:
    """Build a short supporting sentence for the main track ridge chart."""
    if order_df.empty:
        return "No ridge ordering insight is available."

    top_feature_label = str(order_df.index[0])
    top_gap = float(order_df.iloc[0]["median_gap"])

    bottom_feature_label = str(order_df.index[-1])
    bottom_gap = float(order_df.iloc[-1]["median_gap"])

    top_signed_gap = float(order_df.iloc[0]["signed_gap"])

    if top_signed_gap > 0:
        direction_phrase = (
            "the condition-met group is shifted upward relative to the condition-not-met group"
        )
    elif top_signed_gap < 0:
        direction_phrase = (
            "the condition-not-met group is shifted upward relative to the condition-met group"
        )
    else:
        direction_phrase = "the two groups have essentially no directional separation"

    return (
        f"💡 The clearest visible distribution split for {get_track_page_display_label(target_col).lower()} "
                f"is '{get_track_page_display_label(top_feature_label)}' (median gap = {top_gap:.3f}), where {direction_phrase}. "
                f"By contrast, '{get_track_page_display_label(bottom_feature_label)}' shows much weaker visible separation "
        f"({bottom_gap:.3f}), indicating that some track conditions are far more informative "
        f"than others under the current target and settings."
    )

def build_track_ridge_modeling_takeaway(
    order_df,
    target_col: str,
) -> str:
    """Build a short modeling takeaway tied to the visible ridge ordering."""
    if order_df.empty:
        return (
            f"No visible ridge separation is available to assess whether "
                        f"{get_track_page_display_label(target_col).lower()} varies across track conditions."
        )

    top_gap = float(order_df.iloc[0]["median_gap"])
    median_gap = float(order_df["median_gap"].median())

    if top_gap >= 0.30 and median_gap >= 0.15:
        strength_phrase = (
            "multiple visible track conditions produce meaningful distribution shifts"
        )
    elif top_gap >= 0.20:
        strength_phrase = (
            "a small number of visible track conditions produce noticeable distribution shifts"
        )
    else:
        strength_phrase = (
            "most visible track conditions produce only modest distribution shifts"
        )

    return (
                f"While no single binary split fully explains {get_track_page_display_label(target_col).lower()}, "
        f"{strength_phrase}. That pattern suggests the target is shaped by several overlapping "
        f"track-level signals rather than one dominant driver, which supports moving from "
        f"univariate ridge comparisons to multivariate regression."
    )

def assess_track_ridge_quality(
    order_df,
    ridge_density_df,
    min_total_tracks: int = 30,
    min_visible_ridges: int = 2,
) -> dict[str, object]:
    """
    Assess whether the current ridge view has enough data to be interpretable.

    Returns:
        dict with:
            - quality_flag: "Good", "Weak", or "Insufficient"
            - total_tracks_used: approximate number of tracks contributing
            - visible_ridges: number of rendered ridge rows
            - message: optional user-facing warning/error text
            - should_stop: whether the page should stop before rendering chart
    """
    if ridge_density_df is None or ridge_density_df.empty:
        return {
            "quality_flag": "Insufficient",
            "total_tracks_used": 0,
            "visible_ridges": 0,
            "message": (
                "⚠️ The current filters do not leave enough usable data to render "
                "a meaningful ridge comparison. Try broadening the filters or "
                "lowering the minimum group size."
            ),
            "should_stop": True,
        }

    visible_ridges = int(ridge_density_df["feature"].nunique())
    if "n_obs" in ridge_density_df.columns:
        total_tracks_used = int(ridge_density_df["n_obs"].max())
    else:
        total_tracks_used = int(ridge_density_df.shape[0])

    if total_tracks_used < min_total_tracks or visible_ridges < min_visible_ridges:
        return {
            "quality_flag": "Insufficient",
            "total_tracks_used": total_tracks_used,
            "visible_ridges": visible_ridges,
            "message": (
                f"⚠️ The current selection produces only {total_tracks_used} usable tracks "
                f"and {visible_ridges} visible ridge split(s), which is too little for a "
                "stable comparison. Try broadening your filters or lowering the minimum "
                "group size."
            ),
            "should_stop": True,
        }

    if visible_ridges < 4 or total_tracks_used < 75:
        return {
            "quality_flag": "Weak",
            "total_tracks_used": total_tracks_used,
            "visible_ridges": visible_ridges,
            "message": (
                f"⚠️ Ridge comparisons may be fragile under the current settings "
                f"({total_tracks_used} usable tracks, {visible_ridges} visible splits). "
                "Interpret the ranking cautiously."
            ),
            "should_stop": False,
        }

    return {
        "quality_flag": "Good",
        "total_tracks_used": total_tracks_used,
        "visible_ridges": visible_ridges,
        "message": None,
        "should_stop": False,
    }

def main() -> None:
    st.set_page_config(
        page_title="Track Ridge Explorer",
        layout="wide",
    )
    apply_app_styles()

    st.title("Track Ridge Explorer")
    st.write(
        """
        Compare how track-success distributions shift across binary feature splits.
        The page remains track-first, but it can also surface attached film and album
        context when those features are selected. This helps distinguish strong
        univariate separation from weaker or overlapping patterns before moving into
        multivariate regression.
        """
    )

    track_df = load_track_data_explorer_data()

    filter_inputs = get_global_filter_inputs(track_df)

    global_controls = get_global_filter_controls(
        min_year=filter_inputs["min_year"],
        max_year=filter_inputs["max_year"],
        film_genre_options=filter_inputs["film_genre_options"],
        album_genre_options=filter_inputs["album_genre_options"],
    )

    filtered_track_df = filter_dataset(track_df, global_controls).copy()

    if filtered_track_df.empty:
        st.warning("No tracks remain under the current global filters.")
        st.stop()

    all_available_features = sorted(
        {
            feature
            for cols in TRACK_DEFAULT_RIDGE_GROUPS.values()
            for feature in cols
            if feature in filtered_track_df.columns
        }
    )

    ridge_controls = get_track_ridge_controls(
        target_options=TRACK_RIDGE_TARGET_OPTIONS,
        preset_labels=PRESET_LABELS,
        preset_keys=list(PRESET_LABELS.keys()),
        all_available_features=all_available_features,
        default_custom_features=TRACK_DEFAULT_RIDGE_GROUPS["core_audio"],
    )

    target_col = ridge_controls["target_col"]
    preset_key = ridge_controls["preset_key"]
    bins = ridge_controls["bins"]
    smooth_window = ridge_controls["smooth_window"]
    min_group_n = ridge_controls["min_group_n"]
    show_order_table = ridge_controls["show_order_table"]
    show_density_table = ridge_controls["show_density_table"]
    show_ridge_long_sample = ridge_controls["show_ridge_long_sample"]

    if preset_key == "custom":
        selected_features = ridge_controls["selected_features"]
    else:
        selected_features = [
            feature
            for feature in TRACK_DEFAULT_RIDGE_GROUPS[preset_key]
            if feature in filtered_track_df.columns
        ]

    if not selected_features:
        st.warning("Select at least one feature to render the ridge plot.")
        st.stop()

    ridge_outputs = build_track_ridge_prep_outputs(
        track_df=filtered_track_df,
        y_col=target_col,
        feature_groups={"selected": selected_features},
    )

    try:
        phase2_outputs = build_track_ridge_phase2_outputs(
            ridge_outputs=ridge_outputs,
            bins=bins,
            smooth_window=smooth_window,
            min_group_n=min_group_n,
            order_method="median_gap",
            row_gap=2.5,
            density_scale=11.0,
            label_y_offset=0.0,
        )
    except ValueError:
        st.error(
            "⚠️ No feature groups meet the minimum group size under the current filters.\n\n"
            "👉 Try lowering the minimum group size or broadening your filters."
        )
        st.stop()

    order_df = phase2_outputs["order_df"]
    n_ridges = len(phase2_outputs["labels_df"])
    chart_height = max(500, n_ridges * 32)

    ridge_quality = assess_track_ridge_quality(
        order_df=order_df,
        ridge_density_df=phase2_outputs["ridge_density_df"],
    )

    if ridge_quality["should_stop"]:
        st.error(ridge_quality["message"])
        st.stop()

    if ridge_quality["message"] is not None:
        st.warning(ridge_quality["message"])

    subtitle_lines = [
                f"Target: {get_track_page_display_label(target_col)} | Visible feature splits: {n_ridges}",
        "Rows are ordered by absolute median gap between condition-met and condition-not-met groups.",
    ]

    chart = create_track_ridge_chart(
        ridge_chart_df=phase2_outputs["ridge_chart_df"],
        labels_df=phase2_outputs["labels_df"],
        y_col=target_col,
        title_text=f"{get_track_page_display_label(target_col)} distributions by feature group",
        subtitle_lines=subtitle_lines,
        width=780,
        height=chart_height,
        left_padding=320,
    )

    st.caption(
        build_track_ridge_scope_caption(
            target_col=target_col,
            preset_key=preset_key,
            selected_features=selected_features,
            n_ridges=n_ridges,
            bins=bins,
            smooth_window=smooth_window,
            min_group_n=min_group_n,
        )
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Usable tracks", f"{ridge_quality['total_tracks_used']:,}")
    with col2:
        st.metric("Visible ridge splits", f"{ridge_quality['visible_ridges']:,}")
    with col3:
        quality_label = ridge_quality["quality_flag"]
        if quality_label == "Good":
            display_quality = "✅ Good"
        elif quality_label == "Weak":
            display_quality = "⚠️ Weak"
        else:
            display_quality = "❌ Insufficient"

        st.metric("Ridge quality", display_quality)

    render_track_ridge_insight_cards(
        order_df=order_df,
        target_col=target_col,
    )

    st.caption(
        build_selected_feature_caption(
            preset_key=preset_key,
            selected_features=selected_features,
        )
    )

    st.altair_chart(chart, width="stretch")

    st.caption(
        build_track_ridge_supporting_insight(
            order_df=order_df,
            target_col=target_col,
        )
    )

    st.markdown("### 📈 Modeling Takeaway")
    st.caption(
        build_track_ridge_modeling_takeaway(
            order_df=order_df,
            target_col=target_col,
        )
    )

    if show_order_table:
        st.subheader("Feature Ordering Table")
        st.dataframe(
            rename_track_page_columns_for_display(phase2_outputs["order_df"].round(3)),
            width="stretch",
        )

    if show_density_table:
        st.subheader("Density Dataframe Sample")
        st.dataframe(
            rename_track_page_columns_for_display(
                phase2_outputs["ridge_density_df"].head(50)
            ),
            width="stretch",
        )

    if show_ridge_long_sample:
        st.subheader("ridge_long Sample")
        st.dataframe(
            rename_track_page_columns_for_display(ridge_outputs["ridge_long"].head(50)),
            width="stretch",
        )


if __name__ == "__main__":
    main()