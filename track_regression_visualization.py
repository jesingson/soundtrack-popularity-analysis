from __future__ import annotations

import altair as alt
import pandas as pd

from app.ui import get_display_label


def create_track_coefficient_whisker_chart(
    coef_df: pd.DataFrame,
    target_col: str,
) -> alt.Chart:
    """
    Create a coefficient dot-and-whisker plot with 95% confidence bands
    for the selected track target.
    """
    y_order = coef_df["feature"].tolist()

    whiskers = alt.Chart(coef_df).mark_rule().encode(
        x="ci_low:Q",
        x2="ci_high:Q",
        y=alt.Y("feature:N", sort=y_order, title=None),
        color=alt.Color("ci_group:N", title=None),
    )

    dots = alt.Chart(coef_df).mark_circle(size=80).encode(
        x=alt.X("coef:Q", title=f"Effect on {get_display_label(target_col).lower()}"),
        y=alt.Y("feature:N", sort=y_order, title=None),
        color=alt.Color("ci_group:N", title=None, legend=None),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("coef:Q", format=".3f", title="Coefficient"),
            alt.Tooltip("ci_low:Q", format=".3f", title="CI low"),
            alt.Tooltip("ci_high:Q", format=".3f", title="CI high"),
            alt.Tooltip("ci_group:N", title=""),
        ],
    )

    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeDash=[4, 4]
    ).encode(
        x="x:Q"
    )

    chart_height = max(700, len(coef_df) * 36)

    chart = (whiskers + dots + zero).properties(
        width=750,
        height=chart_height,
        title={
            "text": "Track regression coefficients with 95% confidence intervals",
            "subtitle": [
                "Dots are coefficient estimates; whiskers are 95% confidence intervals.",
                "Color indicates whether the confidence interval crosses 0.",
            ],
        },
    )

    return chart