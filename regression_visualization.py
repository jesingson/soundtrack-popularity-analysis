"""Visualization functions for post-regression soundtrack analysis.

This module contains chart-building functions for the regression output
section of the soundtrack analysis pipeline. These functions accept
already-prepared dataframes and return Altair charts for embedding into
the regression HTML report.
"""

import altair as alt
import pandas as pd


def create_vote_count_scatter_chart(
        plot_df: pd.DataFrame,
        line_df: pd.DataFrame
) -> alt.Chart:
    """Create the film vote count versus listener scatterplot chart.

    This function renders the post-regression visualization showing the
    relationship between film vote count and log soundtrack listeners.
    The point and fitted-line data are prepared upstream so this function
    only handles chart construction.

    Args:
        plot_df: Dataframe containing scatterplot points with
            ``film_vote_count`` and ``log_lfm_album_listeners`` columns.
        line_df: Dataframe containing two endpoints for the fitted line,
            using the same columns as ``plot_df``.

    Returns:
        alt.Chart: Layered Altair chart containing the scatterplot points
        and the fitted line.
    """
    scatter = alt.Chart(plot_df).mark_circle(opacity=0.25, size=40).encode(
        x=alt.X("film_vote_count:Q", title="Film vote count (exposure proxy)"),
        y=alt.Y("log_lfm_album_listeners:Q", title="Log soundtrack listeners")
    )

    line = alt.Chart(line_df).mark_line(strokeWidth=3).encode(
        x="film_vote_count:Q",
        y="log_lfm_album_listeners:Q"
    )

    chart = (scatter + line).properties(
        width=650,
        height=400,
        title="Film exposure vs soundtrack popularity (with fitted line)"
    )

    return chart


def create_coefficient_whisker_chart(
        coef_df: pd.DataFrame
) -> alt.Chart:
    """Create a coefficient dot-and-whisker plot with 95% confidence bands.

    This function renders the post-regression coefficient visualization
    from a tidy coefficient dataframe prepared upstream. It shows each
    coefficient estimate, its confidence interval, and whether that
    interval crosses zero.

    Args:
        coef_df: Tidy coefficient dataframe containing at least
            ``feature``, ``coef``, ``ci_low``, ``ci_high``, and
            ``ci_group`` columns.

    Returns:
        alt.Chart: Layered Altair chart with coefficient points,
        confidence interval whiskers, and a vertical zero reference line.
    """
    # Lock the y-order to the sorted coefficient order from the dataframe
    y_order = coef_df["feature"].tolist()

    # Dot-and-whisker plot layers
    whiskers = alt.Chart(coef_df).mark_rule().encode(
        x="ci_low:Q",
        x2="ci_high:Q",
        y=alt.Y("feature:N", sort=y_order, title=None),
        color=alt.Color("ci_group:N", title=None)  # uses theme category palette
    )

    dots = alt.Chart(coef_df).mark_circle(size=80).encode(
        x=alt.X("coef:Q", title="Effect on log soundtrack listeners"),
        y=alt.Y("feature:N", sort=y_order, title=None),
        color=alt.Color("ci_group:N", title=None, legend=None),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("coef:Q", format=".3f", title="Coefficient"),
            alt.Tooltip("ci_low:Q", format=".3f", title="CI low"),
            alt.Tooltip("ci_high:Q", format=".3f", title="CI high"),
            alt.Tooltip("ci_group:N", title="")
        ]
    )

    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeDash=[4, 4]
    ).encode(
        x="x:Q"
    )

    chart = (whiskers + dots + zero).properties(
        width=750,
        height=700,
        title={
            "text": "Regression coefficients with 95% confidence intervals",
                        "subtitle": [
                "Dots are coefficient estimates; whiskers are 95% "
                "confidence intervals.",
                "Color indicates whether the confidence interval crosses 0."
            ]
        }
    )

    return chart
