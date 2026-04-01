"""
Analysis functions for soundtrack feature correlation exploration.

This module contains reusable helpers for correlation-based analysis of
the analysis-ready soundtrack album dataframe. It includes functions for
computing full correlation matrices, preparing feature-level target
correlations, building Altair visualizations, and running lightweight
robustness checks on the resulting correlation story.

The module currently supports three main kinds of analysis output:

- correlation matrices for numeric and boolean features
- visualization-ready datasets and charts, including a heatmap and a
  target-focused lollipop chart
- robustness diagnostics that compare Pearson and Spearman results and
  assess sensitivity to trimming extreme target values

Data preparation logic is kept separate from plotting and reporting
logic so the functions remain easier to test, reuse, and extend.
"""


import pandas as pd
import altair as alt

EXCLUDED_CORRELATION_COLS = {
    "tmdb_id",
    "release_group_mbid",
}

def compute_correlation_matrix(
        album_analytics_df: pd.DataFrame,
        method: str = "spearman"
) -> pd.DataFrame:
    """
    Compute a correlation matrix for numeric and boolean analysis columns.

    This selects analysis-ready numeric fields from the album analytics
    dataframe and calculates their pairwise correlation values using the
    requested method.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        method: Correlation method to use, such as "spearman" or "pearson".

    Returns:
        pd.DataFrame: Correlation matrix for numeric and boolean columns.
    """
    corr_df = album_analytics_df.select_dtypes(include=["number", "bool"])
    # print(corr_df.shape, album_analytics_df.shape)  # Verify that we don't lose columns

    corr_method_df = corr_df.corr(method=method)
    # print(type(corr_method_df))
    return corr_method_df

def corr_to_long(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a square correlation matrix into long format for Altair.

    This reshapes the matrix so each row represents one variable pair
    and its corresponding correlation value.

    Args:
        corr_df: Square correlation matrix.

    Returns:
        pd.DataFrame: Long-form correlation dataframe.
    """
    return (
        corr_df
        .reset_index()
        .melt(id_vars="index", var_name="var_y", value_name="corr")
        .rename(columns={"index": "var_x"})
    )

def plot_correlation_heatmap(
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Heatmap (Spearman)",
) -> alt.Chart:
    """
    Build an Altair heatmap from a correlation matrix.

    This converts the square correlation matrix into long format and
    maps each variable pair to a colored cell with tooltip support.

    Args:
        corr_matrix: Square correlation matrix to visualize.
        title: Title displayed above the heatmap.

    Returns:
        alt.Chart: Altair heatmap chart.
    """

    corr_long = corr_to_long(corr_matrix)  # or corr_pearson

    heatmap = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X(
                "var_x:O",
                title=None,
                sort=None,
                axis=alt.Axis(labelAngle=-45, labelFontSize=13, titleFontSize=13)
            ),
            y=alt.Y(
                "var_y:O",
                title=None,
                sort=None,
                axis=alt.Axis(labelFontSize=13, titleFontSize=13)
            ),
            color=alt.Color(
                "corr:Q",
                scale=alt.Scale(
                    range=["#CC0000", "#F3F0E6", "#1195B2"],
                    domain=[-1, 1],
                    domainMid=0,
                    clamp=True
                ),
                title="Correlation"
            ),
            tooltip=[
                alt.Tooltip("var_x:N", title="X"),
                alt.Tooltip("var_y:N", title="Y"),
                alt.Tooltip("corr:Q", format=".2f", title="ρ"),
            ],
        )
        .properties(
            width=700,
            height=700,
            title=title,
        )
    )

    return heatmap



def prepare_lollipop_data(
        album_analytics_df: pd.DataFrame,
        target_col: str = "log_lfm_album_listeners",
        method: str = "pearson"
) -> pd.DataFrame:
    """Prepare feature-correlation data for the lollipop chart.

    This computes correlations between the target column and all other
    numeric analysis columns, removes the target self-correlation, and
    adds helper fields used for plotting and sorting.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        target_col: Target variable whose feature correlations will be
            visualized.
        method: Correlation method to use for the lollipop analysis.

    Returns:
        pd.DataFrame: Plot-ready dataframe containing feature names,
        correlation values, absolute correlation values, and a zero
        baseline column for lollipop chart rendering.
    """
    corr_df = album_analytics_df.select_dtypes(include=["number", "bool"]).copy()

    # Exclude obvious ID / key fields from correlation ranking.
    keep_cols = [
        col for col in corr_df.columns
        if col not in EXCLUDED_CORRELATION_COLS
    ]
    corr_df = corr_df[keep_cols]

    target = target_col

    corr_sorted = (
        corr_df
        .corr(method=method)[target]
        .drop(target)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    corr_df_plot = (
        corr_sorted
        .rename("corr")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    corr_df_plot["abs_corr"] = corr_df_plot["corr"].abs()
    corr_df_plot["zero"] = 0

    return corr_df_plot

def plot_lollipop_chart(
        corr_df_plot: pd.DataFrame,
) -> alt.Chart:
    """
    Build a lollipop chart from prepared feature-correlation data.

    This plots each feature's correlation with the target variable as a
    horizontal lollipop, using color to distinguish positive and
    negative relationships and tooltips to show exact values.

    Args:
        corr_df_plot: Plot-ready dataframe of feature correlations.

    Returns:
        alt.Chart: Altair lollipop chart.
    """
    x_domain = [
        min(0.0, float(corr_df_plot["corr"].min()) - 0.02),
        max(0.0, float(corr_df_plot["corr"].max()) + 0.02),
    ]

    x_axis = alt.X(
        "corr:Q",
        title="Correlation with log(album listeners)",
        scale=alt.Scale(domain=x_domain)
    )

    y_axis = alt.Y(
        "feature:N",
        sort=alt.SortField("abs_corr", order="descending"),
        title=None
    )

    sticks = alt.Chart(corr_df_plot).mark_rule(strokeWidth=2).encode(
        y=y_axis,
        x=alt.X(
            "zero:Q",
            scale=alt.Scale(domain=x_domain),
            title="Correlation with log(album listeners)"
        ),
        x2="corr:Q",
        color=alt.condition(
            "datum.corr >= 0",
            alt.value("#1195B2"),
            alt.value("#CC0000")
        )
    )

    dots = alt.Chart(corr_df_plot).mark_circle(size=80).encode(
        y=y_axis,
        x=x_axis,
        color=alt.condition(
            "datum.corr >= 0",
            alt.value("#1195B2"),
            alt.value("#CC0000")
        ),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("corr:Q", title="Correlation", format=".3f")
        ]
    )

    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        strokeDash=[4, 4], opacity=0.6
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_domain))
    )

    chart = (sticks + dots + zero_line).properties(
        width=650,
        height=900,
        title={
            "text": "Lollipop Plot: Which features scale with album popularity?",
            "subtitle": """Pearson correlations with log(album listeners); 
            no single feature dominates"""
        }
    )

    return chart

def compute_correlation_method_agreement(
        pearson: pd.Series,
        spearman: pd.Series,
        top_n: int
) -> tuple[int, list[str], float]:
    """Compute top-N overlap and rank agreement across methods.

    Args:
        pearson: Pearson correlations between each feature and the target.
        spearman: Spearman correlations between each feature and the target.
        top_n: Requested number of top features to compare.

    Returns:
        tuple[int, list[str], float]: The effective top-N value used,
        the sorted overlapping feature names, and the rank agreement
        statistic between Pearson and Spearman absolute-correlation
        rankings.
    """
    top_n_used = min(top_n, len(pearson))

    top_pearson = set(pearson.abs().nlargest(top_n_used).index)
    top_spearman = set(spearman.abs().nlargest(top_n_used).index)
    overlap = sorted(top_pearson & top_spearman)

    rank_agreement = (
        pearson.abs().rank(ascending=False)
        .corr(spearman.abs().rank(ascending=False), method="spearman")
    )

    return top_n_used, overlap, rank_agreement

def build_correlation_method_comparison_tables(
        pearson: pd.Series,
        spearman: pd.Series,
        top_n_used: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build summary tables for Pearson versus Spearman comparison.

    Args:
        pearson: Pearson correlations between each feature and the target.
        spearman: Spearman correlations between each feature and the target.
        top_n_used: Number of top features to retain in the compact report
            table.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Full summary dataframe and
        top-features dataframe.
    """
    summary_df = pd.DataFrame({
        "pearson": pearson,
        "spearman": spearman,
    })
    summary_df["abs_pearson"] = summary_df["pearson"].abs()
    summary_df["abs_spearman"] = summary_df["spearman"].abs()
    summary_df["pearson_rank"] = summary_df["abs_pearson"].rank(
        ascending=False,
        method="average",
    )
    summary_df["spearman_rank"] = summary_df["abs_spearman"].rank(
        ascending=False,
        method="average",
    )

    top_features_df = (
        summary_df.sort_values("abs_pearson", ascending=False)
        .head(top_n_used)[
            [
                "pearson",
                "spearman",
                "pearson_rank",
                "spearman_rank",
            ]
        ]
        .round(3)
    )

    return summary_df, top_features_df

def compare_target_correlations_by_method(
        analysis_df:pd.DataFrame,
        target_col: str = "log_lfm_album_listeners",
        top_n: int = 15
) -> dict:
    """
    Compare target-feature correlations under Pearson and Spearman methods.

    This robustness check evaluates whether the same features emerge as the
    strongest correlates of the target variable under two different
    correlation methods:

    - Pearson: sensitive to linear magnitude relationships
    - Spearman: rank-based and more robust to heavy tails or monotonic
      but non-linear relationships

    The function returns both headline metrics and tabular outputs that can
    be logged, tested, or rendered into an HTML report.

    Args:
        analysis_df: Analysis-ready dataframe containing numeric feature
            columns and the target column.
        target_col: Name of the target column whose feature correlations
            should be compared across methods.
        top_n: Number of highest-magnitude target correlations to compare
            across Pearson and Spearman rankings.

    Returns:
        dict: Dictionary containing:
            - "metrics": headline robustness statistics
            - "summary_df": full feature-level comparison table
            - "top_features_df": report-ready top-N comparison table

    Raises:
        ValueError: If the target column is missing, non-numeric, no usable
            feature columns remain, or no complete rows remain after dropping
            missing values.
    """
    if target_col not in analysis_df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in the analysis "
            f"dataframe."
        )

    numeric_df = analysis_df.select_dtypes(include=["number", "bool"]).copy()

    if target_col not in numeric_df.columns:
        raise ValueError(
            f"Target column '{target_col}' is not numeric and cannot be used "
            f"for correlation analysis."
        )

    feature_cols = [col for col in numeric_df.columns if col != target_col]
    if not feature_cols:
        raise ValueError(
            "At least one numeric feature column besides the target is "
            "required for correlation comparison."
        )

    rows_before_dropna = len(numeric_df)
    complete_df = numeric_df.dropna()
    rows_used = len(complete_df)

    if rows_used == 0:
        raise ValueError(
            "No complete rows remain after dropping missing values. "
            "Cannot compare target correlations by method."
        )

    # Correlations with the target (one number per feature)
    pearson = complete_df.corr(method="pearson")[target_col].drop(target_col)
    spearman = complete_df.corr(method="spearman")[target_col].drop(target_col)

    top_n_used, overlap, rank_agreement = compute_correlation_method_agreement(
        pearson=pearson,
        spearman=spearman,
        top_n=top_n,
    )

    # Display: one compact summary table + two headline stats
    summary_df, top_features_df = build_correlation_method_comparison_tables(
        pearson=pearson,
        spearman=spearman,
        top_n_used=top_n_used,
    )

    # print(f"Top-{TOP_N} overlap: {len(overlap)} / {TOP_N}")
    # print(f"Rank agreement (ρ on ranks of |corr|): {rank_agreement:.3f}")
    # print("Overlapping features:", overlap)
    #
    # display(
    #     summary.sort_values("abs_pearson", ascending=False)
    #     .head(TOP_N)[["pearson", "spearman"]]
    #     .round(3)
    # )

    return {
        "metrics": {
            "target_col": target_col,
            "top_n_requested": top_n,
            "top_n_used": top_n_used,
            "rows_before_dropna": rows_before_dropna,
            "rows_used": rows_used,
            "top_n_overlap_count": len(overlap),
            "top_n_overlap_features": overlap,
            "rank_agreement": round(rank_agreement, 3),
        },
        "summary_df": summary_df,
        "top_features_df": top_features_df,
    }



def assess_trimmed_correlation_stability(
        analysis_df: pd.DataFrame,
        target_col: str = "log_lfm_album_listeners",
        trim_q: float = 0.99,
) -> dict:
    """Assess how stable target-feature correlations remain after trimming
    extreme target values.

    This robustness check compares the full correlation vector against a
    trimmed version in which the highest target values are removed. It is
    intended to show whether Pearson and Spearman target correlations are
    broadly stable or heavily influenced by extreme observations.

    Args:
        analysis_df: Analysis-ready dataframe containing numeric feature
            columns and the target column.
        target_col: Name of the target column used to define trimming.
        trim_q: Quantile threshold used to trim the upper tail of the target
            distribution. For example, 0.99 removes the top 1 percent.

    Returns:
        dict: Dictionary containing:
            - "metrics": headline trimming and stability statistics
            - "summary_df": feature-level comparison table for full vs trimmed
              correlations

    Raises:
        ValueError: If the target column is missing, non-numeric, or no
            complete rows remain after dropping missing values.
    """
    if target_col not in analysis_df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in the analysis "
            f"dataframe."
        )

    numeric_df = analysis_df.select_dtypes(include=["number", "bool"]).copy()

    if target_col not in numeric_df.columns:
        raise ValueError(
            f"Target column '{target_col}' is not numeric and cannot be used "
            f"for correlation analysis."
        )

    complete_df = numeric_df.dropna()

    if complete_df.empty:
        raise ValueError(
            "No complete rows remain after dropping missing values. "
            "Cannot assess trimmed correlation stability."
        )

    threshold = complete_df[target_col].quantile(trim_q)
    trimmed_df = complete_df[complete_df[target_col] <= threshold]

    pearson_full = complete_df.corr(method="pearson")[target_col].drop(target_col)
    spearman_full = complete_df.corr(method="spearman")[target_col].drop(target_col)

    pearson_trim = trimmed_df.corr(method="pearson")[target_col].drop(target_col)
    spearman_trim = trimmed_df.corr(method="spearman")[target_col].drop(target_col)

    pearson_stability = pearson_full.corr(pearson_trim, method="pearson")
    spearman_stability = spearman_full.corr(spearman_trim, method="pearson")

    summary_df = pd.DataFrame({
        "pearson_full": pearson_full,
        "pearson_trim": pearson_trim,
        "spearman_full": spearman_full,
        "spearman_trim": spearman_trim,
    })
    summary_df["pearson_abs_diff"] = (
        summary_df["pearson_full"] - summary_df["pearson_trim"]
    ).abs()
    summary_df["spearman_abs_diff"] = (
        summary_df["spearman_full"] - summary_df["spearman_trim"]
    ).abs()

    # print("\nTest 2 — Trim extremes")
    # print(f"Trim threshold (top {(1 - TRIM_Q) * 100:.1f}% removed): {thr:.3f}")
    # print(f"Rows kept: {len(df_trim):,} / {len(df):,}")
    # print(f"Pearson stability (full vs trimmed):  {pearson_stability:.3f}")
    # print(f"Spearman stability (full vs trimmed): {spearman_stability:.3f}")

    return {
        "metrics": {
            "target_col": target_col,
            "trim_q": trim_q,
            "trim_percent_removed": round((1 - trim_q) * 100, 1),
            "trim_threshold": round(threshold, 3),
            "rows_before_trim": len(complete_df),
            "rows_after_trim": len(trimmed_df),
            "pearson_stability": round(pearson_stability, 3),
            "spearman_stability": round(spearman_stability, 3),
        },
        "summary_df": summary_df.round(3),
    }
