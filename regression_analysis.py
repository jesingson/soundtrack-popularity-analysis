"""Regression workflow functions for soundtrack popularity analysis.

This module contains reusable helpers for preparing regression features,
lightly filtering continuous predictors, applying modeling transforms,
finalizing predictors for ordinary least squares estimation, and fitting
the final OLS model. It also includes small post-regression data-prep
helpers used to support HTML reporting and visualization.

The regression workflow is intentionally staged so that each step returns
structured artifacts for debugging, reporting, and downstream charting.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from data_processing import (
    FILM_FEATURES,
    ALBUM_FEATURES,
    DERIVED_AWARD_COLS,
    TARGET_COL,
)

def define_regression_features(
        film_features: list[str],
        album_features: list[str],
        derived_award_cols: list[str],
        target_col: str = "log_lfm_album_listeners",
        threshold: float = 0.05,
) -> dict:
    """
    Define grouped predictor lists for the regression workflow.

    This function organizes candidate regression features into continuous
    and binary groups before any filtering or transformation is applied.
    It also returns the target column name and the light-winnowing
    threshold used later in the regression pipeline.

    Args:
        film_features: List of film-related feature column names available
            in the analysis dataframe.
        album_features: List of album-related feature column names available
            in the analysis dataframe.
        derived_award_cols: List of derived award indicator column names.
        target_col: Name of the regression target column.
        threshold: Minimum absolute Pearson correlation required to keep a
            continuous predictor during the later light-winnowing step.

    Returns:
        dict: Dictionary containing regression feature groups and config
        values used later in the modeling pipeline.
    """
    film_continuous = [
        "film_vote_count",
        "film_popularity",
        "film_budget",
        "film_revenue",
        "film_rating",
        "days_since_film_release",
    ]

    film_binary = [c for c in film_features if c.startswith("film_is_")]

    album_continuous = [
        "days_since_album_release",
        "n_tracks",
        "composer_album_count",
        "album_cohesion_score",
    ]
    album_binary = [c for c in album_features if c not in album_continuous]

    awards_binary = derived_award_cols

    continuous_features = film_continuous + album_continuous
    binary_features = film_binary + album_binary + awards_binary

    return {
        "target_col": target_col,
        "threshold": threshold,
        "film_continuous": film_continuous,
        "film_binary": film_binary,
        "album_continuous": album_continuous,
        "album_binary": album_binary,
        "awards_binary": awards_binary,
        "continuous_features": continuous_features,
        "binary_features": binary_features,
    }

def filter_continuous_features_by_correlation(
        album_analytics_df: pd.DataFrame,
        continuous_features: list[str],
        binary_features: list[str],
        target_col: str,
        threshold: float,
) -> dict:
    """
    Filter continuous predictors by their Pearson correlation with the target.

    This function performs the light-winnowing step of the regression
    workflow. All binary predictors are retained, while continuous
    predictors are filtered using a minimum absolute Pearson correlation
    threshold with the target variable.

    The goal is not aggressive feature selection, but modest noise
    reduction before later transformation and modeling steps.

    Args:
        album_analytics_df: Analysis-ready dataframe containing the target
            column and candidate predictor columns.
        continuous_features: List of continuous predictor column names to
            evaluate against the target.
        binary_features: List of binary predictor column names to retain
            without correlation filtering.
        target_col: Name of the regression target column.
        threshold: Minimum absolute Pearson correlation required to keep a
            continuous predictor.

    Returns:
        dict: Dictionary containing:
            - "cont_corr": Pearson correlations of continuous predictors
              with the target
            - "kept_continuous": Continuous predictors meeting the
              threshold
            - "dropped_continuous": Continuous predictors below the
              threshold
            - "x_cols": Final predictor list consisting of all binary
              features plus retained continuous features
            - "summary_df": Report-ready table of continuous-feature
              correlations with keep/drop status
    """

    df = album_analytics_df[
        binary_features + continuous_features + [target_col]
        ].copy()

    # Pearson correlations for continuous features vs target (pairwise complete)
    cont_corr = (
        df[[target_col] + continuous_features]
        .corr(method="pearson")[target_col]
        .drop(target_col)
    )

    kept_continuous = cont_corr[cont_corr.abs() >= threshold].index.tolist()
    dropped_continuous = cont_corr[cont_corr.abs() < threshold].index.tolist()

    # Final X list for OLS (binary all kept + filtered continuous)
    # Final predictor list:
    # - All binary flags
    # - Only continuous predictors with a minimal linear signal vs target
    x_cols = binary_features + kept_continuous

    summary_df = (
        cont_corr
        .rename("pearson_r")
        .to_frame()
        .assign(
            abs_pearson_r=lambda x: x["pearson_r"].abs(),
            kept=lambda x: x["abs_pearson_r"] >= threshold,
        )
        .sort_values("abs_pearson_r", ascending=False)
    )

    return {
        "cont_corr": cont_corr,
        "kept_continuous": kept_continuous,
        "dropped_continuous": dropped_continuous,
        "x_cols": x_cols,
        "summary_df": summary_df,
    }

def apply_regression_transforms(
        album_analytics_df: pd.DataFrame,
        x_cols: list[str],
        binary_features: list[str],
        target_col: str,
) -> dict:
    """
    Apply regression-specific transforms to the modeling dataframe.

    This function builds the post-winnowing modeling frame, applies
    log transforms to selected heavy-tailed continuous predictors, and
    standardizes all continuous predictors. Binary predictors are left
    unchanged as 0/1 indicators, and the target is assumed to already be
    transformed upstream if needed.

    Args:
        album_analytics_df: Analysis-ready dataframe containing the target
            column and selected predictor columns.
        x_cols: Final predictor columns retained after light winnowing.
        binary_features: List of binary predictor columns that should not
            be transformed or standardized.
        target_col: Name of the regression target column.

    Returns:
        dict: Dictionary containing:
            - "df_model": Transformed modeling dataframe
            - "x_cont": Continuous predictors included in the model
            - "x_bin": Binary predictors included in the model
            - "heavy_tailed": Candidate heavy-tailed predictors
            - "logged_predictors": Heavy-tailed predictors that were
              actually log-transformed
    """
    df_model = album_analytics_df[x_cols + [target_col]].copy()

    # Identify predictor types
    x_cont = [c for c in x_cols if c not in binary_features]  # continuous numeric predictors
    x_bin = [c for c in x_cols if c in binary_features]  # 0/1 flags (genres, awards, etc.)

    # (a) Log-transform ONLY the heavy-tailed exposure variables (if present)
    # SPLOM justification: these are strongly right-skewed and benefit from log compression.
    heavy_tailed = [
        "film_vote_count",
        "film_budget",
        "film_revenue",
        "film_popularity",
    ]

    for c in heavy_tailed:
        if c in x_cont:
            # log1p handles zeros; clip avoids issues if anything weird slipped in
            df_model[c] = np.log1p(df_model[c].clip(lower=0))

    # (b) Z-score all continuous predictors (post-log where applicable)
    # Recommended because it puts continuous predictors on a common scale and makes
    # coefficients easier to compare (does NOT change model fit).
    for c in x_cont:
        mu = df_model[c].mean(skipna=True)
        sd = df_model[c].std(skipna=True, ddof=0)

        # Avoid divide-by-zero if a column is constant
        if sd == 0 or np.isnan(sd):
            df_model[c] = df_model[c] - mu
        else:
            df_model[c] = (df_model[c] - mu) / sd

    # Cohesion is missing for albums with insufficient valid track-audio data.
    # After standardization, impute missing cohesion scores to 0 so these rows
    # stay in the model. The companion binary flag album_cohesion_has_audio_data
    # captures whether the cohesion value was actually observed.
    if "album_cohesion_score" in df_model.columns:
        df_model["album_cohesion_score"] = df_model["album_cohesion_score"].fillna(0.0)

    logged_predictors = [c for c in heavy_tailed if c in x_cont]

    # Binary predictors stay as 0/1 (no scaling, no transforms)
    # Target is already log-transformed upstream in your pipeline.

    # df_model is now ready for Step 4 (OLS)

    return {
        "df_model": df_model,
        "x_cont": x_cont,
        "x_bin": x_bin,
        "heavy_tailed": heavy_tailed,
        "logged_predictors": logged_predictors,
    }

def finalize_regression_predictors(
        df_model: pd.DataFrame,
        target_col: str = "log_lfm_album_listeners",
) -> dict:
    """
    Finalize the regression modeling dataframe before OLS fitting.

    This function applies the last predictor-level cleanup decisions
    before model fitting. It removes selected columns to reduce
    multicollinearity and converts film genre indicator columns from
    boolean to integer format so all predictors are consistently numeric
    for downstream modeling and reporting.

    Args:
        df_model: Transformed modeling dataframe containing the target
            column and selected predictor columns.
        target_col: Name of the regression target column.

    Returns:
        dict: Dictionary containing:
            - "df_model_final": Final modeling dataframe after predictor
              cleanup
            - "dropped_columns": Predictor columns removed before fitting
            - "film_genre_cols": Film genre indicator columns converted
              to integer type
            - "x_cols": Final predictor columns used for OLS, excluding
              the target
    """
    # print(df_model.columns)

    # To prevent multicollinearity:
    # 1) Drop days_since_film_release (nearly perfectly correlated with days_since_album_release)
    # 2) Keep only ONE film exposure proxy (film_vote_count) and drop the rest
    dropped_columns = [
        "days_since_film_release",
        "film_budget",
        "film_revenue",
    ]

    df_model_final = df_model.drop(columns=dropped_columns, errors="ignore")


    # The film_is columns are boolean -- convert them to int
    film_genre_cols = [c for c in df_model_final.columns if c.startswith("film_is_")]

    df_model_final[film_genre_cols] = df_model_final[film_genre_cols].astype(int)
    x_cols = [c for c in df_model_final.columns if c != target_col]

    # print("Final model:", df_model_final.columns)
    return {
        "df_model_final": df_model_final,
        "dropped_columns": dropped_columns,
        "film_genre_cols": film_genre_cols,
        "x_cols": x_cols,
    }

def fit_final_ols_model(
        df_model_final: pd.DataFrame,
        target_col: str = "log_lfm_album_listeners",
) -> dict:
    """
    Fit the final OLS regression model.

    This function prepares the finalized modeling dataframe for ordinary
    least squares regression by coercing columns to numeric types,
    dropping incomplete rows, splitting predictors from the target, and
    fitting an OLS model with an added intercept term.

    Args:
        df_model_final: Final modeling dataframe containing the target
            column and cleaned predictor columns.
        target_col: Name of the regression target column.

    Returns:
        dict: Dictionary containing:
            - "results": Fitted statsmodels OLS results object
            - "model_df_reg": Numeric modeling dataframe used for fitting
            - "x_cols": Final predictor columns used in the model
            - "n_rows": Number of rows used in the fitted model
            - "n_predictors": Number of predictor columns used before
              adding the intercept
    """
    # ------------------------------------------------------------
    # Final OLS regression (minimal + correct ordering)
    # ------------------------------------------------------------

    y_col = target_col

    # Use all remaining columns except the target
    x_cols = [c for c in df_model_final.columns if c != y_col]

    # ------------------------------------------------------------
    # 1) Build modeling frame and force numeric types
    # ------------------------------------------------------------

    model_df_reg = df_model_final[[y_col] + x_cols].copy()

    # Coerce everything to numeric (required by statsmodels)
    model_df_reg = model_df_reg.apply(pd.to_numeric, errors="coerce")

    # Drop rows with any missing values
    model_df_reg = model_df_reg.dropna()

    # print("Modeling rows:", model_df_reg.shape[0])
    # print("Predictors:", len(x_cols))

    # ------------------------------------------------------------
    # 2) Split into X and y
    # ------------------------------------------------------------

    label = model_df_reg[y_col]
    features = model_df_reg[x_cols]

    # Add intercept
    features = sm.add_constant(features)

    # ------------------------------------------------------------
    # 3) Fit OLS model
    # ------------------------------------------------------------

    results = sm.OLS(label, features).fit()

    # print(results.summary())

    return {
        "results": results,
        "model_df_reg": model_df_reg,
        "x_cols": x_cols,
        "n_rows": model_df_reg.shape[0],
        "n_predictors": len(x_cols),
    }


def run_regression_pipeline(
        album_analytics_df: pd.DataFrame,
        target_col: str = TARGET_COL,
        threshold: float = 0.05,
) -> dict:
    """
    Run the full regression-preparation and OLS modeling workflow.

    This orchestrates regression feature grouping, light winnowing of
    continuous predictors, transformation of selected predictors, final
    predictor cleanup, and fitting of the final OLS model.

    Args:
        album_analytics_df: Analysis-ready dataframe used for modeling.
        target_col: Name of the regression target column.
        threshold: Minimum absolute Pearson correlation required to keep a
            continuous predictor during light winnowing.

    Returns:
        dict: Dictionary containing feature configuration, intermediate
        filtering and transformation artifacts, finalized predictors, and
        final OLS results for reporting and visualization.
    """
    feature_config = define_regression_features(
        film_features=FILM_FEATURES,
        album_features=ALBUM_FEATURES,
        derived_award_cols=DERIVED_AWARD_COLS,
        target_col=target_col,
        threshold=threshold,
    )

    filter_results = filter_continuous_features_by_correlation(
        album_analytics_df=album_analytics_df,
        continuous_features=feature_config["continuous_features"],
        binary_features=feature_config["binary_features"],
        target_col=feature_config["target_col"],
        threshold=feature_config["threshold"],
    )

    transform_results = apply_regression_transforms(
        album_analytics_df=album_analytics_df,
        x_cols=filter_results["x_cols"],
        binary_features=feature_config["binary_features"],
        target_col=feature_config["target_col"],
    )

    finalize_results = finalize_regression_predictors(
        df_model=transform_results["df_model"],
        target_col=feature_config["target_col"],
    )

    ols_results = fit_final_ols_model(
        df_model_final=finalize_results["df_model_final"],
        target_col=feature_config["target_col"],
    )

    return {
        "feature_config": feature_config,
        "filter_results": filter_results,
        "transform_results": transform_results,
        "finalize_results": finalize_results,
        "ols_results": ols_results,
    }

def build_vote_count_scatter_data(
        model_df_reg: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build point and fitted-line data for the vote count scatterplot.

    This function prepares the exact data needed for the post-regression
    visualization that shows the relationship between film vote count and
    log soundtrack listeners. It extracts only the required columns, drops
    missing values so the fitted line does not fail, and computes a simple
    best-fit line using NumPy.

    Args:
        model_df_reg: Regression sample dataframe containing at least
            ``film_vote_count`` and ``log_lfm_album_listeners``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A two-item tuple containing:
            1. ``plot_df`` with the scatterplot points
            2. ``line_df`` with two endpoints for the fitted line

    Raises:
        ValueError: If there are fewer than two valid rows available after
            removing missing values.
    """
    plot_df = model_df_reg[
        ["film_vote_count", "log_lfm_album_listeners"]
    ].copy()

    # Drop missing values before fitting so polyfit does not fail or
    # silently produce invalid results.
    plot_df = plot_df.dropna()

    if len(plot_df) < 2:
        raise ValueError(
            "At least two non-null rows are required to fit the vote count "
            "scatterplot line."
        )

    # Fit simple line in Python: y = a*x + b
    x = plot_df["film_vote_count"].to_numpy()
    y = plot_df["log_lfm_album_listeners"].to_numpy()
    a, b = np.polyfit(x, y, 1)  # Fit the best-fitting line across datapoints

    # Make a 2-point line spanning the x-range
    x_min, x_max = float(x.min()), float(x.max())
    line_df = pd.DataFrame({
        "film_vote_count": [x_min, x_max],
        "log_lfm_album_listeners": [a * x_min + b, a * x_max + b]
    })

    return plot_df, line_df

def build_coefficient_plot_df(
        results: sm.regression.linear_model.RegressionResultsWrapper
) -> pd.DataFrame:
    """Build a tidy dataframe for the coefficient confidence interval plot.

    This function converts the fitted statsmodels OLS result object into a
    chart-ready dataframe containing coefficients, confidence intervals,
    significance grouping, and sorting fields. The intercept is excluded
    because it is generally not useful in the coefficient interpretation
    graphic.

    Args:
        results: Fitted statsmodels OLS regression results object.

    Returns:
        pd.DataFrame: Tidy coefficient dataframe with one row per modeled
        predictor and columns for coefficient value, confidence interval
        bounds, zero-crossing group, and absolute coefficient magnitude.
    """
    conf_int_df = results.conf_int()

    # -----------------------------
    # 1) Build tidy coefficient table
    # -----------------------------
    coef_df = (
        pd.DataFrame({
            "feature": results.params.index,
            "coef": results.params.values,
            "ci_low": conf_int_df[0].values,
            "ci_high": conf_int_df[1].values,
        })
        .query("feature != 'const'")
        .copy()
    )

    # CI crosses zero? (True/False)
    coef_df["crosses_zero"] = (
        (coef_df["ci_low"] <= 0) & (coef_df["ci_high"] >= 0)
    )

    # Make it a string category so Altair treats it as two discrete groups
    # reliably.
    coef_df["ci_group"] = coef_df["crosses_zero"].map(
        {True: "CI crosses 0", False: "CI does NOT cross 0"}
    )

    # Sort so the largest absolute effects appear first (top)
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    return coef_df

def build_scatterplot_feature_ranking(
        album_analytics_df: pd.DataFrame,
        continuous_features: list[str] | None = None,
        target_col: str = TARGET_COL,
        method: str = "pearson",
) -> pd.DataFrame:
    """
    Rank continuous features by their univariate relationship with the target.

    This helper is intended for Streamlit-style exploratory scatterplot
    pages. It evaluates each continuous feature against the target using
    a simple pairwise correlation, then derives a univariate R-squared
    value (r^2) so features can be sorted from strongest to weakest
    linear relationship.

    Unlike the multivariate OLS model R-squared, these values reflect
    one-feature-at-a-time relationships only.

    Args:
        album_analytics_df: Analysis-ready dataframe containing candidate
            continuous features and the target column.
        continuous_features: Optional list of continuous features to rank.
            If omitted, the function uses the same continuous feature set
            defined by ``define_regression_features(...)``.
        target_col: Name of the target column.
        method: Correlation method to use. For this use case, "pearson"
            is the default and most interpretable because r^2 corresponds
            to simple linear fit strength.

    Returns:
        pd.DataFrame: Ranked dataframe with one row per feature and the
        following columns:
            - feature
            - corr
            - abs_corr
            - r_squared
            - direction
            - rank

    Raises:
        ValueError: If the target column is missing, no continuous
            features are available, or none of the requested features
            exist in the dataframe.
    """
    if target_col not in album_analytics_df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in the analysis "
            f"dataframe."
        )

    if continuous_features is None:
        feature_config = define_regression_features(
            film_features=FILM_FEATURES,
            album_features=ALBUM_FEATURES,
            derived_award_cols=DERIVED_AWARD_COLS,
            target_col=target_col,
        )
        continuous_features = feature_config["continuous_features"]

    if not continuous_features:
        raise ValueError(
            "At least one continuous feature is required to build the "
            "scatterplot feature ranking."
        )

    available_features = [
        col for col in continuous_features if col in album_analytics_df.columns
    ]
    if not available_features:
        raise ValueError(
            "None of the requested continuous features were found in the "
            "analysis dataframe."
        )

    corr_df = album_analytics_df[[target_col] + available_features].copy()

    corr_series = (
        corr_df
        .corr(method=method)[target_col]
        .drop(target_col)
        .rename("corr")
    )

    ranking_df = (
        corr_series
        .to_frame()
        .assign(
            abs_corr=lambda x: x["corr"].abs(),
            r_squared=lambda x: x["corr"] ** 2,
            direction=lambda x: np.where(x["corr"] >= 0, "positive", "negative"),
        )
        .sort_values("abs_corr", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    ranking_df["rank"] = range(1, len(ranking_df) + 1)

    return ranking_df

def build_exploratory_scatter_data(
        album_analytics_df: pd.DataFrame,
        feature_col: str,
        metadata_df: pd.DataFrame | None = None,
        metadata_cols: list[str] | None = None,
        id_cols: list[str] | None = None,
        target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build a generic scatterplot dataset using regression-style transforms.

    This helper prepares point-level and fitted-line data for exploratory
    scatterplots of one selected continuous feature against the soundtrack
    popularity target. The selected feature is transformed using the same
    logic as the regression workflow:

    - heavy-tailed features are log-transformed when appropriate
    - continuous features are standardized afterward

    The returned point dataframe includes both the transformed x-value
    used for plotting and the raw feature value used for tooltip display.
    If a richer metadata dataframe is supplied, descriptive columns such
    as soundtrack title, film title, or composer can also be merged in.

    Args:
        album_analytics_df: Analysis-ready dataframe containing IDs, the
            selected feature, and the target column.
        feature_col: Continuous feature to plot on the x-axis.
        metadata_df: Optional richer dataframe to merge in for tooltip
            metadata. This is useful because the analysis-ready dataframe
            does not retain descriptive title fields.
        metadata_cols: Optional list of descriptive columns to keep from
            ``metadata_df``. Only columns that actually exist will be used.
        id_cols: Optional identifier columns used to merge metadata.
            Defaults to ``["tmdb_id", "release_group_mbid"]``.
        target_col: Name of the target column to plot on the y-axis.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]:
            1. Point dataframe for scatterplot rendering
            2. Two-point fitted line dataframe
            3. Metrics/metadata dictionary describing the transform and fit

    Raises:
        ValueError: If the selected feature or target column is missing,
            or if fewer than two valid rows remain after preparation.
    """
    if id_cols is None:
        id_cols = ["tmdb_id", "release_group_mbid"]

    if target_col not in album_analytics_df.columns:
        raise ValueError(
            f"Target column '{target_col}' was not found in the analysis "
            f"dataframe."
        )

    if feature_col not in album_analytics_df.columns:
        raise ValueError(
            f"Feature column '{feature_col}' was not found in the analysis "
            f"dataframe."
        )

    # Use the regression feature definitions so we can reuse the exact
    # binary/continuous distinction from the modeling workflow.
    feature_config = define_regression_features(
        film_features=FILM_FEATURES,
        album_features=ALBUM_FEATURES,
        derived_award_cols=DERIVED_AWARD_COLS,
        target_col=target_col,
    )
    binary_features = feature_config["binary_features"]

    if feature_col in binary_features:
        raise ValueError(
            f"Feature '{feature_col}' is binary. This helper is intended "
            f"for continuous-feature scatterplots only."
        )

    required_cols = [c for c in id_cols if c in album_analytics_df.columns]
    source_df = album_analytics_df[required_cols + [feature_col, target_col]].copy()

    # Preserve raw feature values for tooltip use before any transforms.
    source_df[f"{feature_col}_raw"] = source_df[feature_col]

    # Reuse the same regression transform logic applied in the model
    # workflow so the exploratory plot geometry matches the regression view.
    transform_results = apply_regression_transforms(
        album_analytics_df=source_df,
        x_cols=[feature_col],
        binary_features=binary_features,
        target_col=target_col,
    )
    df_model = transform_results["df_model"].copy()

    # Attach IDs and raw feature values back to the transformed frame.
    df_model[required_cols] = source_df[required_cols]
    df_model[f"{feature_col}_raw"] = source_df[f"{feature_col}_raw"]

    plot_df = df_model.rename(
        columns={
            feature_col: "x_value",
            target_col: "y_value",
            f"{feature_col}_raw": "x_raw_value",
        }
    )

    # Bring in optional descriptive metadata for tooltips.
    if metadata_df is not None:
        if metadata_cols is None:
            metadata_cols = []

        available_meta_cols = [c for c in metadata_cols if c in metadata_df.columns]
        available_id_cols = [c for c in id_cols if c in metadata_df.columns]

        if available_meta_cols and available_id_cols:
            meta_merge_df = metadata_df[available_id_cols + available_meta_cols].drop_duplicates()
            plot_df = plot_df.merge(
                meta_merge_df,
                on=[c for c in available_id_cols if c in plot_df.columns],
                how="left",
            )

    # Remove incomplete rows after transforms / metadata merge.
    plot_df = plot_df.dropna(subset=["x_value", "y_value"])

    if len(plot_df) < 2:
        raise ValueError(
            "At least two non-null rows are required to build the "
            "exploratory scatterplot."
        )

    x = plot_df["x_value"].to_numpy()
    y = plot_df["y_value"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)

    x_min, x_max = float(x.min()), float(x.max())
    line_df = pd.DataFrame({
        "x_value": [x_min, x_max],
        "y_value": [slope * x_min + intercept, slope * x_max + intercept],
    })

    corr = float(plot_df["x_value"].corr(plot_df["y_value"], method="pearson"))

    metrics = {
        "feature_col": feature_col,
        "target_col": target_col,
        "rows_used": len(plot_df),
        "pearson_r": corr,
        "r_squared": corr ** 2,
        "is_logged": feature_col in transform_results["logged_predictors"],
        "is_standardized": feature_col in transform_results["x_cont"],
        "x_axis_label": (
            f"{feature_col} (log-transformed, standardized)"
            if feature_col in transform_results["logged_predictors"]
            else f"{feature_col} (standardized)"
        ),
        "tooltip_raw_col": "x_raw_value",
    }

    return plot_df, line_df, metrics