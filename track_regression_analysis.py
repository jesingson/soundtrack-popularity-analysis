from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


TRACK_REGRESSION_TARGET_OPTIONS = [
    "log_lfm_track_playcount",
    "log_lfm_track_listeners",
    "spotify_popularity",
]

TRACK_CONTINUOUS_FEATURES = [
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

TRACK_BINARY_FEATURES = [
    "is_first_track",
    "is_last_track",
    "is_instrumental",
    "is_major_mode",
]

TRACK_HEAVY_TAILED_FEATURES = [
    "duration_seconds",
    "tempo",
    "film_vote_count",
]

TRACK_FILM_CONTROL_FEATURES = [
    "film_year",
    "film_vote_count",
    "film_popularity",
    "film_budget",
    "film_revenue",
    "film_rating",
    "film_runtime_min",
    "days_since_film_release",
]

TRACK_ALBUM_CONTROL_FEATURES = [
    "n_tracks",
    "album_release_lag_days",
    "composer_album_count",
    "album_cohesion_score",
]

TRACK_CONTEXT_BINARY_FEATURES = [
    "film_is_action",
    "film_is_adventure",
    "film_is_animation",
    "film_is_comedy",
    "film_is_crime",
    "film_is_documentary",
    "film_is_drama",
    "film_is_family",
    "film_is_fantasy",
    "film_is_history",
    "film_is_horror",
    "film_is_music",
    "film_is_mystery",
    "film_is_romance",
    "film_is_science_fiction",
    "film_is_tv_movie",
    "film_is_thriller",
    "film_is_war",
    "film_is_western",
    "ambient_experimental",
    "classical_orchestral",
    "electronic",
    "hip_hop_rnb",
    "pop",
    "rock",
    "world_folk",
    "bafta_nominee",
]

TRACK_PIPELINE_ARTIFACT_EXCLUSIONS = {
    "album_cohesion_has_audio_data",
    "track_audio_feature_count",
    "track_has_any_audio_features",
}

def define_track_regression_features(
    track_df: pd.DataFrame,
    target_col: str = "log_lfm_track_playcount",
    threshold: float = 0.05,
) -> dict:
    """
    Define grouped predictor lists for the track regression workflow.

    Args:
        track_df: Track-level dataframe.
        target_col: Selected regression target column.
        threshold: Minimum absolute Pearson correlation required to keep a
            continuous predictor during the light filtering step.

    Returns:
        dict: Regression feature configuration.
    """
    available_continuous = [
        col for col in TRACK_CONTINUOUS_FEATURES
        if col in track_df.columns
    ]

    available_binary = [
        col for col in TRACK_BINARY_FEATURES
        if col in track_df.columns
    ]

    available_film_controls = [
        col for col in TRACK_FILM_CONTROL_FEATURES
        if col in track_df.columns
    ]

    available_album_controls = [
        col for col in TRACK_ALBUM_CONTROL_FEATURES
        if col in track_df.columns
    ]

    available_context_binary = [
        col for col in TRACK_CONTEXT_BINARY_FEATURES
        if col in track_df.columns
    ]

    return {
        "target_col": target_col,
        "threshold": threshold,
        "continuous_features": available_continuous,
        "binary_features": available_binary,
        "film_controls": available_film_controls,
        "album_controls": available_album_controls,
        "context_binary_features": available_context_binary,
    }


def get_track_leakage_exclusions(target_col: str) -> set[str]:
    """
    Return predictors that should be excluded for the selected target.

    This prevents obvious leakage or near-duplicate target relationships.
    """
    exclusions = set()

    if target_col == "log_lfm_track_playcount":
        exclusions.update({"log_lfm_track_listeners", "lfm_track_listeners"})
    elif target_col == "log_lfm_track_listeners":
        exclusions.update({"log_lfm_track_playcount", "lfm_track_playcount"})
    elif target_col == "spotify_popularity":
        exclusions.update({"popularity"})

    exclusions.update({
        "lfm_album_listeners",
        "lfm_album_playcount",
        "album_cohesion_has_audio_data",
        "track_audio_feature_count",
        "track_has_any_audio_features",
    })
    exclusions.update(TRACK_PIPELINE_ARTIFACT_EXCLUSIONS)

    return exclusions

def get_track_filter_based_exclusions(global_controls: dict | None) -> set[str]:
    """
    Return predictor columns that should be excluded because the user has
    already conditioned the dataset on them via global filters.

    Examples:
    - If the user filters to Animation films, `film_is_animation` should
      not also be used as a predictor.
    - If the user filters to Pop albums, `pop` should not also be used as
      a predictor.

    Continuous range filters like film_year are NOT excluded, because the
    filtered dataframe can still retain variation within the selected range.
    """
    if not global_controls:
        return set()

    exclusions = set()

    film_genres = global_controls.get("selected_film_genres", []) or []
    for genre in film_genres:
        slug = str(genre).strip().lower().replace("&", "and").replace(" ", "_")
        exclusions.add(f"film_is_{slug}")

    album_genres = global_controls.get("selected_album_genres", []) or []
    album_genre_map = {
        "Ambient/Experimental": "ambient_experimental",
        "Classical/Orchestral": "classical_orchestral",
        "Electronic": "electronic",
        "Hip-Hop/R&B": "hip_hop_rnb",
        "Pop": "pop",
        "Rock": "rock",
        "World/Folk": "world_folk",
    }
    for genre in album_genres:
        if genre in album_genre_map:
            exclusions.add(album_genre_map[genre])

    return exclusions

def filter_continuous_track_features_by_correlation(
    track_df: pd.DataFrame,
    continuous_features: list[str],
    binary_features: list[str],
    control_features: list[str],
    target_col: str,
    threshold: float,
) -> dict:
    """
    Filter continuous track predictors by Pearson correlation with the target.

    Track-level continuous predictors are lightly screened by threshold.
    Film/album control features are always retained when supplied, even if
    their raw pairwise correlation with the selected target is weak.
    """
    leakage_exclusions = get_track_leakage_exclusions(target_col)

    continuous_features = [
        col for col in continuous_features
        if col not in leakage_exclusions
    ]
    binary_features = [
        col for col in binary_features
        if col not in leakage_exclusions
    ]
    control_features = [
        col for col in control_features
        if col not in leakage_exclusions
    ]

    df = track_df[
        binary_features + continuous_features + control_features + [target_col]
    ].copy()

    cont_corr = (
        df[[target_col] + continuous_features]
        .corr(method="pearson")[target_col]
        .drop(target_col)
    )

    kept_continuous = cont_corr[cont_corr.abs() >= threshold].index.tolist()
    dropped_continuous = cont_corr[cont_corr.abs() < threshold].index.tolist()

    x_cols = binary_features + kept_continuous + control_features

    summary_df = (
        cont_corr
        .rename("pearson_r")
        .to_frame()
        .assign(
            abs_pearson_r=lambda x: x["pearson_r"].abs(),
            kept=lambda x: x["abs_pearson_r"] >= threshold,
            feature_role="screened track feature",
        )
        .sort_values("abs_pearson_r", ascending=False)
    )

    if control_features:
        control_summary_df = pd.DataFrame(
            {
                "pearson_r": [pd.NA] * len(control_features),
                "abs_pearson_r": [pd.NA] * len(control_features),
                "kept": [True] * len(control_features),
                "feature_role": ["forced-in context control"] * len(control_features),
            },
            index=control_features,
        )
        summary_df = pd.concat([summary_df, control_summary_df], axis=0)

    return {
        "cont_corr": cont_corr,
        "kept_continuous": kept_continuous,
        "dropped_continuous": dropped_continuous,
        "kept_controls": control_features,
        "x_cols": x_cols,
        "summary_df": summary_df,
    }


def apply_track_regression_transforms(
    track_df: pd.DataFrame,
    x_cols: list[str],
    binary_features: list[str],
    target_col: str,
) -> dict:
    """
    Apply regression-specific transforms to the track modeling dataframe.
    """
    df_model = track_df[x_cols + [target_col]].copy()

    x_cont = [c for c in x_cols if c not in binary_features]
    x_bin = [c for c in x_cols if c in binary_features]

    for c in TRACK_HEAVY_TAILED_FEATURES:
        if c in x_cont:
            df_model[c] = np.log1p(df_model[c].clip(lower=0))

    for c in x_cont:
        mu = df_model[c].mean(skipna=True)
        sd = df_model[c].std(skipna=True, ddof=0)

        if sd == 0 or np.isnan(sd):
            df_model[c] = df_model[c] - mu
        else:
            df_model[c] = (df_model[c] - mu) / sd

    for c in x_bin:
        df_model[c] = (
            pd.to_numeric(df_model[c], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    logged_predictors = [c for c in TRACK_HEAVY_TAILED_FEATURES if c in x_cont]

    return {
        "df_model": df_model,
        "x_cont": x_cont,
        "x_bin": x_bin,
        "logged_predictors": logged_predictors,
    }


def finalize_track_regression_predictors(
    df_model: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    Apply final cleanup before OLS fitting.

    This step removes a few intentionally excluded predictors to reduce
    obvious redundancy among track-level features.
    """
    dropped_columns = [
        col for col in [
            "track_speech_texture_score",
            "is_first_three_tracks",
            "is_first_five_tracks",
        ]
        if col in df_model.columns
    ]

    df_model_final = df_model.drop(columns=dropped_columns, errors="ignore").copy()
    x_cols = [c for c in df_model_final.columns if c != target_col]

    return {
        "df_model_final": df_model_final,
        "dropped_columns": dropped_columns,
        "x_cols": x_cols,
    }


def fit_final_track_ols_model(
    df_model_final: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    Fit the final OLS model for the selected track target.
    """
    y_col = target_col
    x_cols = [c for c in df_model_final.columns if c != y_col]

    model_df_reg = df_model_final[[y_col] + x_cols].copy()
    model_df_reg = model_df_reg.apply(pd.to_numeric, errors="coerce").dropna()

    label = model_df_reg[y_col]
    features = model_df_reg[x_cols]
    features = sm.add_constant(features)

    results = sm.OLS(label, features).fit()

    return {
        "results": results,
        "model_df_reg": model_df_reg,
        "x_cols": x_cols,
        "n_rows": model_df_reg.shape[0],
        "n_predictors": len(x_cols),
    }


def run_track_regression_pipeline(
    track_df: pd.DataFrame,
    target_col: str = "log_lfm_track_playcount",
    threshold: float = 0.05,
    include_context_controls: bool = True,
    global_controls: dict | None = None,
    excluded_features: list[str] | None = None,
) -> dict:
    """
    Run the full track regression pipeline.
    """
    feature_config = define_track_regression_features(
        track_df=track_df,
        target_col=target_col,
        threshold=threshold,
    )

    hard_excluded_features = {
        "album_cohesion_has_audio_data",
        "track_audio_feature_count",
        "track_has_any_audio_features",
    }
    excluded_feature_set = set(excluded_features or []) | hard_excluded_features

    feature_config["continuous_features"] = [
        col for col in feature_config["continuous_features"]
        if col not in excluded_feature_set
    ]
    feature_config["binary_features"] = [
        col for col in feature_config["binary_features"]
        if col not in excluded_feature_set
    ]
    feature_config["film_controls"] = [
        col for col in feature_config["film_controls"]
        if col not in excluded_feature_set
    ]
    feature_config["album_controls"] = [
        col for col in feature_config["album_controls"]
        if col not in excluded_feature_set
    ]
    feature_config["context_binary_features"] = [
        col for col in feature_config["context_binary_features"]
        if col not in excluded_feature_set
    ]

    filter_based_exclusions = get_track_filter_based_exclusions(global_controls)

    feature_config["continuous_features"] = [
        col for col in feature_config["continuous_features"]
        if col not in filter_based_exclusions
    ]
    feature_config["binary_features"] = [
        col for col in feature_config["binary_features"]
        if col not in filter_based_exclusions
    ]
    feature_config["film_controls"] = [
        col for col in feature_config["film_controls"]
        if col not in filter_based_exclusions
    ]
    feature_config["album_controls"] = [
        col for col in feature_config["album_controls"]
        if col not in filter_based_exclusions
    ]
    feature_config["context_binary_features"] = [
        col for col in feature_config["context_binary_features"]
        if col not in filter_based_exclusions
    ]
    if include_context_controls:
        control_features = (
                feature_config["film_controls"]
                + feature_config["album_controls"]
        )
        binary_features = (
                feature_config["binary_features"]
                + feature_config["context_binary_features"]
        )
    else:
        control_features = []
        binary_features = feature_config["binary_features"]

    filter_results = filter_continuous_track_features_by_correlation(
        track_df=track_df,
        continuous_features=feature_config["continuous_features"],
        binary_features=binary_features,
        control_features=control_features,
        target_col=feature_config["target_col"],
        threshold=feature_config["threshold"],
    )

    transform_results = apply_track_regression_transforms(
        track_df=track_df,
        x_cols=filter_results["x_cols"],
        binary_features=binary_features,
        target_col=feature_config["target_col"],
    )

    finalize_results = finalize_track_regression_predictors(
        df_model=transform_results["df_model"],
        target_col=feature_config["target_col"],
    )

    ols_results = fit_final_track_ols_model(
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


def build_track_coefficient_plot_df(
    results: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    """
    Build a tidy coefficient dataframe for track regression charts.
    """
    conf_int_df = results.conf_int()

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

    coef_df["crosses_zero"] = (
        (coef_df["ci_low"] <= 0) & (coef_df["ci_high"] >= 0)
    )

    coef_df["ci_group"] = coef_df["crosses_zero"].map(
        {True: "CI crosses 0", False: "CI does NOT cross 0"}
    )

    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)

    return coef_df