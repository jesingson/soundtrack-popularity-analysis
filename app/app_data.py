from pathlib import Path

import pandas as pd
import streamlit as st

import data_processing as dp
import regression_analysis as reg
import ridge_analysis as reg_ridge


@st.cache_data
def load_source_data(
    album_file_path: str = "data/albums.csv",
    wide_file_path: str = "data/wide.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw source CSV files.

    Args:
        album_file_path: Path to the album-level CSV.
        wide_file_path: Path to the wide-format CSV.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Raw albums and wide dataframes.
    """
    album_path = Path(album_file_path)
    wide_path = Path(wide_file_path)

    albums_df, wide_df = dp.load_input_data(
        str(album_path),
        str(wide_path),
    )
    return albums_df, wide_df

@st.cache_data
def load_explorer_data(
    album_file_path: str = "data/albums.csv",
    wide_file_path: str = "data/wide.csv",
) -> pd.DataFrame:
    """
    Load source CSVs and build the exploration-ready album dataframe.

    Args:
        album_file_path: Path to the album-level CSV.
        wide_file_path: Path to the wide-format CSV.

    Returns:
        pd.DataFrame: Album-level exploration dataframe with descriptive
        metadata and selected engineered features.
    """
    albums_df, wide_df = load_source_data(
        album_file_path=album_file_path,
        wide_file_path=wide_file_path,
    )
    return dp.build_album_explorer_dataset(albums_df, wide_df)

@st.cache_data
def load_track_explorer_data(
    album_file_path: str = "data/albums.csv",
    wide_file_path: str = "data/wide.csv",
) -> pd.DataFrame:
    """
    Load source CSVs and build the exploration-ready track dataframe.

    Args:
        album_file_path: Path to the album-level CSV.
        wide_file_path: Path to the wide-format CSV.

    Returns:
        pd.DataFrame: Track-level exploration dataframe enriched with album
        metadata and helper fields for Page 6.
    """
    albums_df, wide_df = load_source_data(
        album_file_path=album_file_path,
        wide_file_path=wide_file_path,
    )
    return dp.build_track_explorer_dataset(albums_df, wide_df)

@st.cache_data
def load_analysis_data(
    album_file_path: str = "data/albums.csv",
    wide_file_path: str = "data/wide.csv",
) -> pd.DataFrame:
    """
    Load source CSVs and build the analysis-ready album dataframe.

    Args:
        album_file_path: Path to the album-level CSV.
        wide_file_path: Path to the wide-format CSV.

    Returns:
        pd.DataFrame: Analysis-ready album dataframe.
    """
    albums_df, wide_df = load_source_data(
        album_file_path=album_file_path,
        wide_file_path=wide_file_path,
    )
    album_analytics_df = dp.build_album_analytics(albums_df, wide_df)
    return album_analytics_df


@st.cache_data
def load_regression_results(
    album_file_path: str = "data/albums.csv",
    wide_file_path: str = "data/wide.csv",
    target_col: str = dp.TARGET_COL,
    threshold: float = 0.05,
) -> dict:
    """
    Load analysis data and run the full regression pipeline.

    Args:
        album_file_path: Path to the album-level CSV.
        wide_file_path: Path to the wide-format CSV.
        target_col: Regression target column.
        threshold: Minimum absolute Pearson correlation required to keep
            a continuous predictor.

    Returns:
        dict: Full regression pipeline results dictionary.
    """
    album_analytics_df = load_analysis_data(
        album_file_path=album_file_path,
        wide_file_path=wide_file_path,
    )

    regression_results = reg.run_regression_pipeline(
        album_analytics_df=album_analytics_df,
        target_col=target_col,
        threshold=threshold,
    )
    return regression_results

@st.cache_data
def load_ridge_dynamic_groups(
    album_file_path: str = "data/albums.csv",
    wide_file_path: str = "data/wide.csv",
    y_col: str = dp.TARGET_COL,
    top_n: int = 8,
) -> dict:
    """
    Load analysis data and compute dynamic ridge feature groups.

    Args:
        album_file_path: Path to the album-level CSV.
        wide_file_path: Path to the wide-format CSV.
        y_col: Target column used for ranking features.
        top_n: Number of features to retain in each dynamic ridge group.

    Returns:
        dict: Dynamic ridge groups keyed by preset name.
    """
    album_analytics_df = load_analysis_data(
        album_file_path=album_file_path,
        wide_file_path=wide_file_path,
    )

    return reg_ridge.build_dynamic_ridge_groups(
        albums_df=album_analytics_df,
        y_col=y_col,
        top_n=top_n,
    )