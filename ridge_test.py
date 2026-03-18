import argparse
from pathlib import Path
from typing import Callable

import data_processing as dp
import ridge_analysis as ridge
import ridge_visualization as ridge_viz


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for ridge test execution.

    Returns:
        argparse.Namespace: Parsed test arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run targeted validation checks for ridge analysis prep."
    )
    parser.add_argument(
        "album_file_path",
        nargs="?",
        default="data/albums.csv",
        help="Path to the album-level CSV file.",
    )
    parser.add_argument(
        "wide_file_path",
        nargs="?",
        default="data/wide.csv",
        help="Path to the wide-format CSV file.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "config",
            "dynamic_groups",
            "viz_df",
            "ridge_long",
            "group_counts",
            "density",
            "order",
            "chart_df",
            "labels_df",
            "chart",
            "feature_group_counts",
            "feature_density_rows",
            "single_density",
        ],
        help="Specific ridge tests to run.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Top N features to use for dynamic ridge groups.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of bins used for ridge density estimation.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Odd-valued smoothing window for ridge density estimation.",
    )
    parser.add_argument(
        "--min-group-n",
        type=int,
        default=10,
        help="Minimum observations required per feature-group density.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="film_vote_count",
        help="Feature to use for feature-specific ridge checks.",
    )
    return parser.parse_args()


def load_album_analytics(
    album_file_path: str,
    wide_file_path: str,
):
    """
    Load source data and build the analysis-ready album dataframe.

    Args:
        album_file_path: Path to the album-level CSV.
        wide_file_path: Path to the wide-format CSV.

    Returns:
        pd.DataFrame: Analysis-ready album dataframe.
    """
    albums_df, wide_df = dp.load_input_data(album_file_path, wide_file_path)
    album_analytics_df = dp.build_album_analytics(albums_df, wide_df)
    return album_analytics_df


def test_config(ridge_outputs: dict, phase2_outputs: dict | None = None) -> None:
    """
    Print the resolved ridge feature configuration.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Optional Phase 2 outputs dictionary.
    """
    ridge_config = ridge_outputs["ridge_config"]

    print("=== Ridge Config ===")
    print("All features:")
    print(ridge_config["all_features"])
    print()

    print("Available ridge groups:")
    for group_name, cols in ridge_config["feature_groups_available"].items():
        print(f"{group_name} ({len(cols)}): {cols}")
    print()

    print("Continuous-like features:")
    print(ridge_config["cont_like_features"])
    print()

    print("Award count features:")
    print(ridge_config["award_count_features"])
    print()

    print("Binary flag features:")
    print(ridge_config["binary_flag_features"])
    print()


def test_dynamic_groups(
    ridge_outputs: dict,
    phase2_outputs: dict | None = None,
) -> None:
    """
    Print the computed dynamic ridge groups.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Optional Phase 2 outputs dictionary.
    """
    print("=== Dynamic Ridge Groups ===")
    for group_name, cols in ridge_outputs["dynamic_groups"].items():
        print(f"{group_name} ({len(cols)}): {cols}")
    print()


def test_viz_df(ridge_outputs: dict, phase2_outputs: dict | None = None) -> None:
    """
    Print shape and preview of the ridge working dataframe.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Optional Phase 2 outputs dictionary.
    """
    viz_df = ridge_outputs["viz_df"]

    print("=== viz_df ===")
    print(viz_df.shape)
    print(viz_df.head())
    print()


def test_ridge_long(ridge_outputs: dict, phase2_outputs: dict | None = None) -> None:
    """
    Print shape and preview of the long-form ridge dataframe.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Optional Phase 2 outputs dictionary.
    """
    ridge_long = ridge_outputs["ridge_long"]

    print("=== ridge_long ===")
    print(ridge_long.shape)
    print(ridge_long.head(10))
    print()


def test_group_counts(
    ridge_outputs: dict,
    phase2_outputs: dict | None = None,
) -> None:
    """
    Print observed feature-group counts from the long-form ridge dataframe.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Optional Phase 2 outputs dictionary.
    """
    ridge_long = ridge_outputs["ridge_long"]

    print("=== Ridge Group Counts ===")
    print(ridge_long.groupby(["feature", "group"], observed=True).size())
    print()


def test_density(ridge_outputs: dict, phase2_outputs: dict | None = None) -> None:
    """
    Print shape and preview of the precomputed ridge density dataframe.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Phase 2 outputs dictionary.
    """
    ridge_density_df = phase2_outputs["ridge_density_df"]

    print("=== ridge_density_df ===")
    print(ridge_density_df.shape)
    print(ridge_density_df.head(10))
    print()


def test_order(ridge_outputs: dict, phase2_outputs: dict | None = None) -> None:
    """
    Print the ridge ordering summary.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Phase 2 outputs dictionary.
    """
    order_df = phase2_outputs["order_df"]

    print("=== Ridge Order Table ===")
    print(order_df)
    print()


def test_chart_df(
    ridge_outputs: dict,
    phase2_outputs: dict | None = None,
) -> None:
    """
    Print shape and preview of the chart-ready ridge dataframe.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Phase 2 outputs dictionary.
    """
    ridge_chart_df = phase2_outputs["ridge_chart_df"]

    print("=== ridge_chart_df ===")
    print(ridge_chart_df.shape)
    print(ridge_chart_df.head(10))
    print()


def test_labels_df(
    ridge_outputs: dict,
    phase2_outputs: dict | None = None,
) -> None:
    """
    Print the chart label dataframe.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Phase 2 outputs dictionary.
    """
    labels_df = phase2_outputs["labels_df"]

    print("=== labels_df ===")
    print(labels_df.shape)
    print(labels_df.head(20))
    print()


def test_feature_group_counts(
    ridge_outputs: dict,
    phase2_outputs: dict | None = None,
    feature: str = "film_vote_count",
) -> None:
    """
    Print group counts for one feature from the long ridge dataframe.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Optional Phase 2 outputs dictionary.
        feature: Feature whose group counts should be printed.
    """
    ridge_long = ridge_outputs["ridge_long"]
    one = ridge_long[ridge_long["feature"] == feature].copy()

    print("=== Feature Group Counts ===")
    print(f"Feature: {feature}")
    print(one["group"].value_counts(dropna=False))
    print()


def test_feature_density_rows(
    ridge_outputs: dict,
    phase2_outputs: dict | None = None,
    feature: str = "film_vote_count",
) -> None:
    """
    Print the density rows for one feature from the precomputed density table.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Phase 2 outputs dictionary.
        feature: Feature whose density rows should be printed.
    """
    ridge_density_df = phase2_outputs["ridge_density_df"]
    one = ridge_density_df[ridge_density_df["feature"] == feature].copy()

    print("=== Feature Density Rows ===")
    print(f"Feature: {feature}")
    print(one.head(20))
    print()

    print("=== Density Row Counts by Group ===")
    print(one.groupby("group", observed=True).size())
    print()


def test_single_density(
    ridge_outputs: dict,
    phase2_outputs: dict | None = None,
    feature: str = "film_vote_count",
) -> None:
    """
    Build and save a single-feature density chart.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Phase 2 outputs dictionary.
        feature: Feature to visualize.
    """
    chart = ridge_viz.create_single_feature_density_chart(
        ridge_density_df=phase2_outputs["ridge_density_df"],
        feature=feature,
    )

    output_path = Path("outputs") / f"ridge_single_density_{feature}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_path))

    print("=== Single Feature Density Chart ===")
    print(f"Feature: {feature}")
    print(f"Saved chart to: {output_path}")
    print()


def test_chart(ridge_outputs: dict, phase2_outputs: dict | None = None) -> None:
    """
    Build and save a test ridge chart HTML file.

    Args:
        ridge_outputs: Output dictionary from build_ridge_prep_outputs().
        phase2_outputs: Phase 2 outputs dictionary.
    """
    chart = ridge_viz.create_ridge_chart(
        ridge_chart_df=phase2_outputs["ridge_chart_df"],
        labels_df=phase2_outputs["labels_df"],
    )

    output_path = Path("outputs") / "ridge_test_chart.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_path))

    print("=== Ridge Chart ===")
    print(f"Saved chart to: {output_path}")
    print()


def main() -> None:
    """
    Run selected ridge validation tests from the command line.
    """
    args = parse_args()

    album_analytics_df = load_album_analytics(
        args.album_file_path,
        args.wide_file_path,
    )

    ridge_outputs = ridge.build_ridge_prep_outputs(
        album_analytics_df,
        feature_groups={
            "core_static": ridge.DEFAULT_RIDGE_GROUPS["core_static"]
        },
        top_n=args.top_n,
    )

    requested_tests = args.tests
    run_all = "all" in requested_tests

    phase2_needed = run_all or any(
        test_name in requested_tests
        for test_name in [
            "density",
            "order",
            "chart_df",
            "labels_df",
            "chart",
            "feature_density_rows",
            "single_density",
        ]
    )

    phase2_outputs = None
    if phase2_needed:
        phase2_outputs = ridge.build_ridge_phase2_outputs(
            ridge_outputs=ridge_outputs,
            bins=args.bins,
            smooth_window=args.smooth_window,
            min_group_n=args.min_group_n,
        )

    test_registry: dict[str, Callable[[dict, dict | None], None]] = {
        "config": test_config,
        "dynamic_groups": test_dynamic_groups,
        "viz_df": test_viz_df,
        "ridge_long": test_ridge_long,
        "group_counts": test_group_counts,
        "density": test_density,
        "order": test_order,
        "chart_df": test_chart_df,
        "labels_df": test_labels_df,
        "chart": test_chart,
    }

    selected_tests = list(test_registry.keys()) if run_all else requested_tests.copy()

    if "feature_group_counts" in selected_tests:
        test_feature_group_counts(
            ridge_outputs,
            phase2_outputs,
            feature=args.feature,
        )
        selected_tests.remove("feature_group_counts")

    if "feature_density_rows" in selected_tests:
        test_feature_density_rows(
            ridge_outputs,
            phase2_outputs,
            feature=args.feature,
        )
        selected_tests.remove("feature_density_rows")

    if "single_density" in selected_tests:
        test_single_density(
            ridge_outputs,
            phase2_outputs,
            feature=args.feature,
        )
        selected_tests.remove("single_density")

    for test_name in selected_tests:
        test_registry[test_name](ridge_outputs, phase2_outputs)


if __name__ == "__main__":
    main()