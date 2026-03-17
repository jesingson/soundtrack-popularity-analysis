"""Command-line entry point for the soundtrack analysis reporting pipeline.

This script orchestrates the end-to-end workflow for the productionized
soundtrack analysis project. It loads the input datasets, builds the
analysis-ready album dataframe, runs correlation and regression analyses,
generates HTML reports, and optionally opens those reports in a browser.

The workflow is organized across helper modules:

- ``data_processing`` for loading and transforming source data
- ``analysis`` for correlation diagnostics and exploratory analysis
- ``regression_analysis`` for regression pipeline logic and plot data prep
- ``regression_visualization`` for post-regression chart rendering
"""
import argparse
import webbrowser
from pathlib import Path

import pandas as pd
import altair as alt
from statsmodels.regression.linear_model import RegressionResultsWrapper

# Our modules
from utils.altair_config import configure_altair
import data_processing as dp
import analysis as an
import regression_analysis as reg
import regression_visualization as reg_viz

# Press the green button in the gutter to run the script.
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for input files and output options.

    This collects the album and wide CSV paths, the output directory
    for generated charts, and the optional browser-opening flag.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("album_file_path",
                        type=str,
                        help="Album file (CSV) location")
    parser.add_argument("wide_file_path",
                        type=str,
                        help="Wide file (CSV) location")
    parser.add_argument("--output-dir",
                        type = str,
                        default = "outputs",
                        help = "Directory where output charts will be saved")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open saved chart outputs in the default web browser",
    )

    args = parser.parse_args()
    return args

def save_chart(chart: alt.TopLevelMixin, output_path: Path) -> None:
    """
    Save an Altair chart to an output file.

    Args:
        chart: Altair chart object to save.
        output_path: Output file path.
    """
    chart.save(str(output_path))

def append_html_section(
        report_path: Path,
        section_html: str
) -> None:
    """
    Append a custom HTML section to an existing Altair-generated HTML file.

    Args:
        report_path: Path to the saved HTML report.
        section_html: HTML content to insert before the closing body tag.
    """
    html_text = report_path.read_text(encoding="utf-8")

    if "</body>" in html_text:
        html_text = html_text.replace("</body>", f"{section_html}\n</body>")
    else:
        html_text += section_html

    report_path.write_text(html_text, encoding="utf-8")

def build_robustness_html(
        metrics: dict,
        top_features_html: str
) -> str:
    """
    Build the HTML section for the correlation robustness check.

    Args:
        metrics: Robustness summary metrics dictionary.
        top_features_html: HTML table for the top feature comparison.

    Returns:
        str: HTML section for the robustness report content.
    """
    overlap_features_str = ", ".join(metrics["top_n_overlap_features"])
    rows_used_str = (
        f"{metrics['rows_used']} / {metrics['rows_before_dropna']}"
    )
    top_overlap_str = (
        f"{metrics['top_n_overlap_count']} / {metrics['top_n_used']}"
    )

    return f"""
<section style="max-width: 1100px; margin: 32px auto; padding: 24px;">
    <h2>Correlation Robustness Check</h2>
    <p>
        This diagnostic compares Pearson and Spearman target-feature
        correlations to assess whether the same general feature story
        emerges across methods.
    </p>
    <ul>
        <li><strong>Target column:</strong> {metrics["target_col"]}</li>
        <li><strong>Rows used:</strong> {rows_used_str}</li>
        <li><strong>Top-{metrics["top_n_used"]} overlap:</strong> {top_overlap_str}</li>
        <li><strong>Rank agreement:</strong> {metrics["rank_agreement"]}</li>
        <li><strong>Overlapping features:</strong> {overlap_features_str}</li>
    </ul>
    <h3>Top Features Comparison</h3>
    {top_features_html}
</section>
"""


def build_trim_html(trim_metrics: dict) -> str:
    """
    Build the HTML section for the outlier sensitivity check.

    Args:
        trim_metrics: Outlier sensitivity metrics dictionary.

    Returns:
        str: HTML section for trimmed correlation diagnostics.
    """
    trim_rows_str = (
        f"{trim_metrics['rows_after_trim']} / "
        f"{trim_metrics['rows_before_trim']}"
    )

    return f"""
<section style="max-width: 1100px; margin: 32px auto; padding: 24px;">
    <h2>Outlier Sensitivity Check</h2>
    <ul>
        <li><strong>Target column:</strong> {trim_metrics["target_col"]}</li>
        <li><strong>Trim threshold:</strong> {trim_metrics["trim_threshold"]}</li>
        <li><strong>Rows kept after trim:</strong> {trim_rows_str}</li>
        <li><strong>Pearson stability:</strong> {trim_metrics["pearson_stability"]}</li>
        <li><strong>Spearman stability:</strong> {trim_metrics["spearman_stability"]}</li>
    </ul>
</section>
"""


def run_correlation_reporting(
        album_analytics_df: pd.DataFrame,
        output_dir: Path
) -> Path:
    """
    Run correlation diagnostics and save the combined HTML report.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        output_dir: Directory where the HTML report should be saved.

    Returns:
        Path: Saved correlation report path.
    """
    print("Calculating and visualizing correlation analyses")
    corr_matrix = an.compute_correlation_matrix(
        album_analytics_df,
        method="spearman",
    )
    heatmap = an.plot_correlation_heatmap(
        corr_matrix,
        title="Correlation Heatmap (Spearman)",
    )

    print("Visualizing lollipop plot...")
    corr_df = an.prepare_lollipop_data(
        album_analytics_df,
        method="pearson"
    )
    lollipop = an.plot_lollipop_chart(corr_df)

    print("Running robustness check...")
    robustness_results = an.compare_target_correlations_by_method(
        analysis_df=album_analytics_df,
        target_col="log_lfm_album_listeners",
        top_n=15,
    )
    metrics = robustness_results["metrics"]
    top_features_df = robustness_results["top_features_df"]

    top_features_html = top_features_df.to_html(
        classes="table table-striped",
        border=0,
    )
    robustness_html = build_robustness_html(metrics, top_features_html)

    trim_results = an.assess_trimmed_correlation_stability(
        analysis_df=album_analytics_df,
        target_col="log_lfm_album_listeners",
        trim_q=0.99,
    )
    trim_html = build_trim_html(trim_results["metrics"])

    combined_report = heatmap & lollipop
    report_path = output_dir / "correlation_report.html"
    save_chart(combined_report, report_path)

    append_html_section(report_path, robustness_html)
    append_html_section(report_path, trim_html)

    print(f"Saved combined report to {report_path}")
    return report_path


def build_regression_report_html(
        regression_results: dict,
        filter_summary_html: str,
        ols_summary_html: str,
        vote_count_chart_html: str,
        coefficient_chart_html: str
) -> str:
    """
    Build the full regression HTML report.

    Args:
        regression_results: Full regression pipeline results dictionary.
        filter_summary_html: HTML table of continuous feature correlations.
        ols_summary_html: Plain-text OLS summary wrapped in HTML.
        vote_count_chart_html: Embedded Altair HTML fragment for the
            vote count scatterplot.
        coefficient_chart_html: Embedded Altair HTML fragment for the
            coefficient plot.

    Returns:
        str: Complete regression report HTML document.
    """
    section_style = "max-width: 1100px; margin: 32px auto; padding: 24px;"

    filter_results = regression_results["filter_results"]
    transform_results = regression_results["transform_results"]
    finalize_results = regression_results["finalize_results"]
    ols_results = regression_results["ols_results"]
    feature_config = regression_results["feature_config"]

    continuous_kept_str = (
        f"{len(filter_results['kept_continuous'])} / "
        f"{len(feature_config['continuous_features'])}"
    )
    logged_predictors_str = ", ".join(transform_results["logged_predictors"])
    dropped_columns_str = ", ".join(finalize_results["dropped_columns"])

    return f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Regression Report</title>
</head>
<body>
    <section style="{section_style}">
        <h1>Regression Analysis Report</h1>
        <p>
            This report summarizes the regression preparation workflow,
            predictor filtering, transformations, final predictor cleanup,
            and OLS model results.
        </p>

        <h2>Feature Filtering Summary</h2>
        <ul>
            <li><strong>Target column:</strong> {feature_config["target_col"]}</li>
            <li><strong>Correlation threshold:</strong> {feature_config["threshold"]}</li>
            <li><strong>Continuous predictors kept:</strong> {continuous_kept_str}</li>
            <li><strong>Binary predictors retained:</strong>
                {len(feature_config["binary_features"])}</li>
            <li><strong>Predictors after filtering:</strong>
                {len(filter_results["x_cols"])}</li>
        </ul>

        <h3>Continuous Feature Correlations</h3>
        {filter_summary_html}

        <h2>Transform Summary</h2>
        <ul>
            <li><strong>Continuous predictors standardized:</strong>
                {len(transform_results["x_cont"])}</li>
            <li><strong>Binary predictors unchanged:</strong>
                {len(transform_results["x_bin"])}</li>
            <li><strong>Logged predictors:</strong>
                {logged_predictors_str}</li>
        </ul>

        <h2>Final Predictor Cleanup</h2>
        <ul>
            <li><strong>Dropped columns:</strong> {dropped_columns_str}</li>
            <li><strong>Final predictors used in OLS:</strong>
                {len(finalize_results["x_cols"])}</li>
        </ul>

        <h2>OLS Results</h2>
        <ul>
            <li><strong>Modeling rows used:</strong> {ols_results["n_rows"]}</li>
            <li><strong>Predictor count:</strong> {ols_results["n_predictors"]}</li>
        </ul>
        {ols_summary_html}

        <h2>Film Exposure vs Soundtrack Popularity</h2>
        <p>
            This scatterplot shows the relationship between film vote count
            and log soundtrack listeners, along with a fitted trend line.
        </p>
        {vote_count_chart_html}

        <h2>Coefficient Plot</h2>
        <p>
            This dot-and-whisker chart shows coefficient estimates and
            95% confidence intervals for the final regression model.
        </p>
        {coefficient_chart_html}
    </section>
</body>
</html>
"""

def build_regression_chart_html(
        model_df_reg: pd.DataFrame,
        results: RegressionResultsWrapper
) -> tuple[str, str]:
    """
    Build embedded HTML fragments for the post-regression charts.

    Args:
        model_df_reg: Regression modeling dataframe used for the final OLS fit.
        results: Fitted statsmodels OLS results object.

    Returns:
        tuple[str, str]: Embedded HTML for the vote count scatterplot and
        the coefficient plot.
    """
    plot_df, line_df = reg.build_vote_count_scatter_data(model_df_reg)
    vote_count_chart = reg_viz.create_vote_count_scatter_chart(
        plot_df,
        line_df,
    )

    coef_df = reg.build_coefficient_plot_df(results)
    coefficient_chart = reg_viz.create_coefficient_whisker_chart(coef_df)

    # Must explicitly use two different div ids so that both render.
    vote_count_chart_html = vote_count_chart.to_html(
        fullhtml=False,
        output_div="vote_count_chart",
    )
    coefficient_chart_html = coefficient_chart.to_html(
        fullhtml=False,
        output_div="coefficient_chart",
    )

    return vote_count_chart_html, coefficient_chart_html

def run_regression_reporting(
        album_analytics_df: pd.DataFrame,
        output_dir: Path
) -> Path:
    """
    Run the regression workflow and save the regression HTML report.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        output_dir: Directory where the regression report should be saved.

    Returns:
        Path: Saved regression report path.
    """
    print("Running regression...")
    regression_results = reg.run_regression_pipeline(
        album_analytics_df=album_analytics_df,
        target_col="log_lfm_album_listeners",
        threshold=0.05,
    )

    ols_results = regression_results["ols_results"]
    model_df_reg = ols_results["model_df_reg"]
    results = ols_results["results"]

    print("Visualizing regression...")
    vote_count_chart_html, coefficient_chart_html = build_regression_chart_html(
        model_df_reg,
        results,
    )

    filter_summary_html = regression_results["filter_results"][
        "summary_df"
    ].round(3).to_html(
        classes="table table-striped",
        border=0,
    )
    ols_summary_html = f"<pre>{results.summary().as_text()}</pre>"

    regression_html = build_regression_report_html(
        regression_results=regression_results,
        filter_summary_html=filter_summary_html,
        ols_summary_html=ols_summary_html,
        vote_count_chart_html=vote_count_chart_html,
        coefficient_chart_html=coefficient_chart_html,
    )

    regression_report_path = output_dir / "regression_report.html"
    regression_report_path.write_text(regression_html, encoding="utf-8")

    return regression_report_path

def main() -> None:
    """Run the full soundtrack analysis workflow from the command line.

    This function parses command-line arguments, loads the source datasets,
    builds the analysis-ready dataframe, generates correlation and regression
    HTML reports, and optionally opens the saved reports in the default
    browser.
    """
    args = parse_args()

    print("Loading data...")
    albums_df, wide_df = dp.load_input_data(
        args.album_file_path,
        args.wide_file_path,
    )

    print("Transforming data and building core analytics table...")
    album_analytics_df = dp.build_album_analytics(albums_df, wide_df)

    print("Setting up visualization configs...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_altair()

    report_path = run_correlation_reporting(album_analytics_df, output_dir)
    regression_report_path = run_regression_reporting(
        album_analytics_df,
        output_dir,
    )

    print("Opening report...")
    if args.open_browser:
        webbrowser.open(report_path.resolve().as_uri())
        webbrowser.open(regression_report_path.resolve().as_uri())

if __name__ == '__main__':
    main()
