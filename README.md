# Soundtrack Popularity Analysis Pipeline

## Overview

This repository contains a productionized Python implementation of a data analysis workflow originally developed in a Jupyter Notebook. The goal of the project is to analyze factors associated with soundtrack album popularity by combining film-level features with album-level metadata and listener statistics.

The pipeline performs the following tasks:

1. Loads soundtrack album and track-level data from CSV files
2. Cleans and engineers analytical features
3. Performs exploratory correlation analysis
4. Runs a regression workflow to model soundtrack popularity
5. Generates HTML reports containing charts and statistical outputs

The original notebook included additional exploratory components such as hypothesis testing. For the purposes of this production script conversion, the implemented workflow focuses on the core **data preparation, correlation analysis, and regression modeling pipeline** needed for reproducible analysis.

---

# Quick Start (30 seconds)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the analysis

```bash
python main.py \
    data/albums.csv \
    data/wide.csv \
    --output-dir output \
    --open-browser
```

3. View the generated reports

```
output/correlation_report.html
output/regression_report.html
```

---

# Repository Structure

```
.
├── main.py
├── data_processing.py
├── analysis.py
├── regression_analysis.py
├── regression_visualization.py
├── requirements.txt
├── README.md
└── data/
    ├── albums.csv
    └── wide.csv
```

### main.py

Command-line entry point for the pipeline.

This script orchestrates the full workflow:

- loading input datasets
- building the analysis dataframe
- running correlation analysis
- running regression analysis
- generating HTML reports

---

### data_processing.py

Contains functions responsible for:

- loading input CSV files
- feature engineering
- preparing the final album-level analysis dataset

Features created include:

- track counts per soundtrack
- award nomination counts
- composer album counts
- normalized genre indicator flags

---

### analysis.py

Provides correlation-based analysis utilities including:

- correlation matrix computation
- correlation heatmap visualization
- feature correlation lollipop chart
- robustness diagnostics comparing Pearson vs Spearman correlations
- sensitivity tests that trim extreme target values

---

### regression_analysis.py

Implements the regression workflow:

1. Defines candidate predictors
2. Filters weak continuous predictors
3. Applies transformations (log transforms and standardization)
4. Removes predictors that may introduce multicollinearity
5. Fits a final OLS regression model using Statsmodels

This module also prepares datasets used for regression visualizations.

---

### regression_visualization.py

Contains Altair chart construction functions used in the regression report:

- film vote count vs soundtrack listener scatterplot
- coefficient dot-and-whisker plot with confidence intervals

---

# Data Inputs

The script expects two CSV files.

## albums.csv

Album-level dataset containing soundtrack metadata and listener statistics.

Example columns include:

- `release_group_mbid`
- `tmdb_id`
- `log_lfm_album_listeners`
- genre indicators
- award nominations
- film metadata

---

## wide.csv

Track-level dataset used to derive additional album features.

Example columns include:

- `release_group_mbid`
- `tmdb_id`
- `track_id`

These datasets are merged and transformed to construct the final analysis dataframe used by the pipeline.

---

# Environment Setup

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

---

## 2. Create a virtual environment

Example using Python venv:

```bash
python -m venv testenv
```

Activate it.

### Mac / Linux

```bash
source testenv/bin/activate
```

### Windows

```bash
testenv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

- pandas
- numpy
- altair
- statsmodels

---

# Running the Analysis

The pipeline is executed through `main.py`.

Example command:

```bash
python main.py \
    data/albums.csv \
    data/wide.csv \
    --output-dir output \
    --open-browser
```

---

# Command Line Arguments

| Argument | Description |
|--------|--------|
| `album_file_path` | Positional path to the album-level dataset (e.g., `data/albums.csv`) |
| `wide_file_path` | Positional path to the track-level dataset (e.g., `data/wide.csv`) |
| `--output-dir` | Directory where HTML reports will be written |
| `--open-browser` | Optional flag that automatically opens generated reports in the default web browser |

Example without opening the browser:

```bash
python main.py \
    data/albums.csv \
    data/wide.csv \
    --output-dir output
```

---

# Outputs

The script generates two HTML reports.

## Correlation Report

```
correlation_report.html
```

Includes:

- correlation heatmap
- lollipop chart showing feature correlations with album popularity
- robustness diagnostics comparing Pearson and Spearman correlations
- trimmed distribution sensitivity analysis

---

## Regression Report

```
regression_report.html
```

Includes:

- regression predictor filtering summary
- full OLS regression results table
- scatterplot showing film exposure vs soundtrack popularity
- coefficient visualization with confidence intervals

---

# Code Quality and Best Practices

This project was designed following Python best practices.

### Modularity

The codebase is organized into logical modules:

- data preparation
- exploratory analysis
- regression modeling
- visualization

Each function has a **single responsibility** and clearly defined inputs and outputs.

---

### Documentation

All modules and functions include:

- detailed docstrings
- type hints
- explanatory comments for non-obvious logic

---

### Linting

The code passes pylint with a perfect score:

```
10.00 / 10
```

This confirms compliance with:

- PEP8 style guidelines
- proper naming conventions
- maintainable code structure

---

# Testing

The script was tested in a fresh Python environment to ensure:

- all dependencies install correctly
- the pipeline runs without errors
- reports generate successfully

The modular structure of the project makes it straightforward to extend with unit tests if desired.

---

# Assignment Context

This repository satisfies the requirements of the **MLP Productionalization Assignment**, which required converting a previously developed Jupyter Notebook into a reusable Python script.

The project demonstrates:

- modular Python architecture
- reusable functions with single responsibilities
- type hints and documentation
- linting compliance
- configurable command-line execution
- reproducible analytical outputs

The original source notebook is also included in the repository for reference during grading and comparison.

While the original notebook contained additional exploratory components (such as hypothesis testing), the production script focuses on the **core reproducible analytical workflow** required to generate consistent outputs from new datasets.