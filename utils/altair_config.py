"""
Project-wide Altair runtime configuration utilities.

This module centralizes the Altair setup that should be applied once per
script run before charts are created. Keeping renderer, data transformer,
and theme activation logic in one place avoids repeating the same setup
code across `main.py`, `analysis.py`, and any future visualization
modules.

Because this project generates visualizations from Python scripts rather
than a notebook environment, it is helpful to define a single shared
configuration step that prepares Altair for chart creation and export.
"""

import altair as alt
from utils.viz_theme import enable


def configure_altair() -> None:
    """
    Configure Altair for script-based chart generation.

    This enables HTML rendering, disables Altair's default max-row
    limit, and activates the project's custom visualization theme.

    Returns:
        None
    """
    alt.renderers.enable("html")
    alt.data_transformers.disable_max_rows()
    enable()