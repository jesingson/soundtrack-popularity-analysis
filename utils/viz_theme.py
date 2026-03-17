"""
Shared Altair theme utilities for the soundtrack analysis project.

This module defines the project's reusable chart styling so visualizations
created in different scripts or modules share the same typography, sizing,
axis treatment, and color palette. Centralizing these settings keeps the
visual design consistent and avoids repeating theme configuration logic
throughout the codebase.

In addition to the custom theme definition, this module provides a helper
for enabling the theme in the current Python session and a small sizing
utility for applying standard chart dimensions.
"""

import altair as alt

# Standardized chart dimensions that should look good in a report
DEFAULT_WIDTH = 560
DEFAULT_HEIGHT = 320

def team_theme():
    """
    Return the project's shared Altair theme configuration.

    This defines consistent chart styling for fonts, axes, legends,
    and categorical colors across all visualizations.

    Returns:
        dict: Altair theme configuration dictionary.
    """
    return {
        "config": {
            # Optional neutral background inspired by the Elegant Wedding palette:
            # Use only when a soft, editorial feel is desired.
            # Disabled by default for report figures, since white backgrounds
            # reproduce more reliably in Google Docs and PDFs.
            "background": "white",
            "view": {
                "fill": "white",
                "stroke": "transparent"   # removes the plot-area border for Tufte-minimalism
            },

            # Typography (Lato + fallbacks if Lato isn't available)
            "font": "Lato",

            # Style for the main chart title text (only shows up if you set .properties(title="..."))
            "title": {
                "font": "Lato",
                "fontSize": 16,
                "fontWeight": 600,
                "color": "#111111",            # main title color
                "subtitleColor": "#5E17A6",   # subtitle color
                "anchor": "middle", # "start" left-aligns titles for a report-style look
            },
            # Axis defaults: readable labels, light gridlines, subtle axis/tick styling.
            "axis": {
                "labelFont": "Lato",
                "titleFont": "Lato",
                "labelFontSize": 12,
                "titleFontSize": 12,

                # Tufte-minimalist settings
                "grid": False,      # no gridlines
                "ticks": False,     # no tick marks
                "domain": False,    # no axis baseline

                # Leave the colors on in case we turn things back on
                "gridColor": "#e9e9e9",
                "tickColor": "#cccccc",
                "domainColor": "#cccccc",
            },

            # Legend typography to match axes (keeps multi-chart docs feeling cohesive).
            "legend": {
                "labelFont": "Lato",
                "titleFont": "Lato",
                "labelFontSize": 12,
                "titleFontSize": 12,
            },

            # Mark defaults: slightly larger points + thicker lines for readability in docs.
            "point": {"filled": True, "size": 60},
            "line": {"strokeWidth": 2},

            # Categorical color palette inspired by the "Elegant Wedding" palette:
            # https://www.color-hex.com/color-palette/1054967
            # Note: the original palette includes a very light beige (#f5f5dc),
            # which we intentionally exclude here to avoid low-contrast series colors
            # on white report backgrounds.
            "range": {
                "category": [
                    "#7922CC", "#1195B2", "#CC0000", "#CE7E00",
                    "#5E17A6", "#0E7C93", "#9E0000", "#A86600",
                    "#3F1D5C", "#1F6F5B", "#8C4A00"
                ]
            }
        }
    }

def enable():
    """
    Register and enable the shared Altair theme.

    Call this once before chart creation so Altair visualizations
    use the project's standard styling.

    Returns:
        None
    """
    alt.themes.register("team_theme", team_theme)
    alt.themes.enable("team_theme")

def sized(chart, width = DEFAULT_WIDTH, height = DEFAULT_HEIGHT):
    """
    Apply standard dimensions to an Altair chart.

    This keeps chart sizing consistent across the project without
    repeating width and height properties in every chart definition.

    Args:
        chart: Altair chart to resize.
        width: Target chart width in pixels.
        height: Target chart height in pixels.

    Returns:
        alt.Chart: Chart with width and height applied.
    """
    return chart.properties(width=width, height=height)