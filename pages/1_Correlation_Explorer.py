import streamlit as st

import analysis as an
from app.app_data import load_analysis_data
from app.app_controls import get_correlation_controls


def render_lollipop_section(
    album_analytics_df,
    method: str,
    top_n: int,
    show_table: bool,
) -> None:
    """
    Render the lollipop chart section.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        method: Correlation method.
        top_n: Number of top features to display.
        show_table: Whether to show the underlying dataframe.
    """
    st.subheader("Feature Correlations with Album Popularity")

    corr_df_plot = an.prepare_lollipop_data(
        album_analytics_df=album_analytics_df,
        target_col="log_lfm_album_listeners",
        method=method,
    ).head(top_n).copy()

    chart = an.plot_lollipop_chart(corr_df_plot)
    st.altair_chart(chart, use_container_width=True)

    if show_table:
        st.dataframe(corr_df_plot, use_container_width=True)


def render_heatmap_section(
    album_analytics_df,
    method: str,
    show_table: bool,
) -> None:
    """
    Render the correlation heatmap section.

    Args:
        album_analytics_df: Analysis-ready album dataframe.
        method: Correlation method.
        show_table: Whether to show the underlying matrix and long table.
    """
    st.subheader("Correlation Heatmap")

    corr_matrix = an.compute_correlation_matrix(
        album_analytics_df=album_analytics_df,
        method=method,
    )
    heatmap = an.plot_correlation_heatmap(
        corr_matrix=corr_matrix,
        title=f"Correlation Heatmap ({method.title()})",
    )

    st.altair_chart(heatmap, use_container_width=True)

    if show_table:
        st.write("Correlation matrix")
        st.dataframe(corr_matrix, use_container_width=True)

        st.write("Long-form heatmap source")
        st.dataframe(an.corr_to_long(corr_matrix), use_container_width=True)


def main() -> None:
    """
    Run the correlation explorer page.
    """
    st.set_page_config(
        page_title="Correlation Explorer",
        layout="wide",
    )

    st.title("Correlation Explorer")
    st.write(
        """
        Explore pairwise feature relationships and feature-level correlations
        with soundtrack album popularity.
        """
    )

    controls = get_correlation_controls()
    album_analytics_df = load_analysis_data()

    st.caption(
        f"Rows loaded: {len(album_analytics_df):,} | "
        f"Method: {controls['method'].title()}"
    )

    render_lollipop_section(
        album_analytics_df=album_analytics_df,
        method=controls["method"],
        top_n=controls["top_n"],
        show_table=controls["show_lollipop_table"],
    )

    st.divider()

    render_heatmap_section(
        album_analytics_df=album_analytics_df,
        method=controls["method"],
        show_table=controls["show_heatmap_table"],
    )


if __name__ == "__main__":
    main()