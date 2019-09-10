import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def corrplot_(
    df=None,
    mask_type="numerical",
    figsize=(14, 14),
    fontsize=8,
    cpalette=(10, 220),
):
    """ Global function that produces customised correlation plot reducing redundancy. """

    if df is None:
        raise ReferenceError("\nDataFrame not found.")
    corr_data = df.corr()

    # Creates whitespace mask over upper right triangle section for repeated features
    upper_triangle_mask = np.zeros_like(corr_data, dtype=np.bool)
    upper_triangle_mask[np.triu_indices_from(upper_triangle_mask)] = True

    # Generates MatPlotLib subplot objects
    fig, ax = plt.subplots(figsize=figsize)

    # Calculates relative maximum from correlational data
    vmax = np.abs(corr_data.values[~upper_triangle_mask]).max()

    # Creates correlational heatmap with simple color intensity relative to distribution
    cmap = sns.diverging_palette(cpalette[0], cpalette[1], as_cmap=True)
    sns.heatmap(
        corr_data,
        mask=upper_triangle_mask,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        linecolor="lightgray",
        linewidths=1,
        ax=ax,
    )

    # Overlays feature names and corr. data values over whitespace mask
    for iterator in range(len(corr_data)):
        ax.text(
            iterator + 0.5,
            iterator + 0.5,
            corr_data.columns[iterator],
            ha="center",
            va="center",
            rotation=45,
        )

        for jterator in range(iterator + 1, len(corr_data)):
            value = "{:.3f}".format(corr_data.values[iterator, jterator])

            # Switch-case for numerical whitespace mask
            if mask_type == "numerical":
                ax.text(
                    jterator + 0.5,
                    (iterator + 0.5),
                    value,
                    ha="center",
                    va="center",
                )

            # Switch-case for categorical whitespace mask
            if mask_type == "categorical":
                ax.text(
                    jterator + 0.5,
                    (iterator + 0.5),
                    _value_to_category(value),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
    ax.axis("off")
