import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

custom_pallete = {
    "red": "#BF616A",
    "orange": "#D08770",
    "yellow": "#EBCB8B",
    "green": "#A3BE8C",
    "purple": "#B48EAD",
    "light_green": "#8FBCBB",
    "light_gray_blue": "#88C0D0",
    "light_blue": "#81A1C1",
    "blue": "#5E81AC",
}

def plt_distr(df, plot_info):

    df_description = df[plot_info["field"]].describe(include="all")

    text_for_histplot = f"""
    Медиана: {df_description["50%"]}  
    Среднее: {df_description["mean"]} 

    Минимальная длина текста: {df_description["min"]} 

    Квантиль q=0.25: {df_description["25%"]} 
    Квантиль q=0.75: {df_description["75%"]} 

    Максимальная длина текста: {df_description["max"]}
    """
    text_for_histplot = text_for_histplot + plot_info["annotation"]

    plt.figure(figsize=(10, 7), dpi=500)

    plt.hist(
        df[plot_info["field"]],
        bins=100,
        density=True,
        color=custom_pallete["light_gray_blue"],
    )

    df[plot_info["field"]].plot.kde(
        color=custom_pallete["blue"], label=f"KDE {plot_info['field']}"
    )

    plt.vlines(
        x=df_description["50%"],
        ymax=1,
        ymin=0,
        colors=custom_pallete["red"],
        linewidth=2,
        label="Медиана",
    )

    plt.vlines(
        x=(
            df_description["25%"],
            df_description["75%"],
        ),
        ymax=1,
        ymin=0,
        colors=custom_pallete["yellow"],
        linewidth=2,
        label="0.25 и 0.75 квантили",
    )

    plt.annotate(
        text_for_histplot,
        xy=plot_info["ann_xy"],
        bbox=dict(boxstyle="square", fc="w", alpha=0.7),
        fontsize=12,
    )

    plt.xlabel(plot_info["xlabel"])
    plt.ylabel(plot_info["ylabel"])

    plt.xlim(plot_info["xlim"][0], plot_info["xlim"][1])
    plt.ylim(plot_info["ylim"][0], plot_info["ylim"][1])

    plt.title(plot_info["title"])

    plt.legend()

    plt.show()
