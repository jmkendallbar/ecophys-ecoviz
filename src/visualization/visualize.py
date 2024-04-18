import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LABEL_COLORS = {
    "Certain REM Sleep": "#FCBE46",
    "Putative REM Sleep": "#FCBE46",
    "HV Slow Wave Sleep": "#41b6c4",
    "LV Slow Wave Sleep": "#c7e9b4",
    "Drowsiness": "#BBA9CF",
    "Quiet Waking": "#225ea8",
    "Active Waking": "#0c2c84",
    "Unscorable": "#D7D7D7"
}

def plot_feature_density(df, feat_col, labels_col, filter_val=0.01):
    """
    plots the density of a feature, colored by the label
    df: DataFrame with data
    feat_col: name of the column with 1Hz data of the feature
    labels_col: name of the column with sleep label
    filter: boolean for amount to filter out on either extreme
    """
    fig, ax = plt.subplots()
    df = df[
        (df[feat_col] > df[feat_col].quantile(filter_val)) &
        (df[feat_col] < df[feat_col].quantile(1 - filter_val))
    ]
    ax = sns.kdeplot(df, x=feat_col, hue=labels_col, palette=LABEL_COLORS)
