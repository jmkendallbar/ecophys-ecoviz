import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pytz
import seaborn as sns

LABEL_COLORS = {
    "Certain REM Sleep": "#FCBE46",
    "Putative REM Sleep": "#FCBE46",
    "HV Slow Wave Sleep": "#41b6c4",
    "LV Slow Wave Sleep": "#c7e9b4",
    "Drowsiness": "#BBA9CF",
    "Quiet Waking": "#225ea8",
    "Active Waking": "#0c2c84",
    "Unscorable": "#D7D7D7",
    "SWS": "#41b6c4",
    "REM": "#FCBE46"
}

PST_TIMEZONE = pytz.timezone('America/Los_Angeles')

def plot_feature_density(df, feat_col, labels_col, title=None, filter_val=0.01):
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
    ax = sns.kdeplot(df, x=feat_col, hue=labels_col, common_norm=False, palette=LABEL_COLORS)
    if title is not None:
        ax.set_title(title)
    return ax


def smooth_outliers(a, q=0.01):
    a = a.copy()
    lower = np.quantile(a, q)
    upper = np.quantile(a, 1-q)
    a[a <= lower] = np.nan
    a[a >= upper] = np.nan
    a = a.interpolate(method='nearest')
    return a

def plot_nap(df, labels_col, nap_start_dt, nap_end_dt, preds=None, labels_ordered=('Unscorable', 'Active Waking', 'Quiet Waking', 'Drowsiness', 'SWS', 'REM'),
             figsize=(8, 14)):
    # Subplots
    nplots = 9 if preds is None else 10
    fig = plt.figure(figsize=figsize)
    # Create subplots manually using gridspec so that we can make the first subplot bigger
    ax = []
    gs = gridspec.GridSpec(nplots, 1, height_ratios=[1]*(nplots-1) + [1.75])
    for i in range(nplots):
        ax.append(fig.add_subplot(gs[i]))

    # Get nap df
    nap_df = df[(df.index >= nap_start_dt) & (df.index <= nap_end_dt)].copy()
    simple_sleep_codes = list(labels_ordered)
    labels_nap = pd.concat([pd.Series(simple_sleep_codes), nap_df[labels_col]], ignore_index=True).copy()

    # Plot True Labels on last subplot
    # Plot each segment with a different color
    label_colors_subset = {key: LABEL_COLORS[key] for key in labels_ordered}
    label_keys, label_colors = list(label_colors_subset.keys()), list(label_colors_subset.values())
    labels_as_ints = np.array([label_keys.index(val) for val in labels_nap]) # integer value represents the index in the label_keys
    ax[-1].matshow(labels_as_ints.reshape(1, -1), aspect='auto', cmap=(ListedColormap(label_colors)))
    patches = [Patch(color=LABEL_COLORS[label], label=label) for label in labels_ordered]
    ax[-1].legend(handles=patches)
    ax[-1].set_yticks([])
    ax[-1].set_ylabel('True\nLabel')

    # Plot Predictions
    if preds is not None:
        preds_nap = pd.concat([pd.Series(simple_sleep_codes),
                              preds[(preds.index >= nap_start_dt) & (preds.index <= nap_end_dt)]],
                              ignore_index=True)
        preds_as_ints = np.array([label_keys.index(val) for val in preds_nap]) # integer value represents the index in the label_keys
        ax[-2].matshow(preds_as_ints.reshape(1, -1), aspect='auto', cmap=(ListedColormap(label_colors)))
        ax[-2].set_yticks([])
        ax[-2].set_ylabel('Pred.\nLabel')

    # Absolute Delta Power
    abs_delta_power_subplot = 0
    nap_df['delta_power'] = nap_df['yasa_eeg_sdelta'] + nap_df['yasa_eeg_fdelta']
    nap_df['delta_power'] = smooth_outliers(nap_df['delta_power'])
    sns.lineplot(nap_df['delta_power'], ax=ax[abs_delta_power_subplot], color='mediumspringgreen')
    ax[abs_delta_power_subplot].axes.set_yticks([nap_df['delta_power'].min(), nap_df['delta_power'].mean(), nap_df['delta_power'].max()])
    ax[abs_delta_power_subplot].set_ylabel('Absolute\nDelta Power')

    # Relative Delta Power
    rel_delta_power_subplot=abs_delta_power_subplot+1
    nap_df['delta_power_relative'] = nap_df['yasa_eeg_sdelta_relative'] + nap_df['yasa_eeg_fdelta_relative']
    sns.lineplot(nap_df['delta_power_relative'], ax=ax[rel_delta_power_subplot], color='turquoise')
    ax[rel_delta_power_subplot].axes.set_yticks([nap_df['delta_power_relative'].min(), nap_df['delta_power_relative'].mean(), nap_df['delta_power_relative'].max()])
    ax[rel_delta_power_subplot].set_ylabel('Relative\nDelta Power')

    # EEG Standard Deviation
    eeg_std_dev_subplot=rel_delta_power_subplot+1
    sns.lineplot(nap_df['yasa_eeg_std'], ax=ax[eeg_std_dev_subplot], color='lightgreen')
    ax[eeg_std_dev_subplot].axes.set_yticks([nap_df['yasa_eeg_std'].min(), nap_df['yasa_eeg_std'].mean(), nap_df['yasa_eeg_std'].max()])
    ax[eeg_std_dev_subplot].set_ylabel('EEG Standard\nDeviation')

    # Raw Heart Rate
    heart_rate_subplot=eeg_std_dev_subplot+1
    sns.lineplot(nap_df['Heart Rate'], ax=ax[heart_rate_subplot], color='red')
    ax[heart_rate_subplot].axes.set_yticks([nap_df['Heart Rate'].min(), nap_df['Heart Rate'].mean(), nap_df['Heart Rate'].max()])
    ax[heart_rate_subplot].set_ylabel('Raw\nHeart Rate')

    # Heart Rate Mean
    heart_rate_mean_subplot=heart_rate_subplot+1
    sns.lineplot(nap_df['yasa_heartrate_mean'], ax=ax[heart_rate_mean_subplot], color='lightcoral')
    ax[heart_rate_mean_subplot].axes.set_yticks([nap_df['yasa_heartrate_mean'].min(), nap_df['yasa_heartrate_mean'].mean(), nap_df['yasa_heartrate_mean'].max()])
    ax[heart_rate_mean_subplot].set_ylabel('Heart Rate\nRolling Mean')

    # Heart Rate Very Low Frequency Power Ratio
    heart_rate_vlf_power_subplot=heart_rate_mean_subplot+1
    sns.lineplot(nap_df['yasa_heartrate_vlf_perc'], ax=ax[heart_rate_vlf_power_subplot], color='indianred')
    ax[heart_rate_vlf_power_subplot].axes.set_yticks([nap_df['yasa_heartrate_vlf_perc'].min(), nap_df['yasa_heartrate_vlf_perc'].mean(), nap_df['yasa_heartrate_vlf_perc'].max()])
    ax[heart_rate_vlf_power_subplot].set_ylabel('Heart Rate\nVLF Power\nRatio')

    # Pressure
    pressure_subplot=heart_rate_vlf_power_subplot+1
    sns.lineplot(nap_df['Pressure Mean'], ax=ax[pressure_subplot], color='lightblue')
    ax[pressure_subplot].axes.set_yticks([nap_df['Pressure Mean'].min(), nap_df['Pressure Mean'].mean(), nap_df['Pressure Mean'].max()])
    ax[pressure_subplot].set_ylabel('Pressure')
    # Add x ticks to pressure plot and format it nicely
    ax[pressure_subplot].xaxis.set_ticks(nap_df.index.values)
    ax[pressure_subplot].set_xticklabels(nap_df.index.values, rotation=45)
    ax[pressure_subplot].xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    ax[pressure_subplot].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M:%S', tz=PST_TIMEZONE))
    # Add blank subplot below pressure plot to allow space for axis labels
    ax[pressure_subplot+1].set_visible(False)

    # Get rid of x ticks in all plots except for pressure plot
    for i in range(len(ax)):
        if i != pressure_subplot:
            ax[i].set_xticks([])
            ax[i].set_xlabel('')

    plt.subplots_adjust(wspace=0, hspace=0.25)
    plt.show()

    # Return figure
    return fig, ax