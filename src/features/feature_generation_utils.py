
import antropy as ant
import datetime
from matplotlib import mlab as mlab
import matplotlib.pyplot as plt
import mne
from mne.filter import filter_data
import numpy as np
import pandas as pd
from scipy.integrate import simpson
import scipy.signal as sp_sig
import scipy.stats as sp_stats
from sleepecg import detect_heartbeats
import wfdb
import wfdb.processing
from yasa import sliding_window
from yasa import bandpower_from_psd_ndarray

# -----------------|
# COMMON FUNCTIONS |
# -----------------|
def generate_features_sequential(data, recording_start_datetime, data_type='EEG', sfreq=500, window_length_sec=3600, buffer_length_sec=300, epoch_size_sec=32,
                                 welch_window_sec=4, step_size=4, bands=None):
    """
    Generates features 1 hour at a time to preserve memory
    data: either eeg or heart rate data to generate features for
    recording_start_datetime: datetime of the beginning of the dataset
    data_type: either 'EEG' if data is an EEG channel, or 'HR' if data is derived heart rate
    sfreq: frequency of data, default 500 Hz
    window_length_sec: length of calculation window in seconds, default is one hour
    buffer_length_sec: length of buffer in seconds to include before and after calculation window (this helps the features near the endpoints be more accurate)
    epoch_size_sec: size of the epoch to use (in seconds) for EEG or heart rate feature calculation
    welch_window_sec: size of the welch window to use for EEG or heart rate frequency band welch spectral power calculation
    """
    feature_dfs = []
    # Start at "0", with the first epoch being centered around the first label
    window_size = int(sfreq * window_length_sec)
    buffer_size = int(sfreq * buffer_length_sec)
    for i in range(0, len(data), window_size):
        print (round(100 * i/len(data), 2), '% complete', sep='', end='\r')
        indices = [i, i + window_size]
        indices_extra_window = [indices[0] - buffer_size, indices[1] + buffer_size]
        if indices_extra_window[0] < 0:
            indices_extra_window[0] = 0
        if indices_extra_window[1] > len(data):
            indices_extra_window[1] = len(data)
        if indices[1] > len(data):
            indices[1] = len(data)
        data_subset = data[indices_extra_window[0]:indices_extra_window[1]]
        if data_type == 'EEG':
            features_subset = get_features_yasa_eeg(data_subset, sfreq=sfreq, epoch_window_sec=epoch_size_sec, welch_window_sec=welch_window_sec, step_size=step_size, bands=bands)
        elif data_type == 'HR':
            features_subset = get_features_yasa_heartrate(data_subset, sfreq=sfreq, epoch_window_sec=epoch_size_sec, welch_window_sec=welch_window_sec, step_size=step_size)
        
        # Add the starting index in seconds to the epoch index, then subset the returned features to only the current window (get rid of the buffer on each side)
        features_subset['epoch_index'] = features_subset['epoch_index'] + indices_extra_window[0] // 500
        feature_dfs.append(features_subset[features_subset['epoch_index'].between(indices[0]//500, indices[1]//500, inclusive='both')])
    print('100.00% complete!')
    print()

    features_df = pd.concat(feature_dfs).drop_duplicates(subset='epoch_index').reset_index(drop=True)
    features_df['yasa_time'] = pd.to_timedelta(features_df['epoch_index'], unit='s') + recording_start_datetime
    # insert start and end time to fill values for full dataset
    recording_end_datetime = recording_start_datetime + datetime.timedelta(seconds=len(data)//sfreq - 1)
    features_df = pd.concat([features_df, pd.DataFrame({'yasa_time': [recording_start_datetime]})], ignore_index=True)
    features_df = pd.concat([features_df, pd.DataFrame({'yasa_time': [recording_end_datetime]})], ignore_index=True)
    # Ffill and Bfill for if step size is >1 second
    fill_limit = epoch_size_sec // 2
    features_df = (
        features_df.set_index('yasa_time', drop=True).sort_index().resample('1S')
        .ffill(limit=fill_limit).bfill(limit=fill_limit+1)
    )
    features_df = features_df.drop('epoch_index', axis=1)
    return features_df


# ---------------------------------------------------------------------------------------------------------------------------------------
# https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho
def get_rolling_mean_std(a, freq=500, window_sec=30, step_size=1):
    """
    Returns a tuple of (rolling_mean, rolling_std) with the rolling mean and standard deviation over a window, each being arrays
        sampled at 1Hz calculating mean/std within the specified rolling window
    a: array to calculate rolling mean and standard deviation for
    start_index: index to start calculations (note this will use data from before the start_index to inform first datapoint if available)
    end_index: index to stop calculations (exclusive)
    freq: frequency of data
    window_sec: size of rolling window in seconds
    step_size: step size to increment data calculations (step_size=1 means function will return data at a 1 Hz sampling frequency)
        
    Example: 
    """
    window_length = window_sec*freq
    step_idx = int(step_size*freq)
    if window_length % 2 != 0:
        print('Note: (window * frequency) is not divisible by 2, will be rounded down to ' + str(window_length//2))
    rolling_mean = []
    rolling_std = []
    for i in range(0, len(a), step_idx):
        window_start = int(i - window_length/2)
        window_end = int(i + window_length/2)
        if window_start < 0:
            window_start = 0
            rolling_mean.append(np.nan)
            rolling_std.append(np.nan)
            # print(f'Window too large for index {i}; not enough preceding point')
        elif window_end > len(a):
            window_start = len(a)
            rolling_mean.append(np.nan)
            rolling_std.append(np.nan)
            # print(f'Window too large for index {i}; not enough following point')
        else:
            rolling_mean.append(np.mean(a[window_start:window_end]))
            rolling_std.append(np.std(a[window_start:window_end]))
    return (np.array(rolling_mean), np.array(rolling_std))

# ---------------------------------------------------------------------------------------------------------------------------------------

# -------------|
# ECG FEATURES |
# -------------|

# ---------------------------------------------------------------------------------------------------------------------------------------
def get_heart_rate(ecg_data, fs=500, search_radius=200, filter_threshold=200):
    """
    Gets heart rate at 1 Hz and extrapolates it to the same frequency as input data
    ecg_data: vector of ecg data
    fs: frequency of data
    search_radius: search radius to look for peaks (200 ~= 150 bpm upper bound)
    filter_threshold: threshold above which to throw out values (filter_threshold=200 would throw out any value above 200 bpm
                      and impute it from its neighbors)
    """
    rpeaks = detect_heartbeats(ecg_data, fs) # using sleepecg
    rpeaks_corrected = wfdb.processing.correct_peaks(
        ecg_data, rpeaks, search_radius=search_radius, smooth_window_size=50, peak_dir="up"
    )
    # MIGHT HAVE TO UPDATE search_radius
    heart_rates = [60 / ((rpeaks_corrected[i+1] - rpeaks_corrected[i]) / fs) for i in range(len(rpeaks_corrected) - 1)]
    # Create a heart rate array matching the frequency of the ECG trace
    hr_data = np.zeros_like(ecg_data)
    # Assign heart rate values to the intervals between R-peaks
    for i in range(len(rpeaks_corrected) - 1):
        start_idx = rpeaks_corrected[i]
        end_idx = rpeaks_corrected[i+1]
        hr_data[start_idx:end_idx] = heart_rates[i]
    hr_data = pd.Series(hr_data)
    filled = hr_data.isna().sum()
    hr_data[hr_data > filter_threshold] = np.nan
    hr_data = hr_data.interpolate(method='quadratic', order=5).fillna('ffill').fillna('bfill')
    print('Filled', filled, 'bad heart rate values')
    return hr_data

# ---------------------------------------------------------------------------------------------------------------------------------------
def get_features_yasa_heartrate(heart_rate_data, sfreq=500, epoch_window_sec=512, welch_window_sec=512, step_size=32):
    """
    Gets heartrate features using similar code & syntax as YASA's feature generation, calculates deterministic features as well as spectral features
    heart_rate_data: heart rate vector data (must already be processed from an ECG, this function does NOT take ECG data)
    sfreq: sampling frequeuency, by default 500 Hz
    epoch_window_sec: size of the epoch rolling window to use
    welch_window_sec: size of the welch window for power spectral density calculations (this affects the low frequeuncy power and very low frequency power calculations, etc.)
    step_size: how big of a step size to use, in seconds
    """
    dt_filt = filter_data(
        heart_rate_data, sfreq, l_freq=0, h_freq=1, verbose=False
    )
    
    # - Extract epochs. Data is now of shape (n_epochs, n_samples).
    times, epochs = sliding_window(dt_filt, sf=sfreq, window=epoch_window_sec, step=step_size)
    times = times + epoch_window_sec // 2 # add window/2 to the times to make the epochs "centered" around the times
    
    window_length = sfreq*welch_window_sec
    kwargs_welch = dict(
        window='hann',
        nperseg=window_length, # a little more than  4 minutes
        noverlap=window_length//2,
        scaling='density',
        average='median'
    )
    bands = [
        (0.0033, 0.04, 'vlf'),
        (0.04, 0.15, 'lf'),
        (0.15, 0.4, 'hf')
    ]
    freqs, psd = sp_sig.welch(epochs, sfreq, **kwargs_welch)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)
    
    feat = {}
    
    # Calculate standard descriptive statistics
    hmob, hcomp = ant.hjorth_params(epochs, axis=1)
    
    feat = {
        "mean": np.mean(epochs, axis=1),
        "std": np.std(epochs, ddof=1, axis=1),
        "iqr": sp_stats.iqr(epochs, rng=(25, 75), axis=1),
        "skew": sp_stats.skew(epochs, axis=1),
        "kurt": sp_stats.kurtosis(epochs, axis=1),
        "hmob": hmob,
        "hcomp": hcomp,
    }
    
    # Bandpowers
    for j, (_, _, b) in enumerate(bands):
        feat[b] = bp[j]

    feat['lf/hf'] = feat['lf'] / feat['hf']
    feat['p_total'] = feat['vlf'] + feat['lf'] + feat['hf']

    # compute relative and normalized power measures
    perc_factor = 100 / feat['p_total']
    feat['vlf_perc'] = feat['vlf'] * perc_factor
    feat['lf_perc'] = feat['lf'] * perc_factor
    feat['hf_perc'] = feat['hf'] * perc_factor

    nu_factor = 100 / (feat['lf'] + feat['hf'])
    feat['lf_nu'] = feat['lf'] * nu_factor
    feat['hf_nu'] = feat['hf'] * nu_factor

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    feat["epoch_index"] = times # added window / 2 to the times above to center the epochs on the times
    return feat


# -------------|
# EEG FEATURES |
# -------------|
def get_features_yasa_eeg(eeg_data, freq_broad=(0.4, 30), sfreq=500, welch_window_sec=4, epoch_window_sec=32, step_size=4, bands=None):
    """
    Gets ECG features using similar code & syntax as YASA's feature generation, calculates deterministic features as well as spectral features
    eeg_data: EEG raw vector data
    freq_broad: broad range frequency of EEG (this is used for "absolute power" calculations, and as a divisor for calculating overall relative power)
    sfreq: sampling frequeuency, by default 500 Hz
    epoch_window_sec: size of the epoch rolling window to use
    welch_window_sec: size of the welch window for power spectral density calculations (this affects the low frequeuncy power and very low frequency power calculations, etc.)
    step_size: how big of a step size to use, in seconds
    bands: optional parameter to override the default bands used, for exmaple if you'd like more specific bands than just sdelta, fdelta, theta, alpha, etc
    """
    # Welch keyword arguments
    kwargs_welch = dict(window="hamming", nperseg=welch_window_sec*sfreq, average="median")
    if bands is None:
        bands = [
            (0.4, 1, "sdelta"),
            (1, 4, "fdelta"),
            (4, 8, "theta"),
            (8, 12, "alpha"),
            (12, 16, "sigma"),
            (16, 30, "beta"),
        ]
    
    #  Preprocessing
    # - Filter the data
    dt_filt = filter_data(
        eeg_data, sfreq, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False
    )
    # - Extract epochs. Data is now of shape (n_epochs, n_samples).
    times, epochs = sliding_window(dt_filt, sf=sfreq, window=epoch_window_sec, step=step_size)
    times = times + epoch_window_sec // 2 # add window/2 to the times to make the epochs "centered" around the times

    window_length = sfreq*welch_window_sec
    kwargs_welch = dict(
        window='hann',
        nperseg=window_length, # a little more than  4 minutes
        noverlap=window_length//2,
        scaling='density',
        average='median'
    )

    # Calculate standard descriptive statistics
    hmob, hcomp = ant.hjorth_params(epochs, axis=1)

    feat = {
        "std": np.std(epochs, ddof=1, axis=1),
        "iqr": sp_stats.iqr(epochs, rng=(25, 75), axis=1),
        "skew": sp_stats.skew(epochs, axis=1),
        "kurt": sp_stats.kurtosis(epochs, axis=1),
        "nzc": ant.num_zerocross(epochs, axis=1),
        "hmob": hmob,
        "hcomp": hcomp,
    }

    # Calculate spectral power features (for EEG + EOG)
    freqs, psd = sp_sig.welch(epochs, sfreq, **kwargs_welch)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)

    for j, (_, _, b) in enumerate(bands):
        feat[b] = bp[j]

    # Add power ratios for EEG
    if 'sdelta' in feat:
        delta = feat["sdelta"] + feat["fdelta"]
        feat["dt"] = delta / feat["theta"]
        feat["ds"] = delta / feat["sigma"]
        feat["db"] = delta / feat["beta"]
        feat["at"] = feat["alpha"] / feat["theta"]
    
    # Add total power
    idx_broad = np.logical_and(freqs >= freq_broad[0], freqs <= freq_broad[1])
    dx = freqs[1] - freqs[0]
    feat["abspow"] = simpson(psd[:, idx_broad], dx=dx)

    # Add relative power standard deviation
    for j, (_, _, b) in enumerate(bands):
        feat[b + '_std'] = [np.std(bp[j])] * len(bp[j])

    # Add relative power bands
    for j, (_, _, b) in enumerate(bands):
        feat[b + '_relative'] = bp[j] / feat["abspow"]

    # Calculate entropy and fractal dimension features
    feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
    feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
    feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)
    feat["epoch_index"] = times

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    for col in feat.columns:
        if col != 'yasa_time':
            feat[col] = pd.to_numeric(feat[col])
    return feat
