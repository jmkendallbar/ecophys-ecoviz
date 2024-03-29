import yasa
import mne
#import os
import scipy
#import glob
#import six
import wfdb
#import pytz
#import sklearn
#import pomegranate
#import pyedflib
#import sleepecg
import datetime
import wfdb.processing
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#import entropy as ent
import random
import seaborn as sns
from matplotlib import mlab as mlab
from sleepecg import detect_heartbeats
from scipy.integrate import simpson
from scipy.signal import hann, welch
#import matplotlib.gridspec as gs

# -----------------|
# COMMON FUNCTIONS |
# -----------------|

# MAYBE?
def get_features_yasa(a, start_index, end_index, freq_broad=(0.4, 30), freq=500, welch_window_sec=5, epoch_window_sec=30, step_size=15):
    import os
    import mne
    import glob
    import joblib
    import logging
    import numpy as np
    import pandas as pd
    import antropy as ant
    import scipy.signal as sp_sig
    import scipy.stats as sp_stats
    import matplotlib.pyplot as plt
    from mne.filter import filter_data
    from sklearn.preprocessing import robust_scale

    from yasa import sliding_window
    from yasa import bandpower_from_psd_ndarray
    # Welch keyword arguments

    kwargs_welch = dict(window="hamming", nperseg=welch_window_sec*freq, average="median")
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
        a[start_index:end_index], freq, l_freq=freq_broad[0], h_freq=freq_broad[1], verbose=False
    )
    # - Extract epochs. Data is now of shape (n_epochs, n_samples).
    times, epochs = sliding_window(dt_filt, sf=freq, window=epoch_window_sec, step=step_size)

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
    freqs, psd = sp_sig.welch(epochs, freq, **kwargs_welch)
    bp = bandpower_from_psd_ndarray(psd, freqs, bands=bands)

    for j, (_, _, b) in enumerate(bands):
        feat[b] = bp[j]

    # Add power ratios for EEG
    delta = feat["sdelta"] + feat["fdelta"]
    feat["dt"] = delta / feat["theta"]
    feat["ds"] = delta / feat["sigma"]
    feat["db"] = delta / feat["beta"]
    feat["at"] = feat["alpha"] / feat["theta"]

    # Add total power
    idx_broad = np.logical_and(freqs >= freq_broad[0], freqs <= freq_broad[1])
    dx = freqs[1] - freqs[0]
    feat["abspow"] = np.trapz(psd[:, idx_broad], dx=dx)

    # Calculate entropy and fractal dimension features
    feat["perm"] = np.apply_along_axis(ant.perm_entropy, axis=1, arr=epochs, normalize=True)
    feat["higuchi"] = np.apply_along_axis(ant.higuchi_fd, axis=1, arr=epochs)
    feat["petrosian"] = ant.petrosian_fd(epochs, axis=1)
    feat["epoch_index"] = times + epoch_window_sec // 2 # Add window/2 so the epoch is centered around the time value

    # Convert to dataframe
    feat = pd.DataFrame(feat)
    for col in feat.columns:
        if col != 'yasa_time':
            feat[col] = pd.to_numeric(feat[col])
    return feat


    
# ---------------------------------------------------------------------------------------------------------------------------------------
# Use this for delta power and absolute power
def get_rolling_band_power_welch(a, start_index, end_index, freq_range=(0.5, 4), ref_power=0.001, freq=500, window_sec=2, step_size=1):
    """
    Gets rolling band power for specified frequency range, data frequency and window size
    a: array to calculate delta power 
    start_index: start index in the input array a to start calculating delta power
    end_index: end index in the input array a to stop calculating delta power
    freq_range: range of frequencies in form of (lower, upper) to calculate power of
    ref_power: arbitrary reference power to divide the windowed delta power by (used for scaling)
    freq: frequency of the input array
    window_sec: window size in seconds to calculate delta power (if the window is longer than the step size there will be overlap)
    step_size: step size in seconds to calculate delta power in windows (if 1, function returns an array with 1Hz power calculations)

    Example: get_rolling_band_power(eeg_data, 0, len(eeg_data), freq_range=(0.5, 4), ref_power=0.001, freq=500, window_sec=2, step_size=1)
    """
    def get_band_power_welch(a, start_index, end_index, freq_range=(0.5, 4), ref_power=0.001, freq=500):
        lower_freq = freq_range[0]
        upper_freq = freq_range[1]
        window_length = int(window_sec*freq)
        # TODO: maybe edit this later so there is a buffer before and after?
        windowed_data = a[start_index:end_index] * hann(window_length)
        freqs, psd = welch(windowed_data, window='hann', fs=freq, nperseg=window_length, noverlap=window_length//2)
        freq_res = freqs[1] - freqs[0]
        # Find the index corresponding to the delta frequency range
        delta_idx = (freqs >= lower_freq) & (freqs <= upper_freq)
        # Integral approximation of the spectrum using parabola (Simpson's rule)
        delta_power = simpson(10 * np.log10(psd[delta_idx] / ref_power), dx=freq_res)
        # Sum the power within the delta frequency range
        return delta_power
    
    # Get rolling delta power using the helper function above
    window_length = window_sec*freq
    step_idx = int(step_size*freq) # step size in array indices
    if window_length % 2 != 0:
        print('Note: (window * frequency) is not divisible by 2, will be rounded down to ' + str(window_length - 1))
    rolling_band_power = []
    for i in range(start_index, end_index, step_idx):
        window_start = int(i - window_length/2)
        window_end = int(i + window_length/2)
        if window_start < 0:
            rolling_band_power.append(np.nan)
            # print(f'Window too large for index {i}; not enough preceding point')
        elif window_end > len(a):
            rolling_band_power.append(np.nan)
            # print(f'Window too large for index {i}; not enough following point')
        else:
            rolling_band_power.append(get_band_power_welch(a, window_start, window_end, freq_range=freq_range, ref_power=ref_power, freq=freq))
    return np.array(rolling_band_power)


# ---------------------------------------------------------------------------------------------------------------------------------------
# For some reason this works better for low frequency power (0.001, 0.05) than welch
def get_rolling_band_power_fourier_sum(a, start_index, end_index, freq_range=(0.001, 0.05), ref_power=0.001, freq=500, window_sec=60, 
                                       step_size=1):
    """
    Gets rolling band power for specified frequency range, data frequency and window size
    a: array to calculate delta power 
    start_index: start index in the input array a to start calculating delta power
    end_index: end index in the input array a to stop calculating delta power
    freq_range: range of frequencies in form of (lower, upper) to calculate power of
    ref_power: arbitrary reference power to divide the windowed delta power by (used for scaling)
    freq: frequency of the input array
    window_sec: window size in seconds to calculate delta power (if the window is longer than the step size there will be overlap)
    step_size: step size in seconds to calculate delta power in windows (if 1, function returns an array with 1Hz power calculations)

    Example: get_rolling_band_power(eeg_data, 0, len(eeg_data), freq_range=(0.5, 4), ref_power=0.001, freq=500, window_sec=2, step_size=1)
    """

    def get_band_power_fourier_sum(a):
        """
        Helper function to get delta spectral power for one array
        """
        # Perform Fourier transform
        fft_data = np.fft.fft(a)
        # Compute the power spectrum
        power_spectrum = np.abs(fft_data)**2
        # Frequency resolution
        freq_resolution = freq / len(a)
        # Find the indices corresponding to the delta frequency range
        delta_freq_indices = np.where((np.fft.fftfreq(len(a), 1/freq) >= freq_range[0]) & 
                                      (np.fft.fftfreq(len(a), 1/freq) <= freq_range[1]))[0]
        # Compute the delta spectral power
        delta_power = 10 * np.log(np.sum(power_spectrum[delta_freq_indices]) * freq_resolution / ref_power)

        return delta_power

    window_length = window_sec*freq
    step_idx = int(step_size*freq)
    if window_length % 2 != 0:
        print('Note: (window * frequency) is not divisible by 2, will be rounded down to ' + str(window_length - 1))
    rolling_band_power = []
    for i in range(start_index, end_index, step_idx):
        window_start = int(i - window_length/2)
        window_end = int(i + window_length/2)
        if window_start < 0:
            rolling_band_power.append(np.nan)
            # print(f'Window too large for index {i}; not enough preceding point')
        elif window_end > len(a):
            rolling_band_power.append(np.nan)
            # print(f'Window too large for index {i}; not enough following point')
        else:
            rolling_band_power.append(get_band_power_fourier_sum(a[window_start:window_end]))
    return np.array(rolling_band_power)

# ---------------------------------------------------------------------------------------------------------------------------------------
# https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho
def get_rolling_mean_std(a, start_index, end_index, freq=500, window_sec=30, step_size=1):
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
    for i in range(start_index, end_index, step_idx):
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
    # there is probably a function out here that does this but just imputing with closest non-na direct left and direct right neighbor
    # i = 0
    # start_fill = -1
    # end_fill = -1
    # count_filled = 0
    # while i < len(hr_data):
    #     if hr_data[i] == hr_data[i]:
    #         if end_fill != -1:
    #             # calculate window of previous 5 seconds and following 5 seconds, and sample from that
    #             # do this so that the heart rate standard deviation isn't 0, this makes downstream features np.nan or -np.inf
    #             offset = int(fs * 5)
    #             if start_fill >= offset:
    #                 window_prev = hr_data[start_fill-offset:start_fill]
    #             else:
    #                 window_prev = hr_data[0:start_fill]
    #             if end_fill+1 < len(hr_data) - offset:
    #                 window_after = hr_data[end_fill+1:end_fill+1+offset]
    #             else:
    #                 window_after = hr_data[end_fill+1:len(hr_data)]
    #             windows_combined = pd.Series(np.concatenate([window_prev, window_after])).dropna()
    #             hr_data[start_fill:end_fill] = windows_combined.sample(end_fill - start_fill, replace=True).to_list()
    #             # hr_data[start_fill:end_fill] = np.mean([hr_data[start_fill-1], hr_data[end_fill+1]])
    #             count_filled += (end_fill - start_fill)
    #             start_fill = -1
    #             end_fill = -1
    #     else:
    #         if start_fill == -1:
    #             start_fill = i
    #         else:
    #             end_fill = i
    #     i += 1
    print('Filled:', filled)
    return hr_data

# ---------------------------------------------------------------------------------------------------------------------------------------

# -------------|
# EEG FEATURES |
# -------------|

# ---------------------------------------------------------------------------------------------------------------------------------------
def get_rolling_zero_crossings(a, start_index, end_index, freq=500, window_sec=1, step_size=1):
    """
    Get the zero-crossings of an array with a rolling window
    a: array to find zero crossings for
    start_index: index of the start position
    end_index: index of the end position (exclusive)
    window_sec: window in seconds
    step_size: step size in seconds (step_size of 1 would mean returend data will be 1 Hz)

    Example: get_rolling_zero_crossings(eeg_data, 0, len(eeg_data), freq=500, window_sec=1, step_size=1)
    """
    window_length = window_sec*freq
    step_idx = int(step_size * freq)
    if window_length % 2 != 0:
        print('Note: (window * frequency) is not divisible by 2, will be rounded down to ' + str(window_length - 1))
    rolling_zero_crossings = []
    for i in range(start_index, end_index, step_idx):
        window_start = int(i - window_length/2)
        window_end = int(i + window_length/2)
        if window_start < 0:
            #window_start = 0
            rolling_zero_crossings.append(np.nan)
            # print(f'Window too large for index {i}; not enough preceding point')
        elif window_end > len(a):
            #window_end = len(a)
            rolling_zero_crossings.append(np.nan)
            # print(f'Window too large for index {i}; not enough following point')
        else:
            rolling_zero_crossings.append(((a[window_start:window_end-1] * a[window_start+1:window_end]) < 0).sum() / window_sec)
    return np.array(rolling_zero_crossings)

# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------

# OLD STUFF
# This function is too slow although it performs slightly better
def get_rolling_band_power_multitaper(a, start_index, end_index, freq_range=(0.5, 4), ref_power=1e-13, freq=500, window_sec=2,
                                      step_size=1, in_dB=True):
    """
    Gets rolling band power for specified frequency range, data frequency and window size
    a: array to calculate delta power 
    start_index: start index in the input array a to start calculating delta power
    end_index: end index in the input array a to stop calculating delta power
    freq_range: range of frequencies in form of (lower, upper) to calculate power of
    ref_power: arbitrary reference power to divide the windowed delta power by (used for scaling)
    freq: frequency of the input array
    window_sec: window size in seconds to calculate delta power (if the window is longer than the step size there will be overlap)
    step_size: step size in seconds to calculate delta power in windows (if 1, function returns an array with 1Hz power calculations)
    in_dB: boolean for whether to convert the output into decibals

    Example: get_rolling_band_power_multitaper(eeg_data, 0, len(eeg_data), freq_range=(0.5, 4), ref_power=0.001) # get delta frequency
    """
    def get_band_power_multitaper(a, start_index, end_index):

        # TODO: maybe edit this later so there is a buffer before and after?
        psd, freqs = mne.time_frequency.psd_array_multitaper(a[start_index:end_index], sfreq=freq,
                                                                   fmin=freq_range[0], fmax=freq_range[1], adaptive=True, 
                                                                   normalization='full', verbose=False)
        freq_res = freqs[1] - freqs[0]
        # Find the index corresponding to the delta frequency range
        delta_idx = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        # Integral approximation of the spectrum using parabola (Simpson's rule)
        delta_power = psd[delta_idx] / ref_power
        if in_dB:
            delta_power = simpson(10 * np.log10(delta_power), dx=freq_res)
        else:
            delta_power = np.mean(delta_power)
        # Sum the power within the delta frequency range
        return delta_power

    window_length = window_sec*freq
    step_idx = int(step_size*freq)
    if window_length % 2 != 0:
        print('Note: (window * frequency) is not divisible by 2, will be rounded down to ' + str(window_length - 1))
    rolling_band_power = []
    for i in range(start_index, end_index, step_idx):
        window_start = int(i - window_length/2)
        window_end = int(i + window_length/2)
        if window_start < 0:
            rolling_band_power.append(np.nan)
            # print(f'Window too large for index {i}; not enough preceding point')
        elif window_end > len(a):
            rolling_band_power.append(np.nan)
            # print(f'Window too large for index {i}; not enough following point')
        else:
            rolling_band_power.append(get_band_power_multitaper(a, window_start, window_end))
    return np.array(rolling_band_power)