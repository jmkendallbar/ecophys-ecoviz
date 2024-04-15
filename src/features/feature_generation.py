import feature_extraction as seal_fe # Seal feature extraction
import datetime
import json
import mne
import os
import pandas as pd
import numpy as np
import pytz
import sys

# ------------------|
# Feature Generator |
# ------------------|
DEFAULT_CONFIG = {
    'ECG Channel': 'ECG_Raw_Ch1',
    'EEG Channel': 'EEG_ICA5',
    'Pressure Channel': 'Pressure',
    'GyrZ Channel': 'GyrZ',
    'ODBA Channel': 'ODBA',
    'Start Seconds': 0, # Start time in seconds to begin feature extraction
    'End Seconds': None, # End time in seconds to stop feature extraction
    'Step Size': 1, # Step size in seconds to perform feature calculations (if 1, returned data will be 1 Hz, if 2, returned data will be 0.5 Hz)
    'Heart Rate Search Radius': 200, # Search radius in sample points (this number is affected by frequency of data) to search for R peaks in heart rate
    'Heart Rate Filter Threshold': 200, # Heart rate BPM threshold above which to throw out values and fill with surrounding values (for sensor noise)
    'Delta Power Reference Power': 1e-14, # Reference Power (in V / Hz) to divide the delta power by
    'Delta Power Calculation Window': 30, # Window size in seconds to use for delta power calculation ([index - size/2, index + size/2])
    'Zero Crossings Calculation Window': 10, # Window size in seconds to use for EEG zero crossings calculation ([index - size/2, index + size/2])
    'Absolute Power Reference Power': 1e-14, # Reference Power (in V / Hz) to divide the absolute power by
    'Absolute Power Calculation Window': 30, # Window size in seconds to use for absolute power calculation ([index - size/2, index + size/2])
    'Heart Rate Mean/STD Calculation Window': 30, # Window size in seconds to use for the heart rate mean and standard deviation calculations
    'HR VLF Power Freq Range': [0.001, 0.05], # Frequency range to use for heart rate "Very Low Frequency" Power calculation
    'HR VLF Power Calculation Window': 30, # Window size in seconds to use for the heart rate VLF Power calculation
    'HR VLF Power Reference Power': 1, # Reference Power (in V / Hz) to divide the HR VLF power by
    'HR VLF Power STD Calculation Window': 60, # Window size in seconds to use for heart rate VLF Power standard deviation
    'Pressure Freq': 500, # Frequency of Pressure data
    'Pressure Calculation Window': 30, # Window size in seconds to use for pressure mean and standard deviation
    'ODBA Freq': 500, # Frequency of ODBA data
    'ODBA Calculation Window': 30, # Window size in seconds to use for ODBA mean and standard deviation
    'GyrZ Freq': 500, # Frequency of GyrZ data
    'GyrZ Calculation Window': 30, # Window size in seconds to use for GyrZ mean and standard deviation
    'YASA Epoch Window Size': 30, # Window size in seconds to use for YASA feature epochs
    'YASA Welch Window Size': 5, # Window size in seconds to use for YASA welch power spectral density calculation
    'YASA Step Size': 15, # Step size in seconds of windows for YASA epochs
}
def generate_features(path_to_edf, output_csv_path=None, config_file_path=None, add_yasa_features=True):
    config = DEFAULT_CONFIG
    # Load Config
    if config_file_path is not None:
        config_json = json.load(config_file_path)
        for key in config_json.keys():
            if key in config.keys():
                config[key] = config_json[key]
            else:
                print(f'Unknown Config Option: {key}')
    # Read EDF
    
    raw = mne.io.read_raw_edf(path_to_edf, include=[config['ECG Channel'], config['EEG Channel'], config['Pressure Channel'],
                                                    config['ODBA Channel'], config['GyrZ Channel']], preload=True)
    sfreq = raw.info['sfreq']
    edf_start_time = raw.info['meas_date']
    if config['End Seconds'] is None:
        config['End Seconds'] = int(len(raw) / sfreq)
    # Define the PST timezone
    pst_timezone = pytz.timezone('America/Los_Angeles')
    # Convert to datetime object in PST
    if isinstance(edf_start_time, datetime.datetime):
        # If it's already a datetime object, just replace the timezone
        recording_start_datetime = edf_start_time.replace(tzinfo=None).astimezone(pst_timezone)
        # for some reason using .replace(tzinfo=...) does weird shit - offsets based of LMT instead of UTC and gets confusing
        # recording_start_datetime = edf_start_time.replace(tzinfo=pst_timezone)
    elif isinstance(edf_start_time, (int, float)):
        # Convert timestamp to datetime in PST
        recording_start_datetime = pst_timezone.localize(datetime.datetime.fromtimestamp(edf_start_time))
    datetime_range = recording_start_datetime + pd.to_timedelta(range(config['Start Seconds'], config['End Seconds'],
                                                                config['Step Size']), unit='s')
    # Get heart rates
    start_index = int(config['Start Seconds'] * sfreq)
    end_index = int(config['End Seconds'] * sfreq)
    
    print('starting heart rate')
    heart_rate = seal_fe.get_heart_rate(raw.get_data([config['ECG Channel']])[0, start_index:end_index].copy(),
                                        fs=sfreq, search_radius=config['Heart Rate Search Radius'],
                                        filter_threshold=config['Heart Rate Filter Threshold'])
    print('done', end='\n\n' + '-'*50 + '\n')

    print('starting hr_mean / hr_std')
    # Rolling mean and standard deviation of heart rate
    hr_mean, hr_std = seal_fe.get_rolling_mean_std(heart_rate, 0, len(heart_rate),
                                                   window_sec=config['Heart Rate Mean/STD Calculation Window'], freq=sfreq,
                                                   step_size=config['Step Size'])
    hr_std[hr_std == -np.inf] = 0 
    print('done', end='\n\n' + '-'*50 + '\n')
    print('starting vlf power')
    # Very low frequency (VLF) power of heart rate
    hr_vlf_power = seal_fe.get_rolling_band_power_fourier_sum(heart_rate, 0, len(heart_rate), freq_range=config['HR VLF Power Freq Range'],
                                                              window_sec=config['HR VLF Power Calculation Window'], freq=sfreq,
                                                              ref_power=config['HR VLF Power Reference Power'],
                                                              step_size=config['Step Size'])
    print('done', end='\n\n' + '-'*50 + '\n')
    # print('starting vlf power std')
    # Standard deviation of VLF power of heart rate
    # _, hr_vlf_power_std = seal_fe.get_rolling_mean_std(hr_vlf_power, 0, len(hr_vlf_power), freq=1/config['Step Size'],
    #                                                   window_sec=config['HR VLF Power STD Calculation Window'], step_size=config['Step Size'])
    # print('done', end='\n\n' + '-'*50 + '\n')
    hr_vlf_power_std = pd.Series(hr_vlf_power).rolling(config['HR VLF Power STD Calculation Window']).std()


    # Get Pressure rolling mean and standard deviation
    pressure_data = raw.get_data([config['Pressure Channel']])[0].copy()
    pressure_start_index = int(config['Start Seconds'] * config['Pressure Freq'])
    pressure_end_index = int(config['End Seconds'] * config['Pressure Freq'])
    pressure_mean, pressure_std = seal_fe.get_rolling_mean_std(pressure_data, pressure_start_index, pressure_end_index, 
                                                               freq=config['Pressure Freq'], 
                                                               window_sec=config['Pressure Calculation Window'],
                                                               step_size=config['Step Size'])
    # Get ODBA rolling mean and standard deviation
    odba_data = raw.get_data([config['ODBA Channel']])[0].copy()
    odba_start_index = int(config['Start Seconds'] * config['ODBA Freq'])
    odba_end_index = int(config['End Seconds'] * config['ODBA Freq'])
    odba_mean, odba_std = seal_fe.get_rolling_mean_std(odba_data, odba_start_index, odba_end_index, freq=config['ODBA Freq'],
                                                    window_sec=config['ODBA Calculation Window'], step_size=config['Step Size'])
    # Get GyrZ rolling mean and standard deviation
    gyrz_data = raw.get_data([config['GyrZ Channel']])[0].copy()
    gyrz_start_index = int(config['Start Seconds'] * config['ODBA Freq'])
    gyrz_end_index = int(config['End Seconds'] * config['ODBA Freq'])
    gyrz_mean, gyrz_std = seal_fe.get_rolling_mean_std(gyrz_data, gyrz_start_index, gyrz_end_index, freq=config['GyrZ Freq'],
                                                    window_sec=config['GyrZ Calculation Window'], step_size=config['Step Size'])
    # Create features dataframe
    features_df = pd.DataFrame({
        'Time': datetime_range,
        # 'Low Freq Delta Power (0.4 - 1)': eeg_low_delta_power,
        # 'High Freq Delta Power (1 - 4)': eeg_high_delta_power,
        # 'Total Delta Power (0.4 - 4)': eeg_delta_power,
        # 'Theta Power': eeg_theta_power,
        # 'Alpha Power': eeg_alpha_power,
        # 'Sigma Power': eeg_sigma_power,
        # 'Beta Power': eeg_beta_power,
        # 'Delta/Theta Power Ratio': eeg_delta_power / eeg_beta_power,
        # 'Delta/Sigma Power Ratio': eeg_delta_power / eeg_sigma_power,
        # 'Delta/Beta Power Ratio': eeg_delta_power / eeg_beta_power,
        # 'Alpha/Theta Power Ratio': eeg_alpha_power / eeg_theta_power,
        # 'Rolling Zero Crossings': eeg_zero_crossings,
        # 'Rolling Absolute Power': eeg_absolute_power,
        'Heart Rate': [heart_rate[i] for i in range(0, len(heart_rate), 500)], # Downsample from 500 Hz to 1 Hz
        'Heart Rate Mean': hr_mean,
        'Heart Rate Std.Dev': hr_std,
        'Heart Rate Very Low Frequency Power': hr_vlf_power,
        'Heart Rate VLF Power Std.Dev': hr_vlf_power_std,
        'Pressure Mean': pressure_mean,
        'Pressure Std.Dev': pressure_std,
        'ODBA Mean': odba_mean,
        'ODBA Std.Dev': odba_std,
        'GyrZ Mean': gyrz_mean,
        'GyrZ Std.Dev': gyrz_std
    })
    if add_yasa_features == True:
        print('starting yasa features')
        eeg_data = raw.get_data([config['EEG Channel']])[0].copy()
        yasa_features_eeg = seal_fe.get_features_yasa_eeg(eeg_data, start_index, end_index, freq_broad=(0.4, 30), freq=500,
                                                          welch_window_sec=config['YASA Welch Window Size'],
                                                          epoch_window_sec=config['YASA Epoch Window Size'],
                                                          step_size=config['YASA Step Size'])
        print('done', end='\n\n' + '-'*50 + '\n')
        yasa_features_eeg['yasa_time'] = recording_start_datetime + pd.to_timedelta(yasa_features_eeg['epoch_index'], unit='s')
        yasa_features_eeg = yasa_features_eeg.drop('epoch_index', axis=1)
        # Upsample yasa epochs to 1 seconds
        # Forward fill half the step size, back fill the rest (so that filled values are centered around the calculated window)
        fill_limit = config['YASA Step Size'] // 2
        yasa_features_eeg = (
            yasa_features_eeg.set_index('yasa_time', drop=True).resample('1S')
            .ffill(limit=fill_limit).bfill(limit=fill_limit+1)
        ).reset_index(names='yasa_time')
        # Add in missing indices (first epoch cuts off starting times, last epoch cuts off ending times)
        return_indices = datetime_range
        add_indices = return_indices[~return_indices.isin(yasa_features_eeg['yasa_time'])]
        yasa_feature_cols = yasa_features_eeg.columns[yasa_features_eeg.columns != 'yasa_time']
        # Fill in missing beginning and ending sections
        yasa_features_eeg = pd.concat([
            yasa_features_eeg.set_index('yasa_time', drop=True),
            pd.DataFrame(
                index=add_indices,
                columns = yasa_feature_cols,
                data=[[np.nan for i in range(len(yasa_features_eeg.columns) - 1)]] * len(add_indices))
        ], axis=0).sort_index().fillna('ffill', limit=fill_limit+1).fillna('bfill', limit=fill_limit+1).reset_index(names='yasa_time')
        yasa_cols = yasa_features_eeg.columns
        yasa_cols = ['yasa_eeg_' + colname if colname != 'yasa_time' else colname for colname in yasa_cols]
        yasa_features_eeg.columns = yasa_cols
        
        # concat feature dataframes together
        if len(yasa_features_eeg) == len(features_df):
            
            before_len = len(features_df)
            features_df = features_df.merge(yasa_features_eeg, how='outer', left_on='Time', right_on='yasa_time')
            after_len = len(features_df)
            if before_len != after_len:
                print('Some indices during yasa&feature merge did not line up. Are your YASA epoch and window sizes sensible?')
        else:
            return(features_df, yasa_features_eeg)

        # if len(yasa_features_ecg) == len(features_df):
        #     features_df = pd.concat([features_df, yasa_features_ecg], axis=1)
        # else:
        #     return((features_df, yasa_features_eeg, yasa_features_ecg))
    print('NA Values:\n', features_df.isna().sum())
    # Write to CSV
    if output_csv_path is not None:
        features_df.to_csv(output_csv_path)
    return features_df

# ---------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    if not (2 <= len(sys.argv) <= 3):
        print('Usage: python feature_generation.py <Input_EDF_Filepath> <Output_CSV_Filepath> [Config_Filepath]')
        exit(-1)
    input_edf_path = sys.argv[1]
    output_csv_path = sys.argv[2]
    config_file_path = sys.argv[3] if len(sys.argv) > 3 else None
    print(input_edf_path, output_csv_path, config_file_path, sep='\n')
    if input_edf_path[-4:] != '.edf':
        print(f'Input EDF must end in .edf')
        exit(-1)
    elif not os.path.exists(input_edf_path):
        print(f'Input EDF {input_edf_path} does not exist.')
        exit(-1)
    if output_csv_path[-4:] != '.csv':
        print('Output file name must end in .csv')
        exit(-1)
    if config_file_path is not None:
        if config_file_path[-5:] != '.json':
            print('Config file name must end in .json')
            exit(-1)
        elif not os.path.exists(config_file_path):
            print(f'Config file {config_file_path} does not exist.')
            exit(-1)
    generate_features(input_edf_path, output_csv_path, config_file_path)
