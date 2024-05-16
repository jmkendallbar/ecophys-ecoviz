from argparse import ArgumentParser
import datetime
import json
import mne
import numpy as np
import os
import pandas as pd
import pytz
import warnings
warnings.simplefilter('ignore')

# custom functions
if __name__ != '__main__' and 'src' in __name__: # when being imported from the context of the src package
    import src.features.feature_generation_utils as seal_fe # Seal feature extraction
else:
    import feature_generation_utils as seal_fe # Seal feature extraction

# ------------------|
# Feature Generator |
# ------------------|
DEFAULT_CONFIG = {
    'ECG Channel': 'ECG_Raw_Ch1', # ECG Channel
    'EEG Channel': 'EEG_ICA5', # EEG Channel
    'Pressure Channel': 'Pressure', # Pressure/Depth
    'GyrZ Channel': 'GyrZ', # Gyroscope Z
    'ODBA Channel': 'ODBA', # Overall dynamic body acceleration
    'Step Size': 1, # Step size in seconds to perform feature calculations (if 1, returned data will be 1 Hz, if 2, returned data will be 0.5 Hz)
    'Heart Rate Search Radius': 200, # Search radius in sample points (this number is affected by frequency of data) to search for R peaks in heart rate
    'Heart Rate Filter Threshold': 200, # Heart rate BPM threshold above which to throw out values and fill with surrounding values (for sensor noise)
    'Pressure Freq': 25, # Frequency of Pressure data
    'Pressure Calculation Window': 30, # Window size in seconds to use for pressure mean and standard deviation
    'ODBA Freq': 25, # Frequency of ODBA data
    'ODBA Calculation Window': 30, # Window size in seconds to use for ODBA mean and standard deviation
    'GyrZ Freq': 25, # Frequency of GyrZ data
    'GyrZ Calculation Window': 30, # Window size in seconds to use for GyrZ mean and standard deviation
    'YASA EEG Epoch Window Size': 30, # Window size in seconds to use for YASA feature epochs
    'YASA EEG Welch Window Size': 4, # Window size in seconds to use for YASA welch power spectral density calculation
    'YASA EEG Step Size': 1, # Step size in seconds of windows for YASA epochs
    'YASA Heart Rate Epoch Window Size': 60, # Window size in seconds to use for YASA heart rate feature epochs
    'YASA Heart Rate Welch Window Size': 4, # Window size in seconds to use for YASA heart rate welch power spectral density calculation
    'YASA Heart Rate Step Size': 1, # Step size in seconds of windows for YASA heart rate epochs
}
def generate_features(path_to_edf, output_csv_path=None, custom_config=None):
    """
    This is the main function for generating features given a path to an EDF file
    path_to_edf: absolute or relative path to the EDF file
    output_csv_path: output path to save a .csv file containing the features, if None, only returns the features
    custom_config: optional variable to provide custom config features. This is expected to either be a python dictionary or a path to a JSON file.
                   Any config values not specified will use the defaults in the above DEFAULT_CONFIG dictionary
    """
    # Load Config
    config = DEFAULT_CONFIG
    if custom_config is not None:
        # if config is a string, attempt to load the config from file
        if type(custom_config) is str:
            config_json = json.load(config)
            for key in config_json.keys():
                if key in config.keys():
                    config[key] = config_json[key]
                else:
                    print(f'Unknown Config Option: {key}')
        # if config is a dict, load default config then replace the default values with all values specified in config
        elif type(custom_config) is dict:
            for key, val in custom_config.items():
                if key in config.keys():
                    config[key] = val
                else:
                    print(f'Unknown Config Option: {key}')
        else:
            print('Unknown type for parameter custom_config, using default')
    # Read EDF
    
    raw = mne.io.read_raw_edf(path_to_edf, include=[config['ECG Channel'], config['EEG Channel'], config['Pressure Channel'],
                                                    config['ODBA Channel'], config['GyrZ Channel']], preload=False)
    sfreq = raw.info['sfreq']
    edf_start_time = raw.info['meas_date']

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

    ecg_raw = mne.io.read_raw_edf(path_to_edf, include=config['ECG Channel'], preload=True, verbose=False).get_data(config['ECG Channel'])[0]
    datetime_range = recording_start_datetime + pd.to_timedelta(range(0, len(ecg_raw) // 500, 1), unit='s')
    # Get heart rates
    print('\n' + '-'*50)
    print('starting heart rate')
    heart_rate = seal_fe.get_heart_rate(ecg_raw, fs=sfreq, search_radius=config['Heart Rate Search Radius'],
                                        filter_threshold=config['Heart Rate Filter Threshold'])
    del(ecg_raw) # delete raw to free up memory
    print('done', end='\n\n' + '-'*50 + '\n')

    # Create features dataframe
    features_dict = {
        'Time': datetime_range,
        'Heart Rate': [heart_rate[i] for i in range(0, len(heart_rate), 500)], # Downsample from 500 Hz to 1 Hz
    }

    if config['Pressure Channel'] is not None:
        if config['Pressure Channel'] in raw.ch_names:
            # Get Pressure rolling mean and standard deviation
            pressure_raw = mne.io.read_raw_edf(path_to_edf, include=config['Pressure Channel'], preload=True, verbose=False).get_data(config['Pressure Channel'])[0]
            pressure_mean, pressure_std = seal_fe.get_rolling_mean_std(pressure_raw,
                                                                    freq=config['Pressure Freq'], 
                                                                    window_sec=config['Pressure Calculation Window'],
                                                                    step_size=config['Step Size'])
            features_dict['Pressure Mean'] = pressure_mean
            features_dict['Pressure Std.Dev'] = pressure_std
            del(pressure_raw) # delete raw to free up memory
        else:
            print(config['Pressure Channel'], 'not found in the channels of the input edf file, excluding...')

    if config['ODBA Channel'] is not None:
        if config['ODBA Channel'] in raw.ch_names:
            # Get ODBA rolling mean and standard deviation
            odba_raw = mne.io.read_raw_edf(path_to_edf, include=config['ODBA Channel'], preload=True, verbose=False).get_data(config['ODBA Channel'])[0]
            odba_mean, odba_std = seal_fe.get_rolling_mean_std(odba_raw, freq=config['ODBA Freq'], window_sec=config['ODBA Calculation Window'], step_size=config['Step Size'])
            features_dict['ODBA Mean'] = odba_mean
            features_dict['ODBA Std.Dev'] = odba_std
            del(odba_raw) # delete raw to free up memory
        else:
            print(config['ODBA Channel'], 'not found in the channels of the input edf file, excluding...')
    
    if config['GyrZ Channel'] is not None:
        if config['GyrZ Channel'] in raw.ch_names:
            # Get GyrZ rolling mean and standard deviation
            gyrz_raw = mne.io.read_raw_edf(path_to_edf, include=config['GyrZ Channel'], preload=True, verbose=False).get_data(config['GyrZ Channel'])[0]
            gyrz_mean, gyrz_std = seal_fe.get_rolling_mean_std(gyrz_raw, freq=config['GyrZ Freq'], window_sec=config['GyrZ Calculation Window'], step_size=config['Step Size'])
            features_dict['GyrZ Mean'] = gyrz_mean
            features_dict['GyrZ Std.Dev'] = gyrz_std
            del(gyrz_raw) # delete raw to free up memory
        else:
            print(config['GyrZ Channel'], 'not found in the channels of the input edf file, excluding...')

    print('Starting Features (and lengths of feature vectors):')
    for key, val in features_dict.items():
        print(key, len(val))
    print('-'*50)
    features_df = pd.DataFrame(features_dict)

    print('starting yasa EEG features')
    eeg_raw = mne.io.read_raw_edf(path_to_edf, include=[config['EEG Channel']], preload=True, verbose=False).get_data(config['EEG Channel'])[0]
    yasa_features_eeg = seal_fe.generate_features_sequential(eeg_raw, recording_start_datetime, data_type='EEG', sfreq=500, window_length_sec=3600, buffer_length_sec=300,
                                                             epoch_size_sec=config['YASA EEG Epoch Window Size'],
                                                             welch_window_sec=config['YASA EEG Welch Window Size'],
                                                             step_size=config['YASA EEG Step Size'])
    del(eeg_raw)
    print('done', end='\n\n' + '-'*50 + '\n')

    # Add yasa_eeg prefix to separate from yasa heart rate features
    yasa_eeg_cols = yasa_features_eeg.columns
    yasa_eeg_cols = ['yasa_eeg_' + colname if colname != 'yasa_time' else colname for colname in yasa_eeg_cols]
    yasa_features_eeg.columns = yasa_eeg_cols


    print('starting yasa heart rate features')
    yasa_features_heartrate = seal_fe.generate_features_sequential(heart_rate.values, recording_start_datetime, data_type='HR', sfreq=500,
                                                                   epoch_size_sec=config['YASA Heart Rate Epoch Window Size'],
                                                                   welch_window_sec=config['YASA Heart Rate Welch Window Size'],
                                                                   step_size=config['YASA Heart Rate Step Size'])
    print('done', end='\n\n' + '-'*50 + '\n')

    # Add yasa_heartrate prefix to separate from yasa eeg features
    yasa_heartrate_cols = yasa_features_heartrate.columns
    yasa_heartrate_cols = ['yasa_heartrate_' + colname if colname != 'yasa_time' else colname for colname in yasa_heartrate_cols]
    yasa_features_heartrate.columns = yasa_heartrate_cols

    features_df = features_df.set_index('Time', drop=True).sort_index()
    # concat feature dataframes together
    if len(yasa_features_eeg) == len(features_df) and len(yasa_features_heartrate) == len(features_df):
        
        before_len = len(features_df)
        features_df = features_df.merge(yasa_features_eeg, how='outer', left_index=True, right_index=True)
        features_df = features_df.merge(yasa_features_heartrate, how='outer', left_index=True, right_index=True)
        after_len = len(features_df)
        if before_len != after_len:
            print('Some indices during yasa&feature merge did not line up. Are your YASA epoch and window sizes sensible?')
    else:
        return(features_df, yasa_features_eeg, yasa_features_heartrate)

    print('NA Values:\n', features_df.isna().sum())
    # Write to CSV
    if output_csv_path is not None:
        print(f'Writing to output file: {output_csv_path}')
        features_df.to_csv(output_csv_path)
    return features_df

def generate_features_separated(path_to_raw_folder, seal_name, output_csv_path=None, custom_config=None):
    """
    Helper function to generate features for if the input edf file is split into multiple days. First concatenates them and saves the edf, then generates features
    """
    def concatenate_raws(folder, seal_name):
        """
        Function to concatenate raw edf objects into one file
        """
        folder_files = os.listdir(folder)
        seal_files = sorted([file for file in folder_files if seal_name in file and 'DAY' in file])
        
        # Define the PST timezone
        pst_timezone = pytz.timezone('America/Los_Angeles')
        
        # Define array to hold raw objects and array to hold start datetimes
        raws = []
        start_datetimes = []
        for seal_file in seal_files:
            print(seal_file)
            raw = mne.io.read_raw_edf(f'{folder}/{seal_file}',
                                    include=['ECG_Raw_Ch1', 'REEG2_Pruned_Ch7', 'GyrZ', 'MagZ', 'Depth'],
                                    preload=True, verbose=False)
            fs = raw.info['sfreq']
            duration = len(raw) / fs
            
            # Extract the measurement date (start time) from raw.info
            start_time = raw.info['meas_date']

            # Convert to datetime object in PST
            if isinstance(start_time, datetime.datetime):
                # If it's already a datetime object, just replace the timezone
                recording_start_datetime = start_time.replace(tzinfo=None).astimezone(pst_timezone)
            elif isinstance(start_time, (int, float)):
                # Convert timestamp to datetime in PST
                recording_start_datetime = pst_timezone.localize(datetime.datetime.fromtimestamp(start_time))
            print('Start:', recording_start_datetime)
            print('End:', recording_start_datetime + datetime.timedelta(seconds=duration))
            start_datetimes.append(recording_start_datetime)
            raws.append(raw)
            print()
            
        # sort raw objects by their start datetime
        raws = [raw for _, raw in sorted(zip(start_datetimes, raws))]

        return mne.concatenate_raws(raws)
    print('Concatenating', seal_name)
    path_to_edf_folder = path_to_raw_folder + '/01_edf_data'
    seal_raw = concatenate_raws(path_to_edf_folder, seal_name)
    mne.export.export_raw(f'{path_to_edf_folder}/{seal_name}_05_ALL_PROCESSED.edf', seal_raw, fmt='edf',
                        overwrite=True)
    del(seal_raw)
    print('Calculating features for', seal_name)
    features_df = generate_features(
        f'{path_to_edf_folder}/{seal_name}_ALL_PROCESSED.edf',
        output_csv_path=output_csv_path,
        custom_config=custom_config
    )
    path_to_hypnogram_folder = path_to_raw_folder + '/02_hypnogram_data'
    labels = pd.read_csv(f'{path_to_hypnogram_folder}/{seal_name}_06_Hypnogram_JKB_1Hz.csv')
    labels['R.Time'] = pd.to_datetime(labels['R.Time']).dt.tz_localize('America/Los_Angeles')
    labels = labels.set_index('R.Time', drop=True)[['Simple.Sleep.Code']]
    features_df = features_df.merge(labels, how='outer', left_index=True, right_index=True)
    features_df.to_csv(f'/data/process/features/{seal_name}_07_features_with_labels.csv')
    features_df.name = f'{seal_name}_features'
    return features_df

# ---------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    """
    For help strings, run python feature_generation.py --help
    """
    WED_INPUT_FILE = 'data/raw/01_edf_data/test12_Wednesday_05_ALL_PROCESSED.edf'
    WED_OUTPUT_FILE = 'data/processed/features/test12_Wednesday_07_features_with_labels.csv'
    WED_LABELS_FILE = 'data/raw/02_hypnogram_data/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv'
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, help="input .edf filepath",
                        default=WED_INPUT_FILE)
    parser.add_argument("-o", "--output", dest="output", type=str, help="output .csv filepath",
                        default=WED_OUTPUT_FILE)
    parser.add_argument("-l", "--labels", dest="labels", type=str, default=None,
                        help="optional .csv filepath with 1Hz labels")
    parser.add_argument("-c", "--config", dest="config", type=str, default=None,
                        help="JSON configuration for feature calculation hyperparameters")

    args = parser.parse_args()
    if args.input == WED_INPUT_FILE and args.output == WED_OUTPUT_FILE:
        args.labels = WED_LABELS_FILE

    # Check parameters point to actual files
    if args.input[-4:] != '.edf':
        print(f'Input EDF must end in .edf')
        exit(-1)
    elif not os.path.exists(args.input):
        print(f'Input EDF {args.input} does not exist.')
        exit(-1)
    if args.output[-4:] != '.csv':
        print(f'Output file {args.output} must end in .csv')
        exit(-1)
    if args.labels is not None:
        if args.labels[-4:] != '.csv':
            print(f'Config file {args.labels} must end in .csv')
            exit(-1)
        elif not os.path.exists(args.labels):
            print(f'Config file {args.labels} does not exist.')
            exit(-1)
    if args.config is not None:
        if args.config[-5:] != '.json':
            print(f'Config file {args.config} must end in .json')
            exit(-1)
        elif not os.path.exists(args.config):
            print(f'Config file {args.config} does not exist.')
            exit(-1)
    
    # generate features, attach labels, save to file
    if args.labels is not None:
        features_df = generate_features(args.input, output_csv_path=None, custom_config=args.config)
        labels = pd.read_csv(args.labels)
        labels['R.Time'] = pd.to_datetime(labels['R.Time']).dt.tz_localize('America/Los_Angeles')
        labels = labels.set_index('R.Time', drop=True)[['Simple.Sleep.Code']]
        features_df = features_df.merge(labels, how='outer', left_index=True, right_index=True)
        features_df.to_csv(args.output)
    
    # generate features, save to file
    else:
        generate_features(args.input, output_csv_path=args.output, custom_config=args.config)
