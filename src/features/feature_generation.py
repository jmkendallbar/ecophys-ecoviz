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
    'YASA EEG Epoch Window Size': 32, # Window size in seconds to use for YASA feature epochs
    'YASA EEG Welch Window Size': 4, # Window size in seconds to use for YASA welch power spectral density calculation
    'YASA EEG Step Size': 4, # Step size in seconds of windows for YASA epochs
    'YASA Heart Rate Epoch Window Size': 64, # Window size in seconds to use for YASA heart rate feature epochs
    'YASA Heart Rate Welch Window Size': 16, # Window size in seconds to use for YASA heart rate welch power spectral density calculation
    'YASA Heart Rate Step Size': 8, # Step size in seconds of windows for YASA heart rate epochs
}
def generate_features(path_to_edf, output_csv_path=None, custom_config=None, hours_at_a_time=1):
    """
    This is the main function for generating features given a path to an EDF file
    path_to_edf: absolute or relative path to the EDF file
    output_csv_path: output path to save a .csv file containing the features, if None, only returns the features
    custom_config: optional variable to provide custom config features. This is expected to either be a python dictionary or a path to a JSON file.
                   Any config values not specified will use the defaults in the above DEFAULT_CONFIG dictionary
    hours_at_a_time: how many hours at a time to calculate eeg and heart rate features. This is a high memory process, so change this to work for your machine
    """
    # Load Config
    config = dict(DEFAULT_CONFIG)
    if custom_config is not None:
        # if config is a string, attempt to load the config from file
        if type(custom_config) is str:
            with open(custom_config) as f:
                config_json = json.load(f)
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
    if type(path_to_edf) == str:
        raw = mne.io.read_raw_edf(path_to_edf, include=[config['ECG Channel'], config['EEG Channel'], config['Pressure Channel'],
                                                    config['ODBA Channel'], config['GyrZ Channel']], preload=False)
    if type(path_to_edf) == mne.io.edf.edf.RawEDF:
        raw = path_to_edf
        print('Passed in MNE Raw EDF object, generating features from raw object')
    
    sfreq = int(raw.info['sfreq'])
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

    if type(path_to_edf) == mne.io.edf.edf.RawEDF:
        ecg_raw = raw.get_data(config['ECG Channel'])[0]
    else:
        ecg_raw = mne.io.read_raw_edf(path_to_edf, include=config['ECG Channel'], preload=True, verbose=False).get_data(config['ECG Channel'])[0]
    datetime_range = recording_start_datetime + pd.to_timedelta(range(0, len(ecg_raw) // sfreq, 1), unit='s')
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
        'Heart Rate': [heart_rate[i] for i in range(0, len(heart_rate), sfreq)], # Downsample from 500 Hz to 1 Hz
    }

    if config['Pressure Channel'] is not None:
        if config['Pressure Channel'] in raw.ch_names:
            # Get Pressure rolling mean and standard deviation
            if type(path_to_edf) == mne.io.edf.edf.RawEDF:
                pressure_raw = raw.get_data(config['Pressure Channel'])[0]
            else:
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
            if type(path_to_edf) == mne.io.edf.edf.RawEDF:
                odba_raw = raw.get_data(config['ODBA Channel'])[0]
            else:
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
            if type(path_to_edf) == mne.io.edf.edf.RawEDF:
                gyrz_raw = raw.get_data(config['GyrZ Channel'])[0]
            else:
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
    if type(path_to_edf) == mne.io.edf.edf.RawEDF:
        eeg_raw = raw.get_data(config['EEG Channel'])[0]
    else:
        eeg_raw = mne.io.read_raw_edf(path_to_edf, include=[config['EEG Channel']], preload=True, verbose=False).get_data(config['EEG Channel'])[0]
    yasa_features_eeg = seal_fe.generate_features_sequential(eeg_raw, recording_start_datetime, data_type='EEG', sfreq=500,
                                                             window_length_sec=hours_at_a_time*60*60, buffer_length_sec=300,
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
                                                                   window_length_sec=hours_at_a_time*60*60, buffer_length_sec=300,
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

def generate_features_separated(path_to_edf_folder, seal_name, output_folder, labels_df=None, custom_config=None):
    """
    Helper function to generate features for if the input edf file is split into multiple days. First concatenates them and saves the edf, then generates features
    """
    os.makedirs(os.path.join(output_folder, seal_name), exist_ok=True)
    # Function to load DAY1, DAY 2, ... edfs and join them into one ALL edf
    def concatenate_raws(path_to_edf_folder, seal_name):
        folder_files = os.listdir(path_to_edf_folder)
        seal_files = sorted([file for file in folder_files if seal_name in file and 'DAY' in file])
        
        # Define the PST timezone
        pst_timezone = pytz.timezone('America/Los_Angeles')
        
        # Define array to hold raw objects and array to hold start datetimes
        raws = []
        start_datetimes = []
        for seal_file in seal_files:
            print(seal_file)
            input_file = os.path.join(path_to_edf_folder, seal_file)
            raw = mne.io.read_raw_edf(input_file,
                                      include=['ECG_Raw_Ch1', 'REEG2_Pruned_Ch7', 'GyrZ', 'MagZ', 'ODBA', 'Depth'],
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
            print('Start:', recording_start_datetime.strftime('%F %T.%f')[:-3])
            print('End:', (recording_start_datetime + datetime.timedelta(seconds=duration)).strftime('%F %T.%f')[:-3])
            start_datetimes.append(recording_start_datetime)
            raws.append(raw)
            print()
            
        # sort raw objects by their start datetime
        raws = [(raw, start_datetime) for start_datetime, raw in sorted(zip(start_datetimes, raws))]
        
        first_start_datetime = raws[0][1]
        
        # crop raw objects so there is no overlap
        for i in range(len(raws)-1):
            raw, start_datetime = raws[i]
            duration = (raws[i+1][1] - start_datetime).total_seconds()
            raw.crop(0, duration, include_tmax=False)
        
        # crop ending filled flatline data from final raw
        last_raw_ecg = raws[-1][0].get_data('ECG_Raw_Ch1')[0].copy()
        final_val = last_raw_ecg[-1]
        last_flatline_idx = len(last_raw_ecg) - 1
        for i in range(len(last_raw_ecg)-1, 0, -1):
            if last_raw_ecg[i] == final_val:
                last_flatline_idx = i
            else:
                break
        
        # remove start datetimes from raws array
        raws = [raw[0] for raw in raws]
        # remove flatline end
        raws[-1].crop(0, last_flatline_idx//500, include_tmax=False)
        
        concatted_raw = mne.concatenate_raws(raws)
        concatted_raw.set_meas_date(first_start_datetime.replace(tzinfo=datetime.timezone.utc))
        return concatted_raw

    print('Concatenating', seal_name)
    seal_raw = concatenate_raws(path_to_edf_folder, seal_name)
    print('Calculating features for', seal_name)
    config = dict(DEFAULT_CONFIG)

    # set frequencies to 500 Hz because we are using a concatenated raw object, so everything gets upsampled to the max frequency
    config['Pressure Freq'] = 500
    config['ODBA Freq'] = 500
    config['GyrZ Freq'] = 500

    welch_epoch_changed = False
    if custom_config is not None:
        # if config is a string, attempt to load the config from file
        if type(custom_config) is str:
            with open(custom_config) as f:
                config_json = json.load(f)
                for key in config_json.keys():
                    if key in config.keys():
                        if ('Welch' in key or 'Epoch' in key) and config[key] != config_json[key]:
                            welch_epoch_changed = True
                        config[key] = config_json[key]
                    else:
                        print(f'Unknown Config Option: {key}')
        # if config is a dict, load default config then replace the default values with all values specified in config
        elif type(custom_config) is dict:
            for key, val in custom_config.items():
                if key in config.keys():
                    if ('Welch' in key or 'Epoch' in key) and config[key] != val:
                        welch_epoch_changed = True
                    config[key] = val
                else:
                    print(f'Unknown Config Option: {key}')
        else:
            print('Unknown type for parameter custom_config, using default')

    if welch_epoch_changed:
        # If config file has manually set the welch or epoch, then only run it for the specified welch/epoch setting
        eeg_epoch_size = config['YASA EEG Epoch Window Size']
        eeg_welch_size = config['YASA EEG Welch Window Size']
        heartrate_epoch_size = config['YASA Heart Rate Epoch Window Size']
        heartrate_welch_size = config['YASA Heart Rate Welch Window Size']
        setting_string = f'EE_{eeg_epoch_size}_EW_{eeg_welch_size}_HE_{heartrate_epoch_size}_HW_{heartrate_welch_size}'
        features_df = generate_features(
            seal_raw,
            custom_config=config
        )
        if labels_df is not None:
            features_df = features_df.merge(labels, how='outer', left_index=True, right_index=True)
        output_file = os.path.join(output_folder, seal_name, f'{seal_name}_07_features_with_labels_{setting_string}.csv')
        features_df.to_csv(output_file)
    else:
        for eeg_epoch_size, eeg_welch_size, heartrate_epoch_size, heartrate_welch_size in [
            (128, 16, 256, 256), # best settings for detecting active waking
            (64, 1, 128, 128), # best settings for detecting quiet waking
            (128, 1, 128, 64), # best settings for detecting drowsiness
            (64, 2, 512, 64), # best settings for detecting slow wave sleep
            (64, 16, 512, 128), # best settings for detecting REM sleep
        ]:
            print(f'Generating features for:\n\t' +
                    f'EEG epoch size:\t{eeg_epoch_size}\n\t' +
                    f'EEG welch size:\t{eeg_welch_size}\n\t' +
                    f'Heart Rate epoch size:\t{heartrate_epoch_size}\n\t' +
                    f'Heart Rate welch size:\t{heartrate_welch_size}\n')
            setting_string = f'EE_{eeg_epoch_size}_EW_{eeg_welch_size}_HE_{heartrate_epoch_size}_HW_{heartrate_welch_size}'
            config['YASA EEG Epoch Window Size'] = eeg_epoch_size
            config['YASA EEG Welch Window Size'] = eeg_welch_size
            config['YASA EEG Step Size'] = config['YASA EEG Epoch Window Size'] // 8
            config['YASA Heart Rate Epoch Window Size'] = heartrate_epoch_size
            config['YASA Heart Rate Welch Window Size'] = heartrate_welch_size
            config['YASA Heart Rate Step Size'] = config['YASA Heart Rate Epoch Window Size'] // 8
            features_df = generate_features(
                seal_raw,
                custom_config=config
            )
            if labels_df is not None:
                features_df = features_df.merge(labels, how='outer', left_index=True, right_index=True)
            output_file = os.path.join(output_folder, seal_name, f'{seal_name}_07_features_with_labels_{setting_string}.csv')
            features_df.to_csv(output_file)
    print(f'Features written to output folder: {output_folder}/{seal_name}')

# ---------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    """
    For help strings, run python feature_generation.py --help
    """
    WED_INPUT_FILE = 'data/raw/01_edf_data/test12_Wednesday_05_ALL_PROCESSED.edf'
    WED_OUTPUT_FILE = 'data/processed/features/test12_Wednesday_07_features_with_labels.csv'
    WED_LABELS_FILE = 'data/raw/02_hypnogram_data/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv'
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str,
                        help="input .edf filepath if reading one file, or input folder containing split seal .edf files if reading separated files",
                        default=WED_INPUT_FILE)
    parser.add_argument("-o", "--output", dest="output", type=str,
                        help="output .csv filepath if reading one file, or output folder if input is split seal .edf files if reading separated files",
                        default=WED_OUTPUT_FILE)
    parser.add_argument("-t", "--is_separated", dest="is_separated", type=bool,
                        help="whether to read multiple .edf files from input folder, or read one single .edf file, True = multiple, False = single",
                        default=False)
    parser.add_argument("-s", "--seal_name", dest="seal_name", type=str,
                        help="seal name that is in the split .edf files, to be used with a folder --input",
                        default=None)
    parser.add_argument("-l", "--labels", dest="labels", type=str, default=None,
                        help="optional .csv filepath with 1Hz labels")
    parser.add_argument("-c", "--config", dest="config", type=str, default=None,
                        help="JSON configuration for feature calculation hyperparameters")

    args = parser.parse_args()
    if args.input == WED_INPUT_FILE and args.output == WED_OUTPUT_FILE:
        args.labels = WED_LABELS_FILE

    # Check parameters point to actual files
    if args.input[-4:] != '.edf' and not args.is_separated:
        print(f'Input EDF must end in .edf')
        exit(-1)
    elif not os.path.exists(args.input):
        print(f'Input EDF {args.input} does not exist.')
        exit(-1)
    if args.output[-4:] != '.csv' and not args.is_separated:
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
        if not args.is_separated:
            print(f'Reading from single .edf file: {args.input}')
            features_df = generate_features(args.input, output_csv_path=None, custom_config=args.config)
            labels = pd.read_csv(args.labels)
            labels['R.Time'] = pd.to_datetime(labels['R.Time']).dt.tz_localize('America/Los_Angeles')
            labels = labels.set_index('R.Time', drop=True)[['Simple.Sleep.Code']]
            features_df = features_df.merge(labels, how='outer', left_index=True, right_index=True)
            features_df.to_csv(args.output)
        else:
            print(f'Reading from every .edf file with {args.seal_name} in filename inside {args.input} folder')
            labels = pd.read_csv(args.labels)
            labels['R.Time'] = pd.to_datetime(labels['R.Time']).dt.tz_localize('America/Los_Angeles')
            labels = labels.set_index('R.Time', drop=True)[['Simple.Sleep.Code']]
            generate_features_separated(args.input, args.seal_name, args.output, labels_df=labels, custom_config=args.config)
    
    # generate features, save to file
    else:
        if not args.is_separated:
            generate_features(args.input, output_csv_path=args.output, custom_config=args.config)
        else:
            generate_features_separated(args.input, args.seal_name, args.output, custom_config=args.config)
