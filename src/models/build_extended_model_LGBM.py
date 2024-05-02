import pytz
import numpy as np
import pandas as pd
import os
import warnings
import joblib
from argparse import ArgumentParser
from lightgbm import LGBMClassifier

def build_lgbm_model(training_df, target_col, outfile=None):
    model = LGBMClassifier(learning_rate=0.005, n_estimators=400, num_leaves=10, n_jobs=8)
    X = training_df.drop(target_col, axis=1)
    y = training_df[target_col]
    model.fit(X, y)
    if outfile is not None:
        joblib.dump(model, outfile)
    return model

if __name__ == '__main__':
    BASIC_FEATURES_PATH = 'data/processed/Wednesday_features_with_labels_v3.csv'
    EEG_FEATURES_PATH = 'data/interim/feature_discovery/EEG/Wednesday_feature_discovery_EEG.csv'
    ECG_FEATURES_PATH = 'data/interim/feature_discovery/ECG/Wednesday_feature_discovery_ECG.csv'
    MODEL_OUTPUT_FILE = 'models/lightgbm_extended_features_model.pkl'
    VISUALIZATION_OUTPUT_FOLDER = 'reports/figures/feature_importances'

    parser = ArgumentParser()
    parser.add_argument("-b", "--basic_features", dest="basic_features", type=str, help="input basic features .csv filepath",
                        default=BASIC_FEATURES_PATH)
    parser.add_argument("-e", "--eeg", dest="eeg", type=str, help="input EEG training features .csv filepath",
                        default=EEG_FEATURES_PATH)
    parser.add_argument("-c", "--ecg", dest="ecg", type=str, help="input ECG training features .csv filepath",
                        default=ECG_FEATURES_PATH)
    parser.add_argument("-o", "--output", dest="output", type=str, help="output model .pkl filepath",
                        default=MODEL_OUTPUT_FILE)
    parser.add_argument("-v", "--visual_output", dest="visual_output", type=str, help="output folder to save feature importance visualizations",
                        default=VISUALIZATION_OUTPUT_FOLDER)

    args = parser.parse_args()
    
    # Check parameters point to actual files
    if args.basic_features[-4:] != '.csv':
        print(f'Input basic features file must end in .csv')
        exit(-1)
    if not os.path.exists(args.basic_features):
        print(f'Input basic features file {args.basic_features} does not exist.')
        exit(-1)
    if args.eeg[-4:] != '.csv':
        print(f'Input basic features file must end in .csv')
        exit(-1)
    if not os.path.exists(args.eeg):
        print(f'Input EEG features file {args.eeg} does not exist.')
        exit(-1)
    if args.ecg[-4:] != '.csv':
        print(f'Input basic features file must end in .csv')
        exit(-1)
    if not os.path.exists(args.ecg):
        print(f'Input ECG features file {args.ecg} does not exist.')
        exit(-1)
    if args.output[-4:] != '.pkl':
        print(f'Output file {args.output} must end in .pkl')
        exit(-1)
    
    basic_features_training_df = pd.read_csv(args.basic_features, index_col=0)
    basic_features_training_df = basic_features_training_df[[col for col in basic_features_training_df.columns if 'yasa' not in col]]
    print(basic_features_training_df.columns)
    eeg_training_df = pd.read_csv(args.eeg, index_col=0)
    ecg_training_df = pd.read_csv(args.ecg, index_col=0)
    pst_timezone = pytz.timezone('America/Los_Angeles')
    basic_features_training_df.index = pd.DatetimeIndex(basic_features_training_df.index, tz=pst_timezone)
    eeg_training_df.index = pd.DatetimeIndex(eeg_training_df.index, tz=pst_timezone)
    ecg_training_df.index = pd.DatetimeIndex(ecg_training_df.index, tz=pst_timezone)
    training_df = pd.concat([eeg_training_df, ecg_training_df], axis=1)
    del(eeg_training_df)
    del(ecg_training_df)
    build_lgbm_model(training_df, 'Simple.Sleep.Code', MODEL_OUTPUT_FILE)