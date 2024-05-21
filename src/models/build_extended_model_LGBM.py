from argparse import ArgumentParser
import joblib
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tabulate import tabulate
import warnings
warnings.simplefilter('ignore')

# custom functions
if __name__ != '__main__' and 'src' in __name__: # when being imported from the context of the src package
    from src.models.build_model_LGBM import evaluate_model
    from src.models.build_model_LGBM import build_model_LGBM
else:
    from build_model_LGBM import evaluate_model
    from build_model_LGBM import build_model_LGBM

def load_from_csvs(basic_features_csv, eeg_features_csv, heartrate_features_csv):
    """
    Loads features and concatenates them from input basic features, EEG features, and heart rate features
    basic_features_csv: path to basic features .csv, this is the output of running feature_generation.py
    eeg_features_csv: path to EEG features .csv, this is part of the output of running feature_generation_extended.py
    heartrate_features_csv: path to heart rate features .csv, this is part of the output of running feature_generation_extended.py
    """
    # PST timezone
    pst_timezone = pytz.timezone('America/Los_Angeles')

    # load basic features
    basic_features_df = pd.read_csv(basic_features_csv, index_col=0)
    basic_features_df.index = pd.DatetimeIndex(basic_features_df.index, tz=pst_timezone)

    # load eeg features
    eeg_features_df = pd.read_csv(eeg_features_csv, index_col=0)
    eeg_features_df.index = pd.DatetimeIndex(eeg_features_df.index, tz=pst_timezone)
    if 'Simple.Sleep.Code' in eeg_features_df.columns:
        eeg_features_df = eeg_features_df.drop('Simple.Sleep.Code', axis=1)

    # load heartrate features
    heartrate_features_df = pd.read_csv(heartrate_features_csv, index_col=0)
    heartrate_features_df.index = pd.DatetimeIndex(heartrate_features_df.index, tz=pst_timezone)
    if 'Simple.Sleep.Code' in heartrate_features_df.columns:
        heartrate_features_df = heartrate_features_df.drop('Simple.Sleep.Code', axis=1)
    
    print('Basic Features Time Start and End\t', basic_features_df.index[0], basic_features_df.index[-1], sep='\t')
    print('EEG Features Time Start and End\t', eeg_features_df.index[0], eeg_features_df.index[-1], sep='\t')
    print('Heart Rate Features Time Start and End', heartrate_features_df.index[0], heartrate_features_df.index[-1], sep='\t')

    training_df = pd.concat([basic_features_df, eeg_features_df, heartrate_features_df], axis=1)
    return training_df


def generate_feature_importance_plots(model, output_viz_folder):
    """
    Generate plots showing feature importance, aggregated by welch size, epoch size, and feature name
    model: a fit LGBM model, created with the extended features, with the feature_importances_ and feature_name_ attribute
    output_viz_folder: folder to save feature importances plots into as .png's
    """
    feat_importances = pd.Series(model.feature_importances_,
                                              index=model.feature_name_)
    feat_importances = feat_importances.rename('feature_importance')
    feat_importances.index = feat_importances.index.rename('feature')
    feat_importances.to_csv(f'{output_viz_folder}/feature_importances_extended.csv')
    # Build dictionaries for each category
    eeg_welch_importances_agg = {
        welch_num: [] for welch_num in [1, 2, 4, 8, 16]
    }
    hr_welch_importances_agg = {
        welch_num: [] for welch_num in [64, 128, 256, 512]
    }
    eeg_epoch_importances_agg = {
        epoch_num: [] for epoch_num in [16, 32, 64, 128, 256]
    }
    hr_epoch_importances_agg = {
        epoch_num: [] for epoch_num in [128, 256, 512]
    }
    eeg_freq_range_agg = {
        freq_range: [] for freq_range in ['0_0.5', '0.5_1'] + [f'{i}_{i+1}' for i in range(1,30)]
    }
    hr_features = ['HR_mean', 'HR_std', 'HR_iqr', 'HR_skew', 'HR_kurt', 'HR_hmob', 'HR_hcomp', 'HR_vlf',
                'HR_lf', 'HR_hf', 'HR_lf/hf', 'HR_p_total', 'HR_vlf_perc', 'HR_lf_perc', 'HR_hf_perc',
                'HR_lf_nu', 'HR_hf_nu']
    hr_importances_agg = {
        hr_feature: [] for hr_feature in hr_features
    }
    # Other EEG features
    other_eeg_feature_names = ['std', 'iqr', 'skew', 'kurt', 'nzc', 'hmob', 'hcomp', 'perm', 'higuchi', 'petrosian']
    other_eeg_features_agg = {
        other_feature: [] for other_feature in other_eeg_feature_names
    }
    for other_feature in other_eeg_feature_names:
        other_eeg_features_agg[f'{other_feature}_relative'] = []

    # Non-eeg features
    other_feature_names = [other_feature.replace(' ', '_') for other_feature in
                        ['Pressure Mean', 'Pressure Std.Dev', 'ODBA Mean', 'ODBA Std.Dev',
                            'GyrZ Mean', 'GyrZ Std.Dev']]
    other_features_agg = {
        other_feature: [] for other_feature in other_feature_names
    }

    # Append feature importances to each category
    for feature, importance in feat_importances.items():
        if 'WELCH' in feature:
            welch_num = int(re.findall('WELCH_[0-9]+', feature)[0][6:])
            if 'HR_' in feature:
                hr_welch_importances_agg[welch_num].append(importance)
            else:
                eeg_welch_importances_agg[welch_num].append(importance)
        if 'EPOCH' in feature:
            epoch_num = int(re.findall('EPOCH_[0-9]+', feature)[0][6:])
            if 'HR_' in feature:
                hr_epoch_importances_agg[epoch_num].append(importance)
            else:
                eeg_epoch_importances_agg[epoch_num].append(importance)
        for freq_range in eeg_freq_range_agg.keys():
            if '_' + freq_range in feature:
                eeg_freq_range_agg[freq_range].append(importance)
        for other_eeg_feature in other_eeg_feature_names:
            if other_eeg_feature in feature and 'HR' not in feature:
                if 'relative' in feature:
                    other_eeg_features_agg[f'{other_eeg_feature}_relative'].append(importance)
                else:
                    other_eeg_features_agg[f'{other_eeg_feature}'].append(importance)
        for other_feature in other_feature_names:
            if other_feature == feature:
                other_features_agg[other_feature].append(importance)
        if 'HR_' in feature:
            for hr_feature in hr_features:
                hr_feature_extracted = re.findall('HR_.*', feature)[0]
                if hr_feature == hr_feature_extracted:
                    hr_importances_agg[hr_feature].append(importance)

    # DataFrame of each category feature importance distributions
    def append_zeroes_to_shorter_columns(df_dict):
        max_length = max(len(arr) for arr in df_dict.values())

        # Iterate through the dictionary and append zeros to shorter arrays
        for key, arr in df_dict.items():
            while len(arr) < max_length:
                arr.append(0)
    
    # Make all df_dicts have the same length for each columns
    for df_dict in [eeg_welch_importances_agg,
                    eeg_epoch_importances_agg,
                    eeg_freq_range_agg,
                    other_eeg_features_agg,
                    hr_welch_importances_agg,
                    hr_epoch_importances_agg,
                    hr_importances_agg,
                    other_features_agg]:
        append_zeroes_to_shorter_columns(df_dict)
        
    eeg_welch_importance_df = pd.DataFrame(eeg_welch_importances_agg)
    eeg_epoch_importance_df = pd.DataFrame(eeg_epoch_importances_agg)
    eeg_freq_range_importance_df = pd.DataFrame(eeg_freq_range_agg)
    other_eeg_feature_df = pd.DataFrame(other_eeg_features_agg)
    hr_welch_importance_df = pd.DataFrame(hr_welch_importances_agg)
    hr_epoch_importance_df = pd.DataFrame(hr_epoch_importances_agg)
    hr_features_df = pd.DataFrame(hr_importances_agg)
    other_features_df = pd.DataFrame(other_features_agg)

    eeg_welch_importance_df.columns = [f'WELCH_{col}_seconds' for col in eeg_welch_importance_df.columns]
    eeg_epoch_importance_df.columns = [f'EPOCH_{col}_seconds' for col in eeg_epoch_importance_df.columns]
    eeg_freq_range_importance_df.columns = [f'FREQ_RANGE_{col}' for col in eeg_freq_range_importance_df.columns]
    hr_welch_importance_df.columns = [f'WELCH_{col}_seconds' for col in hr_welch_importance_df.columns]
    hr_epoch_importance_df.columns = [f'EPOCH_{col}_seconds' for col in hr_epoch_importance_df.columns]


    # Make plots ---
    # EEG welch
    fig, ax = plt.subplots()
    ax.barh(width=eeg_welch_importance_df.sum(), y=eeg_welch_importance_df.columns)
    ax.set_title('WELCH window size importance')
    fig.savefig(f'{output_viz_folder}/EEG_Welch_Importance.png', bbox_inches='tight')

    # EEG epoch
    fig, ax = plt.subplots()
    ax.barh(width=eeg_epoch_importance_df.sum(), y=eeg_epoch_importance_df.columns)
    ax.set_title('EPOCH size importance')
    fig.savefig(f'{output_viz_folder}/EEG_Epoch_Importance.png', bbox_inches='tight')

    # EEG freq ranges
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(width=eeg_freq_range_importance_df.sum(), y=eeg_freq_range_importance_df.columns)
    ax.set_title('Frequency range importance')
    fig.savefig(f'{output_viz_folder}/EEG_Frequency_Range_Importance.png', bbox_inches='tight')

    # EEG other features
    fig, ax = plt.subplots()
    ax.barh(width=other_eeg_feature_df.sum(), y=other_eeg_feature_df.columns)
    ax.set_title('Other EEG features\' importance')
    fig.savefig(f'{output_viz_folder}/EEG_Other_Feature_Importance.png', bbox_inches='tight')

    # Heart Rate welch
    fig, ax = plt.subplots()
    ax.barh(width=hr_welch_importance_df.sum(), y=hr_welch_importance_df.columns)
    ax.set_title('Heart Rate WELCH size importance')
    fig.savefig(f'{output_viz_folder}/Heart_Rate_Welch_Importance.png', bbox_inches='tight')

    # Heart Rate epoch
    fig, ax = plt.subplots()
    ax.barh(width=hr_epoch_importance_df.sum(), y=hr_epoch_importance_df.columns)
    ax.set_title('Heart Rate EPOCH size importance')
    fig.savefig(f'{output_viz_folder}/Heart_Rate_Epoch_Importance.png', bbox_inches='tight')

    # Heart Rate other
    fig, ax = plt.subplots()
    ax.barh(width=hr_features_df.sum(), y=hr_features_df.columns)
    ax.set_title('Heart Rate features\' importance')
    fig.savefig(f'{output_viz_folder}/Heart_Rate_Feature_Importance.png', bbox_inches='tight')

    # Movement & Pressure
    fig, ax = plt.subplots()
    ax.barh(width=other_features_df.sum(), y=other_features_df.columns)
    ax.set_title('Other features\' importance')
    fig.savefig(f'{output_viz_folder}/Other_Feature_Importance.png', bbox_inches='tight')

    print(f'Feature importance plots saved to {output_viz_folder}')


if __name__ == '__main__':
    """
    For help strings, run python build_extended_model_LGBM.py --help
    """
    TRAINING_FEATURES_FILE = 'data/processed/Wednesday_features_with_labels_v3.csv'
    EEG_FEATURES_FILE = 'data/interim/feature_discovery/EEG/Wednesday_feature_discovery_EEG.csv'
    HEARTRATE_FEATURES_FILE = 'data/interim/feature_discovery/ECG/Wednesday_feature_discovery_ECG.csv'
    MODEL_OUTPUT_FILE = 'models/lightgbm_model_extended.pkl'
    CONFUSION_MATRIX_OUTPUT_FILE = 'models/lightgbm_model_extended_confusion_matrix.csv'
    FEATURE_IMPORTANCE_VIZ_OUTPUT_FOLDER = 'reports/figures/feature_discovery'
    best_params = {'learning_rate': 0.005, 'n_estimators': 400, 'num_leaves': 10}
    parser = ArgumentParser()
    parser.add_argument("-b", "--basic_features", dest="basic_features", type=str, help="basic training features .csv filepath (for GyrZ, MagZ, ODBA, Pressure)",
                        default=TRAINING_FEATURES_FILE)
    parser.add_argument("-e", "--eeg_features", dest="eeg_features", type=str, help="input eeg features .csv filepath (containing all concatenated EPOCH & WELCH settings)",
                        default=EEG_FEATURES_FILE)
    parser.add_argument("-c", "--ecg_features", dest="ecg_features", type=str, help="input ecg heart rate features .csv filepath (containing all concatenated EPOCH & WELCH settings)",
                        default=HEARTRATE_FEATURES_FILE)
    parser.add_argument("-o", "--output", dest="output", type=str, help="output model .pkl filepath",
                        default=MODEL_OUTPUT_FILE)
    parser.add_argument("-m", "--matrix", dest="matrix", type=str, help="output confusion matrix .csv filepath",
                        default=CONFUSION_MATRIX_OUTPUT_FILE)
    parser.add_argument("-v", "--viz_folder", dest="viz_folder", type=str, help="output folder for feature importance visualizations",
                        default=FEATURE_IMPORTANCE_VIZ_OUTPUT_FOLDER)

    args = parser.parse_args()

    # Check parameters point to actual files
    for arg in vars(args):
        val = getattr(args, arg)
        if arg in ['basic_features', 'eeg_features', 'ecg_features']:
            if val[-4:] != '.csv':
                print(f'Input file {val} for argument {arg} must end in ".csv"')
                exit(-1)
            elif not os.path.exists(val):
                print(f'Input file {val} for argument {arg} does not exist in filesystem')
                exit(-1)
        if arg == 'output':
            if val[-4:] != '.pkl':
                print(f'Output model file {val} for argument {arg} must end in ".pkl"')
                exit(-1)
        if arg == 'matrix': 
            if val[-4:] != '.csv':
                print(f'Output confusion matrix file {val} for argument {arg} must end in ".csv"')
                exit(-1)
    
    training_df = load_from_csvs(args.basic_features, args.eeg_features, args.ecg_features)
    evaluate_model(training_df, 'Simple.Sleep.Code', args.matrix, params=best_params)
    model = build_model_LGBM(training_df, 'Simple.Sleep.Code', args.output, params=best_params)
    generate_feature_importance_plots(model, args.viz_folder)