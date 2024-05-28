from argparse import ArgumentParser
import joblib
from lightgbm import LGBMClassifier
import numpy as np
import os
import pandas as pd
import pytz
import re
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
    from build_extended_model_LGBM import load_from_csvs


def iterate_single_models(training_df, target_col):
    """
    Iterates through every (EEG Setting, ECG Setting) pair and stores the accuracy and per-class accuracy
    training_df: pandas dataframe containing features and target (only one column should be the target variable while the rest are features)
    target_col: string name of the target column
    """
    # heart rate settings
    hr_settings = []
    hr_string = r'EPOCH_[0-9]+_WELCH_[0-9]+_HR'
    for col in training_df.columns:
        matches = re.findall(hr_string, col)
        if len(matches) == 1:
            hr_settings.append(matches[0])
    # eeg settings
    eeg_settings = []
    eeg_string = r'EPOCH_[0-9]+_WELCH_[0-9]+_EEG'
    for col in training_df.columns:
        matches = re.findall(eeg_string, col)
        if len(matches) == 1:
            eeg_settings.append(matches[0])
    # movement & pressure columns and target column
    general_columns = [col for col in training_df.columns if ('EEG' not in col and 'HR' not in col and 'yasa' not in col and col != target_col)] + [target_col]

    # only get unique settings
    hr_settings = sorted(list(set(hr_settings)))
    eeg_settings = sorted(list(set(eeg_settings)))

    # iterate through every (single_heartrate_setting, single_eeg_setting) combination
    setting_outputs = []
    num_iterations = len(eeg_settings) * len(hr_settings)
    counter = 0
    for eeg_setting_string in eeg_settings:
        for hr_setting_string in hr_settings:
            print(f'{100*(counter/num_iterations):.2f}% done. \tStarting: ({eeg_setting_string}, {hr_setting_string})' + ' '*30, end='\r')
            col_filter = (
                (training_df.columns.str.contains(eeg_setting_string)) | # eeg columns
                (training_df.columns.str.contains(hr_setting_string)) | # heart reate columns
                (training_df.columns.isin(general_columns)) # general movement columns & true labels
            )
            simple_model_df = training_df.loc[:,col_filter].copy()
            overall_accuracies, mean_class_accuracies, conf_matrices_combined, summed_conf_matr = evaluate_model(simple_model_df, target_col, verbosity=-1)
            setting_outputs.append((eeg_setting_string, hr_setting_string, overall_accuracies, mean_class_accuracies, conf_matrices_combined, summed_conf_matr))
            counter += 1
    print('100% complete' + ' '*50)
    # make accuracy dataframe from output
    setting_accuracy_df_dict = {
        'EEG Setting': [],
        'Heart Rate Setting': [],
        'Overall Accuracy': []
    }
    for sleep_state in ['Active Waking', 'Quiet Waking', 'Drowsiness', 'SWS', 'REM']:
        setting_accuracy_df_dict[f'{sleep_state} Accuracy'] = []
    
    # append values
    for setting_output in setting_outputs:
        setting_accuracy_df_dict['EEG Setting'].append(setting_output[0])
        setting_accuracy_df_dict['Heart Rate Setting'].append(setting_output[1])
        setting_accuracy_df_dict['Overall Accuracy'].append(np.mean(setting_output[2]))
        for sleep_state in ['Active Waking', 'Quiet Waking', 'Drowsiness', 'SWS', 'REM']:
            setting_accuracy_df_dict[f'{sleep_state} Accuracy'].append(setting_output[3][sleep_state])
    # make dataframe from dictionary
    setting_accuracy_df = pd.DataFrame(setting_accuracy_df_dict)

    return setting_accuracy_df, setting_outputs # return setting accuracy dataframe and model outputs (in case we want predictions and feature importances later)

def build_refined_model_LGBM(training_df, target_col, features_outfile, setting_accuracies_outfile=None, outfile=None, matrix_outfile=None):
    """
    Discovers and builds the best "refined" model, using only the minimum best settings for EEG and ECG features
    training_df: pandas dataframe containing features and target (only one column should be the target variable while the rest are features)
    target_col: string name of the target column
    setting_accuracies_outfile: folder to save the setting accuracies .csv
    outfile: output filepath to store the model as a .pkl
    matrix_outfile: output filepath to store the confusion matrix as a .csv
    """
    setting_accuracy_df, setting_outputs = iterate_single_models(training_df, target_col)
    if setting_accuracies_outfile is not None:
        setting_accuracy_df.to_csv(setting_accuracies_outfile)
    settings_to_include = []
    for sleep_state in ['Active Waking', 'Quiet Waking', 'Drowsiness', 'SWS', 'REM']:
        # order accuracies by current sleep_state
        sleep_state_accuracies = setting_accuracy_df[['EEG Setting', 'Heart Rate Setting', f'{sleep_state} Accuracy']].sort_values(by=f'{sleep_state} Accuracy', ascending=False)

        # get the best combination of settings for the current sleep state
        best_setting = sleep_state_accuracies.iloc[0]
        best_eeg_setting, best_heartrate_setting = best_setting['EEG Setting'], best_setting['Heart Rate Setting']
        settings_to_include.append(best_eeg_setting)
        settings_to_include.append(best_heartrate_setting)
    
    # drop duplicate settings if they are best in multiple sleep states, so we only include it once in the refined model
    settings_to_include = list(set(settings_to_include))
    
    # Add basic movement and pressure features, and target column
    settings_to_include.extend([col for col in training_df.columns if ('EEG' not in col and 'HR' not in col and 'yasa' not in col and col != target_col)])
    settings_to_include.append(target_col)

    custom_sorted = [setting for setting in settings_to_include if 'EEG' in setting] + \
                    [setting for setting in settings_to_include if 'HR' in setting]
    print('Final Included Settings:\n\t', '\n\t'.join(custom_sorted))
    
    # Build final refined model dataframe using only columns in the refined settings
    refined_training_df = training_df.loc[:, [any(setting in col for setting in settings_to_include) for col in training_df.columns]]

    # Build and evaluate refined model
    model = build_model_LGBM(refined_training_df, target_col, outfile=outfile) # also saves model to output .pkl file
    overall_accuracies, mean_class_accuracies, conf_matrices_combined, summed_conf_matr = evaluate_model(refined_training_df, target_col, outfile=matrix_outfile, verbosity=-1) # also saved confusion matrix to output .csv file
    print("Overall accuracy: ", round(np.mean(overall_accuracies) * 100, 2), '%', sep='')
    print()
    print("Mean class accuracies across folds:")
    print(mean_class_accuracies)
    print()
    print('Overall confusion matrix:')
    print(tabulate(summed_conf_matr, headers=summed_conf_matr.columns))

    # Save features dataframe
    refined_training_df.to_csv(features_outfile)

if __name__ == '__main__':
    """
    For help strings, run python build_refined_model_LGBM.py --help
    """
    TRAINING_FEATURES_FILE = 'data/processed/features/test12_Wednesday_07_features_with_labels.csv'
    EEG_FEATURES_FILE = 'data/interim/feature_discovery/EEG/Wednesday_feature_discovery_EEG.csv'
    HEARTRATE_FEATURES_FILE = 'data/interim/feature_discovery/ECG/Wednesday_feature_discovery_ECG.csv'
    FEATURES_OUTPUT_FILE = 'data/processed/features/test12_Wednesday_08_refined_features_with_labels.csv'
    MODEL_OUTPUT_FILE = 'models/lightgbm_model_refined.pkl'
    CONFUSION_MATRIX_OUTPUT_FILE = 'models/lightgbm_model_refined_confusion_matrix.csv'
    SETTING_ACCURIES_OUTPUT_FILE = 'data/interim/setting_accuracies.csv'
    best_params = {'learning_rate': 0.005, 'n_estimators': 400, 'num_leaves': 10}
    parser = ArgumentParser()
    parser.add_argument("-b", "--basic_features", dest="basic_features", type=str, help="basic training features .csv filepath (for GyrZ, MagZ, ODBA, Pressure)",
                        default=TRAINING_FEATURES_FILE)
    parser.add_argument("-e", "--eeg_features", dest="eeg_features", type=str, help="input eeg features .csv filepath (containing all concatenated EPOCH & WELCH settings)",
                        default=EEG_FEATURES_FILE)
    parser.add_argument("-c", "--ecg_features", dest="ecg_features", type=str, help="input ecg heart rate features .csv filepath (containing all concatenated EPOCH & WELCH settings)",
                        default=HEARTRATE_FEATURES_FILE)
    parser.add_argument("-f", "--features", dest="features", type=str, help="output features .csv filepath",
                        default=FEATURES_OUTPUT_FILE)
    parser.add_argument("-o", "--output", dest="output", type=str, help="output model .pkl filepath",
                        default=MODEL_OUTPUT_FILE)
    parser.add_argument("-m", "--matrix", dest="matrix", type=str, help="output confusion matrix .csv filepath",
                        default=CONFUSION_MATRIX_OUTPUT_FILE)
    parser.add_argument("-a", "--setting_accuracies", dest="setting_accuracies", type=str, help="output setting accuracies .csv filepath",
                        default=SETTING_ACCURIES_OUTPUT_FILE)

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
    build_refined_model_LGBM(training_df, 'Simple.Sleep.Code', args.features, args.setting_accuracies, args.output, args.matrix)