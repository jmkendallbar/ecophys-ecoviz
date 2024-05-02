import pytz
import numpy as np
import pandas as pd
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
    TRAINING_FEATURES_FILE = 'data/processed/Wednesday_features_with_labels_v3.csv'
    MODEL_OUTPUT_FILE = 'models/lightgbm_model.pkl'
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, help="input training features .csv filepath",
                        default=TRAINING_FEATURES_FILE)
    parser.add_argument("-o", "--output", dest="output", type=str, help="output model .pkl filepath",
                        default=MODEL_OUTPUT_FILE)

    args = parser.parse_args()
    
    # Check parameters point to actual files
    if args.input[-4:] != '.csv':
        print(f'Input training features file must end in .csv')
        exit(-1)
    elif not os.path.exists(args.input):
        print(f'Input CSV {args.input} does not exist.')
        exit(-1)
    if args.output[-4:] != '.pkl':
        print(f'Output file {args.output} must end in .pkl')
        exit(-1)
    
    training_df = pd.read_csv(TRAINING_FEATURES_FILE, index_col=0)
    pst_timezone = pytz.timezone('America/Los_Angeles')
    training_df.index = pd.DatetimeIndex(training_df.index, tz=pst_timezone)
    build_lgbm_model(training_df, 'Simple.Sleep.Code', MODEL_OUTPUT_FILE)