from argparse import ArgumentParser
import joblib
from lightgbm import LGBMClassifier
import numpy as np
import os
import pandas as pd
import pytz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tabulate import tabulate
import warnings
warnings.simplefilter('ignore')

def build_model_LGBM(training_df, target_col, outfile=None, params={'learning_rate': 0.005, 'n_estimators': 400, 'num_leaves': 10}):
    """
    Builds an LGBM model with a training dataframe and target column name
    training_df: pandas dataframe containing features and target (only one column should be the target variable while the rest are features)
    target_col: string name of the target column
    outfile: location to save the trained model as a .pkl file
    params: parameters to pass into LGBM, by default uses parameters that were found using GridSearchCV on the seal Wednesday
    """
    model_df = training_df.dropna()
    model = LGBMClassifier(**params, n_jobs=8)
    X = model_df.drop(target_col, axis=1)
    y = model_df[target_col]
    model.fit(X, y)
    if outfile is not None:
        joblib.dump(model, outfile)
    return model


def evaluate_model(training_df, target_col, outfile=None, params={'learning_rate': 0.005, 'n_estimators': 400, 'num_leaves': 10},
                   return_preds=False, verbosity=1):
    """
    Evaluates the performance of an LGBM model on the training_df features in predicting the target_col, using k-fold validation
    training_df: pandas dataframe containing features and target (only one column should be the target variable while the rest are features)
    target_col: string name of the target column
    outfile: location to save the confusion matrix as a .csv
    params: parameters to pass into LGBM, by default uses parameters that were found using GridSearchCV on the seal Wednesday
    return_preds: whether to return the predictions
    verbosity: level of verbosity for the LGBMClassifier and its performance (-1 = none, 0 = model eval but not LightGBM, 1 = all)
    """
    model_df = training_df.dropna()
    # Initialize k-fold
    n_splits = 5  # Define the number of splits for k-fold
    skf = KFold(n_splits=n_splits, shuffle=False)

    # Initialize arrays to store accuracies
    class_accuracies = []
    overall_accuracies = []
    conf_matrices = []

    X, y = model_df.drop(target_col, axis=1), model_df[target_col]
    kfold_preds = []
    # Perform k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        if verbosity >= 0:
            print(f'Fold {fold + 1}/{n_splits}')
        kmodel = LGBMClassifier(**params, verbosity=verbosity, n_jobs=8)
        
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Train the model without unscorable so we aren't teachin the model to predict it
        train_filter = y_tr != 'Unscorable'
        X_tr, y_tr = X_tr.loc[train_filter], y_tr.loc[train_filter]
        
        # Train the model
        kmodel.fit(X_tr, y_tr)
        
        # Predict on validation set
        y_pred = pd.Series(kmodel.predict(X_val), index=y_val.index)
        kfold_preds.append(y_pred)

        # Calculate accuracy per class
        class_accuracy = []
        for class_label in np.sort(np.unique(y_val)):
            class_accuracy.append(np.sum((y_pred == y_val) & (y_val == class_label)) / 
                                np.sum(y_val == class_label))

        class_accuracies.append(pd.Series(class_accuracy, index=np.unique(y_val)))
        overall_accuracies.append(np.mean(y_pred == y_val))

        custom_order = ['Active Waking', 'Quiet Waking', 'Drowsiness', 'SWS', 'REM', 'Unscorable']
        y_labels_ordered = [val for val in custom_order if val in np.unique(y)]
        conf_matr = confusion_matrix(y_val, y_pred, labels=y_labels_ordered)
        conf_matr = pd.DataFrame(conf_matr,
                                 index=['True_' + label for label in y_labels_ordered],
                                 columns=['Predicted_'+ label for label in y_labels_ordered])
        conf_matr.index = conf_matr.index.str.replace(' ', '_')
        conf_matr.columns = conf_matr.columns.str.replace(' ', '_')
        conf_matrices.append(conf_matr)

    summed_conf_matr = conf_matrices[0]
    for conf_matr in conf_matrices[1:]:
        summed_conf_matr = summed_conf_matr.add(conf_matr, fill_value=0)
    # Calculate mean accuracy per class across folds
    mean_class_accuracies = pd.concat(class_accuracies, axis=1).mean(axis=1).round(4) * 100
    if verbosity >= 0:
        print("Overall accuracy: ", round(np.mean(overall_accuracies) * 100, 2), '%', sep='')
        print()
        print("Mean class accuracies across folds:")
        print(mean_class_accuracies)
        print()
        print('Overall confusion matrix:')
        print(tabulate(summed_conf_matr, headers=summed_conf_matr.columns))

    for i, conf_matr in enumerate(conf_matrices):
        conf_matr.index = conf_matr.index + f'_FOLD_{i}'
    conf_matrices_combined = pd.concat(conf_matrices)
    if outfile is not None:
        conf_matrices_combined.to_csv(outfile)
    
    if return_preds:
        return (overall_accuracies, mean_class_accuracies, conf_matrices_combined, summed_conf_matr, kfold_preds)
    return (overall_accuracies, mean_class_accuracies, conf_matrices_combined, summed_conf_matr)


if __name__ == '__main__':
    """
    For help strings, run python build_model_LGBM.py --help
    """
    TRAINING_FEATURES_FILE = 'data/processed/features/Wednesday_features_with_labels_v3.csv'
    MODEL_OUTPUT_FILE = 'models/lightgbm_model_basic.pkl'
    CONFUSION_MATRIX_OUTPUT_FILE = 'models/lightgbm_model_basic_confusion_matrix.csv'
    best_params = {'learning_rate': 0.005, 'n_estimators': 400, 'num_leaves': 10}
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, help="input training features .csv filepath",
                        default=TRAINING_FEATURES_FILE)
    parser.add_argument("-o", "--output", dest="output", type=str, help="output model .pkl filepath",
                        default=MODEL_OUTPUT_FILE)
    parser.add_argument("-c", "--matrix", dest="matrix", type=str, help="output confusion matrix .csv filepath",
                        default=CONFUSION_MATRIX_OUTPUT_FILE)

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
    
    training_df = pd.read_csv(args.input, index_col=0)
    pst_timezone = pytz.timezone('America/Los_Angeles')
    training_df.index = pd.DatetimeIndex(training_df.index, tz=pst_timezone)
    evaluate_model(training_df, 'Simple.Sleep.Code', args.matrix, params=best_params)
    build_model_LGBM(training_df, 'Simple.Sleep.Code', args.output, params=best_params)