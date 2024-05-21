
# EcoViz Ecophysiology Seal Sleep Classifier

<!-- badges: start -->
<!-- badges: end -->


### Folder structure (after running all make targets)

```
├── DESCRIPTION
├── Makefile
├── README.md
├── data
│   ├── README.md
│   ├── interim
│   │   ├── feature_discovery
│   │   │   ├── test12_Wednesday_feature_discovery_ECG.csv
│   │   │   └── test12_Wednesday_feature_discovery_EEG.csv
│   │   └── settings_accuracies.csv
│   ├── processed
│   │   └── test12_Wednesday_07_features_with_labels.csv
│   └── raw
│       ├── 01_edf_data
│       │   └── test12_Wednesday_05_ALL_PROCESSED.edf
│       └── 02_hypnogram_data
│           └── test12_Wednesday_06_Hypnogram_JKB_1Hz.csv
├── models
│   ├── lightgbm_model_basic.pkl
│   ├── lightgbm_model_extended.pkl
│   ├── lightgbm_model_extended_confusion_matrix.csv
│   ├── lightgbm_model_refined.pkl
│   └── lightgbm_model_refined_confusion_matrix.csv
├── notebooks
│   ├── 00_heart_rate_peak_detection.ipynb
│   ├── 01_visualize_feature_density.ipynb
│   ├── 02_visualize_naps_features_with_labels.ipynb
│   ├── 03_basic_model_evaluation???.ipynb
│   ├── 04_lgbm_feature_discovery.ipynb
│   ├── 05_refined_model_evaluation.ipynb
│   └── README.md
├── paper
│   └── README.md
├── reports
│   └── figures
│       ├── README.md
│       ├── feature_discovery
│       │   ├── EEG_Epoch_Importance.png
│       │   ├── EEG_Frequency_Range_Importance.png
│       │   ├── EEG_Other_Feature_Importance.png
│       │   ├── EEG_Welch_Importance.png
│       │   ├── Heart_Rate_Epoch_Importance.png
│       │   ├── Heart_Rate_Feature_Importance.png
│       │   ├── Heart_Rate_Welch_Importance.png
│       │   ├── Other_Feature_Importance.png
│       │   └── feature_importances_extended.csv
│       └── labchart
│           ├── light-sleep-spectral-power.png
│           ├── rem.png
│           ├── slow-wave-1.png
│           ├── slow-wave-2-spectral-power.png
│           └── slow-wave-2.png
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── features
│   │   ├── feature_generation.py
│   │   ├── feature_generation_extended.py
│   │   └── feature_generation_utils.py
│   ├── models
│   │   ├── build_extended_model_LGBM.py
│   │   ├── build_model_LGBM.py
│   │   └── build_refined_model_LGBM.py
└── └── visualization
        └── visualize.py
```

* `data/` - this holds raw, processed (features), and interim (feature discovery) data 
* `models/` - basic, 
* `notebooks/` - literate programming files explaining the pipeline and analyzing visualizations
* `paper/` - the analysis manuscript
* `reports/` - report-facing figures and analyses
* `src/` - source files supporting the pipeline (feature generation, model building, visualization)

### Computational environment

```
├── Makefile
└── requirements.txt
├── src
│   ├── __init__.py
```

These are conventional files used for pipeline consolidation and recording package dependencies. 

`Makefile` is a file used to allow running the pipeline from the command-line using `make`
`requirements.txt` is a file used to record Python package dependencies. Users can install dependencies by running `pip install -r requirements.txt` at the command line.
`__init__.py` is a file used to by Python that allows jupyter notebooks and external python programs to recognize the src folder as a Python package.

## Makefile pipeline steps

Here's how the pipeline flows from the Makefile (this will run the pipeline on the seal Wednesday).

#### `make download`

- This downloads the necessary raw .edf file and .csv hypnogram file from figshare

#### `make features`

- This makes the features from the .edf file using the naive settings for epoch size and welch window size (used for power spectral density)

#### `make model`

- This makes the basic model using the naive features, and prints the performance and saves both the model and confusion matrix.

#### `make extended_features`

- This is the most important step for feature exploration and discovery; it re-generates the features many times using different settings for epoch size and welch window size to discovery what settings are "best"

#### `make extended_model`

- This trains a LightGBM model using the extended features, which allows for the LightGBM to discover which features are most important, and then the script generate plots that visualize the aggregate importance of different epoch and welch settings, and the aggregated feature importance of the features themselves

#### `make refined_model`

- This trains a LightGBM model on each single setting from the extended model, to find which setting is best at predicting each sleep state: Active Waking, Quiet Waking, Drowsiness, SWS, REM; the script then saves the subset features, model, and confusion matrix to the output files

## Python script usage

Each python script besides visualize.py and feature_generation_utils.py can be run from the command-line as main, and has different options which when provided can change its usage. Each make target (besides make download) calls one of these python scripts, as follows:
    - `make features`: python src/features/feature_generation.py
    - `make model`: python src/models/build_model_LGBM.py
    - `make features_extended`: python src/features/feature_generation_extended.py
    - `make model_extended`: python src/models/build_extended_model_LGBM.py
    - `make refined_model`: python src/models/build_refined_model_LGBM.py

Each of these python scripts have different options available, and these options can be viewed by adding **--help** after calling it via `python` in the command-line. Note that all of these scripts have optional parameters, and by default will run as if running on test12_Wednesday. 

#### python src/features/feature_generation.py --help

Generates the sleep features from an EDF file.

```
usage: feature_generation.py [-h] [-i INPUT] [-o OUTPUT] [-t IS_SEPARATED] [-s SEAL_NAME] [-l LABELS] [-c CONFIG]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input .edf filepath if reading one file, or input folder containing split seal .edf files if reading separated files
  -o OUTPUT, --output OUTPUT
                        output .csv filepath if reading one file, or output folder if input is split seal .edf files if reading separated files
  -t IS_SEPARATED, --is_separated IS_SEPARATED
                        whether to read multiple .edf files from input folder, or read one single .edf file, True = multiple, False = single
  -s SEAL_NAME, --seal_name SEAL_NAME
                        seal name that is in the split .edf files, to be used with a folder --input
  -l LABELS, --labels LABELS
                        optional .csv filepath with 1Hz labels
  -c CONFIG, --config CONFIG
                        JSON configuration for feature calculation hyperparameters
```
    - `-i`: input .edf file to use for feature generation. This **must** have an ECG and EEG Channel, and if it is different than *ECG_Raw_Ch1* and *EEG_ICA5*, you must specify this in a config.json file
    - `-o`: output .csv filepath - where to save the features file. This should be the full or relative filepath and include the file name with extension
    - `-t`: whether the input EDF is split into multiple days (if it is, then the input -i shoud be a path to a folder, and the output -o should be a path to a folder)
    - `-s`: to be used with -t True, the seal name to look for in the folder; will concatenate all EDF files with that seal name in chronological order (looking up their recording start times)
    - `-l`: optional path to 1Hz hypnogram labels .csv to include in the features output as a 'Simple.Sleep.Code' column
    - `-c`: JSON config file to overwrite the default settings in the feature_generation.py python script. These settings and their default values are (you can copy this JSON dictionary and save it as a text file with the .json extension, replacing the values you would like to change):

```
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
```

Example: (input is one .EDF file):

```
python src/features/feature_generation.py -i data/raw/01_edf_data/test12_Wednesday_05_ALL_PROCESSED.edf -o data/processed/features/test12_Wednesday_07_features_with_labels.csv -l data/raw/02_hypnogram_data/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv
```
Note that -s is only needed if -t is set to True, and if -c is excluded the program uses the default configuration

Example: (input is all the .EDF files in data/raw/01_edf_data containing test33_HypoactiveHeidi in the file name, output is the folder data/processed) - the program will create a folder named test33_HypoactiveHeidi inside data/processed and save all feature files to that folder

```
python src/features/feature_generation.py -i data/raw/01_edf_data -o data/processed -t True -s test33_HypoactiveHeidi -l data/raw/02_hypnogram_data/test33_HypoactiveHeidi_06_Hypnogram_JKB_1Hz.csv -c data/raw/config/HypoactiveHeidi_config.json
```

#### python src/models/build_model_LGBM.py --help

Trains a LightGBM classifier on the basic features and saves the model and confusion matrix to a file.

```
usage: build_model_LGBM.py [-h] [-i INPUT] [-o OUTPUT] [-c MATRIX]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input training features .csv filepath
  -o OUTPUT, --output OUTPUT
                        output model .pkl filepath
  -c MATRIX, --matrix MATRIX
                        output confusion matrix .csv filepath
```
    - `-i`: input .csv training features file to use for model training and validation.
    - `-o`: output .pkl model filepath - where to save the trained model using joblib.
    - `-c`: output confusion matrix .csv filepath - where to save the confusion matrix file.

#### python src/features/feature_generation_extended.py --help

Generates the extended features to use for feature discovery.

```
usage: feature_generation_extended.py [-h] [-i INPUT] [-d OUTPUT_DIR] [-o FILE_NAME] [-e EEG] [-c ECG] [-l LABELS]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input .edf filepath
  -d OUTPUT_DIR, --output-dir OUTPUT_DIR
                        output folder filepath (will create an ECG and EEG subfolder)
  -o FILE_NAME, --output-file-name FILE_NAME
                        name of output file
  -e EEG, --eeg EEG     channel name for eeg
  -c ECG, --ecg ECG     channel name for ecg
  -l LABELS, --labels LABELS
                        optional .csv filepath with 1Hz labels
```
    - `-i`: input .edf file to use for feature generation. This must have an ECG and EEG channel, specified with the -e and -c options
    - `-d`: output directory to create an ECG and EEG feature discovery .csv
    - `-o`: filename prefix to use when saving the ECG and EEG feature .csv files
    - `-e`: channel to use for EEG
    - `-c`: channel to use for ECG
    - `-l`: filepath to 1Hz hypnogram labels to add as a 'Simple.Sleep.Code' column in feature dataframe .csv

#### python src/models/build_extended_model_LGBM.py --help

Builds a model using all of the features generated in the feature_generation_extended

```
usage: build_extended_model_LGBM.py [-h] [-b BASIC_FEATURES] [-e EEG_FEATURES] [-c ECG_FEATURES] [-o OUTPUT] [-m MATRIX] [-v VIZ_FOLDER]

options:
  -h, --help            show this help message and exit
  -b BASIC_FEATURES, --basic_features BASIC_FEATURES
                        basic training features .csv filepath (for GyrZ, MagZ, ODBA, Pressure)
  -e EEG_FEATURES, --eeg_features EEG_FEATURES
                        input eeg features .csv filepath (containing all concatenated EPOCH & WELCH settings)
  -c ECG_FEATURES, --ecg_features ECG_FEATURES
                        input ecg heart rate features .csv filepath (containing all concatenated EPOCH & WELCH settings)
  -o OUTPUT, --output OUTPUT
                        output model .pkl filepath
  -m MATRIX, --matrix MATRIX
                        output confusion matrix .csv filepath
  -v VIZ_FOLDER, --viz_folder VIZ_FOLDER
                        output folder for feature importance visualizations
```
    - `-b`: basic features .csv filepath (for GyrZ, MagZ, ODBA, Pressure)
    - `-e`: EEG features .csv filepath (output from feature_generation_extended)
    - `-c`: ECG features .csv filepath (output from feature_generation_extended)
    - `-o`: output model .pkl filepath
    - `-m`: output confusion matrix .csv filepath
    - `-v`: output folder in which to save feature importance visualizations

#### python src/models/build_refined_model_LGBM.py --help

Builds a model using only the best feature settings for each sleep state. For example, if the features_extended generates EEG features for every epoch size in (16, 32, 64, 128, 256, 512) and every welch size in (1, 2, 4, 8, 16), this results in 30 combinations of settings for generating features. The refined model will only contain the best five settings: the best pairing for active waking, the best pairing for quiet waking, the best for drowsiness, and SWS, and REM. It will then do the same for the heart rate features, picking only the top five settings.

```
usage: build_refined_model_LGBM.py [-h] [-i INPUT] [-o OUTPUT]

options:
  -h, --help            show this help message and exit
  -b BASIC_FEATURES, --basic_features BASIC_FEATURES
                        basic training features .csv filepath (for GyrZ, MagZ, ODBA, Pressure)
  -e EEG_FEATURES, --eeg_features EEG_FEATURES
                        input eeg features .csv filepath (containing all concatenated EPOCH & WELCH settings)
  -c ECG_FEATURES, --ecg_features ECG_FEATURES
                        input ecg heart rate features .csv filepath (containing all concatenated EPOCH & WELCH settings)
  -f FEATURES, --features FEATURES
                        output features .csv filepath
  -o OUTPUT, --output OUTPUT
                        output model .pkl filepath
  -m MATRIX, --matrix MATRIX
                        output confusion matrix .csv filepath
  -a SETTING_ACCURACIES, --setting_accuracies SETTING_ACCURACIES
                        output setting accuracies .csv filepath
```
    - `-b`: basic features .csv filepath (for GyrZ, MagZ, ODBA, Pressure)
    - `-e`: EEG features .csv filepath (output from feature_generation_extended)
    - `-c`: ECG features .csv filepath (output from feature_generation_extended)
    - `-f`: output model .pkl filepath
    - `-o`: output refined features .csv filepath - this will save a subset of the extended features, only the columns calculated using the "top 5" settings
    - `-m`: output confusion matrix .csv filepath
    - `-a`: output setting accuracies .csv filepath - this will save the accuracies of the models fit only using one setting for EEG and one setting for ECG

## Start to finish - Running the pipeline on a new seal (or a new animal)

1. This requires an EDF file. To start, make sure you know what channels are inside your input .EDF file. The most important for this pipeline are the desired EEG and ECG channels, as well as Pressure (or Depth), MagZ, and ODBA (if applicable). To set these channels, create a JSON file named config.json (you can put it anywhere but inside data/interim would work well), with the following content:
    - You can change any preferences you would like to, but most importantly make sure to set *ECG Channel* and *EEG Channel*. For Pressure, MagZ, and ODBA, if you have the channel data in your EDF, then change the value to whatever your channel is named. If you do not have one or all of these channels, you can set the value to *null* (without quotes, otherwise the program will look for a channel named "null").

```
{
    "ECG Channel": "ECG_Raw_Ch1",
    "EEG Channel": "EEG_ICA5",
    "Pressure Channel": "Pressure",
    "GyrZ Channel": "GyrZ",
    "ODBA Channel": "ODBA",
    "Step Size": 1,
    "Heart Rate Search Radius": 200,
    "Heart Rate Filter Threshold": 200,
    "Pressure Freq": 25,
    "Pressure Calculation Window": 30,
    "ODBA Freq": 25,
    "ODBA Calculation Window": 30,
    "GyrZ Freq": 25,
    "GyrZ Calculation Window": 30,
    "YASA EEG Epoch Window Size": 30,
    "YASA EEG Welch Window Size": 4,
    "YASA EEG Step Size": 1,
    "YASA Heart Rate Epoch Window Size": 60,
    "YASA Heart Rate Welch Window Size": 4,
    "YASA Heart Rate Step Size": 1
}
```

2. Run `python feature_generation.py [-i PATH_TO_INPUT_EDF] [-o PATH_TO_OUTPUT_FEATURES_CSV] [-l PATH_TO_HYNPOGRAM_CSV] [-c PATH_TO_CONFIG_JSON]`
    - -l is optional, so if you wish to generate features and make predictions with a pre-built model, use it without a hypnogram file, but if you wish to build a model with these generate features, you must provide a hypnogram file to -l for the next steps to function properly

3. Run `python build_model_LGBM.py build_model_LGBM.py [-i INPUT] [-o OUTPUT] [-c MATRIX]`

4. Run `python feature_generation_extended.py [-i INPUT] [-d OUTPUT_DIR] [-o FILE_NAME] [-e EEG] [-c ECG] [-l LABELS]`
    - This step allows you to build a model that include the features calculated at a wide variety of epoch and welch window sizes, and also also you to calculated the "refined model", which only include the best welch and epoch sizes for each sleep state

5. Run `python build_extended_model_LGBM.py [-b BASIC_FEATURES] [-e EEG_FEATURES] [-c ECG_FEATURES] [-o OUTPUT] [-m MATRIX] [-v VIZ_FOLDER]`
    - This step builds a model that includes all the features generation in step 4 (as long as LightGBM finds the features to be useful)
    - This step also generates all of the feature importance plots that can be seen (as generate from the seal Wednesday) in the notebook *04_lgbm_feature_discovery.ipynb*

6. Run `python build_refined_model_LGBM.py build_refined_model_LGBM.py [-i INPUT] [-o OUTPUT]`
    - This step will output a features file containing only features generated with a setting found to be the most important for one of the sleep states
    - It will also save a confusion matrix to wherever the model is saved.
