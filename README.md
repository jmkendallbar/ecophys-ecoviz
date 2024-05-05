
# EcoViz Ecophysiology Seal Sleep Classifier

<!-- badges: start -->
<!-- badges: end -->

This research compendium applies a `lightgbm` gradient boosted decision tree model to score sleep stages in a non-model system, the northern elephant seal (*Mirounga angustirostris*). We apply a three step process to build and refine our model:
- **Step 1:** First, we train and test a model based on features assumed to be important based on previous studies and manual data review
- **Step 2:** Next, we try a naive approach where we calculate features based on a range of settings including epochs from Xs to Xs and FFT window sizes from Xs to Xs. We plot the resulting feature importances and the accuracy of models with different feature settings. We use these to inform features for step 3.
- **Step 3:** Based on the results from Step 2, we calculate the accuracy for detecting each sleep stage given every combination of feature settings. For example, if a model to detect the sleep stage `Active Waking` (alone) was most accurate with epochs and Welch window sizes of 64s and 16s for EEG and 256s and 64s for Heart Rate, we would use those settings 

### Folder structure

```
├── 00_Data_Exploration.ipynb
├── DESCRIPTION
├── README.md
├── data
│   └── README.md
├── figs
│   ├── README.md
│   ├── light-sleep-spectral-power.png
│   ├── rem.png
│   ├── slow-wave-1.png
│   ├── slow-wave-2-spectral-power.png
│   └── slow-wave-2.png
├── notebooks
│   ├── 00_initial_feature_extraction.ipynb
│   ├── 01_initial_models.ipynb
│   ├── 01a_YASA_feature_extraction.ipynb
│   ├── 02_advanced_feature_generation_and_models.ipynb
│   ├── 03_recursive_feature_elimination.ipynb
│   ├── 04_lgbm_feature_viz.ipynb
│   ├── Basic GBTM Example.ipynb
│   ├── Boosted Decision Tree Experiments.ipynb
│   └── README.md
├── output
│   └── README.md
├── paper
│   └── README.md
├── pipeline
│   ├── 00_download_data.R
│   ├── 00_download_data.py
│   ├── _run_pipeline.R
│   └── _run_pipeline.py
├── requirements.txt
├── src
│   ├── Python
│   │   ├── README.md
│   │   ├── feature_extraction.py
│   │   └── feature_generation.py
│   └── R
│       └── README.md
└── traintest
    ├── test
    │   └── README.md
    └── train
        ├── README.md
        └── welch_feature_importance.csv
```

* `data/` - raw data only
* `figs/` - static figures generated during pipeline
* `notebooks/` - literate programming files (e.g., Jupyter, Quarto) explaining the pipeline
* `output/` - derived data generated during pipeline
* `paper/` - the analysis manuscript (preferably in .md format)
* `pipeline/` - pipeline scripts
* `src/` - source files supporting the pipeline (e.g., functions, classes, constants)
* `traintest/` - specifies train/test splits for reproducibility

### Computational environment

```
├── DESCRIPTION
└── requirements.txt
```

These are conventional files used for recording package dependencies. 

`DESCRIPTION` is a file used by R packages to capture package metadata. It is useful for analyses because it records package dependencies following commonly used conventions. Add packages (with version numbers) to the `Imports` field. When other users clone your repo, they can install dependencies by calling `devtools::install_deps()` in R.

`requirements.txt` is a file used to record Python package dependencies. Users can install dependencies by running `pip install -r requirements.txt` at the command line.

### Pipeline steps

Here's how the pipeline flows.

#### 00_initial_feature_extraction.ipynb
Initial notebook used for feature extraction, not the prettiest of notebooks, but walks through heart rate extraction from ECG using peak detection. Also goes through each of the features and plots their distributions to get a first-pass glimpse at the feature space.

#### 01_initial_features_and_models.ipynb
Uses the features created in *00_initial_feature_extraction* to create a few rudimentary scikit-learn machine learning models, including a Support Vector Machine Classifier (SVC), K Nearest Neighbors Classifier (KNN), and Random Forest Classifier (RFC). Also performs grid searching on each of these estimators to find ideal or close-to-ideal parameterizations of these models.

#### 01a_YASA_feature_extraction.ipynb
Has a simple demonstration of applying the YASA sleep staging algorithm built for humans onto one seal. While the performance is not the best, many of the features implemented by YASA were used for this project, and in some cases their code was used as well (but adjusted for seals). Note that YASA requires a LOT of memory and the kernel often crashes for me if I have other notebooks running or too many Google Chrome tabs open.

#### 02_advanced_feature_generation_and_models.ipynb
Uses the (semi) finalized feature extraction code in ***feature_extraction.py*** and ***feature_generation.py*** to generate all the available features for sleep detection created so far. Also creates a RFC using the grid-searched parameters from ***01_initial_features_and_models***, and plots the predictions against the true labels and features for a few naps

#### 03_recursive_feature_elimination.ipynb
Performs recursive feature elimination using the model from ***02_advanced_feature_generation_and_models.ipynb*** to explore which features are the most informative for seal sleep prediction.

#### 04_lgbm_feature_viz.ipynb
Separating EEG bandpower features into smaller bands (from delta to 0-0.5 Hz, 0.5-1Hz, 1-2Hz, etc.) to  Using gradient boosted decision tree lightGBM. Output is a CSV with `WELCH` `_` Resolution (length in seconds of window) `_` Name of feature (if two numbers, range of frequencies; if a name, type of feature).

### Feature extraction usage

To run the full feature extraction start to finish, run from the terminal: `python feature_generation.py <Input_EDF_Filepath> <Output_CSV_Filepath> [Config_Filepath]`. The Config_Filepath should be a path to a json file that has key value pairs like the ones in the DEFAULT_CONFIG variable at the top of the feature_generation.py script. These parameters can be used to adjust window size, step size, and other parameters used by some of the feature functions (although some of the parameters have not been tested thoroughly so adjust them at your own risk). Any config keys not defined by the config file will use the default values defined at the top of feature_generation.py

### Model usage

Something here about the models we tried and what went well

### Use case interpretation

```
├── notebooks
└── paper
```

The literate programming documents in `notebooks/` explain the pipeline components. They should render relatively quickly, so keep long-running commands in the pipeline scripts. Keep rendering times short by using the processed data in `outputs/`.

Rendering the documents in `notebooks/` should be automated by the last pipeline script (e.g., `pipeline/10_render_notebooks.R`). For Jupyter notebooks, render them to HTML using `nbconvert`: `jupyter nbconvert --to html notebooks/your_notebook.ipynb`. For Quarto documents, use `quarto render notebooks/your_notebook.qmd --to html`.

`paper/` contains a manuscript describing your use case. It should preferably be in Markdown or a literate programming script.
