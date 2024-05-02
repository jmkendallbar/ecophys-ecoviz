SRC_DIR=src

EDF_DATA_DIR=data/raw/01_edf_data
HYPNO_DATA_DIR=data/raw/02_hypnogram_data
FEAT_DISC_DIR=data/interim/feature_discovery

FEATURES_DIR=data/processed/features
FEATURES_EXT_DIR=data/processed/features_extended

MODELS_DIR=models

all: download features model visualization evaluation


# Download data
data/raw/01_edf_data/test12_Wednesday_05_ALL_PROCESSED.edf:
	wget -O $(EDF_DATA_DIR)/test12_Wednesday_05_ALL_PROCESSED.edf https://figshare.com/ndownloader/files/45224134
	wget -O $(HYPNO_DATA_DIR)/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv https://figshare.com/ndownloader/files/45224131

download: data/raw/01_edf_data/test12_Wednesday_05_ALL_PROCESSED.edf

# Simple features - Step 1
data/processed/features/test12_Wednesday_05_features.csv: download
	python src/features/feature_generation.py

features: data/processed/features/test12_Wednesday_05_features.csv

models/lgbm_model_basic.pkl: features
	python src/models/build_model_LGBM.py

model: models/lgbm_model_basic.pkl

# Extended features - Optional step 2
data/interim/feature_discovery/EEG/Wednesday_feature_discovery_EEG.csv: download
	python src/features/feature_generation_extended.py

features_extended: data/interim/feature_discovery/EEG/Wednesday_feature_discovery_EEG.csv features

models/lgbm_model_extended.pkl: features_extended
	python src/models/build_extended_model_LGBM.py

model_extended: models/lgbm_model_extended.pkl 

# Refined model using features generated with best epoch & welch settings - Step 3 (will use default output unless step 2 has run)
models/lgbm_model_refined.pkl:
	python src/models/build_refined_model_LGBM.py

refined_model: models/lgbm_model_refined.pkl 

# Visualization targets
visualize_features: $(FEATURES_DIR)/ $(HYPNOGRAM_FILE)

visualize_predictions: $(FEATURES_FILE) $(HYPNOGRAM_FILE) $(MODEL_FILE)

visualize_feature_importance: $(FEATURE_IMPORTANCE_FILE)

evaluation: $(FEATURES_FILE) $(HYPNOGRAM_FILE) $(MODEL_FILE)

notebook:
	jupyter notebook --notebook-dir=$(pwd)