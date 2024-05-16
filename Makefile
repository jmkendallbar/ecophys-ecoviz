SRC_DIR=src

EDF_DATA_DIR=data/raw/01_edf_data
HYPNO_DATA_DIR=data/raw/02_hypnogram_data
FEAT_DISC_DIR=data/interim/feature_discovery
FEATURES_DIR=data/processed/features

MODELS_DIR=models

all: download features model visualization

# Make directories
make_directories:
	mkdir -p $(EDF_DATA_DIR)
	mkdir -p $(HYPNO_DATA_DIR)
	mkdir -p $(FEAT_DISC_DIR)
	mkdir -p $(FEAT_DISC_DIR)/EEG
	mkdir -p $(FEAT_DISC_DIR)/ECG
	mkdir -p $(FEATURES_DIR)
	mkdir -p $(FEATURES_DIR)
	mkdir -p reports/figures/feature_discovery
	mkdir -p reports/figures/model_evaluation

# Download data
$(EDF_DATA_DIR)/test12_Wednesday_05_ALL_PROCESSED.edf:
	wget -O $(EDF_DATA_DIR)/test12_Wednesday_05_ALL_PROCESSED.edf https://figshare.com/ndownloader/files/46036731

$(HYPNO_DATA_DIR)/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv:
	wget -O $(HYPNO_DATA_DIR)/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv https://figshare.com/ndownloader/files/46036728

download: $(EDF_DATA_DIR)/test12_Wednesday_05_ALL_PROCESSED.edf $(HYPNO_DATA_DIR)/test12_Wednesday_06_Hypnogram_JKB_1Hz.csv

# Simple features - Step 1
$(FEATURES_DIR)/test12_Wednesday_07_features_with_labels.csv: download
	python src/features/feature_generation.py

features: $(FEATURES_DIR)/test12_Wednesday_07_features_with_labels.csv

models/lightgbm_model_basic.pkl: $(FEATURES_DIR)/test12_Wednesday_07_features_with_labels.csv
	python src/models/build_model_LGBM.py

model: models/lightgbm_model_basic.pkl

# Extended features - Optional step 2
$(FEAT_DISC_DIR)/EEG/Wednesday_feature_discovery_EEG.csv:
	python src/features/feature_generation_extended.py

features_extended: $(FEAT_DISC_DIR)/EEG/Wednesday_feature_discovery_EEG.csv $(FEAT_DISC_DIR)/ECG/Wednesday_feature_discovery_ECG.csv download features

models/lightgbm_model_extended.pkl:
	python src/models/build_extended_model_LGBM.py

model_extended: models/lightgbm_model_extended.pkl features_extended

# Refined model using features generated with best epoch & welch settings - Step 3 (will use default output unless step 2 has run)
models/lgbm_model_refined.pkl:
	python src/models/build_refined_model_LGBM.py

refined_model: models/lgbm_model_refined.pkl 

# Visualization targets
visualize_basic: $(FEATURES_DIR)/ $(HYPNOGRAM_FILE)

visualize_extended: $(FEATURES_FILE) $(HYPNOGRAM_FILE) $(MODEL_FILE)

visualize_refined: $(FEATURE_IMPORTANCE_FILE)


notebook:
	jupyter notebook --notebook-dir=$(pwd)