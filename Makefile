SRC_DIR=src

EDF_DATA_DIR=data/raw/01_edf_data
HYPNO_DATA_DIR=data/raw/02_hypnogram_data

FEATURES_DIR=data/processed/features
FEATURES_EXT_DIR=data/processed/features_extended

MODELS_DIR=models

all: download features model visualization evaluation


# Download data
data/raw/01_edf_data/test12_Wednesday_05_ALL_PROCESSED.edf:
	python src/data/download_data.py

download: data/raw/01_edf_data/test12_Wednesday_05_ALL_PROCESSED.edf

# Simple features
data/processed/features/test12_Wednesday_05_features.csv: download
	python src/features/feature_generation.py

features: data/processed/features/test12_Wednesday_05_features.csv

# Extended features
data/processed/features_extended/test12_Wednesday_05_features_extended.csv: download
	python src/features/feature_generation_extended.py

features_extended: data/processed/features_extended/test12_Wednesday_05_features_extended.csv

# Build model
model: data/processed/features/test12_Wednesday_05_features.csv

model_extended: $(MODELS_DIR)/lgbm_model_extended.pkl

visualize_features: $(FEATURES_DIR)/ $(HYPNOGRAM_FILE)

visualize_predictions: $(FEATURES_FILE) $(HYPNOGRAM_FILE) $(MODEL_FILE)

visualize_feature_importance: $(FEATURE_IMPORTANCE_FILE)

evaluation: $(FEATURES_FILE) $(HYPNOGRAM_FILE) $(MODEL_FILE)

notebook:
	jupyter notebook --notebook-dir=$(pwd)