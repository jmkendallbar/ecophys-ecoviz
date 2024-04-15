SRC_DIR=src
DATA_DIR=data
MODELS_DIR=models

EDF_FILE=$(MY_EDF_FILE)
FEATURES_FILE=$(MY_FEATURES_FILE)

all: features model visualization evaluation

features: $(EDF_FILE)

model: $(FEATURES_FILE)

visualization: $(FEATURES_FILE) $(MODEL_FILE)

evaluation: $(FEATURES_FILE) $(MODEL_FILE)