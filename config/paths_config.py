import os


# DATA INGESTION ########################

RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


# DATA PROCESSING ######################

PROCESSED_DIR = "artifacts/processesed_data"
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")

# MODEL TAINING ########################

MODEL_OUTPUT_PATH = "artifacts/models/lgbm_model_2.pkl"

# PREPROCESING PIPELINE ################

PREPROCESSING_PIPELINE_PATH = "artifacts/pipeline/preprocessing_pipeline.pkl"
# FEATURE_SELECTION_PIPELINE_PATH = "artifacts/pipeline/feature_selection_pipeline.pkl"
