from datetime import datetime

CURRENT_DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
ARTIFACT_DIR = 'artifact'


# for mongo db related
DATABASE_NAME = 'laptop_price_dataset_DB'
COLLECTION_NAME= 'laptop_price_dataset'


"data ingestion related constants"
DATA_INGESTION_DIR_NAME = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE = 'feature_store' # here raw data will be save that will come from the Databse
DATA_INGESTION_INGESTED_DIR_NAME = 'ingestion'
TRAINING_FILE_PATH_NAME = 'train.csv'
TEST_FILE_PATH_NAME  = 'test.csv'
TRAIN_TEST_SPLIT_RATIO = 0.2






