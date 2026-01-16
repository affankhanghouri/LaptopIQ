from datetime import datetime
import os 


# for aws connection
REGION_NAME = "us-east-1"


TARGET_COLUMN = "Price"

CURRENT_DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
ARTIFACT_DIR = 'artifact'
SCHEMA_FILE_PATH = os.path.join('config','schema.yaml')
MODEL_SCHEMA_FILE_PATH = os.path.join('config' ,'model.yaml')


TRAIN_FILE_NAME= 'train.csv'
TEST_FILE_NAME= 'test.csv'
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"


MODEL_FILE_NAME = "model.pkl"



# for mongo db related
DATABASE_NAME = 'laptop_price_dataset_DB'
COLLECTION_NAME= 'laptop_price_dataset'


# data ingestion related constants

DATA_INGESTION_DIR_NAME = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE = 'feature_store' # here raw data will be save that will come from the Databse
DATA_INGESTION_INGESTED_DIR_NAME = 'ingestion'
TRAINING_FILE_PATH_NAME = 'train.csv'
TEST_FILE_PATH_NAME  = 'test.csv'
TRAIN_TEST_SPLIT_RATIO = 0.2




# data validation related constansts "

DATA_VALIDATION_DIR_NAME = 'data_validation'




# data Transformation related constants

DATA_TRANSFORMATION_DIR_NAME= 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR  =  "transformed_object"



# Model trainer realated constants

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")








"""
MODEL EVALUATION related constant 
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.2
MODEL_BUCKET_NAME = "laptop-model2026"
MODEL_PUSHER_S3_KEY = "model-registry"
