from dataclasses import dataclass
from src.constants_component import *
import os

@dataclass
class Pipeline:
    artifact_dir : str = ARTIFACT_DIR
    current_date_time : str = CURRENT_DATE_TIME


pipeline = Pipeline()



@dataclass
class DataIngestionConfig:

    data_ingestion_dir_name : str = os.path.join(pipeline.artifact_dir , pipeline.current_date_time , DATA_INGESTION_DIR_NAME)
    data_ingestion_feature_store_dir : str = os.path.join(data_ingestion_dir_name , DATA_INGESTION_FEATURE_STORE)
    data_ingestion_feature_store_file: str = os.path.join( data_ingestion_feature_store_dir , "laptop.csv")
    training_file_path : str = os.path.join(data_ingestion_dir_name , DATA_INGESTION_INGESTED_DIR_NAME , TRAINING_FILE_PATH_NAME)
    test_file_path : str = os.path.join(data_ingestion_dir_name , DATA_INGESTION_INGESTED_DIR_NAME , TEST_FILE_PATH_NAME)
    train_test_split_ratio : float = TRAIN_TEST_SPLIT_RATIO


@dataclass
class DataValidationConfig:

    data_validation_dir_name = os.path.join(pipeline.artifact_dir , DATA_VALIDATION_DIR_NAME)


