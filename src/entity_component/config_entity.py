from dataclasses import dataclass
from src.constants_component import *
import os


@dataclass
class Pipeline:
    artifact_dir: str = ARTIFACT_DIR
    current_date_time: str = CURRENT_DATE_TIME


pipeline = Pipeline()


# --------------------- DATA INGESTION --------------------- #
@dataclass
class DataIngestionConfig:

    data_ingestion_dir_name: str = os.path.join(
        pipeline.artifact_dir,
        pipeline.current_date_time,
        DATA_INGESTION_DIR_NAME
    )

    data_ingestion_feature_store_dir: str = os.path.join(
        data_ingestion_dir_name,
        DATA_INGESTION_FEATURE_STORE
    )

    data_ingestion_feature_store_file: str = os.path.join(
        data_ingestion_feature_store_dir,
        "laptop.csv"
    )

    training_file_path: str = os.path.join(
        data_ingestion_dir_name,
        DATA_INGESTION_INGESTED_DIR_NAME,
        TRAINING_FILE_PATH_NAME
    )

    test_file_path: str = os.path.join(
        data_ingestion_dir_name,
        DATA_INGESTION_INGESTED_DIR_NAME,
        TEST_FILE_PATH_NAME
    )

    train_test_split_ratio: float = TRAIN_TEST_SPLIT_RATIO


# --------------------- DATA VALIDATION --------------------- #
@dataclass
class DataValidationConfig:

    data_validation_dir_name: str = os.path.join(
        pipeline.artifact_dir,
        pipeline.current_date_time,
        DATA_VALIDATION_DIR_NAME
    )


# --------------------- DATA TRANSFORMATION --------------------- #
@dataclass
class DataTransformationConfig:

    data_transformation_dir_name: str = os.path.join(
        pipeline.artifact_dir,
        pipeline.current_date_time,
        DATA_TRANSFORMATION_DIR_NAME
    )

    transformed_train_file_path: str = os.path.join(
        data_transformation_dir_name,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TRAIN_FILE_NAME.replace("csv", "npy")
    )

    transformed_test_file_path: str = os.path.join(
        data_transformation_dir_name,
        DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        TEST_FILE_NAME.replace("csv", "npy")
    )

    transformed_object_file_path: str = os.path.join(
        data_transformation_dir_name,
        DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
        PREPROCSSING_OBJECT_FILE_NAME
    )
