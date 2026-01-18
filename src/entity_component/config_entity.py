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



# ------------------------------------ Model Trainer ------------------------



@dataclass
class ModelTrainerConfig:
    model_trainer_dir:  str = os.path.join(
        pipeline.artifact_dir,
        pipeline.current_date_time,
        MODEL_TRAINER_DIR_NAME
    )
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    expected_r2_score: float = EXPECTED_SCORE
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH






@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME



@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME





@dataclass
class LaptopPricePredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME
