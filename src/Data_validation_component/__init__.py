from src.entity_component.config_entity import DataValidationConfig
from src.entity_component.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.logging_component import logger
from src.exception_component import MyException
from src.utils_component.main_utils import read_yaml_file, write_yaml_file
from src.constants_component import *
import sys
import pandas as pd
from pandas import DataFrame

from evidently import Report
from evidently.presets import DataDriftPreset


class DataValidation:
    """
    DataValidation class handles:
    1. Validating dataset columns against schema.
    2. Detecting data drift between training and testing datasets.
    3. Returning DataValidationArtifact with status, message, and drift report path.
    """

    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact
    ):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            logger.error("Error occurred in DataValidation constructor")
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        """Read CSV file into DataFrame."""
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(self, df: DataFrame) -> bool:
        """
        Validate that the dataframe has the required number of columns
        as per schema.
        """
        try:
            logger.info("Validating number of columns")
            required_columns = self._schema_config.get("columns", [])
            status = len(df.columns) == len(required_columns)
            logger.info(f"Required columns present: {status}")
            return status
        except Exception as e:
            logger.error("Error occurred in validate_number_of_columns")
            raise MyException(e, sys)

    def is_all_columns_present(self, df: DataFrame) -> bool:
        """
        Check if all required columns exist in the dataframe.
        """
        try:
            required_columns = self._schema_config.get("columns", [])
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            logger.error("Error occurred in is_all_columns_present")
            raise MyException(e, sys)
        





        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Perform full data validation:
        1. Column validation
        2. Dataset drift detection
        Returns DataValidationArtifact with status, message, and drift report path.
        """
        try:
            logger.info("Starting data validation process")
            validation_error_msg = ""

            # Load train and test datasets
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            logger.info(f"Train columns: {list(train_df.columns)}")
            logger.info(f"Expected columns: {list(self._schema_config['columns'])}")

            # Column count validation
            if not self.validate_number_of_columns(df=train_df):
                validation_error_msg += "Train dataframe has missing columns. "
            if not self.validate_number_of_columns(df=test_df):
                validation_error_msg += "Test dataframe has missing columns. "

            # Column existence validation
            if not self.is_all_columns_present(df=train_df):
                validation_error_msg += "Train dataframe missing required columns. "
            if not self.is_all_columns_present(df=test_df):
                validation_error_msg += "Test dataframe missing required columns. "

            # Determine validation status
            validation_status = len(validation_error_msg) == 0

          
            # Return artifact
            return DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
               
            )

        except Exception as e:
            logger.error("Error occurred in initiate_data_validation")
            raise MyException(e, sys)
