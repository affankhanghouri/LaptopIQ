from src.entity_component.config_entity import DataIngestionConfig
from src.entity_component.artifact_entity import DataIngestionArtifact
from src.constants_component import *
from src.logging_component import logger
from src.exception_component import MyException
from src.data_access.get_data_in_correct_order_module import GetData
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config=DataIngestionConfig):
        """
        Initialize DataIngestion class with config and artifact.
        """
        self.data_ingestion_config = data_ingestion_config
        self.data_ingestion_artifact = DataIngestionArtifact(
            self.data_ingestion_config.training_file_path,
            self.data_ingestion_config.test_file_path
        )

    def import_data_and_put_into_feature_store(self) -> pd.DataFrame:
        """
        Import data from MongoDB and store it in feature store as CSV.
        """
        try:
            logger.info(
                "Entered DataIngestion.import_data_and_put_into_feature_store method"
            )

            # Step 1: Get data from MongoDB
            gf = GetData()
            dataFrame = gf.get_data_in_correct_form(
                database_name=DATABASE_NAME, collection_name=COLLECTION_NAME
            )
            logger.info(f"Shape of dataframe : {dataFrame.shape}")

           
           
            # Step 2: Ensure feature store directory exists
            feature_store_dir = self.data_ingestion_config.data_ingestion_feature_store_dir
            os.makedirs(feature_store_dir, exist_ok=True)

            # Step 3: Save dataframe to feature store CSV
            feature_store_file_path = self.data_ingestion_config.data_ingestion_feature_store_file
            logger.info(f"Saving imported data to feature store at: {feature_store_file_path}")
            dataFrame.to_csv(feature_store_file_path, index=False, header=True)

            return dataFrame

        except Exception as e:
            logger.error(
                "Error occurred in DataIngestion.import_data_and_put_into_feature_store"
            )
            raise MyException(e, sys)


    def split_data_as_train_test(self, dataFrame: pd.DataFrame):
        """
        Split the dataframe into train and test sets based on configured ratio.
        """
        try:
            logger.info("Entered DataIngestion.split_data_as_train_test method")

            # Step 1: Train-test split
            train_set, test_set = train_test_split(
                dataFrame,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42  # For reproducibility
            )
            logger.info("Performed train-test split on the dataframe")

            # Step 2: Ensure directories for train/test files exist
            train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            test_dir = os.path.dirname(self.data_ingestion_config.test_file_path)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Step 3: Export train and test sets
            logger.info("Exporting train and test datasets to CSV")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logger.info("Exported train and test datasets successfully")

        except Exception as e:
            logger.error("Error occurred in DataIngestion.split_data_as_train_test")
            raise MyException(e, sys) from e



    def Initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process: fetch, feature store, split.
        """
        logger.info("Entered DataIngestion.Initiate_data_ingestion method")

        try:
            # Step 1: Import data and put into feature store
            dataframe = self.import_data_and_put_into_feature_store()

            # Step 2: Split data into train and test sets
            self.split_data_as_train_test(dataframe)
            logger.info("Train-test split completed successfully")

            # Step 3: Log and return artifact
            logger.info(f"Data ingestion artifact: {self.data_ingestion_artifact}")
            return self.data_ingestion_artifact

        except Exception as e:
            logger.error("Error occurred in DataIngestion.Initiate_data_ingestion")
            raise MyException(e, sys)
