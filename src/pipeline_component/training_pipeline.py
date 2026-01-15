import sys
from src.Data_Ingestion_component import DataIngestion
from src.entity_component.config_entity import DataIngestionConfig 
from src.entity_component.artifact_entity import DataIngestionArtifact
from src.logging_component import logger
from src.exception_component import MyException


class TrainingPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def start_data_ingestion (self) -> DataIngestionArtifact:

        try:
            logger.info("Entered the start_data_ingestion method of TrainPipeline class")
            logger.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.Initiate_data_ingestion()
            logger.info("Got the train_set and test_set from mongodb")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e   


    def run_pipeline(self):

        
        try:
            # Stage 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
        
            logger.info(f"Pipeline completed successfully.")

            
        except Exception as e:
            raise MyException(e, sys)    

    