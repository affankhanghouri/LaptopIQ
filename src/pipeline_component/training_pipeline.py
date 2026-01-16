import sys

from src.Data_Ingestion_component import DataIngestion
from src.Data_validation_component import DataValidation
from src.Data_transformation_component import DataTransformation
from src.Model_Trainer_component import ModelTrainer
from src.Model_evaluation_component import ModelEvaluation

from src.entity_component.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

from src.entity_component.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact , 
    ModelEvaluationArtifact
    
)

from src.logging_component import logger
from src.exception_component import MyException


class TrainingPipeline:
    """
    TrainingPipeline orchestrates the end-to-end ML training workflow.
    Each stage:
    - Takes config(s)
    - Consumes previous stage artifact(s)
    - Produces a new artifact
    """

    def __init__(self):
        try:
            logger.info("Initializing TrainingPipeline")

            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
            self.model_evaluation_config = ModelEvaluationConfig()

        except Exception as e:
            raise MyException(e, sys)
        



    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Stage 1: Data Ingestion
        """
        try:
            logger.info("Starting data ingestion stage")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.Initiate_data_ingestion()

            logger.info("Data ingestion completed successfully")
            logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e



    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ) -> DataValidationArtifact:
        """
        Stage 2: Data Validation
        """
        try:
            logger.info("Starting data validation stage")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config,
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logger.info("Data validation completed successfully")
            logger.info(f"Data Validation Artifact: {data_validation_artifact}")

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e



       

       
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise MyException(e, sys)
        

    
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logger.info("Started model training stage")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            artifact = model_trainer.initiate_model_trainer()
            logger.info("Completed model training stage")
            return artifact
        except Exception as e:
            raise MyException(e, sys)



    
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                               data_ingestion_artifact=data_ingestion_artifact,
                                               model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)     
    
        

        
    def run_pipeline(self):
        """
        Runs the complete training pipeline sequentially.
        """
        try:
            logger.info("Training pipeline started")

            # =========================
            # Stage 1: Data Ingestion
            # =========================
            data_ingestion_artifact = self.start_data_ingestion()

            # =========================
            # Stage 2: Data Validation
            # =========================
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
           

            if not data_validation_artifact.validation_status:
                raise Exception(
                    f"Data validation failed: {data_validation_artifact.message}"
                )
            

            #================================
            # Stage 3: Data Transformation
            #=================================

            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)



            # ==================================
            # stage 4  : Model trainer
            # ==================================

            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            # ==================================
            # Model Evaluation 
            # ====================================

            
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            




            logger.info("Training pipeline completed successfully")

        except Exception as e:
            logger.error("Training pipeline failed")
            raise MyException(e, sys)
