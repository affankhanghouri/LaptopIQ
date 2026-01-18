from src.entity_component.config_entity import ModelPusherConfig
from src.entity_component.artifact_entity import ModelPusherArtifact , ModelEvaluationArtifact
from src.logging_component import logger
from src.exception_component import MyException
from src.entity_component.s3_estimator import LaptopTrainedModelEstimator
from src.cloud_storage.aws_storage import SimpleStorageService
import sys

class ModelPusher :

    def __init__(self , model_pusher_config : ModelPusherConfig , model_evaluation_artifact : ModelEvaluationArtifact) :


        self.model_pusher_config = model_pusher_config
        self.model_evaluation_artifact =  model_evaluation_artifact 
        self.s3 = SimpleStorageService()
        self.laptop_price_estimator = LaptopTrainedModelEstimator(bucket_name=self.model_pusher_config.bucket_name
                                                                  , model_path=self.model_pusher_config.s3_model_key_path)
        

    def initiate_model_pusher(self) -> ModelPusherArtifact :

        logger.info("Entered initiate_model_pusher method of ModelPusher class")

        try :
            logger.info("Uploading artifact folder to s3 bucket ")

            self.laptop_price_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(self.model_pusher_config.bucket_name , self.model_pusher_config.s3_model_key_path)   

            
            logger.info("Uploaded artifacts folder to s3 bucket")
            logger.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logger.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        
        except Exception as e:
            logger.error('ERROR OCCURED IN MODEL PUSHER CLASS in initiate_model_pusher methdo')
            raise MyException(e, sys) from e 
