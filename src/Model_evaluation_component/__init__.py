
import sys
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import r2_score

from src.entity_component.config_entity import ModelEvaluationConfig
from src.entity_component.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.exception_component import MyException
from src.constants_component import *
from src.entity_component.s3_estimator import LaptopTrainedModelEstimator
from src.entity_component.estimator import ModelPredictor

from src.logging_component import logger



@dataclass
class EvaluateModelResponse:
    trained_model_r2_score: float
    best_model_r2_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logger.info("Initializing ModelEvaluation class")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            logger.exception("Error during initialization of ModelEvaluation")
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[LaptopTrainedModelEstimator]:
        """
        Fetches the production model from S3 if available
        """
        try:
            logger.info("Entering get_best_model()")
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            estimator = LaptopTrainedModelEstimator(bucket_name=bucket_name, model_path=model_path)

            if estimator.is_model_present():
                logger.info(f"Production model found at {model_path}")
                return estimator
            logger.info("No production model found")
            return None
        except Exception as e:
            logger.exception("Error while fetching the best model from S3")
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate trained regression model vs production model using r2_score
        """
        try:
            logger.info("Entering evaluate_model()")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            X, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            logger.info(f"Test data loaded: {X.shape[0]} samples, {X.shape[1]} features")

            trained_model_r2_score = self.model_trainer_artifact.metric_artifact.r2_score
            logger.info(f"Trained model r2_score: {trained_model_r2_score:.4f}")

            best_model_r2_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_pred_best_model = best_model.predict(X)
                best_model_r2_score = r2_score(y, y_pred_best_model)
                logger.info(f"Best model r2_score: {best_model_r2_score:.4f}")

            tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score
            is_model_accepted = trained_model_r2_score > tmp_best_model_score
            difference = trained_model_r2_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_r2_score=trained_model_r2_score,
                best_model_r2_score=best_model_r2_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

            logger.info(f"Evaluation Result -> Accepted: {is_model_accepted}, Difference: {difference:.4f}")
            logger.info("Exiting evaluate_model()")
            return result

        except Exception as e:
            logger.exception("Error during model evaluation")
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Orchestrates the full model evaluation process
        """
        try:
            logger.info("Starting initiate_model_evaluation()")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logger.info(f"Model evaluation artifact created: {model_evaluation_artifact}")
            logger.info("Exiting initiate_model_evaluation()")
            return model_evaluation_artifact

        except Exception as e:
            logger.exception("Error during initiate_model_evaluation")
            raise MyException(e, sys) from e
