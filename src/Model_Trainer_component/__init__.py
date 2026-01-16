from src.Model_Trainer_component.ModelFactoryModule import ModelFactory
from src.utils_component.main_utils import read_yaml_file, load_object
import sys
import os
import numpy as np
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception_component import MyException
from src.logging_component import logger

from src.entity_component.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact
)
from src.entity_component.config_entity import ModelTrainerConfig
from src.entity_component.estimator import ModelPredictor
from src.constants_component import MODEL_SCHEMA_FILE_PATH


class ModelTrainer:
    """
    Production Model Trainer (REGRESSION)
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.model_schema = read_yaml_file(MODEL_SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def load_data(self):
        try:
            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path,
                allow_pickle=True
            )
            test_arr = np.load(
                self.data_transformation_artifact.transformed_test_file_path,
                allow_pickle=True
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return X_train, y_train, X_test, y_test

        except Exception as e:
            raise MyException(e, sys)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        try:
            model_config = self.model_schema["model"]

            logger.info(f"Training model: {model_config['name']}")

            model = ModelFactory.get_model(
                model_config["name"],
                model_config["params"]
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            logger.info(
                f"Regression Metrics -> R2: {r2}, MAE: {mae}, RMSE: {rmse}"
            )

            return model, r2, mae, rmse

        except Exception as e:
            raise MyException(e, sys)

    def save_model(self, model):
        try:
            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            model_predictor = ModelPredictor(
                preprocessing_object=preprocessor,
                trained_model_object=model
            )

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            joblib.dump(
                model_predictor,
                self.model_trainer_config.trained_model_file_path
            )

            logger.info("Regression model saved successfully")

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logger.info("Model Trainer started")

            X_train, y_train, X_test, y_test = self.load_data()

            model, r2, mae, rmse = self.train_and_evaluate(
                X_train, y_train, X_test, y_test
            )

            if r2 < self.model_trainer_config.expected_r2_score:
                raise Exception(
                    f"R2 {r2} is less than expected "
                    f"{self.model_trainer_config.expected_r2_score}"
                )

            self.save_model(model)

            metric_artifact = RegressionMetricArtifact(
                r2_score=r2,
                mean_absolute_error=mae,
                root_mean_squared_error=rmse
            )

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise MyException(e, sys)
