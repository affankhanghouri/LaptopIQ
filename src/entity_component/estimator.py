
from src.logging_component import logger
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from src.exception_component import MyException
import sys



class ModelPredictor:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        A generic wrapper for ML models to handle transformation and prediction.
        :param preprocessing_object: Input preprocessing Pipeline object (e.g., sklearn Pipeline)
        :param trained_model_object: Trained model object (e.g., XGBoost, Random Forest)
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> object:
        """
        Transforms raw input data and returns predictions.
        """
        # Dynamically get the class name for cleaner logging
        class_name = self.__class__.__name__
        logger.info(f"Entered predict method of {class_name}")

        try:
            logger.info("Applying preprocessing transformations to input data")
            transformed_features = self.preprocessing_object.transform(dataframe)

            logger.info("Generating predictions from the trained model")
            predictions = self.trained_model_object.predict(transformed_features)

            logger.info(f"Successfully completed prediction in {class_name}")
            return predictions

        except Exception as e:
            # sys is used here to provide detailed traceback information
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"ModelPredictor(model={type(self.trained_model_object).__name__})"

    def __str__(self):
        return f"Generic Predictor for {type(self.trained_model_object).__name__}"