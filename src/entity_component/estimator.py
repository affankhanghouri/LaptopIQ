from src.logging_component import logger
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from src.exception_component import MyException
import sys
from src.Data_transformation_component import DataTransformation
import numpy as np

class ModelPredictor:
    """
    Generic wrapper for ML models to handle feature engineering,
    preprocessing, and prediction safely.
    """
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Preprocessing pipeline (sklearn Pipeline / ColumnTransformer)
        :param trained_model_object: Trained ML model (XGBoost, RandomForest, etc.)
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Predict using the trained model and preprocessing pipeline.
        Handles missing columns in prediction data.
        """
        logger.info(f"Entered predict method of {self.__class__.__name__}")

        try:
            # -----------------------------
            # Feature engineering
            # -----------------------------
            logger.info("Applying feature engineering to input data")
            dataframe = DataTransformation.feature_engineering_for_prediction(
                data=dataframe
            )

            # -----------------------------
            # Ensure all required columns exist
            # -----------------------------
            required_columns = (
                DataTransformation.schema["numerical_features"] +
                DataTransformation.schema["categorical_features"]
            )

            for col in required_columns:
                if col not in dataframe.columns:
                    if col in DataTransformation.schema["categorical_features"]:
                        dataframe[col] = "unknown"
                    else:
                        dataframe[col] = 0

            # Reorder columns exactly as in training
            dataframe = dataframe[required_columns]

            # -----------------------------
            # Preprocessing
            # -----------------------------
            logger.info("Applying preprocessing transformations")
            transformed_features = self.preprocessing_object.transform(dataframe)

            # -----------------------------
            # Prediction
            # -----------------------------
            logger.info("Generating predictions from the trained model")
            predictions = self.trained_model_object.predict(transformed_features)

            # -----------------------------
            # Reverse log transform
            # -----------------------------
            predictions = np.exp(predictions)

            logger.info("Successfully completed prediction")
            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise MyException(e, sys) from e

    def __repr__(self):
        return f"ModelPredictor(model={type(self.trained_model_object).__name__})"

    def __str__(self):
        return f"Generic Predictor for {type(self.trained_model_object).__name__}"
