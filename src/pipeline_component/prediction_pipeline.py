# pipeline_component/prediction_pipeline.py

import sys
from pandas import DataFrame
from src.exception_component import MyException
from src.logging_component import logger
from src.entity_component.config_entity import LaptopPricePredictorConfig
from src.entity_component.s3_estimator import LaptopTrainedModelEstimator


class LaptopData:
    """
    Class to handle the input data for laptop prediction.
    """

    def __init__(
        self,
        Company: str,
        TypeName: str,
        Inches: float,
        ScreenResolution: str,
        Cpu: str,
        Ram: int,
        Memory: str,
        Gpu: str,
        OpSys: str,
        Weight: float,
    ):
        try:
            self.Company = Company
            self.TypeName = TypeName
            self.Inches = Inches
            self.ScreenResolution = ScreenResolution
            self.Cpu = Cpu
            self.Ram = Ram
            self.Memory = Memory
            self.Gpu = Gpu
            self.OpSys = OpSys
            self.Weight = Weight

        except Exception as e:
            raise MyException(e, sys) from e

    def get_input_data_frame(self) -> DataFrame:
        """
        Converts input data into a pandas DataFrame with column names
        matching the dataset schema.
        """
        try:
            input_data = {
                "Company": [self.Company],
                "TypeName": [self.TypeName],
                "Inches": [self.Inches],
                "ScreenResolution": [self.ScreenResolution],
                "Cpu": [self.Cpu],
                "Ram": [self.Ram],
                "Memory": [self.Memory],
                "Gpu": [self.Gpu],
                "OpSys": [self.OpSys],
                "Weight": [self.Weight],
            }

            logger.info("Laptop input DataFrame created")
            return DataFrame(input_data)

        except Exception as e:
            raise MyException(e, sys) from e


class LaptopPredictor:
    """
    Class to handle the prediction pipeline for laptops.
    """

    def __init__(
        self,
        prediction_pipeline_config: LaptopPricePredictorConfig = LaptopPricePredictorConfig(),
    ):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config

        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, dataframe: DataFrame):
        """
        Returns the model prediction for the given input DataFrame.
        """
        try:
            logger.info("Entered predict method of LaptopPredictor")

            # Load the LaptopEstimator model (preprocessing + trained model)
            model: LaptopTrainedModelEstimator = LaptopTrainedModelEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            ).load_model()

            # Perform prediction
            prediction = model.predict(dataframe)

            logger.info("Laptop prediction completed successfully")
            return prediction

        except Exception as e:
            raise MyException(e, sys) from e
