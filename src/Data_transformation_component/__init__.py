import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.logging_component import logger
from src.exception_component import MyException

from src.entity_component.config_entity import DataTransformationConfig
from src.entity_component.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact
)

from src.utils_component.helper_functions_ import (
    categorize_cpu,
    extract_memory,
    categorize_gpu,
    categorize_opsys
)

from src.utils_component.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)

from src.constants_component import SCHEMA_FILE_PATH


class DataTransformation:
    """
    Notebook-faithful, schema-driven data transformation.
    """

    # ==================================================
    # CLASS-LEVEL SCHEMA (used by static methods)
    # ==================================================
    schema = read_yaml_file(SCHEMA_FILE_PATH)

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact
    ):
        try:
            self.config = data_transformation_config
            self.ingestion_artifact = data_ingestion_artifact
            logger.info("DataTransformation initialized")
        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    # TRAINING FEATURE ENGINEERING (UNCHANGED)
    # ==================================================
    @staticmethod
    def apply_custom_feature_engineering(data: DataFrame) -> DataFrame:
        """
        EXACTLY mirrors laptop.ipynb feature engineering.
        TRAINING ONLY (Price is required).
        """
        try:
            logger.info("Applying feature engineering (training)")

            data = data.copy()

            # Ram & Weight
            data['Ram'] = data['Ram'].str.replace('GB', '', regex=False).astype(int)
            data['Weight'] = data['Weight'].str.replace('kg', '', regex=False).astype(float)

            # IQR outlier removal (uses Price)
            Q1 = data['Price'].quantile(0.25)
            Q3 = data['Price'].quantile(0.75)
            IQR = Q3 - Q1

            data = data[
                (data['Price'] >= Q1 - 1.5 * IQR) &
                (data['Price'] <= Q3 + 1.5 * IQR)
            ]

            # Screen
            data['Touchscreen'] = data['ScreenResolution'].apply(
                lambda x: 1 if 'Touchscreen' in x else 0
            )
            data['IPS'] = data['ScreenResolution'].apply(
                lambda x: 1 if 'IPS' in x else 0
            )

            res = data['ScreenResolution'].str.split('x', expand=True)
            data['X_res'] = res[0].str.extract(r'(\d+)$').astype(int)
            data['Y_res'] = res[1].astype(int)

            data['ppi'] = (
                ((data['X_res'] ** 2 + data['Y_res'] ** 2) ** 0.5)
                / data['Inches']
            )

            data = data.drop(
                columns=['ScreenResolution', 'X_res', 'Y_res', 'Inches']
            )

            # CPU
            data['Cpu_Category'] = data['Cpu'].apply(categorize_cpu)
            data = data.drop(columns=['Cpu'])

            # Memory
            data['SSD'] = data['Memory'].apply(lambda x: extract_memory(x, 'ssd'))
            data['HDD'] = data['Memory'].apply(lambda x: extract_memory(x, 'hdd'))
            data['Flash_Storage'] = data['Memory'].apply(lambda x: extract_memory(x, 'flash'))
            data['Hybrid'] = data['Memory'].apply(lambda x: extract_memory(x, 'hybrid'))
            data = data.drop(columns=['Memory'])

            # GPU (filtering allowed in training)
            data['Gpu_category'] = data['Gpu'].apply(categorize_gpu)
            data = data[data['Gpu_category'] != 'other']
            data = data.drop(columns=['Gpu'])

            # OS
            data['categorize_opsys'] = data['OpSys'].apply(categorize_opsys)
            data = data.drop(columns=['OpSys'])

            # Schema enforcement
            data = data[DataTransformation.schema["columns_after_transformation"]]

            return data.reset_index(drop=True)

        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    # PREDICTION FEATURE ENGINEERING (SAFE)
    # ==================================================
    @staticmethod
    def feature_engineering_for_prediction(data: DataFrame) -> DataFrame:
        """
        Feature engineering for inference.
        Does NOT use Price.
        Does NOT drop rows.
        """
        try:
            logger.info("Applying feature engineering (prediction)")

            data = data.copy()

            # Ram & Weight
            data['Ram'] = data['Ram'].str.replace('GB', '', regex=False).astype(int)
            data['Weight'] = data['Weight'].str.replace('kg', '', regex=False).astype(float)

            # Screen
            data['Touchscreen'] = data['ScreenResolution'].apply(
                lambda x: 1 if 'Touchscreen' in x else 0
            )
            data['IPS'] = data['ScreenResolution'].apply(
                lambda x: 1 if 'IPS' in x else 0
            )

            res = data['ScreenResolution'].str.split('x', expand=True)
            data['X_res'] = res[0].str.extract(r'(\d+)$').astype(int)
            data['Y_res'] = res[1].astype(int)

            data['ppi'] = (
                ((data['X_res'] ** 2 + data['Y_res'] ** 2) ** 0.5)
                / data['Inches']
            )

            data = data.drop(
                columns=['ScreenResolution', 'X_res', 'Y_res', 'Inches']
            )

            # CPU
            data['Cpu_Category'] = data['Cpu'].apply(categorize_cpu)
            data = data.drop(columns=['Cpu'])

            # Memory
            data['SSD'] = data['Memory'].apply(lambda x: extract_memory(x, 'ssd'))
            data['HDD'] = data['Memory'].apply(lambda x: extract_memory(x, 'hdd'))
            data['Flash_Storage'] = data['Memory'].apply(lambda x: extract_memory(x, 'flash'))
            data['Hybrid'] = data['Memory'].apply(lambda x: extract_memory(x, 'hybrid'))
            data = data.drop(columns=['Memory'])

            # GPU (NO FILTERING)
            data['Gpu_category'] = data['Gpu'].apply(categorize_gpu)
            data = data.drop(columns=['Gpu'])

            # OS
            data['categorize_opsys'] = data['OpSys'].apply(categorize_opsys)
            data = data.drop(columns=['OpSys'])

            # Schema enforcement
            data = data[DataTransformation.schema["columns_after_transformation_for_prediction"]]

            return data.reset_index(drop=True)

        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            num_features = DataTransformation.schema["numerical_features"]
            cat_features = DataTransformation.schema["categorical_features"]

            num_pipeline = Pipeline(
                steps=[("scaler", StandardScaler())]
            )

            cat_pipeline = Pipeline(
                steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
            )

            return ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, num_features),
                    ("cat", cat_pipeline, cat_features)
                ]
            )

        except Exception as e:
            raise MyException(e, sys)

    # ==================================================
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")

            train_df = self.read_data(self.ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.ingestion_artifact.test_file_path)

            train_df = DataTransformation.apply_custom_feature_engineering(train_df)
            test_df = DataTransformation.apply_custom_feature_engineering(test_df)

            target = DataTransformation.schema["target_column"][0]

            X_train = train_df.drop(columns=[target])
            y_train = np.log(train_df[target])

            X_test = test_df.drop(columns=[target])
            y_test = np.log(test_df[target])

            preprocessor = self.get_data_transformer_object()

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            save_object(self.config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.config.transformed_test_file_path, test_arr)

            logger.info("Data transformation completed")

            return DataTransformationArtifact(
                transformed_object_file_path=self.config.transformed_object_file_path,
                transformed_train_file_path=self.config.transformed_train_file_path,
                transformed_test_file_path=self.config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys)
