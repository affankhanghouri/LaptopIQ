from src.configuration_component.mongodb_connection import MongoDB_Client
from src.logging_component import logger
from src.exception_component import MyException
from src.constants_component import *
import pandas as pd

class GetData:
    """
    Class to fetch data from MongoDB and return it as a pandas DataFrame.
    """


    def __init__(self):
        try:
            mongo_db_instance = MongoDB_Client()
            self.mongo_db_client = mongo_db_instance.client
            logger.info("MongoDB client initialized successfully.")
        except Exception as e:
            logger.error("MongoDB connection could not be established.")
            raise MyException("MongoDB connection initialization failed", e)





    def get_data_in_correct_form(self, database_name=DATABASE_NAME, collection_name=COLLECTION_NAME):
        """
        Fetches data from the given MongoDB collection and returns it as a DataFrame.
        Drops the 'Unnamed: 0' column if it exists.
        """
        try:
            logger.info(f"Fetching data from database '{database_name}', collection '{collection_name}'...")
            collection = self.mongo_db_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            # Drop 'Unnamed: 0' column if it exists
            logger.debug(f'these are the columns : {df.columns}')
            if 'Unnamed: 0' in df.columns:
                df.drop(columns='Unnamed: 0', inplace=True)

            if '_id' in df.columns:
                df.drop(columns='_id' , inplace=True)  


            logger.debug(f'these are the columns after dropping : {df.columns}')    


 
            logger.info(f"Data fetched successfully. Number of records: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Error occurred while fetching data from database '{database_name}', collection '{collection_name}'.")
            raise MyException("Data fetching failed", e)
