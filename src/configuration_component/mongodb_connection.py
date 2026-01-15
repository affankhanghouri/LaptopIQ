import pymongo
from dotenv import load_dotenv
load_dotenv()
import os
from src.logging_component import logger
from src.exception_component import MyException
from src.constants_component import *

class MongoDB_Client:
    """
    MongoDB client singleton.
    Ensures only one MongoClient is created.
    """

    client = None

    def __init__(self):
        try:
            logger.info("Attempting to build MongoDB connection...")

            if MongoDB_Client.client is None:
                mongo_db_url = os.getenv("MONGO_DB_CONNECTION_URL")
                if not mongo_db_url:
                    logger.error("MongoDB connection URL is not set in environment variables")
                    raise MyException("MongoDB URL is missing in environment variables")

                # Initialize MongoClient
                MongoDB_Client.client = pymongo.MongoClient(
                    mongo_db_url,
                    serverSelectionTimeoutMS=5000,
                    heartbeatFrequencyMS=10000
                )

            self.client = MongoDB_Client.client
            self.database = self.client[DATABASE_NAME]
            logger.info("MongoDB connection established successfully.")

        except Exception as e:
            logger.exception(f"MongoDB client initialization failed: {e}")
            raise MyException("MongoDB client initialization failed", e)
