from src.dsProject.logger import logging
from src.dsProject.exceptions import CustomException
import sys
from src.dsProject.components.data_ingestion import DataIngestionConfig
from src.dsProject.components.data_ingestion import DataIngestion

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception raised")
        raise CustomException(e, sys)