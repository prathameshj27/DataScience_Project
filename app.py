from src.dsProject.logger import logging
from src.dsProject.exceptions import CustomException
import sys
from src.dsProject.components.data_ingestion import DataIngestionConfig
from src.dsProject.components.data_ingestion import DataIngestion
from src.dsProject.components.data_transformation import DataTransformation, DataTransformationConfig
if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        
    except Exception as e:
        logging.info("Custom Exception raised")
        raise CustomException(e, sys)