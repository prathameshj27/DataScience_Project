from src.dsProject.logger import logging
from src.dsProject.exceptions import CustomException
import sys
from src.dsProject.components.data_ingestion import DataIngestionConfig
from src.dsProject.components.data_ingestion import DataIngestion
from src.dsProject.components.data_transformation import DataTransformation, DataTransformationConfig
from src.dsProject.components.model_trainer import Model_Trainer_Config, Model_Trainer

if __name__=="__main__":
    logging.info("The execution has started")

    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        
        ##Model Training
        model_trainer = Model_Trainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
        
    except Exception as e:
        logging.info("Custom Exception raised")
        raise CustomException(e, sys)