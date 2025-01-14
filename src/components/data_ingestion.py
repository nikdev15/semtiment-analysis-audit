import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from datasets import load_dataset

from dataclasses import dataclass

#from src.components.data_transformation import DataTransformation
#from src.components.data_transformation import DataTransformationConfig

#from src.components.model_trainer import ModelTrainerConfig
#from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.parquet")
    test_data_path: str=os.path.join('artifacts',"test.parquet")
    raw_data_path: str=os.path.join('artifacts',"data.parquet")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=load_dataset("FinanceInc/auditor_sentiment")
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            #df.to_parquet(self.ingestion_config.raw_data_path,header=True)
            train_set = df['train']
            test_set = df['test']

            train_set.to_parquet(self.ingestion_config.train_data_path)

            test_set.to_parquet(self.ingestion_config.test_data_path)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    #data_transformation=DataTransformation()
    #train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))