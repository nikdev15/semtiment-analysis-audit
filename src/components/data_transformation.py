import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from transformers import BertTokenizer
import tensorflow as tf

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            checkpoint = 'bert-base-uncased'

            tokenizer  = BertTokenizer.from_pretrained(checkpoint)

            return tokenizer
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=train_path
            test_df=test_path

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="label"

            input_feature_train_df=train_df['sentence']
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df['sentence']
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            seq_len = 512
            input_feature_train_arr=preprocessing_obj(input_feature_train_df, max_length=seq_len, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='pt')
            def prep_arry_data(text_array):
                tokens = preprocessing_obj.batch_encode_plus(text_array, max_length=512,truncation=True, padding='max_length',add_special_tokens=True, return_token_type_ids=False,return_tensors='tf')
                return {'input_ids': tf.cast(tokens['input_ids'], tf.float64),'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}
            input_feature_test_arr=prep_arry_data(input_feature_test_df)

            num_samples= len(train_df)
            arr = pd.DataFrame(train_df['label'], columns=['label'])
            arr_values = arr['label'].sort_values().unique().tolist()
            labels = np.zeros((num_samples, len(arr_values)))
            labels[np.arange(num_samples), arr['label'].tolist()] = 1
            ds = tf.data.Dataset.from_tensor_slices((input_feature_train_arr['input_ids'], input_feature_train_arr['attention_mask'], labels))
            def map_func(input_ids, masks, labels):
                return {'input_ids': input_ids, 'attention_mask': masks}, labels
            ds = ds.map(map_func)
            ds_train = ds.shuffle(10000).batch(12, drop_remainder=True)
            ds_test = input_feature_test_arr

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                ds_train,
                ds_test,
                input_feature_train_arr['input_ids'].shape[0],
                target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)