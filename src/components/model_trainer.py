import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from transformers import TFAutoModel
import tensorflow as tf

from src.utils import save_object,load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array,length,test_values):
        try:
            logging.info("Split training and val input data")
            size = int((length / 12) * 0.8)
            train_ds = train_array.take(size)
            val_ds = train_array.skip(size)

            model = TFAutoModel.from_pretrained('bert-base-uncased')

            input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
            mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')
            embeddings = model.bert(input_ids, attention_mask=mask)[1]
            layer_1 = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
            layer_2 = tf.keras.layers.Dense(3, activation='softmax', name='outputs')(layer_1)

            sentiment_model = tf.keras.Model(inputs=[input_ids, mask], outputs=layer_2)
            sentiment_model.layers[2].trainable = True

            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=1e-6)
            loss = tf.keras.losses.CategoricalCrossentropy()
            acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

            sentiment_model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

            history = sentiment_model.fit(train_ds,validation_data=val_ds,epochs=3)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=sentiment_model
            )
            
            label_mapping = {0: 'Negative',1: 'Neutral',2: 'Positive'}
            probs =sentiment_model.predict(test_array)
            pred = np.argmax(probs, axis=1)

            df_test = pd.DataFrame(test_values, columns = ['label'])
            df_pred = pd.DataFrame(pred, columns = ['label'])

            accuracy = accuracy_score(df_pred, df_test) * 100
            classificationreport=classification_report(test_values,pred)

            return classificationreport

        except Exception as e:
            raise CustomException(e,sys)