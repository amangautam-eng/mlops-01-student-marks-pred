import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file= os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns=['writing_score','reading_score']
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical column scaling completed")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encode",OneHotEncoder()),
                    ("scaling",StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical column encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("categorical_pipeline",cat_pipeline,categorical_columns)
                ]

            )

            return preprocessor



            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data")
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"

            x_train=train_df.drop(columns=[target_column_name])
            y_train=train_df[target_column_name]

            x_test=test_df.drop(columns=[target_column_name])
            y_test=test_df[target_column_name]

            logging.info("Applying preprocessor obj on train and test data")


            x_train_transformed=preprocessing_obj.fit_transform(x_train)
            x_test_transformed=preprocessing_obj.transform(x_test)

            train_arr=np.c_[x_train_transformed,np.array(y_train)]
            test_arr=np.c_[x_test_transformed,np.array(y_test)]

            logging.info("Sa_ve preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file,
            )


        except Exception as e:
            raise CustomException(e,sys)
            

            