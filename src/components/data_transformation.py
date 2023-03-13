import os
import sys 
# from data_ingestion import DataIngestion
import pandas as pd
import numpy as np
import pandas as pd

import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.exceptions import CustomException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import logging 
import os
from dataclasses import dataclass
from src.utils import utility

# class DataTransformer:
#     def __init__(self, yaml_file_path):
#         self.yaml_file_path = yaml_file_path
#         self.di_obj = DataIngestion()
    
#     def load_data(self):
#         self.train_path, self.test_path = self.di_obj.initiate_data_ingestion(self.yaml_file_path) 
#         self.train_set = pd.read_csv(self.train_path)
#         self.test_set = pd.read_csv(self.test_path)

#         self.X_train, self.y_train = self.train_set.drop(columns=['math_score']), self.train_set[['math_score']]
#         self.X_test, self.y_test = self.test_set.drop(columns=['math_score']), self.test_set[['math_score']]

#     def transform(self):
#         self.num_features_train = self.X_train.select_dtypes(exclude="object").columns
#         self.cat_features_train = self.X_train.select_dtypes(include="object").columns

#         self.num_features_test = self.X_test.select_dtypes(exclude="object").columns
#         self.cat_features_test = self.X_test.select_dtypes(include="object").columns

#         self.numeric_transformer = StandardScaler()
#         self.oh_transformer = OneHotEncoder()

#         self.preprocessor = ColumnTransformer(
#             [
#                 ("OneHotEncoder", self.oh_transformer, self.cat_features_train),
#                 ("StandardScaler", self.numeric_transformer, self.num_features_train),        
#             ]
#         )
#         self.X_train = self.preprocessor.fit_transform(self.X_train)
#         self.X_test = self.preprocessor.transform(self.X_test)
        
#         return self.X_train, self.X_test, self.y_train, self.y_test

@dataclass       
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
              ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline",cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[[target_column_name]]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[[target_column_name]]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info(f"data transformed")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"Saved preprocessing object.")

            utility.save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)