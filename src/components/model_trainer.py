import os
import sys 
from dataclasses import dataclass 

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 

from src.exceptions import CustomException 
from src.logger import logging 

from src.utils import utility  
# class ModelTrainer:
#     def __init__(self, yaml_file_path):
#         self.data_transformer = DataTransformer(yaml_file_path)
#         self.data_transformer.load_data()
#         self.X_train, self.X_test, self.y_train, self.y_test = self.data_transformer.transform()

#     def evaluate_model(self, true, predicted):
#         self.mae = mean_absolute_error(true, predicted)
#         self.mse = mean_squared_error(true, predicted)
#         self.rmse = np.sqrt(mean_squared_error(true, predicted))
#         self.r2_square = r2_score(true, predicted)
#         return self.mae, self.rmse, self.r2_square
    
#     def training(self):

#         self.models = {
#             "Linear Regression": LinearRegression(),
#             "Lasso": Lasso(),
#             "Ridge": Ridge(),
#             "K-Neighbors Regressor": KNeighborsRegressor(),
#             "Decision Tree": DecisionTreeRegressor(),
#             "Random Forest Regressor": RandomForestRegressor(),
#             "XGBRegressor": XGBRegressor(), 
#             "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#             "AdaBoost Regressor": AdaBoostRegressor()
#         }
#         self.model_list = []
#         self.r2_list =[]

#         for i in range(len(list(self.models))):
#             self.model = list(self.models.values())[i]
#             self.model.fit(self.X_train, self.y_train) # Train model

#             # Make predictions
#             self.y_train_pred = self.model.predict(self.X_train)
#             self.y_test_pred = self.model.predict(self.X_test)
    
#             # Evaluate Train and Test dataset
#             self.model_train_mae , self.model_train_rmse, self.model_train_r2 = self.evaluate_model(self.y_train, self.y_train_pred)

#             self.model_test_mae , self.model_test_rmse, self.model_test_r2 = self.evaluate_model(self.y_test, self.y_test_pred)

    
#             print(list(self.models.keys())[i])
#             self.model_list.append(list(self.models.keys())[i])
    
#             print('Model performance for Training set')
#             print("- Root Mean Squared Error: {:.4f}".format(self.model_train_rmse))
#             print("- Mean Absolute Error: {:.4f}".format(self.model_train_mae))
#             print("- R2 Score: {:.4f}".format(self.model_train_r2))

#             print('----------------------------------')
            
#             print('Model performance for Test set')
#             print("- Root Mean Squared Error: {:.4f}".format(self.model_test_rmse))
#             print("- Mean Absolute Error: {:.4f}".format(self.model_test_mae))
#             print("- R2 Score: {:.4f}".format(self.model_test_r2))
#             self.r2_list.append(self.model_test_r2)
            
#             print('='*35)
#             print('\n')
#         self.best_r2 = max(self.r2_list)
#         self.ind = self.r2_list.index(self.best_r2)
#         self.best_model = self.model_list[self.ind]
#         return self.best_r2, self.best_model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Catboost regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = utility.evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset")

            utility.save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            
            r_squared = r2_score(y_test, predicted)
            print("R2_score: ",r_squared)
            return r_squared 
        except Exception as e:
            raise CustomException(e, sys)