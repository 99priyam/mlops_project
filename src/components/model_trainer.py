from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
from sklearn.model_selection import train_test_split
from data_transformation import DataTransformer
import numpy as np

class ModelTrainer:
    def __init__(self, yaml_file_path):
        self.data_transformer = DataTransformer(yaml_file_path)
        self.data_transformer.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_transformer.transform()

    def evaluate_model(self, true, predicted):
        self.mae = mean_absolute_error(true, predicted)
        self.mse = mean_squared_error(true, predicted)
        self.rmse = np.sqrt(mean_squared_error(true, predicted))
        self.r2_square = r2_score(true, predicted)
        return self.mae, self.rmse, self.r2_square
    
    def training(self):

        self.models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
        }
        self.model_list = []
        self.r2_list =[]

        for i in range(len(list(self.models))):
            self.model = list(self.models.values())[i]
            self.model.fit(self.X_train, self.y_train) # Train model

            # Make predictions
            self.y_train_pred = self.model.predict(self.X_train)
            self.y_test_pred = self.model.predict(self.X_test)
    
            # Evaluate Train and Test dataset
            self.model_train_mae , self.model_train_rmse, self.model_train_r2 = self.evaluate_model(self.y_train, self.y_train_pred)

            self.model_test_mae , self.model_test_rmse, self.model_test_r2 = self.evaluate_model(self.y_test, self.y_test_pred)

    
            print(list(self.models.keys())[i])
            self.model_list.append(list(self.models.keys())[i])
    
            print('Model performance for Training set')
            print("- Root Mean Squared Error: {:.4f}".format(self.model_train_rmse))
            print("- Mean Absolute Error: {:.4f}".format(self.model_train_mae))
            print("- R2 Score: {:.4f}".format(self.model_train_r2))

            print('----------------------------------')
            
            print('Model performance for Test set')
            print("- Root Mean Squared Error: {:.4f}".format(self.model_test_rmse))
            print("- Mean Absolute Error: {:.4f}".format(self.model_test_mae))
            print("- R2 Score: {:.4f}".format(self.model_test_r2))
            self.r2_list.append(self.model_test_r2)
            
            print('='*35)
            print('\n')
        self.best_r2 = max(self.r2_list)
        self.ind = self.r2_list.index(self.best_r2)
        self.best_model = self.model_list[self.ind]
        return self.best_r2, self.best_model

if __name__ == "__main__":
    obj = ModelTrainer('src\\components\\yaml_file.yaml')
    r2_score, model_name = obj.training()
    print("Best Model is {0} with {1} r-squared score".format(model_name, np.round(r2_score,2)))
