import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")

class Modeler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_claim = None
        self.df_model = None
        self.feature_cols = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_clean_data(self):
        self.df = pd.read_csv(self.file_path, sep="|")
        self.df_claim = self.df[self.df['TotalClaims'] > 0].copy()
        self.df_claim['CapitalOutstanding'] = pd.to_numeric(self.df_claim['CapitalOutstanding'], errors='coerce')
        self.df_claim.dropna(subset=['TotalClaims', 'CapitalOutstanding'], inplace=True)

    def select_features(self):
        feature_cols = [
    'cubiccapacity', 'kilowatts', 'CapitalOutstanding', 'SumInsured', 
    'NewVehicle', 'Gender', 'Province', 'VehicleType'
]

        self.df_model = self.df_claim[feature_cols + ['TotalClaims']].dropna()

        # Encode categoricals
        self.df_model = pd.get_dummies(self.df_model, columns=['NewVehicle', 'Gender', 'Province', 'VehicleType'], drop_first=True)
    
    def split_data(self):
        X = self.df_model.drop('TotalClaims', axis=1)
        y = self.df_model['TotalClaims']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_and_evaluate_model(self, model, name):
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        r2 = r2_score(self.y_test, preds)
        print(f"{name} — RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return model
    
    def run_models(self):
        # Linear Regression
        lr = self.train_and_evaluate_model(LinearRegression(), "Linear Regression")

        # Random Forest
        rf = self.train_and_evaluate_model(RandomForestRegressor(random_state=42), "Random Forest")

        # XGBoost
        xgb = self.train_and_evaluate_model(XGBRegressor(random_state=42, n_jobs=-1), "XGBoost")

        return lr, rf, xgb
    
    def explain_model(self, model):
        explainer = shap.Explainer(model)
        shap_values = explainer(self.X_test)

        # Plot SHAP Summary
        shap.plots.beeswarm(shap_values)
        plt.show()
