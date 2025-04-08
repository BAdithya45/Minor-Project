import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load('../models/saved_models.pkl')['random_forest']  # You can change this to 'lightgbm' or 'linear_regression'

# Load dataset again
data = pd.read_csv('C:\\dataset.csv')

# Use the same features you trained with
features = ['Outdoor Temp (°C)', 'Water Flow Rate (L/s)', 'Outdoor Humidity (%)']
X = data[features]

# Subset for SHAP performance (optional)
X_sample = X.sample(100, random_state=42)

# Create TreeExplainer and get SHAP values
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

# SHAP summary plot
shap.summary_plot(shap_values, X_sample, show=True)
