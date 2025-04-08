import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ✅ Correct dataset path
file_path = "C:\\dataset.csv"
data = pd.read_csv(file_path)
print("Available columns:", data.columns.tolist())

# ✅ Replace with your actual column names
X = data[["Air Temp (°C)", "Water Flow Rate (L/s)", "Outdoor Humidity (%)"]]
y = data["Cooling Tower Efficiency (%)"]

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("LR MSE:", mean_squared_error(y_test, lr.predict(X_test)))

# ✅ Train Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print("RF MSE:", mean_squared_error(y_test, rf.predict(X_test)))

# ✅ Train LightGBM
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
print("LGBM MSE:", mean_squared_error(y_test, lgb_model.predict(X_test)))

# ✅ Save the models
models_folder = "..\\models"
os.makedirs(models_folder, exist_ok=True)

joblib.dump(lr, os.path.join(models_folder, "linear_regression_model.joblib"))
joblib.dump(rf, os.path.join(models_folder, "random_forest_model.joblib"))
joblib.dump(lgb_model, os.path.join(models_folder, "lightgbm_model.joblib"))

models = {
    "linear_regression": lr,
    "random_forest": rf,
    "lightgbm": lgb_model  # Fixed: using lgb_model instead of lgb
}

joblib.dump(models, "../models/saved_models.pkl")
print("✅ Models saved to models/saved_models.pkl")


