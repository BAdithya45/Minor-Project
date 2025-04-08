from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from efficiency.efficiency_recommender import suggest_improvements

app = Flask(__name__)

# ✅ Load trained model from your custom path
MODEL_PATH = r"C:\Users\Adithya Bhaskar\Desktop\cooling-tower-xai-project\models\saved_models.pkl"
model = joblib.load(MODEL_PATH)["random_forest"]  # Replace with "lightgbm", etc. if needed

# 🔑 Input features expected
FEATURE_COLUMNS = [
    "Outdoor Temp (°C)", "Outdoor Humidity (%)", "Wind Speed (m/s)",
    "Water Inlet Temp (°C)", "Water Outlet Temp (°C)", "Air Temp (°C)",
    "Water Flow Rate (L/s)", "Air Velocity (m/s)", "Energy Consumption (kWh)"
]

@app.route("/")
def home():
    return "🌬️ Cooling Tower Efficiency Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict_efficiency():
    try:
        data = request.get_json()

        # ✅ Validate all required features
        missing = [feature for feature in FEATURE_COLUMNS if feature not in data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # 🔄 Prepare data for prediction
        input_data = np.array([data[feature] for feature in FEATURE_COLUMNS]).reshape(1, -1)

        # 🔮 Predict cooling tower efficiency
        predicted_efficiency = model.predict(input_data)[0]

        # 🧠 Suggestions to improve efficiency
        suggestions = suggest_improvements(data)

        # 🌬️ Analyze fan speed from air velocity
        fan_status = fan_speed_status(data["Air Velocity (m/s)"])

        return jsonify({
            "Predicted Efficiency (%)": round(predicted_efficiency, 2),
            "Fan Speed Insight": fan_status,
            "Suggestions": suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def fan_speed_status(air_velocity):
    if air_velocity < 2:
        return "⬆️ Low fan speed — consider increasing airflow."
    elif 2 <= air_velocity <= 4:
        return "✅ Optimal fan speed."
    else:
        return "⚠️ High fan speed — check for overuse or energy inefficiency."

if __name__ == "__main__":
    app.run(debug=True)
