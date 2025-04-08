# src/efficiency/efficiency_recommender.py

def suggest_improvements(data: dict) -> dict:
    """
    Suggest operational improvements based on input sensor data.
    Expects a dictionary with at least:
    - Air Velocity (m/s)
    - Cooling Tower Efficiency (%)
    - Water Inlet Temp (°C)
    - Water Outlet Temp (°C)
    - Energy Consumption (kWh)
    """
    suggestions = []

    # Extract values
    air_velocity = data.get("Air Velocity (m/s)", None)
    efficiency = data.get("Cooling Tower Efficiency (%)", None)
    inlet_temp = data.get("Water Inlet Temp (°C)", None)
    outlet_temp = data.get("Water Outlet Temp (°C)", None)
    energy_use = data.get("Energy Consumption (kWh)", None)

    # Safety check
    if None in (air_velocity, efficiency, inlet_temp, outlet_temp, energy_use):
        return {"error": "Missing required fields for suggestion generation."}

    # Temperature difference
    temp_diff = inlet_temp - outlet_temp

    # Suggestion logic
    if efficiency < 70:
        suggestions.append("⚠️ Efficiency is below optimal. Consider increasing fan speed (Air Velocity).")
        if air_velocity < 3.0:
            suggestions.append("🔧 Air velocity is low. Increase fan speed to improve heat dissipation.")

    if temp_diff < 5:
        suggestions.append("🌡️ Temperature drop across the tower is too low. Inspect water flow rate and check fan operation.")

    if energy_use > 100:
        suggestions.append("⚡ High energy consumption detected. Optimize PID or reduce fan speed during low-load periods.")

    if air_velocity > 5.0 and efficiency > 85:
        suggestions.append("✅ System is highly efficient with high air velocity. Consider reducing fan speed slightly to save energy.")

    if not suggestions:
        suggestions.append("✅ System appears to be operating within optimal parameters.")

    return {
        "Efficiency (%)": efficiency,
        "Air Velocity (m/s)": air_velocity,
        "Temp Drop (°C)": round(temp_diff, 2),
        "Suggestions": suggestions
    }
# For standalone testing
if __name__ == "__main__":
    sample_data = {
        "Air Velocity (m/s)": 2.5,
        "Cooling Tower Efficiency (%)": 65.0,
        "Water Inlet Temp (°C)": 35.0,
        "Water Outlet Temp (°C)": 31.0,
        "Energy Consumption (kWh)": 120
    }

    suggestions = suggest_improvements(sample_data)
    from pprint import pprint
    pprint(suggestions)

