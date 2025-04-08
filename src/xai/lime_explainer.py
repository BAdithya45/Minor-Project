import pandas as pd
import joblib
import lime
import lime.lime_tabular
import numpy as np
import webbrowser
from pathlib import Path
import os

def create_lime_explanation(data_path: str, model_name: str = "random_forest", instance_idx: int = None) -> dict:
    """Create LIME explanation for a cooling tower model prediction"""
    try:
        # === Load data ===
        data = pd.read_csv(data_path)
        
        # Select features and target
        features = ["Air Temp (°C)", "Water Flow Rate (L/s)", "Outdoor Humidity (%)"]
        target = "Cooling Tower Efficiency (%)"
        
        if not all(col in data.columns for col in features + [target]):
            raise ValueError("Required columns not found in dataset")
            
        X = data[features]
        y = data[target]
        
        # === Load model ===
        model_path = r"C:\Users\Adithya Bhaskar\Desktop\cooling-tower-xai-project\models\saved_models.pkl"
        models = joblib.load(model_path)
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
        model = models[model_name]
        
        # === Initialize LIME explainer ===
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=features,
            mode="regression"
        )
        
        # === Explain instance ===
        if instance_idx is None:
            instance_idx = np.random.randint(0, len(X))
        exp = explainer.explain_instance(X.iloc[instance_idx].values, model.predict, num_features=len(features))
        
        # === Save explanation ===
        # Create outputs directory in project root
        output_dir = Path(r"C:\Users\Adithya Bhaskar\Desktop\cooling-tower-xai-project\outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Save HTML explanation
        html_path = output_dir / f"lime_explanation_{model_name}_{instance_idx}.html"
        exp.save_to_file(str(html_path))
        
        # Print actual vs predicted values
        actual = y.iloc[instance_idx]
        predicted = model.predict([X.iloc[instance_idx]])[0]
        
        print("\n=== LIME Explanation Results ===")
        print(f"Model: {model_name}")
        print(f"Instance Index: {instance_idx}")
        print(f"Actual Value: {actual:.2f}%")
        print(f"Predicted Value: {predicted:.2f}%")
        print(f"Explanation saved to: {html_path}")
        
        # Open in browser
        webbrowser.open(f'file://{html_path}')
        
        return {
            'explainer': explainer,
            'explanation': exp,
            'instance_index': instance_idx,
            'actual_value': actual,
            'predicted_value': predicted,
            'html_path': html_path
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LIME explanations for cooling tower predictions")
    parser.add_argument("--dataset", type=str, default=r"C:\dataset.csv",
                       help="Path to dataset CSV file")
    parser.add_argument("--model", default="random_forest",
                       choices=["linear_regression", "random_forest", "lightgbm"],
                       help="Model to explain")
    parser.add_argument("--index", type=int, help="Specific instance index to explain (optional)")
    
    args = parser.parse_args()
    create_lime_explanation(args.dataset, args.model, args.index)
