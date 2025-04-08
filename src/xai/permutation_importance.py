import os
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import argparse

def calculate_permutation_importance(dataset_path: str, model_name: str = "random_forest", n_repeats: int = 30) -> Dict[str, Any]:
    try:
        # Validate dataset path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        # Load the dataset
        data = pd.read_csv(dataset_path)
        
        # Select relevant features and target (matching training data)
        features = ["Air Temp (°C)", "Water Flow Rate (L/s)", "Outdoor Humidity (%)"]
        target = "Cooling Tower Efficiency (%)"
        
        if not all(col in data.columns for col in features + [target]):
            raise ValueError("Required columns not found in dataset")
        
        X = data[features]
        y = data[target]
        
        # Use specific model path
        model_path = r"C:\Users\Adithya Bhaskar\Desktop\cooling-tower-xai-project\models\saved_models.pkl"
        
        # Validate model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model with error handling
        try:
            models = joblib.load(model_path)
            if model_name not in models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
            model = models[model_name]
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
            
        # Compute permutation importance
        print(f"Computing permutation importance for {model_name}...")
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
        
        # Plot the results
        sorted_idx = result.importances_mean.argsort()
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
        plt.xlabel("Permutation Importance")
        plt.title(f"Permutation Feature Importance ({model_name})")
        plt.tight_layout()
        
        # Save the plot
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"permutation_importance_{model_name}.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        plt.show()
        
        return {
            'importances_mean': result.importances_mean,
            'importances_std': result.importances_std,
            'features': features,
            'sorted_idx': sorted_idx
        }
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate permutation importance for cooling tower models")
    parser.add_argument("--dataset", type=str, default=str(Path("C:\dataset.csv")), help="Path to dataset CSV file")
    parser.add_argument("--model", default="random_forest", 
                       choices=["linear_regression", "random_forest", "lightgbm"],
                       help="Model to analyze")
    parser.add_argument("--repeats", type=int, default=30, help="Number of permutation repeats")
    
    args = parser.parse_args()
    calculate_permutation_importance(args.dataset, args.model, args.repeats)
