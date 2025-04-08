from pathlib import Path
from typing import Dict, Any
import sys
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def create_pdp_plot(dataset_path: str, model_name: str = "random_forest") -> Dict[str, Any]:
    try:
        # Validate paths
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
            
        # Load dataset
        data = pd.read_csv(dataset_path)
        
        # Select relevant features (matching training data)
        features = ["Air Temp (°C)", "Water Flow Rate (L/s)", "Outdoor Humidity (%)"]
        target = "Cooling Tower Efficiency (%)"
        
        if not all(col in data.columns for col in features + [target]):
            raise ValueError("Required columns not found in dataset")
            
        X = data[features]
        y = data[target]
        
        # Use absolute model path
        model_path = r"C:\Users\Adithya Bhaskar\Desktop\cooling-tower-xai-project\models\saved_models.pkl"
        
        # Load model with error handling
        try:
            models = joblib.load(model_path)
            if model_name not in models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
            model = models[model_name]
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
        
        # Create PDP plot using the new API
        fig, ax = plt.subplots(figsize=(12, 8))
        display = PartialDependenceDisplay.from_estimator(
            model, X, features,
            n_jobs=-1,
            grid_resolution=50,
            random_state=42
        )
        plt.tight_layout()
        
        # Save plot
        output_dir = Path(r"C:\Users\Adithya Bhaskar\Desktop\cooling-tower-xai-project\outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"pdp_plot_{model_name}.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        
        plt.show()
        
        return {
            'figure': fig,
            'features': features,
            'model_name': model_name,
            'display': display
        }
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PDP plots for cooling tower models")
    parser.add_argument("--dataset", type=str, default=r"C:\dataset.csv", 
                       help="Path to dataset CSV file")
    parser.add_argument("--model", default="random_forest",
                       choices=["linear_regression", "random_forest", "lightgbm"],
                       help="Model to analyze")
    
    args = parser.parse_args()
    create_pdp_plot(args.dataset, args.model)
