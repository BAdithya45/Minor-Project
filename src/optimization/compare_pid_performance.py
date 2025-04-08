from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimization.pid_controller import PIDController
from typing import List, Tuple
import sys
import argparse

def load_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate dataset."""
    try:
        df = pd.read_csv(data_path)
        required_cols = ["Setpoint Temp (°C)", "Water Outlet Temp (°C)", "Kp", "Ki", "Kd"]
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
            
        # Drop rows with missing values in key columns
        df.dropna(subset=required_cols, inplace=True)
        
        if len(df) == 0:
            raise ValueError("No valid data rows after cleaning")
        
        return (
            df["Setpoint Temp (°C)"].values,
            df["Water Outlet Temp (°C)"].values,
            df["Kp"].values,
            df["Ki"].values,
            df["Kd"].values
        )
    except Exception as e:
        print(f"Error loading data: {str(e)}", file=sys.stderr)
        raise

def evaluate_pid_performance(setpoints: np.ndarray, actual_temps: np.ndarray, 
                           kp: float, ki: float, kd: float) -> List[float]:
    """Evaluate PID controller performance."""
    if not all(isinstance(x, (int, float)) for x in [kp, ki, kd]):
        raise ValueError("PID parameters must be numeric")
        
    errors = []
    pid = PIDController(Kp=kp, Ki=ki, Kd=kd)
    
    for i in range(len(setpoints)):
        output = pid.compute(setpoints[i], actual_temps[i])
        corrected_temp = actual_temps[i] + output
        error = abs(setpoints[i] - corrected_temp)
        errors.append(error)
        pid.reset()  # Reset controller state between iterations
        
    return errors

def main():
    try:
        parser = argparse.ArgumentParser(description="Compare PID controller performance")
        parser.add_argument("--data", type=Path, default=Path("dataset.csv"),
                          help="Path to dataset CSV file")
        parser.add_argument("--results", type=Path, default=Path("pid_parameters.csv"),
                          help="Path to optimization results CSV file")
        args = parser.parse_args()
        
        # Setup paths
        script_dir = Path(__file__).parent
        data_path = script_dir.parent.parent / "data" / args.data
        results_path = script_dir / "outputs" / args.results
        
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        # Load data
        setpoints, actual_temps, kp_vals, ki_vals, kd_vals = load_data(data_path)
        
        # Evaluate original PID performance
        original_errors = evaluate_pid_performance(setpoints, actual_temps, 
                                                np.mean(kp_vals), np.mean(ki_vals), np.mean(kd_vals))
        avg_original_error = np.mean(original_errors)
        print(f"⚙️ Average Error using original PID values: {avg_original_error:.4f}")
        
        # Load optimized parameters
        try:
            opt_results = pd.read_csv(results_path)
            best_Kp = opt_results['Kp'].iloc[-1]
            best_Ki = opt_results['Ki'].iloc[-1]
            best_Kd = opt_results['Kd'].iloc[-1]
        except Exception as e:
            print(f"Warning: Could not load optimized parameters: {e}")
            best_Kp, best_Ki, best_Kd = 0.1, 0.0, 0.0
            
        # Evaluate optimized PID performance
        optimized_errors = evaluate_pid_performance(setpoints, actual_temps, 
                                                 best_Kp, best_Ki, best_Kd)
        avg_optimized_error = np.mean(optimized_errors)
        print(f"🏁 Average Error using optimized PID values: {avg_optimized_error:.4f}")
        
        # Visualize results
        plt.figure(figsize=(12, 6))
        plt.plot(original_errors, label='Original PID', alpha=0.7)
        plt.plot(optimized_errors, label='Optimized PID', alpha=0.7)
        plt.xlabel('Time Step')
        plt.ylabel('Absolute Error')
        plt.title('PID Controller Performance Comparison')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_dir = script_dir / "outputs"
        output_dir.mkdir(exist_ok=True)
        try:
            plt.savefig(output_dir / "pid_comparison.png")
            print(f"📊 Performance comparison plot saved to {output_dir}/pid_comparison.png")
        except Exception as e:
            print(f"Warning: Failed to save plot: {e}", file=sys.stderr)
        
        plt.show()
        plt.close()  # Cleanup
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
