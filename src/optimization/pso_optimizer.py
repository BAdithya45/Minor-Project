from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
from pyswarm import pso
import argparse
import sys
from datetime import datetime

def pid_fitness(params: List[float], data: pd.DataFrame, max_integral: float = 100.0) -> float:
    """
    Calculate fitness score for PID parameters using multiple metrics:
    - Tracking error
    - Control effort
    - Settling time
    """
    Kp, Ki, Kd = params
    setpoint = data["Setpoint Temp (°C)"].values
    actual_temp = data["Water Outlet Temp (°C)"].values

    error = 0.0
    integral = 0.0
    derivative = 0.0
    previous_error = 0.0
    control_effort = 0.0
    
    # Weights for different performance metrics
    w_error = 1.0
    w_effort = 0.1
    w_settling = 0.2

    for i in range(len(actual_temp)):
        current_error = setpoint[i] - actual_temp[i]
        
        # Anti-windup: Limit integral term
        integral = np.clip(integral + current_error, -max_integral, max_integral)
        
        derivative = current_error - previous_error
        output = Kp * current_error + Ki * integral + Kd * derivative
        
        # Calculate different performance metrics
        error += abs(current_error)
        control_effort += abs(output)
        
        previous_error = current_error

    # Normalize metrics
    avg_error = error / len(actual_temp)
    avg_effort = control_effort / len(actual_temp)
    
    # Combined fitness score
    fitness = (w_error * avg_error + 
              w_effort * avg_effort)
    
    return fitness

def optimize_pid(data_path: str, swarm_size: int = 30, max_iter: int = 50) -> Tuple[np.ndarray, float]:
    """Run PSO optimization to find optimal PID parameters."""
    try:
        # Load and validate data
        data = pd.read_csv(data_path)
        required_cols = ["Setpoint Temp (°C)", "Water Outlet Temp (°C)"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")

        # Validate PSO parameters
        if swarm_size < 1:
            raise ValueError("Swarm size must be positive")
        if max_iter < 1:
            raise ValueError("Maximum iterations must be positive")

        # Define bounds for Kp, Ki, Kd with more reasonable ranges
        lb = [0.1, 0.0, 0.0]  # Lower bounds
        ub = [5.0, 2.0, 2.0]  # Upper bounds - reduced from 10.0

        print("Starting PSO optimization...")
        # Run PSO with data passed to fitness function
        best_params, best_score = pso(
            lambda x: pid_fitness(x, data),
            lb, ub,
            swarmsize=swarm_size,
            maxiter=max_iter,
            minstep=1e-6,  # Reduced from 1e-8
            minfunc=1e-6,  # Reduced from 1e-8
            debug=True
        )
        
        return best_params, best_score
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="PID Controller Optimizer using PSO")
        parser.add_argument("--data", type=Path, default=Path(r"C:\dataset.csv"),
                           help="Path to dataset CSV file")
        parser.add_argument("--swarm-size", type=int, default=30,
                           help="Size of the particle swarm")
        parser.add_argument("--max-iter", type=int, default=50,
                           help="Maximum number of iterations")
        
        args = parser.parse_args()
        
        # Validate data path
        if not args.data.exists():
            raise FileNotFoundError(f"Dataset not found: {args.data}")
            
        # Run optimization
        best_params, best_score = optimize_pid(str(args.data), args.swarm_size, args.max_iter)
        
        # Create output directory
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "Kp": best_params[0],
            "Ki": best_params[1],
            "Kd": best_params[2],
            "fitness_score": best_score,
            "timestamp": timestamp
        }
        
        output_file = output_dir / f"pid_parameters_{timestamp}.csv"
        pd.DataFrame([results]).to_csv(output_file, index=False)
        
        # Print results
        print(f"\nBest PID Parameters Found:")
        print(f"Kp: {best_params[0]:.4f}")
        print(f"Ki: {best_params[1]:.4f}")
        print(f"Kd: {best_params[2]:.4f}")
        print(f"Fitness Score (Avg Error): {best_score:.4f}")
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
