
import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the visualization function from run_all_kaggle.py
# We'll use a trick to import it without running main
import run_all_kaggle

def main():
    results_dir = Path("results/experiments/2d_optimization")
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    # Find CSVs using rglob (the fix)
    csv_files = list(results_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files.")
    
    if not csv_files:
        print("No CSV files found even with rglob.")
        return

    # Call the visualization function
    print("Triggering visualization...")
    try:
        run_all_kaggle.create_experiment_visualizations(
            '2D_Optimization', 
            str(results_dir.parent.parent), 
            [str(f) for f in csv_files]
        )
        print("Visualization trigger complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
