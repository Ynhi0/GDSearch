"""Regenerate interactive visualizations only (helper script)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_all_kaggle import generate_interactive_visualizations

generate_interactive_visualizations(r'C:/Users/MPhuc/Downloads/results/results_full', r'C:/Users/MPhuc/Downloads/results/results_full/visualizations')
print('Interactive regeneration complete')