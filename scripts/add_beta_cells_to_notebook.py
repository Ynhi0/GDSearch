#!/usr/bin/env python3
"""
Add Beta Sensitivity 2D cells to Kaggle notebook
"""
import json
from pathlib import Path

def add_beta_sensitivity_cells():
    """Add beta sensitivity 2D visualization cells to the notebook."""
    
    notebook_path = Path('kaggle/gdsearch_kaggle_runner.ipynb')
    
    # Read existing notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # New cells to add (just before the "Package Results" section)
    new_cells = [
        {
            "cell_type": "markdown",
            "id": "beta_2d_header",
            "metadata": {},
            "source": [
                "## Beta Sensitivity 2D Visualizations (Optional)\n",
                "\n",
                "**NEW - February 2026:** Generate publication-quality 2D trajectory visualizations for thesis.\n",
                "\n",
                "**Purpose:**\n",
                "- Visualize β impact on Momentum optimizer trajectory\n",
                "- Analyze β1, β2 impact on Adam dynamics\n",
                "- Generate figures for Vietnamese research proposal requirements\n",
                "\n",
                "**Time:** ~5 minutes for demos (optional, can be skipped)\n",
                "\n",
                "**Research Alignment:**\n",
                "- Vietnamese Proposal: *\"khảo sát ảnh hưởng của β, β1, β2 lên quỹ đạo và độ ổn định\"*\n",
                "- Outputs: Publication-quality trajectory plots, heatmaps, metrics comparisons"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "beta_2d_demos",
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# RUN BETA SENSITIVITY 2D VISUALIZATIONS\n",
                "# ========================================\n",
                "# Set to True to generate 2D visualization plots for thesis\n",
                "# Set to False to skip (saves ~5 minutes)\n",
                "\n",
                "RUN_BETA_2D_DEMOS = False  # Change to True to enable\n",
                "\n",
                "if RUN_BETA_2D_DEMOS:\n",
                "    print(\"=\"*80)\n",
                "    print(\"RUNNING BETA SENSITIVITY 2D VISUALIZATIONS\")\n",
                "    print(\"=\"*80)\n",
                "    print(\"This generates trajectory plots for thesis figures...\\n\")\n",
                "    \n",
                "    import subprocess\n",
                "    from pathlib import Path\n",
                "    \n",
                "    # Run the demo script\n",
                "    result = subprocess.run(\n",
                "        [sys.executable, 'run_beta_2d_demos.py'],\n",
                "        cwd=WORKING_DIR,\n",
                "        capture_output=True,\n",
                "        text=True,\n",
                "        timeout=600  # 10 minute timeout\n",
                "    )\n",
                "    \n",
                "    print(result.stdout)\n",
                "    if result.returncode != 0:\n",
                "        print(\"\\nWarnings/Errors:\")\n",
                "        print(result.stderr)\n",
                "    \n",
                "    # Show generated files\n",
                "    beta_results_dir = WORKING_DIR / 'results' / 'beta_sensitivity_2d'\n",
                "    if beta_results_dir.exists():\n",
                "        print(\"\\n\" + \"=\"*80)\n",
                "        print(\"GENERATED VISUALIZATIONS\")\n",
                "        print(\"=\"*80)\n",
                "        \n",
                "        png_files = list(beta_results_dir.rglob('*.png'))\n",
                "        csv_files = list(beta_results_dir.rglob('*.csv'))\n",
                "        \n",
                "        print(f\"\\nPlots: {len(png_files)} PNG files\")\n",
                "        for png in png_files[:5]:  # Show first 5\n",
                "            print(f\"  - {png.relative_to(WORKING_DIR)}\")\n",
                "        \n",
                "        print(f\"\\nData: {len(csv_files)} CSV files\")\n",
                "        for csv in csv_files:\n",
                "            print(f\"  - {csv.relative_to(WORKING_DIR)}\")\n",
                "        \n",
                "        print(\"\\n💡 These visualizations are suitable for thesis inclusion.\")\n",
                "        print(\"=\"*80)\n",
                "    \n",
                "else:\n",
                "    print(\"=\"*80)\n",
                "    print(\"BETA SENSITIVITY 2D VISUALIZATIONS: SKIPPED\")\n",
                "    print(\"=\"*80)\n",
                "    print(\"To enable: Set RUN_BETA_2D_DEMOS = True in the cell above\")\n",
                "    print(\"\\nThis is optional and takes ~5 minutes.\")\n",
                "    print(\"Useful for generating thesis trajectory visualization figures.\")\n",
                "    print(\"=\"*80)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "beta_2d_custom",
            "metadata": {},
            "outputs": [],
            "source": [
                "# CUSTOM BETA SENSITIVITY EXPERIMENT (Optional)\n",
                "# ===============================================\n",
                "# Uncomment and customize this cell to run your own beta sweep\n",
                "\n",
                "# Example: Custom Momentum β sweep with more values\n",
                "# cmd = [\n",
                "#     sys.executable, 'src/experiments/beta_sensitivity_2d.py',\n",
                "#     '--optimizer', 'Momentum',\n",
                "#     '--function', 'rosenbrock',\n",
                "#     '--beta-values', '0.3,0.5,0.7,0.8,0.9,0.95,0.99',\n",
                "#     '--max-iters', '500'\n",
                "# ]\n",
                "# subprocess.run(cmd, cwd=WORKING_DIR)\n",
                "\n",
                "# Example: Adam β1×β2 grid search\n",
                "# cmd = [\n",
                "#     sys.executable, 'src/experiments/beta_sensitivity_2d.py',\n",
                "#     '--optimizer', 'Adam',\n",
                "#     '--function', 'saddle_point',\n",
                "#     '--beta1-values', '0.7,0.8,0.9,0.95',\n",
                "#     '--beta2-values', '0.9,0.95,0.99,0.999',\n",
                "#     '--max-iters', '300'\n",
                "# ]\n",
                "# subprocess.run(cmd, cwd=WORKING_DIR)\n",
                "\n",
                "print(\"Custom beta sensitivity cell (currently commented out)\")\n",
                "print(\"Uncomment examples above to run customized experiments.\")"
            ]
        }
    ]
    
    # Find insertion point (before "Package Results")
    insertion_index = None
    for i, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if 'Package Results' in source:
                insertion_index = i
                break
    
    if insertion_index is None:
        # If not found, add before last 2 cells (download cells)
        insertion_index = len(notebook['cells']) - 2
    
    # Insert new cells
    for idx, new_cell in enumerate(new_cells):
        notebook['cells'].insert(insertion_index + idx, new_cell)
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✅ Added {len(new_cells)} cells to {notebook_path}")
    print(f"   Inserted at position {insertion_index}")
    print(f"   Total cells now: {len(notebook['cells'])}")

if __name__ == '__main__':
    add_beta_sensitivity_cells()
