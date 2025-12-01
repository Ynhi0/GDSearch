#!/usr/bin/env python3
"""
Organize GDSearch Output Structure

Reorganizes experiment outputs into a clear, descriptive structure:

results/
├── experiments/              # Raw per-run CSVs
│   ├── mnist/
│   ├── cifar10/
│   ├── nlp/
│   └── medical/
├── analysis/                 # Statistical and convergence analysis
│   ├── 00_basic_statistics.csv
│   ├── 01_convergence_rates.csv
│   └── 02_statistical_comparison.csv
├── visualizations/           # Plots and charts
│   ├── interactive/         # HTML interactive plots
│   └── static/              # PNG/PDF static plots
└── reports/                  # Markdown summaries
    └── 00_EXPERIMENT_SUMMARY.md
"""

import shutil
from pathlib import Path

def organize_results_directory():
    """Reorganize results directory with descriptive structure"""
    
    results_root = Path("results")
    if not results_root.exists():
        print("No results directory found")
        return
    
    print("="*80)
    print(" "*20 + "ORGANIZING OUTPUT STRUCTURE")
    print("="*80)
    
    # Create organized structure
    experiments_dir = results_root / "experiments"
    analysis_dir = results_root / "analysis"
    viz_dir = results_root / "visualizations"
    viz_interactive = viz_dir / "interactive"
    viz_static = viz_dir / "static"
    reports_dir = results_root / "reports"
    
    for d in [experiments_dir, analysis_dir, viz_interactive, viz_static, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Move experiment CSVs to experiments/
    for dataset in ['mnist', 'cifar10', 'nlp', 'medical', 'resnet', 'highdim']:
        old_dir = results_root / dataset
        if old_dir.exists() and old_dir.is_dir():
            new_dir = experiments_dir / dataset
            if new_dir.exists():
                shutil.rmtree(new_dir)
            shutil.move(str(old_dir), str(new_dir))
            print(f"✓ Moved {dataset}/ to experiments/{dataset}/")
    
    # Move analysis files
    analysis_files = [
        ('convergence_analysis.csv', '01_convergence_rates.csv'),
        ('statistical_comparison.csv', '02_statistical_comparison.csv'),
        ('basic_statistics.csv', '00_basic_statistics.csv'),
        ('basic_stats.csv', '00_basic_stats_summary.csv'),
    ]
    
    for old_name, new_name in analysis_files:
        old_path = results_root / old_name
        if old_path.exists():
            new_path = analysis_dir / new_name
            shutil.move(str(old_path), str(new_path))
            print(f"✓ Moved {old_name} to analysis/{new_name}")
    
    # Move plots to visualizations/
    old_plots = results_root / "plots"
    if old_plots.exists() and old_plots.is_dir():
        for html_file in old_plots.glob("*.html"):
            shutil.move(str(html_file), str(viz_interactive / html_file.name))
            print(f"✓ Moved {html_file.name} to visualizations/interactive/")
        
        # Remove empty plots dir
        if not list(old_plots.iterdir()):
            old_plots.rmdir()
    
    # Move any HTML files in root to visualizations/
    for html_file in results_root.glob("*.html"):
        shutil.move(str(html_file), str(viz_interactive / html_file.name))
        print(f"✓ Moved {html_file.name} to visualizations/interactive/")
    
    # Move reports
    report_files = [
        ('FINAL_SUMMARY_REPORT.md', '00_EXPERIMENT_SUMMARY.md'),
        ('EXPERIMENT_SUMMARY.md', '00_EXPERIMENT_SUMMARY.md'),
    ]
    
    for old_name, new_name in report_files:
        old_path = results_root / old_name
        if old_path.exists():
            new_path = reports_dir / new_name
            if new_path.exists():
                new_path.unlink()
            shutil.move(str(old_path), str(new_path))
            print(f"✓ Moved {old_name} to reports/{new_name}")
    
    # Create directory structure documentation
    readme_path = results_root / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# GDSearch Experiment Results\n\n")
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write("results/\n")
        f.write("├── experiments/              # Raw per-run experiment data\n")
        f.write("│   ├── mnist/               # MNIST CSVs (per seed, per optimizer)\n")
        f.write("│   ├── cifar10/             # CIFAR-10 CSVs\n")
        f.write("│   ├── nlp/                 # NLP sentiment CSVs\n")
        f.write("│   ├── medical/             # Medical segmentation CSVs\n")
        f.write("│   ├── resnet/              # ResNet18 CSVs\n")
        f.write("│   └── highdim/             # High-dimensional optimization CSVs\n")
        f.write("├── analysis/                 # Post-experiment analysis\n")
        f.write("│   ├── 00_basic_statistics.csv          # Mean/std/min/max per optimizer\n")
        f.write("│   ├── 01_convergence_rates.csv         # Convergence analysis\n")
        f.write("│   └── 02_statistical_comparison.csv    # t-tests, effect sizes, p-values\n")
        f.write("├── visualizations/           # Plots and charts\n")
        f.write("│   ├── interactive/         # HTML plots (Plotly) - open in browser\n")
        f.write("│   └── static/              # PNG/PDF static plots\n")
        f.write("└── reports/                  # Markdown summaries\n")
        f.write("    └── 00_EXPERIMENT_SUMMARY.md         # Comprehensive experiment report\n")
        f.write("```\n\n")
        f.write("## File Naming Conventions\n\n")
        f.write("### Experiment CSVs\n")
        f.write("- Format: `{DATASET}_{MODEL}_{OPTIMIZER}_seed{SEED}.csv`\n")
        f.write("- Example: `MNIST_MLP_Adam_seed42.csv`\n\n")
        f.write("### Analysis Files\n")
        f.write("- Numbered prefix for logical ordering (00, 01, 02...)\n")
        f.write("- Descriptive names indicating content\n\n")
        f.write("### Interactive Plots\n")
        f.write("- Format: `{dataset}_optimizer_comparison.html`\n")
        f.write("- Open in web browser for interactive pan/zoom/hover\n\n")
        f.write("## Quick Access\n\n")
        f.write("```bash\n")
        f.write("# View convergence analysis\n")
        f.write("cat analysis/01_convergence_rates.csv | column -t -s,\n\n")
        f.write("# View statistical comparisons\n")
        f.write("cat analysis/02_statistical_comparison.csv | column -t -s,\n\n")
        f.write("# Open interactive plots\n")
        f.write("open visualizations/interactive/*.html  # macOS\n")
        f.write("xdg-open visualizations/interactive/*.html  # Linux\n\n")
        f.write("# Read summary report\n")
        f.write("cat reports/00_EXPERIMENT_SUMMARY.md\n")
        f.write("```\n\n")
        f.write("## Using Results in Python\n\n")
        f.write("```python\n")
        f.write("import pandas as pd\n")
        f.write("from pathlib import Path\n\n")
        f.write("results = Path('results')\n\n")
        f.write("# Load convergence data\n")
        f.write("conv = pd.read_csv(results / 'analysis/01_convergence_rates.csv')\n")
        f.write("print(conv.groupby('optimizer')['convergence_rate'].mean())\n\n")
        f.write("# Load statistical tests\n")
        f.write("stats = pd.read_csv(results / 'analysis/02_statistical_comparison.csv')\n")
        f.write("significant = stats[stats['is_significant']]\n")
        f.write("print(significant[['optimizer_1', 'optimizer_2', 'mean_diff', 'p_value']])\n\n")
        f.write("# Load specific experiment\n")
        f.write("mnist_adam = pd.read_csv(results / 'experiments/mnist/MNIST_MLP_Adam_seed42.csv')\n")
        f.write("print(mnist_adam[['epoch', 'test_acc']].tail())\n")
        f.write("```\n")
    
    print(f"\n✓ Created {readme_path}")
    
    print("\n" + "="*80)
    print("✅ ORGANIZATION COMPLETE")
    print("="*80)
    print(f"\nOrganized structure in: {results_root}/")
    print("  📁 experiments/     - Raw experiment CSVs")
    print("  📊 analysis/        - Statistical analysis")
    print("  📈 visualizations/  - Interactive & static plots")
    print("  📄 reports/         - Summary reports")
    print("\nSee results/README.md for detailed documentation")
    print("="*80)


if __name__ == '__main__':
    organize_results_directory()
